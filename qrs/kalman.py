"""Pillar 1 -- Time-varying beta via the Kalman filter.

Model (state-space form)
------------------------
Observation:  r_t^stock = alpha_t + beta_t * r_t^mkt + eps_t,   eps_t ~ N(0, R)
State:        [alpha_t, beta_t]' = [alpha_{t-1}, beta_{t-1}]' + eta_t,
              eta_t ~ N(0, diag(q_alpha, q_beta))

So: state vector x_t = [alpha_t, beta_t]', F = I_2, H_t = [1, r_t^mkt].
H_t is TIME-VARYING (it *is* the market return) -- this is why the filter
learns a lot about beta on big-market days and almost nothing on flat days:
the Kalman gain scales with H_t.

Q and R are estimated by maximum likelihood: the filter's innovations nu_t
and their variances S_t define the log-likelihood of the data
(the "prediction error decomposition"), which we maximise with scipy.
This replaces the arbitrary "why a 60-day window?" of rolling OLS with a
parameter estimated from the data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize


@dataclass
class KalmanResult:
    alpha_filt: np.ndarray   # filtered E[alpha_t | data up to t]
    beta_filt: np.ndarray    # filtered E[beta_t  | data up to t]
    alpha_smooth: np.ndarray  # smoothed E[alpha_t | ALL data]
    beta_smooth: np.ndarray   # smoothed E[beta_t  | ALL data]
    beta_filt_se: np.ndarray   # sqrt of filtered variance of beta
    beta_smooth_se: np.ndarray  # sqrt of smoothed variance of beta
    q_alpha: float
    q_beta: float
    r_obs: float
    loglik: float


def _filter_pass(y, x_mkt, q_alpha, q_beta, r_obs, x0, P0):
    """One forward Kalman pass. Returns filtered moments, predictions, loglik."""
    T = len(y)
    Q = np.diag([q_alpha, q_beta])

    x_pred = np.zeros((T, 2));   P_pred = np.zeros((T, 2, 2))
    x_filt = np.zeros((T, 2));   P_filt = np.zeros((T, 2, 2))
    loglik = 0.0

    x_prev, P_prev = x0.copy(), P0.copy()
    for t in range(T):
        # ---- PREDICT: state persists (F = I), uncertainty inflates by Q ----
        x_p = x_prev
        P_p = P_prev + Q

        # ---- UPDATE: precision-weighted correction toward the innovation ----
        H = np.array([1.0, x_mkt[t]])            # time-varying observation map
        nu = y[t] - H @ x_p                       # innovation (the surprise)
        S = H @ P_p @ H + r_obs                   # innovation variance (scalar)
        K = (P_p @ H) / S                         # Kalman gain (2-vector)

        x_f = x_p + K * nu
        P_f = P_p - np.outer(K, H) @ P_p          # (I - K H) P_p

        # Gaussian log-density of the innovation -> prediction error decomposition
        loglik += -0.5 * (np.log(2.0 * np.pi * S) + nu**2 / S)

        x_pred[t], P_pred[t] = x_p, P_p
        x_filt[t], P_filt[t] = x_f, P_f
        x_prev, P_prev = x_f, P_f

    return x_pred, P_pred, x_filt, P_filt, loglik


def _smoother_pass(x_pred, P_pred, x_filt, P_filt):
    """Rauch-Tung-Striebel backward pass: re-estimate each day's state
    using the WHOLE sample (past and future). F = I simplifies the algebra."""
    T = x_filt.shape[0]
    x_s = x_filt.copy()
    P_s = P_filt.copy()
    for t in range(T - 2, -1, -1):
        # Smoother gain J_t = P_filt[t] F' P_pred[t+1]^{-1}  (F = I)
        J = P_filt[t] @ np.linalg.inv(P_pred[t + 1])
        x_s[t] = x_filt[t] + J @ (x_s[t + 1] - x_pred[t + 1])
        P_s[t] = P_filt[t] + J @ (P_s[t + 1] - P_pred[t + 1]) @ J.T
    return x_s, P_s


def fit_time_varying_beta(
    stock_ret: np.ndarray,
    mkt_ret: np.ndarray,
    estimate_q_alpha: bool = False,
) -> KalmanResult:
    """Fit the time-varying alpha/beta model with (q_beta, R) -- and optionally
    q_alpha -- chosen by maximum likelihood.

    By default alpha is treated as near-constant (q_alpha pinned tiny):
    daily alpha is minuscule and hard to identify; letting it drift freely
    tends to soak up variation that belongs to beta.
    """
    y = np.asarray(stock_ret, dtype=float)
    x = np.asarray(mkt_ret, dtype=float)
    if y.shape != x.shape:
        raise ValueError("stock and market return series must be equal length")

    # Diffuse-ish prior: start beta at the OLS estimate with generous variance,
    # so early observations dominate the prior quickly.
    X = np.column_stack([np.ones_like(x), x])
    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    x0 = np.array([ols[0], ols[1]])
    P0 = np.diag([1e-4, 1.0])

    var_y = np.var(y)
    TINY_Q_ALPHA = 1e-12

    def neg_loglik(params):
        # Optimise in log-space to enforce positivity of the variances.
        if estimate_q_alpha:
            la, lb, lr = params
            qa = np.exp(la)
        else:
            lb, lr = params
            qa = TINY_Q_ALPHA
        qb, r = np.exp(lb), np.exp(lr)
        *_, ll = _filter_pass(y, x, qa, qb, r, x0, P0)
        return -ll

    # Sensible starting guesses: idio variance ~ residual variance of OLS,
    # beta drift a few orders of magnitude smaller.
    resid_var = max(np.var(y - X @ ols), 1e-10)
    if estimate_q_alpha:
        start = np.log([1e-10, 1e-5, resid_var])
    else:
        start = np.log([1e-5, resid_var])

    res = optimize.minimize(neg_loglik, start, method="Nelder-Mead",
                            options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000})

    if estimate_q_alpha:
        q_alpha, q_beta, r_obs = np.exp(res.x)
    else:
        q_alpha = TINY_Q_ALPHA
        q_beta, r_obs = np.exp(res.x)

    x_pred, P_pred, x_filt, P_filt, ll = _filter_pass(y, x, q_alpha, q_beta, r_obs, x0, P0)
    x_s, P_s = _smoother_pass(x_pred, P_pred, x_filt, P_filt)

    return KalmanResult(
        alpha_filt=x_filt[:, 0], beta_filt=x_filt[:, 1],
        alpha_smooth=x_s[:, 0], beta_smooth=x_s[:, 1],
        beta_filt_se=np.sqrt(np.maximum(P_filt[:, 1, 1], 0)),
        beta_smooth_se=np.sqrt(np.maximum(P_s[:, 1, 1], 0)),
        q_alpha=float(q_alpha), q_beta=float(q_beta), r_obs=float(r_obs),
        loglik=float(ll),
    )


def rolling_ols_beta(stock_ret: np.ndarray, mkt_ret: np.ndarray, window: int = 60) -> np.ndarray:
    """The crude baseline the Kalman filter improves on (for comparison plots)."""
    y, x = np.asarray(stock_ret), np.asarray(mkt_ret)
    out = np.full(len(y), np.nan)
    for t in range(window, len(y)):
        ys, xs = y[t - window:t], x[t - window:t]
        xc = xs - xs.mean()
        denom = (xc**2).sum()
        out[t] = (xc * (ys - ys.mean())).sum() / denom if denom > 0 else np.nan
    return out
