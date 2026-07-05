"""Pillar 2 -- Conditional volatility via GARCH(1,1).

sigma_t^2 = omega + alpha * a_{t-1}^2 + beta * sigma_{t-1}^2

- alpha: REACTION (how hard vol jumps on yesterday's shock)
- beta:  PERSISTENCE (how much of yesterday's variance carries over)
- alpha + beta < 1 required for stationarity; long-run variance
  = omega / (1 - alpha - beta).

We fit with both Normal and Student-t innovations and compare by AIC/BIC:
even after GARCH soaks up volatility clustering, standardized residuals of
equity returns are fat-tailed, so t usually wins -- and the tail choice
matters enormously for VaR (Pillar 4).

Scaling note: the arch library's optimizer is happiest when returns are in
PERCENT (multiply log returns by 100). We handle the scaling internally and
always report/forecast in decimal units so downstream code never has to care.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model

SCALE = 100.0  # decimal returns -> percent for fitting


@dataclass
class GarchFit:
    dist: str                 # "normal" or "t"
    omega: float              # in DECIMAL^2 units (rescaled back)
    alpha: float
    beta: float
    nu: float | None          # t degrees of freedom (None for normal)
    persistence: float        # alpha + beta
    uncond_vol_daily: float   # sqrt(omega / (1 - alpha - beta)), decimal
    aic: float
    bic: float
    cond_vol: pd.Series       # in-sample sigma_t, decimal units
    std_resid: pd.Series      # a_t / sigma_t -- should be ~iid if model is right
    _res: object              # underlying arch results (for forecasting)


def fit_garch(returns: pd.Series, dist: str = "t", mean: str = "Zero") -> GarchFit:
    """Fit GARCH(1,1) on a decimal log-return series.

    mean="Zero" is a deliberate choice for daily data: the daily mean is
    tiny relative to vol and estimating it adds noise. Swap to "Constant"
    to check robustness.
    """
    r_pct = returns.dropna() * SCALE
    am = arch_model(r_pct, mean=mean, vol="GARCH", p=1, q=1, dist=dist)
    res = am.fit(disp="off")

    p = res.params
    omega_pct = float(p["omega"])           # percent^2 units
    alpha = float(p["alpha[1]"])
    beta = float(p["beta[1]"])
    nu = float(p["nu"]) if "nu" in p.index else None

    persistence = alpha + beta
    omega_dec = omega_pct / SCALE**2
    uncond_var = omega_dec / (1 - persistence) if persistence < 1 else np.nan

    return GarchFit(
        dist=dist, omega=omega_dec, alpha=alpha, beta=beta, nu=nu,
        persistence=persistence,
        uncond_vol_daily=float(np.sqrt(uncond_var)) if persistence < 1 else np.nan,
        aic=float(res.aic), bic=float(res.bic),
        cond_vol=res.conditional_volatility / SCALE,
        std_resid=res.std_resid,
        _res=res,
    )


def compare_distributions(returns: pd.Series) -> pd.DataFrame:
    """Fit Normal vs Student-t innovations; lower AIC/BIC wins."""
    rows = []
    for dist in ("normal", "t"):
        f = fit_garch(returns, dist=dist)
        rows.append({
            "dist": dist, "alpha": f.alpha, "beta": f.beta,
            "persistence": f.persistence, "nu": f.nu,
            "uncond_vol_annual": f.uncond_vol_daily * np.sqrt(252),
            "aic": f.aic, "bic": f.bic,
        })
    return pd.DataFrame(rows).set_index("dist")


def simulate_paths(fit: GarchFit, horizon: int, n_paths: int = 20_000,
                   seed: int | None = 0) -> np.ndarray:
    """Simulate n_paths x horizon future DECIMAL returns from the fitted model.

    This is the engine for GARCH-based VaR: draw standardized innovations z
    from the fitted distribution, iterate the variance recursion forward from
    TODAY's conditional variance, accumulate returns. Because we start from
    current sigma_t, the resulting risk numbers respond to current conditions
    -- the whole point versus a constant-vol model.
    """
    rng = np.random.default_rng(seed)
    omega, alpha, beta = fit.omega, fit.alpha, fit.beta

    # Standardized innovations from the fitted distribution
    if fit.dist == "t":
        nu = fit.nu
        z = rng.standard_t(nu, size=(n_paths, horizon))
        z /= np.sqrt(nu / (nu - 2.0))          # rescale to unit variance
    else:
        z = rng.standard_normal((n_paths, horizon))

    # Launch from the last in-sample conditional variance and shock
    sigma2 = np.full(n_paths, fit.cond_vol.iloc[-1] ** 2)
    last_a = fit.std_resid.iloc[-1] * fit.cond_vol.iloc[-1]
    a_prev = np.full(n_paths, last_a)

    rets = np.zeros((n_paths, horizon))
    for h in range(horizon):
        sigma2 = omega + alpha * a_prev**2 + beta * sigma2
        a = np.sqrt(sigma2) * z[:, h]
        rets[:, h] = a
        a_prev = a
    return rets
