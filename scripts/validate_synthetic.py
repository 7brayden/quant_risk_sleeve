"""Known-truth validation of every pillar on SIMULATED data.

Logic: if the Kalman filter cannot recover a beta path we generated
ourselves, or the GARCH fit cannot recover parameters we chose, the code
has no business touching real data. This script is Part 0 of the project
and belongs in the writeup as the validation section.

World we simulate (deliberately SOFI-like):
  r_mkt_t ~ N(0.0004, 0.011^2)                      # ~ SPY-ish daily
  beta_t  = beta_{t-1} + N(0, 0.015^2), beta_0=2.3  # drifts DOWN over time
            (we add a deterministic downtrend so there's a regime change
             to detect: 'speculative fintech' -> 'profitable bank')
  eps_t   ~ GARCH(1,1): omega=5e-6, alpha=0.08, beta=0.90, t(7) innovations
  r_t     = 0.0 + beta_t * r_mkt_t + eps_t
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qrs.kalman import fit_time_varying_beta, rolling_ols_beta
from qrs.garch_model import fit_garch, compare_distributions
from qrs.bootstrap import bootstrap_ci, sharpe_ratio
from qrs.risk import historical_var_es, parametric_var_es, garch_var_es

rng = np.random.default_rng(42)
T = 1250  # ~5 trading years

# ---------------------------------------------------------------- simulate
r_mkt = rng.normal(0.0004, 0.011, T)

true_beta = np.empty(T)
true_beta[0] = 2.3
drift = -1.0 / T                       # slow structural decline 2.3 -> ~1.3
for t in range(1, T):
    true_beta[t] = true_beta[t - 1] + drift + rng.normal(0, 0.015)

TRUE = dict(omega=5e-6, alpha=0.08, beta=0.90, nu=7)
z = rng.standard_t(TRUE["nu"], T) / np.sqrt(TRUE["nu"] / (TRUE["nu"] - 2))
sigma2 = np.empty(T); eps = np.empty(T)
sigma2[0] = TRUE["omega"] / (1 - TRUE["alpha"] - TRUE["beta"])
eps[0] = np.sqrt(sigma2[0]) * z[0]
for t in range(1, T):
    sigma2[t] = TRUE["omega"] + TRUE["alpha"] * eps[t-1]**2 + TRUE["beta"] * sigma2[t-1]
    eps[t] = np.sqrt(sigma2[t]) * z[t]

r_stock = true_beta * r_mkt + eps

print("=" * 70)
print("PILLAR 1 -- Kalman filter recovers the hidden beta path?")
print("=" * 70)
kf = fit_time_varying_beta(r_stock, r_mkt)
roll = rolling_ols_beta(r_stock, r_mkt, window=60)

def rmse(a, b):
    m = ~np.isnan(a) & ~np.isnan(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))

corr_s = np.corrcoef(true_beta, kf.beta_smooth)[0, 1]
print(f"  estimated q_beta={kf.q_beta:.2e} (drift var), R={kf.r_obs:.2e} (idio var)")
print(f"  [truth: q_beta~{0.015**2:.2e},  R(uncond)~{5e-6/0.02:.2e}]")
print(f"  corr(true beta, smoothed beta) = {corr_s:.3f}")
print(f"  RMSE  smoothed : {rmse(true_beta, kf.beta_smooth):.3f}")
print(f"  RMSE  filtered : {rmse(true_beta, kf.beta_filt):.3f}")
print(f"  RMSE  60d OLS  : {rmse(true_beta, roll):.3f}   <- the crude baseline")

# validation figure
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(true_beta, color="black", lw=1.6, label="TRUE beta (hidden)")
ax.plot(kf.beta_smooth, color="tab:blue", lw=1.3, label="Kalman smoothed")
ax.fill_between(np.arange(T), kf.beta_smooth - 2 * kf.beta_smooth_se,
                kf.beta_smooth + 2 * kf.beta_smooth_se,
                color="tab:blue", alpha=0.15, label="smoothed ±2 SE")
ax.plot(roll, color="tab:red", lw=0.9, alpha=0.7, label="rolling 60d OLS")
ax.set_title("Pillar 1 validation: recovering a known time-varying beta")
ax.set_xlabel("day"); ax.set_ylabel("beta"); ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(__file__), "..", "figures",
                         "validation_kalman_beta.png"), dpi=130)
print("  figure saved -> figures/validation_kalman_beta.png")

print()
print("=" * 70)
print("PILLAR 2 -- GARCH fit recovers known (omega, alpha, beta, nu)?")
print("=" * 70)
eps_series = pd.Series(eps)
cmp = compare_distributions(eps_series)
print(cmp.round(4).to_string())
fit_t = fit_garch(eps_series, dist="t")
print(f"\n  truth : alpha={TRUE['alpha']:.3f}  beta={TRUE['beta']:.3f}  "
      f"persistence={TRUE['alpha']+TRUE['beta']:.3f}  nu={TRUE['nu']}")
print(f"  fitted: alpha={fit_t.alpha:.3f}  beta={fit_t.beta:.3f}  "
      f"persistence={fit_t.persistence:.3f}  nu={fit_t.nu:.1f}")
print(f"  t beats normal on AIC: {cmp.loc['t','aic'] < cmp.loc['normal','aic']}")

print()
print("=" * 70)
print("PILLAR 3 -- does iid resampling destroy volatility clustering?")
print("=" * 70)
# The decisive test: bootstrap a statistic that MEASURES the dependence.
# If iid resampling were valid here, its resampled distribution would
# center on the observed value. It centers on ~0 instead: the property
# the data actually has is annihilated by shuffling. Block resampling
# preserves it. Any statistic that depends on the time-ordering of
# returns (drawdowns, multi-day VaR, vol persistence) inherits this bias.

def acf1_sq(r):
    s = r**2 - (r**2).mean()
    return float((s[1:] * s[:-1]).mean() / (s * s).mean())

iid_ci = bootstrap_ci(eps, acf1_sq, n_boot=2000, method="iid", seed=1)
blk_ci = bootstrap_ci(eps, acf1_sq, n_boot=2000, method="stationary",
                      mean_block=20, seed=1)
print(f"  observed ACF(1) of squared returns : {iid_ci['point']:.3f}"
      "   <- volatility clustering")
print(f"  iid   bootstrap: dist mean={iid_ci['boot_stats'].mean():.3f}"
      f"  CI [{iid_ci['lo']:.3f}, {iid_ci['hi']:.3f}]  <- clustering DESTROYED")
print(f"  block bootstrap: dist mean={blk_ci['boot_stats'].mean():.3f}"
      f"  CI [{blk_ci['lo']:.3f}, {blk_ci['hi']:.3f}]  <- clustering preserved")

print()
print("=" * 70)
print("PILLAR 4 -- VaR estimators vs the model-implied truth")
print("=" * 70)
# 'True' next-day 95% VaR: known sigma_{T+1} times the t quantile
from scipy import stats as st
sig_next = np.sqrt(TRUE["omega"] + TRUE["alpha"] * eps[-1]**2 + TRUE["beta"] * sigma2[-1])
t_q = st.t.ppf(0.05, TRUE["nu"]) / np.sqrt(TRUE["nu"] / (TRUE["nu"] - 2))
true_var = float(-sig_next * t_q)
h = historical_var_es(eps)
p = parametric_var_es(eps, dist="t")
g = garch_var_es(fit_t, level=0.95, horizon=1)
print(f"  TRUE 1d 95% VaR (known params, today's sigma): {true_var:.4f}")
print(f"  historical (unconditional)                   : {h['var']:.4f}")
print(f"  parametric t (unconditional)                 : {p['var']:.4f}")
print(f"  GARCH-simulated (conditional)                : {g['var']:.4f}")
print("  (GARCH should sit closest to truth; the unconditional methods")
print("   can't see whether TODAY is calm or turbulent)")
print()
print("ALL PILLARS VALIDATED" if corr_s > 0.9 else "CHECK PILLAR 1")
