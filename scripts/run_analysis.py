"""End-to-end risk analysis of the SOFI/QBTS sleeve on REAL data.

Run this LOCALLY (needs internet for yfinance):
    pip install yfinance
    python scripts/run_analysis.py

Produces:
  figures/beta_paths.png        -- Kalman beta migration per asset
  figures/cond_vol.png          -- GARCH conditional volatility per asset
  results_summary.txt           -- all headline numbers

Note the flagship step at the end (bootstrap CI on the GARCH-VaR) refits
a GARCH per bootstrap replicate -- a few minutes of compute. Reduce
N_BOOT for a quick pass.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qrs.data import load_prices, log_returns, portfolio_returns
from qrs.kalman import fit_time_varying_beta, rolling_ols_beta
from qrs.garch_model import compare_distributions, fit_garch
from qrs.bootstrap import bootstrap_ci, sharpe_ratio, max_drawdown, block_length_sensitivity
from qrs.risk import historical_var_es, parametric_var_es, garch_var_es, bootstrap_var_ci

# ---------------------------------------------------------------- config
TICKERS = ["SOFI", "QBTS"]
MARKET = "SPY"
WEIGHTS = {"SOFI": 0.5, "QBTS": 0.5}   # <-- set to your actual sleeve weights
START = "2021-06-01"                    # SOFI SPAC-era start; QBTS joins Aug 2022
VAR_LEVEL = 0.95
N_BOOT_VAR = 200                        # flagship step cost; raise overnight

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)
lines = []
def say(s=""):
    print(s); lines.append(str(s))

# ---------------------------------------------------------------- data
prices = load_prices(TICKERS + [MARKET], start=START)
rets = log_returns(prices)
say(f"Sample: {rets.index[0].date()} -> {rets.index[-1].date()}  ({len(rets)} days)")
say("NOTE: common-day alignment trims history to QBTS's listing (Aug 2022).")
say("For a longer SOFI-only beta study, rerun with TICKERS=['SOFI'].")

# ---------------------------------------------------------------- Pillar 1
say("\n" + "=" * 70 + "\nPILLAR 1 -- time-varying beta (Kalman smoothed)\n" + "=" * 70)
fig, ax = plt.subplots(figsize=(11, 5))
for tkr in TICKERS:
    kf = fit_time_varying_beta(rets[tkr].values, rets[MARKET].values)
    say(f"{tkr}: beta start={kf.beta_smooth[0]:.2f}  "
        f"end={kf.beta_smooth[-1]:.2f} (+/-{2*kf.beta_smooth_se[-1]:.2f})  "
        f"min={kf.beta_smooth.min():.2f}  max={kf.beta_smooth.max():.2f}  "
        f"q_beta={kf.q_beta:.2e}")
    ax.plot(rets.index, kf.beta_smooth, label=f"{tkr} (Kalman smoothed)")
    ax.fill_between(rets.index, kf.beta_smooth - 2*kf.beta_smooth_se,
                    kf.beta_smooth + 2*kf.beta_smooth_se, alpha=0.15)
    ax.plot(rets.index, rolling_ols_beta(rets[tkr].values, rets[MARKET].values),
            lw=0.7, alpha=0.5, linestyle="--", label=f"{tkr} (60d OLS)")
ax.axhline(1.0, color="grey", lw=0.8)
ax.set_title("Time-varying market beta vs SPY"); ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "beta_paths.png"), dpi=130)

# ---------------------------------------------------------------- Pillar 2
say("\n" + "=" * 70 + "\nPILLAR 2 -- GARCH(1,1): Normal vs Student-t\n" + "=" * 70)
fits = {}
fig, ax = plt.subplots(figsize=(11, 4.5))
for tkr in TICKERS:
    cmp = compare_distributions(rets[tkr])
    best = cmp["aic"].idxmin()
    fits[tkr] = fit_garch(rets[tkr], dist=best)
    f = fits[tkr]
    say(f"\n{tkr} (best dist by AIC: {best})")
    say(cmp.round(4).to_string())
    say(f"  persistence={f.persistence:.3f}  "
        f"annualised uncond vol={f.uncond_vol_daily*np.sqrt(252):.1%}")
    ax.plot(rets.index, f.cond_vol * np.sqrt(252), label=f"{tkr} cond vol (ann.)")
ax.set_title("GARCH conditional volatility (annualised)")
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "cond_vol.png"), dpi=130)

# ---------------------------------------------------------------- Pillar 3
say("\n" + "=" * 70 + "\nPILLAR 3 -- block-bootstrap CIs on sleeve statistics\n" + "=" * 70)
port = portfolio_returns(rets[TICKERS], WEIGHTS)
for name, stat in [("Sharpe (ann.)", sharpe_ratio), ("Max drawdown", max_drawdown)]:
    r = bootstrap_ci(port.values, stat, n_boot=5000, mean_block=20)
    say(f"{name:14s}: {r['point']:.3f}   95% CI [{r['lo']:.3f}, {r['hi']:.3f}]")
say("\nBlock-length sensitivity (Sharpe):")
for row in block_length_sensitivity(port.values, sharpe_ratio):
    say(f"  L={row['mean_block']:>3}: CI [{row['lo']:.3f}, {row['hi']:.3f}]")

# ---------------------------------------------------------------- Pillar 4
say("\n" + "=" * 70 + "\nPILLAR 4 -- 1-day 95% VaR / ES of the sleeve\n" + "=" * 70)
port_fit = fit_garch(port, dist="t")
h = historical_var_es(port.values, VAR_LEVEL)
p = parametric_var_es(port.values, VAR_LEVEL, dist="t")
g = garch_var_es(port_fit, VAR_LEVEL, horizon=1)
say(f"historical    : VaR={h['var']:.2%}  ES={h['es']:.2%}   (unconditional)")
say(f"parametric-t  : VaR={p['var']:.2%}  ES={p['es']:.2%}   (unconditional)")
say(f"GARCH-sim (t) : VaR={g['var']:.2%}  ES={g['es']:.2%}   (conditional on TODAY)")
g10 = garch_var_es(port_fit, VAR_LEVEL, horizon=10)
say(f"GARCH-sim 10d : VaR={g10['var']:.2%}  ES={g10['es']:.2%}")

say(f"\nFlagship: block-bootstrap CI on the GARCH-VaR (n_boot={N_BOOT_VAR})...")
bv = bootstrap_var_ci(port.values, level=VAR_LEVEL, n_boot=N_BOOT_VAR)
say(f"1-day {VAR_LEVEL:.0%} VaR = {bv['var_point']:.2%},  "
    f"90% CI [{bv['lo']:.2%}, {bv['hi']:.2%}]  "
    f"(n_eff={bv['n_effective']})")
say("\nInterpretation: the CI is ESTIMATION uncertainty of the risk number;")
say("model-form uncertainty (is GARCH(1,1)-t right?) is additional and")
say("belongs in the limitations section.")

with open(os.path.join(os.path.dirname(__file__), "..", "results_summary.txt"), "w") as fh:
    fh.write("\n".join(lines))
say("\nSaved: results_summary.txt, figures/beta_paths.png, figures/cond_vol.png")
