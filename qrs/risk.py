"""Pillar 4 -- Value at Risk and Expected Shortfall.

Conventions: VaR and ES are reported as POSITIVE decimal loss fractions.
"1-day 95% VaR = 0.032" means: on 95% of days the loss is below 3.2%.
ES is the average loss GIVEN you're beyond VaR -- coherent where VaR is not,
and the number regulators have shifted toward. Report both.

Three estimators, in increasing sophistication:
1. historical  -- empirical quantile of past returns (assumes past ~ future)
2. parametric  -- Normal or Student-t quantile scaled by sample vol
3. garch_var   -- simulate forward paths from the fitted GARCH launched at
                  TODAY's conditional vol; risk responds to current regime.

The flagship composition: garch_var gives the point estimate,
bootstrap_var_ci puts an honest confidence interval around it.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as st

from .bootstrap import bootstrap_ci
from .garch_model import GarchFit, fit_garch, simulate_paths


# ---- 1. Historical ------------------------------------------------------

def historical_var_es(returns: np.ndarray, level: float = 0.95) -> dict:
    r = np.asarray(returns)
    q = np.quantile(r, 1 - level)          # e.g. 5th percentile (a negative return)
    tail = r[r <= q]
    return {"var": float(-q), "es": float(-tail.mean()) if len(tail) else float(-q)}


# ---- 2. Parametric ------------------------------------------------------

def parametric_var_es(returns: np.ndarray, level: float = 0.95,
                      dist: str = "t") -> dict:
    r = np.asarray(returns)
    mu, sd = r.mean(), r.std(ddof=1)
    a = 1 - level
    if dist == "normal":
        z = st.norm.ppf(a)
        es_z = st.norm.pdf(z) / a                     # standard normal ES factor
        var = -(mu + sd * z)
        es = -(mu - sd * es_z)
    elif dist == "t":
        nu, loc, scale = st.t.fit(r)                  # fit df from the data
        q = st.t.ppf(a, nu, loc, scale)
        var = -q
        # ES for Student-t (standard closed form), then rescale
        t_q = (q - loc) / scale
        es_std = (st.t.pdf(t_q, nu) / a) * (nu + t_q**2) / (nu - 1)
        es = -(loc - scale * es_std)
    else:
        raise ValueError(dist)
    return {"var": float(var), "es": float(es)}


# ---- 3. GARCH-simulation (conditional) ----------------------------------

def garch_var_es(fit: GarchFit, level: float = 0.95, horizon: int = 1,
                 n_paths: int = 50_000, seed: int = 0) -> dict:
    """VaR/ES from simulated forward paths of the fitted GARCH.

    horizon > 1 aggregates the h daily log returns per path (log returns
    add across time), capturing the vol dynamics over the window rather
    than a naive sqrt(h) scaling.
    """
    paths = simulate_paths(fit, horizon=horizon, n_paths=n_paths, seed=seed)
    agg = paths.sum(axis=1)                # h-day log return per path
    q = np.quantile(agg, 1 - level)
    tail = agg[agg <= q]
    return {"var": float(-q), "es": float(-tail.mean()), "horizon": horizon}


# ---- The flagship: CI on the GARCH-VaR itself ---------------------------

def bootstrap_var_ci(returns, level: float = 0.95, horizon: int = 1,
                     n_boot: int = 200, mean_block: float | None = None,
                     dist: str = "t", ci: float = 0.90,
                     n_paths: int = 5_000, seed: int = 0) -> dict:
    """Block-bootstrap CI for the GARCH-simulation VaR.

    Each bootstrap replicate: resample the return HISTORY in blocks
    -> refit GARCH -> recompute simulated VaR. The spread of those VaRs
    is the estimation uncertainty of the risk number itself.

    n_boot is modest (default 200) because each replicate refits a GARCH;
    this is the expensive-but-honest step. Run overnight with more if needed.

    NOTE: this captures PARAMETER-estimation uncertainty. Model-form
    uncertainty (is GARCH(1,1)-t the right model at all?) is not in the
    interval -- say so in the writeup.
    """
    import pandas as pd
    r = np.asarray(returns, dtype=float)

    def var_stat(sample: np.ndarray) -> float:
        try:
            f = fit_garch(pd.Series(sample), dist=dist)
            return garch_var_es(f, level=level, horizon=horizon,
                                n_paths=n_paths, seed=seed)["var"]
        except Exception:
            return np.nan  # rare degenerate resample; excluded below

    res = bootstrap_ci(r, var_stat, n_boot=n_boot,
                       mean_block=mean_block, ci=ci, seed=seed)
    boot = res["boot_stats"]
    ok = boot[~np.isnan(boot)]
    lo, hi = np.percentile(ok, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return {"var_point": res["point"], "lo": float(lo), "hi": float(hi),
            "n_effective": int(ok.size), "level": level, "ci": ci}
