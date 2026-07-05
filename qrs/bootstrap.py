"""Pillar 3 -- Block bootstrap for dependent (financial) data.

Why not the ordinary bootstrap: resampling individual days iid destroys
volatility clustering. Any statistic that depends on the time-ordering of
returns -- drawdowns, multi-day VaR, vol persistence, GARCH parameters --
is then computed on data that no longer has the property, biasing the
interval (for risk statistics, typically toward overconfidence: the one
failure mode a risk tool cannot afford). For statistics insensitive to
ordering (e.g. the mean of serially-uncorrelated returns) the harm is
small -- which is itself worth demonstrating, see validate_synthetic.py.
Resampling contiguous blocks preserves the local dependence.

Two flavours:
- moving-block: fixed block length L
- stationary (Politis & Romano 1994): block length ~ Geometric(1/L),
  which makes the resampled series stationary and reduces sensitivity to
  any single arbitrary L. This is the default.

Block-length heuristic: L ~ n^(1/3) as a starting point; always check
sensitivity across a range of L (see block_length_sensitivity).
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def stationary_bootstrap_indices(n: int, mean_block: float,
                                 rng: np.random.Generator) -> np.ndarray:
    """Index array for one stationary-bootstrap resample of length n.

    Mechanism: pick a random start; at each step, with prob p = 1/mean_block
    jump to a fresh random start (new block), else continue sequentially
    (wrapping around the end of the sample, per Politis-Romano).
    """
    p = 1.0 / mean_block
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    # Vectorised: draw all "new block?" coin flips and fresh starts up front
    new_block = rng.random(n) < p
    fresh = rng.integers(0, n, size=n)
    for t in range(1, n):
        idx[t] = fresh[t] if new_block[t] else (idx[t - 1] + 1) % n
    return idx


def moving_block_indices(n: int, block_len: int,
                         rng: np.random.Generator) -> np.ndarray:
    """Index array for one fixed-length moving-block resample."""
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n - block_len + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_len) for s in starts])
    return idx[:n]


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_boot: int = 5_000,
    mean_block: float | None = None,
    method: str = "stationary",
    ci: float = 0.95,
    seed: int | None = 0,
) -> dict:
    """Percentile CI for `statistic(data)` under a block bootstrap.

    `data` may be 1-D (a return series) or 2-D (T x assets: rows are
    resampled together, preserving cross-asset correlation -- important
    when the statistic is a portfolio quantity).
    """
    data = np.asarray(data)
    n = data.shape[0]
    if mean_block is None:
        mean_block = max(round(n ** (1 / 3)), 2)  # heuristic starting point
    rng = np.random.default_rng(seed)

    stats = np.empty(n_boot)
    for b in range(n_boot):
        if method == "stationary":
            idx = stationary_bootstrap_indices(n, mean_block, rng)
        elif method == "block":
            idx = moving_block_indices(n, int(mean_block), rng)
        elif method == "iid":  # kept ONLY to demonstrate why it's wrong here
            idx = rng.integers(0, n, size=n)
        else:
            raise ValueError(f"unknown method {method!r}")
        stats[b] = statistic(data[idx])

    lo, hi = np.percentile(stats, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return {
        "point": float(statistic(data)),
        "lo": float(lo), "hi": float(hi),
        "se": float(stats.std(ddof=1)),
        "mean_block": float(mean_block),
        "boot_stats": stats,
    }


def block_length_sensitivity(data, statistic, lengths=(5, 10, 20, 40, 60),
                             n_boot=2_000, seed=0) -> "list[dict]":
    """Re-run the CI across block lengths -- if conclusions flip with L,
    say so in the writeup. Robustness beats a single arbitrary choice."""
    out = []
    for L in lengths:
        r = bootstrap_ci(data, statistic, n_boot=n_boot, mean_block=L, seed=seed)
        out.append({"mean_block": L, "lo": r["lo"], "hi": r["hi"], "se": r["se"]})
    return out


# ---- Common statistics -------------------------------------------------

def sharpe_ratio(returns: np.ndarray, periods: int = 252) -> float:
    r = np.asarray(returns)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods)) if sd > 0 else np.nan


def max_drawdown(returns: np.ndarray) -> float:
    """Max peak-to-trough drawdown of the cumulative log-return path
    (returned as a POSITIVE fraction, e.g. 0.35 = -35%)."""
    cum = np.cumsum(np.asarray(returns))
    peak = np.maximum.accumulate(cum)
    return float(np.max(1.0 - np.exp(cum - peak)))
