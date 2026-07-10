from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def estimate_noise_table(field_csv: str | Path, metric: str = 'log_skill') -> pd.DataFrame:
    """Estimate per-site sampling noise from replicated field observations.

    Returns one row per (market, budget, theta_id).  Sites with one replicate keep
    NaN variance; surrogate.py later replaces those with a robust pooled default.
    """
    df = pd.read_csv(field_csv)
    rows = []
    for (market, b, theta_id, theta), g in df.groupby(['market', 'b', 'theta_id', 'theta'], dropna=False):
        vals = g[metric].dropna().to_numpy(dtype=float)
        rec = {'market': market, 'b': int(b), 'theta_id': theta_id, 'theta': theta, 'metric': metric, 'n': int(len(vals))}
        rec['mean'] = float(np.mean(vals)) if len(vals) else np.nan
        rec['var'] = float(np.var(vals, ddof=1)) if len(vals) > 1 else np.nan
        rec['std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
        rec['sem'] = float(rec['std'] / np.sqrt(len(vals))) if len(vals) > 1 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def pooled_noise(noise_df: pd.DataFrame, min_noise: float = 1e-6) -> float:
    vals = noise_df['var'].dropna().to_numpy(dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    return float(max(np.median(vals), min_noise)) if len(vals) else float(min_noise)
