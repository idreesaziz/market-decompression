from __future__ import annotations

from typing import Dict, Iterable, Mapping
import numpy as np


def paired_skill(mse_transfer: float, mse_forecast: float, eps: float = 1e-12, clip_ratio: float | None = 1e6) -> Dict[str, float | bool]:
    mf = max(float(mse_forecast), eps)
    mt = max(float(mse_transfer), eps)
    ratio = mt / mf
    if clip_ratio is not None:
        ratio = min(max(ratio, 1.0 / float(clip_ratio)), float(clip_ratio))
    skill_T = 1.0 - ratio
    log_skill = float(np.log(mf) - np.log(mt))
    return {
        'skill_T': float(skill_T),
        'log_skill': float(log_skill),
        'ratio': float(ratio),
        'hit': bool(mt < mf),
    }


def bootstrap_ci(vals: np.ndarray, n_boot: int = 2000, seed: int = 123, q=(0.025, 0.975)) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return (float('nan'), float('nan'))
    if len(vals) == 1:
        return (float(vals[0]), float(vals[0]))
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(vals, size=(int(n_boot), len(vals)), replace=True), axis=1)
    lo, hi = np.quantile(means, q)
    return float(lo), float(hi)


def summarize_values(vals: Iterable[float], n_boot: int = 2000, seed: int = 123) -> Dict[str, float | int]:
    arr = np.asarray(list(vals), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {'n': 0, 'mean': float('nan'), 'median': float('nan'), 'std': float('nan'), 'ci95_low': float('nan'), 'ci95_high': float('nan')}
    lo, hi = bootstrap_ci(arr, n_boot=n_boot, seed=seed)
    return {
        'n': int(len(arr)),
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        'ci95_low': lo,
        'ci95_high': hi,
    }
