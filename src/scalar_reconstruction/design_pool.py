from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .io_utils import theta_key, theta_to_text, stable_id


def normalize_theta(theta: np.ndarray, lo: float = -3.0, hi: float = 3.0) -> np.ndarray:
    return (np.asarray(theta, dtype=float) - lo) / (hi - lo)


def denormalize_theta(x: np.ndarray, lo: float = -3.0, hi: float = 3.0) -> np.ndarray:
    return lo + (hi - lo) * np.asarray(x, dtype=float)


def sobol_unit(n: int, d: int = 6, seed: int = 0) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, d), dtype=float)
    eng = qmc.Sobol(d=d, scramble=True, seed=int(seed))
    return eng.random(int(n))


def sobol_theta_pool(n: int, d: int = 6, seed: int = 0, lo: float = -3.0, hi: float = 3.0) -> pd.DataFrame:
    X = denormalize_theta(sobol_unit(n, d=d, seed=seed), lo=lo, hi=hi)
    return theta_dataframe(X, design_source='candidate_pool')


def maximin_select(unit_pool: np.ndarray, n: int, seed: int = 0, initial: np.ndarray | None = None) -> np.ndarray:
    unit_pool = np.asarray(unit_pool, dtype=float)
    rng = np.random.default_rng(seed)
    if len(unit_pool) == 0 or n <= 0:
        return np.zeros((0, unit_pool.shape[1] if unit_pool.ndim == 2 else 0))
    selected = []
    if initial is not None and len(initial) > 0:
        for x in np.asarray(initial, dtype=float):
            selected.append(x)
    else:
        selected.append(unit_pool[int(rng.integers(len(unit_pool)))])
    while len(selected) < n:
        S = np.vstack(selected)
        d2 = ((unit_pool[:, None, :] - S[None, :, :]) ** 2).sum(axis=2)
        min_d2 = d2.min(axis=1)
        idx = int(np.argmax(min_d2))
        selected.append(unit_pool[idx])
    return np.vstack(selected[:n])


def dedup_thetas(df: pd.DataFrame, decimals: int = 6) -> pd.DataFrame:
    seen = set(); keep = []
    for _, row in df.iterrows():
        theta = [float(row[f'theta_{j+1}']) for j in range(6)] if 'theta_1' in df.columns else json.loads(row['theta'])
        key = theta_key(theta, decimals=decimals)
        if key not in seen:
            seen.add(key); keep.append(True)
        else:
            keep.append(False)
    out = df.loc[keep].copy().reset_index(drop=True)
    return out


def theta_dataframe(thetas: np.ndarray, design_source: str, extra: Dict[str, object] | None = None) -> pd.DataFrame:
    rows = []
    for i, th in enumerate(np.asarray(thetas, dtype=float)):
        rec = {'theta': theta_to_text(th.tolist()), 'design_source': design_source, 'design_index': int(i)}
        for j, v in enumerate(th.tolist()):
            rec[f'theta_{j+1}'] = float(v)
        rec['theta_id'] = stable_id({'theta': [round(float(x), 8) for x in th.tolist()]})
        if extra:
            rec.update(extra)
        rows.append(rec)
    return pd.DataFrame(rows)


def read_theta_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'theta' not in df.columns and all(f'theta_{j+1}' in df.columns for j in range(6)):
        df['theta'] = [theta_to_text([row[f'theta_{j+1}'] for j in range(6)]) for _, row in df.iterrows()]
    if 'theta_id' not in df.columns:
        df['theta_id'] = [stable_id({'theta': [round(float(x), 8) for x in json.loads(t)]}) for t in df['theta']]
    return df
