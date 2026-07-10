from __future__ import annotations

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from .config import FullConfig
from .design_pool import sobol_unit, denormalize_theta, theta_dataframe
from .io_utils import ensure_dir, write_json_atomic


def quadratic_features(unit_x: np.ndarray) -> np.ndarray:
    X = np.asarray(unit_x, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    n, d = X.shape
    cols = [np.ones(n), *[X[:, j] for j in range(d)], *[X[:, j] ** 2 for j in range(d)]]
    for i in range(d):
        for j in range(i + 1, d):
            cols.append(X[:, i] * X[:, j])
    return np.vstack(cols).T


def select_ioptimal(unit_pool: np.ndarray, n: int, ridge: float = 1e-3, seed: int = 0) -> np.ndarray:
    Xpool = np.asarray(unit_pool, dtype=float)
    Phi = quadratic_features(Xpool)
    p = Phi.shape[1]
    if int(n) < p:
        raise ValueError(
            f'I-optimal quadratic design requested n={n}, but the full quadratic basis has p={p} terms. '
            f'Use n >= {p}; recommended n >= 40–48 for noisy scalar-field reconstruction.'
        )
    if int(n) < int(np.ceil(1.5 * p)):
        warnings.warn(
            f'I-optimal design n={n} is close to saturated for p={p} quadratic terms; '
            'Hessian/interaction estimates may be unstable under noise.',
            RuntimeWarning,
            stacklevel=2,
        )

    C = (Phi.T @ Phi) / len(Phi)
    Ainv = np.eye(p) / float(ridge)
    selected_idx: list[int] = []
    available = np.ones(len(Xpool), dtype=bool)

    # Seed with a point near center for numerical stability/intercept coverage;
    # the report records this so it is not mistaken for pure greedy optimality.
    center_idx = int(np.argmin(((Xpool - 0.5) ** 2).sum(axis=1)))
    for _ in range(int(n)):
        P = Phi[available]
        idxs = np.flatnonzero(available)
        B = Ainv @ C @ Ainv
        numer = np.einsum('ij,jk,ik->i', P, B, P)
        denom = 1.0 + np.einsum('ij,jk,ik->i', P, Ainv, P)
        reductions = numer / np.maximum(denom, 1e-12)
        if not selected_idx:
            chosen = center_idx if available[center_idx] else int(idxs[np.argmax(reductions)])
        else:
            chosen = int(idxs[np.argmax(reductions)])
        x = Phi[chosen]
        Ax = Ainv @ x
        Ainv = Ainv - np.outer(Ax, Ax) / max(1.0 + float(x @ Ax), 1e-12)
        available[chosen] = False
        selected_idx.append(chosen)
    return Xpool[selected_idx]


def make_ioptimal_design(cfg: FullConfig, out_dir: str | Path, n: int | None = None) -> Path:
    out_dir = ensure_dir(out_dir)
    n = int(n or cfg.design.ioptimal_n)
    pool = sobol_unit(max(cfg.design.pool_size, n * 32), d=cfg.design.theta_dim, seed=cfg.design.seed + 303)
    selected_unit = select_ioptimal(pool, n=n, ridge=cfg.design.ioptimal_ridge, seed=cfg.design.seed)
    theta = denormalize_theta(selected_unit, lo=cfg.design.theta_min, hi=cfg.design.theta_max)
    df = theta_dataframe(theta, 'i_optimal_quadratic', extra={'role': 'ioptimal', 'initial_replicates': int(cfg.design.reconstruction_reps)})
    path = out_dir / 'theta_ioptimal.csv'
    df.to_csv(path, index=False)
    write_json_atomic(out_dir / 'theta_ioptimal_report.json', {
        'n': n,
        'pool_size': int(len(pool)),
        'ridge': float(cfg.design.ioptimal_ridge),
        'basis_terms': int(quadratic_features(np.zeros((1, cfg.design.theta_dim))).shape[1]),
        'center_seeded': True,
    })
    return path
