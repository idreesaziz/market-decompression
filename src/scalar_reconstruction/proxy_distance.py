from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Sequence
import hashlib
import json
import os
import warnings

import numpy as np
import pandas as pd

from .config import FullConfig, market_by_name
from .data import load_target
from .design_pool import read_theta_csv
from .io_utils import ensure_dir, theta_to_text


def _acf(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) <= lag + 2:
        return 0.0
    a = x[:-lag] - np.mean(x[:-lag]); b = x[lag:] - np.mean(x[lag:])
    den = np.sqrt(np.sum(a*a) * np.sum(b*b))
    return float(np.sum(a*b) / den) if den > 0 else 0.0


def return_features(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=float); r = r[np.isfinite(r)]
    if len(r) == 0:
        return np.zeros(20)
    mu = np.mean(r); sd = np.std(r) + 1e-12
    z = (r - mu) / sd
    qs = np.quantile(r, [0.01, 0.05, 0.50, 0.95, 0.99])
    absr = np.abs(r)
    rv20 = np.array([np.sqrt(np.mean(r[max(0, i-19):i+1]**2)+1e-12) for i in range(len(r))])
    jump = np.mean(absr > 3.0 * sd)
    return np.asarray([
        mu, sd, float(np.mean(z**3)), float(np.mean(z**4)), *qs.tolist(),
        float(np.mean(absr)), float(np.std(absr)),
        _acf(r, 1), _acf(r, 5), _acf(r, 10), _acf(absr, 1), _acf(absr, 5), _acf(absr, 10),
        float(np.mean(rv20)), float(np.std(rv20)), float(jump)
    ], dtype=float)




def window_features_matrix(r: np.ndarray, window: int = 192, stride: int = 32, max_windows: int = 512) -> np.ndarray:
    """Return a matrix of feature embeddings over rolling return windows.

    This supports a stronger realism proxy than MMD on raw scalar returns: MMD
    over multivariate window-level financial summaries. It captures volatility
    clustering, tails, and autocorrelation differences that marginal-return
    distances miss.
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < max(8, window):
        return return_features(r)[None, :]
    starts = np.arange(0, len(r) - window + 1, max(1, stride))
    if len(starts) > max_windows:
        # Deterministic spread across the series, not random subsampling.
        idx = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
        starts = starts[idx]
    feats = [return_features(r[int(i):int(i) + window]) for i in starts]
    return np.vstack(feats)

def feature_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    denom = np.maximum(np.abs(a) + np.abs(b), 1e-8)
    return float(np.sqrt(np.mean(((a - b) / denom) ** 2)))


def wasserstein_1d(a: np.ndarray, b: np.ndarray, n_quantiles: int = 101) -> float:
    qs = np.linspace(0.0, 1.0, n_quantiles)
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def _standardize_pooled(A: np.ndarray, B: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Z-score A and B using pooled feature statistics.

    This is essential for feature-window MMD: the financial summary vector
    contains coordinates with very different raw scales (e.g. mean returns,
    volatility, skewness, kurtosis, ACFs). Without pooled standardization, an
    isotropic RBF MMD is dominated by whichever feature has the largest numeric
    variance, usually kurtosis, rather than measuring multivariate realism.
    """
    X = np.vstack([A, B])
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > eps), sd, 1.0)
    return (A - mu) / sd, (B - mu) / sd


def mmd_rbf_features(
    A: np.ndarray,
    B: np.ndarray,
    gamma: float | None = None,
    *,
    standardize: bool = True,
) -> float:
    """Biased squared MMD with an RBF kernel.

    For multivariate feature embeddings, ``standardize=True`` uses pooled
    z-scoring before applying the isotropic RBF kernel. This prevents large-scale
    features such as kurtosis from overwhelming small-scale but important
    features such as volatility, quantiles, and autocorrelations. For raw
    one-dimensional returns, standardization is harmless and makes the median
    heuristic comparable across markets.
    """
    A = np.asarray(A, dtype=float); B = np.asarray(B, dtype=float)
    if A.ndim == 1: A = A[:, None]
    if B.ndim == 1: B = B[:, None]
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
    if standardize:
        A, B = _standardize_pooled(A, B)
    X = np.vstack([A, B])
    if gamma is None:
        # Median heuristic on the standardized pooled feature space.
        D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
        med = np.median(D[D > 0]) if np.any(D > 0) else 1.0
        gamma = 1.0 / max(med, 1e-12)
    def K(U, V):
        return np.exp(-gamma * ((U[:, None, :] - V[None, :, :]) ** 2).sum(axis=2))
    return float(np.mean(K(A, A)) + np.mean(K(B, B)) - 2.0 * np.mean(K(A, B)))


def _synthetic_returns(theta: Sequence[float], cfg: FullConfig, n: int, seed: int = 0) -> np.ndarray:
    """Generate synthetic returns for proxy-distance calculations.

    In a real paper run, failures in the real simulator must be visible rather
    than silently replaced by a toy fallback. The fallback is therefore allowed
    only when the environment variable
    ``SCALAR_RECON_ALLOW_SYNTHETIC_FALLBACK=1`` is set, which is useful for dry
    runs and package smoke tests.
    """
    try:
        from mde_search.simulator import SimulatorConfig, generate_synthetic_dataset
        sim_cfg = SimulatorConfig(L=cfg.data.seq_len, horizons=tuple(cfg.data.horizons), module=cfg.train.family)
        raw = generate_synthetic_dataset([float(x) for x in theta], n, seed=seed, cfg=sim_cfg)
        if isinstance(raw, dict) and 'returns' in raw:
            return np.asarray(raw['returns'], dtype=float).ravel()
        if isinstance(raw, dict) and 'X' in raw:
            return np.asarray(raw['X'], dtype=float)[:, :, 0].ravel()
        raise RuntimeError(f"Simulator returned unsupported object type/keys: {type(raw)}")
    except Exception as e:
        allow_fallback = os.environ.get('SCALAR_RECON_ALLOW_SYNTHETIC_FALLBACK', '').strip().lower() in {'1', 'true', 'yes'}
        if not allow_fallback:
            raise RuntimeError(
                'Failed to generate synthetic returns from mde_search.simulator. '
                'Set SCALAR_RECON_ALLOW_SYNTHETIC_FALLBACK=1 only for dry-run smoke tests '
                'if you explicitly want the deterministic toy fallback.'
            ) from e
        warnings.warn(
            f'Using deterministic toy synthetic fallback for proxy distances because simulator failed: {e!r}',
            RuntimeWarning,
            stacklevel=2,
        )
    # Deterministic fallback for dry runs only: pseudo synthetic from theta.
    key = json.dumps([round(float(x), 8) for x in theta], sort_keys=True).encode('utf-8')
    h = int(hashlib.blake2b(key, digest_size=8).hexdigest(), 16) % (2**32 - 1)
    rng = np.random.default_rng((int(seed) + h) % (2**32 - 1))
    scale = 0.01 * (1.0 + 0.15 * np.tanh(float(theta[1])))
    return rng.standard_t(df=5, size=n) * scale


def compute_proxy_distances(config_path: str | Path, theta_csv: str | Path, out_dir: str | Path) -> Path:
    from .config import load_config
    cfg = load_config(config_path)
    out_dir = ensure_dir(out_dir)
    design = read_theta_csv(theta_csv)
    rows = []
    target_feats: Dict[str, np.ndarray] = {}
    target_windows: Dict[str, np.ndarray] = {}
    target_returns: Dict[str, np.ndarray] = {}
    for m in cfg.markets:
        raw = load_target(m, cfg.data)
        r = np.asarray(raw['returns'], dtype=float)
        target_returns[m.name] = r
        target_feats[m.name] = return_features(r)
        target_windows[m.name] = window_features_matrix(r, window=cfg.data.seq_len, stride=max(1, cfg.data.seq_len // 6), max_windows=512)
    for _, row in design.iterrows():
        theta = [float(row[f'theta_{j+1}']) for j in range(6)]
        sim_r = _synthetic_returns(theta, cfg, n=5000, seed=cfg.design.seed + 1200)
        sim_feat = return_features(sim_r)
        sim_windows = window_features_matrix(sim_r, window=cfg.data.seq_len, stride=max(1, cfg.data.seq_len // 6), max_windows=512)
        for m in cfg.markets:
            rr = target_returns[m.name]
            rec = {
                'theta_id': row['theta_id'], 'theta': theta_to_text(theta), 'market': m.name,
                'D_feature': feature_distance(sim_feat, target_feats[m.name]),
                'D_wasserstein_returns': wasserstein_1d(sim_r, rr),
                'D_mmd_returns': mmd_rbf_features(sim_r[:2000, None], rr[-2000:, None], standardize=True),
                'D_mmd_feature_windows': mmd_rbf_features(sim_windows, target_windows[m.name], standardize=True),
            }
            for j, v in enumerate(theta): rec[f'theta_{j+1}'] = float(v)
            rows.append(rec)
    path = out_dir / 'proxy_distance.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
