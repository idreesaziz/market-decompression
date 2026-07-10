from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .config import FullConfig
from .design_pool import normalize_theta, denormalize_theta, sobol_unit
from .io_utils import ensure_dir


@dataclass
class GroupSurrogate:
    market: str
    b: int
    metric: str
    theta: np.ndarray
    y_mean: np.ndarray
    n_rep: np.ndarray
    noise_var: np.ndarray
    model: object | None
    fallback: Dict[str, np.ndarray] | None
    y_offset: float = 0.0
    y_scale: float = 1.0
    lo: float = -3.0
    hi: float = 3.0

    def predict(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict the latent transfer field F at theta.

        Internally sklearn/fallback models are fit on manually standardized targets. We
        unstandardize both the posterior mean and latent posterior std here. This avoids
        sklearn's normalize_y/alpha unit mismatch: the alpha vector must be in the same
        units as the y values passed to fit().
        """
        X = normalize_theta(np.asarray(theta, dtype=float), self.lo, self.hi)
        if X.ndim == 1:
            X = X[None, :]
        if self.model is not None:
            mu_s, std_s = self.model.predict(X, return_std=True)
            mu = self.y_offset + self.y_scale * np.asarray(mu_s, dtype=float)
            std = abs(self.y_scale) * np.asarray(std_s, dtype=float)
            return mu, np.maximum(std, 1e-9)
        # Fixed-kernel fallback, also trained in standardized y units.
        fb = self.fallback or {}
        Xtr = fb['X']; Kinv = fb['Kinv']; y_s = fb['y_s']; y0_s = fb['y0_s']; amp = fb['amp']; ell = fb['ell']
        Kx = amp * np.exp(-0.5 * ((X[:, None, :] - Xtr[None, :, :]) ** 2).sum(axis=2) / max(float(ell) ** 2, 1e-12))
        mu_s = y0_s + Kx @ Kinv @ (y_s - y0_s)
        v_s = np.maximum(float(amp) - np.einsum('ij,jk,ik->i', Kx, Kinv, Kx), 1e-9)
        mu = self.y_offset + self.y_scale * mu_s
        std = abs(self.y_scale) * np.sqrt(v_s)
        return np.asarray(mu, dtype=float), np.maximum(np.asarray(std, dtype=float), 1e-9)

    def prob_positive(self, theta: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        from math import erf, sqrt
        mu, std = self.predict(theta)
        z = (mu - threshold) / np.maximum(std, 1e-9)
        return 0.5 * (1.0 + np.vectorize(erf)(z / sqrt(2.0)))


def _theta_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[[f'theta_{j+1}' for j in range(6)]].to_numpy(dtype=float)


def _fit_one(g: pd.DataFrame, cfg: FullConfig, metric: str) -> GroupSurrogate:
    rows = []
    for theta_id, gg in g.groupby('theta_id'):
        vals = gg[metric].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        row = gg.iloc[0].copy()
        row['y_mean'] = float(np.mean(vals))
        row['n_rep'] = int(len(vals))
        row['noise_var'] = float(np.var(vals, ddof=1)) if len(vals) > 1 else np.nan
        rows.append(row)
    sdf = pd.DataFrame(rows)
    if sdf.empty:
        raise ValueError('No observations for surrogate group.')

    Xtheta = _theta_matrix(sdf)
    X = normalize_theta(Xtheta, cfg.design.theta_min, cfg.design.theta_max)
    y = sdf['y_mean'].to_numpy(dtype=float)
    nrep = sdf['n_rep'].to_numpy(dtype=float)
    noise = sdf['noise_var'].to_numpy(dtype=float)

    finite_noise = noise[np.isfinite(noise) & (noise > 0)]
    default_noise = float(np.median(finite_noise)) if len(finite_noise) else float(cfg.surrogate.default_noise)
    noise = np.where(np.isfinite(noise) & (noise > 0), noise, default_noise)
    noise = np.maximum(noise, float(cfg.surrogate.min_noise))

    # Observation variance of the replicate mean in original metric units.
    alpha_orig = noise / np.maximum(nrep, 1.0) + float(cfg.surrogate.min_noise)

    # Manual standardization keeps y and alpha in the same units. sklearn's
    # normalize_y=True would rescale y internally but leave alpha untouched.
    y_offset = float(np.mean(y))
    y_scale = float(np.std(y, ddof=0))
    if not np.isfinite(y_scale) or y_scale < 1e-8:
        y_scale = 1.0
    y_s = (y - y_offset) / y_scale
    alpha_s = alpha_orig / (y_scale ** 2)
    alpha_s = np.maximum(alpha_s, float(cfg.surrogate.min_noise) / (y_scale ** 2))

    model = None
    fallback = None
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF
        if cfg.surrogate.kernel.lower() == 'rbf':
            base = RBF(length_scale=np.ones(X.shape[1]) * cfg.surrogate.lengthscale,
                       length_scale_bounds=(0.05, 2.0))
        else:
            base = Matern(length_scale=np.ones(X.shape[1]) * cfg.surrogate.lengthscale,
                          length_scale_bounds=(0.05, 2.0), nu=2.5)
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * base
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha_s,
            normalize_y=False,
            n_restarts_optimizer=2,
            random_state=cfg.design.seed,
        )
        model.fit(X, y_s)
    except Exception:
        amp = float(np.var(y_s) + np.mean(alpha_s) + 1e-6)
        ell = float(cfg.surrogate.lengthscale)
        K = amp * np.exp(-0.5 * ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2) / max(ell ** 2, 1e-12))
        K[np.diag_indices_from(K)] += alpha_s
        Kinv = np.linalg.pinv(K)
        fallback = {
            'X': X,
            'alpha_s': alpha_s,
            'Kinv': Kinv,
            'y_s': y_s,
            'y0_s': np.array(float(np.mean(y_s))),
            'amp': np.array(amp),
            'ell': np.array(ell),
        }

    return GroupSurrogate(
        str(g['market'].iloc[0]),
        int(g['b'].iloc[0]),
        metric,
        Xtheta,
        y,
        nrep,
        noise,
        model,
        fallback,
        y_offset=y_offset,
        y_scale=y_scale,
        lo=cfg.design.theta_min,
        hi=cfg.design.theta_max,
    )


def fit_surrogates(field_csv: str | Path, cfg: FullConfig, metric: str | None = None) -> Dict[tuple[str, int], GroupSurrogate]:
    metric = metric or cfg.surrogate.metric
    df = pd.read_csv(field_csv)
    if metric not in df.columns:
        raise ValueError(f"metric={metric!r} missing from {field_csv}")
    out: Dict[tuple[str, int], GroupSurrogate] = {}
    for (market, b), g in df.groupby(['market', 'b']):
        out[(str(market), int(b))] = _fit_one(g, cfg, metric)
    return out


def write_prediction_grid(field_csv: str | Path, cfg: FullConfig, out_dir: str | Path, metric: str | None = None) -> Path:
    out_dir = ensure_dir(out_dir)
    sur = fit_surrogates(field_csv, cfg, metric=metric)
    unit = sobol_unit(cfg.surrogate.prediction_pool_size, d=cfg.design.theta_dim, seed=cfg.design.seed + 808)
    theta = denormalize_theta(unit, cfg.design.theta_min, cfg.design.theta_max)
    rows = []
    for (market, b), model in sur.items():
        mu, sd = model.predict(theta)
        pp = model.prob_positive(theta)
        for i, th in enumerate(theta):
            rec = {
                'market': market,
                'b': b,
                'pred_mean': float(mu[i]),
                'pred_std': float(sd[i]),
                'prob_positive': float(pp[i]),
            }
            for j, v in enumerate(th):
                rec[f'theta_{j+1}'] = float(v)
            rows.append(rec)
    df = pd.DataFrame(rows)
    path = out_dir / 'surrogate_predictions.csv'
    df.to_csv(path, index=False)
    return path
