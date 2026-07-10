from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from .config import load_config
from .jobs import runtime_paths


def _theta_key_df(df: pd.DataFrame, decimals: int = 8) -> pd.Series:
    cols = [f'theta_{j+1}' for j in range(6)]
    rounded = df[cols].astype(float).round(decimals).astype(str)
    return rounded.agg('|'.join, axis=1)


def posterior_volume_ci(
    config_path: str | Path,
    field_csv: str | Path,
    n_samples: int = 500,
    threshold: float = 0.0,
    ci_levels: tuple = (0.90, 0.95),
    seed: int = 42,
) -> Path:
    """Estimate positive-transfer envelope volume with posterior uncertainty.

    Samples posterior fields F_b(theta;R) ~ GP(mu, Sigma) for each (market, budget)
    and computes envelope volume (fraction of theta-grid with F > threshold) for each
    sample. Reports mean volume + CI across samples.
    """
    from .config import load_config
    from .surrogate import fit_surrogates
    from .design_pool import sobol_unit, denormalize_theta
    cfg = load_config(config_path)
    paths = runtime_paths(cfg)
    sur = fit_surrogates(field_csv, cfg)
    rng = np.random.default_rng(seed)
    unit = sobol_unit(cfg.surrogate.prediction_pool_size, d=cfg.design.theta_dim, seed=cfg.design.seed + 808)
    theta = denormalize_theta(unit, cfg.design.theta_min, cfg.design.theta_max)
    # Classify theta by region
    core_mask = np.all(np.abs(theta) <= 3.0, axis=1)
    rows = []
    for (market, b), model in sur.items():
        mu, sd = model.predict(theta)
        # Sample n_samples posterior realizations: f_i ~ N(mu_j, sd_j^2) for each grid point j
        z = rng.standard_normal((n_samples, len(theta)))
        samples = mu[None, :] + sd[None, :] * z  # (n_samples, n_grid)
        # Envelope volume per sample
        vol_samples = np.mean(samples > threshold, axis=1)  # (n_samples,)
        vol_core_samples = np.mean(samples[:, core_mask] > threshold, axis=1) if core_mask.any() else np.zeros(n_samples)
        vol_shell_samples = np.mean(samples[:, ~core_mask] > threshold, axis=1) if (~core_mask).any() else np.zeros(n_samples)
        rec: Dict = {'market': market, 'b': int(b), 'threshold': threshold, 'n_grid': len(theta), 'n_samples': n_samples}
        rec['vol_mean'] = float(np.mean(vol_samples))
        rec['vol_std'] = float(np.std(vol_samples, ddof=1))
        rec['vol_core_mean'] = float(np.mean(vol_core_samples))
        rec['vol_shell_mean'] = float(np.mean(vol_shell_samples))
        for ci in ci_levels:
            alpha = 1.0 - ci
            rec[f'vol_ci{int(ci*100)}_low'] = float(np.percentile(vol_samples, 100 * alpha / 2))
            rec[f'vol_ci{int(ci*100)}_high'] = float(np.percentile(vol_samples, 100 * (1 - alpha / 2)))
        # Also report plug-in estimate from posterior mean
        rec['vol_plugin'] = float(np.mean(mu > threshold))
        rec['vol_prob_gt0'] = float(np.mean(0.5 * (1 + np.array([model.prob_positive(theta[j:j+1])[0] for j in range(0, min(100, len(theta)))]))))
        rows.append(rec)
    out = paths['tables_dir'] / 'posterior_volume_ci.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def envelope_tables(config_path: str | Path, prediction_csv: str | Path, threshold: float = 0.0, gamma: float = 0.5) -> tuple[Path, Path]:
    cfg = load_config(config_path)
    paths = runtime_paths(cfg)
    df = pd.read_csv(prediction_csv).copy()
    df['theta_key'] = _theta_key_df(df)

    rows = []
    for (market, b), g in df.groupby(['market', 'b']):
        hard = g['pred_mean'].to_numpy(dtype=float) > threshold
        pp = g['prob_positive'].to_numpy(dtype=float)
        rows.append({
            'market': market,
            'b': int(b),
            'threshold': float(threshold),
            'hard_volume': float(np.mean(hard)) if len(hard) else np.nan,
            'prob_volume': float(np.mean(pp)) if len(pp) else np.nan,
            'confident_volume': float(np.mean(pp > gamma)) if len(pp) else np.nan,
            'n_grid': int(len(g)),
        })
    vol_path = paths['tables_dir'] / 'envelope_volume.csv'
    pd.DataFrame(rows).to_csv(vol_path, index=False)

    # Pairwise Jaccard overlaps within each budget. Align by theta_key rather
    # than row order so adaptive/incomplete grids cannot silently miscompare.
    rows = []
    for b, gb in df.groupby('b'):
        frames = {}
        for market, g in gb.groupby('market'):
            tmp = g[['theta_key', 'pred_mean']].copy()
            tmp = tmp.drop_duplicates('theta_key')
            tmp[f'mask_{market}'] = tmp['pred_mean'].astype(float) > threshold
            frames[market] = tmp[['theta_key', f'mask_{market}']]
        markets = sorted(frames)
        for i in range(len(markets)):
            for j in range(i + 1, len(markets)):
                ma, mb = markets[i], markets[j]
                merged = frames[ma].merge(frames[mb], on='theta_key', how='inner')
                if merged.empty:
                    rows.append({
                        'b': int(b), 'market_a': ma, 'market_b': mb,
                        'jaccard': np.nan, 'intersection': 0, 'union': 0,
                        'n_common': 0,
                    })
                    continue
                a = merged[f'mask_{ma}'].to_numpy(dtype=bool)
                c = merged[f'mask_{mb}'].to_numpy(dtype=bool)
                union = int(np.logical_or(a, c).sum())
                inter = int(np.logical_and(a, c).sum())
                rows.append({
                    'b': int(b),
                    'market_a': ma,
                    'market_b': mb,
                    'jaccard': float(inter / union) if union else np.nan,
                    'intersection': inter,
                    'union': union,
                    'n_common': int(len(merged)),
                })
    jac_path = paths['tables_dir'] / 'envelope_overlap_jaccard.csv'
    pd.DataFrame(rows).to_csv(jac_path, index=False)
    return vol_path, jac_path
