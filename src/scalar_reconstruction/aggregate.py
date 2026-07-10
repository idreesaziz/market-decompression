from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

from .config import load_config
from .io_utils import read_json
from .jobs import runtime_paths
from .skill_score import summarize_values


def _flatten_result(rec: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    theta = rec.get('theta', [])
    metrics = rec.get('metrics', {})
    for b_text, m in metrics.items():
        # Per-budget finetune_lr is stored in metrics[b]['finetune_lr'] when a schedule is used.
        # Fall back to job-level rec['finetune_lr'] for backward compatibility.
        b_lr = m.get('finetune_lr')
        finetune_lr = float(b_lr) if b_lr is not None else (float(rec['finetune_lr']) if rec.get('finetune_lr') is not None else np.nan)
        row = {
            'job_id': rec.get('job_id'), 'stage': rec.get('stage'), 'market': rec.get('market') or rec.get('target_name'),
            'family': rec.get('family'), 'theta_id': rec.get('theta_id'), 'theta': json.dumps(theta, separators=(',', ':')),
            'design_source': rec.get('design_source'), 'role': rec.get('role'), 'control_type': rec.get('control_type'),
            'action_type': rec.get('action_type'), 'acquisition': rec.get('acquisition'), 'replicate_index': rec.get('replicate_index'),
            'seed': rec.get('seed'), 'b': int(b_text),
            'finetune_lr': finetune_lr,
            'skill_T': float(m.get('skill_T', m.get('T', np.nan))), 'log_skill': float(m.get('log_skill', np.nan)), 'ratio': float(m.get('ratio', np.nan)), 'hit': bool(m.get('hit', False)),
            'mse_forecast_only': float(m.get('mse_forecast_only', np.nan)), 'mse_transfer': float(m.get('mse_transfer', m.get('mse_forecast_plus_latent', np.nan))),
            'elapsed_s': rec.get('elapsed_s'), 'eval_split': rec.get('eval_split'), 'eval_protocol': rec.get('eval_protocol'),
        }
        for j, v in enumerate(theta):
            row[f'theta_{j+1}'] = float(v)
        for i, v in enumerate(m.get('T_by_horizon', []) or []):
            row[f'skill_T_h{i}'] = float(v)
        for i, v in enumerate(m.get('log_skill_by_horizon', []) or []):
            row[f'log_skill_h{i}'] = float(v)
        rows.append(row)
    return rows


def collect_results(config_path: str | Path) -> pd.DataFrame:
    cfg = load_config(config_path); paths = runtime_paths(cfg)
    rows: List[Dict[str, Any]] = []
    for p in sorted(paths['results_dir'].glob('*.json')):
        try:
            rows.extend(_flatten_result(read_json(p)))
        except Exception as e:
            rows.append({'job_id': p.stem, 'error': repr(e)})
    df = pd.DataFrame(rows)
    out = paths['tables_dir'] / 'field_observations.csv'
    df.to_csv(out, index=False)
    # Keep a top-level copy for convenience.
    df.to_csv(paths['run_dir'] / 'scalar_transfer_evals_flat.csv', index=False)
    return df


def summarize_field(config_path: str | Path) -> Path:
    cfg = load_config(config_path); paths = runtime_paths(cfg)
    df = collect_results(config_path)
    rows = []
    if df.empty:
        out = paths['tables_dir'] / 'skill_score_summary.csv'; pd.DataFrame().to_csv(out, index=False); return out
    gcols = ['market', 'b', 'theta_id', 'theta', 'design_source', 'finetune_lr']
    for key, g in df.groupby(gcols, dropna=False):
        rec = dict(zip(gcols, key))
        for j in range(6):
            rec[f'theta_{j+1}'] = float(g[f'theta_{j+1}'].iloc[0])
        for metric in ['skill_T', 'log_skill']:
            s = summarize_values(g[metric].dropna().to_numpy(dtype=float))
            for k, v in s.items():
                rec[f'{metric}_{k}'] = v
        rec['hit_rate'] = float(np.mean(g['hit'].astype(bool))) if len(g) else np.nan
        rec['n_jobs'] = int(len(g))
        rows.append(rec)
    out_df = pd.DataFrame(rows)
    out = paths['tables_dir'] / 'skill_score_summary.csv'
    out_df.to_csv(out, index=False)
    return out


def _bootstrap_ci(vals: np.ndarray, n_boot: int = 2000, seed: int = 42, ci_levels: tuple = (0.90, 0.95)) -> Dict[str, float]:
    """Bootstrap over replicate-level observations. Returns CI bounds and prob_gt0."""
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {f'ci{int(ci*100)}_low': np.nan for ci in ci_levels} | {f'ci{int(ci*100)}_high': np.nan for ci in ci_levels} | {'bootstrap_prob_mean_gt0': np.nan}
    rng = np.random.default_rng(seed)
    means = np.array([np.mean(rng.choice(vals, size=len(vals), replace=True)) for _ in range(n_boot)])
    out: Dict[str, float] = {}
    for ci in ci_levels:
        alpha = 1.0 - ci
        lo = float(np.percentile(means, 100 * alpha / 2))
        hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
        out[f'ci{int(ci*100)}_low'] = lo
        out[f'ci{int(ci*100)}_high'] = hi
    out['bootstrap_prob_mean_gt0'] = float(np.mean(means > 0.0))
    return out


def bootstrap_summary(config_path: str | Path, n_boot: int = 2000) -> Path:
    """Compute bootstrap CIs per (market, b, theta_id). Paper-grade replicate-level inference."""
    cfg = load_config(config_path); paths = runtime_paths(cfg)
    df = collect_results(config_path)
    if df.empty:
        out = paths['tables_dir'] / 'bootstrap_summary.csv'
        pd.DataFrame().to_csv(out, index=False)
        return out
    rows = []
    gcols = ['market', 'b', 'theta_id', 'design_source', 'role']
    for key, g in df.groupby([c for c in gcols if c in df.columns], dropna=False):
        rec = dict(zip([c for c in gcols if c in df.columns], key if isinstance(key, tuple) else [key]))
        for j in range(6):
            if f'theta_{j+1}' in g.columns:
                rec[f'theta_{j+1}'] = float(g[f'theta_{j+1}'].iloc[0])
        rec['n_replicates'] = int(len(g))
        rec['finetune_lr'] = float(g['finetune_lr'].iloc[0]) if 'finetune_lr' in g.columns else np.nan
        for metric in ['log_skill', 'skill_T', 'mse_forecast_only', 'mse_transfer']:
            if metric not in g.columns:
                continue
            vals = g[metric].dropna().to_numpy(dtype=float)
            rec[f'{metric}_mean'] = float(np.mean(vals)) if len(vals) else np.nan
            rec[f'{metric}_median'] = float(np.median(vals)) if len(vals) else np.nan
            rec[f'{metric}_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            if metric == 'log_skill':
                ci_res = _bootstrap_ci(vals, n_boot=n_boot)
                rec.update({f'log_skill_{k}': v for k, v in ci_res.items()})
        if 'hit' in g.columns:
            rec['hit_rate'] = float(np.mean(g['hit'].astype(bool)))
        if 'mse_forecast_only' in g.columns and 'mse_transfer' in g.columns:
            fo = g['mse_forecast_only'].dropna().to_numpy(dtype=float)
            tr = g['mse_transfer'].dropna().to_numpy(dtype=float)
            if len(fo) > 0 and len(tr) > 0:
                rec['ratio_mse_transfer_to_forecast_only'] = float(np.mean(tr) / max(np.mean(fo), 1e-12))
        rows.append(rec)
    out_df = pd.DataFrame(rows)
    out = paths['tables_dir'] / 'bootstrap_summary.csv'
    out_df.to_csv(out, index=False)
    return out


def noise_summary(config_path: str | Path) -> Path:
    cfg = load_config(config_path); paths = runtime_paths(cfg)
    df = collect_results(config_path)
    rows = []
    if not df.empty:
        for (market, b, theta_id, theta), g in df.groupby(['market', 'b', 'theta_id', 'theta'], dropna=False):
            rec = {'market': market, 'b': int(b), 'theta_id': theta_id, 'theta': theta, 'n': int(len(g)), 'hit_rate': float(np.mean(g['hit'].astype(bool)))}
            for j in range(6): rec[f'theta_{j+1}'] = float(g[f'theta_{j+1}'].iloc[0])
            for metric in ['skill_T', 'log_skill']:
                vals = g[metric].dropna().to_numpy(dtype=float)
                rec[f'{metric}_mean'] = float(np.mean(vals)) if len(vals) else np.nan
                rec[f'{metric}_var'] = float(np.var(vals, ddof=1)) if len(vals) > 1 else np.nan
                rec[f'{metric}_std'] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            rows.append(rec)
    out = paths['tables_dir'] / 'noise_summary.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    return out
