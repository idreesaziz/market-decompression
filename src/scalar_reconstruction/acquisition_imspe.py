from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from .config import FullConfig
from .design_pool import sobol_unit, denormalize_theta, theta_dataframe, normalize_theta
from .surrogate import fit_surrogates
from .io_utils import ensure_dir, stable_id, theta_to_text


def make_imspe_actions(field_csv: str | Path, cfg: FullConfig, out_dir: str | Path, n_actions: int = 24, metric: str | None = None, explore_frac: float = 0.6) -> Path:
    """Approximate batched replicate-vs-explore actions.

    This is not a full hetGP IMSPE implementation.  It is an engineering-compatible
    approximation that exposes the same action space: either explore a new candidate
    from a Sobol pool or replicate an existing theta with high posterior/noise value.
    The R/hetGP bridge can replace this selector later without changing job schema.
    """
    out_dir = ensure_dir(out_dir)
    metric = metric or cfg.surrogate.metric
    df = pd.read_csv(field_csv)
    sur = fit_surrogates(field_csv, cfg, metric=metric)
    pool_unit = sobol_unit(cfg.design.pool_size, d=cfg.design.theta_dim, seed=cfg.design.seed + 909)
    pool_theta = denormalize_theta(pool_unit, cfg.design.theta_min, cfg.design.theta_max)
    # Exploration score: average posterior variance across all market/budget surrogates.
    score = np.zeros(len(pool_theta))
    for m in sur.values():
        _, sd = m.predict(pool_theta)
        score += sd ** 2
    score /= max(1, len(sur))
    explore_n = int(round(n_actions * explore_frac))
    repl_n = int(n_actions) - explore_n
    chosen = []
    order = np.argsort(-score)
    selected_unit = []
    for idx in order:
        x = pool_unit[idx]
        if all(np.linalg.norm(x - y) >= cfg.design.diverse_min_distance for y in selected_unit):
            selected_unit.append(x)
            chosen.append(idx)
        if len(chosen) >= explore_n:
            break
    rows = []
    for rank, idx in enumerate(chosen):
        th = pool_theta[idx]
        rec = {'action_type': 'explore', 'acquisition': 'approx_imspe_explore', 'acquisition_score': float(score[idx]), 'action_rank': rank, 'theta': theta_to_text(th.tolist()), 'theta_id': stable_id({'theta': [round(float(x), 8) for x in th.tolist()]}), 'design_source': 'imspe_explore', 'initial_replicates': int(cfg.design.reconstruction_reps)}
        for j, v in enumerate(th): rec[f'theta_{j+1}'] = float(v)
        rows.append(rec)
    # Replication score: replicate sites with high estimated noise and/or high posterior uncertainty.
    if repl_n > 0 and not df.empty:
        gcols = ['theta_id', 'theta', *[f'theta_{j+1}' for j in range(6)]]
        reps = []
        for theta_id, gg in df.groupby('theta_id'):
            th = gg[[f'theta_{j+1}' for j in range(6)]].iloc[0].to_numpy(dtype=float)
            score_rep = 0.0
            for (market, b), m in sur.items():
                sub = gg[(gg['market'] == market) & (gg['b'] == b)]
                a = max(1, len(sub))
                vals = sub[metric].dropna().to_numpy(dtype=float)
                nv = float(np.var(vals, ddof=1)) if len(vals) > 1 else float(cfg.surrogate.default_noise)
                _, sd = m.predict(th)
                score_rep += float(sd[0] ** 2) * nv / (a + 1.0)
            reps.append((score_rep, theta_id, th))
        reps.sort(reverse=True, key=lambda x: x[0])
        for k, (sr, tid, th) in enumerate(reps[:repl_n]):
            rec = {'action_type': 'replicate', 'acquisition': 'approx_imspe_replicate', 'acquisition_score': float(sr), 'action_rank': explore_n + k, 'theta': theta_to_text(th.tolist()), 'theta_id': str(tid), 'design_source': 'imspe_replicate', 'initial_replicates': 1}
            for j, v in enumerate(th): rec[f'theta_{j+1}'] = float(v)
            rows.append(rec)
    path = out_dir / 'adaptive_imspe_actions.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
