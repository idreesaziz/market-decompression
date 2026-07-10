from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .config import FullConfig
from .design_pool import sobol_unit, denormalize_theta, maximin_select, theta_dataframe, dedup_thetas
from .io_utils import ensure_dir, stable_id, theta_to_text, write_json_atomic


def make_seed_design(cfg: FullConfig, out_dir: str | Path) -> Path:
    out_dir = ensure_dir(out_dir)
    d = cfg.design.theta_dim
    lo, hi = cfg.design.theta_min, cfg.design.theta_max
    pool = sobol_unit(max(cfg.design.pool_size, cfg.design.seed_spacefill_n * 8), d=d, seed=cfg.design.seed + 101)
    selected_unit = maximin_select(pool, cfg.design.seed_spacefill_n, seed=cfg.design.seed + 202)
    selected_theta = denormalize_theta(selected_unit, lo=lo, hi=hi)
    sf = theta_dataframe(selected_theta, design_source='seed_spacefill', extra={'role': 'spacefill', 'initial_replicates': int(cfg.design.seed_spacefill_reps)})
    anchors = []
    for i, a in enumerate(cfg.design.anchors):
        th = [float(x) for x in a['theta']]
        rec = {'theta': theta_to_text(th), 'design_source': 'seed_anchor', 'design_index': i, 'role': a.get('role', 'anchor'), 'anchor_name': a.get('name', f'anchor_{i}'), 'initial_replicates': int(cfg.design.anchor_reps)}
        for j, v in enumerate(th):
            rec[f'theta_{j+1}'] = float(v)
        rec['theta_id'] = stable_id({'theta': [round(float(x), 8) for x in th]})
        anchors.append(rec)
    df = pd.concat([pd.DataFrame(anchors), sf], ignore_index=True)
    before = len(df)
    df = dedup_thetas(df)
    path = out_dir / 'seed_design.csv'
    df.to_csv(path, index=False)
    write_json_atomic(out_dir / 'seed_design_report.json', {'before_dedup': before, 'after_dedup': int(len(df)), 'spacefill_n': int(cfg.design.seed_spacefill_n), 'anchor_n': len(cfg.design.anchors)})
    return path
