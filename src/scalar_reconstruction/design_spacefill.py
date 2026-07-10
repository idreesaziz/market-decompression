from __future__ import annotations

from pathlib import Path
import numpy as np

from .config import FullConfig
from .design_pool import sobol_unit, maximin_select, denormalize_theta, theta_dataframe
from .io_utils import ensure_dir, write_json_atomic


def make_spacefill_design(cfg: FullConfig, out_dir: str | Path, n: int | None = None) -> Path:
    out_dir = ensure_dir(out_dir)
    n = int(n or cfg.design.spacefill_n)
    pool = sobol_unit(max(cfg.design.pool_size, n * 32), d=cfg.design.theta_dim, seed=cfg.design.seed + 404)
    selected_unit = maximin_select(pool, n=n, seed=cfg.design.seed + 405)
    theta = denormalize_theta(selected_unit, lo=cfg.design.theta_min, hi=cfg.design.theta_max)
    df = theta_dataframe(theta, 'sobol_maximin', extra={'role': 'spacefill', 'initial_replicates': int(cfg.design.reconstruction_reps)})
    path = out_dir / 'theta_spacefill.csv'
    df.to_csv(path, index=False)
    write_json_atomic(out_dir / 'theta_spacefill_report.json', {'n': n, 'pool_size': int(len(pool))})
    return path
