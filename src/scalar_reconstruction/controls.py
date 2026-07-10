from __future__ import annotations

from pathlib import Path
import pandas as pd

from .config import FullConfig
from .io_utils import ensure_dir, stable_id, theta_to_text


def make_null_control_design(cfg: FullConfig, out_dir: str | Path) -> Path:
    """Write a small set of explicit controls.

    These are encoded as theta controls first.  If the evaluation pipeline later adds
    a special null/scrambled source generator, the `control_type` column is already
    available in the schema.
    """
    out_dir = ensure_dir(out_dir)
    controls = [
        ('center_control', [0, 0, 0, 0, 0, 0], 'center'),
        ('negative_extreme', [3, 3, -3, 3, 3, -3], 'mismatched_extreme'),
        ('all_low', [-3, -3, -3, -3, -3, -3], 'mismatched_extreme'),
        ('all_high', [3, 3, 3, 3, 3, 3], 'mismatched_extreme'),
    ]
    rows = []
    for i, (name, th, ctype) in enumerate(controls):
        rec = {'theta': theta_to_text(th), 'theta_id': stable_id({'theta': [round(float(x),8) for x in th]}), 'design_source': 'control', 'design_index': i, 'role': ctype, 'control_name': name, 'control_type': ctype, 'initial_replicates': int(cfg.design.anchor_reps)}
        for j, v in enumerate(th): rec[f'theta_{j+1}'] = float(v)
        rows.append(rec)
    path = out_dir / 'theta_controls.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
