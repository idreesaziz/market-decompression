from __future__ import annotations

"""Optional R/hetGP bridge placeholder.

The core Python package does not require R.  This module writes clean CSV inputs and
contains a thin runner hook so the same field-observation schema can later be used
with CRAN `hetGP` and the noisy-contour acquisitions from that ecosystem.
"""
from pathlib import Path
import subprocess
from typing import Sequence


def write_hetgp_input(field_csv: str | Path, out_csv: str | Path, metric: str = 'log_skill') -> Path:
    import pandas as pd
    df = pd.read_csv(field_csv)
    cols = ['market', 'b', 'theta_id', *[f'theta_{j+1}' for j in range(6)], metric]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns for hetGP input: {missing}')
    out = Path(out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    df[cols].rename(columns={metric: 'y'}).to_csv(out, index=False)
    return out


def run_r_script(script_path: str | Path, args: Sequence[str]) -> int:
    cmd = ['Rscript', str(script_path), *map(str, args)]
    return subprocess.call(cmd)
