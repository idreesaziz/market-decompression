from __future__ import annotations
from pathlib import Path
import pandas as pd
from .design_pool import read_theta_csv, dedup_thetas
from .io_utils import ensure_dir, write_json_atomic


def make_union(paths: list[str | Path], out_path: str | Path) -> Path:
    frames = [read_theta_csv(p) for p in paths if Path(p).exists()]
    if not frames:
        raise ValueError('No design CSVs found to union.')
    df = pd.concat(frames, ignore_index=True, sort=False)
    before = len(df)
    out = dedup_thetas(df).reset_index(drop=True)
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    out.to_csv(out_path, index=False)
    write_json_atomic(out_path.with_suffix('.report.json'), {'before_dedup': before, 'after_dedup': int(len(out)), 'sources': [str(p) for p in paths]})
    return out_path
