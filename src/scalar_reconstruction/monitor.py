from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd

from .aggregate import collect_results, summarize_field
from .config import load_config
from .io_utils import read_json
from .jobs import runtime_paths


def _count(p: Path) -> int:
    return len(list(p.glob('*.json')))


def _fmt_age(ts) -> str:
    if not ts: return 'n/a'
    s = max(0.0, time.time() - float(ts))
    if s < 60: return f'{s:.0f}s'
    if s < 3600: return f'{s/60:.1f}m'
    return f'{s/3600:.1f}h'


def print_status(config_path: str | Path, top_k: int = 8, summarize: bool = True) -> None:
    cfg = load_config(config_path); paths = runtime_paths(cfg); jobs = paths['jobs_dir']
    pending = _count(jobs/'pending'); running = _count(jobs/'running'); done = _count(jobs/'done'); failed = _count(jobs/'failed')
    total = pending + running + done + failed
    progress = 100.0 * done / total if total else 0.0
    print('=' * 96)
    print(f"Scalar reconstruction: {paths['run_dir']}")
    print(f"Jobs: total={total} done={done} running={running} pending={pending} failed={failed} progress={progress:.1f}%")
    hbs = sorted(paths['state_dir'].glob('worker_*.heartbeat.json'))
    if hbs:
        print('\nWorkers:')
        for hb in hbs:
            try:
                obj = read_json(hb)
                print(f"  {obj.get('worker_id')}: status={obj.get('status')} job={obj.get('job_id','-')} jobs_done={obj.get('jobs_done',0)} heartbeat_age={_fmt_age(obj.get('time'))}")
            except Exception as e:
                print(f'  {hb.name}: unreadable ({e})')
    if summarize:
        try:
            summary_path = summarize_field(config_path)
            df = pd.read_csv(summary_path)
            if not df.empty:
                print('\nTop mean log-skill by market/budget/theta:')
                cols = ['market','b','design_source','log_skill_mean','log_skill_std','log_skill_n','hit_rate','theta']
                cols = [c for c in cols if c in df.columns]
                print(df.sort_values('log_skill_mean', ascending=False)[cols].head(top_k).to_string(index=False, max_colwidth=42))
                obs = collect_results(config_path)
                if not obs.empty:
                    by = obs.groupby(['market','b'])['log_skill'].agg(['count','mean','std']).reset_index()
                    print('\nBy market/budget log-skill:')
                    print(by.to_string(index=False))
            else:
                print('\nNo result summaries yet.')
        except Exception as e:
            print(f'\nSummary unavailable yet: {e}')
    print('=' * 96)
