from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import pandas as pd

from .config import FullConfig, load_config, config_to_dict
from .design_pool import read_theta_csv
from .io_utils import ensure_dir, file_lock, read_json, stable_id, write_json_atomic


def runtime_paths(cfg: FullConfig) -> Dict[str, Path]:
    run_dir = ensure_dir(cfg.runtime.run_dir)
    return {
        'run_dir': run_dir,
        'jobs_dir': ensure_dir(run_dir / cfg.runtime.jobs_dir),
        'results_dir': ensure_dir(run_dir / cfg.runtime.results_dir),
        'logs_dir': ensure_dir(run_dir / cfg.runtime.logs_dir),
        'state_dir': ensure_dir(run_dir / 'state'),
        'design_dir': ensure_dir(run_dir / 'design'),
        'tables_dir': ensure_dir(run_dir / 'tables'),
        'targets_dir': ensure_dir(run_dir / 'targets'),
    }


def ensure_job_dirs(jobs_dir: Path) -> None:
    for s in ['pending', 'running', 'done', 'failed']:
        ensure_dir(jobs_dir / s)


def _job_path(jobs_dir: Path, status: str, job_id: str) -> Path:
    return jobs_dir / status / f'{job_id}.json'


def write_job(jobs_dir: Path, payload: Mapping[str, Any]) -> str:
    ensure_job_dirs(jobs_dir)
    job_id = str(payload.get('job_id') or stable_id(payload))
    obj = dict(payload); obj['job_id'] = job_id
    # Do not duplicate completed jobs. Existing pending/running are left untouched.
    if any(_job_path(jobs_dir, status, job_id).exists() for status in ['pending', 'running', 'done']):
        return job_id
    write_json_atomic(_job_path(jobs_dir, 'pending', job_id), obj)
    return job_id


def claim_next_job(jobs_dir: Path, worker_id: str, lock_timeout_s: float = 120.0) -> Path | None:
    ensure_job_dirs(jobs_dir)
    with file_lock(jobs_dir / '.claim.lock', timeout_s=lock_timeout_s):
        pending = sorted((jobs_dir / 'pending').glob('*.json'))
        if not pending:
            return None
        src = pending[0]
        job = read_json(src)
        job['claimed_by'] = worker_id
        job['claimed_at'] = time.time()
        dst = _job_path(jobs_dir, 'running', str(job['job_id']))
        write_json_atomic(dst, job)
        src.unlink(missing_ok=True)
        return dst


def mark_job_done(jobs_dir: Path, running_path: Path, result_path: Path) -> None:
    # Search running/ first, fall back to pending/ in case requeue-orphaned moved it.
    job_id = running_path.stem
    job = None
    for src_path in [running_path, _job_path(jobs_dir, 'pending', job_id)]:
        if src_path.exists():
            try:
                job = read_json(src_path)
                src_path.unlink(missing_ok=True)
                break
            except FileNotFoundError:
                continue
    if job is None:
        return
    job['done_at'] = time.time(); job['result_path'] = str(result_path)
    dst = _job_path(jobs_dir, 'done', job_id)
    write_json_atomic(dst, job)


def mark_job_failed(jobs_dir: Path, running_path: Path, traceback_text: str) -> None:
    job_id = running_path.stem
    job = None
    for src_path in [running_path, _job_path(jobs_dir, 'pending', job_id)]:
        if src_path.exists():
            try:
                job = read_json(src_path)
                src_path.unlink(missing_ok=True)
                break
            except FileNotFoundError:
                continue
    if job is None:
        return
    job['failed_at'] = time.time(); job['traceback'] = traceback_text
    job['retries'] = int(job.get('retries', 0)) + 1
    dst = _job_path(jobs_dir, 'failed', str(job['job_id']))
    write_json_atomic(dst, job)


def requeue_failed(config_path: str | Path, max_retries: int | None = None) -> int:
    cfg = load_config(config_path); paths = runtime_paths(cfg); jobs_dir = paths['jobs_dir']
    max_retries = int(max_retries if max_retries is not None else cfg.runtime.max_job_retries)
    n = 0
    for p in sorted((jobs_dir / 'failed').glob('*.json')):
        job = read_json(p)
        if int(job.get('retries', 0)) <= max_retries:
            dst = _job_path(jobs_dir, 'pending', str(job['job_id']))
            write_json_atomic(dst, job)
            p.unlink(missing_ok=True)
            n += 1
    return n


def make_jobs_from_design(
    config_path: str | Path,
    design_csv: str | Path,
    stage: str,
    markets: Sequence[str] | None = None,
    budgets: Sequence[int] | None = None,
    default_replicates: int | None = None,
    eval_split: str = 'val',
    finetune_lr_schedule: Mapping[str, float] | None = None,
) -> Path:
    cfg = load_config(config_path); paths = runtime_paths(cfg); jobs_dir = paths['jobs_dir']; ensure_job_dirs(jobs_dir)
    df = read_theta_csv(design_csv)
    markets = list(markets or [m.name for m in cfg.markets])
    budgets = [int(x) for x in (budgets or cfg.design.reconstruction_budgets)]
    base_seed = cfg.design.seed + 50_000 + (int(stable_id({'stage': stage}, 8), 16) % 10_000)
    lr_sched = {str(k): float(v) for k, v in finetune_lr_schedule.items()} if finetune_lr_schedule else {}
    job_ids = []
    for i, row in df.iterrows():
        theta = [float(row[f'theta_{j+1}']) for j in range(6)]
        reps = int(row.get('initial_replicates', default_replicates or cfg.design.reconstruction_reps))
        # Per-row LR schedule overrides the call-level one if present in CSV.
        row_sched_raw = row.get('finetune_lr_schedule')
        if row_sched_raw and str(row_sched_raw) not in ('', 'nan', 'None'):
            import json as _json
            try:
                row_sched = {str(k): float(v) for k, v in _json.loads(str(row_sched_raw)).items()}
            except Exception:
                row_sched = lr_sched
        else:
            row_sched = lr_sched
        for market in markets:
            for r in range(reps):
                payload = {
                    'stage': stage,
                    'market': market,
                    'family': cfg.train.family,
                    'theta_id': str(row['theta_id']),
                    'theta': theta,
                    'design_source': str(row.get('design_source', stage)),
                    'role': row.get('role') if str(row.get('role', '')) not in ('', 'nan', 'None') else None,
                    'control_type': row.get('control_type') if str(row.get('control_type', '')) not in ('', 'nan', 'None') else None,
                    'action_type': row.get('action_type') if str(row.get('action_type', '')) not in ('', 'nan', 'None') else None,
                    'acquisition': row.get('acquisition') if str(row.get('acquisition', '')) not in ('', 'nan', 'None') else None,
                    'replicate_index': int(r),
                    'seed': int(base_seed + i * 1009 + r * 37 + (int(stable_id({'market': market}, 8), 16) % 997)),
                    'label_budgets': budgets,
                    'eval_split': eval_split,
                    'finetune_lr_schedule': row_sched,
                    'created_at': time.time(),
                }
                payload['job_id'] = stable_id({'stage': stage, 'market': market, 'theta_id': payload['theta_id'], 'replicate_index': r, 'budgets': budgets, 'eval_split': eval_split, 'lr_sched': row_sched})
                job_ids.append(write_job(jobs_dir, payload))
    manifest = {'stage': stage, 'design_csv': str(design_csv), 'n_jobs': len(job_ids), 'job_ids': job_ids, 'markets': markets, 'budgets': budgets, 'config': config_to_dict(cfg)}
    out = paths['run_dir'] / f'{stage}_manifest.json'
    write_json_atomic(out, manifest)
    return out


def requeue_orphaned_running(config_path: str | Path, older_than_s: float = 600.0) -> int:
    cfg = load_config(config_path); paths = runtime_paths(cfg); jobs_dir = paths['jobs_dir']
    n = 0; now = time.time()
    for p in sorted((jobs_dir / 'running').glob('*.json')):
        try:
            age = now - p.stat().st_mtime
            if age >= older_than_s:
                job = read_json(p)
                write_json_atomic(_job_path(jobs_dir, 'pending', str(job['job_id'])), job)
                p.unlink(missing_ok=True); n += 1
        except FileNotFoundError:
            pass
    return n
