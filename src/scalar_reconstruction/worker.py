from __future__ import annotations

import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict

from .config import load_config, market_by_name
from .data import load_target
from .eval_transfer import execute_transfer_job
from .io_utils import append_jsonl, ensure_dir, read_json, write_json_atomic
from .jobs import claim_next_job, mark_job_done, mark_job_failed, runtime_paths


def set_thread_env(threads: int | None) -> None:
    if threads is None:
        return
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[name] = str(int(threads))


def run_worker(config_path: str | Path, worker_id: str, threads: int | None = None, max_jobs: int | None = None) -> None:
    cfg = load_config(config_path); paths = runtime_paths(cfg); jobs_dir = paths['jobs_dir']; results_dir = paths['results_dir']; logs_dir = paths['logs_dir']
    set_thread_env(threads)
    heartbeat = paths['state_dir'] / f'worker_{worker_id}.heartbeat.json'
    worker_log = logs_dir / f'worker_{worker_id}.jsonl'
    # Lazy cache targets per market.
    target_cache: Dict[str, Any] = {}
    jobs_done = 0
    while True:
        write_json_atomic(heartbeat, {'worker_id': worker_id, 'time': time.time(), 'status': 'waiting', 'jobs_done': jobs_done})
        running_path = claim_next_job(jobs_dir, worker_id=worker_id, lock_timeout_s=cfg.runtime.lock_timeout_s)
        if running_path is None:
            append_jsonl(worker_log, {'time': time.time(), 'worker_id': worker_id, 'event': 'no_pending_jobs'})
            break
        job = read_json(running_path); job_id = str(job['job_id']); market = str(job['market'])
        write_json_atomic(heartbeat, {'worker_id': worker_id, 'time': time.time(), 'status': 'running', 'job_id': job_id, 'jobs_done': jobs_done})
        append_jsonl(worker_log, {'time': time.time(), 'worker_id': worker_id, 'event': 'start', 'job_id': job_id, 'job': job})
        try:
            if market not in target_cache:
                target_cache[market] = load_target(market_by_name(cfg, market), cfg.data)
            result = execute_transfer_job(job, cfg, target_cache[market])
            result_path = results_dir / f'{job_id}.json'
            write_json_atomic(result_path, result)
            append_jsonl(paths['run_dir'] / 'scalar_transfer_evals.jsonl', result)
            mark_job_done(jobs_dir, running_path, result_path)
            jobs_done += 1
            append_jsonl(worker_log, {'time': time.time(), 'worker_id': worker_id, 'event': 'done', 'job_id': job_id, 'result_path': str(result_path)})
        except Exception as e:
            tb = traceback.format_exc()
            mark_job_failed(jobs_dir, running_path, tb)
            append_jsonl(worker_log, {'time': time.time(), 'worker_id': worker_id, 'event': 'failed', 'job_id': job_id, 'error': repr(e), 'traceback': tb})
        if max_jobs is not None and jobs_done >= int(max_jobs):
            break
        time.sleep(float(cfg.runtime.worker_sleep_s))
    write_json_atomic(heartbeat, {'worker_id': worker_id, 'time': time.time(), 'status': 'stopped', 'jobs_done': jobs_done})
