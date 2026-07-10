from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, List, Dict


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json_default(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
    except Exception:
        pass
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def read_json(path: str | Path) -> Any:
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)


def write_json_atomic(path: str | Path, obj: Any, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=indent, sort_keys=True, default=_json_default)
        f.write('\n')
    os.replace(tmp, path)


def append_jsonl(path: str | Path, obj: Mapping[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    line = json.dumps(obj, sort_keys=True, default=_json_default)
    with path.open('a', encoding='utf-8') as f:
        f.write(line + '\n')


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@contextlib.contextmanager
def file_lock(lock_path: str | Path, timeout_s: float = 120.0, poll_s: float = 0.25) -> Iterator[Path]:
    lock_path = Path(lock_path)
    ensure_dir(lock_path.parent)
    start = time.time()
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"pid={os.getpid()} time={time.time()}\n".encode('utf-8'))
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
                if age > timeout_s:
                    lock_path.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.time() - start > timeout_s:
                raise TimeoutError(f"Timed out waiting for lock {lock_path}")
            time.sleep(poll_s)
    try:
        yield lock_path
    finally:
        if fd is not None:
            os.close(fd)
        lock_path.unlink(missing_ok=True)


def stable_id(payload: Mapping[str, Any], n: int = 16) -> str:
    import hashlib
    canonical = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=_json_default)
    return hashlib.sha1(canonical.encode('utf-8')).hexdigest()[:n]


def parse_theta_text(theta: str | list[float] | tuple[float, ...]) -> list[float]:
    if isinstance(theta, str):
        xs = json.loads(theta)
    else:
        xs = list(theta)
    if len(xs) != 6:
        raise ValueError(f"theta must have length 6, got {len(xs)}")
    return [float(x) for x in xs]


def theta_key(theta: list[float] | tuple[float, ...], decimals: int = 6) -> tuple[float, ...]:
    return tuple(round(float(x), decimals) for x in theta)


def theta_to_text(theta: list[float] | tuple[float, ...]) -> str:
    return json.dumps([float(x) for x in theta], separators=(',', ':'))
