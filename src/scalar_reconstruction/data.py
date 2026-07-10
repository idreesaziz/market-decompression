from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd

from .config import DataBuildConfig, MarketConfig


def _infer_time_col(df: pd.DataFrame) -> str | None:
    candidates = [c for c in df.columns if str(c).lower() in {"time", "timestamp", "date", "datetime", "open_time", "close_time"}]
    return candidates[0] if candidates else None


def read_market_frame(market: MarketConfig) -> pd.DataFrame:
    path = Path(market.csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Market CSV not found for {market.name}: {path}")
    df = pd.read_csv(path)
    tcol = market.time_col or _infer_time_col(df)
    if tcol and tcol in df.columns:
        s = df[tcol]
        # Binance sometimes uses millisecond timestamps.
        if pd.api.types.is_numeric_dtype(s):
            vals = pd.to_numeric(s, errors='coerce')
            unit = 'ms' if vals.dropna().median() > 10_000_000_000 else 's'
            df[tcol] = pd.to_datetime(vals, errors='coerce', utc=True, unit=unit)
        else:
            df[tcol] = pd.to_datetime(s, errors='coerce', utc=True)
        df = df.sort_values(tcol).drop_duplicates(tcol).reset_index(drop=True)
    if market.max_rows is not None and len(df) > int(market.max_rows):
        df = df.iloc[-int(market.max_rows):].copy().reset_index(drop=True)
    return df


def extract_returns_and_times(df: pd.DataFrame, market: MarketConfig, build: DataBuildConfig) -> Tuple[np.ndarray, np.ndarray | None]:
    tcol = market.time_col or _infer_time_col(df)
    times = None
    if market.return_col and market.return_col in df.columns:
        r = pd.to_numeric(df[market.return_col], errors='coerce').to_numpy(dtype=np.float64)
        if tcol and tcol in df.columns:
            times = pd.to_datetime(df[tcol], utc=True, errors='coerce').to_numpy()
    else:
        if market.price_col not in df.columns:
            raise ValueError(f"{market.name} CSV must contain price_col={market.price_col!r} or return_col={market.return_col!r}. Columns={list(df.columns)}")
        px = pd.to_numeric(df[market.price_col], errors='coerce').to_numpy(dtype=np.float64)
        good = np.isfinite(px) & (px > 0)
        if tcol and tcol in df.columns:
            all_times = pd.to_datetime(df[tcol], utc=True, errors='coerce').to_numpy()
            px = px[good]
            all_times = all_times[good]
            r = np.diff(np.log(px))
            times = all_times[1:]
        else:
            px = px[good]
            r = np.diff(np.log(px))
    finite = np.isfinite(r)
    r = r[finite]
    if times is not None:
        times = times[finite]
    qlo, qhi = build.winsorize_quantiles
    lo, hi = np.nanquantile(r, [float(qlo), float(qhi)])
    r = np.clip(r, lo, hi).astype(np.float64)
    return r, times


def _to_epoch_seconds(arr) -> np.ndarray:
    """Convert any datetime-like array to integer seconds since epoch.

    Handles both pandas datetime64[ns] (pandas < 3) and datetime64[us] (pandas >= 3)
    by detecting the scale from the magnitude of the int64 representation.
    Dates in 2000-2100: ns epoch ~1.6e18, us epoch ~1.6e15; threshold 1e16 splits cleanly.
    """
    int64 = pd.to_datetime(arr, utc=True).astype('int64').to_numpy()
    scale = 1_000_000_000 if (len(int64) > 0 and int64[0] > 10 ** 16) else 1_000_000
    return int64 // scale


def _contiguous_mask(times: np.ndarray | None, start: int, length: int, expected_seconds: int) -> bool:
    if times is None:
        return True
    window = times[start:start + length]
    if len(window) != length:
        return False
    if np.any(pd.isna(window)):
        return False
    ts = _to_epoch_seconds(window)
    deltas = np.diff(ts)
    return bool(np.all(deltas == int(expected_seconds)))


def build_target_from_returns(returns: np.ndarray, times: np.ndarray | None, market: MarketConfig, build: DataBuildConfig) -> Dict[str, np.ndarray]:
    r = np.asarray(returns, dtype=np.float64)
    L = int(build.seq_len)
    horizons = tuple(int(h) for h in build.horizons)
    max_h = max(horizons)
    n_possible = r.size - L - max_h + 1
    if n_possible <= 0:
        raise ValueError(f"Too few returns for {market.name}: got {r.size}, need > {L + max_h}")
    X_rows = []
    Y_rows = []
    starts = []
    for i in range(n_possible):
        if not _contiguous_mask(times, i, L + max_h, market.expected_interval_seconds):
            continue
        tape_r = r[i:i + L]
        rv_proxy = np.zeros(L, dtype=np.float64)
        for t in range(L):
            lo = max(0, t - 9)
            rv_proxy[t] = np.sqrt(np.mean(tape_r[lo:t + 1] ** 2) + 1e-12)
        X = np.zeros((L, 3), dtype=np.float32)
        X[:, 0] = tape_r.astype(np.float32)
        X[:, 1] = np.abs(tape_r).astype(np.float32)
        X[:, 2] = rv_proxy.astype(np.float32)
        Y = np.zeros((len(horizons),), dtype=np.float32)
        future_start = i + L
        for j, h in enumerate(horizons):
            future = r[future_start:future_start + h]
            Y[j] = float(np.sum(future ** 2))
        X_rows.append(X)
        Y_rows.append(Y)
        starts.append(i)
    if len(X_rows) < int(build.min_rows):
        raise ValueError(f"Only {len(X_rows)} contiguous windows built for {market.name}; min_rows={build.min_rows}")
    out: Dict[str, np.ndarray] = {
        'X': np.stack(X_rows).astype(np.float32),
        'Y': np.stack(Y_rows).astype(np.float32),
        'returns': r.astype(np.float32),
        'horizons': np.asarray(horizons, dtype=np.int32),
        'valid_window_starts': np.asarray(starts, dtype=np.int64),
    }
    if times is not None:
        out['return_times_ns'] = pd.to_datetime(times, utc=True).astype('int64').to_numpy().astype(np.int64)
    return out


def load_target(market: MarketConfig, build: DataBuildConfig) -> Dict[str, np.ndarray]:
    df = read_market_frame(market)
    returns, times = extract_returns_and_times(df, market, build)
    target = build_target_from_returns(returns, times, market, build)
    target['target_name'] = np.asarray([market.name])
    return target


def chronological_split_indices(n: int, train_frac: float, val_frac: float, gap: int = 0) -> Dict[str, np.ndarray]:
    if not (0 < train_frac < 1 and 0 <= val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError('invalid split fractions')
    n = int(n); gap = max(0, int(gap))
    train_end = int(np.floor(n * train_frac))
    val_end = int(np.floor(n * (train_frac + val_frac)))
    train = np.arange(0, max(0, train_end), dtype=np.int64)
    val = np.arange(min(n, train_end + gap), max(train_end + gap, val_end), dtype=np.int64)
    test = np.arange(min(n, val_end + gap), n, dtype=np.int64)
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError(f"Empty split: train={len(train)}, val={len(val)}, test={len(test)}, n={n}, gap={gap}")
    return {'train': train, 'val': val, 'test': test}


def subset_target(raw: Mapping[str, np.ndarray], idx: np.ndarray) -> Dict[str, np.ndarray]:
    idx = np.asarray(idx, dtype=np.int64)
    out: Dict[str, np.ndarray] = {}
    for k, v in raw.items():
        if isinstance(v, np.ndarray) and len(v.shape) > 0 and v.shape[0] == raw['X'].shape[0]:
            out[k] = v[idx]
        else:
            out[k] = v
    return out


def audit_market(market: MarketConfig, build: DataBuildConfig) -> Dict[str, object]:
    df = read_market_frame(market)
    returns, times = extract_returns_and_times(df, market, build)
    target = build_target_from_returns(returns, times, market, build)
    splits = chronological_split_indices(target['X'].shape[0], build.split_train_frac, build.split_val_frac, build.eval_gap)
    gaps = None
    if times is not None:
        ts = _to_epoch_seconds(times)
        gaps = int(np.sum(np.diff(ts) != int(market.expected_interval_seconds)))
    return {
        'market': market.name,
        'csv_path': market.csv_path,
        'raw_rows': int(len(df)),
        'returns': int(len(returns)),
        'timestamp_gaps': gaps,
        'windows': int(target['X'].shape[0]),
        'train_windows': int(len(splits['train'])),
        'val_windows': int(len(splits['val'])),
        'test_windows': int(len(splits['test'])),
        'seq_len': int(build.seq_len),
        'horizons': [int(x) for x in build.horizons],
        'eval_gap': int(build.eval_gap),
    }
