# Data

The raw price data is **not included** in this repository (files are large and
subject to Binance ToS). This directory is a placeholder.

## What the pipeline uses

| Field | Value |
|---|---|
| Exchange | Binance |
| Interval | 1-hour candles (OHLCV) |
| Markets | BTCUSDT · ETHUSDT · SOLUSDT |
| Train window | 2021-01-01 – 2022-12-31 |
| Val window | 2023-01-01 – 2023-08-31 |
| Test window | 2023-09-01 – 2024-03-31 (held out; not used in discovery) |

## Format

Each market is a single CSV file named `<MARKET>_1h.csv`:

```
open_time,open,high,low,close,volume
1609459200000,29000.5,29300.0,28900.0,29100.0,4512.3
...
```

`open_time` is a Unix timestamp in milliseconds (UTC).

## Download instructions

```python
import requests, pandas as pd, time

def fetch_binance_klines(symbol, interval="1h", start_ms=None, end_ms=None):
    url = "https://api.binance.com/api/v3/klines"
    rows = []
    params = dict(symbol=symbol, interval=interval, limit=1000)
    if start_ms:
        params["startTime"] = start_ms
    while True:
        r = requests.get(url, params=params, timeout=10).json()
        if not r:
            break
        rows.extend(r)
        last_ts = r[-1][0]
        if end_ms and last_ts >= end_ms:
            break
        params["startTime"] = last_ts + 1
        time.sleep(0.2)
    df = pd.DataFrame(rows, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_vol","n_trades","taker_buy_base","taker_buy_quote","ignore"
    ])
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)
    return df[["open_time","open","high","low","close","volume"]]

import datetime
START = int(datetime.datetime(2021, 1, 1).timestamp() * 1000)
END   = int(datetime.datetime(2024, 4, 1).timestamp() * 1000)

for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
    df = fetch_binance_klines(sym, start_ms=START, end_ms=END)
    df.to_csv(f"data/{sym}_1h.csv", index=False)
    print(f"{sym}: {len(df)} rows")
```

After downloading, point `configs/experiment.json` → `"data_root": "data/"`.

## Normalization

All normalization (z-score of log-returns) is fitted on the **training split
only** and applied consistently to val and test. The pipeline enforces this via
`_normalize_target_X_with_train_stats()` in `src/scalar_reconstruction/eval_transfer.py`.
No information from val or test leaks into the model or the feature statistics.
