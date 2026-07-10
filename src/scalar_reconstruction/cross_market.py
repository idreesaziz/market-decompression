from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .config import load_config
from .jobs import runtime_paths


def cross_market_correlations(config_path: str | Path, prediction_csv: str | Path) -> Path:
    cfg = load_config(config_path); paths = runtime_paths(cfg)
    df = pd.read_csv(prediction_csv)
    rows = []
    theta_cols = [f'theta_{j+1}' for j in range(6)]
    for b, gb in df.groupby('b'):
        # build a stable theta key by row order/rounded theta
        gb = gb.copy()
        gb['theta_key'] = gb[theta_cols].round(6).astype(str).agg('|'.join, axis=1)
        pivot = gb.pivot_table(index='theta_key', columns='market', values='pred_mean', aggfunc='mean')
        markets = list(pivot.columns)
        for i in range(len(markets)):
            for j in range(i+1, len(markets)):
                sub = pivot[[markets[i], markets[j]]].dropna()
                corr = float(sub[markets[i]].corr(sub[markets[j]])) if len(sub) > 2 else np.nan
                rows.append({'b': int(b), 'market_a': markets[i], 'market_b': markets[j], 'field_corr': corr, 'n_grid': int(len(sub))})
    out = paths['tables_dir'] / 'cross_market_correlations.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def budget_deformation(config_path: str | Path, prediction_csv: str | Path, threshold: float = 0.0) -> Path:
    cfg = load_config(config_path); paths = runtime_paths(cfg)
    df = pd.read_csv(prediction_csv)
    theta_cols = [f'theta_{j+1}' for j in range(6)]
    df = df.copy(); df['theta_key'] = df[theta_cols].round(6).astype(str).agg('|'.join, axis=1)
    rows = []
    for market, g in df.groupby('market'):
        masks = {}
        for b, gb in g.groupby('b'):
            masks[int(b)] = gb.set_index('theta_key')['pred_mean'] > threshold
        if 5 in masks and 10 in masks:
            both = pd.concat([masks[5], masks[10]], axis=1, join='inner').dropna()
            a = both.iloc[:,0].to_numpy(bool); c = both.iloc[:,1].to_numpy(bool)
            union = np.logical_or(a,c).sum(); inter = np.logical_and(a,c).sum()
            rows.append({'market': market, 'b_a': 5, 'b_b': 10, 'jaccard': float(inter/union) if union else np.nan, 'intersection': int(inter), 'union': int(union)})
    out = paths['tables_dir'] / 'budget_deformation.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    return out
