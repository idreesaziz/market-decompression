from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .config import FullConfig
from .design_pool import sobol_unit, denormalize_theta
from .surrogate import fit_surrogates
from .io_utils import ensure_dir, stable_id, theta_to_text


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def levelset_scores(
    surrogates,
    theta: np.ndarray,
    beta: float = 2.0,
    acquisition: str = 'straddle',
    threshold: float = 0.0,
) -> np.ndarray:
    """Score candidate theta for noisy level-set/envelope refinement.

    The returned score is always a non-negative reward, so scores can be summed
    across market/budget surrogates without penalizing market-specific boundary
    points. Earlier versions used MCU=-|z|, which is mathematically fine for a
    single group but wrong when summed across groups because all off-contour
    groups add negative penalties.
    """
    total = np.zeros(len(theta), dtype=float)
    for m in surrogates.values():
        mu, sd = m.predict(theta)
        sd = np.maximum(sd, 1e-9)
        z = (mu - threshold) / sd
        p = m.prob_positive(theta, threshold=threshold)
        acq = acquisition.lower()
        if acq == 'straddle':
            # Credit only points whose uncertainty band reaches the contour.
            s = np.maximum(0.0, np.sqrt(beta) * sd - np.abs(mu - threshold))
        elif acq == 'mcu':
            # Maximum contour uncertainty: high near the zero contour, bounded
            # and non-negative so multi-group summation remains meaningful.
            s = np.exp(-0.5 * z ** 2)
        elif acq == 'mee':
            # Local misclassification probability under posterior sign prob.
            s = np.minimum(p, 1.0 - p)
        elif acq in {'icu', 'entropy'}:
            # Information/uncertainty in the sign classification.
            s = _entropy(p)
        elif acq == 'tmse':
            # Targeted MSE: uncertainty weighted by proximity to contour.
            s = sd ** 2 * np.exp(-0.5 * z ** 2)
        elif acq == 'csur':
            # Cheap cSUR proxy: sign entropy weighted by posterior std.
            s = _entropy(p) * sd
        else:
            raise ValueError(f'unknown level-set acquisition {acquisition}')
        total += np.maximum(np.asarray(s, dtype=float), 0.0)
    return total


def make_levelset_actions(
    field_csv: str | Path,
    cfg: FullConfig,
    out_dir: str | Path,
    n_actions: int | None = None,
    metric: str | None = None,
    acquisition: str = 'straddle',
) -> Path:
    out_dir = ensure_dir(out_dir)
    n_actions = int(n_actions or cfg.design.levelset_n)
    metric = metric or cfg.surrogate.metric
    sur = fit_surrogates(field_csv, cfg, metric=metric)
    unit = sobol_unit(cfg.design.pool_size, d=cfg.design.theta_dim, seed=cfg.design.seed + 1001)
    theta = denormalize_theta(unit, cfg.design.theta_min, cfg.design.theta_max)
    score = levelset_scores(sur, theta, beta=cfg.surrogate.beta_levelset, acquisition=acquisition)
    order = np.argsort(-score)
    rows = []
    selected = []
    for idx in order:
        x = unit[idx]
        if all(np.linalg.norm(x - y) >= cfg.design.diverse_min_distance for y in selected):
            selected.append(x)
            th = theta[idx]
            rec = {
                'action_type': 'explore',
                'acquisition': acquisition,
                'acquisition_score': float(score[idx]),
                'action_rank': len(rows),
                'theta': theta_to_text(th.tolist()),
                'theta_id': stable_id({'theta': [round(float(t), 8) for t in th.tolist()]}),
                'design_source': f'levelset_{acquisition}',
                'initial_replicates': int(cfg.design.reconstruction_reps),
            }
            for j, v in enumerate(th):
                rec[f'theta_{j+1}'] = float(v)
            rows.append(rec)
        if len(rows) >= n_actions:
            break
    path = out_dir / f'levelset_{acquisition}_actions.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
