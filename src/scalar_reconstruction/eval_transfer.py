from __future__ import annotations

import copy
import time
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .skill_score import paired_skill


def _normalize_target_X_with_train_stats(raw: Mapping[str, np.ndarray], train_idx: np.ndarray):
    from mde_search.training import TensorPack
    from mde_search.utils import standardize
    X = np.asarray(raw['X'], dtype=np.float32)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    _, x_mean, x_std = standardize(X[train_idx])
    Xz, _, _ = standardize(X, x_mean, x_std)
    return TensorPack(Xz.astype(np.float32), np.asarray(raw['Y'], dtype=np.float32), raw.get('Z'))


def _prepare_chrono_budget(target, support_pool: np.ndarray, eval_pool: np.ndarray, b: int, seed: int):
    from mde_search.utils import standardize_2d
    support_pool = np.asarray(support_pool, dtype=np.int64)
    eval_pool = np.asarray(eval_pool, dtype=np.int64)
    if b > len(support_pool):
        raise ValueError(f'label budget b={b} exceeds support pool size {len(support_pool)}')
    rng = np.random.default_rng(seed)
    support_idx = np.sort(rng.choice(support_pool, size=int(b), replace=False))
    eval_idx = eval_pool
    Y_support = target.Y[support_idx]
    Y_eval = target.Y[eval_idx]
    Y_support_z, y_mean, y_std = standardize_2d(Y_support)
    Y_eval_z, _, _ = standardize_2d(Y_eval, y_mean, y_std)
    return target.X[support_idx], Y_support_z, target.X[eval_idx], Y_eval_z, support_idx, eval_idx


def matched_transfer_eval_real_chrono(
    theta: Sequence[float],
    target_raw: Mapping[str, np.ndarray],
    label_budgets: Sequence[int],
    seed: int,
    train_cfg,
    sim_cfg,
    model_cfg=None,
    eval_split: str = 'val',
    train_frac: float = 0.60,
    val_frac: float = 0.20,
    gap: int = 64,
    finetune_lr_schedule: dict | None = None,
) -> Dict[str, Any]:
    """Evaluate scalar transfer field F_b(θ;R) across label budgets.

    finetune_lr_schedule: optional dict mapping str(b) -> float, e.g. {"5": 0.0008, "10": 0.0004}.
    When provided, each budget uses its specified finetune_lr instead of train_cfg.finetune_lr.
    """
    import torch
    from mde_search.model import MarketTransformer, ModelConfig
    from mde_search.simulator import generate_synthetic_dataset
    from mde_search.training import evaluate_mse, finetune_head, make_loader, normalize_source, pack_to_tensors, train_pretrain
    from mde_search.utils import get_device, set_seed
    from scalar_reconstruction.data import chronological_split_indices

    if eval_split not in {'val', 'test'}:
        raise ValueError("eval_split must be 'val' or 'test'")
    set_seed(seed)
    device = get_device(train_cfg.device)
    model_cfg = model_cfg or ModelConfig(seq_len=sim_cfg.L, n_horizons=len(sim_cfg.horizons))
    n = int(np.asarray(target_raw['X']).shape[0])
    splits = chronological_split_indices(n, train_frac=train_frac, val_frac=val_frac, gap=gap)
    support_pool = splits['train']
    eval_pool = splits[eval_split]
    raw_source = generate_synthetic_dataset(theta, train_cfg.synthetic_episodes, seed=seed + 17, cfg=sim_cfg)
    source = normalize_source(raw_source)
    target = _normalize_target_X_with_train_stats(target_raw, support_pool)
    source_ds = pack_to_tensors(source, include_z=True)
    loader_seed = seed + 31
    loader_only = make_loader(source_ds, train_cfg.batch_size, shuffle=True, seed=loader_seed, num_workers=train_cfg.num_workers)
    loader_plus = make_loader(source_ds, train_cfg.batch_size, shuffle=True, seed=loader_seed, num_workers=train_cfg.num_workers)
    set_seed(seed + 101)
    initial = MarketTransformer(model_cfg)
    initial_state = copy.deepcopy(initial.state_dict())
    forecast_only = MarketTransformer(model_cfg); forecast_plus = MarketTransformer(model_cfg)
    forecast_only.load_state_dict(initial_state); forecast_plus.load_state_dict(initial_state)
    train_pretrain(forecast_only, loader_only, train_cfg, use_latent=False, epochs=train_cfg.pretrain_epochs, device=device)
    train_pretrain(forecast_plus, loader_plus, train_cfg, use_latent=True, epochs=train_cfg.pretrain_epochs, device=device)
    metrics: Dict[str, Dict[str, Any]] = {}; split_meta: Dict[str, Dict[str, Any]] = {}
    for b in label_budgets:
        eff_lr = float(finetune_lr_schedule[str(int(b))]) if (finetune_lr_schedule and str(int(b)) in finetune_lr_schedule) else train_cfg.finetune_lr
        b_train_cfg = _make_train_cfg_with_lr(train_cfg, eff_lr) if eff_lr != train_cfg.finetune_lr else train_cfg
        X_tr, Y_tr, X_ev, Y_ev, support_idx, eval_idx = _prepare_chrono_budget(target, support_pool, eval_pool, int(b), seed=seed + 251)
        m_only = finetune_head(forecast_only, X_tr, Y_tr, b_train_cfg, seed=seed + 401 + int(b), device=device)
        m_plus = finetune_head(forecast_plus, X_tr, Y_tr, b_train_cfg, seed=seed + 401 + int(b), device=device)
        mse_only, mse_only_h = evaluate_mse(m_only, X_ev, Y_ev, b_train_cfg, device)
        mse_plus, mse_plus_h = evaluate_mse(m_plus, X_ev, Y_ev, b_train_cfg, device)
        skill = paired_skill(mse_plus, mse_only)
        T_h = 1.0 - mse_plus_h / np.maximum(mse_only_h, 1e-12)
        log_h = np.log(np.maximum(mse_only_h, 1e-12)) - np.log(np.maximum(mse_plus_h, 1e-12))
        metrics[str(int(b))] = {
            'skill_T': float(skill['skill_T']), 'log_skill': float(skill['log_skill']), 'ratio': float(skill['ratio']), 'hit': bool(skill['hit']),
            'T': float(skill['skill_T']),  # backward compatible alias
            'T_by_horizon': T_h.astype(float).tolist(), 'log_skill_by_horizon': log_h.astype(float).tolist(),
            'mse_forecast_only': float(mse_only), 'mse_transfer': float(mse_plus), 'mse_forecast_plus_latent': float(mse_plus),
            'mse_forecast_only_by_horizon': mse_only_h.astype(float).tolist(), 'mse_transfer_by_horizon': mse_plus_h.astype(float).tolist(),
            'finetune_lr': float(eff_lr),
        }
        split_meta[str(int(b))] = {'support_start': int(support_idx[0]), 'support_end': int(support_idx[-1]), 'eval_start': int(eval_idx[0]), 'eval_end': int(eval_idx[-1]), 'n_support': int(len(support_idx)), 'n_eval': int(len(eval_idx))}
    return {'theta': [float(x) for x in theta], 'seed': int(seed), 'proxy': False, 'synthetic_episodes': int(train_cfg.synthetic_episodes), 'pretrain_epochs': int(train_cfg.pretrain_epochs), 'eval_split': eval_split, 'chronological_split': {k: [int(v[0]), int(v[-1]), int(len(v))] for k, v in splits.items()}, 'budget_split_meta': split_meta, 'metrics': metrics}


def _make_train_cfg_with_lr(train_cfg, finetune_lr: float):
    """Return a new TrainConfig identical to train_cfg but with a different finetune_lr."""
    from mde_search.training import TrainConfig
    return TrainConfig(
        synthetic_episodes=train_cfg.synthetic_episodes,
        proxy_synthetic_episodes=getattr(train_cfg, 'proxy_synthetic_episodes', train_cfg.synthetic_episodes),
        pretrain_epochs=train_cfg.pretrain_epochs,
        proxy_pretrain_epochs=getattr(train_cfg, 'proxy_pretrain_epochs', train_cfg.pretrain_epochs),
        finetune_epochs=train_cfg.finetune_epochs,
        batch_size=train_cfg.batch_size,
        lr=train_cfg.lr,
        finetune_lr=float(finetune_lr),
        latent_weight=train_cfg.latent_weight,
        num_workers=train_cfg.num_workers,
        device=train_cfg.device,
    )


def build_train_model_sim_cfg(cfg, family: str, finetune_lr_override: float | None = None):
    from mde_search.model import ModelConfig
    from mde_search.simulator import SimulatorConfig
    from mde_search.training import TrainConfig
    sim_cfg = SimulatorConfig(L=cfg.data.seq_len, horizons=tuple(cfg.data.horizons), module=family)
    model_cfg = ModelConfig(seq_len=sim_cfg.L, n_horizons=len(sim_cfg.horizons))
    effective_finetune_lr = float(finetune_lr_override) if finetune_lr_override is not None else cfg.train.finetune_lr
    train_cfg = TrainConfig(
        synthetic_episodes=cfg.train.synthetic_episodes_full,
        proxy_synthetic_episodes=cfg.train.synthetic_episodes_full,
        pretrain_epochs=cfg.train.pretrain_epochs,
        proxy_pretrain_epochs=cfg.train.pretrain_epochs,
        finetune_epochs=cfg.train.finetune_epochs,
        batch_size=cfg.train.batch_size,
        lr=cfg.train.lr,
        finetune_lr=effective_finetune_lr,
        latent_weight=cfg.train.latent_weight,
        num_workers=cfg.train.num_workers,
        device=cfg.train.device,
    )
    return sim_cfg, model_cfg, train_cfg


def execute_transfer_job(job: Mapping[str, Any], cfg, target_raw: Mapping[str, Any]) -> Dict[str, Any]:
    family = str(job.get('family', cfg.train.family))
    theta = [float(x) for x in job['theta']]
    seed = int(job['seed'])
    label_budgets = [int(x) for x in job['label_budgets']]
    # Per-budget LR schedule takes precedence; fall back to per-job override; fall back to config default.
    finetune_lr_schedule = job.get('finetune_lr_schedule') or {}
    finetune_lr_override = job.get('finetune_lr')  # legacy single-value override
    sim_cfg, model_cfg, train_cfg = build_train_model_sim_cfg(cfg, family, finetune_lr_override=finetune_lr_override)
    eval_split = str(job.get('eval_split') or ('test' if str(job.get('stage')) == 'test_confirm' else 'val'))
    t0 = time.time()
    rec = matched_transfer_eval_real_chrono(
        theta, target_raw, label_budgets, seed, train_cfg, sim_cfg, model_cfg,
        eval_split=eval_split, train_frac=cfg.data.split_train_frac,
        val_frac=cfg.data.split_val_frac, gap=cfg.data.eval_gap,
        finetune_lr_schedule=finetune_lr_schedule or None,
    )
    elapsed = time.time() - t0
    out = dict(rec)
    out.update({
        'job_id': job.get('job_id'), 'stage': job.get('stage'), 'market': job.get('market'), 'target_name': job.get('market'), 'family': family,
        'theta_id': job.get('theta_id'), 'design_source': job.get('design_source'), 'role': job.get('role'),
        'replicate_index': job.get('replicate_index'), 'action_type': job.get('action_type'), 'acquisition': job.get('acquisition'),
        'control_type': job.get('control_type'), 'eval_protocol': cfg.train.eval_protocol, 'elapsed_s': float(elapsed), 'completed_at': time.time(),
        'finetune_lr': float(train_cfg.finetune_lr),
        'finetune_lr_schedule': dict(finetune_lr_schedule) if finetune_lr_schedule else {},
    })
    return out
