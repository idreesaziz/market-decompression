from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List

from .io_utils import read_json


@dataclass(frozen=True)
class MarketConfig:
    name: str
    csv_path: str
    symbol: str | None = None
    time_col: str | None = None
    price_col: str = "close"
    return_col: str | None = None
    volume_col: str | None = None
    max_rows: int | None = 25_000
    expected_interval_seconds: int = 3600


@dataclass(frozen=True)
class DataBuildConfig:
    min_rows: int = 1_000
    seq_len: int = 192
    horizons: List[int] = field(default_factory=lambda: [5, 20, 60])
    feature_set: str = "returns_abs_rv"
    split_train_frac: float = 0.60
    split_val_frac: float = 0.20
    eval_gap: int = 64
    winsorize_quantiles: List[float] = field(default_factory=lambda: [0.0005, 0.9995])


@dataclass(frozen=True)
class DesignConfig:
    theta_dim: int = 6
    theta_min: float = -3.0
    theta_max: float = 3.0
    seed: int = 20260622
    pool_size: int = 4096
    seed_spacefill_n: int = 24
    seed_spacefill_reps: int = 2
    anchor_reps: int = 8
    ioptimal_n: int = 48
    spacefill_n: int = 40
    levelset_n: int = 24
    reconstruction_reps: int = 3
    calibration_budgets: List[int] = field(default_factory=lambda: [5, 10, 20, 40])
    reconstruction_budgets: List[int] = field(default_factory=lambda: [5, 10])
    # These are full-family 6D anchors. They are optional but useful for the BTC noise gate.
    anchors: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"name": "center", "theta": [0, 0, 0, 0, 0, 0], "role": "anchor"},
        {"name": "stage2_stable_full_fc555", "theta": [-1.106714816763997, 0.5829579290002584, -2.0108915716409683, -2.178876992315054, -1.3317950740456581, -1.655008664354682], "role": "stage2_positive"},
        {"name": "stage2_unstable_spike_729e", "theta": [2.6849591452628374, -0.8402213808149099, 2.832493392750621, -1.652742963284254, -1.8377255089581013, 1.5106442850083113], "role": "stage2_unstable"},
        {"name": "stage2_negative_bfe317", "theta": [2.935733925551176, 1.262687299400568, -2.945072801783681, 0.3439173046499491, 0.23808459006249905, -2.272738991305232], "role": "negative_control"},
        {"name": "axis_z2_neg", "theta": [0, -1.5, 0, 0, 0, 0], "role": "axis_control"},
        {"name": "axis_z6_neg", "theta": [0, 0, 0, 0, 0, -1.5], "role": "axis_control"},
    ])
    ioptimal_ridge: float = 1e-3
    min_dedup_distance: float = 1e-6
    diverse_min_distance: float = 0.18  # normalized space


@dataclass(frozen=True)
class TrainingRunConfig:
    synthetic_episodes_full: int = 2048
    pretrain_epochs: int = 12
    finetune_epochs: int = 180
    batch_size: int = 64
    lr: float = 2e-4
    finetune_lr: float = 8e-4
    latent_weight: float = 1.0
    device: str = "cpu"
    num_workers: int = 0
    eval_protocol: str = "matched_source_full"
    family: str = "full"


@dataclass(frozen=True)
class SurrogateConfig:
    metric: str = "log_skill"  # log_skill is the default modeling target; skill_T remains reported.
    kernel: str = "matern52"
    lengthscale: float = 0.35
    min_noise: float = 1e-6
    default_noise: float = 0.05
    beta_levelset: float = 2.0
    prediction_pool_size: int = 4096


@dataclass(frozen=True)
class RuntimeConfig:
    run_dir: str = "runs/scalar_crypto3_v1"
    jobs_dir: str = "jobs"
    results_dir: str = "results"
    logs_dir: str = "logs"
    heartbeat_s: int = 20
    lock_timeout_s: float = 120.0
    worker_sleep_s: float = 2.0
    max_job_retries: int = 1


@dataclass(frozen=True)
class FullConfig:
    markets: List[MarketConfig] = field(default_factory=list)
    data: DataBuildConfig = field(default_factory=DataBuildConfig)
    design: DesignConfig = field(default_factory=DesignConfig)
    train: TrainingRunConfig = field(default_factory=TrainingRunConfig)
    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _merge_dict(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(default)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def config_to_dict(cfg: FullConfig) -> Dict[str, Any]:
    return asdict(cfg)


def load_config(path: str | Path) -> FullConfig:
    path = Path(path)
    if path.suffix.lower() in {'.yaml', '.yml'}:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError('YAML config requested but PyYAML is not installed.') from e
        with path.open('r', encoding='utf-8') as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = read_json(path)
    base = asdict(FullConfig())
    merged = _merge_dict(base, raw)
    markets = [MarketConfig(**m) for m in merged.get('markets', [])]
    if not markets:
        raise ValueError('Config must define at least one market in `markets`.')
    return FullConfig(
        markets=markets,
        data=DataBuildConfig(**merged.get('data', {})),
        design=DesignConfig(**merged.get('design', {})),
        train=TrainingRunConfig(**merged.get('train', {})),
        surrogate=SurrogateConfig(**merged.get('surrogate', {})),
        runtime=RuntimeConfig(**merged.get('runtime', {})),
    )


def market_by_name(cfg: FullConfig, name: str) -> MarketConfig:
    for m in cfg.markets:
        if m.name == name or (m.symbol and m.symbol == name):
            return m
    raise KeyError(f"Market {name!r} not found. Available: {[m.name for m in cfg.markets]}")
