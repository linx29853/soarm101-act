from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DatasetSettings:
    repo_id: str
    root: Path
    fps: int


@dataclass(frozen=True)
class PolicySettings:
    type: str
    n_obs_steps: int
    chunk_size: int
    n_action_steps: int
    temporal_ensemble_coeff: float | None
    dim_model: int
    n_heads: int
    dim_feedforward: int
    n_encoder_layers: int
    n_decoder_layers: int
    use_vae: bool
    latent_dim: int
    n_vae_encoder_layers: int
    dropout: float
    kl_weight: float
    optimizer_lr: float
    optimizer_lr_backbone: float
    optimizer_weight_decay: float
    pretrained_backbone_weights: str | None
    use_amp: bool
    push_to_hub: bool


@dataclass(frozen=True)
class TrainingSettings:
    output_dir: Path
    job_name: str
    device: str
    batch_size: int
    num_workers: int
    steps: int
    log_freq: int
    save_freq: int
    seed: int
    wandb: bool


@dataclass(frozen=True)
class ExperimentConfig:
    version: int
    task: str
    dataset: DatasetSettings
    policy: PolicySettings
    training: TrainingSettings
    source_path: Path


def _resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def _require(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing '{section}.{key}' in experiment configuration")
    return mapping[key]


def load_config(path: str | Path, project_root: Path = PROJECT_ROOT) -> ExperimentConfig:
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config_path = config_path.resolve()

    with config_path.open(encoding="utf-8") as file:
        raw = json.load(file)

    dataset_raw = _require(raw, "dataset", "root")
    policy_raw = _require(raw, "policy", "root")
    training_raw = _require(raw, "training", "root")

    dataset = DatasetSettings(
        repo_id=str(_require(dataset_raw, "repo_id", "dataset")),
        root=_resolve_path(_require(dataset_raw, "root", "dataset"), project_root),
        fps=int(dataset_raw.get("fps", 30)),
    )
    policy = PolicySettings(
        type=str(policy_raw.get("type", "act")),
        n_obs_steps=int(policy_raw.get("n_obs_steps", 1)),
        chunk_size=int(policy_raw.get("chunk_size", 100)),
        n_action_steps=int(policy_raw.get("n_action_steps", 1)),
        temporal_ensemble_coeff=policy_raw.get("temporal_ensemble_coeff", 0.01),
        dim_model=int(policy_raw.get("dim_model", 512)),
        n_heads=int(policy_raw.get("n_heads", 8)),
        dim_feedforward=int(policy_raw.get("dim_feedforward", 3200)),
        n_encoder_layers=int(policy_raw.get("n_encoder_layers", 4)),
        n_decoder_layers=int(policy_raw.get("n_decoder_layers", 1)),
        use_vae=bool(policy_raw.get("use_vae", True)),
        latent_dim=int(policy_raw.get("latent_dim", 32)),
        n_vae_encoder_layers=int(policy_raw.get("n_vae_encoder_layers", 4)),
        dropout=float(policy_raw.get("dropout", 0.1)),
        kl_weight=float(policy_raw.get("kl_weight", 10.0)),
        optimizer_lr=float(policy_raw.get("optimizer_lr", 1e-5)),
        optimizer_lr_backbone=float(policy_raw.get("optimizer_lr_backbone", 1e-5)),
        optimizer_weight_decay=float(policy_raw.get("optimizer_weight_decay", 1e-4)),
        pretrained_backbone_weights=policy_raw.get(
            "pretrained_backbone_weights", "ResNet18_Weights.IMAGENET1K_V1"
        ),
        use_amp=bool(policy_raw.get("use_amp", True)),
        push_to_hub=bool(policy_raw.get("push_to_hub", False)),
    )
    training = TrainingSettings(
        output_dir=_resolve_path(
            training_raw.get("output_dir", "outputs/train/act_so101"), project_root
        ),
        job_name=str(training_raw.get("job_name", "act_so101")),
        device=str(training_raw.get("device", "cuda")),
        batch_size=int(training_raw.get("batch_size", 8)),
        num_workers=int(training_raw.get("num_workers", 4)),
        steps=int(training_raw.get("steps", 100_000)),
        log_freq=int(training_raw.get("log_freq", 100)),
        save_freq=int(training_raw.get("save_freq", 10_000)),
        seed=int(training_raw.get("seed", 1000)),
        wandb=bool(training_raw.get("wandb", False)),
    )

    config = ExperimentConfig(
        version=int(raw.get("version", 1)),
        task=str(_require(raw, "task", "root")),
        dataset=dataset,
        policy=policy,
        training=training,
        source_path=config_path,
    )
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    if config.policy.type != "act":
        raise ValueError("This project currently targets policy.type='act'")
    if not config.dataset.repo_id.strip():
        raise ValueError("dataset.repo_id cannot be empty")
    if not config.task.strip():
        raise ValueError("task cannot be empty")
    if config.policy.n_obs_steps != 1:
        raise ValueError("The LeRobot ACT implementation supports n_obs_steps=1")
    if config.policy.n_action_steps > config.policy.chunk_size:
        raise ValueError("policy.n_action_steps cannot exceed policy.chunk_size")
    if config.policy.temporal_ensemble_coeff is not None and config.policy.n_action_steps != 1:
        raise ValueError("Temporal ensembling requires policy.n_action_steps=1")
    if config.policy.dim_model % config.policy.n_heads != 0:
        raise ValueError("policy.dim_model must be divisible by policy.n_heads")
    if config.policy.chunk_size <= 0:
        raise ValueError("policy.chunk_size must be positive")
    if config.training.batch_size <= 0:
        raise ValueError("training.batch_size must be positive")
    if config.training.steps <= 0:
        raise ValueError("training.steps must be positive")
    if config.training.save_freq <= 0 or config.training.log_freq <= 0:
        raise ValueError("training.save_freq and training.log_freq must be positive")


def policy_dict(config: ExperimentConfig) -> dict[str, Any]:
    policy = config.policy
    return {
        "type": policy.type,
        "n_obs_steps": policy.n_obs_steps,
        "chunk_size": policy.chunk_size,
        "n_action_steps": policy.n_action_steps,
        "temporal_ensemble_coeff": policy.temporal_ensemble_coeff,
        "dim_model": policy.dim_model,
        "n_heads": policy.n_heads,
        "dim_feedforward": policy.dim_feedforward,
        "n_encoder_layers": policy.n_encoder_layers,
        "n_decoder_layers": policy.n_decoder_layers,
        "use_vae": policy.use_vae,
        "latent_dim": policy.latent_dim,
        "n_vae_encoder_layers": policy.n_vae_encoder_layers,
        "dropout": policy.dropout,
        "kl_weight": policy.kl_weight,
        "optimizer_lr": policy.optimizer_lr,
        "optimizer_lr_backbone": policy.optimizer_lr_backbone,
        "optimizer_weight_decay": policy.optimizer_weight_decay,
        "pretrained_backbone_weights": policy.pretrained_backbone_weights,
        "use_amp": policy.use_amp,
        "push_to_hub": policy.push_to_hub,
        "device": config.training.device,
    }


def train_config_dict(config: ExperimentConfig) -> dict[str, Any]:
    training = config.training
    return {
        "dataset": {
            "repo_id": config.dataset.repo_id,
            "root": str(config.dataset.root),
        },
        "env": None,
        "policy": policy_dict(config),
        "output_dir": str(training.output_dir),
        "job_name": training.job_name,
        "resume": False,
        "seed": training.seed,
        "num_workers": training.num_workers,
        "batch_size": training.batch_size,
        "steps": training.steps,
        "eval_freq": 0,
        "log_freq": training.log_freq,
        "save_checkpoint": True,
        "save_freq": training.save_freq,
        "use_policy_training_preset": True,
        "optimizer": None,
        "scheduler": None,
        "eval": {"n_episodes": 1, "batch_size": 1, "use_async_envs": False},
        "wandb": {
            "enable": training.wandb,
            "disable_artifact": False,
            "project": "soarm101-act",
            "entity": None,
            "notes": config.task,
            "run_id": None,
            "mode": None,
        },
    }
