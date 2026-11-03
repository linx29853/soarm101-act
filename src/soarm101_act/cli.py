from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .commands import python_module_command, record_command, rollout_command, runtime_environment
from .config import PROJECT_ROOT, ExperimentConfig, load_config, train_config_dict


def _camera_arguments(args: argparse.Namespace) -> dict:
    camera_json = args.cameras or (
        '{"front":{"type":"opencv","index_or_path":"/dev/video0",'
        '"width":640,"height":480,"fps":30}}'
    )
    cameras = json.loads(camera_json)
    if not isinstance(cameras, dict) or not cameras:
        raise ValueError("Camera configuration must be a non-empty JSON object")
    return cameras


def _require_value(value: str | None, env_name: str, option_name: str) -> str:
    resolved = value
    if not resolved:
        import os

        resolved = os.environ.get(env_name)
    if not resolved:
        raise ValueError(f"Set {option_name} or {env_name}")
    return resolved


def _train_args(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    from dataclasses import replace

    training = replace(
        config.training,
        output_dir=Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else config.training.output_dir,
        device=args.device or config.training.device,
        batch_size=args.batch_size or config.training.batch_size,
        num_workers=args.num_workers if args.num_workers is not None else config.training.num_workers,
        steps=args.steps or config.training.steps,
        save_freq=args.save_freq or config.training.save_freq,
        wandb=args.wandb if args.wandb is not None else config.training.wandb,
    )
    return replace(config, training=training)


def _write_train_config(config: ExperimentConfig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(train_config_dict(config), indent=2) + "\n", encoding="utf-8")
    return path


def _check(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    failures: list[str] = []
    warnings: list[str] = []

    if sys.version_info < (3, 10):
        failures.append(f"Python >=3.10 is required, found {sys.version.split()[0]}")

    lerobot_src = PROJECT_ROOT / "src" / "lerobot" / "src" / "lerobot"
    if not lerobot_src.is_dir():
        failures.append("LeRobot checkout is missing; run `bash scripts/bootstrap.sh`")
    if not shutil.which("lerobot-train"):
        warnings.append("lerobot-train is not on PATH; run `bash scripts/bootstrap.sh --install`")

    dataset_info = config.dataset.root / "meta" / "info.json"
    if not dataset_info.is_file():
        warnings.append(f"Dataset metadata is missing: {dataset_info}")

    try:
        import torch
    except ImportError:
        warnings.append("PyTorch is not installed in the active Python environment")
    else:
        if config.training.device == "cuda" and not torch.cuda.is_available():
            warnings.append("training.device=cuda, but CUDA is unavailable")
        if config.training.device == "cuda" and torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            print(f"CUDA device: {properties.name} ({properties.total_memory / 2**30:.1f} GiB)")

    print(f"Config: {config.source_path}")
    print(f"Task: {config.task}")
    print(f"Dataset root: {config.dataset.root}")
    print(f"ACT: chunk={config.policy.chunk_size}, dim={config.policy.dim_model}, "
          f"VAE={config.policy.use_vae}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    for failure in failures:
        print(f"ERROR: {failure}")

    if failures or (args.strict and warnings):
        return 1
    print("Environment check passed")
    return 0


def _train(args: argparse.Namespace) -> int:
    config = _train_args(load_config(args.config), args)
    if not (config.dataset.root / "meta" / "info.json").is_file():
        raise FileNotFoundError(
            f"Dataset metadata not found at {config.dataset.root}. Record or download a LeRobot v2.1 dataset."
        )
    config_path = _write_train_config(config, Path(args.effective_config).expanduser().resolve())
    command = python_module_command("lerobot.scripts.train", f"--config_path={config_path}")
    print(f"Effective training config: {config_path}")
    print("Running:", " ".join(command))
    return subprocess.run(command, env=runtime_environment(), cwd=PROJECT_ROOT, check=False).returncode


def _record(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    follower_port = _require_value(args.follower_port, "SOARM_FOLLOWER_PORT", "--follower-port")
    leader_port = _require_value(args.leader_port, "SOARM_LEADER_PORT", "--leader-port")
    command = record_command(
        config,
        follower_port=follower_port,
        leader_port=leader_port,
        cameras=_camera_arguments(args),
        episodes=args.episodes,
    )
    print("Running:", " ".join(command))
    return subprocess.run(command, env=runtime_environment(), cwd=PROJECT_ROOT, check=False).returncode


def _rollout(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    follower_port = _require_value(args.follower_port, "SOARM_FOLLOWER_PORT", "--follower-port")
    dataset_root = (
        Path(args.dataset_root).expanduser().resolve()
        if args.dataset_root
        else PROJECT_ROOT / "data" / "datasets" / "so101_eval"
    )
    if dataset_root.exists() and any(dataset_root.iterdir()):
        raise FileExistsError(f"Evaluation dataset directory is not empty: {dataset_root}")
    command = rollout_command(
        config,
        policy_path=Path(args.policy_path).expanduser().resolve(),
        follower_port=follower_port,
        cameras=_camera_arguments(args),
        dataset_root=dataset_root,
        episodes=args.episodes,
    )
    print("Running:", " ".join(command))
    return subprocess.run(command, env=runtime_environment(), cwd=PROJECT_ROOT, check=False).returncode


def _train_smoke(args: argparse.Namespace) -> int:
    command = python_module_command("scripts.smoke_act", "--config", args.config)
    return subprocess.run(command, env=runtime_environment(), cwd=PROJECT_ROOT, check=False).returncode


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="soarm101-act")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Check environment, dataset, and configuration")
    check.add_argument("--config", default="configs/act_so101.json")
    check.add_argument("--strict", action="store_true")
    check.set_defaults(func=_check)

    train = subparsers.add_parser("train", help="Train ACT on a LeRobot dataset")
    train.add_argument("--config", default="configs/act_so101.json")
    train.add_argument("--output-dir")
    train.add_argument("--device")
    train.add_argument("--batch-size", type=int)
    train.add_argument("--num-workers", type=int)
    train.add_argument("--steps", type=int)
    train.add_argument("--save-freq", type=int)
    train.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=None)
    train.add_argument(
        "--effective-config", default="outputs/configs/act_so101_train.json"
    )
    train.set_defaults(func=_train)

    record = subparsers.add_parser("record", help="Record SO-ARM101 demonstrations")
    record.add_argument("--config", default="configs/act_so101.json")
    record.add_argument("--follower-port")
    record.add_argument("--leader-port")
    record.add_argument("--cameras", help="Camera JSON object")
    record.add_argument("--episodes", type=int, default=50)
    record.set_defaults(func=_record)

    rollout = subparsers.add_parser("rollout", help="Run a trained ACT checkpoint")
    rollout.add_argument("--config", default="configs/act_so101.json")
    rollout.add_argument("--policy-path", required=True)
    rollout.add_argument("--follower-port")
    rollout.add_argument("--cameras", help="Camera JSON object")
    rollout.add_argument("--dataset-root")
    rollout.add_argument("--episodes", type=int, default=10)
    rollout.set_defaults(func=_rollout)

    smoke = subparsers.add_parser("train-smoke", help="Run a synthetic ACT train step")
    smoke.add_argument("--config", default="configs/act_so101.json")
    smoke.set_defaults(func=_train_smoke)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        raise SystemExit(args.func(args))
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
