from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .config import ExperimentConfig, PROJECT_ROOT


LEROBOT_SOURCE = PROJECT_ROOT / "src" / "lerobot" / "src"


def runtime_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    paths = [str(LEROBOT_SOURCE), str(PROJECT_ROOT / "src")]
    existing = env.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def python_module_command(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", module, *args]


def record_command(
    config: ExperimentConfig,
    *,
    follower_port: str,
    leader_port: str,
    cameras: dict,
    dataset_root: Path | None = None,
    repo_id: str | None = None,
    episodes: int = 50,
    robot_id: str = "so101_follower",
    leader_id: str = "so101_leader",
    reset_time_s: int = 30,
) -> list[str]:
    dataset_root = dataset_root or config.dataset.root
    repo_id = repo_id or config.dataset.repo_id
    return python_module_command(
        "lerobot.record",
        "--robot.type=so101_follower",
        f"--robot.port={follower_port}",
        f"--robot.id={robot_id}",
        f"--robot.cameras={json.dumps(cameras, separators=(',', ':'))}",
        "--teleop.type=so101_leader",
        f"--teleop.port={leader_port}",
        f"--teleop.id={leader_id}",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={dataset_root}",
        f"--dataset.single_task={config.task}",
        f"--dataset.fps={config.dataset.fps}",
        f"--dataset.num_episodes={episodes}",
        f"--dataset.reset_time_s={reset_time_s}",
        "--dataset.push_to_hub=false",
        "--display_data=false",
    )


def rollout_command(
    config: ExperimentConfig,
    *,
    policy_path: str | Path,
    follower_port: str,
    cameras: dict,
    dataset_root: Path,
    episodes: int,
    robot_id: str = "so101_follower",
) -> list[str]:
    return python_module_command(
        "lerobot.record",
        "--robot.type=so101_follower",
        f"--robot.port={follower_port}",
        f"--robot.id={robot_id}",
        f"--robot.cameras={json.dumps(cameras, separators=(',', ':'))}",
        f"--policy.path={policy_path}",
        "--dataset.repo_id=local/so101_eval",
        f"--dataset.root={dataset_root}",
        f"--dataset.single_task={config.task}",
        f"--dataset.fps={config.dataset.fps}",
        f"--dataset.num_episodes={episodes}",
        "--dataset.push_to_hub=false",
        "--display_data=false",
    )
