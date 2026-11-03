from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from soarm101_act.commands import record_command, rollout_command
from soarm101_act.config import load_config


class CommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.config = load_config("configs/act_so101.json", self.root)
        self.cameras = {
            "front": {
                "type": "opencv",
                "index_or_path": "/dev/video0",
                "width": 640,
                "height": 480,
                "fps": 30,
            }
        }

    def test_record_disables_hub_upload(self) -> None:
        command = record_command(
            self.config,
            follower_port="/dev/ttyACM0",
            leader_port="/dev/ttyACM1",
            cameras=self.cameras,
            episodes=5,
        )
        self.assertIn("--robot.type=so101_follower", command)
        self.assertIn("--teleop.type=so101_leader", command)
        self.assertIn("--dataset.push_to_hub=false", command)
        self.assertIn("--dataset.num_episodes=5", command)

    def test_rollout_uses_policy_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = rollout_command(
                self.config,
                policy_path="/tmp/checkpoint",
                follower_port="/dev/ttyACM0",
                cameras=self.cameras,
                dataset_root=Path(directory),
                episodes=3,
            )
        self.assertIn("--policy.path=/tmp/checkpoint", command)
        self.assertIn("--dataset.num_episodes=3", command)
        self.assertFalse(any(argument.startswith("--teleop.") for argument in command))


if __name__ == "__main__":
    unittest.main()
