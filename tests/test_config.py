from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from soarm101_act.config import load_config, train_config_dict, validate_config


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.config = load_config("configs/act_so101.json", self.project_root)

    def test_reference_configuration_matches_act_parameters(self) -> None:
        self.assertEqual(self.config.policy.chunk_size, 100)
        self.assertEqual(self.config.policy.n_action_steps, 1)
        self.assertEqual(self.config.policy.temporal_ensemble_coeff, 0.01)
        self.assertEqual(self.config.policy.latent_dim, 32)
        self.assertEqual(self.config.policy.kl_weight, 10.0)

    def test_training_config_contains_lerobot_sections(self) -> None:
        serialized = train_config_dict(self.config)
        self.assertEqual(serialized["policy"]["type"], "act")
        self.assertIsNone(serialized["env"])
        self.assertFalse(serialized["policy"]["push_to_hub"])
        self.assertEqual(serialized["dataset"]["root"], str(self.config.dataset.root))

    def test_temporal_ensemble_rejects_action_chunks(self) -> None:
        policy = replace(self.config.policy, n_action_steps=10)
        invalid = replace(self.config, policy=policy)
        with self.assertRaisesRegex(ValueError, "Temporal ensembling"):
            validate_config(invalid)

    def test_json_config_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(train_config_dict(self.config)), encoding="utf-8")
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["steps"], 100_000)


if __name__ == "__main__":
    unittest.main()
