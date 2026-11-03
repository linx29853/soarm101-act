from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/act_so101.json")
    args = parser.parse_args()

    try:
        import torch

        from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
    except ImportError as error:
        print(f"ERROR: ACT dependencies are unavailable: {error}", file=sys.stderr)
        print("Install them with `bash scripts/bootstrap.sh --install`.", file=sys.stderr)
        return 1

    state_dim = 6
    action_dim = 6
    image_shape = (3, 64, 64)
    chunk_size = 8

    features = {
        "observation.images.front": PolicyFeature(
            type=FeatureType.VISUAL, shape=image_shape
        ),
        "observation.state": PolicyFeature(
            type=FeatureType.STATE, shape=(state_dim,)
        ),
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
    }
    config = ACTConfig(
        n_obs_steps=1,
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        temporal_ensemble_coeff=None,
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        dim_model=64,
        n_heads=4,
        dim_feedforward=128,
        n_encoder_layers=1,
        n_decoder_layers=1,
        use_vae=True,
        latent_dim=8,
        n_vae_encoder_layers=1,
        dropout=0.0,
        kl_weight=10.0,
        input_features={
            "observation.images.front": features["observation.images.front"],
            "observation.state": features["observation.state"],
        },
        output_features={"action": features["action"]},
        normalization_mapping={
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
        device="cpu",
    )

    stats = {
        "observation.images.front": {
            "mean": torch.zeros(1, 3, 1, 1),
            "std": torch.ones(1, 3, 1, 1),
            "min": torch.zeros(1, 3, 1, 1),
            "max": torch.ones(1, 3, 1, 1),
        },
        "observation.state": {
            "mean": torch.zeros(1, state_dim),
            "std": torch.ones(1, state_dim),
            "min": -torch.ones(1, state_dim),
            "max": torch.ones(1, state_dim),
        },
        "action": {
            "mean": torch.zeros(1, action_dim),
            "std": torch.ones(1, action_dim),
            "min": -torch.ones(1, action_dim),
            "max": torch.ones(1, action_dim),
        },
    }

    torch.manual_seed(0)
    policy = ACTPolicy(config, dataset_stats=stats)
    policy.train()
    batch = {
        "observation.images.front": torch.rand(2, *image_shape),
        "observation.state": torch.randn(2, state_dim),
        "action": torch.randn(2, chunk_size, action_dim),
        "action_is_pad": torch.zeros(2, chunk_size, dtype=torch.bool),
    }
    loss, metrics = policy.forward(batch)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)

    if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
        print("ERROR: ACT smoke test produced a non-finite loss or gradient", file=sys.stderr)
        return 1

    policy.eval()
    with torch.inference_mode():
        action = policy.select_action(
            {
                "observation.images.front": batch["observation.images.front"],
                "observation.state": batch["observation.state"],
            }
        )
    if tuple(action.shape) != (2, action_dim):
        print(f"ERROR: Unexpected action shape {tuple(action.shape)}", file=sys.stderr)
        return 1

    print(f"ACT smoke test passed: loss={loss.item():.4f}, grad_norm={grad_norm.item():.4f}")
    print(f"Metrics: {metrics}")
    print(f"Action shape: {tuple(action.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
