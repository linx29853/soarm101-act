#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${SOARM_PYTHON:-${HOME}/miniconda3/envs/lerobot/bin/python}"
exec "${PYTHON_BIN}" scripts/record_with_terminal_keys.py \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5CE5122712-if00 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras='{"front":{"type":"opencv","index_or_path":"/dev/video6","width":640,"height":480,"fps":30,"fourcc":"MJPG"},"handeye":{"type":"opencv","index_or_path":"/dev/video4","width":640,"height":480,"fps":30}}' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5CE5122659-if00 \
  --teleop.id=my_awesome_leader_arm \
  --display_data=false \
  --dataset.repo_id=local/so101_cube_trial \
  --dataset.root="${ROOT_DIR}"/data/datasets/so101_cube_trial \
  --dataset.single_task="Pick up the cube and place it into the box." \
  --dataset.fps=30 \
  --dataset.num_episodes=5 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=30 \
  --dataset.push_to_hub=false
