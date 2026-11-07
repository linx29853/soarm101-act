#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${SOARM_PYTHON:-${HOME}/miniconda3/envs/lerobot/bin/python}"
plan=$("${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path

root = Path('data/datasets/so101_task')
count = 0
if root.exists():
    info = json.loads((root / 'meta/info.json').read_text())
    count = info['total_episodes']
    if not isinstance(count, int) or count < 1:
        raise SystemExit('Existing dataset has no saved episodes; inspect it before continuing.')
    for index in range(count):
        chunk = index // info['chunks_size']
        paths = [root / info['data_path'].format(episode_chunk=chunk, episode_index=index)]
        for key, feature in info['features'].items():
            if feature.get('dtype') == 'video':
                paths.append(root / info['video_path'].format(
                    episode_chunk=chunk, episode_index=index, video_key=key))
        for path in paths:
            if not path.is_file() or path.stat().st_size == 0:
                raise SystemExit(f'Missing or empty saved episode file: {path}')
print(count, max(0, 50 - count), 'true' if root.exists() else 'false')
PY
)
read -r saved remaining resume <<< "$plan"
echo "Saved episodes: $saved; remaining: $remaining"
if [ "$remaining" -eq 0 ]; then
  echo 'Dataset already contains at least 50 episodes.'
  exit 0
fi
exec "${PYTHON_BIN}" scripts/record_with_terminal_keys.py \
  --resume="$resume" \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5CE5122712-if00 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras='{"front":{"type":"opencv","index_or_path":"/dev/video6","width":640,"height":480,"fps":30,"fourcc":"MJPG"},"handeye":{"type":"opencv","index_or_path":"/dev/video4","width":640,"height":480,"fps":30}}' \
  --teleop.type=so101_leader \
  --teleop.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5CE5122659-if00 \
  --teleop.id=my_awesome_leader_arm \
  --display_data=false \
  --dataset.repo_id=local/so101_task \
  --dataset.root="${ROOT_DIR}"/data/datasets/so101_task \
  --dataset.single_task="Pick up the cube and place it into the box." \
  --dataset.fps=30 \
  --dataset.num_episodes="$remaining" \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=60 \
  --dataset.push_to_hub=false
