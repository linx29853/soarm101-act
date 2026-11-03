#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_DIR="${ROOT_DIR}/src/lerobot"
LEROBOT_COMMIT="882c80d446a63a44868c67ae535467af32ce0e80"
PATCH_FILE="${ROOT_DIR}/patches/0001-so101-adapter.patch"
INSTALL=false

if [[ "${1:-}" == "--install" ]]; then
  INSTALL=true
fi

if [[ -d "${LEROBOT_DIR}/.git" ]]; then
  echo "Using existing LeRobot checkout: ${LEROBOT_DIR}"
else
  mkdir -p "${ROOT_DIR}/src"
  git clone https://github.com/huggingface/lerobot.git "${LEROBOT_DIR}"
  git -C "${LEROBOT_DIR}" checkout "${LEROBOT_COMMIT}"
fi

CURRENT_COMMIT="$(git -C "${LEROBOT_DIR}" rev-parse HEAD)"
if [[ "${CURRENT_COMMIT}" != "${LEROBOT_COMMIT}" ]]; then
  echo "Expected LeRobot ${LEROBOT_COMMIT}, found ${CURRENT_COMMIT}" >&2
  echo "Use a clean checkout at the pinned commit before applying the project patch." >&2
  exit 1
fi

if git -C "${LEROBOT_DIR}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "SO-101 adapter patch is already applied"
else
  git -C "${LEROBOT_DIR}" apply --check "${PATCH_FILE}"
  git -C "${LEROBOT_DIR}" apply "${PATCH_FILE}"
  echo "Applied SO-101 adapter patch"
fi

if [[ "${INSTALL}" == "true" ]]; then
  python3 -m pip install -e "${LEROBOT_DIR}[feetech]" -e "${ROOT_DIR}"
fi

echo "LeRobot ${LEROBOT_COMMIT} is ready"
