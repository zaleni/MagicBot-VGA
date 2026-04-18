#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJ_ROOT}/src:${PROJ_ROOT}:${PYTHONPATH:-}"

cd "${PROJ_ROOT}"

WS_URL="${WS_URL:-ws://127.0.0.1:8000}"
PROMPT="${PROMPT:-Clear the junk and items off the desktop.}"
FRAME_RATE="${FRAME_RATE:-60}"
IMAGE_HISTORY_INTERVAL="${IMAGE_HISTORY_INTERVAL:-15}"
MAX_PUBLISH_STEP="${MAX_PUBLISH_STEP:-10000}"

ARGS=(
  python evaluation/Real_Lift2/real_lift2_inference.py
  --ws_url="${WS_URL}"
  --prompt="${PROMPT}"
  --frame_rate="${FRAME_RATE}"
  --image_history_interval="${IMAGE_HISTORY_INTERVAL}"
  --max_publish_step="${MAX_PUBLISH_STEP}"
)

if [[ -n "${DATA_CONFIG:-}" ]]; then
  ARGS+=(--data="${DATA_CONFIG}")
fi

if [[ "${USE_BASE:-false}" == "true" ]]; then
  ARGS+=(--use_base)
fi

if [[ -n "${FIXED_BODY_HEIGHT:-}" ]]; then
  ARGS+=(--fixed_body_height="${FIXED_BODY_HEIGHT}")
fi

if [[ -n "${GRIPPER_GATE:-}" ]]; then
  ARGS+=(--gripper_gate="${GRIPPER_GATE}")
fi

"${ARGS[@]}"
