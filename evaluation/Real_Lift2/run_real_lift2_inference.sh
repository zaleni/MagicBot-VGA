#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REAL_LIFT2_RUNTIME_ROOT="${REAL_LIFT2_RUNTIME_ROOT:-/home/arx/ROS2_LIFT_Play/act}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export REAL_LIFT2_RUNTIME_ROOT
export PYTHONPATH="${REAL_LIFT2_RUNTIME_ROOT}:${PROJ_ROOT}/src:${PROJ_ROOT}:${PYTHONPATH:-}"

cd "${PROJ_ROOT}"

WS_URL="${WS_URL:-ws://127.0.0.1:8000}"
PROMPT="${PROMPT:-Clear the junk and items off the desktop.}"
FRAME_RATE="${FRAME_RATE:-60}"
IMAGE_HISTORY_INTERVAL="${IMAGE_HISTORY_INTERVAL:-15}"
MAX_PUBLISH_STEP="${MAX_PUBLISH_STEP:-10000}"
RECORD_MODE="${RECORD_MODE:-}"
STATE_DIM="${STATE_DIM:-}"
ACTION_DIM="${ACTION_DIM:-}"
PREFETCH_LEAD_STEPS="${PREFETCH_LEAD_STEPS:-}"
LOG_TIMING_EVERY="${LOG_TIMING_EVERY:-}"
SEED="${SEED:-}"
SAFE_STOP_BODY_HEIGHT="${SAFE_STOP_BODY_HEIGHT:-}"
SAFE_STOP_PUBLISH_STEPS="${SAFE_STOP_PUBLISH_STEPS:-}"

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

if [[ -n "${SEED}" ]]; then
  ARGS+=(--seed="${SEED}")
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

if [[ -n "${RECORD_MODE}" ]]; then
  ARGS+=(--record="${RECORD_MODE}")
fi

if [[ -n "${STATE_DIM}" ]]; then
  ARGS+=(--state_dim="${STATE_DIM}")
fi

if [[ -n "${ACTION_DIM}" ]]; then
  ARGS+=(--action_dim="${ACTION_DIM}")
fi

if [[ -n "${PREFETCH_LEAD_STEPS}" ]]; then
  ARGS+=(--prefetch_lead_steps="${PREFETCH_LEAD_STEPS}")
fi

if [[ -n "${LOG_TIMING_EVERY}" ]]; then
  ARGS+=(--log_timing_every="${LOG_TIMING_EVERY}")
fi

if [[ -n "${SAFE_STOP_BODY_HEIGHT}" ]]; then
  ARGS+=(--safe_stop_body_height="${SAFE_STOP_BODY_HEIGHT}")
fi

if [[ -n "${SAFE_STOP_PUBLISH_STEPS}" ]]; then
  ARGS+=(--safe_stop_publish_steps="${SAFE_STOP_PUBLISH_STEPS}")
fi

"${ARGS[@]}"
