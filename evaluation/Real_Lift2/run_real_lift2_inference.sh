#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REAL_LIFT2_RUNTIME_ROOT="${REAL_LIFT2_RUNTIME_ROOT:-/home/arx/ROS2_LIFT_Play/act}"
ENTRY_SCRIPT="${PROJ_ROOT}/evaluation/Real_Lift2/real_lift2_inference.py"
RUNTIME_MODULE="${PROJ_ROOT}/evaluation/Real_Lift2/real_lift2_runtime.py"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export REAL_LIFT2_RUNTIME_ROOT
export PYTHONPATH="${REAL_LIFT2_RUNTIME_ROOT}:${PROJ_ROOT}/src:${PROJ_ROOT}:${PYTHONPATH:-}"

cd "${PROJ_ROOT}"

for required_file in "${ENTRY_SCRIPT}" "${RUNTIME_MODULE}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required deployment file is missing:"
    echo "  ${required_file}"
    echo "Please sync the full evaluation/Real_Lift2 directory to the target machine."
    exit 1
  fi
done

WS_URL="${WS_URL:-ws://127.0.0.1:8000}"
PROMPT="${PROMPT:-Clear the junk and items off the desktop.}"
FRAME_RATE="${FRAME_RATE:-60}"
IMAGE_HISTORY_INTERVAL="${IMAGE_HISTORY_INTERVAL:-15}"
SEND_IMAGE_HEIGHT="${SEND_IMAGE_HEIGHT:-}"
SEND_IMAGE_WIDTH="${SEND_IMAGE_WIDTH:-}"
MAX_PUBLISH_STEP="${MAX_PUBLISH_STEP:-10000}"
RECORD_MODE="${RECORD_MODE:-}"
STATE_DIM="${STATE_DIM:-}"
ACTION_DIM="${ACTION_DIM:-}"
INFERENCE_MODE="${INFERENCE_MODE:-}"
PREFETCH_LEAD_STEPS="${PREFETCH_LEAD_STEPS:-}"
LOG_TIMING_EVERY="${LOG_TIMING_EVERY:-}"
SEED="${SEED:-}"
SAFE_STOP_BODY_HEIGHT="${SAFE_STOP_BODY_HEIGHT:-}"
SAFE_STOP_PUBLISH_STEPS="${SAFE_STOP_PUBLISH_STEPS:-}"
SAFE_STOP_HOME_ARMS="${SAFE_STOP_HOME_ARMS:-}"
SAFE_STOP_HOME_PUBLISH_STEPS="${SAFE_STOP_HOME_PUBLISH_STEPS:-}"

ARGS=(
  python "${ENTRY_SCRIPT}"
  --ws_url="${WS_URL}"
  --prompt="${PROMPT}"
  --frame_rate="${FRAME_RATE}"
  --image_history_interval="${IMAGE_HISTORY_INTERVAL}"
  --max_publish_step="${MAX_PUBLISH_STEP}"
)

if [[ -n "${SEND_IMAGE_HEIGHT}" ]]; then
  ARGS+=(--send_image_height="${SEND_IMAGE_HEIGHT}")
fi

if [[ -n "${SEND_IMAGE_WIDTH}" ]]; then
  ARGS+=(--send_image_width="${SEND_IMAGE_WIDTH}")
fi

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

if [[ -n "${INFERENCE_MODE}" ]]; then
  ARGS+=(--inference_mode="${INFERENCE_MODE}")
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

if [[ "${SAFE_STOP_HOME_ARMS:-false}" == "true" ]]; then
  ARGS+=(--safe_stop_home_arms)
fi

if [[ -n "${SAFE_STOP_HOME_PUBLISH_STEPS}" ]]; then
  ARGS+=(--safe_stop_home_publish_steps="${SAFE_STOP_HOME_PUBLISH_STEPS}")
fi

"${ARGS[@]}"
