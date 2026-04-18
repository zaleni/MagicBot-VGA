#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJ_ROOT}/src:${PROJ_ROOT}:${PYTHONPATH:-}"

cd "${PROJ_ROOT}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-bfloat16}"
INFER_HORIZON="${INFER_HORIZON:-30}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-Clear the junk and items off the desktop.}"
DISABLE_3D_TEACHER_FOR_EVAL="${DISABLE_3D_TEACHER_FOR_EVAL:-true}"

if [[ -z "${CHECKPOINT_DIR}" ]]; then
  echo "Please set CHECKPOINT_DIR to a MagicBot checkpoint step dir or pretrained_model dir."
  exit 1
fi

ARGS=(
  python evaluation/Real_Lift2/serve_magicbot_policy.py
  --ckpt_path="${CHECKPOINT_DIR}"
  --host="${HOST}"
  --port="${PORT}"
  --device="${DEVICE}"
  --dtype="${DTYPE}"
  --infer_horizon="${INFER_HORIZON}"
  --resize_size="${RESIZE_SIZE}"
  --default_prompt="${DEFAULT_PROMPT}"
)

case "${DISABLE_3D_TEACHER_FOR_EVAL,,}" in
  true|1|yes|y|on)
    ARGS+=(--disable_3d_teacher_for_eval)
    ;;
  false|0|no|n|off)
    ARGS+=(--no-disable_3d_teacher_for_eval)
    ;;
  *)
    echo "Invalid DISABLE_3D_TEACHER_FOR_EVAL=${DISABLE_3D_TEACHER_FOR_EVAL}"
    echo "Expected one of: true/false, 1/0, yes/no, on/off"
    exit 1
    ;;
esac

if [[ -n "${STATS_KEY:-}" ]]; then
  ARGS+=(--stats_key="${STATS_KEY}")
fi

if [[ -n "${STATS_PATH:-}" ]]; then
  ARGS+=(--stats_path="${STATS_PATH}")
fi

if [[ -n "${ACTION_MODE:-}" ]]; then
  ARGS+=(--action_mode="${ACTION_MODE}")
fi

if [[ -n "${QWEN3_VL_PROCESSOR_PATH:-}" ]]; then
  ARGS+=(--qwen3_vl_processor_path="${QWEN3_VL_PROCESSOR_PATH}")
fi

if [[ -n "${QWEN3_VL_PRETRAINED_PATH:-}" ]]; then
  ARGS+=(--qwen3_vl_pretrained_path="${QWEN3_VL_PRETRAINED_PATH}")
fi

if [[ -n "${COSMOS_TOKENIZER_PATH_OR_NAME:-}" ]]; then
  ARGS+=(--cosmos_tokenizer_path_or_name="${COSMOS_TOKENIZER_PATH_OR_NAME}")
fi

if [[ -n "${DA3_MODEL_PATH_OR_NAME:-}" ]]; then
  ARGS+=(--da3_model_path_or_name="${DA3_MODEL_PATH_OR_NAME}")
fi

if [[ -n "${DA3_CODE_ROOT:-}" ]]; then
  ARGS+=(--da3_code_root="${DA3_CODE_ROOT}")
fi

"${ARGS[@]}"
