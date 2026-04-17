#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJ_ROOT}"

PRETRAINED_CKPT="${PRETRAINED_CKPT:-outputs/cubev2/your_libero_run/checkpoints/last/pretrained_model}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_goal}"
TASK_ID="${TASK_ID:-}"
STATS_KEY="${STATS_KEY:-franka}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
INFER_HORIZON="${INFER_HORIZON:-}"
VIDEO_DIR="${VIDEO_DIR:-${PROJ_ROOT}/evaluation/Libero/output/${TASK_SUITE_NAME}}"

QWEN3_VL_PRETRAINED_PATH="${QWEN3_VL_PRETRAINED_PATH:-}"
QWEN3_VL_PROCESSOR_PATH="${QWEN3_VL_PROCESSOR_PATH:-}"
COSMOS_TOKENIZER_PATH_OR_NAME="${COSMOS_TOKENIZER_PATH_OR_NAME:-}"
DA3_MODEL_PATH_OR_NAME="${DA3_MODEL_PATH_OR_NAME:-}"
DA3_CODE_ROOT="${DA3_CODE_ROOT:-}"

ARGS=(
  --args.ckpt_path "${PRETRAINED_CKPT}"
  --args.task_suite_name "${TASK_SUITE_NAME}"
  --args.stats_key "${STATS_KEY}"
  --args.num_trials_per_task "${NUM_TRIALS_PER_TASK}"
  --args.video_dir "${VIDEO_DIR}"
  --args.disable_3d_teacher_for_eval true
)

if [[ -n "${TASK_ID}" ]]; then
  ARGS+=(--args.task_id "${TASK_ID}")
fi

if [[ -n "${INFER_HORIZON}" ]]; then
  ARGS+=(--args.infer_horizon "${INFER_HORIZON}")
fi

if [[ -n "${QWEN3_VL_PRETRAINED_PATH}" ]]; then
  ARGS+=(--args.qwen3_vl_pretrained_path "${QWEN3_VL_PRETRAINED_PATH}")
fi

if [[ -n "${QWEN3_VL_PROCESSOR_PATH}" ]]; then
  ARGS+=(--args.qwen3_vl_processor_path "${QWEN3_VL_PROCESSOR_PATH}")
fi

if [[ -n "${COSMOS_TOKENIZER_PATH_OR_NAME}" ]]; then
  ARGS+=(--args.cosmos_tokenizer_path_or_name "${COSMOS_TOKENIZER_PATH_OR_NAME}")
fi

if [[ -n "${DA3_MODEL_PATH_OR_NAME}" ]]; then
  ARGS+=(--args.da3_model_path_or_name "${DA3_MODEL_PATH_OR_NAME}")
fi

if [[ -n "${DA3_CODE_ROOT}" ]]; then
  ARGS+=(--args.da3_code_root "${DA3_CODE_ROOT}")
fi

python evaluation/Libero/inference.py "${ARGS[@]}"
