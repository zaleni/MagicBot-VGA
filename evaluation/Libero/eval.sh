#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

LIBERO_HOME="${LIBERO_HOME:-}"
if [[ -n "${LIBERO_HOME}" ]]; then
  export PYTHONPATH="${LIBERO_HOME}:${PYTHONPATH:-}"
elif [[ -d "${PROJ_ROOT}/LIBERO" ]]; then
  export PYTHONPATH="${PROJ_ROOT}/LIBERO:${PYTHONPATH:-}"
elif [[ -d "${PROJ_ROOT}/third_party/LIBERO" ]]; then
  export PYTHONPATH="${PROJ_ROOT}/third_party/LIBERO:${PYTHONPATH:-}"
fi

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
DISABLE_3D_TEACHER_FOR_EVAL="${DISABLE_3D_TEACHER_FOR_EVAL:-true}"

ARGS=(
  --args.ckpt_path "${PRETRAINED_CKPT}"
  --args.task_suite_name "${TASK_SUITE_NAME}"
  --args.stats_key "${STATS_KEY}"
  --args.num_trials_per_task "${NUM_TRIALS_PER_TASK}"
  --args.video_dir "${VIDEO_DIR}"
)

case "${DISABLE_3D_TEACHER_FOR_EVAL,,}" in
  true|1|yes|y|on)
    ARGS+=(--args.disable_3d_teacher_for_eval)
    ;;
  false|0|no|n|off)
    ARGS+=(--no-args.disable_3d_teacher_for_eval)
    ;;
  *)
    echo "Invalid DISABLE_3D_TEACHER_FOR_EVAL=${DISABLE_3D_TEACHER_FOR_EVAL}"
    echo "Expected one of: true/false, 1/0, yes/no, on/off"
    exit 1
    ;;
esac

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
