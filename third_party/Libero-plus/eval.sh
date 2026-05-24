#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJ_ROOT}"

PRETRAINED_CKPT="${PRETRAINED_CKPT:-}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-}"
TASK_SUITE_NAMES="${TASK_SUITE_NAMES:-libero_spatial,libero_object,libero_goal,libero_10}"
TASK_ID="${TASK_ID:-}"

SEED="${SEED:-7}"
STATS_KEY="${STATS_KEY:-franka}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-1}"
INFER_HORIZON="${INFER_HORIZON:-}"
WS_URL="${WS_URL:-}"
CLASSIFICATION_PATH="${CLASSIFICATION_PATH:-/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/LIBERO-plus/libero/libero/benchmark/task_classification.json}"
MODE_TAG="local"
if [[ -n "${WS_URL}" ]]; then
  MODE_TAG="split_ws"
fi
VIDEO_ROOT="${VIDEO_ROOT:-${SCRIPT_DIR}/output}"

QWEN3_VL_PRETRAINED_PATH="${QWEN3_VL_PRETRAINED_PATH:-}"
QWEN3_VL_PROCESSOR_PATH="${QWEN3_VL_PROCESSOR_PATH:-}"
COSMOS_TOKENIZER_PATH_OR_NAME="${COSMOS_TOKENIZER_PATH_OR_NAME:-}"
DA3_MODEL_PATH_OR_NAME="${DA3_MODEL_PATH_OR_NAME:-}"
DA3_CODE_ROOT="${DA3_CODE_ROOT:-}"
DISABLE_3D_TEACHER_FOR_EVAL="${DISABLE_3D_TEACHER_FOR_EVAL:-true}"

ARGS=(
  --args.seed "${SEED}"
  --args.stats_key "${STATS_KEY}"
  --args.num_trials_per_task "${NUM_TRIALS_PER_TASK}"
  --args.video_dir "${VIDEO_ROOT}"
)

if [[ -n "${TASK_SUITE_NAME}" ]]; then
  ARGS+=(--args.task_suite_name "${TASK_SUITE_NAME}")
else
  ARGS+=(--args.task_suite_names "${TASK_SUITE_NAMES}")
fi

if [[ -n "${PRETRAINED_CKPT}" ]]; then
  ARGS+=(--args.ckpt_path "${PRETRAINED_CKPT}")
fi

if [[ -n "${WS_URL}" ]]; then
  ARGS+=(--args.ws_url "${WS_URL}")
fi

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

if [[ -n "${CLASSIFICATION_PATH}" ]]; then
  ARGS+=(--args.classification_path "${CLASSIFICATION_PATH}")
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

if [[ -z "${PRETRAINED_CKPT}" && -z "${WS_URL}" ]]; then
  echo "Please set either PRETRAINED_CKPT for local evaluation or WS_URL for split websocket policy serving."
  exit 1
fi

echo "LIBERO_HOME      : ${LIBERO_HOME:-unset}"
echo "Task suite name  : ${TASK_SUITE_NAME:-<multi-suite>}"
echo "Task suite names : ${TASK_SUITE_NAMES}"
echo "Task id          : ${TASK_ID:-all}"
echo "Eval mode        : ${MODE_TAG}"
echo "Output root      : ${VIDEO_ROOT}"

python "${SCRIPT_DIR}/inference.py" "${ARGS[@]}"
