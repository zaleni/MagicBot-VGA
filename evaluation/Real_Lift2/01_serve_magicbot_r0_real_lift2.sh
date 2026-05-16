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
PORT="${PORT:-8102}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-bfloat16}"
INFER_HORIZON="${INFER_HORIZON:-24}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-Sweep the trash into the dustpan using a broom.}"
STATS_KEY="${STATS_KEY:-real_lift2}"
STATS_PATH="${STATS_PATH:-}"
ACTION_MODE="${ACTION_MODE:-}"

RTC_ENABLED="${RTC_ENABLED:-false}"
DISABLE_3D_TEACHER_FOR_EVAL="${DISABLE_3D_TEACHER_FOR_EVAL:-true}"

WAN_MODEL_ID="${WAN_MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B}"
WAN_TOKENIZER_MODEL_ID="${WAN_TOKENIZER_MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B}"
MAGICBOT_R0_ASSET_ROOT="${MAGICBOT_R0_ASSET_ROOT:-${PROJ_ROOT}/checkpoints/magicbot_r0}"
ACTION_DIT_PRETRAINED_PATH="${ACTION_DIT_PRETRAINED_PATH:-${MAGICBOT_R0_ASSET_ROOT}/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
FUTURE_3D_PRETRAINED_PATH="${FUTURE_3D_PRETRAINED_PATH:-${MAGICBOT_R0_ASSET_ROOT}/Future3DExpert_linear_interp_Wan22_alphascale_768hdim.pt}"
MAGICBOT_R0_LOAD_TEXT_ENCODER="${MAGICBOT_R0_LOAD_TEXT_ENCODER:-true}"
MAGICBOT_R0_REDIRECT_COMMON_FILES="${MAGICBOT_R0_REDIRECT_COMMON_FILES:-true}"
MAGICBOT_R0_SKIP_DIT_LOAD_FROM_PRETRAIN="${MAGICBOT_R0_SKIP_DIT_LOAD_FROM_PRETRAIN:-true}"
MAGICBOT_R0_STATE_KEY="${MAGICBOT_R0_STATE_KEY:-default}"
MAGICBOT_R0_VIDEO_HEIGHT="${MAGICBOT_R0_VIDEO_HEIGHT:-384}"
MAGICBOT_R0_VIDEO_WIDTH="${MAGICBOT_R0_VIDEO_WIDTH:-320}"
MAGICBOT_R0_STANDARDIZE_VIDEO_SIZE_BY_CAMERAS="${MAGICBOT_R0_STANDARDIZE_VIDEO_SIZE_BY_CAMERAS:-true}"
MAGICBOT_R0_CONCAT_MULTI_CAMERA="${MAGICBOT_R0_CONCAT_MULTI_CAMERA:-robotwin}"
MAGICBOT_R0_TEXT_EMBED_CACHE_DIR="${MAGICBOT_R0_TEXT_EMBED_CACHE_DIR:-${TEXT_EMBED_CACHE_DIR:-}}"
MAGICBOT_R0_CONTEXT_LEN="${MAGICBOT_R0_CONTEXT_LEN:-}"

if [[ -z "${CHECKPOINT_DIR}" ]]; then
  echo "Please set CHECKPOINT_DIR to a MagicBot_R0 checkpoint step dir or pretrained_model dir."
  exit 1
fi

case "${RTC_ENABLED,,}" in
  true|1|yes|y|on)
    echo "MagicBot_R0 Real_Lift2 RTC serving is not implemented yet. Use sync mode with RTC_ENABLED=false."
    exit 1
    ;;
  false|0|no|n|off)
    ;;
  *)
    echo "Invalid RTC_ENABLED=${RTC_ENABLED}"
    echo "Expected one of: true/false, 1/0, yes/no, on/off"
    exit 1
    ;;
esac

ARGS=(
  python evaluation/Real_Lift2/model_server.py
  --ckpt_path="${CHECKPOINT_DIR}"
  --host="${HOST}"
  --port="${PORT}"
  --device="${DEVICE}"
  --dtype="${DTYPE}"
  --infer_horizon="${INFER_HORIZON}"
  --default_prompt="${DEFAULT_PROMPT}"
  --stats_key="${STATS_KEY}"
  --magicbot_r0_model_id="${WAN_MODEL_ID}"
  --magicbot_r0_tokenizer_model_id="${WAN_TOKENIZER_MODEL_ID}"
  --magicbot_r0_action_dit_pretrained_path="${ACTION_DIT_PRETRAINED_PATH}"
  --magicbot_r0_future_3d_pretrained_path="${FUTURE_3D_PRETRAINED_PATH}"
  --magicbot_r0_state_key="${MAGICBOT_R0_STATE_KEY}"
  --magicbot_r0_video_height="${MAGICBOT_R0_VIDEO_HEIGHT}"
  --magicbot_r0_video_width="${MAGICBOT_R0_VIDEO_WIDTH}"
  --magicbot_r0_concat_multi_camera="${MAGICBOT_R0_CONCAT_MULTI_CAMERA}"
)

if [[ -n "${NUM_INFERENCE_STEPS}" ]]; then
  ARGS+=(--num_inference_steps="${NUM_INFERENCE_STEPS}")
fi

if [[ -n "${STATS_PATH}" ]]; then
  ARGS+=(--stats_path="${STATS_PATH}")
fi

if [[ -n "${ACTION_MODE}" ]]; then
  ARGS+=(--action_mode="${ACTION_MODE}")
fi

if [[ -n "${MAGICBOT_R0_TEXT_EMBED_CACHE_DIR}" ]]; then
  ARGS+=(--magicbot_r0_text_embedding_cache_dir="${MAGICBOT_R0_TEXT_EMBED_CACHE_DIR}")
fi

if [[ -n "${MAGICBOT_R0_CONTEXT_LEN}" ]]; then
  ARGS+=(--magicbot_r0_context_len="${MAGICBOT_R0_CONTEXT_LEN}")
fi

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

case "${MAGICBOT_R0_LOAD_TEXT_ENCODER,,}" in
  true|1|yes|y|on)
    ARGS+=(--magicbot_r0_load_text_encoder)
    ;;
  false|0|no|n|off)
    ARGS+=(--no-magicbot_r0_load_text_encoder)
    ;;
  *)
    echo "Invalid MAGICBOT_R0_LOAD_TEXT_ENCODER=${MAGICBOT_R0_LOAD_TEXT_ENCODER}"
    exit 1
    ;;
esac

case "${MAGICBOT_R0_REDIRECT_COMMON_FILES,,}" in
  true|1|yes|y|on)
    ARGS+=(--magicbot_r0_redirect_common_files)
    ;;
  false|0|no|n|off)
    ARGS+=(--no-magicbot_r0_redirect_common_files)
    ;;
  *)
    echo "Invalid MAGICBOT_R0_REDIRECT_COMMON_FILES=${MAGICBOT_R0_REDIRECT_COMMON_FILES}"
    exit 1
    ;;
esac

case "${MAGICBOT_R0_SKIP_DIT_LOAD_FROM_PRETRAIN,,}" in
  true|1|yes|y|on)
    ARGS+=(--magicbot_r0_skip_dit_load_from_pretrain)
    ;;
  false|0|no|n|off)
    ARGS+=(--no-magicbot_r0_skip_dit_load_from_pretrain)
    ;;
  *)
    echo "Invalid MAGICBOT_R0_SKIP_DIT_LOAD_FROM_PRETRAIN=${MAGICBOT_R0_SKIP_DIT_LOAD_FROM_PRETRAIN}"
    exit 1
    ;;
esac

case "${MAGICBOT_R0_STANDARDIZE_VIDEO_SIZE_BY_CAMERAS,,}" in
  true|1|yes|y|on)
    ARGS+=(--magicbot_r0_standardize_video_size_by_cameras)
    ;;
  false|0|no|n|off)
    ARGS+=(--no-magicbot_r0_standardize_video_size_by_cameras)
    ;;
  *)
    echo "Invalid MAGICBOT_R0_STANDARDIZE_VIDEO_SIZE_BY_CAMERAS=${MAGICBOT_R0_STANDARDIZE_VIDEO_SIZE_BY_CAMERAS}"
    exit 1
    ;;
esac

if [[ -n "${LOAD_DEVICE:-}" ]]; then
  ARGS+=(--load_device="${LOAD_DEVICE}")
fi

if [[ -n "${DA3_MODEL_PATH_OR_NAME:-}" ]]; then
  ARGS+=(--da3_model_path_or_name="${DA3_MODEL_PATH_OR_NAME}")
fi

if [[ -n "${DA3_CODE_ROOT:-}" ]]; then
  ARGS+=(--da3_code_root="${DA3_CODE_ROOT}")
fi

"${ARGS[@]}"
