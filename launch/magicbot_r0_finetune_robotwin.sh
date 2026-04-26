#!/usr/bin/env bash
set -euo pipefail

###############################################################################
################################# ENV config ##################################

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-6390}
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

PROC_PER_NODE="${PROC_PER_NODE:-8}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export WANDB_MODE=${WANDB_MODE:-offline}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false

###############################################################################
############################## TRAINING config ################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
echo "SCRIPT_DIR = ${SCRIPT_DIR}"
echo "PROJ_ROOT  = ${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJ_ROOT}"

POLICY="MagicBot_R0"
MAGICBOT_R0_VARIANT="${MAGICBOT_R0_VARIANT:-magicbot_r0}"
case "${MAGICBOT_R0_VARIANT}" in
  magicbot_r0|magicbot_r0_joint)
    ;;
  *)
    echo "Unsupported MagicBot_R0 variant=${MAGICBOT_R0_VARIANT}. Expected magicbot_r0 or magicbot_r0_joint."
    exit 1
    ;;
esac

WAN_MODEL_ID="${WAN_MODEL_ID:-Wan-AI/Wan2.2-TI2V-5B}"
WAN_TOKENIZER_MODEL_ID="${WAN_TOKENIZER_MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B}"
MAGICBOT_R0_REDIRECT_COMMON_FILES="${MAGICBOT_R0_REDIRECT_COMMON_FILES:-true}"
MAGICBOT_R0_ASSET_ROOT="${MAGICBOT_R0_ASSET_ROOT:-${PROJ_ROOT}/checkpoints/magicbot_r0}"
ACTION_DIT_PRETRAINED_PATH="${ACTION_DIT_PRETRAINED_PATH:-${MAGICBOT_R0_ASSET_ROOT}/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
FUTURE_3D_PRETRAINED_PATH="${FUTURE_3D_PRETRAINED_PATH:-${MAGICBOT_R0_ASSET_ROOT}/Future3DExpert_linear_interp_Wan22_alphascale_768hdim.pt}"
NATIVE_MAGICBOT_R0_CHECKPOINT_PATH="${NATIVE_MAGICBOT_R0_CHECKPOINT_PATH:-}"
LOAD_TEXT_ENCODER="${LOAD_TEXT_ENCODER:-false}"

ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/DATASET/RoboTwin-LeRobot-v30}"
ROBOTWIN_REQUIRE_THREE_CAMERAS="${ROBOTWIN_REQUIRE_THREE_CAMERAS:-true}"
if [[ "${LOAD_TEXT_ENCODER}" == "true" ]]; then
  TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR:-}"
else
  TEXT_EMBED_CACHE_DIR="${TEXT_EMBED_CACHE_DIR:-${PROJ_ROOT}/outputs/MagicBot_R0/text_embeds/robotwin}"
fi
USE_EXTERNAL_STATS="${USE_EXTERNAL_STATS:-false}"
NORMALIZATION_STATS_PATH="${NORMALIZATION_STATS_PATH:-}"
DATASET_EXTERNAL_STATS_PATH="${DATASET_EXTERNAL_STATS_PATH:-}"
DATASET_EXTERNAL_STATS_ROOT="${DATASET_EXTERNAL_STATS_ROOT:-/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/Foundation-Moodel/norm_stats}"
DATASET_EXTERNAL_STATS_ROBOT_TYPE="${DATASET_EXTERNAL_STATS_ROBOT_TYPE:-aloha}"
VALIDATE_DATASETS="${VALIDATE_DATASETS:-true}"
VIDEO_BACKEND="${VIDEO_BACKEND:-}"
USE_DIST_LOADING="${USE_DIST_LOADING:-false}"

ACTION_TYPE="${ACTION_TYPE:-abs}"
ACTION_DIM="${ACTION_DIM:-14}"
PROPRIO_DIM="${PROPRIO_DIM:-14}"
ACTION_HORIZON="${ACTION_HORIZON:-32}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
NUM_FRAMES="${NUM_FRAMES:-33}"
ACTION_VIDEO_FREQ_RATIO="${ACTION_VIDEO_FREQ_RATIO:-4}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-384}"
VIDEO_WIDTH="${VIDEO_WIDTH:-320}"
CONCAT_MULTI_CAMERA="${CONCAT_MULTI_CAMERA:-robotwin}"
STANDARDIZE_VIDEO_SIZE_BY_CAMERAS="${STANDARDIZE_VIDEO_SIZE_BY_CAMERAS:-true}"
NORM_DEFAULT_MODE="${NORM_DEFAULT_MODE:-z-score}"

BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
STEPS="${STEPS:-200000}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-}"
SAVE_FREQ="${SAVE_FREQ:-20000}"
LOG_FREQ="${LOG_FREQ:-25}"
NUM_WORKERS="${NUM_WORKERS:-12}"

LR="${LR:-1.0e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1.0e-2}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
DECAY_LR="${DECAY_LR:-5.0e-6}"
LAMBDA_VIDEO="${LAMBDA_VIDEO:-1.0}"
LAMBDA_ACTION="${LAMBDA_ACTION:-1.0}"
LAMBDA_3D="${LAMBDA_3D:-0.01}"
DA3_NUM_VIEWS="${DA3_NUM_VIEWS:-3}"
PROCESSOR_NUM_OUTPUT_CAMERAS="${PROCESSOR_NUM_OUTPUT_CAMERAS:-${DA3_NUM_VIEWS}}"
FUTURE_3D_TOKENS_PER_VIEW="${FUTURE_3D_TOKENS_PER_VIEW:-144}"
FUTURE_3D_VIEW_ATTENTION_LAYOUT="${FUTURE_3D_VIEW_ATTENTION_LAYOUT:-${CONCAT_MULTI_CAMERA}}"
FUTURE_3D_QUERY_MODE="${FUTURE_3D_QUERY_MODE:-query_token}"
FUTURE_3D_QUERY_NOISE_SCALE="${FUTURE_3D_QUERY_NOISE_SCALE:-0.5}"
FUTURE_3D_QUERY_NOISE_MIN_SIGMA="${FUTURE_3D_QUERY_NOISE_MIN_SIGMA:-0.0}"
FUTURE_3D_QUERY_NOISE_MAX_SIGMA="${FUTURE_3D_QUERY_NOISE_MAX_SIGMA:-0.5}"
FUTURE_3D_QUERY_SIGMA_SOURCE="${FUTURE_3D_QUERY_SIGMA_SOURCE:-constant}"
FUTURE_3D_SLOT_POS_SCALE="${FUTURE_3D_SLOT_POS_SCALE:-0.5}"
DA3_MODEL_PATH_OR_NAME="${DA3_MODEL_PATH_OR_NAME:-/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/DATASET/model/DA3-LARGE-1-1}"
DA3_VARIANT="${DA3_VARIANT:-large}"
DA3_CODE_ROOT="${DA3_CODE_ROOT:-}"
DA3_TEACHER_PROCESS_RES="${DA3_TEACHER_PROCESS_RES:-504}"
LOG_DA3_TEACHER_TIMING="${LOG_DA3_TEACHER_TIMING:-true}"
FUTURE_3D_TARGET_INDEX="${FUTURE_3D_TARGET_INDEX:--1}"
DTYPE="${DTYPE:-bfloat16}"
MAGICBOT_R0_CHECKPOINT_MIXED_ATTN="${MAGICBOT_R0_CHECKPOINT_MIXED_ATTN:-false}"

IMAGE_KEYS="${IMAGE_KEYS:-[\"cam_high\",\"cam_left_wrist\",\"cam_right_wrist\"]}"
IMAGE_RAW_SHAPES="${IMAGE_RAW_SHAPES:-[[3,480,640],[3,480,640],[3,480,640]]}"
IMAGE_SHAPES="${IMAGE_SHAPES:-[[3,240,320],[3,240,320],[3,240,320]]}"
ACTION_KEYS="${ACTION_KEYS:-[\"default\"]}"
ACTION_RAW_SHAPES="${ACTION_RAW_SHAPES:-[${ACTION_DIM}]}"
ACTION_SHAPES="${ACTION_SHAPES:-[${ACTION_DIM}]}"
STATE_KEYS="${STATE_KEYS:-[\"default\"]}"
STATE_RAW_SHAPES="${STATE_RAW_SHAPES:-[${PROPRIO_DIM}]}"
STATE_SHAPES="${STATE_SHAPES:-[${PROPRIO_DIM}]}"
PROCESSOR_DELTA_ACTION_DIM_MASK="${PROCESSOR_DELTA_ACTION_DIM_MASK:-{\"default\":[true,true,true,true,true,true,false,true,true,true,true,true,true,false]}}"

case "${ACTION_TYPE}" in
  abs|delta)
    ;;
  *)
    echo "Unsupported ACTION_TYPE=${ACTION_TYPE}. Expected abs or delta."
    exit 1
    ;;
esac

case "${USE_EXTERNAL_STATS}" in
  true|false)
    ;;
  *)
    echo "Unsupported USE_EXTERNAL_STATS=${USE_EXTERNAL_STATS}. Expected true or false."
    exit 1
    ;;
esac

if [[ "${USE_EXTERNAL_STATS}" == "true" ]]; then
  if [[ -z "${NORMALIZATION_STATS_PATH}" ]]; then
    if [[ -n "${DATASET_EXTERNAL_STATS_PATH}" ]]; then
      NORMALIZATION_STATS_PATH="${DATASET_EXTERNAL_STATS_PATH}"
    elif [[ -n "${DATASET_EXTERNAL_STATS_ROOT}" ]]; then
      NORMALIZATION_STATS_PATH="${DATASET_EXTERNAL_STATS_ROOT}/${DATASET_EXTERNAL_STATS_ROBOT_TYPE}/${ACTION_TYPE}/stats.json"
    fi
  fi

  if [[ -z "${NORMALIZATION_STATS_PATH}" ]]; then
    echo "USE_EXTERNAL_STATS=true but no normalization stats path could be resolved."
    echo "Set NORMALIZATION_STATS_PATH, DATASET_EXTERNAL_STATS_PATH, or DATASET_EXTERNAL_STATS_ROOT."
    exit 1
  fi
else
  NORMALIZATION_STATS_PATH=""
fi

case "${DTYPE}" in
  bfloat16)
    ACCELERATE_MIXED_PRECISION="bf16"
    ;;
  float16)
    ACCELERATE_MIXED_PRECISION="fp16"
    ;;
  float32)
    ACCELERATE_MIXED_PRECISION="no"
    ;;
  *)
    echo "Unsupported DTYPE=${DTYPE}. Expected one of: bfloat16, float16, float32"
    exit 1
    ;;
esac

discover_dataset_dirs() {
  local root="$1"
  if [[ -z "${root}" || ! -d "${root}" ]]; then
    return 0
  fi

  find -L "${root}" -path "*/meta/info.json" 2>/dev/null \
    | while read -r info_path; do
        ds_dir="$(dirname "$(dirname "${info_path}")")"
        if [[ ! -d "${ds_dir}/data" && ! -d "${ds_dir}/videos" ]]; then
          continue
        fi
        if [[ "${ROBOTWIN_REQUIRE_THREE_CAMERAS}" == "true" ]]; then
          python -c 'import json, sys
info = json.load(open(sys.argv[1], encoding="utf-8"))
image_keys = json.loads(sys.argv[2])
features = set(info.get("features", {}).keys())
required = {"observation.state", "action"}
required.update(f"observation.images.{key}" for key in image_keys)
raise SystemExit(0 if required.issubset(features) else 1)
' "${info_path}" "${IMAGE_KEYS}" || continue
        fi
        echo "${ds_dir}"
      done \
    | sort -u
}

mapfile -t DATASET_REPO_IDS < <(discover_dataset_dirs "${ROBOTWIN_ROOT}")

if [[ ${#DATASET_REPO_IDS[@]} -eq 0 ]]; then
  echo "No valid RoboTwin LeRobot datasets found under ROBOTWIN_ROOT=${ROBOTWIN_ROOT}"
  if [[ "${ROBOTWIN_REQUIRE_THREE_CAMERAS}" == "true" ]]; then
    echo "The default filter keeps only datasets with IMAGE_KEYS=${IMAGE_KEYS}."
    echo "Set ROBOTWIN_REQUIRE_THREE_CAMERAS=false if you intentionally want to include other layouts."
  fi
  exit 1
fi

if [[ ! -f "${ACTION_DIT_PRETRAINED_PATH}" ]]; then
    echo "Missing ActionDiT backbone: ${ACTION_DIT_PRETRAINED_PATH}"
    echo "Generate MagicBot_R0 expert backbones with:"
    echo "  python src/lerobot/scripts/magicbot_r0_preprocess_expert_backbones.py --expert both --action-output \"${ACTION_DIT_PRETRAINED_PATH}\" --future-3d-output \"${FUTURE_3D_PRETRAINED_PATH}\" --action-dim ${ACTION_DIM} --da3-num-views ${DA3_NUM_VIEWS} --future-3d-tokens-per-view ${FUTURE_3D_TOKENS_PER_VIEW} --device cuda --dtype bfloat16"
    exit 1
fi

if [[ ! -f "${FUTURE_3D_PRETRAINED_PATH}" ]]; then
    echo "Missing Future3DExpert backbone: ${FUTURE_3D_PRETRAINED_PATH}"
    echo "Generate MagicBot_R0 expert backbones with:"
    echo "  python src/lerobot/scripts/magicbot_r0_preprocess_expert_backbones.py --expert both --action-output \"${ACTION_DIT_PRETRAINED_PATH}\" --future-3d-output \"${FUTURE_3D_PRETRAINED_PATH}\" --action-dim ${ACTION_DIM} --da3-num-views ${DA3_NUM_VIEWS} --future-3d-tokens-per-view ${FUTURE_3D_TOKENS_PER_VIEW} --device cuda --dtype bfloat16"
    exit 1
fi

if [[ "${LOAD_TEXT_ENCODER}" != "true" && ! -d "${TEXT_EMBED_CACHE_DIR}" ]]; then
  echo "LOAD_TEXT_ENCODER=false but TEXT_EMBED_CACHE_DIR does not exist: ${TEXT_EMBED_CACHE_DIR}"
  echo "Generate a RoboTwin repo list first with:"
  echo "  python src/lerobot/scripts/magicbot_r0_discover_robotwin_repos.py --robotwin-root \"${ROBOTWIN_ROOT}\" --output-file outputs/MagicBot_R0/_repo_id_files/robotwin.txt --require-three-cameras ${ROBOTWIN_REQUIRE_THREE_CAMERAS}"
  echo "Then precompute text embeddings with:"
  echo "  python src/lerobot/scripts/magicbot_r0_precompute_text_embeds.py --repo-id-file outputs/MagicBot_R0/_repo_id_files/robotwin.txt --text-embedding-cache-dir \"${TEXT_EMBED_CACHE_DIR}\" --device cuda"
  echo "Or set LOAD_TEXT_ENCODER=true."
  exit 1
fi

if [[ -n "${NORMALIZATION_STATS_PATH}" && ! -f "${NORMALIZATION_STATS_PATH}" ]]; then
  echo "NORMALIZATION_STATS_PATH does not exist: ${NORMALIZATION_STATS_PATH}"
  exit 1
fi

if [[ "${VALIDATE_DATASETS}" == "true" ]]; then
  echo "Validating RoboTwin dataset mappings..."
  for ds_dir in "${DATASET_REPO_IDS[@]}"; do
    info_path="${ds_dir}/meta/info.json"
    python -c 'import json, sys
info = json.load(open(sys.argv[1], encoding="utf-8"))
image_keys = json.loads(sys.argv[3])
features = set(info.get("features", {}).keys())
required = {"observation.state", "action"}
required.update(f"observation.images.{key}" for key in image_keys)
print("{} -> robot_type={}, features={}".format(sys.argv[2], info.get("robot_type"), len(features)))
missing = sorted(required - features)
if missing:
    raise SystemExit(f"Missing required MagicBot_R0 RobotWin features for {sys.argv[2]}: {missing}")
' "${info_path}" "${ds_dir}" "${IMAGE_KEYS}"
  done
else
  echo "Skipping per-dataset validation (VALIDATE_DATASETS=${VALIDATE_DATASETS})."
fi

echo "Discovered ${#DATASET_REPO_IDS[@]} RoboTwin datasets under ${ROBOTWIN_ROOT}"
printf '  %s\n' "${DATASET_REPO_IDS[@]}"

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs/${POLICY}}"
BOOTSTRAP_TAG="${BOOTSTRAP_TAG:-magicbot_r0_backbone}"
JOB_NAME="${JOB_NAME:-${POLICY}-${MAGICBOT_R0_VARIANT}-robotwin-3d-${ACTION_TYPE}-${BOOTSTRAP_TAG}-finetune-$(date +'%Y_%m_%d_%H_%M_%S')}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${JOB_NAME}"
REPO_ID_FILE_DIR="${BASE_OUTPUT_DIR}/_repo_id_files"
mkdir -p "${REPO_ID_FILE_DIR}"
REPO_ID_FILE="${REPO_ID_FILE_DIR}/${JOB_NAME}.txt"
printf '%s\n' "${DATASET_REPO_IDS[@]}" > "${REPO_ID_FILE}"

echo "MAGICBOT_R0_VARIANT=${MAGICBOT_R0_VARIANT}"
echo "ACTION_TYPE=${ACTION_TYPE}"
echo "ACTION_DIM=${ACTION_DIM}, PROPRIO_DIM=${PROPRIO_DIM}"
echo "NORM_DEFAULT_MODE=${NORM_DEFAULT_MODE}"
echo "USE_EXTERNAL_STATS=${USE_EXTERNAL_STATS}"
echo "NORMALIZATION_STATS_PATH=${NORMALIZATION_STATS_PATH:-<auto-compute>}"
echo "ACTION_DIT_PRETRAINED_PATH=${ACTION_DIT_PRETRAINED_PATH}"
echo "FUTURE_3D_PRETRAINED_PATH=${FUTURE_3D_PRETRAINED_PATH}"
echo "NUM_FRAMES=${NUM_FRAMES}, ACTION_HORIZON=${ACTION_HORIZON}, ACTION_VIDEO_FREQ_RATIO=${ACTION_VIDEO_FREQ_RATIO}"
echo "VIDEO_SIZE=[${VIDEO_HEIGHT},${VIDEO_WIDTH}], CONCAT_MULTI_CAMERA=${CONCAT_MULTI_CAMERA}"
echo "STANDARDIZE_VIDEO_SIZE_BY_CAMERAS=${STANDARDIZE_VIDEO_SIZE_BY_CAMERAS}"
echo "Future3D: LAMBDA_3D=${LAMBDA_3D}, DA3_NUM_VIEWS=${DA3_NUM_VIEWS}, TOKENS_PER_VIEW=${FUTURE_3D_TOKENS_PER_VIEW}, VIEW_LAYOUT=${FUTURE_3D_VIEW_ATTENTION_LAYOUT}"
echo "Future3D query: MODE=${FUTURE_3D_QUERY_MODE}, NOISE_SCALE=${FUTURE_3D_QUERY_NOISE_SCALE}, SIGMA=[${FUTURE_3D_QUERY_NOISE_MIN_SIGMA},${FUTURE_3D_QUERY_NOISE_MAX_SIGMA}], SIGMA_SOURCE=${FUTURE_3D_QUERY_SIGMA_SOURCE}, SLOT_POS_SCALE=${FUTURE_3D_SLOT_POS_SCALE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

ARGS=(
    --multi_gpu
    --mixed_precision="${ACCELERATE_MIXED_PRECISION}"
    --num_processes="${NUM_PROCESSES}"
    --num_machines="${NODE_COUNT}"
    --machine_rank="${NODE_RANK}"
    --main_process_ip="${MASTER_ADDR}"
    --main_process_port="${MASTER_PORT}"
    src/lerobot/scripts/lerobot_train.py

    --output_dir="${OUTPUT_DIR}"
    --num_workers="${NUM_WORKERS}"
    --job_name="${JOB_NAME}"

    --policy.type=${POLICY}
    --policy.repo_id=lerobot_lab/${POLICY}
    --policy.push_to_hub=false
    --policy.variant="${MAGICBOT_R0_VARIANT}"
    --policy.model_id="${WAN_MODEL_ID}"
    --policy.tokenizer_model_id="${WAN_TOKENIZER_MODEL_ID}"
    --policy.redirect_common_files="${MAGICBOT_R0_REDIRECT_COMMON_FILES}"
    --policy.action_dit_pretrained_path="${ACTION_DIT_PRETRAINED_PATH}"
    --policy.future_3d_pretrained_path="${FUTURE_3D_PRETRAINED_PATH}"
    --policy.load_text_encoder="${LOAD_TEXT_ENCODER}"
    --policy.dtype="${DTYPE}"
    --policy.mot_checkpoint_mixed_attn="${MAGICBOT_R0_CHECKPOINT_MIXED_ATTN}"
    --policy.action_dim="${ACTION_DIM}"
    --policy.proprio_dim="${PROPRIO_DIM}"
    --policy.action_horizon="${ACTION_HORIZON}"
    --policy.n_action_steps="${N_ACTION_STEPS}"
    --policy.num_inference_steps="${NUM_INFERENCE_STEPS}"
    --policy.lambda_video="${LAMBDA_VIDEO}"
    --policy.lambda_action="${LAMBDA_ACTION}"
    --policy.lambda_3d="${LAMBDA_3D}"
    --policy.da3_num_views="${DA3_NUM_VIEWS}"
    --policy.future_3d_tokens_per_view="${FUTURE_3D_TOKENS_PER_VIEW}"
    --policy.future_3d_view_attention_layout="${FUTURE_3D_VIEW_ATTENTION_LAYOUT}"
    --policy.future_3d_query_mode="${FUTURE_3D_QUERY_MODE}"
    --policy.future_3d_query_noise_scale="${FUTURE_3D_QUERY_NOISE_SCALE}"
    --policy.future_3d_query_noise_min_sigma="${FUTURE_3D_QUERY_NOISE_MIN_SIGMA}"
    --policy.future_3d_query_noise_max_sigma="${FUTURE_3D_QUERY_NOISE_MAX_SIGMA}"
    --policy.future_3d_query_sigma_source="${FUTURE_3D_QUERY_SIGMA_SOURCE}"
    --policy.future_3d_slot_pos_scale="${FUTURE_3D_SLOT_POS_SCALE}"
    --policy.da3_model_path_or_name="${DA3_MODEL_PATH_OR_NAME}"
    --policy.da3_variant="${DA3_VARIANT}"
    --policy.da3_teacher_process_res="${DA3_TEACHER_PROCESS_RES}"
    --policy.log_da3_teacher_timing="${LOG_DA3_TEACHER_TIMING}"
    --policy.action_norm_default_mode="${NORM_DEFAULT_MODE}"
    --policy.optimizer_lr="${LR}"
    --policy.optimizer_weight_decay="${WEIGHT_DECAY}"
    --policy.train_num_epochs="${NUM_EPOCHS}"

    --dataset.type=${POLICY}
    --dataset.repo_id="multidata_from_file"
    --dataset.repo_id_file="${REPO_ID_FILE}"
    --dataset.action_mode="${ACTION_TYPE}"
    --dataset.use_external_stats="${USE_EXTERNAL_STATS}"
    --dataset.image_keys="${IMAGE_KEYS}"
    --dataset.image_raw_shapes="${IMAGE_RAW_SHAPES}"
    --dataset.image_shapes="${IMAGE_SHAPES}"
    --dataset.action_keys="${ACTION_KEYS}"
    --dataset.action_raw_shapes="${ACTION_RAW_SHAPES}"
    --dataset.action_shapes="${ACTION_SHAPES}"
    --dataset.state_keys="${STATE_KEYS}"
    --dataset.state_raw_shapes="${STATE_RAW_SHAPES}"
    --dataset.state_shapes="${STATE_SHAPES}"
    --dataset.num_frames="${NUM_FRAMES}"
    --dataset.action_video_freq_ratio="${ACTION_VIDEO_FREQ_RATIO}"
    --dataset.video_size="[${VIDEO_HEIGHT},${VIDEO_WIDTH}]"
    --dataset.standardize_video_size_by_cameras="${STANDARDIZE_VIDEO_SIZE_BY_CAMERAS}"
    --dataset.context_len=128
    --dataset.val_set_proportion=0.0
    --dataset.skip_padding_as_possible=false
    --dataset.concat_multi_camera="${CONCAT_MULTI_CAMERA}"
    --dataset.processor_norm_default_mode="${NORM_DEFAULT_MODE}"
    --dataset.processor_num_output_cameras="${PROCESSOR_NUM_OUTPUT_CAMERAS}"
    --dataset.processor_action_output_dim="${ACTION_DIM}"
    --dataset.processor_proprio_output_dim="${PROPRIO_DIM}"
    --dataset.processor_delta_action_dim_mask="${PROCESSOR_DELTA_ACTION_DIM_MASK}"
    --dataset.future_3d_target_index="${FUTURE_3D_TARGET_INDEX}"

    --seed=42
    --batch_size="${BATCH_SIZE}"
    --gradient_accumulation_steps="${GRAD_ACCUM_STEPS}"
    --steps="${STEPS}"
    --save_freq="${SAVE_FREQ}"
    --log_freq="${LOG_FREQ}"

    --wandb.enable=true
    --wandb.project=MagicBot_R0
    --wandb.mode=${WANDB_MODE}
)

if [[ -n "${TEXT_EMBED_CACHE_DIR}" ]]; then
    ARGS+=(--dataset.text_embedding_cache_dir="${TEXT_EMBED_CACHE_DIR}")
fi

if [[ -n "${WARMUP_STEPS}" ]]; then
    ARGS+=(--policy.scheduler_warmup_steps="${WARMUP_STEPS}")
fi

if [[ -n "${DECAY_LR}" ]]; then
    ARGS+=(--policy.scheduler_decay_lr="${DECAY_LR}")
fi

if [[ -n "${TRAIN_MAX_STEPS}" ]]; then
    ARGS+=(--policy.train_max_steps="${TRAIN_MAX_STEPS}")
fi

if [[ -n "${NORMALIZATION_STATS_PATH}" ]]; then
    ARGS+=(--dataset.normalization_stats_path="${NORMALIZATION_STATS_PATH}")
fi

if [[ -n "${NATIVE_MAGICBOT_R0_CHECKPOINT_PATH}" ]]; then
    ARGS+=(--policy.native_checkpoint_path="${NATIVE_MAGICBOT_R0_CHECKPOINT_PATH}")
fi

if [[ -n "${VIDEO_BACKEND}" ]]; then
    ARGS+=(--dataset.video_backend="${VIDEO_BACKEND}")
fi

if [[ -n "${DA3_CODE_ROOT}" ]]; then
    ARGS+=(--policy.da3_code_root="${DA3_CODE_ROOT}")
fi

if [[ "${USE_DIST_LOADING}" == "true" ]]; then
    echo "USE_DIST_LOADING=true is not supported for MagicBot_R0 in this framework."
    echo "Leave USE_DIST_LOADING=false so Accelerate can shard the dataloader correctly."
    exit 1
fi

accelerate launch "${ARGS[@]}"
