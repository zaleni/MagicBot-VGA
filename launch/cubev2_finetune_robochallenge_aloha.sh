#!/usr/bin/env bash
set -euo pipefail

###############################################################################
################################# ENV config ##################################

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-6379}"
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

PROC_PER_NODE="${PROC_PER_NODE:-8}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

###############################################################################
############################## TRAINING config ################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
echo "SCRIPT_DIR = ${SCRIPT_DIR}"
echo "PROJ_ROOT  = ${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}/src:${PYTHONPATH:-}"

cd "${PROJ_ROOT}"

POLICY="cubev2"
POLICY_INIT_PATH="${POLICY_INIT_PATH:-${PRETRAINED_PATH:-/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/Foundation-Moodel/outputs/cubev2/cubev2-multidata-delta-pretrain-2026_04_07_07_42_16/checkpoints/300000/pretrained_model}}"
QWEN3_VL_PRETRAINED_PATH="${QWEN3_VL_PRETRAINED_PATH:-/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/DATASET/model/Qwen3-VL-2B-Instruct}"
QWEN3_VL_PROCESSOR_PATH="${QWEN3_VL_PROCESSOR_PATH:-${QWEN3_VL_PRETRAINED_PATH}}"
COSMOS_TOKENIZER_PATH_OR_NAME="${COSMOS_TOKENIZER_PATH_OR_NAME:-/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/DATASET/model/Cosmos-Tokenizer-CI8x8}"
DA3_MODEL_PATH_OR_NAME="${DA3_MODEL_PATH_OR_NAME:-/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/DATASET/model/DA3-LARGE-1-1}"
DA3_VARIANT="${DA3_VARIANT:-auto}"
DA3_ALIGNMENT_MODE="${DA3_ALIGNMENT_MODE:-query_decoder}"
DA3_CODE_ROOT="${DA3_CODE_ROOT:-}"

ROBOCHALLENGE_ROOT="${ROBOCHALLENGE_ROOT:-${DATASET_ROOT:-/inspire/qb-ilm/project/embodied-basic-model/zhangjianing-253108140206/DATASET/Robochallengev2_lerobotv3/Robochallenge_aloha_v3}}"
DATASET_DIR="${DATASET_DIR:-}"
DATASET_DIRS_FILE="${DATASET_DIRS_FILE:-}"

DEFAULT_ROBOCHALLENGE_ALOHA_REGULAR_TASKS="put_the_books_back stamp_positioning wipe_the_blackboard scoop_with_a_small_spoon"
DEFAULT_ROBOCHALLENGE_ALOHA_EXTRA_TASKS="wrap_with_a_soft_cloth paint_jam pack_the_items put_the_pencil_case_into_the_schoolbag pack_the_toothbrush_holder lint_roller_remove_dirt"
# Keep empty strings intentional: set ROBOCHALLENGE_ALOHA_EXTRA_TASKS="" for regular-only training.
ROBOCHALLENGE_ALOHA_REGULAR_TASKS="${ROBOCHALLENGE_ALOHA_REGULAR_TASKS-${DEFAULT_ROBOCHALLENGE_ALOHA_REGULAR_TASKS}}"
ROBOCHALLENGE_ALOHA_EXTRA_TASKS="${ROBOCHALLENGE_ALOHA_EXTRA_TASKS-${DEFAULT_ROBOCHALLENGE_ALOHA_EXTRA_TASKS}}"
ROBOCHALLENGE_ALOHA_TASKS="${ROBOCHALLENGE_ALOHA_TASKS-}"
if [[ -n "${ROBOCHALLENGE_ALOHA_TASKS}" ]]; then
  if [[ "${ROBOCHALLENGE_ALOHA_TASKS}" == "all" ]]; then
    ROBOCHALLENGE_ALOHA_SELECTED_TASKS="${DEFAULT_ROBOCHALLENGE_ALOHA_REGULAR_TASKS} ${DEFAULT_ROBOCHALLENGE_ALOHA_EXTRA_TASKS}"
  else
    ROBOCHALLENGE_ALOHA_SELECTED_TASKS="${ROBOCHALLENGE_ALOHA_TASKS}"
  fi
else
  ROBOCHALLENGE_ALOHA_SELECTED_TASKS="${ROBOCHALLENGE_ALOHA_REGULAR_TASKS} ${ROBOCHALLENGE_ALOHA_EXTRA_TASKS}"
fi
ROBOCHALLENGE_ALOHA_TASK_SET="${ROBOCHALLENGE_ALOHA_TASK_SET-}"
if [[ -z "${ROBOCHALLENGE_ALOHA_TASK_SET}" ]]; then
  ROBOCHALLENGE_ALOHA_TASK_SET="$(
    python - "${ROBOCHALLENGE_ALOHA_SELECTED_TASKS}" "${DEFAULT_ROBOCHALLENGE_ALOHA_REGULAR_TASKS}" "${DEFAULT_ROBOCHALLENGE_ALOHA_EXTRA_TASKS}" <<'PY'
import hashlib
import sys

selected = tuple(name for name in sys.argv[1].split() if name)
regular = tuple(name for name in sys.argv[2].split() if name)
extra = tuple(name for name in sys.argv[3].split() if name)
if not selected:
    raise SystemExit("No RoboChallenge ALOHA tasks selected.")
if selected == regular + extra:
    print("all")
elif selected == regular:
    print("regular_only")
else:
    print("tasks_" + hashlib.sha1(",".join(selected).encode("utf-8")).hexdigest()[:8])
PY
  )"
fi

ACTION_TYPE="${ACTION_TYPE:-delta}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-${CHUNK_SIZE}}"
ENABLE_3D_QUERIES="${ENABLE_3D_QUERIES:-true}"
NUM_3D_QUERY_TOKENS="${NUM_3D_QUERY_TOKENS:-432}"
CUBEV2_ATTENTION_MASK_MODE="${CUBEV2_ATTENTION_MASK_MODE:-default}"
IMAGE_DELTA_INDICES="${IMAGE_DELTA_INDICES:-[0,0,15]}"
LAMBDA_GEN="${LAMBDA_GEN:-0.01}"
LAMBDA_3D="${LAMBDA_3D:-0.01}"

USE_EXTERNAL_STATS="${USE_EXTERNAL_STATS:-true}"
NORM_STATS_ROOT="${NORM_STATS_ROOT:-outputs_robochallenge/norm_stats}"
if [[ "${ROBOCHALLENGE_ALOHA_TASK_SET}" == "all" ]]; then
  DEFAULT_DATASET_EXTERNAL_STATS_PATH="${NORM_STATS_ROOT}/robochallenge_aloha/${ACTION_TYPE}/stats.json"
else
  DEFAULT_DATASET_EXTERNAL_STATS_PATH="${NORM_STATS_ROOT}/robochallenge_aloha/${ROBOCHALLENGE_ALOHA_TASK_SET}/${ACTION_TYPE}/stats.json"
fi
DATASET_EXTERNAL_STATS_PATH="${DATASET_EXTERNAL_STATS_PATH:-${DEFAULT_DATASET_EXTERNAL_STATS_PATH}}"

ENABLE_IMAGE_AUG="${ENABLE_IMAGE_AUG:-false}"
IMAGE_AUG_PRESET="${IMAGE_AUG_PRESET:-pi05}"

BATCH_SIZE="${BATCH_SIZE:-12}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
STEPS="${STEPS:-300000}"
SAVE_FREQ="${SAVE_FREQ:-20000}"
LOG_FREQ="${LOG_FREQ:-100}"
NUM_WORKERS="${NUM_WORKERS:-12}"
DIST_LOADING="${DIST_LOADING:-false}"
WEIGHT_RULES_PATH="${WEIGHT_RULES_PATH:-configs/weight_rules_robochallenge_aloha.yaml}"
DTYPE="${DTYPE:-bfloat16}"

case "${DTYPE}" in
  bfloat16)
    ACCELERATE_MIXED_PRECISION="bf16"
    ;;
  float32)
    ACCELERATE_MIXED_PRECISION="no"
    ;;
  *)
    echo "Unsupported DTYPE=${DTYPE}. Expected bfloat16 or float32."
    exit 1
    ;;
esac

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs_robochallenge/${POLICY}}"
if [[ "${ROBOCHALLENGE_ALOHA_TASK_SET}" == "all" ]]; then
  JOB_TASK_SUFFIX=""
else
  JOB_TASK_SUFFIX="-${ROBOCHALLENGE_ALOHA_TASK_SET}"
fi
JOB_NAME="${JOB_NAME:-${POLICY}-robochallenge_aloha${JOB_TASK_SUFFIX}-${ACTION_TYPE}-chunk${CHUNK_SIZE}-finetune-$(date +'%Y_%m_%d_%H_%M_%S')}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${JOB_NAME}"
REPO_ID_FILE="${REPO_ID_FILE:-${BASE_OUTPUT_DIR}/_repo_id_files/${JOB_NAME}.txt}"

if [[ -z "${POLICY_INIT_PATH}" ]]; then
  echo "Please set POLICY_INIT_PATH or PRETRAINED_PATH to the ALOHA/CubeV2 checkpoint you want to finetune."
  exit 1
fi

if [[ "${ACTION_TYPE}" != "delta" && "${ACTION_TYPE}" != "abs" ]]; then
  echo "ACTION_TYPE must be abs or delta, got ${ACTION_TYPE}"
  exit 1
fi

if [[ "${DIST_LOADING}" != "true" && "${DIST_LOADING}" != "false" ]]; then
  echo "DIST_LOADING must be true or false, got ${DIST_LOADING}"
  exit 1
fi

if [[ -n "${WEIGHT_RULES_PATH}" && ! -f "${WEIGHT_RULES_PATH}" ]]; then
  echo "WEIGHT_RULES_PATH does not exist: ${WEIGHT_RULES_PATH}"
  exit 1
fi

is_robochallenge_aloha_info() {
  local info_path="$1"
  python - "${info_path}" <<'PY'
import json
import sys
from pathlib import Path

info = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
robot_type = str(info.get("robot_type", ""))
features = set((info.get("features") or {}).keys())
required_base = {
    "observation.state",
    "action",
}
camera_schemas = (
    {
        "observation.images.head",
        "observation.images.left",
        "observation.images.right",
    },
    {
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    },
)
ok = (
    robot_type in {"aloha", "ALOHA"}
    and required_base.issubset(features)
    and any(camera_schema.issubset(features) for camera_schema in camera_schemas)
)
raise SystemExit(0 if ok else 1)
PY
}

discover_dataset_dirs() {
  local root="$1"
  if [[ -z "${root}" || ! -d "${root}" ]]; then
    return 0
  fi

  while IFS= read -r -d '' info_path; do
    if is_robochallenge_aloha_info "${info_path}"; then
      dirname "$(dirname "${info_path}")"
    fi
  done < <(find -L "${root}" -path "*/meta/info.json" -print0 2>/dev/null) | sort -u
}

read_dataset_dirs_file() {
  local path="$1"
  local line=""
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%$'\r'}"
    if [[ -n "${line//[[:space:]]/}" ]]; then
      echo "${line}"
    fi
  done < "${path}"
}

filter_dataset_dirs_by_task_names() {
  local task_names="$1"
  shift
  python - "${task_names}" "$@" <<'PY'
import sys
from pathlib import Path

task_names = {name for name in sys.argv[1].split() if name}
if not task_names:
    raise SystemExit("No RoboChallenge ALOHA tasks selected.")

for dataset_dir in sys.argv[2:]:
    path = Path(dataset_dir)
    if path.name in task_names or task_names.intersection(path.parts):
        print(dataset_dir)
PY
}

declare -a DATASET_REPO_IDS=()
if [[ -n "${DATASET_DIRS_FILE}" ]]; then
  if [[ ! -f "${DATASET_DIRS_FILE}" ]]; then
    echo "DATASET_DIRS_FILE does not exist: ${DATASET_DIRS_FILE}"
    exit 1
  fi
  mapfile -t DATASET_REPO_IDS < <(read_dataset_dirs_file "${DATASET_DIRS_FILE}")
elif [[ -n "${DATASET_DIR}" ]]; then
  DATASET_REPO_IDS=("${DATASET_DIR}")
else
  if [[ -z "${ROBOCHALLENGE_ROOT}" ]]; then
    echo "Set ROBOCHALLENGE_ROOT to RoboChallenge_aloha_v3 or the parent Robochallengev2_lerobotv3 directory."
    exit 1
  fi
  mapfile -t DATASET_REPO_IDS < <(discover_dataset_dirs "${ROBOCHALLENGE_ROOT}")
fi

mapfile -t DATASET_REPO_IDS < <(filter_dataset_dirs_by_task_names "${ROBOCHALLENGE_ALOHA_SELECTED_TASKS}" "${DATASET_REPO_IDS[@]}")

if [[ ${#DATASET_REPO_IDS[@]} -eq 0 ]]; then
  echo "No RoboChallenge ALOHA LeRobot datasets found for selected tasks: ${ROBOCHALLENGE_ALOHA_SELECTED_TASKS}"
  exit 1
fi

for ds_dir in "${DATASET_REPO_IDS[@]}"; do
  if [[ ! -f "${ds_dir}/meta/info.json" ]]; then
    echo "meta/info.json not found under dataset dir: ${ds_dir}"
    exit 1
  fi
  resolved="$(
    python - "${ds_dir}/meta/info.json" <<'PY'
import json
import sys
from pathlib import Path
from lerobot.transforms.constants import (
    get_feature_mapping,
    get_image_mapping,
    get_mask_mapping,
    infer_embodiment_variant,
)

info = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
robot_type = info["robot_type"]
features = info.get("features", {})
resolved = infer_embodiment_variant(robot_type, features)
get_feature_mapping(robot_type, features)
get_image_mapping(robot_type, features)
get_mask_mapping(robot_type, features)
print(resolved)
PY
  )"
  if [[ "${resolved}" != "ALOHA" && "${resolved}" != "ALOHA_STARVLA" && "${resolved}" != "aloha" ]]; then
    echo "Expected RoboChallenge ALOHA schema, got resolved_robot_type=${resolved} for ${ds_dir}"
    exit 1
  fi
done

if [[ "${USE_EXTERNAL_STATS}" == "true" && ! -f "${DATASET_EXTERNAL_STATS_PATH}" ]]; then
  echo "Missing external stats: ${DATASET_EXTERNAL_STATS_PATH}"
  echo "Compute them first with: ACTION_TYPE=${ACTION_TYPE} CHUNK_SIZE=${CHUNK_SIZE} ROBOCHALLENGE_ROOT=... bash launch/compute_norm_stats_robochallenge_aloha.sh"
  exit 1
fi

mkdir -p "$(dirname "${REPO_ID_FILE}")"
printf '%s\n' "${DATASET_REPO_IDS[@]}" > "${REPO_ID_FILE}"

echo "POLICY_INIT_PATH=${POLICY_INIT_PATH}"
echo "DATASET_COUNT=${#DATASET_REPO_IDS[@]}"
echo "REPO_ID_FILE=${REPO_ID_FILE}"
echo "ROBOCHALLENGE_ALOHA_TASK_SET=${ROBOCHALLENGE_ALOHA_TASK_SET}"
echo "ROBOCHALLENGE_ALOHA_SELECTED_TASKS=${ROBOCHALLENGE_ALOHA_SELECTED_TASKS}"
echo "ACTION_TYPE=${ACTION_TYPE}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "N_ACTION_STEPS=${N_ACTION_STEPS}"
echo "CUBEV2_ATTENTION_MASK_MODE=${CUBEV2_ATTENTION_MASK_MODE}"
echo "IMAGE_DELTA_INDICES=${IMAGE_DELTA_INDICES}"
echo "BATCH_SIZE(per_device)=${BATCH_SIZE}"
echo "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
echo "DIST_LOADING=${DIST_LOADING}"
echo "WEIGHT_RULES_PATH=${WEIGHT_RULES_PATH:-<disabled>}"
echo "DTYPE=${DTYPE}"
echo "USE_EXTERNAL_STATS=${USE_EXTERNAL_STATS}"
echo "DATASET_EXTERNAL_STATS_PATH=${DATASET_EXTERNAL_STATS_PATH}"
echo "ENABLE_IMAGE_AUG=${ENABLE_IMAGE_AUG}"
echo "IMAGE_AUG_PRESET=${IMAGE_AUG_PRESET}"
echo "JOB_NAME=${JOB_NAME}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
printf '  - %s\n' "${DATASET_REPO_IDS[@]}"

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

    --policy.type="${POLICY}"
    --policy.repo_id="lerobot_lab/${POLICY}"
    --policy.pretrained_path="${POLICY_INIT_PATH}"
    --policy.qwen3_vl_pretrained_path="${QWEN3_VL_PRETRAINED_PATH}"
    --policy.cosmos_tokenizer_path_or_name="${COSMOS_TOKENIZER_PATH_OR_NAME}"
    --policy.push_to_hub=false
    --policy.gradient_checkpointing=false
    --policy.dtype="${DTYPE}"
    --policy.optimizer_lr=5.0e-5
    --policy.scheduler_warmup_steps=3000
    --policy.scheduler_decay_steps="${STEPS}"
    --policy.scheduler_decay_lr=5.0e-6
    --policy.freeze_vision_encoder=false
    --policy.train_expert_only=false
    --policy.train_vlm_only=false
    --policy.qwen3_vl_variant=qwen3_vl_28l
    --policy.action_expert_variant=qwen3_28l
    --policy.chunk_size="${CHUNK_SIZE}"
    --policy.n_action_steps="${N_ACTION_STEPS}"
    --policy.attention_mask_mode="${CUBEV2_ATTENTION_MASK_MODE}"
    --policy.image_delta_indices="${IMAGE_DELTA_INDICES}"
    --policy.enable_3d_queries="${ENABLE_3D_QUERIES}"
    --policy.num_3d_query_tokens="${NUM_3D_QUERY_TOKENS}"
    --policy.lambda_gen="${LAMBDA_GEN}"
    --policy.lambda_3d="${LAMBDA_3D}"
    --policy.da3_model_path_or_name="${DA3_MODEL_PATH_OR_NAME}"
    --policy.da3_variant="${DA3_VARIANT}"
    --policy.da3_alignment_mode="${DA3_ALIGNMENT_MODE}"
    --policy.log_da3_teacher_timing=true

    --dataset.type="${POLICY}"
    --dataset.repo_id="multidata_from_file"
    --dataset.repo_id_file="${REPO_ID_FILE}"
    --dataset.qwen3_vl_processor_path="${QWEN3_VL_PROCESSOR_PATH}"
    --dataset.action_mode="${ACTION_TYPE}"
    --dataset.use_external_stats="${USE_EXTERNAL_STATS}"
    --dataset.dist_loading="${DIST_LOADING}"

    --seed=42
    --batch_size="${BATCH_SIZE}"
    --gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}"
    --steps="${STEPS}"
    --save_freq="${SAVE_FREQ}"
    --log_freq="${LOG_FREQ}"

    --wandb.enable=true
    --wandb.project=CUBEv2
    --wandb.mode="${WANDB_MODE}"
)

if [[ -n "${DA3_CODE_ROOT}" ]]; then
    ARGS+=(--policy.da3_code_root="${DA3_CODE_ROOT}")
fi

if [[ "${USE_EXTERNAL_STATS}" == "true" ]]; then
    ARGS+=(--dataset.external_stats_path="${DATASET_EXTERNAL_STATS_PATH}")
fi

if [[ -n "${WEIGHT_RULES_PATH}" ]]; then
    ARGS+=(--dataset.weight_rules_path="${WEIGHT_RULES_PATH}")
fi

if [[ "${ENABLE_IMAGE_AUG}" == "true" ]]; then
    ARGS+=(
        --dataset.image_transforms.enable=true
        --dataset.image_transforms.preset="${IMAGE_AUG_PRESET}"
    )
fi

accelerate launch "${ARGS[@]}"
