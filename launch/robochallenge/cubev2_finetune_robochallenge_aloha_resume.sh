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
############################### RESUME config #################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
echo "SCRIPT_DIR = ${SCRIPT_DIR}"
echo "PROJ_ROOT  = ${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}/src:${PYTHONPATH:-}"

cd "${PROJ_ROOT}"

POLICY="cubev2"

# Point one of these at the copied run/checkpoint on the new server.
RESUME_RUN_DIR="${RESUME_RUN_DIR:-${RUN_DIR:-}}"
RESUME_STEP="${RESUME_STEP:-}"
RESUME_CHECKPOINT_DIR="${RESUME_CHECKPOINT_DIR:-${CHECKPOINT_DIR:-}}"
RESUME_CONFIG_PATH="${RESUME_CONFIG_PATH:-}"
RESUME_OUTPUT_DIR="${RESUME_OUTPUT_DIR:-${OUTPUT_DIR:-}}"

if [[ -n "${RESUME_RUN_DIR}" && -z "${RESUME_CHECKPOINT_DIR}" ]]; then
  if [[ -n "${RESUME_STEP}" ]]; then
    RESUME_CHECKPOINT_DIR="${RESUME_RUN_DIR%/}/checkpoints/${RESUME_STEP}"
  else
    RESUME_CHECKPOINT_DIR="${RESUME_RUN_DIR%/}/checkpoints/last"
  fi
fi

if [[ -n "${RESUME_CHECKPOINT_DIR}" && -z "${RESUME_CONFIG_PATH}" ]]; then
  RESUME_CONFIG_PATH="${RESUME_CHECKPOINT_DIR%/}/pretrained_model/train_config.json"
fi

if [[ -z "${RESUME_CONFIG_PATH}" ]]; then
  echo "Set RESUME_RUN_DIR, RESUME_CHECKPOINT_DIR, or RESUME_CONFIG_PATH."
  echo "Prefer a numeric checkpoint dir, e.g. RESUME_CHECKPOINT_DIR=/path/to/run/checkpoints/200000"
  exit 1
fi

if [[ ! -f "${RESUME_CONFIG_PATH}" ]]; then
  echo "Resume config not found: ${RESUME_CONFIG_PATH}"
  exit 1
fi

if [[ -z "${RESUME_CHECKPOINT_DIR}" ]]; then
  RESUME_CHECKPOINT_DIR="$(dirname "$(dirname "${RESUME_CONFIG_PATH}")")"
fi

if [[ ! -d "${RESUME_CHECKPOINT_DIR}/training_state" ]]; then
  echo "Missing training_state under checkpoint: ${RESUME_CHECKPOINT_DIR}"
  echo "For true resume, copy the whole checkpoints/<step> directory, not only pretrained_model."
  exit 1
fi

if [[ -z "${RESUME_RUN_DIR}" ]]; then
  RESUME_RUN_DIR="$(dirname "$(dirname "${RESUME_CHECKPOINT_DIR}")")"
fi

CONFIG_JOB_NAME="$(
  python -c 'import json,sys; cfg=json.load(open(sys.argv[1], encoding="utf-8")); print(cfg.get("job_name") or "")' \
    "${RESUME_CONFIG_PATH}"
)"
CONFIG_OUTPUT_DIR="$(
  python -c 'import json,sys; cfg=json.load(open(sys.argv[1], encoding="utf-8")); print(cfg.get("output_dir") or "")' \
    "${RESUME_CONFIG_PATH}"
)"
CONFIG_ACTION_TYPE="$(
  python -c 'import json,sys; cfg=json.load(open(sys.argv[1], encoding="utf-8")); print((cfg.get("dataset") or {}).get("action_mode") or "")' \
    "${RESUME_CONFIG_PATH}"
)"
CONFIG_DTYPE="$(
  python -c 'import json,sys; cfg=json.load(open(sys.argv[1], encoding="utf-8")); print((cfg.get("policy") or {}).get("dtype") or "")' \
    "${RESUME_CONFIG_PATH}"
)"
CONFIG_BATCH_SIZE="$(
  python -c 'import json,sys; cfg=json.load(open(sys.argv[1], encoding="utf-8")); print(cfg.get("batch_size") or "")' \
    "${RESUME_CONFIG_PATH}"
)"
CONFIG_GRADIENT_ACCUMULATION_STEPS="$(
  python -c 'import json,sys; cfg=json.load(open(sys.argv[1], encoding="utf-8")); print(cfg.get("gradient_accumulation_steps") or "")' \
    "${RESUME_CONFIG_PATH}"
)"

JOB_NAME="${JOB_NAME:-${CONFIG_JOB_NAME}}"
RESUME_OUTPUT_DIR="${RESUME_OUTPUT_DIR:-${RESUME_RUN_DIR}}"
ACTION_TYPE="${ACTION_TYPE:-${CONFIG_ACTION_TYPE:-delta}}"
DTYPE="${DTYPE:-${CONFIG_DTYPE:-bfloat16}}"

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

###############################################################################
########################## Cross-server path overrides #########################

# Defaults follow launch/cubev2/cubev2_finetune_real_lift2_abs.sh on the jiangjiahao server.
QWEN3_VL_PRETRAINED_PATH="${QWEN3_VL_PRETRAINED_PATH:-/home/jiangjiahao/data/model/Qwen3-VL-2B-Instruct}"
QWEN3_VL_PROCESSOR_PATH="${QWEN3_VL_PROCESSOR_PATH:-${QWEN3_VL_PRETRAINED_PATH}}"
COSMOS_TOKENIZER_PATH_OR_NAME="${COSMOS_TOKENIZER_PATH_OR_NAME:-/home/jiangjiahao/data/model/Cosmos-Tokenizer-CI8x8}"
DA3_MODEL_PATH_OR_NAME="${DA3_MODEL_PATH_OR_NAME:-/home/jiangjiahao/data/model/DA3-LARGE-1.1}"
DA3_VARIANT="${DA3_VARIANT:-auto}"
DA3_ALIGNMENT_MODE="${DA3_ALIGNMENT_MODE:-query_decoder}"
DA3_CODE_ROOT="${DA3_CODE_ROOT:-}"

OVERRIDE_MODEL_PATHS="${OVERRIDE_MODEL_PATHS:-true}"
OVERRIDE_DATA_PATHS="${OVERRIDE_DATA_PATHS:-true}"

ROBOCHALLENGE_ROOT="${ROBOCHALLENGE_ROOT:-${DATASET_ROOT:-/home/jiangjiahao/data/Robochallengev2_lerobotv3/Robochallenge_aloha_v3}}"
DATASET_DIR="${DATASET_DIR:-}"
DATASET_DIRS_FILE="${DATASET_DIRS_FILE:-}"

DEFAULT_ROBOCHALLENGE_ALOHA_REGULAR_TASKS="put_the_books_back stamp_positioning wipe_the_blackboard scoop_with_a_small_spoon"
DEFAULT_ROBOCHALLENGE_ALOHA_EXTRA_TASKS="wrap_with_a_soft_cloth paint_jam pack_the_items put_the_pencil_case_into_the_schoolbag pack_the_toothbrush_holder lint_roller_remove_dirt"
ROBOCHALLENGE_ALOHA_REGULAR_TASKS="${ROBOCHALLENGE_ALOHA_REGULAR_TASKS-${DEFAULT_ROBOCHALLENGE_ALOHA_REGULAR_TASKS}}"
# Match the previous regular-only run by default:
#   ROBOCHALLENGE_ALOHA_EXTRA_TASKS="" bash launch/robochallenge/cubev2_finetune_robochallenge_aloha.sh
# Set ROBOCHALLENGE_ALOHA_TASKS=all, or pass a non-empty
# ROBOCHALLENGE_ALOHA_EXTRA_TASKS, only when intentionally changing the task set.
ROBOCHALLENGE_ALOHA_EXTRA_TASKS="${ROBOCHALLENGE_ALOHA_EXTRA_TASKS-}"
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

USE_EXTERNAL_STATS="${USE_EXTERNAL_STATS:-true}"
NORM_STATS_ROOT="${NORM_STATS_ROOT:-outputs_robochallenge/norm_stats}"
if [[ "${ROBOCHALLENGE_ALOHA_TASK_SET}" == "all" ]]; then
  DEFAULT_DATASET_EXTERNAL_STATS_PATH="${NORM_STATS_ROOT}/robochallenge_aloha/${ACTION_TYPE}/stats.json"
else
  DEFAULT_DATASET_EXTERNAL_STATS_PATH="${NORM_STATS_ROOT}/robochallenge_aloha/${ROBOCHALLENGE_ALOHA_TASK_SET}/${ACTION_TYPE}/stats.json"
fi
DATASET_EXTERNAL_STATS_PATH="${DATASET_EXTERNAL_STATS_PATH:-${DEFAULT_DATASET_EXTERNAL_STATS_PATH}}"

REPO_ID_FILE="${REPO_ID_FILE:-${RESUME_OUTPUT_DIR%/}/_repo_id_files/robochallenge_aloha_resume_${ROBOCHALLENGE_ALOHA_TASK_SET}_${ACTION_TYPE}.txt}"
WEIGHT_RULES_PATH="${WEIGHT_RULES_PATH:-configs/weight_rules_robochallenge_aloha.yaml}"

NUM_WORKERS="${NUM_WORKERS:-12}"
DIST_LOADING="${DIST_LOADING:-}"
STEPS="${STEPS:-}"
SAVE_FREQ="${SAVE_FREQ:-}"
LOG_FREQ="${LOG_FREQ:-}"
PRESERVE_EFFECTIVE_BATCH="${PRESERVE_EFFECTIVE_BATCH:-true}"
# Original ALOHA finetune used per-device batch 12 with grad accumulation 1.
# Use a smaller micro-batch by default, then increase grad accumulation so each
# optimizer step still sees the same effective batch when possible.
RESUME_MICRO_BATCH_SIZE="${RESUME_MICRO_BATCH_SIZE:-6}"
BATCH_SIZE="${BATCH_SIZE:-${RESUME_MICRO_BATCH_SIZE}}"
if [[ ! "${BATCH_SIZE}" =~ ^[0-9]+$ || "${BATCH_SIZE}" -le 0 ]]; then
  echo "BATCH_SIZE must be a positive integer, got ${BATCH_SIZE}"
  exit 1
fi
if [[ -z "${GRADIENT_ACCUMULATION_STEPS:-}" ]]; then
  if [[ "${PRESERVE_EFFECTIVE_BATCH}" == "true" && "${CONFIG_BATCH_SIZE}" =~ ^[0-9]+$ && "${CONFIG_GRADIENT_ACCUMULATION_STEPS}" =~ ^[0-9]+$ ]]; then
    CONFIG_EFFECTIVE_BATCH_PER_PROCESS=$((CONFIG_BATCH_SIZE * CONFIG_GRADIENT_ACCUMULATION_STEPS))
    if (( CONFIG_EFFECTIVE_BATCH_PER_PROCESS % BATCH_SIZE != 0 )); then
      echo "Cannot preserve effective batch exactly with BATCH_SIZE=${BATCH_SIZE}."
      echo "Checkpoint per-process effective batch is ${CONFIG_EFFECTIVE_BATCH_PER_PROCESS}."
      echo "Use BATCH_SIZE=6 for exact preservation, or set PRESERVE_EFFECTIVE_BATCH=false."
      exit 1
    fi
    GRADIENT_ACCUMULATION_STEPS=$((CONFIG_EFFECTIVE_BATCH_PER_PROCESS / BATCH_SIZE))
  else
    GRADIENT_ACCUMULATION_STEPS="${CONFIG_GRADIENT_ACCUMULATION_STEPS:-1}"
  fi
fi
if [[ ! "${GRADIENT_ACCUMULATION_STEPS}" =~ ^[0-9]+$ || "${GRADIENT_ACCUMULATION_STEPS}" -le 0 ]]; then
  echo "GRADIENT_ACCUMULATION_STEPS must be a positive integer, got ${GRADIENT_ACCUMULATION_STEPS}"
  exit 1
fi
ENABLE_IMAGE_AUG="${ENABLE_IMAGE_AUG:-}"
IMAGE_AUG_PRESET="${IMAGE_AUG_PRESET:-pi05}"

if [[ "${OVERRIDE_MODEL_PATHS}" != "true" && "${OVERRIDE_MODEL_PATHS}" != "false" ]]; then
  echo "OVERRIDE_MODEL_PATHS must be true or false, got ${OVERRIDE_MODEL_PATHS}"
  exit 1
fi

if [[ "${OVERRIDE_DATA_PATHS}" != "true" && "${OVERRIDE_DATA_PATHS}" != "false" ]]; then
  echo "OVERRIDE_DATA_PATHS must be true or false, got ${OVERRIDE_DATA_PATHS}"
  exit 1
fi

if [[ "${PRESERVE_EFFECTIVE_BATCH}" != "true" && "${PRESERVE_EFFECTIVE_BATCH}" != "false" ]]; then
  echo "PRESERVE_EFFECTIVE_BATCH must be true or false, got ${PRESERVE_EFFECTIVE_BATCH}"
  exit 1
fi

if [[ "${ACTION_TYPE}" != "delta" && "${ACTION_TYPE}" != "abs" ]]; then
  echo "ACTION_TYPE must be abs or delta, got ${ACTION_TYPE}"
  exit 1
fi

if [[ -n "${DIST_LOADING}" && "${DIST_LOADING}" != "true" && "${DIST_LOADING}" != "false" ]]; then
  echo "DIST_LOADING must be true, false, or empty, got ${DIST_LOADING}"
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
if [[ "${OVERRIDE_DATA_PATHS}" == "true" ]]; then
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
    echo "Compute them first with: ACTION_TYPE=${ACTION_TYPE} ROBOCHALLENGE_ROOT=... bash launch/compute_norm/compute_norm_stats_robochallenge_aloha.sh"
    exit 1
  fi

  mkdir -p "$(dirname "${REPO_ID_FILE}")"
  printf '%s\n' "${DATASET_REPO_IDS[@]}" > "${REPO_ID_FILE}"
fi

RESUME_STEP_VALUE="<unknown>"
if [[ -f "${RESUME_CHECKPOINT_DIR}/training_state/training_step.json" ]]; then
  RESUME_STEP_VALUE="$(
    python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("step", "<unknown>"))' \
      "${RESUME_CHECKPOINT_DIR}/training_state/training_step.json"
  )"
fi

EFFECTIVE_BATCH_SIZE="<unknown>"
if [[ "${BATCH_SIZE}" =~ ^[0-9]+$ && "${GRADIENT_ACCUMULATION_STEPS}" =~ ^[0-9]+$ ]]; then
  EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * NUM_PROCESSES * GRADIENT_ACCUMULATION_STEPS))
fi

CONFIG_EFFECTIVE_BATCH_SIZE="<unknown>"
if [[ "${CONFIG_BATCH_SIZE}" =~ ^[0-9]+$ && "${CONFIG_GRADIENT_ACCUMULATION_STEPS}" =~ ^[0-9]+$ ]]; then
  CONFIG_EFFECTIVE_BATCH_SIZE=$((CONFIG_BATCH_SIZE * NUM_PROCESSES * CONFIG_GRADIENT_ACCUMULATION_STEPS))
fi

echo "RESUME=true"
echo "RESUME_RUN_DIR=${RESUME_RUN_DIR}"
echo "RESUME_CHECKPOINT_DIR=${RESUME_CHECKPOINT_DIR}"
echo "RESUME_CONFIG_PATH=${RESUME_CONFIG_PATH}"
echo "RESUME_STEP=${RESUME_STEP_VALUE}"
echo "CONFIG_OUTPUT_DIR=${CONFIG_OUTPUT_DIR}"
echo "RESUME_OUTPUT_DIR=${RESUME_OUTPUT_DIR}"
echo "JOB_NAME=${JOB_NAME}"
echo "ACTION_TYPE=${ACTION_TYPE}"
echo "DTYPE=${DTYPE}"
echo "CONFIG_BATCH_SIZE(per_device)=${CONFIG_BATCH_SIZE:-<unset>}"
echo "CONFIG_GRADIENT_ACCUMULATION_STEPS=${CONFIG_GRADIENT_ACCUMULATION_STEPS:-<unset>}"
echo "CONFIG_EFFECTIVE_BATCH_SIZE=${CONFIG_EFFECTIVE_BATCH_SIZE}"
echo "PRESERVE_EFFECTIVE_BATCH=${PRESERVE_EFFECTIVE_BATCH}"
echo "RESUME_MICRO_BATCH_SIZE=${RESUME_MICRO_BATCH_SIZE}"
echo "BATCH_SIZE(per_device)=${BATCH_SIZE}"
echo "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
echo "EFFECTIVE_BATCH_SIZE=${EFFECTIVE_BATCH_SIZE}"
echo "OVERRIDE_MODEL_PATHS=${OVERRIDE_MODEL_PATHS}"
echo "OVERRIDE_DATA_PATHS=${OVERRIDE_DATA_PATHS}"
if [[ "${OVERRIDE_MODEL_PATHS}" == "true" ]]; then
  echo "QWEN3_VL_PRETRAINED_PATH=${QWEN3_VL_PRETRAINED_PATH}"
  echo "QWEN3_VL_PROCESSOR_PATH=${QWEN3_VL_PROCESSOR_PATH}"
  echo "COSMOS_TOKENIZER_PATH_OR_NAME=${COSMOS_TOKENIZER_PATH_OR_NAME}"
  echo "DA3_MODEL_PATH_OR_NAME=${DA3_MODEL_PATH_OR_NAME}"
fi
if [[ "${OVERRIDE_DATA_PATHS}" == "true" ]]; then
  echo "DATASET_COUNT=${#DATASET_REPO_IDS[@]}"
  echo "ROBOCHALLENGE_ALOHA_EXTRA_TASKS=${ROBOCHALLENGE_ALOHA_EXTRA_TASKS:-<empty>}"
  echo "ROBOCHALLENGE_ALOHA_TASK_SET=${ROBOCHALLENGE_ALOHA_TASK_SET}"
  echo "ROBOCHALLENGE_ALOHA_SELECTED_TASKS=${ROBOCHALLENGE_ALOHA_SELECTED_TASKS}"
  echo "REPO_ID_FILE=${REPO_ID_FILE}"
  echo "USE_EXTERNAL_STATS=${USE_EXTERNAL_STATS}"
  echo "DATASET_EXTERNAL_STATS_PATH=${DATASET_EXTERNAL_STATS_PATH}"
  echo "WEIGHT_RULES_PATH=${WEIGHT_RULES_PATH:-<disabled>}"
  printf '  - %s\n' "${DATASET_REPO_IDS[@]}"
fi

ARGS=(
    --multi_gpu
    --mixed_precision="${ACCELERATE_MIXED_PRECISION}"
    --num_processes="${NUM_PROCESSES}"
    --num_machines="${NODE_COUNT}"
    --machine_rank="${NODE_RANK}"
    --main_process_ip="${MASTER_ADDR}"
    --main_process_port="${MASTER_PORT}"
    src/lerobot/scripts/lerobot_train.py

    --resume=true
    --config_path="${RESUME_CONFIG_PATH}"
    --output_dir="${RESUME_OUTPUT_DIR}"
    --num_workers="${NUM_WORKERS}"
)

if [[ -n "${JOB_NAME}" ]]; then
    ARGS+=(--job_name="${JOB_NAME}")
fi

if [[ "${OVERRIDE_MODEL_PATHS}" == "true" ]]; then
    ARGS+=(
        --policy.qwen3_vl_pretrained_path="${QWEN3_VL_PRETRAINED_PATH}"
        --policy.cosmos_tokenizer_path_or_name="${COSMOS_TOKENIZER_PATH_OR_NAME}"
        --policy.da3_model_path_or_name="${DA3_MODEL_PATH_OR_NAME}"
        --policy.da3_variant="${DA3_VARIANT}"
        --policy.da3_alignment_mode="${DA3_ALIGNMENT_MODE}"
        --dataset.qwen3_vl_processor_path="${QWEN3_VL_PROCESSOR_PATH}"
    )

    if [[ -n "${DA3_CODE_ROOT}" ]]; then
        ARGS+=(--policy.da3_code_root="${DA3_CODE_ROOT}")
    fi
fi

if [[ "${OVERRIDE_DATA_PATHS}" == "true" ]]; then
    ARGS+=(
        --dataset.repo_id="multidata_from_file"
        --dataset.repo_id_file="${REPO_ID_FILE}"
        --dataset.action_mode="${ACTION_TYPE}"
        --dataset.use_external_stats="${USE_EXTERNAL_STATS}"
    )

    if [[ "${USE_EXTERNAL_STATS}" == "true" ]]; then
        ARGS+=(--dataset.external_stats_path="${DATASET_EXTERNAL_STATS_PATH}")
    fi

    if [[ -n "${WEIGHT_RULES_PATH}" ]]; then
        ARGS+=(--dataset.weight_rules_path="${WEIGHT_RULES_PATH}")
    fi
fi

if [[ -n "${DIST_LOADING}" ]]; then
    ARGS+=(--dataset.dist_loading="${DIST_LOADING}")
fi

if [[ -n "${STEPS}" ]]; then
    ARGS+=(--steps="${STEPS}")
fi

if [[ -n "${SAVE_FREQ}" ]]; then
    ARGS+=(--save_freq="${SAVE_FREQ}")
fi

if [[ -n "${LOG_FREQ}" ]]; then
    ARGS+=(--log_freq="${LOG_FREQ}")
fi

if [[ -n "${BATCH_SIZE}" ]]; then
    ARGS+=(--batch_size="${BATCH_SIZE}")
fi

if [[ -n "${GRADIENT_ACCUMULATION_STEPS}" ]]; then
    ARGS+=(--gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}")
fi

if [[ "${ENABLE_IMAGE_AUG}" == "true" ]]; then
    ARGS+=(
        --dataset.image_transforms.enable=true
        --dataset.image_transforms.preset="${IMAGE_AUG_PRESET}"
    )
fi

accelerate launch "${ARGS[@]}"
