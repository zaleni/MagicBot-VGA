#!/usr/bin/env bash
set -euo pipefail

###############################################################################
################################# ENV config ##################################

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-6391}
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

DDP_TIMEOUT_SEC="${DDP_TIMEOUT_SEC:-3600}"
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-false}"
PARALLEL_DATASET_LOAD="${PARALLEL_DATASET_LOAD:-true}"
REPO_ASSIGNMENT_LOG_MODE="${REPO_ASSIGNMENT_LOG_MODE:-summary}"
REPO_ASSIGNMENT_LOG_LIMIT="${REPO_ASSIGNMENT_LOG_LIMIT:-4}"
WEIGHT_RULES_DEFAULT_GROUP_MODE="${WEIGHT_RULES_DEFAULT_GROUP_MODE:-error}"
WEIGHT_RULES_DEFAULT_GROUP_LIMIT="${WEIGHT_RULES_DEFAULT_GROUP_LIMIT:-20}"
ADAPTER_LOG_MODE="${ADAPTER_LOG_MODE:-summary}"
ADAPTER_LOG_LIMIT="${ADAPTER_LOG_LIMIT:-3}"
RANK_DEVICE_LOG="${RANK_DEVICE_LOG:-true}"

export LEROBOT_DDP_TIMEOUT_SEC="${DDP_TIMEOUT_SEC}"
export LEROBOT_DDP_FIND_UNUSED_PARAMETERS="${LEROBOT_DDP_FIND_UNUSED_PARAMETERS:-${DDP_FIND_UNUSED_PARAMETERS}}"
export LEROBOT_PARALLEL_DATASET_LOAD="${PARALLEL_DATASET_LOAD}"
export LEROBOT_REPO_ASSIGNMENT_LOG_MODE="${REPO_ASSIGNMENT_LOG_MODE}"
export LEROBOT_REPO_ASSIGNMENT_LOG_LIMIT="${REPO_ASSIGNMENT_LOG_LIMIT}"
export LEROBOT_WEIGHT_RULES_DEFAULT_GROUP_MODE="${WEIGHT_RULES_DEFAULT_GROUP_MODE}"
export LEROBOT_WEIGHT_RULES_DEFAULT_GROUP_LIMIT="${WEIGHT_RULES_DEFAULT_GROUP_LIMIT}"
export LEROBOT_MAGICBOT_R0_ADAPTER_LOG_MODE="${ADAPTER_LOG_MODE}"
export LEROBOT_MAGICBOT_R0_ADAPTER_LOG_LIMIT="${ADAPTER_LOG_LIMIT}"
export LEROBOT_LOG_RANK_DEVICE_MAP="${RANK_DEVICE_LOG}"

###############################################################################
############################### RESUME config #################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
echo "SCRIPT_DIR = ${SCRIPT_DIR}"
echo "PROJ_ROOT  = ${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJ_ROOT}"

DTYPE="${DTYPE:-bfloat16}"
NUM_WORKERS="${NUM_WORKERS:-16}"
RESUME_CONFIG_PATH="${RESUME_CONFIG_PATH:-/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/MagicBot-VGA/outputs/MagicBot_R0/MagicBot_R0-magicbot_r0-multidata-delta-pretrain-2026_05_07_15_59_20/checkpoints/240000/pretrained_model/train_config.json}"
RESUME_CHECKPOINT_DIR="$(dirname "$(dirname "${RESUME_CONFIG_PATH}")")"

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

if [[ ! -f "${RESUME_CONFIG_PATH}" ]]; then
  echo "Resume config not found: ${RESUME_CONFIG_PATH}"
  exit 1
fi

if [[ ! -d "${RESUME_CHECKPOINT_DIR}/training_state" ]]; then
  echo "Resume training_state not found: ${RESUME_CHECKPOINT_DIR}/training_state"
  exit 1
fi

echo "RESUME=true"
echo "RESUME_CONFIG_PATH=${RESUME_CONFIG_PATH}"
echo "RESUME_CHECKPOINT_DIR=${RESUME_CHECKPOINT_DIR}"
echo "NUM_PROCESSES=${NUM_PROCESSES}, NUM_WORKERS=${NUM_WORKERS}, DTYPE=${DTYPE}"

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
    --num_workers="${NUM_WORKERS}"
)

accelerate launch "${ARGS[@]}"
