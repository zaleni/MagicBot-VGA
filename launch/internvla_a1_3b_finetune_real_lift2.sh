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

# InternVLA-A1-3B is registered as the qwena1 policy in this codebase.
POLICY="qwena1"
MODEL_NAME="internvla_a1_3b"

DATASET_DIR="${DATASET_DIR:-/home/jiangjiahao/data/zhenji/table_clean_100_filter}"
DATASET_NAME="${DATASET_NAME:-$(basename "${DATASET_DIR}")}"
DATASET_REPO_ID="${DATASET_REPO_ID:-${DATASET_DIR}}"

POLICY_INIT_PATH="${POLICY_INIT_PATH:-/home/jiangjiahao/data/model/InternVLA-A1-3B}"
QWEN3_VL_PRETRAINED_PATH="${QWEN3_VL_PRETRAINED_PATH:-/home/jiangjiahao/data/model/Qwen3-VL-2B-Instruct}"
QWEN3_VL_PROCESSOR_PATH="${QWEN3_VL_PROCESSOR_PATH:-${QWEN3_VL_PRETRAINED_PATH}}"
COSMOS_TOKENIZER_PATH_OR_NAME="${COSMOS_TOKENIZER_PATH_OR_NAME:-/home/jiangjiahao/data/model/Cosmos-Tokenizer-CI8x8}"

# Kept here only so the baseline prints the same address block as CubeV2 runs.
# InternVLA-A1 does not use DA3 teacher supervision.
DA3_MODEL_PATH_OR_NAME="${DA3_MODEL_PATH_OR_NAME:-/home/jiangjiahao/data/model/DA3-LARGE-1.1}"

ACTION_TYPE="${ACTION_TYPE:-delta}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-${CHUNK_SIZE}}"
LAMBDA_GEN="${LAMBDA_GEN:-0.01}"

USE_EXTERNAL_STATS="${USE_EXTERNAL_STATS:-true}"
NORM_STATS_ROOT="${NORM_STATS_ROOT:-/home/jiangjiahao/data/zhenji/norm_stats}"
DATASET_EXTERNAL_STATS_PATH="${DATASET_EXTERNAL_STATS_PATH:-${NORM_STATS_ROOT}/${ACTION_TYPE}/${DATASET_NAME}/stats.json}"

# Official InternVLA-A1 baseline: no image augmentation.
ENABLE_IMAGE_AUG="${ENABLE_IMAGE_AUG:-false}"

BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-60000}"
SAVE_FREQ="${SAVE_FREQ:-20000}"
LOG_FREQ="${LOG_FREQ:-200}"
NUM_WORKERS="${NUM_WORKERS:-12}"

if [[ "${ENABLE_IMAGE_AUG}" == "true" ]]; then
  echo "This InternVLA-A1 baseline script intentionally disables image augmentation."
  echo "Unset ENABLE_IMAGE_AUG or set ENABLE_IMAGE_AUG=false."
  exit 1
fi

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "DATASET_DIR does not exist: ${DATASET_DIR}"
  exit 1
fi

if [[ ! -f "${DATASET_DIR}/meta/info.json" ]]; then
  echo "meta/info.json not found under DATASET_DIR: ${DATASET_DIR}"
  exit 1
fi

if [[ "${POLICY_INIT_PATH}" == /* && ! -d "${POLICY_INIT_PATH}" ]]; then
  echo "POLICY_INIT_PATH does not exist: ${POLICY_INIT_PATH}"
  exit 1
fi

if [[ "${QWEN3_VL_PRETRAINED_PATH}" == /* && ! -d "${QWEN3_VL_PRETRAINED_PATH}" ]]; then
  echo "QWEN3_VL_PRETRAINED_PATH does not exist: ${QWEN3_VL_PRETRAINED_PATH}"
  exit 1
fi

if [[ "${QWEN3_VL_PROCESSOR_PATH}" == /* && ! -d "${QWEN3_VL_PROCESSOR_PATH}" ]]; then
  echo "QWEN3_VL_PROCESSOR_PATH does not exist: ${QWEN3_VL_PROCESSOR_PATH}"
  exit 1
fi

if [[ "${COSMOS_TOKENIZER_PATH_OR_NAME}" == /* ]]; then
  if [[ ! -f "${COSMOS_TOKENIZER_PATH_OR_NAME}/encoder.jit" || ! -f "${COSMOS_TOKENIZER_PATH_OR_NAME}/decoder.jit" ]]; then
    echo "COSMOS_TOKENIZER_PATH_OR_NAME must contain encoder.jit and decoder.jit: ${COSMOS_TOKENIZER_PATH_OR_NAME}"
    exit 1
  fi
fi

robot_type="$(
  python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["robot_type"])' \
    "${DATASET_DIR}/meta/info.json"
)"

if [[ "${robot_type}" != "real_lift2" ]]; then
  echo "Expected robot_type=real_lift2, got ${robot_type}"
  echo "Please reconvert the dataset or update meta/info.json."
  exit 1
fi

if [[ "${ACTION_TYPE}" != "delta" && "${ACTION_TYPE}" != "abs" ]]; then
  echo "ACTION_TYPE must be abs or delta, got ${ACTION_TYPE}"
  exit 1
fi

if [[ "${USE_EXTERNAL_STATS}" == "true" && ! -f "${DATASET_EXTERNAL_STATS_PATH}" ]]; then
  echo "Missing external stats: ${DATASET_EXTERNAL_STATS_PATH}"
  echo "Compute them first with:"
  echo "  DATASET_DIR=\"${DATASET_DIR}\" NORM_STATS_ROOT=\"${NORM_STATS_ROOT}\" ACTION_TYPE=\"${ACTION_TYPE}\" CHUNK_SIZE=\"${CHUNK_SIZE}\" bash launch/compute_norm_stats_real_lift2_delta.sh"
  exit 1
fi

python -c 'from lerobot.transforms.constants import MASK_MAPPING, FEATURE_MAPPING, IMAGE_MAPPING; import sys; rt=sys.argv[1]; missing=[name for name,m in [("MASK_MAPPING", MASK_MAPPING), ("FEATURE_MAPPING", FEATURE_MAPPING), ("IMAGE_MAPPING", IMAGE_MAPPING)] if rt not in m]; raise SystemExit(0 if not missing else "robot_type=" + rt + " missing in " + ", ".join(missing))' \
  "${robot_type}"

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs_real/${MODEL_NAME}}"
JOB_NAME="${JOB_NAME:-${MODEL_NAME}-real_lift2-${ACTION_TYPE}-chunk${CHUNK_SIZE}-noaug-finetune-$(date +'%Y_%m_%d_%H_%M_%S')}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_OUTPUT_DIR}/${JOB_NAME}}"

echo "DATASET_DIR=${DATASET_DIR}"
echo "DATASET_NAME=${DATASET_NAME}"
echo "DATASET_REPO_ID=${DATASET_REPO_ID}"
echo "robot_type=${robot_type}"
echo "POLICY=${POLICY}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "POLICY_INIT_PATH=${POLICY_INIT_PATH}"
echo "QWEN3_VL_PRETRAINED_PATH=${QWEN3_VL_PRETRAINED_PATH}"
echo "QWEN3_VL_PROCESSOR_PATH=${QWEN3_VL_PROCESSOR_PATH}"
echo "COSMOS_TOKENIZER_PATH_OR_NAME=${COSMOS_TOKENIZER_PATH_OR_NAME}"
echo "DA3_MODEL_PATH_OR_NAME=${DA3_MODEL_PATH_OR_NAME} (unused by InternVLA-A1 baseline)"
echo "ACTION_TYPE=${ACTION_TYPE}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "N_ACTION_STEPS=${N_ACTION_STEPS}"
echo "USE_EXTERNAL_STATS=${USE_EXTERNAL_STATS}"
echo "DATASET_EXTERNAL_STATS_PATH=${DATASET_EXTERNAL_STATS_PATH}"
echo "ENABLE_IMAGE_AUG=${ENABLE_IMAGE_AUG}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "STEPS=${STEPS}"
echo "SAVE_FREQ=${SAVE_FREQ}"
echo "LOG_FREQ=${LOG_FREQ}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

ARGS=(
    --multi_gpu
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
    --policy.dtype=bfloat16
    --policy.optimizer_lr=5.0e-5
    --policy.scheduler_warmup_steps=2000
    --policy.scheduler_decay_steps="${STEPS}"
    --policy.scheduler_decay_lr=5.0e-6
    --policy.freeze_vision_encoder=false
    --policy.train_expert_only=false
    --policy.train_vlm_only=false
    --policy.qwen3_vl_variant=qwen3_vl_28l
    --policy.action_expert_variant=qwen3_28l
    --policy.chunk_size="${CHUNK_SIZE}"
    --policy.n_action_steps="${N_ACTION_STEPS}"
    --policy.lambda_gen="${LAMBDA_GEN}"

    --dataset.type="${POLICY}"
    --dataset.repo_id="${DATASET_REPO_ID}"
    --dataset.qwen3_vl_processor_path="${QWEN3_VL_PROCESSOR_PATH}"
    --dataset.action_mode="${ACTION_TYPE}"
    --dataset.use_external_stats="${USE_EXTERNAL_STATS}"
    --dataset.image_transforms.enable=false

    --seed=42
    --batch_size="${BATCH_SIZE}"
    --steps="${STEPS}"
    --save_freq="${SAVE_FREQ}"
    --log_freq="${LOG_FREQ}"

    --wandb.enable=true
    --wandb.project="${WANDB_PROJECT:-InternVLA_A1_3B}"
    --wandb.mode="${WANDB_MODE}"
)

if [[ "${USE_EXTERNAL_STATS}" == "true" ]]; then
    ARGS+=(--dataset.external_stats_path="${DATASET_EXTERNAL_STATS_PATH}")
fi

accelerate launch "${ARGS[@]}"
