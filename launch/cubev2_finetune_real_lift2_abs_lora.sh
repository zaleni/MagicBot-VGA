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
POLICY_INIT_PATH="${POLICY_INIT_PATH:-/home/jiangjiahao/data/model/MagicBot-VGA-Base}"
QWEN3_VL_PRETRAINED_PATH="${QWEN3_VL_PRETRAINED_PATH:-/home/jiangjiahao/data/model/Qwen3-VL-2B-Instruct}"
QWEN3_VL_PROCESSOR_PATH="${QWEN3_VL_PROCESSOR_PATH:-${QWEN3_VL_PRETRAINED_PATH}}"
COSMOS_TOKENIZER_PATH_OR_NAME="${COSMOS_TOKENIZER_PATH_OR_NAME:-/home/jiangjiahao/data/model/Cosmos-Tokenizer-CI8x8}"
DA3_MODEL_PATH_OR_NAME="${DA3_MODEL_PATH_OR_NAME:-/home/jiangjiahao/data/model/DA3-LARGE-1.1}"
DA3_VARIANT="${DA3_VARIANT:-auto}"
DA3_ALIGNMENT_MODE="${DA3_ALIGNMENT_MODE:-query_decoder}"
DA3_CODE_ROOT="${DA3_CODE_ROOT:-}"

DATASET_DIR="${DATASET_DIR:-/home/jiangjiahao/data/zhenji/data100_0420/scene1_joint_96_60hz_v30}"
DATASET_NAME="${DATASET_NAME:-$(basename "${DATASET_DIR}")}"
DATASET_REPO_ID="${DATASET_REPO_ID:-${DATASET_DIR}}"

ACTION_TYPE="${ACTION_TYPE:-abs}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-${CHUNK_SIZE}}"
ENABLE_3D_QUERIES="${ENABLE_3D_QUERIES:-true}"
NUM_3D_QUERY_TOKENS="${NUM_3D_QUERY_TOKENS:-432}"
LAMBDA_3D="${LAMBDA_3D:-0.01}"

USE_EXTERNAL_STATS="${USE_EXTERNAL_STATS:-true}"
NORM_STATS_ROOT="${NORM_STATS_ROOT:-/home/jiangjiahao/data/zhenji/norm_stats}"
DATASET_EXTERNAL_STATS_PATH="${DATASET_EXTERNAL_STATS_PATH:-${NORM_STATS_ROOT}/${ACTION_TYPE}/${DATASET_NAME}/stats.json}"

BATCH_SIZE="${BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
STEPS="${STEPS:-60000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
LOG_FREQ="${LOG_FREQ:-50}"

###############################################################################
################################ LoRA config ##################################

# JSON array strings are expected here because draccus parses list/tuple CLI
# overrides reliably from JSON-like syntax.
# Default behavior keeps LoRA disabled so this launch script falls back to
# standard full-parameter finetuning unless LORA_MODULES is set explicitly.
LORA_MODULES="${LORA_MODULES:-[]}"
LORA_UNSELECTED_MODE="${LORA_UNSELECTED_MODE:-full}"
LORA_TARGETS="${LORA_TARGETS:-[\"attn\",\"ffn\"]}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32.0}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"

# Optional presets:
#   LORA_MODULES='[]'                      # full finetuning (default)
#   LORA_MODULES='["und"]'                 # only understanding expert LoRA
#   LORA_MODULES='["und","act"]'           # understanding + action expert LoRA
#   LORA_MODULES='["und","gen","act"]'     # full LoRA on all three experts
#   LORA_UNSELECTED_MODE='freeze'          # freeze experts not selected for LoRA
#   LORA_TARGETS='["attn"]'                # attention-only LoRA

if [[ -z "${POLICY_INIT_PATH}" ]]; then
  echo "Please set POLICY_INIT_PATH to the CubeV2 bootstrap checkpoint."
  echo "For backward compatibility, PRETRAINED_PATH is also accepted."
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
  echo "Compute them first with util_scripts/compute_norm_stats_single.py"
  exit 1
fi

python -c 'from lerobot.transforms.constants import MASK_MAPPING, FEATURE_MAPPING, IMAGE_MAPPING; import sys; rt=sys.argv[1]; missing=[name for name,m in [("MASK_MAPPING", MASK_MAPPING), ("FEATURE_MAPPING", FEATURE_MAPPING), ("IMAGE_MAPPING", IMAGE_MAPPING)] if rt not in m]; raise SystemExit(0 if not missing else "robot_type=" + rt + " missing in " + ", ".join(missing))' \
  "${robot_type}"

LORA_JOB_TAG="$(printf '%s' "${LORA_MODULES}" | tr -d '[]\" ' | tr ',' '-')"
if [[ -z "${LORA_JOB_TAG}" ]]; then
  LORA_JOB_TAG="fullft"
fi

BASE_OUTPUT_DIR="outputs_real/${POLICY}"
JOB_NAME="${POLICY}-real_lift2-${ACTION_TYPE}-chunk${CHUNK_SIZE}-lora-${LORA_JOB_TAG}-finetune-$(date +'%Y_%m_%d_%H_%M_%S')"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${JOB_NAME}"

echo "DATASET_DIR=${DATASET_DIR}"
echo "DATASET_NAME=${DATASET_NAME}"
echo "DATASET_REPO_ID=${DATASET_REPO_ID}"
echo "robot_type=${robot_type}"
echo "ACTION_TYPE=${ACTION_TYPE}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "N_ACTION_STEPS=${N_ACTION_STEPS}"
echo "BATCH_SIZE(per_device)=${BATCH_SIZE}"
echo "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
echo "USE_EXTERNAL_STATS=${USE_EXTERNAL_STATS}"
echo "DATASET_EXTERNAL_STATS_PATH=${DATASET_EXTERNAL_STATS_PATH}"
echo "LORA_MODULES=${LORA_MODULES}"
echo "LORA_UNSELECTED_MODE=${LORA_UNSELECTED_MODE}"
echo "LORA_TARGETS=${LORA_TARGETS}"
echo "LORA_RANK=${LORA_RANK}"
echo "LORA_ALPHA=${LORA_ALPHA}"
echo "LORA_DROPOUT=${LORA_DROPOUT}"
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
    --num_workers=12
    --job_name="${JOB_NAME}"

    --policy.type="${POLICY}"
    --policy.repo_id="lerobot_lab/${POLICY}"
    --policy.pretrained_path="${POLICY_INIT_PATH}"
    --policy.qwen3_vl_pretrained_path="${QWEN3_VL_PRETRAINED_PATH}"
    --policy.cosmos_tokenizer_path_or_name="${COSMOS_TOKENIZER_PATH_OR_NAME}"
    --policy.push_to_hub=false
    --policy.gradient_checkpointing=false
    --policy.dtype=bfloat16
    --policy.optimizer_lr=3.5e-5
    --policy.scheduler_warmup_steps=2000
    --policy.scheduler_decay_steps="${STEPS}"
    --policy.scheduler_decay_lr=5.0e-6
    --policy.freeze_vision_encoder=false
    --policy.train_expert_only=false
    --policy.train_vlm_only=false
    --policy.lora_modules="${LORA_MODULES}"
    --policy.lora_unselected_mode="${LORA_UNSELECTED_MODE}"
    --policy.lora_targets="${LORA_TARGETS}"
    --policy.lora_rank="${LORA_RANK}"
    --policy.lora_alpha="${LORA_ALPHA}"
    --policy.lora_dropout="${LORA_DROPOUT}"
    --policy.qwen3_vl_variant=qwen3_vl_28l
    --policy.action_expert_variant=qwen3_28l
    --policy.chunk_size="${CHUNK_SIZE}"
    --policy.n_action_steps="${N_ACTION_STEPS}"
    --policy.enable_3d_queries="${ENABLE_3D_QUERIES}"
    --policy.num_3d_query_tokens="${NUM_3D_QUERY_TOKENS}"
    --policy.lambda_3d="${LAMBDA_3D}"
    --policy.da3_model_path_or_name="${DA3_MODEL_PATH_OR_NAME}"
    --policy.da3_variant="${DA3_VARIANT}"
    --policy.da3_alignment_mode="${DA3_ALIGNMENT_MODE}"
    --policy.log_da3_teacher_timing=true

    --dataset.type="${POLICY}"
    --dataset.repo_id="${DATASET_REPO_ID}"
    --dataset.qwen3_vl_processor_path="${QWEN3_VL_PROCESSOR_PATH}"
    --dataset.action_mode="${ACTION_TYPE}"
    --dataset.use_external_stats="${USE_EXTERNAL_STATS}"

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

accelerate launch "${ARGS[@]}"
