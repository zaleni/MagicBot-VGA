#!/usr/bin/env bash
set -euo pipefail

###############################################################################
################################# ENV config ##################################

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-6384}"
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
DEFAULT_ROBOCHALLENGE_RAW_ROOT="/inspire/qb-ilm/project/embodied-basic-model/zhangjianing-253108140206/DATASET/Robochallenge_table30v2_unzipped"
ROBOCHALLENGE_RAW_ROOT="${ROBOCHALLENGE_RAW_ROOT:-${DATASET_ROOT:-${DEFAULT_ROBOCHALLENGE_RAW_ROOT}}}"
TASK_REGEX="${TASK_REGEX:-}"
TASK_PRESET="${TASK_PRESET:-table30v2_aloha}"
WEIGHTED_TASK_SAMPLING="${WEIGHTED_TASK_SAMPLING:-true}"
TASK_SAMPLING_MODE="${TASK_SAMPLING_MODE:-group_frames_pow}"
TASK_SAMPLING_GAMMA="${TASK_SAMPLING_GAMMA:-0.8}"
REGULAR_TASK_WEIGHT="${REGULAR_TASK_WEIGHT:-1.0}"
EXTRA_TASK_WEIGHT="${EXTRA_TASK_WEIGHT:-0.8}"
# ALOHA table30v2 has 4 regular tasks and 6 extra tasks. Both groups get
# total budget 4.0; inside each group, tasks are allocated by frames^gamma.
REGULAR_TASK_TOTAL_WEIGHT="${REGULAR_TASK_TOTAL_WEIGHT:-4.0}"
EXTRA_TASK_TOTAL_WEIGHT="${EXTRA_TASK_TOTAL_WEIGHT:-4.0}"
FRAME_INTERVAL="${FRAME_INTERVAL:-1}"
STATE_CACHE_DIR="${STATE_CACHE_DIR:-outputs_robochallenge/raw_cache/aloha_states}"
VALIDATE_VIDEOS="${VALIDATE_VIDEOS:-false}"

POLICY_INIT_PATH="${POLICY_INIT_PATH:-${PRETRAINED_PATH:-/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/Foundation-Moodel/outputs/cubev2/cubev2-multidata-delta-pretrain-2026_04_07_07_42_16/checkpoints/300000/pretrained_model}}"
QWEN3_VL_PRETRAINED_PATH="${QWEN3_VL_PRETRAINED_PATH:-/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/DATASET/model/Qwen3-VL-2B-Instruct}"
QWEN3_VL_PROCESSOR_PATH="${QWEN3_VL_PROCESSOR_PATH:-${QWEN3_VL_PRETRAINED_PATH}}"
COSMOS_TOKENIZER_PATH_OR_NAME="${COSMOS_TOKENIZER_PATH_OR_NAME:-/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/DATASET/model/Cosmos-Tokenizer-CI8x8}"
DA3_MODEL_PATH_OR_NAME="${DA3_MODEL_PATH_OR_NAME:-/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/DATASET/model/DA3-LARGE-1-1}"
DA3_VARIANT="${DA3_VARIANT:-auto}"
DA3_ALIGNMENT_MODE="${DA3_ALIGNMENT_MODE:-query_decoder}"
DA3_CODE_ROOT="${DA3_CODE_ROOT:-}"

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
DATASET_EXTERNAL_STATS_PATH="${DATASET_EXTERNAL_STATS_PATH:-${NORM_STATS_ROOT}/robochallenge_raw_aloha/${TASK_PRESET}/${ACTION_TYPE}/chunk${CHUNK_SIZE}_frame${FRAME_INTERVAL}/stats.json}"

ENABLE_IMAGE_AUG="${ENABLE_IMAGE_AUG:-false}"
IMAGE_AUG_PRESET="${IMAGE_AUG_PRESET:-pi05}"

BATCH_SIZE="${BATCH_SIZE:-12}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
STEPS="${STEPS:-300000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
LOG_FREQ="${LOG_FREQ:-50}"
NUM_WORKERS="${NUM_WORKERS:-16}"
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

if [[ -z "${ROBOCHALLENGE_RAW_ROOT}" ]]; then
  echo "Set ROBOCHALLENGE_RAW_ROOT to the RoboChallenge raw root or a single ALOHA task directory."
  exit 1
fi

if [[ -z "${POLICY_INIT_PATH}" ]]; then
  echo "Please set POLICY_INIT_PATH or PRETRAINED_PATH to the CubeV2 checkpoint you want to finetune."
  exit 1
fi

if [[ "${ACTION_TYPE}" != "delta" && "${ACTION_TYPE}" != "abs" ]]; then
  echo "ACTION_TYPE must be abs or delta, got ${ACTION_TYPE}"
  exit 1
fi

if [[ "${USE_EXTERNAL_STATS}" != "true" ]]; then
  echo "RoboChallenge raw ALOHA training requires USE_EXTERNAL_STATS=true."
  exit 1
fi

if [[ ! -f "${DATASET_EXTERNAL_STATS_PATH}" ]]; then
  echo "Missing external stats: ${DATASET_EXTERNAL_STATS_PATH}"
  echo "Compute them first with:"
  echo "  ROBOCHALLENGE_RAW_ROOT=... TASK_PRESET=${TASK_PRESET} ACTION_TYPE=${ACTION_TYPE} CHUNK_SIZE=${CHUNK_SIZE} FRAME_INTERVAL=${FRAME_INTERVAL} bash launch/compute_norm_stats_robochallenge_raw_aloha.sh"
  exit 1
fi

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs_robochallenge/${POLICY}}"
JOB_NAME="${JOB_NAME:-${POLICY}-robochallenge_raw_aloha-${TASK_PRESET}-${ACTION_TYPE}-chunk${CHUNK_SIZE}-frame${FRAME_INTERVAL}-finetune-$(date +'%Y_%m_%d_%H_%M_%S')}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${JOB_NAME}"

echo "POLICY_INIT_PATH=${POLICY_INIT_PATH}"
echo "ROBOCHALLENGE_RAW_ROOT=${ROBOCHALLENGE_RAW_ROOT}"
echo "TASK_REGEX=${TASK_REGEX:-<all ALOHA preset tasks>}"
echo "TASK_PRESET=${TASK_PRESET}"
echo "WEIGHTED_TASK_SAMPLING=${WEIGHTED_TASK_SAMPLING}"
echo "TASK_SAMPLING_MODE=${TASK_SAMPLING_MODE}"
echo "TASK_SAMPLING_GAMMA=${TASK_SAMPLING_GAMMA}"
echo "REGULAR_TASK_WEIGHT=${REGULAR_TASK_WEIGHT}"
echo "EXTRA_TASK_WEIGHT=${EXTRA_TASK_WEIGHT}"
echo "REGULAR_TASK_TOTAL_WEIGHT=${REGULAR_TASK_TOTAL_WEIGHT}"
echo "EXTRA_TASK_TOTAL_WEIGHT=${EXTRA_TASK_TOTAL_WEIGHT}"
echo "ACTION_TYPE=${ACTION_TYPE}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "N_ACTION_STEPS=${N_ACTION_STEPS}"
echo "CUBEV2_ATTENTION_MASK_MODE=${CUBEV2_ATTENTION_MASK_MODE}"
echo "IMAGE_DELTA_INDICES=${IMAGE_DELTA_INDICES}"
echo "FRAME_INTERVAL=${FRAME_INTERVAL}"
echo "STATE_CACHE_DIR=${STATE_CACHE_DIR}"
echo "DATASET_EXTERNAL_STATS_PATH=${DATASET_EXTERNAL_STATS_PATH}"
echo "ENABLE_IMAGE_AUG=${ENABLE_IMAGE_AUG}"
echo "IMAGE_AUG_PRESET=${IMAGE_AUG_PRESET}"
echo "BATCH_SIZE(per_device)=${BATCH_SIZE}"
echo "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
echo "STEPS=${STEPS}"
echo "DTYPE=${DTYPE}"
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

    --dataset.type=robochallenge_raw_aloha
    --dataset.repo_id=robochallenge_raw_aloha
    --dataset.raw_root="${ROBOCHALLENGE_RAW_ROOT}"
    --dataset.embodiment=ALOHA
    --dataset.action_representation=joint
    --dataset.frame_interval="${FRAME_INTERVAL}"
    --dataset.task_preset="${TASK_PRESET}"
    --dataset.weighted_task_sampling="${WEIGHTED_TASK_SAMPLING}"
    --dataset.task_sampling_mode="${TASK_SAMPLING_MODE}"
    --dataset.task_sampling_gamma="${TASK_SAMPLING_GAMMA}"
    --dataset.regular_task_weight="${REGULAR_TASK_WEIGHT}"
    --dataset.extra_task_weight="${EXTRA_TASK_WEIGHT}"
    --dataset.regular_task_total_weight="${REGULAR_TASK_TOTAL_WEIGHT}"
    --dataset.extra_task_total_weight="${EXTRA_TASK_TOTAL_WEIGHT}"
    --dataset.state_cache_dir="${STATE_CACHE_DIR}"
    --dataset.validate_videos="${VALIDATE_VIDEOS}"
    --dataset.qwen3_vl_processor_path="${QWEN3_VL_PROCESSOR_PATH}"
    --dataset.action_mode="${ACTION_TYPE}"
    --dataset.use_external_stats=true
    --dataset.external_stats_path="${DATASET_EXTERNAL_STATS_PATH}"

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

if [[ -n "${TASK_REGEX}" ]]; then
    ARGS+=(--dataset.task_regex="${TASK_REGEX}")
fi

if [[ -n "${DA3_CODE_ROOT}" ]]; then
    ARGS+=(--policy.da3_code_root="${DA3_CODE_ROOT}")
fi

if [[ "${ENABLE_IMAGE_AUG}" == "true" ]]; then
    ARGS+=(
        --dataset.image_transforms.enable=true
        --dataset.image_transforms.preset="${IMAGE_AUG_PRESET}"
    )
fi

accelerate launch "${ARGS[@]}"
