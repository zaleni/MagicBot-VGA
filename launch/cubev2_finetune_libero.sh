#!/usr/bin/env bash
set -euo pipefail

###############################################################################
################################# ENV config ##################################

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-6379}
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

PROC_PER_NODE="${PROC_PER_NODE:-8}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export WANDB_MODE=${WANDB_MODE:-online}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

###############################################################################
############################## TRAINING config ################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
echo "SCRIPT_DIR = ${SCRIPT_DIR}"
echo "PROJ_ROOT  = ${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

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

LIBERO_ROOT="/home/jiangjiahao/data/LIBERO/libero_v30"
USE_DIST_LOADING="${USE_DIST_LOADING:-false}"
VALIDATE_DATASETS="${VALIDATE_DATASETS:-true}"
VIDEO_BACKEND="${VIDEO_BACKEND:-}"

# LIBERO actions are consumed downstream as end-effector deltas in the StarVLA
# evaluation pipeline, so keep the stored action as-is unless the caller
# explicitly overrides this to a different mode.
ACTION_TYPE="${ACTION_TYPE:-abs}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"
N_ACTION_STEPS="${N_ACTION_STEPS:-${CHUNK_SIZE}}"

ENABLE_3D_QUERIES="${ENABLE_3D_QUERIES:-true}"
LAMBDA_3D="${LAMBDA_3D:-0.01}"
GEN_LAMBDA="${GEN_LAMBDA:-0.002}"
NUM_3D_QUERY_TOKENS="${NUM_3D_QUERY_TOKENS:-432}"

USE_EXTERNAL_STATS="${USE_EXTERNAL_STATS:-true}"
DATASET_EXTERNAL_STATS_PATH="/home/jiangjiahao/research/MagicBot-VGA/outputs/norm_stats/libero_all_chunk10/franka/abs/stats.json"
DATASET_EXTERNAL_STATS_ROOT="${DATASET_EXTERNAL_STATS_ROOT:-}"

BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
STEPS="${STEPS:-60000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
LOG_FREQ="${LOG_FREQ:-25}"

discover_dataset_dirs() {
  local root="$1"
  if [[ -z "${root}" || ! -d "${root}" ]]; then
    return 0
  fi

  find -L "${root}" -path "*/meta/info.json" 2>/dev/null \
    | while read -r info_path; do
        ds_dir="$(dirname "$(dirname "${info_path}")")"
        ds_name="$(basename "${ds_dir}")"
        case "${ds_name}" in
          libero_*_lerobot_v30)
            if [[ -d "${ds_dir}/data" || -d "${ds_dir}/videos" ]]; then
              echo "${ds_dir}"
            fi
            ;;
        esac
      done \
    | sort -u
}

mapfile -t DATASET_REPO_IDS < <(discover_dataset_dirs "${LIBERO_ROOT}")

if [[ ${#DATASET_REPO_IDS[@]} -eq 0 ]]; then
  echo "No LIBERO v3.0 datasets found under LIBERO_ROOT=${LIBERO_ROOT}"
  echo "Expected directories like libero_goal_no_noops_1.0.0_lerobot_v30"
  exit 1
fi

if [[ -z "${POLICY_INIT_PATH}" ]]; then
  echo "Please set POLICY_INIT_PATH to the CubeV2 bootstrap checkpoint."
  echo "For backward compatibility, PRETRAINED_PATH is also accepted."
  exit 1
fi

if [[ "${USE_EXTERNAL_STATS}" == "true" && -z "${DATASET_EXTERNAL_STATS_PATH}" && -z "${DATASET_EXTERNAL_STATS_ROOT}" ]]; then
  echo "USE_EXTERNAL_STATS=true but neither DATASET_EXTERNAL_STATS_PATH nor DATASET_EXTERNAL_STATS_ROOT is set."
  exit 1
fi

if [[ "${VALIDATE_DATASETS}" == "true" ]]; then
  echo "Validating LIBERO dataset mappings..."
  for ds_dir in "${DATASET_REPO_IDS[@]}"; do
    info_path="${ds_dir}/meta/info.json"
    python -c 'import json, sys
from lerobot.transforms.constants import infer_embodiment_variant
info = json.load(open(sys.argv[1], encoding="utf-8"))
robot_type = info["robot_type"]
resolved = infer_embodiment_variant(robot_type, info.get("features", {}))
codebase_version = info.get("codebase_version", "unknown")
print(f"{sys.argv[2]} -> codebase={codebase_version}, robot_type={robot_type}, resolved={resolved}")
if codebase_version != "v3.0":
    raise SystemExit(f"Dataset is not v3.0: {sys.argv[2]}")
if resolved != "libero_franka":
    raise SystemExit(f"Unexpected mapping resolution for {sys.argv[2]}: {resolved}")
' "${info_path}" "${ds_dir}"

    if [[ "${USE_EXTERNAL_STATS}" == "true" ]]; then
      if [[ -n "${DATASET_EXTERNAL_STATS_PATH}" ]]; then
        stat_path="${DATASET_EXTERNAL_STATS_PATH}"
      else
        stat_path="${DATASET_EXTERNAL_STATS_ROOT}/franka/${ACTION_TYPE}/stats.json"
      fi
      if [[ ! -f "${stat_path}" ]]; then
        echo "Missing external stats for LIBERO: ${stat_path}"
        exit 1
      fi
    fi
  done
else
  echo "Skipping per-dataset validation (VALIDATE_DATASETS=${VALIDATE_DATASETS})."
fi

echo "Discovered ${#DATASET_REPO_IDS[@]} LIBERO datasets under ${LIBERO_ROOT}"
printf '  %s\n' "${DATASET_REPO_IDS[@]}"

BASE_OUTPUT_DIR="/home/jiangjiahao/data/ckpt/${POLICY}"
JOB_NAME="${POLICY}-libero4-${ACTION_TYPE}-chunk${CHUNK_SIZE}-finetune-$(date +'%Y_%m_%d_%H_%M_%S')"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${JOB_NAME}"
REPO_ID_FILE_DIR="${BASE_OUTPUT_DIR}/_repo_id_files"
mkdir -p "${REPO_ID_FILE_DIR}"
REPO_ID_FILE="${REPO_ID_FILE_DIR}/${JOB_NAME}.txt"
printf '%s\n' "${DATASET_REPO_IDS[@]}" > "${REPO_ID_FILE}"

echo "ACTION_TYPE=${ACTION_TYPE}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "N_ACTION_STEPS=${N_ACTION_STEPS}"
echo "GEN_LAMBDA=${GEN_LAMBDA}"
echo "LAMBDA_3D=${LAMBDA_3D}"
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

    --policy.type=${POLICY}
    --policy.repo_id=lerobot_lab/${POLICY}
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
    --policy.enable_3d_queries="${ENABLE_3D_QUERIES}"
    --policy.num_3d_query_tokens="${NUM_3D_QUERY_TOKENS}"
    --policy.lambda_gen="${GEN_LAMBDA}"
    --policy.lambda_3d="${LAMBDA_3D}"
    --policy.da3_model_path_or_name="${DA3_MODEL_PATH_OR_NAME}"
    --policy.da3_variant="${DA3_VARIANT}"
    --policy.da3_alignment_mode="${DA3_ALIGNMENT_MODE}"
    --policy.log_da3_teacher_timing=true

    --dataset.type=${POLICY}
    --dataset.repo_id="multidata_from_file"
    --dataset.repo_id_file="${REPO_ID_FILE}"
    --dataset.qwen3_vl_processor_path="${QWEN3_VL_PROCESSOR_PATH}"
    --dataset.action_mode="${ACTION_TYPE}"
    --dataset.use_external_stats=${USE_EXTERNAL_STATS}

    --seed=42
    --batch_size="${BATCH_SIZE}"
    --gradient_accumulation_steps="${GRAD_ACCUM_STEPS}"
    --steps="${STEPS}"
    --save_freq="${SAVE_FREQ}"
    --log_freq="${LOG_FREQ}"

    --wandb.enable=true
    --wandb.project=CUBEv2
    --wandb.mode=${WANDB_MODE}
)

if [[ -n "${DA3_CODE_ROOT}" ]]; then
    ARGS+=(--policy.da3_code_root="${DA3_CODE_ROOT}")
fi

if [[ -n "${DATASET_EXTERNAL_STATS_PATH}" ]]; then
    ARGS+=(--dataset.external_stats_path="${DATASET_EXTERNAL_STATS_PATH}")
fi

if [[ -n "${DATASET_EXTERNAL_STATS_ROOT}" ]]; then
    ARGS+=(--dataset.external_stats_root="${DATASET_EXTERNAL_STATS_ROOT}")
fi

if [[ -n "${VIDEO_BACKEND}" ]]; then
    ARGS+=(--dataset.video_backend="${VIDEO_BACKEND}")
fi

if [[ "${USE_DIST_LOADING}" == "true" ]]; then
    ARGS+=(--dataset.dist_loading=true)
fi

accelerate launch "${ARGS[@]}"
