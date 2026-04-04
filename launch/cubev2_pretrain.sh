#!/usr/bin/env bash
set -euo pipefail

###############################################################################
################################# ENV config ##################################

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

WANDB_TOKEN=${WANDB_TOKEN}
CONDA_ROOT=${_CONDA_ROOT}
CONDA_ENV=internvla_a1

source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

wandb login ${WANDB_TOKEN}

###############################################################################

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-6379}
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

PROC_PER_NODE="${PROC_PER_NODE:-2}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

export CUDA_HOME="/usr/local/cuda-12.8"
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

###############################################################################
############################## TRAINING config ################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
echo "SCRIPT_DIR = ${SCRIPT_DIR}"
echo "PROJ_ROOT  = ${PROJ_ROOT}"

cd ${PROJ_ROOT}

POLICY="cubev2"
QWEN3_VL_PRETRAINED_PATH="${QWEN3_VL_PRETRAINED_PATH:-Qwen/Qwen3-VL-2B-Instruct}"
QWEN3_VL_PROCESSOR_PATH="${QWEN3_VL_PROCESSOR_PATH:-${QWEN3_VL_PRETRAINED_PATH}}"
COSMOS_TOKENIZER_PATH_OR_NAME="${COSMOS_TOKENIZER_PATH_OR_NAME:-nvidia/Cosmos-Tokenizer-CI8x8}"
DA3_MODEL_PATH_OR_NAME="${DA3_MODEL_PATH_OR_NAME:-depth-anything/DA3-GIANT-1.1}"
DA3_VARIANT="${DA3_VARIANT:-auto}"
DA3_CODE_ROOT="${DA3_CODE_ROOT:-}"
INTERNDATA_ROOT="${INTERNDATA_ROOT:-/inspire/qb-ilm/project/embodied-basic-model/zhangjianing-253108140206/DATASET/InternData-A1-v30}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/DATASET/RoboTwin-LeRobot-v30}"
ROBOCHALLENGE_ROOT="${ROBOCHALLENGE_ROOT:-/inspire/qb-ilm/project/embodied-basic-model/zhangjianing-253108140206/DATASET/Robochallengev3.0_eef}"
AGIBOT_ROOT="${AGIBOT_ROOT:-/inspire/qb-ilm/project/embodied-basic-model/zhangjianing-253108140206/DATASET/Agibotv3.0}"
EGODEX_LEROBOT_ROOT="${EGODEX_LEROBOT_ROOT:-}"
WEIGHT_RULES_PATH="${WEIGHT_RULES_PATH:-}"

ACTION_TYPE=delta
USE_EXTERNAL_STATS="${USE_EXTERNAL_STATS:-true}"
DATASET_EXTERNAL_STATS_PATH="${DATASET_EXTERNAL_STATS_PATH:-}"
DATASET_EXTERNAL_STATS_ROOT="${DATASET_EXTERNAL_STATS_ROOT:-outputs/norm_stats}"

discover_dataset_dirs() {
  local root="$1"
  if [[ -z "${root}" || ! -d "${root}" ]]; then
    return 0
  fi

  find -L "${root}" -path "*/meta/info.json" 2>/dev/null \
    | while read -r info_path; do
        ds_dir="$(dirname "$(dirname "${info_path}")")"
        if [[ -d "${ds_dir}/data" || -d "${ds_dir}/videos" ]]; then
          echo "${ds_dir}"
        fi
      done \
    | sort -u
}

mapfile -t DATASET_REPO_IDS < <(
  {
    discover_dataset_dirs "${INTERNDATA_ROOT}"
    discover_dataset_dirs "${ROBOTWIN_ROOT}"
    discover_dataset_dirs "${ROBOCHALLENGE_ROOT}"
    discover_dataset_dirs "${AGIBOT_ROOT}"
    discover_dataset_dirs "${EGODEX_LEROBOT_ROOT}"
  } | sort -u
)

if [[ ${#DATASET_REPO_IDS[@]} -eq 0 ]]; then
  echo "No valid LeRobot datasets found."
  echo "Please set one or more of: INTERNDATA_ROOT ROBOTWIN_ROOT ROBOCHALLENGE_ROOT AGIBOT_ROOT EGODEX_LEROBOT_ROOT"
  exit 1
fi

if [[ "${USE_EXTERNAL_STATS}" == "true" ]]; then
  if [[ -n "${DATASET_EXTERNAL_STATS_PATH}" ]]; then
    echo "cubev2_pretrain.sh is a multi-dataset script and does not accept DATASET_EXTERNAL_STATS_PATH."
    echo "Please set DATASET_EXTERNAL_STATS_ROOT instead."
    exit 1
  fi
  if [[ -z "${DATASET_EXTERNAL_STATS_ROOT}" ]]; then
    echo "USE_EXTERNAL_STATS=true but DATASET_EXTERNAL_STATS_ROOT is not set."
    exit 1
  fi
fi

BASE_OUTPUT_DIR="outputs/${POLICY}"
DATASET_NAME="a1"
JOB_NAME="$(date +'%Y_%m_%d_%H_%M_%S')-${POLICY}-${DATASET_NAME}-${ACTION_TYPE}-pretrain"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${JOB_NAME}"
REPO_ID_FILE_DIR="${BASE_OUTPUT_DIR}/_repo_id_files"
mkdir -p "${REPO_ID_FILE_DIR}"
REPO_ID_FILE="${REPO_ID_FILE_DIR}/${JOB_NAME}.txt"
printf '%s\n' "${DATASET_REPO_IDS[@]}" > "${REPO_ID_FILE}"

config_path="xxx/checkpoints/last/pretrained_model/train_config.json"

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
    # --resume=true
    # --config_path=${config_path}

    --policy.type=${POLICY}
    --policy.repo_id=lerobot_lab/${POLICY}
    --policy.qwen3_vl_pretrained_path="${QWEN3_VL_PRETRAINED_PATH}"
    --policy.cosmos_tokenizer_path_or_name="${COSMOS_TOKENIZER_PATH_OR_NAME}"
    --policy.push_to_hub=false
    --policy.gradient_checkpointing=false
    --policy.dtype=bfloat16
    --policy.optimizer_lr=5.0e-5
    --policy.scheduler_warmup_steps=0
    --policy.scheduler_decay_steps=7_000_000
    --policy.scheduler_decay_lr=5.0e-5
    --policy.freeze_vision_encoder=false
    --policy.train_expert_only=false
    --policy.train_vlm_only=false
    --policy.qwen3_vl_variant=qwen3_vl_28l
    --policy.action_expert_variant=qwen3_28l
    --policy.enable_3d_queries=true
    --policy.num_3d_query_tokens=1296
    --policy.lambda_3d=0.05
    --policy.da3_model_path_or_name="${DA3_MODEL_PATH_OR_NAME}"
    --policy.da3_variant="${DA3_VARIANT}"

    --dataset.type=${POLICY}
    --dataset.repo_id="multidata_from_file"
    --dataset.repo_id_file="${REPO_ID_FILE}"
    --dataset.qwen3_vl_processor_path="${QWEN3_VL_PROCESSOR_PATH}"
    --dataset.action_mode="${ACTION_TYPE}"
    --dataset.use_external_stats=${USE_EXTERNAL_STATS}
    --dataset.dist_loading=true

    --seed=42
    --batch_size=16
    --steps=7_000_000
    --save_freq=10000
    --log_freq=200

    --wandb.enable=true
    --wandb.project=lerobot_lab_${POLICY}
    --wandb.mode=offline
)

if [[ -n "${DA3_CODE_ROOT}" ]]; then
    ARGS+=(--policy.da3_code_root="${DA3_CODE_ROOT}")
fi

if [[ -n "${DATASET_EXTERNAL_STATS_ROOT}" ]]; then
    ARGS+=(--dataset.external_stats_root="${DATASET_EXTERNAL_STATS_ROOT}")
fi

if [[ -n "${WEIGHT_RULES_PATH}" ]]; then
    ARGS+=(--dataset.weight_rules_path="${WEIGHT_RULES_PATH}")
fi

accelerate launch "${ARGS[@]}"
