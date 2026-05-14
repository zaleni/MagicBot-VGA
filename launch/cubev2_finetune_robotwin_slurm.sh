#!/usr/bin/env bash
#SBATCH --job-name=cubev2-robotwin
#SBATCH --nodes=1
#SBATCH -p hx
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=56
#SBATCH --mem-per-gpu=250G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Submit all four ablations with:
#   sbatch --array=0-3 launch/cubev2_finetune_robotwin_slurm.sh
# Or run one case with:
#   ABLATION_ID=4 sbatch launch/cubev2_finetune_robotwin_slurm.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_CLUSTER_PROJ_ROOT="/HOME/uestc_jksong/uestc_jksong_1/SSD_POOL/jjhao/MagicBot-VGA"
PROJ_ROOT="${PROJ_ROOT:-${CLUSTER_PROJ_ROOT:-}}"
if [[ -z "${PROJ_ROOT}" ]]; then
  if [[ -d "${DEFAULT_CLUSTER_PROJ_ROOT}" ]]; then
    PROJ_ROOT="${DEFAULT_CLUSTER_PROJ_ROOT}"
  else
    PROJ_ROOT="${LOCAL_PROJ_ROOT}"
  fi
fi
LAUNCH_SCRIPT_PATH="${PROJ_ROOT}/launch/cubev2_finetune_robotwin.sh"

CONDA_SH_PATH="${CONDA_SH_PATH:-/HOME/uestc_jksong/uestc_jksong_1/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-magicbot-vga}"

# Empty POLICY_INIT_PATH means Qwen3-VL initialization plus randomly initialized
# CubeV2 action/generation/3D heads. Set POLICY_INIT_PATH/PRETRAINED_PATH to
# continue from a full CubeV2 checkpoint instead.
POLICY="cubev2"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs/${POLICY}}"
POLICY_INIT_PATH="${POLICY_INIT_PATH:-${PRETRAINED_PATH:-}}"
QWEN3_VL_PRETRAINED_PATH="${QWEN3_VL_PRETRAINED_PATH:-/HOME/uestc_jksong/uestc_jksong_1/SSD_POOL/jjhao/model/Qwen3-VL-2B-Instruct}"
QWEN3_VL_PROCESSOR_PATH="${QWEN3_VL_PROCESSOR_PATH:-${QWEN3_VL_PRETRAINED_PATH}}"
COSMOS_TOKENIZER_PATH_OR_NAME="${COSMOS_TOKENIZER_PATH_OR_NAME:-/HOME/uestc_jksong/uestc_jksong_1/SSD_POOL/jjhao/model/Cosmos-Tokenizer-CI8x8}"
DA3_MODEL_PATH_OR_NAME="${DA3_MODEL_PATH_OR_NAME:-/HOME/uestc_jksong/uestc_jksong_1/SSD_POOL/jjhao/model/DA3-LARGE-1.1}"
DA3_VARIANT="${DA3_VARIANT:-auto}"
DA3_ALIGNMENT_MODE="${DA3_ALIGNMENT_MODE:-query_decoder}"
DA3_CODE_ROOT="${DA3_CODE_ROOT:-}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/HOME/uestc_jksong/uestc_jksong_1/SSD_POOL/jjhao/data/RoboTwin-LeRobot-v30}"

ACTION_TYPE="${ACTION_TYPE:-delta}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-${CHUNK_SIZE}}"
NUM_3D_QUERY_TOKENS="${NUM_3D_QUERY_TOKENS:-432}"

USE_EXTERNAL_STATS="${USE_EXTERNAL_STATS:-true}"
DATASET_EXTERNAL_STATS_PATH="${DATASET_EXTERNAL_STATS_PATH:-}"
DATASET_EXTERNAL_STATS_ROOT="${DATASET_EXTERNAL_STATS_ROOT:-}"
if [[ -z "${DATASET_EXTERNAL_STATS_PATH}" && -z "${DATASET_EXTERNAL_STATS_ROOT}" ]]; then
  DATASET_EXTERNAL_STATS_PATH="/HOME/uestc_jksong/uestc_jksong_1/SSD_POOL/jjhao/data/norm_stats/aloha/${ACTION_TYPE}/stats.json"
fi

WEIGHT_RULES_PATH="${WEIGHT_RULES_PATH:-}"
USE_DIST_LOADING="${USE_DIST_LOADING:-true}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
STEPS="${STEPS:-100000}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
LOG_FREQ="${LOG_FREQ:-25}"
NUM_WORKERS="${NUM_WORKERS:-12}"
DDP_TIMEOUT_SEC="${DDP_TIMEOUT_SEC:-3600}"
export LEROBOT_DDP_TIMEOUT_SEC="${DDP_TIMEOUT_SEC}"

ABLATION_CASE="${ABLATION_CASE:-}"
if [[ -z "${ABLATION_CASE}" && -n "${ABLATION_ID:-}" ]]; then
  ABLATION_CASE="${ABLATION_ID}"
fi
if [[ -z "${ABLATION_CASE}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  if [[ "${SLURM_ARRAY_TASK_ID}" == "0" || "${SLURM_ARRAY_TASK_MIN:-}" == "0" ]]; then
    ABLATION_CASE="$((SLURM_ARRAY_TASK_ID + 1))"
  else
    ABLATION_CASE="${SLURM_ARRAY_TASK_ID}"
  fi
fi

if [[ -n "${ABLATION_CASE}" ]]; then
  case "${ABLATION_CASE}" in
    1|action|action_only|qwen_action)
      ABLATION_NAME="qwen-action"
      LAMBDA_GEN=0
      LAMBDA_3D=0
      ENABLE_3D_QUERIES=false
      CUBEV2_ATTENTION_MASK_MODE=default
      ;;
    2|gen|action_gen|qwen_action_gen)
      ABLATION_NAME="action-gen"
      LAMBDA_GEN=0.01
      LAMBDA_3D=0
      ENABLE_3D_QUERIES=false
      CUBEV2_ATTENTION_MASK_MODE=default
      ;;
    3|3d|action_3d|qwen_action_3d)
      ABLATION_NAME="action-3d"
      LAMBDA_GEN=0
      LAMBDA_3D=0.01
      ENABLE_3D_QUERIES=true
      CUBEV2_ATTENTION_MASK_MODE=default
      ;;
    4|full|causal|action_gen_3d_causal)
      ABLATION_NAME="full-causal"
      LAMBDA_GEN=0.01
      LAMBDA_3D=0.01
      ENABLE_3D_QUERIES=true
      CUBEV2_ATTENTION_MASK_MODE=causal
      ;;
    *)
      echo "Unknown ABLATION_CASE=${ABLATION_CASE}. Use 1-4 or sbatch --array=0-3."
      exit 1
      ;;
  esac
else
  ABLATION_NAME="${ABLATION_NAME:-manual}"
  LAMBDA_GEN="${LAMBDA_GEN:-0.01}"
  LAMBDA_3D="${LAMBDA_3D:-0.01}"
  ENABLE_3D_QUERIES="${ENABLE_3D_QUERIES:-true}"
  CUBEV2_ATTENTION_MASK_MODE="${CUBEV2_ATTENTION_MASK_MODE:-causal}"
fi

if [[ "${CUBEV2_ATTENTION_MASK_MODE}" == "casual" ]]; then
  CUBEV2_ATTENTION_MASK_MODE="causal"
fi

JOB_NAME="${JOB_NAME:-${POLICY}-robotwin-${ABLATION_NAME}-${ACTION_TYPE}-chunk${CHUNK_SIZE}-finetune-$(date +'%Y_%m_%d_%H_%M_%S')}"

# Optional cluster-specific environment bootstrap.
if [[ -n "${ENV_SETUP_SCRIPT:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SETUP_SCRIPT}"
fi

if [[ ! -f "${CONDA_SH_PATH}" ]]; then
  echo "conda.sh not found: ${CONDA_SH_PATH}"
  echo "Set CONDA_SH_PATH explicitly before calling sbatch."
  exit 1
fi

# shellcheck disable=SC1090
source "${CONDA_SH_PATH}"
conda activate "${CONDA_ENV_NAME}"

if [[ ! -d "${PROJ_ROOT}" ]]; then
  echo "Project root does not exist: ${PROJ_ROOT}"
  echo "Set PROJ_ROOT or CLUSTER_PROJ_ROOT explicitly before calling sbatch."
  exit 1
fi

if [[ ! -f "${LAUNCH_SCRIPT_PATH}" ]]; then
  echo "Launch script not found: ${LAUNCH_SCRIPT_PATH}"
  exit 1
fi

if [[ "${ACTION_TYPE}" != "delta" && "${ACTION_TYPE}" != "abs" ]]; then
  echo "ACTION_TYPE must be abs or delta, got ${ACTION_TYPE}"
  exit 1
fi

cd "${PROJ_ROOT}"
echo "Current working directory: $(pwd)"

GPUS_PER_NODE_RAW="${PROC_PER_NODE:-${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}}"
if [[ "${GPUS_PER_NODE_RAW}" =~ ([0-9]+) ]]; then
  export PROC_PER_NODE="${BASH_REMATCH[1]}"
else
  echo "Unable to parse GPU count from GPUS_PER_NODE=${GPUS_PER_NODE_RAW}"
  echo "Set PROC_PER_NODE or GPUS_PER_NODE explicitly before calling sbatch."
  exit 1
fi

export NODE_COUNT="${SLURM_NNODES}"
export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + RANDOM % 10000))}"

export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
# Uncomment and adjust these on clusters that require a fixed NIC / IB device.
# export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
# export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_2,mlx5_3}"

export PROJ_ROOT
export LAUNCH_SCRIPT_PATH
export BASE_OUTPUT_DIR
export POLICY_INIT_PATH
export QWEN3_VL_PRETRAINED_PATH
export QWEN3_VL_PROCESSOR_PATH
export COSMOS_TOKENIZER_PATH_OR_NAME
export DA3_MODEL_PATH_OR_NAME
export DA3_VARIANT
export DA3_ALIGNMENT_MODE
export DA3_CODE_ROOT
export ROBOTWIN_ROOT
export ACTION_TYPE
export CHUNK_SIZE
export N_ACTION_STEPS
export ENABLE_3D_QUERIES
export NUM_3D_QUERY_TOKENS
export CUBEV2_ATTENTION_MASK_MODE
export LAMBDA_GEN
export LAMBDA_3D
export USE_EXTERNAL_STATS
export DATASET_EXTERNAL_STATS_PATH
export DATASET_EXTERNAL_STATS_ROOT
export WEIGHT_RULES_PATH
export USE_DIST_LOADING
export VIDEO_BACKEND
export BATCH_SIZE
export GRADIENT_ACCUMULATION_STEPS
export STEPS
export WARMUP_STEPS
export SAVE_FREQ
export LOG_FREQ
export NUM_WORKERS
export JOB_NAME

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-<none>}"
echo "ABLATION_CASE=${ABLATION_CASE:-<manual>}"
echo "ABLATION_NAME=${ABLATION_NAME}"
echo "NODE_COUNT=${NODE_COUNT}"
echo "PROC_PER_NODE=${PROC_PER_NODE}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR}"
echo "POLICY_INIT_PATH=${POLICY_INIT_PATH:-<scratch>}"
echo "QWEN3_VL_PRETRAINED_PATH=${QWEN3_VL_PRETRAINED_PATH}"
echo "QWEN3_VL_PROCESSOR_PATH=${QWEN3_VL_PROCESSOR_PATH}"
echo "COSMOS_TOKENIZER_PATH_OR_NAME=${COSMOS_TOKENIZER_PATH_OR_NAME}"
echo "DA3_MODEL_PATH_OR_NAME=${DA3_MODEL_PATH_OR_NAME}"
echo "ROBOTWIN_ROOT=${ROBOTWIN_ROOT}"
echo "ACTION_TYPE=${ACTION_TYPE}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "N_ACTION_STEPS=${N_ACTION_STEPS}"
echo "ENABLE_3D_QUERIES=${ENABLE_3D_QUERIES}"
echo "NUM_3D_QUERY_TOKENS=${NUM_3D_QUERY_TOKENS}"
echo "CUBEV2_ATTENTION_MASK_MODE=${CUBEV2_ATTENTION_MASK_MODE}"
echo "LAMBDA_GEN=${LAMBDA_GEN}"
echo "LAMBDA_3D=${LAMBDA_3D}"
echo "USE_EXTERNAL_STATS=${USE_EXTERNAL_STATS}"
echo "DATASET_EXTERNAL_STATS_PATH=${DATASET_EXTERNAL_STATS_PATH}"
echo "DATASET_EXTERNAL_STATS_ROOT=${DATASET_EXTERNAL_STATS_ROOT}"
echo "WEIGHT_RULES_PATH=${WEIGHT_RULES_PATH:-<none>}"
echo "USE_DIST_LOADING=${USE_DIST_LOADING}"
echo "VIDEO_BACKEND=${VIDEO_BACKEND}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
echo "STEPS=${STEPS}"
echo "WARMUP_STEPS=${WARMUP_STEPS}"
echo "SAVE_FREQ=${SAVE_FREQ}"
echo "LOG_FREQ=${LOG_FREQ}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "JOB_NAME=${JOB_NAME}"

srun --jobid "${SLURM_JOB_ID}" \
  --ntasks="${SLURM_NNODES}" \
  --ntasks-per-node=1 \
  --kill-on-bad-exit=1 \
  bash -lc '
    set -euo pipefail
    cd "${PROJ_ROOT}"
    export NODE_RANK="${SLURM_PROCID}"
    echo "Host=$(hostname) NODE_RANK=${NODE_RANK}/${NODE_COUNT} PROC_PER_NODE=${PROC_PER_NODE}"
    exec bash "${LAUNCH_SCRIPT_PATH}"
  '
