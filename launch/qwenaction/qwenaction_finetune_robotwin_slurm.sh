#!/usr/bin/env bash
#SBATCH --job-name=qwenaction-robotwin
#SBATCH --nodes=1
#SBATCH -p hx
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=56
#SBATCH --mem-per-gpu=250G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_CLUSTER_PROJ_ROOT="/HOME/uestc_jksong/uestc_jksong_1/SSD_POOL/jjhao/MagicBot-VGA"
PROJ_ROOT="${PROJ_ROOT:-${CLUSTER_PROJ_ROOT:-}}"
if [[ -z "${PROJ_ROOT}" ]]; then
  if [[ -d "${DEFAULT_CLUSTER_PROJ_ROOT}" ]]; then
    PROJ_ROOT="${DEFAULT_CLUSTER_PROJ_ROOT}"
  else
    PROJ_ROOT="${LOCAL_PROJ_ROOT}"
  fi
fi
LAUNCH_SCRIPT_PATH="${PROJ_ROOT}/launch/qwenaction/qwenaction_finetune_robotwin.sh"

CONDA_SH_PATH="${CONDA_SH_PATH:-/HOME/uestc_jksong/uestc_jksong_1/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-magicbot-vga}"

POLICY="qwenaction"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs/${POLICY}}"
POLICY_INIT_PATH="${POLICY_INIT_PATH:-${PRETRAINED_PATH:-}}"
QWEN3_VL_PRETRAINED_PATH="${QWEN3_VL_PRETRAINED_PATH:-/HOME/uestc_jksong/uestc_jksong_1/SSD_POOL/jjhao/model/Qwen3-VL-2B-Instruct}"
QWEN3_VL_PROCESSOR_PATH="${QWEN3_VL_PROCESSOR_PATH:-${QWEN3_VL_PRETRAINED_PATH}}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/HOME/uestc_jksong/uestc_jksong_1/SSD_POOL/jjhao/data/RoboTwin-LeRobot-v30}"

ACTION_TYPE="${ACTION_TYPE:-delta}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
N_ACTION_STEPS="${N_ACTION_STEPS:-${CHUNK_SIZE}}"

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
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-${GRAD_ACCUM_STEPS:-1}}"
STEPS="${STEPS:-100000}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
SAVE_FREQ="${SAVE_FREQ:-20000}"
LOG_FREQ="${LOG_FREQ:-25}"
NUM_WORKERS="${NUM_WORKERS:-12}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
DDP_TIMEOUT_SEC="${DDP_TIMEOUT_SEC:-3600}"
export LEROBOT_DDP_TIMEOUT_SEC="${DDP_TIMEOUT_SEC}"

INIT_TAG="scratch"
if [[ -n "${POLICY_INIT_PATH}" ]]; then
  INIT_TAG="${BOOTSTRAP_TAG:-pretrained}"
fi
JOB_NAME="${JOB_NAME:-${POLICY}-robotwin-${ACTION_TYPE}-chunk${CHUNK_SIZE}-${INIT_TAG}-action_only-finetune-$(date +'%Y_%m_%d_%H_%M_%S')}"

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
export ROBOTWIN_ROOT
export ACTION_TYPE
export CHUNK_SIZE
export N_ACTION_STEPS
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
export GRADIENT_CHECKPOINTING
export JOB_NAME

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "NODE_COUNT=${NODE_COUNT}"
echo "PROC_PER_NODE=${PROC_PER_NODE}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR}"
echo "POLICY_INIT_PATH=${POLICY_INIT_PATH:-<scratch>}"
echo "QWEN3_VL_PRETRAINED_PATH=${QWEN3_VL_PRETRAINED_PATH}"
echo "QWEN3_VL_PROCESSOR_PATH=${QWEN3_VL_PROCESSOR_PATH}"
echo "ROBOTWIN_ROOT=${ROBOTWIN_ROOT}"
echo "ACTION_TYPE=${ACTION_TYPE}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "N_ACTION_STEPS=${N_ACTION_STEPS}"
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
echo "GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING}"
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
