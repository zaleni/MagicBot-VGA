#!/usr/bin/env bash
#SBATCH --job-name=cubev2-libero
#SBATCH --nodes=2
#SBATCH -p hx
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=56
#SBATCH --mem-per-gpu=200G
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

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
LAUNCH_SCRIPT_PATH="${PROJ_ROOT}/launch/cubev2_finetune_libero.sh"

CONDA_SH_PATH="${CONDA_SH_PATH:-/HOME/uestc_jksong/uestc_jksong_1/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-magicbot-vga}"

# Cluster-side training values forwarded to launch/cubev2_finetune_libero.sh.
POLICY="cubev2"
BASE_OUTPUT_DIR="outputs/${POLICY}"
POLICY_INIT_PATH="/HOME/uestc_jksong/uestc_jksong_1/SSD_POOL/jjhao/model/MagicBot-VGA-Base"
QWEN3_VL_PRETRAINED_PATH="/HOME/uestc_jksong/uestc_jksong_1/SSD_POOL/jjhao/model/Qwen3-VL-2B-Instruct"
QWEN3_VL_PROCESSOR_PATH="${QWEN3_VL_PRETRAINED_PATH}"
COSMOS_TOKENIZER_PATH_OR_NAME="/HOME/uestc_jksong/uestc_jksong_1/SSD_POOL/jjhao/model/Cosmos-Tokenizer-CI8x8"
DA3_MODEL_PATH_OR_NAME="/HOME/uestc_jksong/uestc_jksong_1/SSD_POOL/jjhao/model/DA3-LARGE-1.1"
LIBERO_ROOT="/HOME/uestc_jksong/uestc_jksong_1/SSD_POOL/jjhao/data/libero_v30"
ACTION_TYPE="abs"
CHUNK_SIZE=10
N_ACTION_STEPS=10
ENABLE_3D_QUERIES=true
NUM_3D_QUERY_TOKENS=432
LAMBDA_3D=0.01
GEN_LAMBDA=0.002
USE_EXTERNAL_STATS=true
DATASET_EXTERNAL_STATS_PATH="/HOME/uestc_jksong/uestc_jksong_1/SSD_POOL/jjhao/data/norm_stats/libero_all_chunk10/franka/abs/stats.json"
BATCH_SIZE=8
GRAD_ACCUM_STEPS=1
STEPS=80000
SAVE_FREQ=10000
LOG_FREQ=25
RESUME="${RESUME:-false}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
RESUME_CONFIG_PATH="${RESUME_CONFIG_PATH:-}"
RESUME_CHECKPOINT_DIR="${RESUME_CHECKPOINT_DIR:-}"

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
export LIBERO_ROOT
export ACTION_TYPE
export CHUNK_SIZE
export N_ACTION_STEPS
export ENABLE_3D_QUERIES
export NUM_3D_QUERY_TOKENS
export LAMBDA_3D
export GEN_LAMBDA
export USE_EXTERNAL_STATS
export DATASET_EXTERNAL_STATS_PATH
export BATCH_SIZE
export GRAD_ACCUM_STEPS
export STEPS
export SAVE_FREQ
export LOG_FREQ
export RESUME
export RESUME_RUN_DIR
export RESUME_CONFIG_PATH
export RESUME_CHECKPOINT_DIR

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "NODE_COUNT=${NODE_COUNT}"
echo "PROC_PER_NODE=${PROC_PER_NODE}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR}"
echo "POLICY_INIT_PATH=${POLICY_INIT_PATH}"
echo "QWEN3_VL_PRETRAINED_PATH=${QWEN3_VL_PRETRAINED_PATH}"
echo "QWEN3_VL_PROCESSOR_PATH=${QWEN3_VL_PROCESSOR_PATH}"
echo "COSMOS_TOKENIZER_PATH_OR_NAME=${COSMOS_TOKENIZER_PATH_OR_NAME}"
echo "DA3_MODEL_PATH_OR_NAME=${DA3_MODEL_PATH_OR_NAME}"
echo "LIBERO_ROOT=${LIBERO_ROOT}"
echo "ACTION_TYPE=${ACTION_TYPE}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "N_ACTION_STEPS=${N_ACTION_STEPS}"
echo "ENABLE_3D_QUERIES=${ENABLE_3D_QUERIES}"
echo "NUM_3D_QUERY_TOKENS=${NUM_3D_QUERY_TOKENS}"
echo "LAMBDA_3D=${LAMBDA_3D}"
echo "GEN_LAMBDA=${GEN_LAMBDA}"
echo "USE_EXTERNAL_STATS=${USE_EXTERNAL_STATS}"
echo "DATASET_EXTERNAL_STATS_PATH=${DATASET_EXTERNAL_STATS_PATH}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}"
echo "STEPS=${STEPS}"
echo "SAVE_FREQ=${SAVE_FREQ}"
echo "LOG_FREQ=${LOG_FREQ}"
echo "RESUME=${RESUME}"
if [[ -n "${RESUME_RUN_DIR}" ]]; then
  echo "RESUME_RUN_DIR=${RESUME_RUN_DIR}"
fi
if [[ -n "${RESUME_CHECKPOINT_DIR}" ]]; then
  echo "RESUME_CHECKPOINT_DIR=${RESUME_CHECKPOINT_DIR}"
fi
if [[ -n "${RESUME_CONFIG_PATH}" ]]; then
  echo "RESUME_CONFIG_PATH=${RESUME_CONFIG_PATH}"
fi

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
