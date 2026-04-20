#!/usr/bin/env bash
#SBATCH --job-name=cubev2-libero
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LAUNCH_SCRIPT_REL="launch/cubev2_finetune_libero.sh"

# Optional cluster-specific environment bootstrap.
if [[ -n "${ENV_SETUP_SCRIPT:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_SETUP_SCRIPT}"
fi

if [[ -n "${CONDA_SH_PATH:-}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH_PATH}"
fi

if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  conda activate "${CONDA_ENV_NAME}"
fi

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
export LAUNCH_SCRIPT_REL

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "NODE_COUNT=${NODE_COUNT}"
echo "PROC_PER_NODE=${PROC_PER_NODE}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"

srun --jobid "${SLURM_JOB_ID}" \
  --ntasks="${SLURM_NNODES}" \
  --ntasks-per-node=1 \
  --kill-on-bad-exit=1 \
  bash -lc '
    set -euo pipefail
    cd "${PROJ_ROOT}"
    export NODE_RANK="${SLURM_PROCID}"
    echo "Host=$(hostname) NODE_RANK=${NODE_RANK}/${NODE_COUNT} PROC_PER_NODE=${PROC_PER_NODE}"
    exec bash "${LAUNCH_SCRIPT_REL}"
  '
