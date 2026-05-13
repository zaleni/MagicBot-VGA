#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ACTION_TYPE="${ACTION_TYPE:-abs}"
if [[ "${ACTION_TYPE}" != "abs" ]]; then
  echo "This launcher is for W1 abs finetuning. Got ACTION_TYPE=${ACTION_TYPE}."
  exit 1
fi

export POLICY_INIT_PATH="${POLICY_INIT_PATH:-${PRETRAINED_PATH:-/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/MagicBot-VGA/outputs_robochallenge/cubev2/cubev2-robochallenge_w1_from_raw25w-regular_only-delta-chunk50-finetune-2026_05_12_10_25_28/checkpoints/070000/pretrained_model}}"
export JOB_SOURCE_TAG="${JOB_SOURCE_TAG:-from_delta70k}"
export MASTER_PORT="${MASTER_PORT:-6690}"

# Keep regular-only W1 by default, matching the delta restart run this continues from.
export ROBOCHALLENGE_W1_EXTRA_TASKS="${ROBOCHALLENGE_W1_EXTRA_TASKS-}"

# The action target distribution changes from delta to abs, so use a slightly
# gentler restart unless overridden by the caller.
export OPTIMIZER_LR="${OPTIMIZER_LR:-3.0e-5}"
export SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-800}"
export SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-3.0e-6}"
export STEPS="${STEPS:-140000}"

exec bash "${SCRIPT_DIR}/cubev2_finetune_robochallenge_w1.sh"
