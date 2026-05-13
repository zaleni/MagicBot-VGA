#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ACTION_TYPE="${ACTION_TYPE:-abs}"
if [[ "${ACTION_TYPE}" != "abs" ]]; then
  echo "This launcher is for ALOHA abs finetuning. Got ACTION_TYPE=${ACTION_TYPE}."
  exit 1
fi

export POLICY_INIT_PATH="${POLICY_INIT_PATH:-${PRETRAINED_PATH:-/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/MagicBot-VGA/outputs_robochallenge/cubev2/cubev2-robochallenge_aloha-regular_only-delta-chunk50-restart-bs64-2026_05_10_18_01_44/checkpoints/220000/pretrained_model}}"
export JOB_NAME="${JOB_NAME:-cubev2-robochallenge_aloha-regular_only-abs-from_delta_restart220k-chunk50-finetune-$(date +'%Y_%m_%d_%H_%M_%S')}"
export MASTER_PORT="${MASTER_PORT:-6690}"
export SEED="${SEED:-442}"
export BATCH_SIZE="${BATCH_SIZE:-12}"

# Keep regular-only ALOHA by default, matching the delta restart run this
# continues from.
export ROBOCHALLENGE_ALOHA_EXTRA_TASKS="${ROBOCHALLENGE_ALOHA_EXTRA_TASKS-}"

# Match the W1 abs-from-delta launcher defaults.
export OPTIMIZER_LR="${OPTIMIZER_LR:-3.5e-5}"
export SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-900}"
export SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-3.5e-6}"
export STEPS="${STEPS:-120000}"
export ENABLE_IMAGE_AUG="${ENABLE_IMAGE_AUG:-true}"
export IMAGE_AUG_PRESET="${IMAGE_AUG_PRESET:-pi05}"
export WEIGHT_RULES_PATH="${WEIGHT_RULES_PATH:-configs/weight_rules_robochallenge_aloha_regular_gamma05.yaml}"

exec bash "${SCRIPT_DIR}/cubev2_finetune_robochallenge_aloha_restart.sh"
