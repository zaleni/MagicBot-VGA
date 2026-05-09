#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ACTION_TYPE="${ACTION_TYPE:-delta}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
ROBOCHALLENGE_ROOT="${ROBOCHALLENGE_ROOT:-${DATASET_ROOT:-/inspire/qb-ilm/project/embodied-basic-model/zhangjianing-253108140206/DATASET/Robochallengev2_starvla_lerobotv3}}"
NORM_STATS_ROOT="${NORM_STATS_ROOT:-outputs_robochallenge/norm_stats}"
DATASET_EXTERNAL_STATS_PATH="${DATASET_EXTERNAL_STATS_PATH:-${NORM_STATS_ROOT}/robochallenge_aloha_starvla/${ACTION_TYPE}/stats.json}"
MASTER_PORT="${MASTER_PORT:-6386}"
JOB_NAME="${JOB_NAME:-cubev2-robochallenge_aloha_starvla-${ACTION_TYPE}-chunk${CHUNK_SIZE}-finetune-$(date +'%Y_%m_%d_%H_%M_%S')}"

export ROBOCHALLENGE_ROOT
export ACTION_TYPE
export CHUNK_SIZE
export NORM_STATS_ROOT
export DATASET_EXTERNAL_STATS_PATH
export MASTER_PORT
export JOB_NAME

exec bash "${PROJ_ROOT}/launch/cubev2_finetune_robochallenge_aloha.sh"
