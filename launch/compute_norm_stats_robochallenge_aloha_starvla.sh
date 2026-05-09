#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

ACTION_TYPE="${ACTION_TYPE:-delta}"
ROBOCHALLENGE_ROOT="${ROBOCHALLENGE_ROOT:-${DATASET_ROOT:-/inspire/qb-ilm/project/embodied-basic-model/zhangjianing-253108140206/DATASET/Robochallengev2_starvla_lerobotv3}}"
NORM_STATS_ROOT="${NORM_STATS_ROOT:-outputs_robochallenge/norm_stats}"
NORM_STATS_PATH="${NORM_STATS_PATH:-${NORM_STATS_ROOT}/robochallenge_aloha_starvla/${ACTION_TYPE}/stats.json}"
REPO_ID_FILE="${REPO_ID_FILE:-${NORM_STATS_ROOT}/_repo_id_files/robochallenge_aloha_starvla_${ACTION_TYPE}.txt}"

export ROBOCHALLENGE_ROOT
export ACTION_TYPE
export NORM_STATS_ROOT
export NORM_STATS_PATH
export REPO_ID_FILE

exec bash "${PROJ_ROOT}/launch/compute_norm_stats_robochallenge_aloha.sh"
