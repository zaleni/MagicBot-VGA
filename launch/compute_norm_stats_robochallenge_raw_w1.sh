#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
echo "SCRIPT_DIR = ${SCRIPT_DIR}"
echo "PROJ_ROOT  = ${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}/src:${PYTHONPATH:-}"

cd "${PROJ_ROOT}"

DEFAULT_ROBOCHALLENGE_RAW_ROOT="/inspire/qb-ilm/project/embodied-basic-model/zhangjianing-253108140206/DATASET/Robochallenge_table30v2_unzipped"
ROBOCHALLENGE_RAW_ROOT="${ROBOCHALLENGE_RAW_ROOT:-${DATASET_ROOT:-${DEFAULT_ROBOCHALLENGE_RAW_ROOT}}}"
ACTION_TYPE="${ACTION_TYPE:-delta}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
FRAME_INTERVAL="${FRAME_INTERVAL:-1}"
TASK_PRESET="${TASK_PRESET:-table30v2_w1}"
WEIGHTED_TASK_STATS="${WEIGHTED_TASK_STATS:-true}"
REGULAR_TASK_WEIGHT="${REGULAR_TASK_WEIGHT:-1.0}"
EXTRA_TASK_WEIGHT="${EXTRA_TASK_WEIGHT:-0.8}"
TASK_REGEX="${TASK_REGEX:-}"
STATE_CACHE_DIR="${STATE_CACHE_DIR:-outputs_robochallenge/raw_cache/w1_states}"
NORM_STATS_ROOT="${NORM_STATS_ROOT:-outputs_robochallenge/norm_stats}"
NORM_STATS_PATH="${NORM_STATS_PATH:-${NORM_STATS_ROOT}/robochallenge_raw_w1/${TASK_PRESET}/${ACTION_TYPE}/chunk${CHUNK_SIZE}_frame${FRAME_INTERVAL}/stats.json}"

if [[ -z "${ROBOCHALLENGE_RAW_ROOT}" ]]; then
  echo "Set ROBOCHALLENGE_RAW_ROOT to the RoboChallenge raw root or a single DOS-W1 task directory."
  exit 1
fi

if [[ "${ACTION_TYPE}" != "delta" && "${ACTION_TYPE}" != "abs" ]]; then
  echo "ACTION_TYPE must be abs or delta, got ${ACTION_TYPE}"
  exit 1
fi

ARGS=(
  --raw-root "${ROBOCHALLENGE_RAW_ROOT}"
  --action-mode "${ACTION_TYPE}"
  --chunk-size "${CHUNK_SIZE}"
  --frame-interval "${FRAME_INTERVAL}"
  --task-preset "${TASK_PRESET}"
  --regular-task-weight "${REGULAR_TASK_WEIGHT}"
  --extra-task-weight "${EXTRA_TASK_WEIGHT}"
  --state-cache-dir "${STATE_CACHE_DIR}"
  --output-path "${NORM_STATS_PATH}"
)

if [[ "${WEIGHTED_TASK_STATS}" == "true" ]]; then
  ARGS+=(--weighted-task-stats)
else
  ARGS+=(--no-weighted-task-stats)
fi

if [[ -n "${TASK_REGEX}" ]]; then
  ARGS+=(--task-regex "${TASK_REGEX}")
fi

echo "ROBOCHALLENGE_RAW_ROOT=${ROBOCHALLENGE_RAW_ROOT}"
echo "ACTION_TYPE=${ACTION_TYPE}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "FRAME_INTERVAL=${FRAME_INTERVAL}"
echo "TASK_PRESET=${TASK_PRESET}"
echo "WEIGHTED_TASK_STATS=${WEIGHTED_TASK_STATS}"
echo "REGULAR_TASK_WEIGHT=${REGULAR_TASK_WEIGHT}"
echo "EXTRA_TASK_WEIGHT=${EXTRA_TASK_WEIGHT}"
echo "TASK_REGEX=${TASK_REGEX:-<all W1 tasks>}"
echo "STATE_CACHE_DIR=${STATE_CACHE_DIR}"
echo "NORM_STATS_PATH=${NORM_STATS_PATH}"

python util_scripts/compute_norm_stats_robochallenge_raw_w1.py "${ARGS[@]}"
