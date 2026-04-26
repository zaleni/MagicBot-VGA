#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
echo "SCRIPT_DIR = ${SCRIPT_DIR}"
echo "PROJ_ROOT  = ${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}/src:${PYTHONPATH:-}"

cd "${PROJ_ROOT}"

DATASET_DIR="${DATASET_DIR:-/home/jiangjiahao/data/zhenji/table_clean_100_filter}"
DATASET_NAME="${DATASET_NAME:-$(basename "${DATASET_DIR}")}"
NORM_STATS_ROOT="${NORM_STATS_ROOT:-/home/jiangjiahao/data/zhenji/norm_stats}"
ACTION_TYPE="${ACTION_TYPE:-delta}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
OUTPUT_STATS_PATH="${NORM_STATS_ROOT}/${ACTION_TYPE}/${DATASET_NAME}/stats.json"

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "DATASET_DIR does not exist: ${DATASET_DIR}"
  exit 1
fi

if [[ ! -f "${DATASET_DIR}/meta/info.json" ]]; then
  echo "meta/info.json not found under DATASET_DIR: ${DATASET_DIR}"
  exit 1
fi

if [[ "${ACTION_TYPE}" != "delta" && "${ACTION_TYPE}" != "abs" ]]; then
  echo "ACTION_TYPE must be abs or delta, got ${ACTION_TYPE}"
  exit 1
fi

robot_type="$(
  python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["robot_type"])' \
    "${DATASET_DIR}/meta/info.json"
)"

if [[ "${robot_type}" != "real_lift2" ]]; then
  echo "Expected robot_type=real_lift2, got ${robot_type}"
  exit 1
fi

echo "DATASET_DIR=${DATASET_DIR}"
echo "DATASET_NAME=${DATASET_NAME}"
echo "robot_type=${robot_type}"
echo "ACTION_TYPE=${ACTION_TYPE}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "NORM_STATS_ROOT=${NORM_STATS_ROOT}"
echo "OUTPUT_STATS_PATH=${OUTPUT_STATS_PATH}"

python util_scripts/compute_norm_stats_single.py \
  --action_mode "${ACTION_TYPE}" \
  --chunk_size "${CHUNK_SIZE}" \
  --repo_id "${DATASET_DIR}" \
  --output_dir "${NORM_STATS_ROOT}"

if [[ ! -f "${OUTPUT_STATS_PATH}" ]]; then
  echo "Expected stats file was not created: ${OUTPUT_STATS_PATH}"
  exit 1
fi

echo "Wrote normalization stats: ${OUTPUT_STATS_PATH}"
