#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
echo "SCRIPT_DIR = ${SCRIPT_DIR}"
echo "PROJ_ROOT  = ${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}/src:${PYTHONPATH:-}"

cd "${PROJ_ROOT}"

ROBOCHALLENGE_ROOT="${ROBOCHALLENGE_ROOT:-${DATASET_ROOT:-/inspire/qb-ilm/project/embodied-basic-model/zhangjianing-253108140206/DATASET/Robochallengev2_lerobotv3/Robochallenge_arx5_v3}}"
DATASET_DIR="${DATASET_DIR:-}"
DATASET_DIRS_FILE="${DATASET_DIRS_FILE:-}"
ARX5_TASK_DIRS="${ARX5_TASK_DIRS:-arrange_flowers_temp hang_the_cup pick_out_the_green_blocks press_the_button turn_on_the_light_switch water_the_flowers wipe_the_table}"

ACTION_TYPE="${ACTION_TYPE:-delta}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
MAX_CHUNKS_PER_EPISODE="${MAX_CHUNKS_PER_EPISODE:-}"
MAX_CHUNKS_PER_REPO="${MAX_CHUNKS_PER_REPO:-}"

NORM_STATS_ROOT="${NORM_STATS_ROOT:-outputs_robochallenge/norm_stats}"
NORM_STATS_PATH="${NORM_STATS_PATH:-${NORM_STATS_ROOT}/robochallenge_arx5/${ACTION_TYPE}/stats.json}"
REPO_ID_FILE="${REPO_ID_FILE:-${NORM_STATS_ROOT}/_repo_id_files/robochallenge_arx5_${ACTION_TYPE}.txt}"

if [[ "${ACTION_TYPE}" != "delta" && "${ACTION_TYPE}" != "abs" ]]; then
  echo "ACTION_TYPE must be abs or delta, got ${ACTION_TYPE}"
  exit 1
fi

is_robochallenge_arx5_info() {
  local info_path="$1"
  python - "${info_path}" <<'PY'
import json
import sys
from pathlib import Path

info = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
robot_type = str(info.get("robot_type", ""))
features = set((info.get("features") or {}).keys())
required_base = {
    "observation.state",
    "action",
}
camera_schemas = (
    {
        "observation.images.head",
        "observation.images.left",
        "observation.images.right",
    },
    {
        "observation.images.cam_global",
        "observation.images.cam_arm",
        "observation.images.cam_side",
    },
)
ok = (
    robot_type in {"arx5", "ARX5"}
    and required_base.issubset(features)
    and any(camera_schema.issubset(features) for camera_schema in camera_schemas)
)
raise SystemExit(0 if ok else 1)
PY
}

discover_dataset_dirs() {
  local root="$1"
  if [[ -z "${root}" || ! -d "${root}" ]]; then
    return 0
  fi

  local task=""
  for task in ${ARX5_TASK_DIRS}; do
    local ds_dir="${root%/}/${task}"
    local info_path="${ds_dir}/meta/info.json"
    if [[ ! -f "${info_path}" ]]; then
      echo "Expected ARX5 task directory is missing meta/info.json: ${ds_dir}" >&2
      exit 1
    fi
    if ! is_robochallenge_arx5_info "${info_path}"; then
      echo "Task directory does not match RoboChallenge ARX5 schema: ${ds_dir}" >&2
      exit 1
    fi
    echo "${ds_dir}"
  done
}

read_dataset_dirs_file() {
  local path="$1"
  local line=""
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%$'\r'}"
    if [[ -n "${line//[[:space:]]/}" ]]; then
      echo "${line}"
    fi
  done < "${path}"
}

declare -a DATASET_REPO_IDS=()
if [[ -n "${DATASET_DIRS_FILE}" ]]; then
  if [[ ! -f "${DATASET_DIRS_FILE}" ]]; then
    echo "DATASET_DIRS_FILE does not exist: ${DATASET_DIRS_FILE}"
    exit 1
  fi
  mapfile -t DATASET_REPO_IDS < <(read_dataset_dirs_file "${DATASET_DIRS_FILE}")
elif [[ -n "${DATASET_DIR}" ]]; then
  DATASET_REPO_IDS=("${DATASET_DIR}")
else
  if [[ -z "${ROBOCHALLENGE_ROOT}" ]]; then
    echo "Set ROBOCHALLENGE_ROOT to RoboChallenge_arx5_v3."
    exit 1
  fi
  mapfile -t DATASET_REPO_IDS < <(discover_dataset_dirs "${ROBOCHALLENGE_ROOT}")
fi

if [[ ${#DATASET_REPO_IDS[@]} -eq 0 ]]; then
  echo "No RoboChallenge ARX5 LeRobot datasets found."
  exit 1
fi

for ds_dir in "${DATASET_REPO_IDS[@]}"; do
  if [[ ! -f "${ds_dir}/meta/info.json" ]]; then
    echo "meta/info.json not found under dataset dir: ${ds_dir}"
    exit 1
  fi
  resolved="$(
    python - "${ds_dir}/meta/info.json" <<'PY'
import json
import sys
from pathlib import Path
from lerobot.transforms.constants import infer_embodiment_variant

info = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(infer_embodiment_variant(info["robot_type"], info.get("features", {})))
PY
  )"
  if [[ "${resolved}" != "ARX5" ]]; then
    echo "Expected RoboChallenge ARX5 schema, got resolved_robot_type=${resolved} for ${ds_dir}"
    exit 1
  fi
done

mkdir -p "$(dirname "${REPO_ID_FILE}")" "$(dirname "${NORM_STATS_PATH}")"
printf '%s\n' "${DATASET_REPO_IDS[@]}" > "${REPO_ID_FILE}"

echo "DATASET_COUNT=${#DATASET_REPO_IDS[@]}"
echo "REPO_ID_FILE=${REPO_ID_FILE}"
echo "ACTION_TYPE=${ACTION_TYPE}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "NORM_STATS_PATH=${NORM_STATS_PATH}"
printf '  - %s\n' "${DATASET_REPO_IDS[@]}"

EXTRA_ARGS=()
if [[ -n "${MAX_CHUNKS_PER_EPISODE}" ]]; then
  EXTRA_ARGS+=(--max_chunks_per_episode "${MAX_CHUNKS_PER_EPISODE}")
fi
if [[ -n "${MAX_CHUNKS_PER_REPO}" ]]; then
  EXTRA_ARGS+=(--max_chunks_per_repo "${MAX_CHUNKS_PER_REPO}")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  EXTRA_ARGS+=(--sample_seed "${SAMPLE_SEED}")
fi

python util_scripts/compute_norm_stats_multi.py \
  --action_mode "${ACTION_TYPE}" \
  --chunk_size "${CHUNK_SIZE}" \
  --repo_id_file "${REPO_ID_FILE}" \
  --num_workers "${NUM_WORKERS}" \
  --output_path "${NORM_STATS_PATH}" \
  "${EXTRA_ARGS[@]}"

if [[ ! -f "${NORM_STATS_PATH}" ]]; then
  echo "Expected stats file was not created: ${NORM_STATS_PATH}"
  exit 1
fi

echo "Wrote normalization stats: ${NORM_STATS_PATH}"
