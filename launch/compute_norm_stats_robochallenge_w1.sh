#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
echo "SCRIPT_DIR = ${SCRIPT_DIR}"
echo "PROJ_ROOT  = ${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}/src:${PYTHONPATH:-}"

cd "${PROJ_ROOT}"

ROBOCHALLENGE_ROOT="${ROBOCHALLENGE_ROOT:-${DATASET_ROOT:-/inspire/qb-ilm/project/embodied-basic-model/zhangjianing-253108140206/DATASET/Robochallengev2_lerobotv3/Robochallenge_dos_w1_v3}}"
DATASET_DIR="${DATASET_DIR:-}"
DATASET_DIRS_FILE="${DATASET_DIRS_FILE:-}"

DEFAULT_ROBOCHALLENGE_W1_REGULAR_TASKS="fold_the_clothes tidy_up_the_makeup_table put_in_pen_container hold_the_tray_with_both_hands stack_bowls place_objects_into_desk_drawer"
DEFAULT_ROBOCHALLENGE_W1_EXTRA_TASKS="sweep_the_trash put_the_shoes_back tie_a_knot untie_the_shoelaces"
ROBOCHALLENGE_W1_REGULAR_TASKS="${ROBOCHALLENGE_W1_REGULAR_TASKS-${DEFAULT_ROBOCHALLENGE_W1_REGULAR_TASKS}}"
# Default W1 converted-data finetuning uses only the 6 regular tasks.
# Set ROBOCHALLENGE_W1_TASKS=all or explicitly set ROBOCHALLENGE_W1_EXTRA_TASKS
# if the 4 extra tasks should be included later.
ROBOCHALLENGE_W1_EXTRA_TASKS="${ROBOCHALLENGE_W1_EXTRA_TASKS-}"
ROBOCHALLENGE_W1_TASKS="${ROBOCHALLENGE_W1_TASKS-}"
if [[ -n "${ROBOCHALLENGE_W1_TASKS}" ]]; then
  if [[ "${ROBOCHALLENGE_W1_TASKS}" == "all" ]]; then
    ROBOCHALLENGE_W1_SELECTED_TASKS="${DEFAULT_ROBOCHALLENGE_W1_REGULAR_TASKS} ${DEFAULT_ROBOCHALLENGE_W1_EXTRA_TASKS}"
  else
    ROBOCHALLENGE_W1_SELECTED_TASKS="${ROBOCHALLENGE_W1_TASKS}"
  fi
else
  ROBOCHALLENGE_W1_SELECTED_TASKS="${ROBOCHALLENGE_W1_REGULAR_TASKS} ${ROBOCHALLENGE_W1_EXTRA_TASKS}"
fi
ROBOCHALLENGE_W1_TASK_SET="${ROBOCHALLENGE_W1_TASK_SET-}"
if [[ -z "${ROBOCHALLENGE_W1_TASK_SET}" ]]; then
  ROBOCHALLENGE_W1_TASK_SET="$(
    python - "${ROBOCHALLENGE_W1_SELECTED_TASKS}" "${DEFAULT_ROBOCHALLENGE_W1_REGULAR_TASKS}" "${DEFAULT_ROBOCHALLENGE_W1_EXTRA_TASKS}" <<'PY'
import hashlib
import sys

selected = tuple(name for name in sys.argv[1].split() if name)
regular = tuple(name for name in sys.argv[2].split() if name)
extra = tuple(name for name in sys.argv[3].split() if name)
if not selected:
    raise SystemExit("No RoboChallenge W1 tasks selected.")
if selected == regular + extra:
    print("all")
elif selected == regular:
    print("regular_only")
else:
    print("tasks_" + hashlib.sha1(",".join(selected).encode("utf-8")).hexdigest()[:8])
PY
  )"
fi

ACTION_TYPE="${ACTION_TYPE:-delta}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
MAX_CHUNKS_PER_EPISODE="${MAX_CHUNKS_PER_EPISODE:-}"
MAX_CHUNKS_PER_REPO="${MAX_CHUNKS_PER_REPO:-}"

NORM_STATS_ROOT="${NORM_STATS_ROOT:-outputs_robochallenge/norm_stats}"
if [[ "${ROBOCHALLENGE_W1_TASK_SET}" == "all" ]]; then
  DEFAULT_NORM_STATS_PATH="${NORM_STATS_ROOT}/robochallenge_w1/${ACTION_TYPE}/stats.json"
  DEFAULT_REPO_ID_FILE="${NORM_STATS_ROOT}/_repo_id_files/robochallenge_w1_${ACTION_TYPE}.txt"
else
  DEFAULT_NORM_STATS_PATH="${NORM_STATS_ROOT}/robochallenge_w1/${ROBOCHALLENGE_W1_TASK_SET}/${ACTION_TYPE}/stats.json"
  DEFAULT_REPO_ID_FILE="${NORM_STATS_ROOT}/_repo_id_files/robochallenge_w1_${ROBOCHALLENGE_W1_TASK_SET}_${ACTION_TYPE}.txt"
fi
NORM_STATS_PATH="${NORM_STATS_PATH:-${DEFAULT_NORM_STATS_PATH}}"
REPO_ID_FILE="${REPO_ID_FILE:-${DEFAULT_REPO_ID_FILE}}"

if [[ "${ACTION_TYPE}" != "delta" && "${ACTION_TYPE}" != "abs" ]]; then
  echo "ACTION_TYPE must be abs or delta, got ${ACTION_TYPE}"
  exit 1
fi

is_robochallenge_w1_info() {
  local info_path="$1"
  python - "${info_path}" <<'PY'
import json
import sys
from pathlib import Path

info = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
robot_type = str(info.get("robot_type", ""))
features = set((info.get("features") or {}).keys())
required = {
    "observation.state",
    "action",
    "observation.images.head",
    "observation.images.left",
    "observation.images.right",
}
raise SystemExit(0 if robot_type in {"dos_w1", "dos-w1", "DOS-W1"} and required.issubset(features) else 1)
PY
}

discover_dataset_dirs() {
  local root="$1"
  if [[ -z "${root}" || ! -d "${root}" ]]; then
    return 0
  fi

  while IFS= read -r -d '' info_path; do
    if is_robochallenge_w1_info "${info_path}"; then
      dirname "$(dirname "${info_path}")"
    fi
  done < <(find -L "${root}" -path "*/meta/info.json" -print0 2>/dev/null) | sort -u
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

filter_dataset_dirs_by_task_names() {
  local task_names="$1"
  shift
  python - "${task_names}" "$@" <<'PY'
import sys
from pathlib import Path

task_names = {name for name in sys.argv[1].split() if name}
if not task_names:
    raise SystemExit("No RoboChallenge W1 tasks selected.")

for dataset_dir in sys.argv[2:]:
    path = Path(dataset_dir)
    if path.name in task_names or task_names.intersection(path.parts):
        print(dataset_dir)
PY
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
    echo "Set ROBOCHALLENGE_ROOT to RoboChallenge_dos_w1_v3."
    exit 1
  fi
  mapfile -t DATASET_REPO_IDS < <(discover_dataset_dirs "${ROBOCHALLENGE_ROOT}")
fi

mapfile -t DATASET_REPO_IDS < <(filter_dataset_dirs_by_task_names "${ROBOCHALLENGE_W1_SELECTED_TASKS}" "${DATASET_REPO_IDS[@]}")

if [[ ${#DATASET_REPO_IDS[@]} -eq 0 ]]; then
  echo "No RoboChallenge W1 LeRobot datasets found for selected tasks: ${ROBOCHALLENGE_W1_SELECTED_TASKS}"
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
  if [[ "${resolved}" != "DOS-W1" ]]; then
    echo "Expected RoboChallenge DOS-W1 schema, got resolved_robot_type=${resolved} for ${ds_dir}"
    exit 1
  fi
done

mkdir -p "$(dirname "${REPO_ID_FILE}")" "$(dirname "${NORM_STATS_PATH}")"
printf '%s\n' "${DATASET_REPO_IDS[@]}" > "${REPO_ID_FILE}"

echo "DATASET_COUNT=${#DATASET_REPO_IDS[@]}"
echo "REPO_ID_FILE=${REPO_ID_FILE}"
echo "ROBOCHALLENGE_W1_TASK_SET=${ROBOCHALLENGE_W1_TASK_SET}"
echo "ROBOCHALLENGE_W1_SELECTED_TASKS=${ROBOCHALLENGE_W1_SELECTED_TASKS}"
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
