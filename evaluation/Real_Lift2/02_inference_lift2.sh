#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

shell_type=${SHELL##*/}
shell_exec="exec $shell_type"

MAGICBOT_ROOT="${MAGICBOT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
REAL_LIFT_DIR="${REAL_LIFT_DIR:-${SCRIPT_DIR}}"
ROS2_LIFT_PLAY_ROOT="${ROS2_LIFT_PLAY_ROOT:-/home/arx/ROS2_LIFT_Play}"
LIFT_ROOT="${LIFT_ROOT:-$(dirname "${ROS2_LIFT_PLAY_ROOT}")/LIFT}"
CAN_ROOT="${CAN_ROOT:-${LIFT_ROOT}/ARX_CAN/arx_can}"
BODY_ROOT="${BODY_ROOT:-${LIFT_ROOT}/body/ROS2}"
X5_ROOT="${X5_ROOT:-${LIFT_ROOT}/ARX_X5/ROS2/X5_ws}"
REALSENSE_ROOT="${REALSENSE_ROOT:-${ROS2_LIFT_PLAY_ROOT}/realsense}"
REAL_LIFT2_RUNTIME_ROOT="${REAL_LIFT2_RUNTIME_ROOT:-${ROS2_LIFT_PLAY_ROOT}/act}"

RUN_SCRIPT="${RUN_SCRIPT:-${REAL_LIFT_DIR}/run_real_lift2_inference.sh}"
if [[ ! -f "${RUN_SCRIPT}" ]]; then
  echo "Could not find run_real_lift2_inference.sh under:"
  echo "  ${REAL_LIFT_DIR}"
  echo "Override with REAL_LIFT_DIR=/path/to/evaluation/Real_Lift2 or RUN_SCRIPT=/path/to/run_real_lift2_inference.sh"
  exit 1
fi

for required_dir in "${CAN_ROOT}" "${BODY_ROOT}" "${X5_ROOT}" "${REALSENSE_ROOT}"; do
  if [[ ! -d "${required_dir}" ]]; then
    echo "Expected directory does not exist:"
    echo "  ${required_dir}"
    echo "Override the corresponding *_ROOT variable before running this script."
    exit 1
  fi
done

RUN_ENV="${RUN_ENV:-act}"
WS_URL="${WS_URL:-ws://10.60.43.33:8000}"
PROMPT="${PROMPT:-Clear the junk and items off the desktop.}"
FRAME_RATE="${FRAME_RATE:-60}"
IMAGE_HISTORY_INTERVAL="${IMAGE_HISTORY_INTERVAL:-15}"
MAX_PUBLISH_STEP="${MAX_PUBLISH_STEP:-10000}"
RECORD_MODE="${RECORD_MODE:-Speed}"
USE_BASE="${USE_BASE:-true}"
FIXED_BODY_HEIGHT="${FIXED_BODY_HEIGHT:-16}"
GRIPPER_GATE="${GRIPPER_GATE:-}"
DATA_CONFIG="${DATA_CONFIG:-}"
STATE_DIM="${STATE_DIM:-14}"
ACTION_DIM="${ACTION_DIM:-14}"
PREFETCH_LEAD_STEPS="${PREFETCH_LEAD_STEPS:-10}"
LOG_TIMING_EVERY="${LOG_TIMING_EVERY:-5}"
SEED="${SEED:-0}"
SAFE_STOP_BODY_HEIGHT="${SAFE_STOP_BODY_HEIGHT:-0}"
SAFE_STOP_PUBLISH_STEPS="${SAFE_STOP_PUBLISH_STEPS:-30}"

# CAN
gnome-terminal --title="can1" -- bash -lc "cd '${CAN_ROOT}'; ./arx_can1.sh; exec bash"
sleep 0.3
gnome-terminal --title="can3" -- bash -lc "cd '${CAN_ROOT}'; ./arx_can3.sh; exec bash"
sleep 0.3
gnome-terminal --title="can5" -- bash -lc "cd '${CAN_ROOT}'; ./arx_can5.sh; exec bash"
sleep 0.3

# Body
gnome-terminal --title="body" -- $shell_type -i -c "cd '${BODY_ROOT}'; source install/setup.bash; ros2 launch arx_lift_controller lift.launch.py; $shell_exec"
sleep 1

# Lift
gnome-terminal --title="lift" -- $shell_type -i -c "cd '${X5_ROOT}'; source install/setup.bash; ros2 launch arx_x5_controller v2_joint_control.launch.py; $shell_exec"
sleep 1

# Realsense
gnome-terminal --title="realsense" -- $shell_type -i -c "cd '${REALSENSE_ROOT}'; ./realsense.sh; $shell_exec"
sleep 3

# MagicBot Real_Lift2 inference
gnome-terminal --title="magicbot-real-lift2" -- $shell_type -i -c "cd '${MAGICBOT_ROOT}'; source ~/.bashrc; conda activate '${RUN_ENV}'; REAL_LIFT2_RUNTIME_ROOT='${REAL_LIFT2_RUNTIME_ROOT}' WS_URL='${WS_URL}' PROMPT='${PROMPT}' FRAME_RATE='${FRAME_RATE}' IMAGE_HISTORY_INTERVAL='${IMAGE_HISTORY_INTERVAL}' MAX_PUBLISH_STEP='${MAX_PUBLISH_STEP}' RECORD_MODE='${RECORD_MODE}' USE_BASE='${USE_BASE}' FIXED_BODY_HEIGHT='${FIXED_BODY_HEIGHT}' GRIPPER_GATE='${GRIPPER_GATE}' DATA_CONFIG='${DATA_CONFIG}' STATE_DIM='${STATE_DIM}' ACTION_DIM='${ACTION_DIM}' PREFETCH_LEAD_STEPS='${PREFETCH_LEAD_STEPS}' LOG_TIMING_EVERY='${LOG_TIMING_EVERY}' SEED='${SEED}' SAFE_STOP_BODY_HEIGHT='${SAFE_STOP_BODY_HEIGHT}' SAFE_STOP_PUBLISH_STEPS='${SAFE_STOP_PUBLISH_STEPS}' bash '${RUN_SCRIPT}'; $shell_exec"
