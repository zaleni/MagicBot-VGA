#!/bin/bash
set -euo pipefail

workspace=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

rate=20
duration=3


usage() {
    echo "Usage: $0 [--rate HZ] [--duration SEC]"
    echo "Example: $0"
    echo "Example: $0 --rate 10 --duration 5"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rate)
            rate="${2:-}"
            shift 2
            ;;
        --duration)
            duration="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$rate" || -z "$duration" ]]; then
    echo "Invalid empty argument"
    usage
    exit 1
fi

# Avoid nounset conflict with some ROS setup scripts (e.g. COLCON_TRACE checks).
set +u
source "${workspace}/../../LIFT/ARX_X5/ROS2/X5_ws/install/setup.bash"
set -u

if ! command -v ros2 >/dev/null 2>&1; then
    echo "ros2 command not found after sourcing ROS setup"
    exit 1
fi

home_msg="{joint_pos: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], joint_vel: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], joint_cur: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], end_pos: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}"

echo "[home_arm] Send home command to both arms"
echo "[home_arm]rate=${rate}Hz, duration=${duration}s"

tmp_left="/tmp/home_arm_left.log"
tmp_right="/tmp/home_arm_right.log"

ros2 topic pub -r "${rate}" /arm_master_l_status arx5_arm_msg/msg/RobotStatus "${home_msg}" >"${tmp_left}" 2>&1 &
left_pid=$!

ros2 topic pub -r "${rate}" /arm_master_r_status arx5_arm_msg/msg/RobotStatus "${home_msg}" >"${tmp_right}" 2>&1 &
right_pid=$!

cleanup() {
    kill "${left_pid}" "${right_pid}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

sleep "${duration}"

echo "[home_arm] Done"
