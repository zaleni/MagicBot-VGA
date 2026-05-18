#!/usr/bin/env bash
set -euo pipefail

###############################################################################
################################# ENV config ##################################

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-4545}"
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false

###############################################################################
############################## EVAL config ####################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
echo "SCRIPT_DIR = ${SCRIPT_DIR}"
echo "PROJ_ROOT  = ${PROJ_ROOT}"

cd "${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}/src:${PYTHONPATH:-}"

PRETRAINED_CKPT="${PRETRAINED_CKPT:-}"
if (( $# > 1 )); then
    echo "Usage:"
    echo "  bash evaluation/RoboTwin/${SCRIPT_NAME} [ckpt_dir_or_hf_repo_id]"
    echo "  PRETRAINED_CKPT=/path/to/pretrained_model bash evaluation/RoboTwin/${SCRIPT_NAME}"
    exit 1
fi

if (( $# == 1 )); then
    PRETRAINED_CKPT="$1"
fi

if [[ -z "${PRETRAINED_CKPT}" ]]; then
    echo "PRETRAINED_CKPT is empty."
    echo "Usage:"
    echo "  PRETRAINED_CKPT=/path/to/pretrained_model bash evaluation/RoboTwin/${SCRIPT_NAME}"
    echo "  bash evaluation/RoboTwin/${SCRIPT_NAME} /path/to/pretrained_model"
    exit 1
fi

POLICY_TYPE="${POLICY_TYPE:-qwenaction}"
QWEN3_VL_PRETRAINED_PATH="${QWEN3_VL_PRETRAINED_PATH:-/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/DATASET/model/Qwen3-VL-2B-Instruct}"
QWEN3_VL_PROCESSOR_PATH="${QWEN3_VL_PROCESSOR_PATH:-${QWEN3_VL_PRETRAINED_PATH}}"

BASE_OUTPUT_PATH="${BASE_OUTPUT_PATH:-${PROJ_ROOT}/evaluation/RoboTwin/output_qwenaction_50}"
TASK_CONFIG="${TASK_CONFIG:-demo_randomized}"
START_TASK_IDX="${START_TASK_IDX:-0}"
TASK_COUNT="${TASK_COUNT:-50}"
MAX_TASKS=50

GPU_IDS="${GPU_IDS:-0,1}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-3}"
POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-35}"

RESIZE_SIZE="${RESIZE_SIZE:-224}"
ACTION_MODE="${ACTION_MODE:-delta}"
TEST_NUM="${TEST_NUM:-100}"
SEED="${SEED:-42}"
STATS_KEY="${STATS_KEY:-aloha}"
DTYPE="${DTYPE:-bfloat16}"
IMAGE_HISTORY_INTERVAL="${IMAGE_HISTORY_INTERVAL:-15}"
INFER_HORIZON="${INFER_HORIZON:-24}"
ACTION_HORIZON_SIZE="${ACTION_HORIZON_SIZE:-50}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-unseen}"
LOG_LEVEL="${LOG_LEVEL:-WARNING}"

if [[ -e "${PRETRAINED_CKPT}" && ! -d "${PRETRAINED_CKPT}" ]]; then
    echo "PRETRAINED_CKPT exists but is not a directory: ${PRETRAINED_CKPT}"
    exit 1
fi

if (( START_TASK_IDX < 0 )); then
    echo "START_TASK_IDX must be >= 0, got ${START_TASK_IDX}"
    exit 1
fi

if (( TASK_COUNT <= 0 )); then
    echo "TASK_COUNT must be > 0, got ${TASK_COUNT}"
    exit 1
fi

if (( START_TASK_IDX + TASK_COUNT > MAX_TASKS )); then
    echo "Requested task range exceeds RoboTwin randomized task count (${MAX_TASKS})."
    echo "Got START_TASK_IDX=${START_TASK_IDX}, TASK_COUNT=${TASK_COUNT}"
    exit 1
fi

parse_gpu_ids() {
    local source_string=""
    if [[ -n "${GPU_IDS}" ]]; then
        source_string="${GPU_IDS// /}"
    elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        source_string="${CUDA_VISIBLE_DEVICES// /}"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        mapfile -t GPU_ID_ARRAY < <(nvidia-smi --query-gpu=index --format=csv,noheader)
        return
    else
        GPU_ID_ARRAY=("0")
        return
    fi

    IFS=',' read -r -a GPU_ID_ARRAY <<< "${source_string}"
}

parse_gpu_ids

if (( ${#GPU_ID_ARRAY[@]} == 0 )); then
    echo "No GPU ids resolved. Set GPU_IDS explicitly, for example GPU_IDS=0,1."
    exit 1
fi

if (( MAX_JOBS_PER_GPU <= 0 )); then
    echo "MAX_JOBS_PER_GPU must be > 0, got ${MAX_JOBS_PER_GPU}"
    exit 1
fi

CKPT_TAG="${CKPT_TAG:-qwenaction}"
RUN_NAME="${RUN_NAME:-${CKPT_TAG}-robotwin-$(date +%Y_%m_%d_%H_%M_%S)}"
RUN_OUTPUT_PATH="${BASE_OUTPUT_PATH}/${RUN_NAME}"
mkdir -p "${RUN_OUTPUT_PATH}/tasks"

TASK_END_IDX=$((START_TASK_IDX + TASK_COUNT - 1))

declare -a SLOT_GPU_IDS=()
for gpu_id in "${GPU_ID_ARRAY[@]}"; do
    for ((slot_repeat = 0; slot_repeat < MAX_JOBS_PER_GPU; slot_repeat++)); do
        SLOT_GPU_IDS+=("${gpu_id}")
    done
done

TOTAL_SLOTS=${#SLOT_GPU_IDS[@]}
declare -a SLOT_PIDS
declare -a SLOT_TASKS
declare -a SLOT_OUTPUT_DIRS
declare -a FAILED_TASKS=()

for ((slot_idx = 0; slot_idx < TOTAL_SLOTS; slot_idx++)); do
    SLOT_PIDS[slot_idx]=""
    SLOT_TASKS[slot_idx]=""
    SLOT_OUTPUT_DIRS[slot_idx]=""
done

{
    echo "script: ${SCRIPT_DIR}/${SCRIPT_NAME}"
    echo "pretrained_ckpt: ${PRETRAINED_CKPT}"
    echo "run_output_path: ${RUN_OUTPUT_PATH}"
    echo "task_config: ${TASK_CONFIG}"
    echo "task_range: ${START_TASK_IDX}-${TASK_END_IDX}"
    echo "task_count: ${TASK_COUNT}"
    echo "gpu_ids: ${GPU_ID_ARRAY[*]}"
    echo "max_jobs_per_gpu: ${MAX_JOBS_PER_GPU}"
    echo "policy_type: ${POLICY_TYPE}"
    echo "action_mode: ${ACTION_MODE}"
    echo "test_num: ${TEST_NUM}"
    echo "seed: ${SEED}"
    echo "resize_size: ${RESIZE_SIZE}"
    echo "stats_key: ${STATS_KEY}"
    echo "dtype: ${DTYPE}"
    echo "instruction_type: ${INSTRUCTION_TYPE}"
    echo "qwen3_vl_pretrained_path: ${QWEN3_VL_PRETRAINED_PATH}"
    echo "qwen3_vl_processor_path: ${QWEN3_VL_PROCESSOR_PATH}"
    echo "poll_interval_seconds: ${POLL_INTERVAL_SECONDS}"
} > "${RUN_OUTPUT_PATH}/launch_info.txt"

printf "task_idx\tgpu_id\texit_code\toutput_dir\n" > "${RUN_OUTPUT_PATH}/job_status.tsv"

cleanup() {
    for pid in "${SLOT_PIDS[@]}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
}

trap cleanup INT TERM

write_task_command_file() {
    local gpu_id="$1"
    local task_idx="$2"
    local task_output_dir="$3"

    local -a cmd=(
        python ../../evaluation/RoboTwin/inference.py
        --args.ckpt_path "${PRETRAINED_CKPT}"
        --args.video_dir "${task_output_dir}"
        --args.task_config "${TASK_CONFIG}"
        --args.task_idx "${task_idx}"
        --args.resize_size "${RESIZE_SIZE}"
        --args.action_mode "${ACTION_MODE}"
        --args.test_num "${TEST_NUM}"
        --args.seed "${SEED}"
        --args.stats_key "${STATS_KEY}"
        --args.dtype "${DTYPE}"
        --args.image_history_interval "${IMAGE_HISTORY_INTERVAL}"
        --args.infer_horizon "${INFER_HORIZON}"
        --args.action_horizon_size "${ACTION_HORIZON_SIZE}"
        --args.instruction_type "${INSTRUCTION_TYPE}"
        --args.log_level "${LOG_LEVEL}"
        --args.policy_type "${POLICY_TYPE}"
        --args.qwen3_vl_pretrained_path "${QWEN3_VL_PRETRAINED_PATH}"
        --args.qwen3_vl_processor_path "${QWEN3_VL_PROCESSOR_PATH}"
    )

    {
        printf 'CUDA_VISIBLE_DEVICES=%q ' "${gpu_id}"
        printf '%q ' "${cmd[@]}"
        printf '\n'
    } > "${task_output_dir}/command.txt"
}

launch_task() {
    local slot_idx="$1"
    local task_idx="$2"
    local gpu_id="${SLOT_GPU_IDS[slot_idx]}"
    local task_output_dir="${RUN_OUTPUT_PATH}/tasks/task_$(printf '%02d' "${task_idx}")"
    local task_log_path="${task_output_dir}/run.log"

    mkdir -p "${task_output_dir}"
    write_task_command_file "${gpu_id}" "${task_idx}" "${task_output_dir}"

    (
        set +e
        cd "${PROJ_ROOT}/third_party/RoboTwin"

        CMD=(
            python ../../evaluation/RoboTwin/inference.py
            --args.ckpt_path "${PRETRAINED_CKPT}"
            --args.video_dir "${task_output_dir}"
            --args.task_config "${TASK_CONFIG}"
            --args.task_idx "${task_idx}"
            --args.resize_size "${RESIZE_SIZE}"
            --args.action_mode "${ACTION_MODE}"
            --args.test_num "${TEST_NUM}"
            --args.seed "${SEED}"
            --args.stats_key "${STATS_KEY}"
            --args.dtype "${DTYPE}"
            --args.image_history_interval "${IMAGE_HISTORY_INTERVAL}"
            --args.infer_horizon "${INFER_HORIZON}"
            --args.action_horizon_size "${ACTION_HORIZON_SIZE}"
            --args.instruction_type "${INSTRUCTION_TYPE}"
            --args.log_level "${LOG_LEVEL}"
            --args.policy_type "${POLICY_TYPE}"
            --args.qwen3_vl_pretrained_path "${QWEN3_VL_PRETRAINED_PATH}"
            --args.qwen3_vl_processor_path "${QWEN3_VL_PROCESSOR_PATH}"
        )

        CUDA_VISIBLE_DEVICES="${gpu_id}" "${CMD[@]}" > "${task_log_path}" 2>&1
        exit_code=$?
        printf "%s\n" "${exit_code}" > "${task_output_dir}/exit_code.txt"
        exit "${exit_code}"
    ) &

    local pid=$!
    SLOT_PIDS[slot_idx]="${pid}"
    SLOT_TASKS[slot_idx]="${task_idx}"
    SLOT_OUTPUT_DIRS[slot_idx]="${task_output_dir}"

    echo "[launch] slot=${slot_idx} gpu=${gpu_id} task_idx=${task_idx} pid=${pid}"
}

reap_finished_slots() {
    local updated=0

    for ((slot_idx = 0; slot_idx < TOTAL_SLOTS; slot_idx++)); do
        local pid="${SLOT_PIDS[slot_idx]}"
        if [[ -z "${pid}" ]]; then
            continue
        fi

        if ! kill -0 "${pid}" 2>/dev/null; then
            local exit_code=0
            if wait "${pid}"; then
                exit_code=0
            else
                exit_code=$?
            fi

            local task_idx="${SLOT_TASKS[slot_idx]}"
            local gpu_id="${SLOT_GPU_IDS[slot_idx]}"
            local task_output_dir="${SLOT_OUTPUT_DIRS[slot_idx]}"
            local summary_path="${task_output_dir}/summary.json"

            if [[ "${exit_code}" -ne 0 && -f "${summary_path}" ]]; then
                echo "[warn] slot=${slot_idx} gpu=${gpu_id} task_idx=${task_idx} wrote summary despite exit_code=${exit_code}; treating as completed" >&2
                exit_code=0
                printf "%s\n" "${exit_code}" > "${task_output_dir}/exit_code.txt"
            fi

            printf "%s\t%s\t%s\t%s\n" "${task_idx}" "${gpu_id}" "${exit_code}" "${task_output_dir}" >> "${RUN_OUTPUT_PATH}/job_status.tsv"

            if [[ "${exit_code}" -ne 0 ]]; then
                FAILED_TASKS+=("${task_idx}")
                echo "[fail] slot=${slot_idx} gpu=${gpu_id} task_idx=${task_idx} exit_code=${exit_code}"
            else
                echo "[done] slot=${slot_idx} gpu=${gpu_id} task_idx=${task_idx}"
            fi

            SLOT_PIDS[slot_idx]=""
            SLOT_TASKS[slot_idx]=""
            SLOT_OUTPUT_DIRS[slot_idx]=""
            updated=1
        fi
    done

    return "${updated}"
}

find_free_slot() {
    for ((slot_idx = 0; slot_idx < TOTAL_SLOTS; slot_idx++)); do
        if [[ -z "${SLOT_PIDS[slot_idx]}" ]]; then
            echo "${slot_idx}"
            return 0
        fi
    done
    return 1
}

for ((task_idx = START_TASK_IDX; task_idx <= TASK_END_IDX; task_idx++)); do
    while true; do
        if free_slot="$(find_free_slot)"; then
            launch_task "${free_slot}" "${task_idx}"
            break
        fi
        reap_finished_slots || true
        sleep "${POLL_INTERVAL_SECONDS}"
    done
done

while true; do
    any_running=0
    for pid in "${SLOT_PIDS[@]}"; do
        if [[ -n "${pid}" ]]; then
            any_running=1
            break
        fi
    done

    if (( any_running == 0 )); then
        break
    fi

    reap_finished_slots || true
    sleep "${POLL_INTERVAL_SECONDS}"
done

echo "All requested tasks completed. Outputs: ${RUN_OUTPUT_PATH}"

if (( ${#FAILED_TASKS[@]} > 0 )); then
    echo "Failed task indices: ${FAILED_TASKS[*]}" >&2
    exit 1
fi
