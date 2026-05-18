#!/usr/bin/env bash
set -euo pipefail

###############################################################################
################################# ENV config ##################################

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-6379}"
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

PROC_PER_NODE="${PROC_PER_NODE:-8}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

###############################################################################
############################### RESUME config #################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
echo "SCRIPT_DIR = ${SCRIPT_DIR}"
echo "PROJ_ROOT  = ${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}/src:${PYTHONPATH:-}"

cd "${PROJ_ROOT}"

POLICY="cubev2"

DEFAULT_RESUME_PRETRAINED_MODEL_DIR="/inspire/ssd/project/embodied-basic-model/zhangjianing-253108140206/MagicBot-VGA/outputs/cubev2/2nd-cubev2-robotwin-abs-chunk50-pretrained-causal-gen0.01-3d0.01-finetune-2026_05_17_08_37_26/checkpoints/060000/pretrained_model"

# You can override any of these, for example:
#   RESUME_CHECKPOINT_DIR=/path/to/run/checkpoints/080000 bash launch/cubev2/cubev2_finetune_robotwin_resume.sh
#   RESUME_RUN_DIR=/path/to/run RESUME_STEP=080000 bash launch/cubev2/cubev2_finetune_robotwin_resume.sh
RESUME_PRETRAINED_MODEL_DIR="${RESUME_PRETRAINED_MODEL_DIR:-${PRETRAINED_MODEL_DIR:-${DEFAULT_RESUME_PRETRAINED_MODEL_DIR}}}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-${RUN_DIR:-}}"
RESUME_STEP="${RESUME_STEP:-}"
RESUME_CHECKPOINT_DIR="${RESUME_CHECKPOINT_DIR:-${CHECKPOINT_DIR:-}}"
RESUME_CONFIG_PATH="${RESUME_CONFIG_PATH:-}"
RESUME_OUTPUT_DIR="${RESUME_OUTPUT_DIR:-${OUTPUT_DIR:-}}"

NUM_WORKERS="${NUM_WORKERS:-12}"
STEPS="${STEPS:-}"
SAVE_FREQ="${SAVE_FREQ:-}"
LOG_FREQ="${LOG_FREQ:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"

if [[ -n "${RESUME_RUN_DIR}" && -z "${RESUME_CHECKPOINT_DIR}" ]]; then
  if [[ -n "${RESUME_STEP}" ]]; then
    RESUME_CHECKPOINT_DIR="${RESUME_RUN_DIR%/}/checkpoints/${RESUME_STEP}"
  else
    RESUME_CHECKPOINT_DIR="${RESUME_RUN_DIR%/}/checkpoints/last"
  fi
fi

if [[ -z "${RESUME_CHECKPOINT_DIR}" && -n "${RESUME_PRETRAINED_MODEL_DIR}" ]]; then
  if [[ "$(basename "${RESUME_PRETRAINED_MODEL_DIR%/}")" == "pretrained_model" ]]; then
    RESUME_CHECKPOINT_DIR="$(dirname "${RESUME_PRETRAINED_MODEL_DIR%/}")"
  else
    echo "RESUME_PRETRAINED_MODEL_DIR should point to a pretrained_model directory:"
    echo "  ${RESUME_PRETRAINED_MODEL_DIR}"
    exit 1
  fi
fi

if [[ -n "${RESUME_CHECKPOINT_DIR}" && -z "${RESUME_CONFIG_PATH}" ]]; then
  RESUME_CONFIG_PATH="${RESUME_CHECKPOINT_DIR%/}/pretrained_model/train_config.json"
fi

if [[ -z "${RESUME_CONFIG_PATH}" ]]; then
  echo "Set RESUME_PRETRAINED_MODEL_DIR, RESUME_CHECKPOINT_DIR, RESUME_RUN_DIR, or RESUME_CONFIG_PATH."
  exit 1
fi

if [[ ! -f "${RESUME_CONFIG_PATH}" ]]; then
  echo "Resume config not found: ${RESUME_CONFIG_PATH}"
  exit 1
fi

if [[ -z "${RESUME_CHECKPOINT_DIR}" ]]; then
  RESUME_CHECKPOINT_DIR="$(dirname "$(dirname "${RESUME_CONFIG_PATH}")")"
fi

if [[ ! -d "${RESUME_CHECKPOINT_DIR%/}/training_state" ]]; then
  echo "Missing training_state under checkpoint: ${RESUME_CHECKPOINT_DIR}"
  echo "For true resume, use/copy the whole checkpoints/<step> directory, not only pretrained_model."
  exit 1
fi

if [[ -z "${RESUME_RUN_DIR}" ]]; then
  RESUME_RUN_DIR="$(dirname "$(dirname "${RESUME_CHECKPOINT_DIR%/}")")"
fi

RESUME_OUTPUT_DIR="${RESUME_OUTPUT_DIR:-${RESUME_RUN_DIR}}"

CONFIG_JOB_NAME="$(
  python -c 'import json,sys; cfg=json.load(open(sys.argv[1], encoding="utf-8")); print(cfg.get("job_name") or "")' \
    "${RESUME_CONFIG_PATH}"
)"
CONFIG_OUTPUT_DIR="$(
  python -c 'import json,sys; cfg=json.load(open(sys.argv[1], encoding="utf-8")); print(cfg.get("output_dir") or "")' \
    "${RESUME_CONFIG_PATH}"
)"
CONFIG_STEPS="$(
  python -c 'import json,sys; cfg=json.load(open(sys.argv[1], encoding="utf-8")); print(cfg.get("steps") or "")' \
    "${RESUME_CONFIG_PATH}"
)"
CONFIG_BATCH_SIZE="$(
  python -c 'import json,sys; cfg=json.load(open(sys.argv[1], encoding="utf-8")); print(cfg.get("batch_size") or "")' \
    "${RESUME_CONFIG_PATH}"
)"
CONFIG_GRADIENT_ACCUMULATION_STEPS="$(
  python -c 'import json,sys; cfg=json.load(open(sys.argv[1], encoding="utf-8")); print(cfg.get("gradient_accumulation_steps") or "")' \
    "${RESUME_CONFIG_PATH}"
)"
CONFIG_DATASET_REPO_ID="$(
  python -c 'import json,sys; cfg=json.load(open(sys.argv[1], encoding="utf-8")); print((cfg.get("dataset") or {}).get("repo_id") or "")' \
    "${RESUME_CONFIG_PATH}"
)"
CONFIG_REPO_ID_FILE="$(
  python -c 'import json,sys; cfg=json.load(open(sys.argv[1], encoding="utf-8")); print((cfg.get("dataset") or {}).get("repo_id_file") or "")' \
    "${RESUME_CONFIG_PATH}"
)"

JOB_NAME="${JOB_NAME:-${CONFIG_JOB_NAME}}"

RESUME_REPO_ID_FILE="${RESUME_REPO_ID_FILE:-${REPO_ID_FILE:-}}"
if [[ -z "${RESUME_REPO_ID_FILE}" && -n "${CONFIG_REPO_ID_FILE}" ]]; then
  if [[ "${CONFIG_REPO_ID_FILE}" == /* ]]; then
    repo_id_file_candidate="${CONFIG_REPO_ID_FILE}"
  else
    repo_id_file_candidate="$(dirname "${RESUME_RUN_DIR%/}")/_repo_id_files/$(basename "${CONFIG_REPO_ID_FILE}")"
  fi

  if [[ -f "${repo_id_file_candidate}" ]]; then
    RESUME_REPO_ID_FILE="${repo_id_file_candidate}"
  elif [[ -f "${CONFIG_REPO_ID_FILE}" ]]; then
    RESUME_REPO_ID_FILE="${CONFIG_REPO_ID_FILE}"
  fi
fi

if [[ "${CONFIG_DATASET_REPO_ID}" == "multidata_from_file" && -n "${CONFIG_REPO_ID_FILE}" && -z "${RESUME_REPO_ID_FILE}" ]]; then
  echo "Could not find dataset repo_id_file for resume."
  echo "Config repo_id_file: ${CONFIG_REPO_ID_FILE}"
  echo "Expected near run dir: $(dirname "${RESUME_RUN_DIR%/}")/_repo_id_files/$(basename "${CONFIG_REPO_ID_FILE}")"
  echo "Set RESUME_REPO_ID_FILE=/path/to/the/repo_id_file.txt if it lives elsewhere."
  exit 1
fi

RESUME_STEP_VALUE="<unknown>"
if [[ -f "${RESUME_CHECKPOINT_DIR%/}/training_state/training_step.json" ]]; then
  RESUME_STEP_VALUE="$(
    python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("step", "<unknown>"))' \
      "${RESUME_CHECKPOINT_DIR%/}/training_state/training_step.json"
  )"
fi

echo "RESUME=true"
echo "POLICY=${POLICY}"
echo "RESUME_PRETRAINED_MODEL_DIR=${RESUME_PRETRAINED_MODEL_DIR}"
echo "RESUME_CHECKPOINT_DIR=${RESUME_CHECKPOINT_DIR}"
echo "RESUME_CONFIG_PATH=${RESUME_CONFIG_PATH}"
echo "RESUME_STEP=${RESUME_STEP_VALUE}"
echo "CONFIG_OUTPUT_DIR=${CONFIG_OUTPUT_DIR}"
echo "RESUME_OUTPUT_DIR=${RESUME_OUTPUT_DIR}"
echo "JOB_NAME=${JOB_NAME:-<unset>}"
echo "CONFIG_STEPS=${CONFIG_STEPS:-<unset>}"
echo "CONFIG_BATCH_SIZE(per_device)=${CONFIG_BATCH_SIZE:-<unset>}"
echo "CONFIG_GRADIENT_ACCUMULATION_STEPS=${CONFIG_GRADIENT_ACCUMULATION_STEPS:-<unset>}"
echo "CONFIG_DATASET_REPO_ID=${CONFIG_DATASET_REPO_ID:-<unset>}"
echo "RESUME_REPO_ID_FILE=${RESUME_REPO_ID_FILE:-<config default>}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "NUM_WORKERS=${NUM_WORKERS}"

ARGS=(
    --multi_gpu
    --num_processes="${NUM_PROCESSES}"
    --num_machines="${NODE_COUNT}"
    --machine_rank="${NODE_RANK}"
    --main_process_ip="${MASTER_ADDR}"
    --main_process_port="${MASTER_PORT}"
    src/lerobot/scripts/lerobot_train.py

    --resume=true
    --config_path="${RESUME_CONFIG_PATH}"
    --output_dir="${RESUME_OUTPUT_DIR}"
    --num_workers="${NUM_WORKERS}"
)

if [[ -n "${JOB_NAME}" ]]; then
    ARGS+=(--job_name="${JOB_NAME}")
fi

if [[ -n "${STEPS}" ]]; then
    ARGS+=(--steps="${STEPS}")
fi

if [[ -n "${SAVE_FREQ}" ]]; then
    ARGS+=(--save_freq="${SAVE_FREQ}")
fi

if [[ -n "${LOG_FREQ}" ]]; then
    ARGS+=(--log_freq="${LOG_FREQ}")
fi

if [[ -n "${BATCH_SIZE}" ]]; then
    ARGS+=(--batch_size="${BATCH_SIZE}")
fi

if [[ -n "${GRADIENT_ACCUMULATION_STEPS}" ]]; then
    ARGS+=(--gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}")
fi

if [[ -n "${RESUME_REPO_ID_FILE}" ]]; then
    ARGS+=(--dataset.repo_id_file="${RESUME_REPO_ID_FILE}")
fi

accelerate launch "${ARGS[@]}" "$@"
