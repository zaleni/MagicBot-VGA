# RoboTwin Evaluation

This directory evaluates CubeV2 and MagicBot_R0 on RoboTwin. The shared evaluator is `evaluation/RoboTwin/inference.py`; the shell scripts only set checkpoints, stats, and job scheduling knobs.

## Entry Points

- `eval_randomized_50.sh`: CubeV2 randomized 50-task eval
- `eval_magicbot_50.sh`: MagicBot_R0 abs-action eval
- `eval_magicbot6B_delta.sh`: MagicBot_R0 delta-action eval
- `inference.py`: shared evaluator

## Requirements

- Linux with an NVIDIA GPU
- Python 3.10
- CUDA 12.8
- PyTorch 2.7.1
- `third_party/RoboTwin` checked out and its assets downloaded
- Vulkan runtime installed

For CubeV2, install `transformers==4.57.1` and patch the installed Qwen3-VL code with `src/lerobot/policies/cubev2/transformers_replace/models`.

## Quick Start

CubeV2:

```bash
PRETRAINED_CKPT=/path/to/cubev2/pretrained_model \
QWEN3_VL_PRETRAINED_PATH=/path/to/Qwen3-VL-2B-Instruct \
QWEN3_VL_PROCESSOR_PATH=/path/to/Qwen3-VL-2B-Instruct \
COSMOS_TOKENIZER_PATH_OR_NAME=/path/to/Cosmos-Tokenizer-CI8x8 \
DISABLE_DA3_TEACHER_FOR_EVAL=true \
GPU_IDS=0,1 \
MAX_JOBS_PER_GPU=2 \
bash evaluation/RoboTwin/eval_randomized_50.sh
```

MagicBot_R0:

```bash
PRETRAINED_CKPT=/path/to/MagicBot_R0/pretrained_model \
DISABLE_DA3_TEACHER_FOR_EVAL=true \
GPU_IDS=0,1 \
MAX_JOBS_PER_GPU=2 \
bash evaluation/RoboTwin/eval_magicbot_50.sh
```

For the delta-action variant, use `evaluation/RoboTwin/eval_magicbot6B_delta.sh`.

## Key Env Vars

Shared:

- `PRETRAINED_CKPT`
- `GPU_IDS`
- `MAX_JOBS_PER_GPU`
- `TASK_CONFIG`
- `START_TASK_IDX`
- `TASK_COUNT`
- `TEST_NUM`
- `ACTION_MODE`
- `STATS_KEY`
- `INFER_HORIZON`
- `BINARIZE_GRIPPER`
- `SKIP_GET_OBS_WITHIN_REPLAN`
- `DECODE_IMAGE_FLAG`
- `DISABLE_DA3_TEACHER_FOR_EVAL`

CubeV2 asset overrides:

- `QWEN3_VL_PRETRAINED_PATH`
- `QWEN3_VL_PROCESSOR_PATH`
- `COSMOS_TOKENIZER_PATH_OR_NAME`
- `DA3_MODEL_PATH_OR_NAME`
- `DA3_CODE_ROOT`

MagicBot_R0 asset overrides:

- `WAN_MODEL_ID`
- `WAN_TOKENIZER_MODEL_ID`
- `ACTION_DIT_PRETRAINED_PATH`
- `FUTURE_3D_PRETRAINED_PATH`
- `MAGICBOT_R0_LOAD_TEXT_ENCODER`
- `MAGICBOT_R0_REDIRECT_COMMON_FILES`
- `MAGICBOT_R0_SKIP_DIT_LOAD_FROM_PRETRAIN`
- `MAGICBOT_R0_STATS_PATH`

## Stats

- `MAGICBOT_R0_STATS_PATH` should normally stay empty
- the evaluator now prefers checkpoint `stats.json` first
- if that is missing, it falls back to the config-linked stats path instead of relying on a brittle relative path

## Outputs

Each run writes per-task logs under `evaluation/RoboTwin/output*/tasks/task_##/`, plus:

- `summary.json`
- `summary.txt`
- `job_status.tsv`

## Notes

- `ACTION_MODE` should match the checkpoint/training setup
- `STATS_KEY` is usually `aloha`
- If you need the CVPR 2026 11-task subset, use task ids `[2, 3, 9, 10, 12, 15, 17, 25, 28, 30, 44]`
