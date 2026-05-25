# LIBERO Evaluation

This directory runs LIBERO with a split setup:

- `libero` env runs benchmark rollouts
- `magicbot` env serves policy inference over a local websocket

Both CubeV2 and MagicBot_R0 checkpoints are supported.

## Entry Points

- `01_serve_magicbot_libero.sh`: start the policy server
- `eval.sh`: run the LIBERO benchmark
- `inference.py`: main evaluation logic
- `model_server.py`: local policy server
- `websocket_server.py`, `websocket_client.py`, `libero_remote_client.py`: split-mode transport helpers

## Setup

Recommended:

- `libero` env: official LIBERO stack, plus `tyro`, `imageio`, `websockets`, and `msgpack`
- `magicbot` env: `tyro`, `matplotlib`, `mediapy`, `websockets`, and `msgpack`

For headless machines:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

If you want the helper path, run:

```bash
bash evaluation/Libero/install_libero.sh
```

## Quick Start

CubeV2:

```bash
conda activate magicbot
CHECKPOINT_DIR=/path/to/cubev2/pretrained_model \
STATS_KEY=franka \
ACTION_MODE=abs \
bash evaluation/Libero/01_serve_magicbot_libero.sh

conda activate libero
WS_URL=ws://127.0.0.1:8000 \
TASK_SUITE_NAME=libero_goal \
INFER_HORIZON=10 \
bash evaluation/Libero/eval.sh
```

MagicBot_R0:

```bash
conda activate magicbot
CHECKPOINT_DIR=/path/to/MagicBot_R0/checkpoints/200000/pretrained_model \
LOAD_TEXT_ENCODER=false \
TEXT_EMBED_CACHE_DIR=$PWD/outputs/MagicBot_R0/text_embeds/libero \
STATS_KEY=franka \
bash evaluation/Libero/01_serve_magicbot_libero.sh

conda activate libero
WS_URL=ws://127.0.0.1:8000 \
TASK_SUITE_NAME=libero_goal \
INFER_HORIZON=10 \
bash evaluation/Libero/eval.sh
```

Set `LOAD_TEXT_ENCODER=true` only if you want live text encoding instead of cached embeddings.

## Key Env Vars

Serve:

- `CHECKPOINT_DIR`
- `STATS_KEY`
- `STATS_PATH`
- `ACTION_MODE`
- `LOAD_TEXT_ENCODER`
- `TEXT_EMBED_CACHE_DIR`
- `TEXT_EMBED_CONTEXT_LEN`
- `QWEN3_VL_PRETRAINED_PATH`
- `QWEN3_VL_PROCESSOR_PATH`
- `COSMOS_TOKENIZER_PATH_OR_NAME`
- `DA3_MODEL_PATH_OR_NAME`
- `DA3_CODE_ROOT`

Eval:

- `PRETRAINED_CKPT`
- `WS_URL`
- `TASK_SUITE_NAME`
- `TASK_ID`
- `NUM_TRIALS_PER_TASK`
- `INFER_HORIZON`
- `VIDEO_ROOT`
- `VIDEO_DIR`

## Notes

- `stats.json` is read from `CHECKPOINT_DIR` by default
- `STATS_PATH` is optional and can override that lookup
- `ACTION_MODE` should match the checkpoint/training config
- Local single-process evaluation also works: set `PRETRAINED_CKPT` instead of `WS_URL`

## Outputs

Results are written under `evaluation/Libero/output/<task_suite_name>/` by default, including `summary.json` and per-task videos.

Dataset conversion is only needed for training. If you train on LIBERO, convert the v2.1 datasets to v3.0 and point `DATASET_EXTERNAL_STATS_PATH` or `DATASET_EXTERNAL_STATS_ROOT` to the merged stats.
