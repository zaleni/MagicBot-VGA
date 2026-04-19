# LIBERO Evaluation for CubeV2

This directory contains the local LIBERO evaluation entrypoint for a fine-tuned `CubeV2` checkpoint:

- `inference.py`: runs LIBERO rollouts with a local checkpoint
- `eval.sh`: thin launcher wrapper around `inference.py`

## What You Need

You do need a few extra runtime dependencies and local resources for LIBERO evaluation.

Required software:

- `mujoco`
- `LIBERO` installed from source with `pip install -e .`
- Python packages used by the eval script: `tyro`, `imageio`

Required local model resources:

- Your fine-tuned `CubeV2` checkpoint directory
- Local `Qwen3-VL` weights
- Local `Cosmos-Tokenizer-CI8x8` weights

Not required by default:

- `DA3` weights are not needed for eval, because `eval.sh` passes `--args.disable_3d_teacher_for_eval true`

## Install LIBERO

Example setup:

```bash
conda create -n libero_eval python=3.10 -y
conda activate libero_eval

pip install mujoco tyro imageio

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```

Or use the helper script in this repo:

```bash
bash evaluation/Libero/install_libero.sh
```

If you want the script to activate a conda env first:

```bash
CONDA_ENV=libero_eval bash evaluation/Libero/install_libero.sh
```

Optional but commonly needed on headless servers:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

Quick verification:

```bash
python -c "from libero.libero import benchmark; print('LIBERO OK')"
python -c "import mujoco; print('MuJoCo OK')"
```

## Do You Need Extra LIBERO Assets?

Usually, installing `LIBERO` from source is enough for evaluation, because the script reads task definitions via:

- `libero.libero.benchmark`
- `get_libero_path("bddl_files")`

So in the normal setup, you do not need to separately download LIBERO training datasets just to run evaluation.

## Do You Need Extra Model Files?

Yes, besides the fine-tuned checkpoint itself:

- `Qwen3-VL` is required at eval time
- `Cosmos tokenizer` is required at eval time

Why:

- `CubeV2` always reconstructs the Qwen3-VL backbone from `config.qwen3_vl_pretrained_path`
- `CubeV2` also instantiates the Cosmos tokenizer from `config.cosmos_tokenizer_path_or_name`

By default, `DA3` is disabled for eval in `eval.sh`, so you do not need `DA3` weights unless you intentionally turn that back on.

## Checkpoint Expectations

The eval script expects a checkpoint directory that contains:

```text
pretrained_model/
  config.json
  model.safetensors
  train_config.json
  stats.json
```

In practice, pass either:

- the `pretrained_model` directory directly, or
- a parent checkpoint directory that contains `pretrained_model/`

The script will resolve both forms automatically.

## Basic Usage

Evaluate one full suite:

```bash
PRETRAINED_CKPT=/path/to/checkpoints/last/pretrained_model \
TASK_SUITE_NAME=libero_goal \
QWEN3_VL_PRETRAINED_PATH=/path/to/Qwen3-VL-2B-Instruct \
COSMOS_TOKENIZER_PATH_OR_NAME=/path/to/Cosmos-Tokenizer-CI8x8 \
bash evaluation/Libero/eval.sh
```

Evaluate a single task:

```bash
PRETRAINED_CKPT=/path/to/checkpoints/last/pretrained_model \
TASK_SUITE_NAME=libero_goal \
TASK_ID=0 \
QWEN3_VL_PRETRAINED_PATH=/path/to/Qwen3-VL-2B-Instruct \
COSMOS_TOKENIZER_PATH_OR_NAME=/path/to/Cosmos-Tokenizer-CI8x8 \
bash evaluation/Libero/eval.sh
```

Choose a rollout horizon explicitly:

```bash
PRETRAINED_CKPT=/path/to/checkpoints/last/pretrained_model \
TASK_SUITE_NAME=libero_goal \
INFER_HORIZON=10 \
QWEN3_VL_PRETRAINED_PATH=/path/to/Qwen3-VL-2B-Instruct \
COSMOS_TOKENIZER_PATH_OR_NAME=/path/to/Cosmos-Tokenizer-CI8x8 \
bash evaluation/Libero/eval.sh
```

If `INFER_HORIZON` is not set, the script defaults to:

- `config.n_action_steps`, or
- `config.chunk_size`

which matches chunk-style execution.

## Important Runtime Arguments

Environment variables supported by `eval.sh`:

- `PRETRAINED_CKPT`: checkpoint path
- `TASK_SUITE_NAME`: one of `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`
- `TASK_ID`: optional single-task evaluation
- `NUM_TRIALS_PER_TASK`: default `50`
- `INFER_HORIZON`: optional action chunk execution length
- `VIDEO_DIR`: where rollout videos and summaries are written
- `STATS_KEY`: default `franka`
- `QWEN3_VL_PRETRAINED_PATH`: override Qwen weights path
- `QWEN3_VL_PROCESSOR_PATH`: optional separate processor path
- `COSMOS_TOKENIZER_PATH_OR_NAME`: override Cosmos tokenizer path
- `DA3_MODEL_PATH_OR_NAME`: only needed if you intentionally re-enable DA3 teacher for eval

## State / Action Convention

This eval path assumes the LIBERO convention already used in this repo:

- input state: 8D absolute state
  - `eef_pos(3) + axis_angle(3) + gripper_qpos(2)`
- output action: 7D action
  - `delta_xyz(3) + delta_rot(3) + gripper(1)`

The script normalizes `observation.state` and unnormalizes `action` using `stats.json` from the checkpoint.

## Gripper Note

Default:

```bash
gripper_mode=libero_open_prob
```

That means the 7th model output dimension is treated like an open/close probability and converted to LIBERO env control.

If your checkpoint was trained with a different gripper convention, the relevant logic is in:

- `evaluation/Libero/inference.py`

## Outputs

By default, outputs are written under:

```text
evaluation/Libero/output/<task_suite_name>/
```

Per-task directories contain:

- rollout videos
- optional action arrays
- `summary.json`

## Common Failure Modes

`ImportError: LIBERO is not available`

- install LIBERO from source in the active environment

`stats.json not found` or wrong `STATS_KEY`

- check that your checkpoint saved `pretrained_model/stats.json`
- for current LIBERO runs in this repo, `STATS_KEY=franka` is the expected default

Model path errors for Qwen or Cosmos

- pass explicit overrides through `eval.sh`

Headless rendering issues

- set `MUJOCO_GL=egl`
- set `PYOPENGL_PLATFORM=egl`

## Recommended Minimal Setup

If you already have a trained checkpoint, the minimum extra setup is usually:

1. Install `mujoco`
2. Clone and `pip install -e` LIBERO
3. Make sure local `Qwen3-VL` and `Cosmos` paths exist
4. Run `bash evaluation/Libero/eval.sh`
