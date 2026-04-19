# LIBERO Evaluation for CubeV2

This repo recommends one setup for LIBERO evaluation:

- one machine
- two Python environments
- two processes
- the `MagicBot` environment serves policy inference
- the `LIBERO` environment runs benchmark rollouts
- both sides talk over a local websocket such as `ws://127.0.0.1:8000`

Relevant files:

- `evaluation/Libero/01_serve_magicbot_libero.sh`: start the policy server in the `MagicBot` env
- `evaluation/Libero/eval.sh`: start the LIBERO evaluator in the `LIBERO` env
- `evaluation/Libero/inference.py`: main LIBERO evaluation logic
- `evaluation/Libero/libero_remote_client.py`: websocket client used by the evaluator

## Install

Recommended layout: keep `LIBERO` at `third_party/LIBERO`.

You need dependencies on both sides:

- `LIBERO` eval env: official LIBERO stack plus this repo's evaluator extras
- `MagicBot` serve env: the original model-serving stack plus `websockets` and `msgpack`

Because this is now a separate LIBERO eval environment, it is fine to follow the official LIBERO install flow.

Recommended install:

```bash
conda activate <your_libero_eval_env>
cd third_party/LIBERO
pip install -r requirements.txt
pip install tyro imageio websockets msgpack
pip install -e .
```

If you want the exact upstream stack, follow `third_party/LIBERO/README.md` first, then add:

```bash
pip install tyro imageio websockets msgpack
```

Or use the helper script:

```bash
conda activate <your_libero_eval_env>
bash evaluation/Libero/install_libero.sh
```

If the `MagicBot` environment does not already have websocket serving deps, install:

```bash
conda activate <your_magicbot_env>
pip install websockets msgpack
```

Notes:

- In the split two-environment setup, installing the official `third_party/LIBERO/requirements.txt` is the recommended default.
- The old warning about avoiding upstream requirements only applies if you try to merge LIBERO into the `MagicBot` environment.
- `tyro`, `imageio`, `websockets`, and `msgpack` are extra deps used by this repo's evaluator path and are not part of upstream LIBERO requirements.
- `websockets` and `msgpack` are also required on the `MagicBot` serve side, because `01_serve_magicbot_libero.sh` reuses the websocket server from `evaluation/Real_Lift2`.
- If you intentionally want a lighter install instead of the official stack, use `bash evaluation/Libero/install_libero.sh` with `INSTALL_LIBERO_REQUIREMENTS=false INSTALL_MINIMAL_LIBERO_EVAL_DEPS=true`.

Quick checks:

```bash
python -c "from libero.libero import benchmark; print('LIBERO OK')"
python -c "from libero.libero.envs import OffScreenRenderEnv; import robosuite, bddl; print('LIBERO eval deps OK')"
python -c "import websockets.sync.client, msgpack; print('Split websocket deps OK')"
```

For headless machines, you may also need:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

## Run

1. Start the policy server in the `MagicBot` environment:

```bash
conda activate <your_magicbot_env>

PORT=8000 \
CHECKPOINT_DIR=/path/to/checkpoints/last/pretrained_model \
QWEN3_VL_PRETRAINED_PATH=/path/to/Qwen3-VL-2B-Instruct \
COSMOS_TOKENIZER_PATH_OR_NAME=/path/to/Cosmos-Tokenizer-CI8x8 \
INFER_HORIZON=10 \
bash evaluation/Libero/01_serve_magicbot_libero.sh
```

2. Start the LIBERO benchmark in the `LIBERO` environment:

```bash
conda activate <your_libero_eval_env>

WS_URL=ws://127.0.0.1:8000 \
TASK_SUITE_NAME=libero_goal \
INFER_HORIZON=10 \
VIDEO_ROOT=$PWD/evaluation/Libero/outputs \
bash evaluation/Libero/eval.sh
```

3. Evaluate a single task:

```bash
conda activate <your_libero_eval_env>

WS_URL=ws://127.0.0.1:8000 \
TASK_SUITE_NAME=libero_goal \
TASK_ID=0 \
INFER_HORIZON=10 \
bash evaluation/Libero/eval.sh
```

## Key Args

Serve side:

- `CHECKPOINT_DIR`: checkpoint or `pretrained_model` directory
- `QWEN3_VL_PRETRAINED_PATH`: local Qwen3-VL weights
- `COSMOS_TOKENIZER_PATH_OR_NAME`: local Cosmos tokenizer
- `INFER_HORIZON`: action chunk length
- `PORT`: defaults to `8000`
- `ACTION_MODE`: keep this aligned with training, for example `abs`

Eval side:

- `WS_URL`: local websocket address, usually `ws://127.0.0.1:8000`
- `TASK_SUITE_NAME`: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, or `libero_90`
- `TASK_ID`: optional single-task evaluation
- `NUM_TRIALS_PER_TASK`: defaults to `50`
- `INFER_HORIZON`: optional, otherwise follows the server
- `VIDEO_ROOT`: optional output root, final path becomes `<VIDEO_ROOT>/<TASK_SUITE_NAME>`
- `VIDEO_DIR`: output directory

## Outputs

Default output path:

```text
evaluation/Libero/output/<task_suite_name>/
```

If you only want to switch the outer folder, for example from `output` to `output_0420`, use `VIDEO_ROOT`:

```bash
conda activate <your_libero_eval_env>

WS_URL=ws://127.0.0.1:8000 \
TASK_SUITE_NAME=libero_goal \
INFER_HORIZON=10 \
VIDEO_ROOT=$PWD/evaluation/Libero/output_0420 \
bash evaluation/Libero/eval.sh
```

This writes to:

```text
evaluation/Libero/output_0420/libero_goal/
```

Each task directory contains:

- rollout videos
- optional action arrays
- `summary.json`

## Common Issues

`ImportError: LIBERO is not available`

- Make sure the current env already ran `pip install -e third_party/LIBERO`.
- Or set `LIBERO_HOME=/path/to/LIBERO`.

`ModuleNotFoundError: No module named 'robosuite'` or `No module named 'bddl'`

- Install the official LIBERO requirements: `pip install -r third_party/LIBERO/requirements.txt`
- Or use the lighter fallback install: `pip install pyyaml matplotlib bddl robosuite`

`ModuleNotFoundError: No module named 'websockets'` or `No module named 'msgpack'`

- Install the websocket deps: `pip install websockets msgpack`
- This can happen on either side: the `LIBERO` evaluator or the `MagicBot` policy server.

`stats.json not found` or wrong `STATS_KEY`

- Check that the checkpoint contains `pretrained_model/stats.json`
- Current LIBERO eval in this repo expects `STATS_KEY=franka`

Qwen or Cosmos path errors

- Pass `QWEN3_VL_PRETRAINED_PATH` and `COSMOS_TOKENIZER_PATH_OR_NAME` on the serve side
