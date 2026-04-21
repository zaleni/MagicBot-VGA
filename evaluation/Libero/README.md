# LIBERO Evaluation for CubeV2

This repo recommends one setup for LIBERO evaluation:

- one machine
- two Python environments
- two processes
- the `magicbot` environment serves policy inference
- the `libero` environment runs benchmark rollouts
- both sides talk over a local websocket such as `ws://127.0.0.1:8000`

Relevant files:

- `evaluation/Libero/01_serve_magicbot_libero.sh`: start the policy server in the `magicbot` env
- `evaluation/Libero/eval.sh`: start the LIBERO evaluator in the `libero` env
- `evaluation/Libero/inference.py`: main LIBERO evaluation logic
- `evaluation/Libero/model_server.py`: LIBERO-local policy server entrypoint
- `evaluation/Libero/websocket_server.py`: LIBERO-local websocket server helper
- `evaluation/Libero/libero_remote_client.py`: websocket client used by the evaluator
- `evaluation/Libero/websocket_client.py`: LIBERO-local websocket transport helper
- `evaluation/Libero/msgpack_numpy.py`: LIBERO-local msgpack NumPy codec helper

## Install

Recommended layout: keep `LIBERO` at `third_party/LIBERO`.

You need dependencies on both sides:

- `LIBERO` eval env: official LIBERO stack plus this repo's evaluator extras
- `magicbot` serve env: the original model-serving stack plus `tyro`, `matplotlib`, `mediapy`, `websockets`, and `msgpack`

Because this is now a separate LIBERO eval environment, it is fine to follow the official LIBERO install flow.

Recommended install:

```bash
conda activate libero
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
conda activate libero
bash evaluation/Libero/install_libero.sh
```

If the `magicbot` environment does not already have the extra serve-side deps, install:

```bash
conda activate magicbot
pip install tyro matplotlib mediapy websockets msgpack
```

Notes:

- In the split two-environment setup, installing the official `third_party/LIBERO/requirements.txt` is the recommended default.
- The old warning about avoiding upstream requirements only applies if you try to merge LIBERO into the `magicbot` environment.
- `tyro`, `imageio`, `websockets`, and `msgpack` are extra deps used by this repo's evaluator path and are not part of upstream LIBERO requirements.
- `tyro`, `matplotlib`, `mediapy`, `websockets`, and `msgpack` are also required on the `magicbot` serve side.
- The LIBERO split-eval path now keeps its own client/server websocket helpers, so later `Real_Lift2` refactors do not change LIBERO benchmark behavior.
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

## Dataset Prep

This section is only needed for LIBERO training data. Evaluation-only benchmark runs do not need a LeRobot dataset.

Download the four LIBERO LeRobot datasets:

```bash
python -m pip install -U "huggingface-hub==0.35.3"

export LIBERO_DATA_ROOT=/path/to/LEROBOT_LIBERO_DATA
mkdir -p "$LIBERO_DATA_ROOT"

for REPO in \
  IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot
do
  hf download "$REPO" --repo-type dataset --local-dir "$LIBERO_DATA_ROOT/${REPO##*/}"
done
```

After download, the directories are usually LeRobot v2.1 datasets such as:

- `libero_spatial_no_noops_1.0.0_lerobot`
- `libero_object_no_noops_1.0.0_lerobot`
- `libero_goal_no_noops_1.0.0_lerobot`
- `libero_10_no_noops_1.0.0_lerobot`

CubeV2 LIBERO training expects LeRobot v3.0 directories under `LIBERO_ROOT`, for example `libero_goal_no_noops_1.0.0_lerobot_v30`.

Convert them from the repo root:

```bash
for NAME in \
  libero_spatial_no_noops_1.0.0_lerobot \
  libero_object_no_noops_1.0.0_lerobot \
  libero_goal_no_noops_1.0.0_lerobot \
  libero_10_no_noops_1.0.0_lerobot
do
  PYTHONPATH=./src python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
    --repo-id="$NAME" \
    --root="$LIBERO_DATA_ROOT" \
    --push-to-hub=false
done
```

This keeps the original v2.1 folders and writes sibling v3.0 folders with the `_v30` suffix. Then point `LIBERO_ROOT` to the parent directory that contains those `libero_*_lerobot_v30` folders.

To compute the merged normalization stats used by LIBERO finetuning:

```bash
export LIBERO_STATS_ROOT=/path/to/norm_stats/libero_all_chunk10
export LIBERO_REPO_ID_FILE=/tmp/libero_v30_datasets.txt

printf '%s\n' \
  "$LIBERO_DATA_ROOT/libero_spatial_no_noops_1.0.0_lerobot_v30" \
  "$LIBERO_DATA_ROOT/libero_object_no_noops_1.0.0_lerobot_v30" \
  "$LIBERO_DATA_ROOT/libero_goal_no_noops_1.0.0_lerobot_v30" \
  "$LIBERO_DATA_ROOT/libero_10_no_noops_1.0.0_lerobot_v30" \
  > "$LIBERO_REPO_ID_FILE"

PYTHONPATH=./src python src/lerobot/scripts/lerobot_data_stats.py \
  --repo_id_file "$LIBERO_REPO_ID_FILE" \
  --action_mode abs \
  --chunk_size 10 \
  --output_path "$LIBERO_STATS_ROOT/franka/abs/stats.json"
```

This writes one merged stats file for all four suites. During training, point `DATASET_EXTERNAL_STATS_PATH` to that file, or set `DATASET_EXTERNAL_STATS_ROOT=$LIBERO_STATS_ROOT`.

If you train with a different action mode or chunk size, keep `--action_mode` and `--chunk_size` aligned with the training launcher.

## Run

1. Start the policy server in the `magicbot` environment:

```bash
conda activate magicbot

PORT=8000 \
CHECKPOINT_DIR=/path/to/checkpoints/last/pretrained_model \
QWEN3_VL_PRETRAINED_PATH=/path/to/Qwen3-VL-2B-Instruct \
COSMOS_TOKENIZER_PATH_OR_NAME=/path/to/Cosmos-Tokenizer-CI8x8 \
INFER_HORIZON=10 \
bash evaluation/Libero/01_serve_magicbot_libero.sh
```

2. Start the LIBERO benchmark in the `libero` environment:

```bash
conda activate libero

WS_URL=ws://127.0.0.1:8000 \
TASK_SUITE_NAME=libero_goal \
INFER_HORIZON=10 \
VIDEO_ROOT=$PWD/evaluation/Libero/outputs \
bash evaluation/Libero/eval.sh
```

3. Evaluate a single task:

```bash
conda activate libero

WS_URL=ws://127.0.0.1:8000 \
TASK_SUITE_NAME=libero_goal \
TASK_ID=0 \
INFER_HORIZON=10 \
bash evaluation/Libero/eval.sh
```

`INFER_HORIZON` note:

- Serve side `INFER_HORIZON` should usually follow the checkpoint training setup, for example `10`.
- Eval side `INFER_HORIZON` controls how many steps the evaluator actually executes from each returned chunk.
- If you want shorter replanning during evaluation, keep the serve side at `10` and only reduce the eval side, for example `5`.

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
- `SEED`: defaults to `7`
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
conda activate libero

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
- This can happen on either side: the `LIBERO` evaluator or the `magicbot` policy server.

`stats.json not found` or wrong `STATS_KEY`

- Check that the checkpoint contains `pretrained_model/stats.json`
- Current LIBERO eval in this repo expects `STATS_KEY=franka`

Qwen or Cosmos path errors

- Pass `QWEN3_VL_PRETRAINED_PATH` and `COSMOS_TOKENIZER_PATH_OR_NAME` on the serve side
