# Real_Lift2 Deployment

This directory contains a standalone MagicBot real-robot deployment path.
It does not depend on `openpi` or on the original PI deployment script.

## Files

### Entrypoints

- `model_server.py`
  Starts the MagicBot websocket inference server on the GPU machine.
- `01_serve_magicbot_real_lift2.sh`
  Shell wrapper for starting the server.
- `main.py`
  Thin real-robot deployment entrypoint for MagicBot inference orchestration.
- `run_real_lift2_inference.sh`
  Shell wrapper for the real-robot deployment loop.
- `02_inference_lift2.sh`
  Convenience launcher that starts the full robot-side stack and the Real_Lift2 inference window.

### Robot-Side Runtime

- `inference.py`
  Robot-side inference session logic: sync/async/RTC loops, manual-home interaction, and first-chunk safety confirmation.
- `runtime.py`
  Robot-runtime helpers extracted from the entrypoint: ROS setup, shared memory, safe-stop, and low-level action publish logic.
- `remote_client.py`
  Lightweight client helper for calling the websocket server from your own robot loop.
- `test_remote_server.py`
  Lightweight connectivity checker for validating that the robot-side machine can reach a remote MagicBot websocket server.

### Transport Helpers

- `request_builder.py`
  Builds `images + qpos + history` request payloads.
- `websocket_server.py`
  Local websocket server implementation.
- `websocket_client.py`
  Local websocket client implementation.
- `msgpack_numpy.py`
  NumPy-aware msgpack serialization helpers.

## Recommended Environments

We recommend using two separate environments.

- `serve` environment
  Use the same environment that can already run MagicBot training or offline evaluation.
- `run` environment
  Use the same robot runtime environment that already works with your robot-side ROS control stack.

This split is the safest setup because the two sides depend on different stacks:

- `serve` side
  Mainly needs model-serving dependencies such as `torch`, `transformers`, `numpy`, `msgpack`, `websockets`, `pyyaml`, and the local `lerobot/MagicBot` code.
- `run` side
  Mainly needs robot-runtime dependencies such as `rclpy`, `utils.ros_operator`, `utils.setup_loader`, `numpy`, `msgpack`, `websockets`, and `pyyaml`.

If you really want to run both on one machine, you can still keep two conda environments and launch the two scripts separately.

## DA3 Note For The Serve Side

By default, `01_serve_magicbot_real_lift2.sh` keeps:

```bash
DISABLE_3D_TEACHER_FOR_EVAL=true
```

In that default mode, the server sets `lambda_3d=0` at runtime, so MagicBot will not instantiate the DA3 teacher during model startup.
That means the `serve` environment does not need a separate DA3 runtime by default.

You only need DA3-related setup on the `serve` side if you explicitly want to enable 3D-teacher evaluation, for example by disabling that flag and keeping DA3 active.
In that case, the server must be able to import `depth_anything_3`, either by:

- installing the `depth_anything_3` package into the `serve` environment, or
- passing `DA3_CODE_ROOT=/path/to/standalone/DA3/repo`

## Serve-Side Extra Pip Packages

If your MagicBot training/eval environment is already working, the main deep-learning stack should already be present.
The most common extra runtime packages needed by the websocket `serve` side are:

```bash
pip install "websockets>=14" msgpack pyyaml draccus huggingface_hub datasets pandas pyarrow pillow packaging einops
```

In most setups you should not reinstall `torch`, `torchvision`, or `transformers` here unless your existing MagicBot environment is missing them.

## Start The Server

```bash
CHECKPOINT_DIR=/path/to/outputs_real/.../checkpoints/060000 \
STATS_KEY=real_lift2 \
ACTION_MODE=delta \
INFER_HORIZON=30 \
bash evaluation/Real_Lift2/01_serve_magicbot_real_lift2.sh
```

If `CHECKPOINT_DIR` points to:

- `.../checkpoints/060000`
  the script will automatically resolve `pretrained_model/`
- `.../checkpoints/060000/pretrained_model`
  that also works directly

Typical `serve`-machine checklist:

- GPU machine with the MagicBot training/eval environment
- access to the checkpoint directory
- access to Qwen3-VL / Cosmos resources referenced by the checkpoint or overridden by env vars
- no ROS runtime needed
- DA3 runtime not needed in the default `DISABLE_3D_TEACHER_FOR_EVAL=true` mode

## Start The Real-Robot Loop

```bash
WS_URL=ws://127.0.0.1:8000 \
PROMPT="Clear the junk and items off the desktop." \
SEND_IMAGE_HEIGHT=240 \
SEND_IMAGE_WIDTH=320 \
bash evaluation/Real_Lift2/run_real_lift2_inference.sh
```

If `sync` mode is stable but chunk boundaries still pause for too long, the safest first optimization is usually reducing the robot-side websocket image size, for example `240x320`. The server will still resize/pad again to the model input size.

`main.py` + `inference.py` + `runtime.py` follow the same broad structure as the existing robot deployment loop:

- a ROS/shared-memory process reads robot observations
- an inference process sends camera/state data to the MagicBot websocket server
- returned action chunks are executed step by step
- the first successful inference pauses for manual safety confirmation

Typical `run`-machine checklist:

- the same runtime that already works for your robot deployment stack
- ROS and `rclpy`
- `utils.ros_operator`, `utils.setup_loader`, and related robot-side helpers
- network access to the websocket server
- no MagicBot checkpoint loading needed on this side
- sync the robot-side module set together: `main.py`, `inference.py`, `runtime.py`, `remote_client.py`, `request_builder.py`, `websocket_client.py`, and `msgpack_numpy.py`

## Test Remote Server Connectivity

Before launching the real robot loop, you can verify that the robot-side machine can reach a remote GPU server:

```bash
python evaluation/Real_Lift2/test_remote_server.py \
  --ws_url ws://10.60.43.33:8101
```

If you also want to send a dummy all-zero observation and validate the returned action chunk shape:

```bash
python evaluation/Real_Lift2/test_remote_server.py \
  --ws_url ws://10.60.43.33:8101 \
  --smoke_infer
```

## Minimal Client Usage

```python
from evaluation.Real_Lift2.remote_client import RealLift2RemoteClient

client = RealLift2RemoteClient(
    host="ws://127.0.0.1:8000",
    prompt="Clear the junk and items off the desktop.",
    image_history_interval=15,
)

client.reset()

response = client.infer_step(
    images={
        "head": head_img,
        "left_wrist": left_wrist_img,
        "right_wrist": right_wrist_img,
    },
    qpos=qpos_14d,
    timestep=timestep,
)

actions = response["actions"]  # shape: [T, 14]
```

## Request Format

The server accepts:

```python
{
    "images": {
        "cam_high": np.ndarray,
        "cam_left_wrist": np.ndarray,
        "cam_right_wrist": np.ndarray,
    },
    "state": np.ndarray,   # 14-dim qpos
    "prompt": str,
    "timestep": int,
    "reset": bool,
}
```

Images may be:

- a single frame in `HWC` or `CHW`
- a two-frame history in `THWC` or `TCHW`

The server returns:

```python
{
    "actions": np.ndarray,   # [infer_horizon, action_dim]
    "action": np.ndarray,    # first action
    "server_timing": {...},
}
```
