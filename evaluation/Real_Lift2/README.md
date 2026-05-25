# Real_Lift2 Deployment

MagicBot_R0 deployment for the Lift2 real-robot setup. The flow is split across two machines:

- `serve`: loads the checkpoint and answers websocket requests
- `run`: reads ROS observations and executes returned action chunks

## Entry Points

- `01_serve_magicbot_r0_real_lift2.sh`: MagicBot_R0 server with Lift2 defaults
- `01_serve_magicbot_real_lift2.sh`: lower-level server wrapper for custom setups
- `run_real_lift2_inference.sh`: start the robot-side loop
- `02_inference_lift2.sh`: one-shot launcher for the full stack
- `test_remote_server.py`: connectivity smoke test
- `remote_client.py`: lightweight client wrapper for custom loops

## Runtime

- state and action are 14D
- request images are `cam_high`, `cam_left_wrist`, and `cam_right_wrist`
- the first successful inference waits for manual safety confirmation
- `DISABLE_3D_TEACHER_FOR_EVAL=true` is the default
- `STATS_PATH` is optional; if omitted, the server uses `CHECKPOINT_DIR/stats.json`

## Quick Start

Serve:

```bash
CHECKPOINT_DIR=/path/to/outputs_real/.../checkpoints/060000 \
ACTION_MODE=abs \
INFER_HORIZON=50 \
bash evaluation/Real_Lift2/01_serve_magicbot_r0_real_lift2.sh
```

`CHECKPOINT_DIR` can point to a checkpoint step dir or directly to `pretrained_model/`.
`ACTION_MODE=delta` is also supported when it matches the checkpoint.
`01_serve_magicbot_r0_real_lift2.sh` defaults `STATS_KEY=real_lift2` and the standard MagicBot_R0 asset paths.

Run:

```bash
WS_URL=ws://127.0.0.1:8000 \
PROMPT="Clear the junk and items off the desktop." \
FRAME_RATE=60 \
IMAGE_HISTORY_INTERVAL=15 \
INFERENCE_MODE=sync \
bash evaluation/Real_Lift2/run_real_lift2_inference.sh
```

Only set `SEND_IMAGE_HEIGHT` and `SEND_IMAGE_WIDTH` if you want a bandwidth/latency tradeoff.

## Key Env Vars

Serve:

- `CHECKPOINT_DIR`
- `STATS_KEY`
- `STATS_PATH`
- `ACTION_MODE`
- `INFER_HORIZON`
- `NUM_INFERENCE_STEPS`
- `QWEN3_VL_PRETRAINED_PATH`
- `COSMOS_TOKENIZER_PATH_OR_NAME`
- `DA3_MODEL_PATH_OR_NAME`
- `DA3_CODE_ROOT`

Run:

- `WS_URL`
- `PROMPT`
- `FRAME_RATE`
- `IMAGE_HISTORY_INTERVAL`
- `SEND_IMAGE_HEIGHT`
- `SEND_IMAGE_WIDTH`
- `INFERENCE_MODE`

## Smoke Test

```bash
python evaluation/Real_Lift2/test_remote_server.py --ws_url ws://10.60.43.33:8101
python evaluation/Real_Lift2/test_remote_server.py --ws_url ws://10.60.43.33:8101 --smoke_infer
```

## Notes

- If you enable the 3D teacher, make sure the serve env can import `depth_anything_3`
