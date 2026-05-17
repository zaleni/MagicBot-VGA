# CubeV2 Real Piper Sync Deployment

This folder contains sync-only real-robot deployment helpers for the single-arm
Piper `real_piper` CubeV2 model.

Start the CubeV2 model server on the GPU machine:

```bash
CHECKPOINT_DIR=/path/to/cubev2_real_piper/checkpoints/last/pretrained_model \
STATS_KEY=real_piper \
ACTION_MODE=abs \
INFER_HORIZON=50 \
bash evaluation/Real_Piper/01_serve_cubev2_real_piper_sync.sh
```

Start the ROS1 client on the Piper machine:

```bash
WS_HOST=<server-ip> \
WS_PORT=8000 \
TASK_PROMPT="Sort desktop objects and place them in designated locations." \
bash evaluation/Real_Piper/02_infer_cubev2_real_piper_sync.sh
```

The client sends only two cameras, `cam_high` and `cam_left_wrist`, plus a 7D
single-arm state/action vector. It does not send a right arm or right wrist
camera.

By default the ROS client does one JPEG roundtrip on camera frames to match the
Piper reference script. Set `JPEG_ROUNDTRIP=false` to disable it.

If `INIT_JOINT_POSITION` is set, `MANUAL_RESET=true` enables Enter controls in
the inference terminal: press Enter once to move back to the init pose and pause
the rollout, then press Enter again to clear stale actions and restart inference
from timestep 0.

For MagicBot_R0, train with:

```bash
DATASET_DIR=/path/to/real_piper_lerobot30 \
ACTION_TYPE=abs \
bash launch/magicbot_r0/magicbot_r0_finetune_real_piper.sh
```

For MagicBot_R0 delta training, compute stats first:

```bash
DATASET_DIR=/path/to/real_piper_lerobot30 \
ACTION_TYPE=delta \
CHUNK_SIZE=32 \
bash launch/compute_norm/compute_norm_stats_real_piper_delta.sh
```

Then serve and run the same Piper robot client through the R0 wrappers:

```bash
CHECKPOINT_DIR=/path/to/MagicBot_R0/real_piper/checkpoints/30000/pretrained_model \
STATS_KEY=real_piper \
ACTION_MODE=abs \
bash evaluation/Real_Piper/01_serve_magicbot_r0_real_piper_sync.sh

WS_HOST=<server-ip> \
WS_PORT=8102 \
bash evaluation/Real_Piper/02_infer_magicbot_r0_real_piper_sync.sh
```

For MagicBot_R0 serving, the simplest runtime path is
`MAGICBOT_R0_LOAD_TEXT_ENCODER=true`, which lets the server consume plain text
prompts. If you set `MAGICBOT_R0_LOAD_TEXT_ENCODER=false`, also set
`MAGICBOT_R0_TEXT_EMBED_CACHE_DIR` and precompute the exact deployment prompt
first.
