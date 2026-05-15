# CubeV2 Real Piper Sync Deployment

This folder contains sync-only real-robot deployment helpers for the single-arm
Piper `real_piper` CubeV2 model.

Start the model server on the GPU machine:

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
