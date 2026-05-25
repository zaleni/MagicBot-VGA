# Real Piper Deployment

This folder serves the single-arm Piper setup for both CubeV2 and MagicBot_R0.
The robot client is shared; only the checkpoint and stats assets change.

## Entry Points

| Family | Server | Robot client |
| --- | --- | --- |
| CubeV2 | `01_serve_cubev2_real_piper_sync.sh` | `02_infer_cubev2_real_piper_sync.sh` |
| MagicBot_R0 | `01_serve_magicbot_r0_real_piper_sync.sh` | `02_infer_magicbot_r0_real_piper_sync.sh` |

## Shared Runtime

- the client sends `cam_high` and `cam_left_wrist`
- the action/state vector is 7D
- `JPEG_ROUNDTRIP=true` matches the reference Piper client
- `MANUAL_RESET=true` enables Enter-to-reset when `INIT_JOINT_POSITION` is set

## Common Env

- `CHECKPOINT_DIR`: checkpoint step dir or `pretrained_model` dir
- `STATS_KEY=real_piper`
- `STATS_PATH`: optional; if unset, the server uses `pretrained_model/stats.json`
- `ACTION_MODE`: keep this aligned with the checkpoint
- `INFER_HORIZON`: server-side chunk length
- `QWEN3_VL_PRETRAINED_PATH`, `COSMOS_TOKENIZER_PATH_OR_NAME`: CubeV2 assets
- `MAGICBOT_R0_LOAD_TEXT_ENCODER`: `true` is the simplest MagicBot_R0 path
- `MAGICBOT_R0_TEXT_EMBED_CACHE_DIR`: only needed when loading cached text embeds

## Quick Start

CubeV2:

```bash
CHECKPOINT_DIR=/path/to/cubev2_real_piper/checkpoints/last/pretrained_model \
STATS_KEY=real_piper \
ACTION_MODE=abs \
INFER_HORIZON=50 \
bash evaluation/Real_Piper/01_serve_cubev2_real_piper_sync.sh

WS_HOST=<server-ip> \
WS_PORT=8000 \
bash evaluation/Real_Piper/02_infer_cubev2_real_piper_sync.sh
```

MagicBot_R0:

```bash
CHECKPOINT_DIR=/path/to/MagicBot_R0/real_piper/checkpoints/30000/pretrained_model \
STATS_KEY=real_piper \
ACTION_MODE=abs \
bash evaluation/Real_Piper/01_serve_magicbot_r0_real_piper_sync.sh

WS_HOST=<server-ip> \
WS_PORT=8102 \
bash evaluation/Real_Piper/02_infer_magicbot_r0_real_piper_sync.sh
```

## Notes

- If `MAGICBOT_R0_LOAD_TEXT_ENCODER=false`, also set `MAGICBOT_R0_TEXT_EMBED_CACHE_DIR`.
- `CHECKPOINT_DIR` can point to the checkpoint step dir or directly to `pretrained_model/`.
- If `STATS_PATH` is omitted, the server uses the checkpoint's own `stats.json`.
