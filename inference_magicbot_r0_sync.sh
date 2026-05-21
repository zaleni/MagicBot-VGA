### MagicBot_R0 Real_Lift2 sync-mode startup reference
### RTC is intentionally not enabled for this path yet.

### 1. On the GPU server, start the MagicBot_R0 Real_Lift2 serve
cd ~/research/MagicBot-VGA

conda activate magicbot

### Optional one-time text cache for deployment without loading the T5 text encoder.
### Reuse the training TEXT_EMBED_CACHE_DIR if it already contains this task prompt.
python src/lerobot/scripts/magicbot_r0_precompute_text_embeds.py \
  --text-embedding-cache-dir /home/jjhao/data/text_embeds \
  --model-cache-dir /home/jjhao/data/model \
  --override-instruction "Put the plastic bottle into the trash bin." \
  --context-len 128 \
  --device cuda
### Optional: set DIFFSYNTH_MODEL_BASE_PATH=/path/to/MagicBot_R0/model_cache
### to keep Wan/T5/VAE files off the default ./checkpoints directory.

CHECKPOINT_DIR=/home/jjhao/data/model/zaleni/6B-plasticbottle-delta \
STATS_KEY=real_lift2 \
ACTION_MODE=delta \
DEVICE=cuda \
LOAD_DEVICE=cuda \
HOST=0.0.0.0 \
PORT=9102 \
INFER_HORIZON=32 \
DEFAULT_PROMPT="Put the plastic bottle into the trash bin." \
RTC_ENABLED=false \
MAGICBOT_R0_LOAD_TEXT_ENCODER=false \
MAGICBOT_R0_TEXT_EMBED_CACHE_DIR=/home/jjhao/data/text_embeds \
MAGICBOT_R0_CONTEXT_LEN=128 \
MAGICBOT_R0_SKIP_DIT_LOAD_FROM_PRETRAIN=true \
bash evaluation/Real_Lift2/01_serve_magicbot_r0_real_lift2.sh


### 2. On the robot inference machine, run the existing sync client
cd /home/arx/MagicBot-VGA

RUN_ENV=act \
WS_URL=ws://10.60.45.31:9102 \
PROMPT="Put the plastic bottle into the trash bin." \
FRAME_RATE=24 \
IMAGE_HISTORY_INTERVAL=15 \
INFERENCE_MODE=sync \
LOG_TIMING_EVERY=5 \
bash evaluation/Real_Lift2/02_inference_lift2.sh


### 3. If only the last inference window was stopped, restart just that window
cd /home/arx/MagicBot-VGA
source ~/.bashrc
conda activate act

REAL_LIFT2_RUNTIME_ROOT=/home/arx/ROS2_LIFT_Play/act \
WS_URL=ws://10.60.45.31:9102 \
PROMPT="Put the plastic bottle into the trash bin." \
FRAME_RATE=24 \
IMAGE_HISTORY_INTERVAL=15 \
MAX_PUBLISH_STEP=10000 \
RECORD_MODE=Speed \
USE_BASE=true \
FIXED_BODY_HEIGHT=16 \
STATE_DIM=14 \
ACTION_DIM=14 \
INFERENCE_MODE=sync \
LOG_TIMING_EVERY=5 \
SAFE_STOP_HOME_ARMS=true \
SAFE_STOP_HOME_PUBLISH_STEPS=180 \
SAFE_STOP_BODY_HEIGHT=0 \
SAFE_STOP_PUBLISH_STEPS=30 \
bash evaluation/Real_Lift2/run_real_lift2_inference.sh
