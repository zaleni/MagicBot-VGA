### 1. On the GPU server, start the MagicBot Serve for Real_Lift2
cd ~/research/MagicBot-VGA

conda activate magicbot

CHECKPOINT_DIR=/home/jiangjiahao/data/model/magicbot-lift2-0418 \
QWEN3_VL_PRETRAINED_PATH=/home/jiangjiahao/data/model/Qwen3-VL-2B-Instruct \
QWEN3_VL_PROCESSOR_PATH=/home/jiangjiahao/data/model/Qwen3-VL-2B-Instruct \
COSMOS_TOKENIZER_PATH_OR_NAME=/home/jiangjiahao/data/model/Cosmos-Tokenizer-CI8x8 \
STATS_KEY=real_lift2 \
ACTION_MODE=delta \
DEVICE=cuda \
LOAD_DEVICE=cuda \
COSMOS_DEVICE=cuda \
HOST=0.0.0.0 \
PORT=8101 \
INFER_HORIZON=50 \
NUM_INFERENCE_STEPS=8 \
RTC_ENABLED=true \
RTC_EXECUTION_HORIZON=24 \
RTC_MAX_GUIDANCE_WEIGHT=5.0 \
RTC_PREFIX_ATTENTION_SCHEDULE=exp \
bash evaluation/Real_Lift2/01_serve_magicbot_real_lift2.sh


### 2. On the robot inference machine, only connect to the remote MagicBot Serve
###    This is the normal full startup path and should be the preferred way to
###    launch robot-side inference.
cd /home/arx/MagicBot-VGA

RUN_ENV=act \
WS_URL=ws://10.60.43.33:8101 \
PROMPT="Clear the junk and items off the desktop." \
FRAME_RATE=16 \
IMAGE_HISTORY_INTERVAL=15 \
INFERENCE_MODE=rtc \
PREFETCH_LEAD_STEPS=10 \
LOG_TIMING_EVERY=5 \
RTC_EXECUTION_HORIZON=24 \
RTC_MAX_GUIDANCE_WEIGHT=5.0 \
RTC_QUEUE_THRESHOLD=45 \
RTC_LATENCY_LOOKBACK=10 \
bash evaluation/Real_Lift2/02_inference_lift2.sh


### 3. If you already started CAN/body/lift/realsense and only Ctrl+C'd the final
###    `magicbot-real-lift2` terminal, do NOT rerun 02_inference_lift2.sh.
###    Restart only the last robot inference window with:
###    After restart, give the camera streams a few seconds to warm up. The
###    If camera deque warnings still never recover, then the camera/ROS stack
###    itself needs restarting instead of only this window.
###    In that case, rerun step 2 / 02_inference_lift2.sh rather than this step 3.
cd /home/arx/MagicBot-VGA
source ~/.bashrc
conda activate act

REAL_LIFT2_RUNTIME_ROOT=/home/arx/ROS2_LIFT_Play/act \
WS_URL=ws://10.60.43.33:8101 \
PROMPT="Clear the junk and items off the desktop." \
FRAME_RATE=16 \
IMAGE_HISTORY_INTERVAL=15 \
MAX_PUBLISH_STEP=10000 \
RECORD_MODE=Speed \
USE_BASE=true \
FIXED_BODY_HEIGHT=16 \
STATE_DIM=14 \
ACTION_DIM=14 \
INFERENCE_MODE=rtc \
PREFETCH_LEAD_STEPS=10 \
LOG_TIMING_EVERY=5 \
RTC_EXECUTION_HORIZON=24 \
RTC_MAX_GUIDANCE_WEIGHT=5.0 \
RTC_QUEUE_THRESHOLD=45 \
RTC_LATENCY_LOOKBACK=10 \
SAFE_STOP_HOME_ARMS=true \
SAFE_STOP_HOME_PUBLISH_STEPS=180 \
SAFE_STOP_BODY_HEIGHT=0 \
SAFE_STOP_PUBLISH_STEPS=30 \
bash evaluation/Real_Lift2/run_real_lift2_inference.sh
