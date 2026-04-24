### Real_Lift2 sync-mode startup reference
### Keep this file minimal: only include parameters that materially affect sync deployment.

### 1. On the GPU server, start the MagicBot Serve for Real_Lift2
cd ~/research/MagicBot-VGA

conda activate magicbot

CHECKPOINT_DIR=/home/jjhao/data/model/Lift2-Table_Clean-abs-0424 \
QWEN3_VL_PRETRAINED_PATH=/home/jjhao/data/model/Qwen3-VL-2B-Instruct \
QWEN3_VL_PROCESSOR_PATH=/home/jjhao/data/model/Qwen3-VL-2B-Instruct \
COSMOS_TOKENIZER_PATH_OR_NAME=/home/jjhao/data/model/Cosmos-Tokenizer-CI8x8 \
STATS_KEY=real_lift2 \
ACTION_MODE=abs \
DEVICE=cuda \
LOAD_DEVICE=cuda \
COSMOS_DEVICE=cuda \
HOST=0.0.0.0 \
PORT=8102 \
INFER_HORIZON=50 \
bash evaluation/Real_Lift2/01_serve_magicbot_real_lift2.sh


### 2. On the robot inference machine, normal full startup path
cd /home/arx/MagicBot-VGA

RUN_ENV=act \
WS_URL=ws://10.60.45.31:8102 \
PROMPT="Clear the junk and items off the desktop." \
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
WS_URL=ws://10.60.45.31:8102 \
PROMPT="Clear the junk and items off the desktop." \
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
