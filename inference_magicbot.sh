### 1. On the GPU server, start the MagicBot Serve for Real_Lift2
cd /home/arx/MagicBot-VGA

conda activate magicbot

CHECKPOINT_DIR=/home/arx/MagicBot-VGA/models/magicbot-lift2-0418 \
QWEN3_VL_PRETRAINED_PATH=/home/arx/MagicBot-VGA/models/Qwen3-VL-2B-Instruct \
QWEN3_VL_PROCESSOR_PATH=/home/arx/MagicBot-VGA/models/Qwen3-VL-2B-Instruct \
COSMOS_TOKENIZER_PATH_OR_NAME=nvidia/Cosmos-Tokenizer-CI8x8 \
STATS_KEY=real_lift2 \
ACTION_MODE=delta \
DEVICE=cuda \
LOAD_DEVICE=cuda \
COSMOS_DEVICE=cuda \
HOST=0.0.0.0 \
PORT=8101 \
INFER_HORIZON=50 \
bash evaluation/Real_Lift2/01_serve_magicbot_real_lift2.sh


### 2. On the robot inference machine, only connect to the remote MagicBot Serve
cd /home/arx/MagicBot-VGA

RUN_ENV=act \
WS_URL=ws://10.60.43.33:8101 \
PROMPT="Clear the junk and items off the desktop." \
PREFETCH_LEAD_STEPS=10 \
LOG_TIMING_EVERY=5 \
bash evaluation/Real_Lift2/02_inference_lift2.sh
