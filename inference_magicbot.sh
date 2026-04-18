### 1. Start the MagicBot Serve for Real_Lift2
cd /home/arx/MagicBot-VGA

conda activate magicbot

CHECKPOINT_DIR=/home/arx/MagicBot-VGA/models/magicbot-lift2-0418 \
QWEN3_VL_PRETRAINED_PATH=Qwen/Qwen3-VL-2B-Instruct \
QWEN3_VL_PROCESSOR_PATH=Qwen/Qwen3-VL-2B-Instruct \
COSMOS_TOKENIZER_PATH_OR_NAME=nvidia/Cosmos-Tokenizer-CI8x8 \
STATS_KEY=real_lift2 \
ACTION_MODE=delta \
HOST=127.0.0.1 \
PORT=8000 \
INFER_HORIZON=30 \
bash evaluation/Real_Lift2/01_serve_magicbot_real_lift2.sh


### 2. Run the inference script to connect to the MagicBot Serve and perform inference on Real_Lift2
cd /home/arx/MagicBot-VGA

RUN_ENV=act \
WS_URL=ws://127.0.0.1:8000 \
PROMPT="Clear the junk and items off the desktop." \
bash evaluation/Real_Lift2/02_inference_lift2.sh
