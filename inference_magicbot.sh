### 1. Start the MagicBot Serve for Real_Lift2
cd /home/arx/MagicBot-VGA

conda activate magicbot

CHECKPOINT_DIR=/home/arx/MagicBot-VGA/models/zaleni/magicbot-lift2-0418 \
STATS_KEY=real_lift2 \
ACTION_MODE=delta \
HOST=127.0.0.1 \
PORT=8000 \
bash evaluation/Real_Lift2/01_serve_magicbot_real_lift2.sh


### 2. Run the inference script to connect to the MagicBot Serve and perform inference on Real_Lift2
cd /home/arx/MagicBot-VGA

RUN_ENV=act \
WS_URL=ws://127.0.0.1:8000 \
PROMPT="Clear the junk and items off the desktop." \
bash evaluation/Real_Lift2/02_inference_lift2.sh
