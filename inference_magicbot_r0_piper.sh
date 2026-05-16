### MagicBot_R0 Real_Piper sync-mode startup reference
### RTC is intentionally not enabled for this path.

### 1. On the GPU server, start the MagicBot_R0 Real_Piper serve
cd ~/research/MagicBot-VGA

conda activate magicbot

### Optional one-time text cache for deployment without loading the T5 text encoder.
### Reuse the training TEXT_EMBED_CACHE_DIR if it already contains this task prompt.
# python src/lerobot/scripts/magicbot_r0_precompute_text_embeds.py \
#   --text-embedding-cache-dir /path/to/MagicBot_R0/text_embeds \
#   --model-cache-dir /path/to/MagicBot_R0/model_cache \
#   --override-instruction "Sort desktop objects and place them in designated locations." \
#   --context-len 128 \
#   --device cuda
### Optional: set DIFFSYNTH_MODEL_BASE_PATH=/path/to/MagicBot_R0/model_cache
### to keep Wan/T5/VAE files off the default ./checkpoints directory.

CHECKPOINT_DIR=/path/to/MagicBot_R0/real_piper/checkpoints/30000/pretrained_model \
STATS_KEY=real_piper \
ACTION_MODE=abs \
DEVICE=cuda \
LOAD_DEVICE=cuda \
HOST=0.0.0.0 \
PORT=8102 \
INFER_HORIZON=24 \
DEFAULT_PROMPT="Sort desktop objects and place them in designated locations." \
RTC_ENABLED=false \
MAGICBOT_R0_LOAD_TEXT_ENCODER=false \
MAGICBOT_R0_TEXT_EMBED_CACHE_DIR=/path/to/MagicBot_R0/text_embeds \
MAGICBOT_R0_CONTEXT_LEN=128 \
MAGICBOT_R0_SKIP_DIT_LOAD_FROM_PRETRAIN=true \
MAGICBOT_R0_CONCAT_MULTI_CAMERA=horizontal \
bash evaluation/Real_Piper/01_serve_magicbot_r0_real_piper_sync.sh


### 2. On the Piper robot inference machine, run the ROS1 sync client
cd /home/arx/MagicBot-VGA

source ~/.bashrc
conda activate act

WS_HOST=10.60.43.33 \
WS_PORT=8102 \
TASK_PROMPT="Sort desktop objects and place them in designated locations." \
PUBLISH_RATE=15 \
ACTION_HORIZON=24 \
IMAGE_HISTORY_INTERVAL=15 \
MAX_STEPS=10000 \
FRONT_CAM_TOPIC=/ob_camera_02/color/image_raw \
WRIST_CAM_TOPIC=/ob_camera_01/color/image_raw \
JOINT_STATE_TOPIC=joint_states_single \
JOINT_CMD_TOPIC=js_cmd \
FIRST_INFERENCE_CHECK=true \
START_PROMPT=true \
JPEG_ROUNDTRIP=true \
GRIPPER_POSTPROCESS=true \
EXPECTED_STATS_KEY=real_piper \
bash evaluation/Real_Piper/02_infer_magicbot_r0_real_piper_sync.sh


### 3. Useful optional overrides
### INIT_JOINT_POSITION="0 0 0 0 0 0 0" INIT_WAIT=true bash evaluation/Real_Piper/02_infer_magicbot_r0_real_piper_sync.sh
### SEND_IMAGE_HEIGHT=480 SEND_IMAGE_WIDTH=640 bash evaluation/Real_Piper/02_infer_magicbot_r0_real_piper_sync.sh
### IMAGE_COLOR_MODE=rgb bash evaluation/Real_Piper/02_infer_magicbot_r0_real_piper_sync.sh
