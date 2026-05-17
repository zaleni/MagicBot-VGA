### Real_Piper sync-mode startup reference
### Keep this file minimal: only include parameters that materially affect Piper sync deployment.

### 1. On the GPU server, start the CubeV2 Serve for Real_Piper
cd ~/research/MagicBot-VGA

conda activate magicbot

CHECKPOINT_DIR=/home/jjhao/data/model/zaleni/3B-Rank_RGB-delta \
QWEN3_VL_PRETRAINED_PATH=/home/jiangjiahao/data/model/Qwen3-VL-2B-Instruct \
QWEN3_VL_PROCESSOR_PATH=/home/jiangjiahao/data/model/Qwen3-VL-2B-Instruct \
COSMOS_TOKENIZER_PATH_OR_NAME=/home/jiangjiahao/data/model/Cosmos-Tokenizer-CI8x8 \
STATS_KEY=real_piper \
ACTION_MODE=delta \
DEVICE=cuda \
LOAD_DEVICE=cuda \
COSMOS_DEVICE=cuda \
HOST=0.0.0.0 \
PORT=8202 \
INFER_HORIZON=50 \
DEFAULT_PROMPT="Position red block, green block, and blue block from left to right in the specified sequence." \
bash evaluation/Real_Piper/01_serve_cubev2_real_piper_sync.sh


### 2. On the Piper robot inference machine, start the ROS1 sync client
cd /home/arx/MagicBot-VGA

source ~/.bashrc
conda activate act

WS_HOST=10.60.43.33 \
WS_PORT=8202 \
TASK_PROMPT="Position red block, green block, and blue block from left to right in the specified sequence." \
PUBLISH_RATE=15 \
ACTION_HORIZON=50 \
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
bash evaluation/Real_Piper/02_infer_cubev2_real_piper_sync.sh


### 3. Useful optional overrides
### INIT_JOINT_POSITION="0 0 0 0 0 0 0" INIT_WAIT=true bash evaluation/Real_Piper/02_infer_cubev2_real_piper_sync.sh
### SEND_IMAGE_HEIGHT=480 SEND_IMAGE_WIDTH=640 bash evaluation/Real_Piper/02_infer_cubev2_real_piper_sync.sh
### IMAGE_COLOR_MODE=rgb bash evaluation/Real_Piper/02_infer_cubev2_real_piper_sync.sh
