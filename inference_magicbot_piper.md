# MagicBot/CubeV2 Real_Piper startup note

This note is the CubeV2 Real_Piper deployment checklist.

## 1. GPU server: start CubeV2 Real_Piper serve

Run this on the GPU machine. The robot-side client below should connect to this
machine's IP and port.

```bash
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
```

## 2. Robot: start roscore

```bash
roscore
```

## 3. Robot: start Orbbec cameras

```bash
conda activate deploy
source /home/admin1/MagicBot/Deploy/tmp/OrbbecSDK_develop/orbbec_ws/devel/setup.bash
roslaunch orbbec_camera multi_camera.launch
```

## 4. Robot: start Piper control node

Check CAN first:

```bash
cd /home/admin1/MagicBot/Deploy/Piper_ros
bash can_activate.sh
```

Then start the Piper node in the same window:

```bash
cd /home/admin1/MagicBot/Deploy/gui_4_2
source /home/admin1/MagicBot/Deploy/Piper_ros/devel/setup.bash
conda activate deploy
roslaunch piper start_single_piper.launch can_port:=can0 auto_enable:=true
```

## 5. Robot: start MagicBot/CubeV2 Piper inference

```bash
source /home/admin1/deploy/piper_ros/devel/setup.bash
cd /home/admin1/MagicBot-VGA
conda activate deploy

INIT_POS="-90885.0 38280.0 -47982.0 518.0 68317.0 1278.0 -2100.0"

WS_HOST=10.60.45.31 \
WS_PORT=8202 \
TASK_PROMPT="Position red block, green block, and blue block from left to right in the specified sequence." \
PUBLISH_RATE=24 \
ACTION_HORIZON=50 \
IMAGE_HISTORY_INTERVAL=15 \
MAX_STEPS=10000 \
INIT_JOINT_POSITION="${INIT_POS}" \
INIT_WAIT=true \
MANUAL_RESET=true \
FRONT_CAM_TOPIC=/ob_camera_02/color/image_raw \
WRIST_CAM_TOPIC=/ob_camera_01/color/image_raw \
JOINT_STATE_TOPIC=joint_states_single \
JOINT_CMD_TOPIC=js_cmd \
FIRST_INFERENCE_CHECK=false \
START_PROMPT=true \
JPEG_ROUNDTRIP=true \
GRIPPER_POSTPROCESS=true \
IMAGE_COLOR_MODE=auto \
EXPECTED_STATS_KEY=real_piper \
bash evaluation/Real_Piper/02_infer_cubev2_real_piper_sync.sh
```

If the GPU server is not `10.60.45.31`, only change `WS_HOST`. Keep
`WS_PORT=8202` unless the server command in step 1 uses a different `PORT`.

## Useful checks

```bash
rostopic list
rostopic echo /joint_states_single
rostopic echo -n 1 /ob_camera_02/color/image_raw/encoding
rostopic echo -n 1 /ob_camera_01/color/image_raw/encoding
```

Manual Piper command:

```bash
rostopic pub -1 /js_cmd sensor_msgs/JointState "{name: ['joint0','joint1','joint2','joint3','joint4','joint5','joint6'], position: [-90346,36605,-46908,831,66802,1428,100000]}"
```

## Notes

- Some older Piper clients send a 14D state because their servers were trained
  that way. This MagicBot/CubeV2 `real_piper` path should stay 7D; the server checks
  `target_action_dim=7` and `stats_key=real_piper`.
- The model-side image convention is RGB. `IMAGE_COLOR_MODE=auto` reads the ROS
  message encoding: `rgb8` is kept as RGB, `bgr8` is converted to RGB, and
  unknown encodings fall back to the legacy BGR-to-RGB path. Only force
  `IMAGE_COLOR_MODE=bgr` if the Orbbec topic encoding is wrong or missing but
  the array is known to be BGR.
- `INIT_POS` is the `rank_block_rgb` initial position.
- With `MANUAL_RESET=true` and `INIT_JOINT_POSITION` set, press Enter during
  inference to move back to `INIT_POS` and pause the current rollout. Press
  Enter again after resetting the scene to clear stale actions and start fresh
  inference from timestep 0.
- The inference window sources `/home/admin1/deploy/piper_ros/devel/setup.bash`.
  The Piper control-node window sources
  `/home/admin1/MagicBot/Deploy/Piper_ros/devel/setup.bash`.
