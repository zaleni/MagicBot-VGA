#!/bin/bash
# PI05 Piper 单臂推理客户端启动脚本

# 激活 ROS1 环境
source /home/admin1/deploy/piper_ros/devel/setup.bash

# 进入工作目录
cd /home/admin1/deploy/inference


conda activate deploy 
# 启动 PI05 Piper 客户端
# 请根据实际情况修改以下参数:
#   --ws_host: PI05 服务器地址
#   --ws_port: PI05 服务器端口
#   --task_prompt: 任务描述
#   --publish_rate: 控制频率 (Hz)
#   --front_cam_topic: 前视相机话题
#   --wrist_cam_topic: 腕部相机话题
#   --joint_state_topic: 关节状态话题
#   --joint_cmd_topic: 关节命令话题

# 示例初始位置 (根据实际情况修改)
# INIT_POS="-90346.0 36605.0 -46908.0 831.0 66802.0 1428.0 0.0"

# INIT_POS="-88440.0 41964.0 -65652.0 -4205.0 69901.0 3748.0 0.0" # 机械臂初始位置 table_bussing

INIT_POS="-90885.0 38280.0 -47982.0    518.0  68317.0   1278.0  -2100.0" # 机械臂初始位置 rank_block_rgb

python pi05_piper_infer_yzh.py \
    --ws_host 10.60.45.31 \
    --ws_port 8005 \
    --task_prompt "Position red block, green block, and blue block from left to right in the specified sequence." \
    --publish_rate 20 \
    --action_horizon 50 \
    --max_steps 10000 \
    --init_joint_position $INIT_POS \
    --init_wait \
    --front_cam_topic /ob_camera_02/color/image_raw \
    --wrist_cam_topic /ob_camera_01/color/image_raw \
    --joint_state_topic joint_states_single \
    --joint_cmd_topic js_cmd \
    --first_inference_check 


#--task_prompt "Position red block, green block, and blue block from left to right in the specified sequence." \