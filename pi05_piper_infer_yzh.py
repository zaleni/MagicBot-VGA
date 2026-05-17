#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PI05 Piper 单臂推理客户端

连接到 PI05 WebSocket 服务器，为 Piper 单臂机器人提供推理控制。

参考:
- pi0_infer.py: 原始单臂 ROS1 推理脚本
- inference_N.py: 双臂 ROS2 推理脚本
"""

import argparse
import signal
import sys
import time

import cv2
import numpy as np
import rospy
import ros_numpy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header

try:
    from openpi_client import websocket_client_policy
except ImportError:
    print("Error: openpi_client not found. Please install it first.")
    print("Install: pip install openpi-client")
    sys.exit(1)


def jpeg_compress(img):
    """
    对图像进行 JPEG 压缩以匹配训练时的数据分布。

    Args:
        img: BGR 格式图像 (H, W, 3)

    Returns:
        压缩后的 BGR 图像
    """
    encoded = cv2.imencode('.jpg', img)[1].tobytes()
    return cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)


def build_piper_observation(obs_dict, task_prompt, log_tag="PI05"):
    """
    构建适配单臂 PI05 服务器的观测数据。

    单臂模型配置 (pi05_piper_zhenji_table_bussing_demo_100):
    - 相机: 2个 (cam_high, cam_left_wrist)
    - 状态: 14维 (前7维是原始数据，后7维填充零)

    原始训练数据: 7维 (前6维关节位置 + 1维夹爪)
    训练时扩充: 14维 (原始7维 + 填充7维0)

    Args:
        obs_dict: 从 ROS 获取的原始观测
            {
                "images": {
                    "thirdPerson": (H, W, 3) BGR,
                    "wrist": (H, W, 3) BGR,
                },
                "qpos": (7,) 关节位置
            }
        task_prompt: 任务描述字符串
        log_tag: 日志标签

    Returns:
        dict: {
            "images": {
                "cam_high": (3, H, W) uint8,
                "cam_left_wrist": (3, H, W) uint8,
            },
            "state": (14,) float32,
            "prompt": str
        }
    """
    # 相机名称映射: ROS → PI05
    cam_name_map = {
        "thirdPerson": "cam_high",
        "wrist": "cam_left_wrist",
    }

    # 构建图像
    images = {}
    for ros_cam, pi05_cam in cam_name_map.items():
        img = obs_dict["images"].get(ros_cam)

        if img is None:
            # 相机缺失时填充零图像
            print(f"[{log_tag}] Warning: {ros_cam} camera missing, using zeros")
            images[pi05_cam] = np.zeros((3, 360, 640), dtype=np.uint8)
        else:
            # BGR → RGB, HWC → CHW
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images[pi05_cam] = np.transpose(img_rgb, (2, 0, 1))

    # 构建状态: 7维 → 14维 (原始7维 + 填充7维0)
    qpos_7d = np.asarray(obs_dict["qpos"], dtype=np.float32).flatten()

    if qpos_7d.size < 7:
        # 关节数据不足，填充零
        state_14d = np.zeros((14,), dtype=np.float32)
        state_14d[:qpos_7d.size] = qpos_7d
        print(f"[{log_tag}] Warning: qpos has only {qpos_7d.size} dims, expected 7")
    else:
        state_14d = np.zeros((14,), dtype=np.float32)
        state_14d[:7] = qpos_7d[:7]
        # 后7维保持为0

    return {
        "images": images,
        "state": state_14d,
        "prompt": task_prompt,
    }


def extract_piper_actions(response, current_qpos, robot_dof=7, log_tag="PI05"):
    """
    从服务器响应中提取并优化夹爪控制。
    
    优化策略：
    1. 闭合增强：推理值 <= 62000 时，认为正在执行抓取，强制额外减 10000 以增加握力。
    2. 打开补偿：推理值 >= 65000 时，认为正在松开，额外加 2000 以确保完全张开。
    """
    if not isinstance(response, dict) or "actions" not in response:
        print(f"[{log_tag}] Error: response format invalid")
        return []

    actions = np.asarray(response["actions"], dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)

    robot_actions = []
    
    for step in actions:
        # 取前 7 维
        action = step[:robot_dof].copy()
        raw_gripper = action[6]
        
        # --- 核心后处理逻辑 ---
        # 逻辑 A: 只要处于闭合判定区，就应用恒定偏移
        if raw_gripper <= 62000:
            # 强化闭合：在模型预测的基础上再深扣 10000 个单位
            action[6] = raw_gripper - 10000
            # 防止数值过小导致驱动器报错（根据实际情况设定最小值，假设最小是 -100000）
            action[6] = max(action[6], -100000) 
            
        # 逻辑 B: 只要处于打开判定区，就应用恒定偏移
        elif raw_gripper >= 65000:
            # 补偿打开
            new_val = raw_gripper + 5000
            # 限高 90000，防止超出物理极限
            action[6] = min(new_val, 90000)
            
        robot_actions.append(action)

    # 打印一条日志，确认当前块的夹爪平均处理状态
    if len(robot_actions) > 0:
        avg_gripper = np.mean([a[6] for a in robot_actions])
        # print(f"[{log_tag}] Action block processed. Avg Gripper: {avg_gripper:.1f}")

    return robot_actions

class PiperRosOperator:
    """
    Piper 单臂 ROS1 接口封装。

    功能:
    - 订阅相机图像话题
    - 订阅关节状态话题
    - 发布关节控制命令

    话题配置:
    - 相机订阅: /ob_camera_02/color/image_raw (前视), /ob_camera_01/color/image_raw (腕部)
    - 关节订阅: joint_states_single
    - 关节发布: js_cmd
    """

    def __init__(self, args, log_tag="PI05"):
        self.log_tag = log_tag
        self.args = args

        # 图像缓存
        self.front_img = None
        self.wrist_img = None
        self.front_img_time = None
        self.wrist_img_time = None

        # 关节状态缓存
        self.joint_state = None
        self.joint_state_time = None

        # 同步窗口
        self.time_tolerance = 0.1  # 时间同步容差 (秒)

        self._init_ros()

    def _init_ros(self):
        """初始化 ROS 节点和订阅/发布器。"""
        rospy.init_node('piper_pi05_client', anonymous=True)

        # 订阅相机
        rospy.Subscriber(
            self.args.front_cam_topic,
            Image,
            self.front_cam_callback,
            queue_size=10,
            tcp_nodelay=True
        )
        rospy.Subscriber(
            self.args.wrist_cam_topic,
            Image,
            self.wrist_cam_callback,
            queue_size=10,
            tcp_nodelay=True
        )

        # 订阅关节状态
        rospy.Subscriber(
            self.args.joint_state_topic,
            JointState,
            self.joint_callback,
            queue_size=10,
            tcp_nodelay=True
        )

        # 发布关节命令
        self.joint_pub = rospy.Publisher(
            self.args.joint_cmd_topic,
            JointState,
            queue_size=10
        )

        print(f"[{self.log_tag}] ROS initialized:")
        print(f"  - Front camera: {self.args.front_cam_topic}")
        print(f"  - Wrist camera: {self.args.wrist_cam_topic}")
        print(f"  - Joint state: {self.args.joint_state_topic}")
        print(f"  - Joint command: {self.args.joint_cmd_topic}")

    def front_cam_callback(self, msg):
        """前视相机回调。"""
        self.front_img = ros_numpy.numpify(msg)
        self.front_img_time = msg.header.stamp.to_sec()

    def wrist_cam_callback(self, msg):
        """腕部相机回调。"""
        self.wrist_img = ros_numpy.numpify(msg)
        self.wrist_img_time = msg.header.stamp.to_sec()

    def joint_callback(self, msg):
        """关节状态回调。"""
        self.joint_state = msg
        self.joint_state_time = msg.header.stamp.to_sec()

    def get_observation(self):
        """
        获取时间同步的观测数据。

        Returns:
            dict: {
                "images": {
                    "thirdPerson": (H, W, 3) BGR,
                    "wrist": (H, W, 3) BGR,
                },
                "qpos": (7,) 关节位置
            }
            或 None (如果数据不完整/不同步)
        """
        # 检查数据完整性
        if self.front_img is None:
            return None
        if self.wrist_img is None:
            return None
        if self.joint_state is None:
            return None

        # 检查时间同步
        times = [
            self.front_img_time,
            self.wrist_img_time,
            self.joint_state_time,
        ]
        min_time = min(times)
        max_time = max(times)

        if max_time - min_time > self.time_tolerance:
            print(f"[{self.log_tag}] Warning: Data not synchronized, time diff: {max_time - min_time:.3f}s")
            return None

        # 应用 JPEG 压缩以匹配训练分布
        front_img_compressed = jpeg_compress(self.front_img)
        wrist_img_compressed = jpeg_compress(self.wrist_img)

        return {
            "images": {
                "thirdPerson": front_img_compressed,
                "wrist": wrist_img_compressed,
            },
            "qpos": np.array(self.joint_state.position, dtype=np.float32),
        }

    def publish_joint_command(self, action):
        """
        发布关节控制命令。

        Args:
            action: (7,) 关节位置数组
        """
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.name = ['joint0', 'joint1', 'joint2', 'joint3',
                   'joint4', 'joint5', 'joint6']
        msg.position = action
        self.joint_pub.publish(msg)

    def move_to_initial_position(self, target_position, timeout=10.0, log_tag="PI05"):
        """
        移动到初始位置并等待完成。

        注意：Piper 机器人关节位置使用编码器单位，不是弧度！
        典型值范围：-100000 到 100000

        Args:
            target_position: 目标关节位置 (7,)，编码器单位
            timeout: 超时时间 (秒)
            log_tag: 日志标签
        """
        print(f"[{log_tag}] 移动到初始位置...")
        print(f"[{log_tag}] 目标位置 (编码器单位): {target_position}")

        # 等待接收关节状态数据
        print(f"[{log_tag}] 等待关节状态数据...")
        rate = rospy.Rate(10)  # 10Hz
        wait_start = rospy.Time.now()

        while self.joint_state is None and not rospy.is_shutdown():
            if (rospy.Time.now() - wait_start).to_sec() > 3.0:
                print(f"[{log_tag}] 警告: 3秒内未接收到关节状态数据")
                print(f"[{log_tag}] 请检查 ROS 节点是否正常运行")
                print(f"[{log_tag}] 话题: {self.args.joint_state_topic}")
                return False
            rate.sleep()

        if rospy.is_shutdown():
            return False

        # 获取当前位置
        current_pos = np.array(self.joint_state.position)
        print(f"[{log_tag}] 当前位置: {current_pos}")
        diff = np.abs(np.array(target_position) - current_pos)
        print(f"[{log_tag}] 位置差异: {diff}")

        # 发送目标位置
        self.publish_joint_command(target_position)

        # 等待到达目标位置
        # 编码器单位下的到达阈值（500 约等于 0.05 度）
        POSITION_THRESHOLD = 500

        print(f"[{log_tag}] 等待到达目标位置 (阈值: {POSITION_THRESHOLD})...")
        start_time = rospy.Time.now()
        last_print_time = start_time

        while not rospy.is_shutdown():
            if self.joint_state is None:
                rate.sleep()
                continue

            current_pos = np.array(self.joint_state.position)
            diff = np.abs(np.array(target_position) - current_pos)
            max_diff = np.max(diff)

            # 每2秒打印一次进度
            if (rospy.Time.now() - last_print_time).to_sec() > 2.0:
                print(f"[{log_tag}] 当前最大误差: {max_diff:.1f}")
                last_print_time = rospy.Time.now()

            if max_diff < POSITION_THRESHOLD:
                print(f"[{log_tag}] ✓ 已到达初始位置 (最大误差: {max_diff:.1f})")
                return True

            elapsed = (rospy.Time.now() - start_time).to_sec()
            if elapsed > timeout:
                print(f"[{log_tag}] 警告: 移动超时 ({timeout}s)")
                print(f"[{log_tag}] 当前最大误差: {max_diff:.1f}")
                print(f"[{log_tag}] 继续执行...")
                return False

            rate.sleep()

        return True


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="PI05 Piper 单臂推理客户端"
    )

    # WebSocket 服务器配置
    parser.add_argument(
        '--ws_host',
        type=str,
        default='10.60.43.33',
        help='PI05 服务器地址'
    )
    parser.add_argument(
        '--ws_port',
        type=str,
        default='8005',
        help='PI05 服务器端口'
    )

    # 任务配置
    parser.add_argument(
        '--task_prompt',
        type=str,
        default='Sort desktop objects and place them in designated locations.',
        help='任务描述'
    )

    # 控制参数
    parser.add_argument(
        '--publish_rate',
        type=int,
        default=15,
        help='控制频率 (Hz)'
    )
    parser.add_argument(
        '--action_horizon',
        type=int,
        default=50,
        help='动作块大小 (每次推理返回的步数)'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=10000,
        help='最大控制步数'
    )
    parser.add_argument(
        '--robot_dof',
        type=int,
        default=7,
        help='机器人自由度'
    )

    # ROS 话题配置
    parser.add_argument(
        '--front_cam_topic',
        type=str,
        default='/ob_camera_02/color/image_raw',
        help='前视相机话题'
    )
    parser.add_argument(
        '--wrist_cam_topic',
        type=str,
        default='/ob_camera_01/color/image_raw',
        help='腕部相机话题'
    )
    parser.add_argument(
        '--joint_state_topic',
        type=str,
        default='joint_states_single',
        help='关节状态话题'
    )
    parser.add_argument(
        '--joint_cmd_topic',
        type=str,
        default='js_cmd',
        help='关节命令话题'
    )

    # 安全配置
    parser.add_argument(
        '--first_inference_check',
        action='store_true',
        help='首次推理后进行安全确认'
    )

    # 初始位置配置
    parser.add_argument(
        '--init_joint_position',
        type=float,
        nargs=7,
        default=None,
        help='初始关节位置 (7个浮点数，编码器单位，典型范围: -100000~100000)'
    )
    parser.add_argument(
        '--init_wait',
        action='store_true',
        help='到达初始位置后等待用户确认'
    )

    return parser.parse_args()


def signal_handler(_sig, _frame):
    """信号处理函数。"""
    print("\n[PI05] Caught Ctrl+C, shutting down...")
    rospy.signal_shutdown("User interrupt")
    sys.exit(0)


def main():
    """主函数。"""
    args = parse_args()
    log_tag = "PI05-Piper"

    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)

    print(f"\n[{log_tag}] ================================================")
    print(f"[{log_tag}] PI05 Piper 单臂推理客户端")
    print(f"[{log_tag}] ================================================")
    print(f"[{log_tag}] WebSocket 服务器: {args.ws_host}:{args.ws_port}")
    print(f"[{log_tag}] 任务: {args.task_prompt}")
    print(f"[{log_tag}] 控制频率: {args.publish_rate} Hz")
    print(f"[{log_tag}] 动作块大小: {args.action_horizon}")
    print(f"[{log_tag}] 最大步数: {args.max_steps}")
    print(f"[{log_tag}] 状态维度: 7 → 14 (原始+填充)")
    print(f"[{log_tag}] ================================================\n")

    # 1. 初始化 ROS 接口
    print(f"[{log_tag}] 初始化 ROS 接口...")
    ros_operator = PiperRosOperator(args, log_tag=log_tag)

    # 2. 连接 PI05 服务器
    print(f"[{log_tag}] 连接到 PI05 服务器 {args.ws_host}:{args.ws_port}...")
    ws_client = websocket_client_policy.WebsocketClientPolicy(
        host=args.ws_host,
        port=args.ws_port,
    )

    try:
        metadata = ws_client.get_server_metadata()
        print(f"[{log_tag}] 已连接到服务器!")
        print(f"[{log_tag}] 服务器元数据: {list(metadata.keys())}")
    except Exception as e:
        print(f"[{log_tag}] 连接服务器失败: {e}")
        sys.exit(1)

    # 3. 移动到初始位置
    if args.init_joint_position is not None:
        print(f"\n[{log_tag}] ================================================")
        print(f"[{log_tag}] 初始化机器人位置...")
        print(f"[{log_tag}] ================================================")
        ros_operator.move_to_initial_position(args.init_joint_position, log_tag=log_tag)

        if args.init_wait:
            input(f"\n[{log_tag}] 初始化完成，按回车键继续...")
    else:
        print(f"\n[{log_tag}] 未设置初始位置 (--init_joint_position)")
        print(f"[{log_tag}] 将从当前位置开始控制")

    # 4. 等待用户确认
    input(f"\n[{log_tag}] 按回车键开始控制...")

    # 4. 主控制循环
    print(f"\n[{log_tag}] 开始主控制循环...")
    rate = rospy.Rate(args.publish_rate)
    action_buffer = []
    timestep = 0
    inference_count = 0

    while not rospy.is_shutdown() and timestep < args.max_steps:
        # 获取观测
        obs = ros_operator.get_observation()
        if obs is None:
            print(f"[{log_tag}] 等待传感器数据...")
            rate.sleep()
            continue

        # 定期推理 (每 action_horizon 步或 buffer 为空)
        if timestep % args.action_horizon == 0 or len(action_buffer) == 0:
            print(f"\n[{log_tag}] --- 推理 #{inference_count} (t={timestep}) ---")

            # 构建请求
            obs_req = build_piper_observation(obs, args.task_prompt, log_tag=log_tag)

            # 发送推理请求
            try:
                response = ws_client.infer(obs_req)

                # 解析动作序列
                action_buffer = extract_piper_actions(
                    response, 
                    current_qpos=obs['qpos'], # 传入当前位置
                    robot_dof=args.robot_dof, 
                    log_tag=log_tag
                )

                if len(action_buffer) == 0:
                    print(f"[{log_tag}] 警告: 服务器未返回有效动作")
                    timestep += 1
                    rate.sleep()
                    continue

                print(f"[{log_tag}] 获得 {len(action_buffer)} 步动作")
                print(f"[{log_tag}] 首步动作: {action_buffer[0]}")

                # 首次推理安全确认
                if args.first_inference_check and inference_count == 0:
                    print(f"\n[{log_tag}] ========================================")
                    print(f"[{log_tag}] 首次推理安全确认")
                    print(f"[{log_tag}] 当前关节位置: {obs['qpos']}")
                    print(f"[{log_tag}] 首步预测动作: {action_buffer[0]}")
                    print(f"[{log_tag}] ========================================\n")

                    user_input = input("输入 'y' 继续执行: ").strip().lower()
                    if user_input != 'y':
                        print(f"[{log_tag}] 用户取消，退出")
                        sys.exit(0)

                inference_count += 1

            except Exception as e:
                print(f"[{log_tag}] 推理失败: {e}")
                timestep += 1
                rate.sleep()
                continue

        # 执行动作
        if len(action_buffer) > 0:
            action = action_buffer.pop(0)
            ros_operator.publish_joint_command(action)

        timestep += 1
        rate.sleep()

    print(f"\n[{log_tag}] 控制循环结束")
    print(f"[{log_tag}] 总步数: {timestep}")
    print(f"[{log_tag}] 推理次数: {inference_count}")


if __name__ == '__main__':
    main()
