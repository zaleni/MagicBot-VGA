# -- coding: UTF-8
import os
import sys

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    os.chdir(str(ROOT))


import argparse
import collections
import json
import pickle
from urllib.parse import urlparse
import yaml
import rclpy
import threading

from rclpy.executors import MultiThreadedExecutor

from utils.utils import set_seed  # helper functions

from utils.ros_operator import RosOperator, Rate
from utils.setup_loader import setup_loader

from functools import partial
import signal
import sys

obs_dict = collections.OrderedDict()

import multiprocessing as mp

from multiprocessing.shared_memory import SharedMemory

import numpy as np

try:
    from openpi_client import websocket_client_policy
except Exception:
    websocket_client_policy = None

# WebSocket 客户端单例，避免重复握手/建连
ws_client = None

# 设置打印输出行宽
np.set_printoptions(linewidth=200)

# 禁用科学计数法
np.set_printoptions(suppress=True)


def load_yaml(yaml_file):
    try:
        with open(yaml_file, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: File not found - {yaml_file}")

        return None
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML file - {e}")

        return None


def make_shm_name_dict(args, shapes):
    shm_name_dict = {}
    for cam in args.camera_names:
        shm_name_dict[cam] = f"shm_img_{cam}"
    for state_key in shapes["states"]:
        shm_name_dict[state_key] = f"shm_state_{state_key}"
    shm_name_dict["action"] = "shm_action"

    return shm_name_dict


def create_shm_dict(config, shm_name_dict, shapes, dtypes):
    shm_dict = {}
    for cam, shape in shapes["images"].items():
        size = np.prod(shape) * np.dtype(dtypes[cam]).itemsize
        shm = SharedMemory(name=shm_name_dict[cam], create=True, size=size)
        shm_dict[cam] = (shm, shape, dtypes[cam])
    for state_key, shape in shapes["states"].items():
        size = np.prod(shape) * np.dtype(np.float32).itemsize
        shm = SharedMemory(name=shm_name_dict[state_key], create=True, size=size)
        shm_dict[state_key] = (shm, shape, np.float32)

    action_shape = config['policy_config']['action_dim']
    size = np.prod(action_shape) * np.dtype(np.float32).itemsize
    shm = SharedMemory(name=shm_name_dict["action"], create=True, size=size)
    shm_dict["action"] = (shm, action_shape, np.float32)

    return shm_dict


def connect_shm_dict(shm_name_dict, shapes, dtypes, config):
    shm_dict = {}
    for cam, shape in shapes["images"].items():
        shm = SharedMemory(name=shm_name_dict[cam], create=False)
        shm_dict[cam] = (shm, shape, dtypes[cam])
    for state_key, shape in shapes["states"].items():
        shm = SharedMemory(name=shm_name_dict[state_key], create=False)
        shm_dict[state_key] = (shm, shape, np.float32)

    action_shape = (config['policy_config']['action_dim'],)
    shm = SharedMemory(name=shm_name_dict["action"], create=False)
    shm_dict["action"] = (shm, action_shape, np.float32)

    return shm_dict


def robot_action(action, shm_dict):
    shm, shape, dtype = shm_dict["action"]
    np_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    np_array[:] = action


def get_model_config(args):

    set_seed(args.seed)

    # WebSocket client mode only
    policy_config = {
        'action_dim': args.action_dim,
        'states_dim': args.state_dim,
        'use_base': args.use_base,
    }

    config = {
        'episode_len': args.max_publish_step,
        'state_dim': policy_config['states_dim'],
        'policy_config': policy_config,
        'camera_names': args.camera_names,
    }

    print(f"action_dim={policy_config['action_dim']} state_dim={policy_config['states_dim']}")

    return config


def apply_gripper_gate(action_value, gate):
    min_gripper = 0
    max_gripper = 5

    return min_gripper if action_value < gate else max_gripper


def get_obervations(args, timestep, ros_operator):
    global obs_dict

    rate = Rate(args.frame_rate)
    while True and rclpy.ok():
        obs_dict = ros_operator.get_observation(ts=timestep)
        if not obs_dict:
            print("syn fail")
            rate.sleep()

            continue

        return obs_dict


def init_robot(ros_operator, use_base, connected_event, start_event):
    init0 = [0, 0, 0, 0, 0, 0, 0]
    init1 = [0, 0, 0, 0, 0, 0, 0]

    # 发布初始位置（关节空间姿态）
    ros_operator.follow_arm_publish_continuous(init0, init0)
    # ros_operator.robot_base_shutdown()

    connected_event.set()
    start_event.wait()

    ros_operator.follow_arm_publish_continuous(init1, init1)
    if use_base:
        ros_operator.start_base_control_thread()


def signal_handler(signal, frame, ros_operator):
    print('Caught Ctrl+C / SIGINT signal')

    # 底盘给零
    ros_operator.base_enable = False
    ros_operator.robot_base_shutdown()
    ros_operator.base_control_thread.join()

    sys.exit(0)


def cleanup_shm(names):
    for name in names:
        try:
            shm = SharedMemory(name=name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass


def _build_pi05_request(obs_dict, timestep, camera_names, task_prompt, log_tag):
    """构建官方标准请求格式（匹配 openPI 服务端）。"""
    del timestep, camera_names

    # ROS camera name -> PI05 camera name (must match training/inference server contract)
    cam_name_map = {
        "head": "cam_high",
        "left_wrist": "cam_left_wrist",
        "right_wrist": "cam_right_wrist",
    }

    def _to_chw_uint8(img):
        img_np = np.asarray(img)
        if img_np.ndim == 3 and img_np.shape[-1] == 3:  # HWC -> CHW
            img_np = np.transpose(img_np, (2, 0, 1))
        elif img_np.ndim == 3 and img_np.shape[0] == 3:  # already CHW
            pass
        else:
            raise ValueError(f"[{log_tag}] invalid image rank/shape: {img_np.shape}, expected HWC or CHW with 3 channels")

        if img_np.shape != (3, 480, 640):
            raise ValueError(f"[{log_tag}] invalid image shape: {img_np.shape}, expected (3, 480, 640)")

        if img_np.dtype != np.uint8:
            raise ValueError(f"[{log_tag}] invalid image dtype: {img_np.dtype}, expected uint8")
        return img_np

    images = {}
    for ros_cam, pi_cam in cam_name_map.items():
        ros_img = obs_dict.get("images", {}).get(ros_cam, None)
        if ros_img is None:
            images[pi_cam] = np.zeros((3, 480, 640), dtype=np.uint8)
        else:
            images[pi_cam] = _to_chw_uint8(ros_img)

    qpos = np.asarray(obs_dict.get("qpos", np.zeros((14,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    if qpos.size < 14:
        state = np.zeros((14,), dtype=np.float32)
        state[:qpos.size] = qpos
    else:
        state = qpos[:14].astype(np.float32, copy=False)

    return {
        "images": images,
        "state": state,
        "prompt": task_prompt,
    }


def _call_pi05_server(ws_url, timeout_sec, payload, log_tag):
    """官方客户端调用（自动处理 metadata 握手 + msgpack-numpy 序列化）。"""
    del timeout_sec
    global ws_client

    if websocket_client_policy is None:
        print(f"[{log_tag}] openpi_client 不可用，请检查环境安装")
        return None

    try:
        if ws_client is None:
            parsed = urlparse(ws_url)
            host = parsed.hostname or "10.60.43.33"
            port = parsed.port or 8000
            ws_client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
            try:
                metadata = ws_client.get_server_metadata()
                print(f"[{log_tag}] 官方客户端连接成功, metadata keys: {list(metadata.keys())}")
            except Exception:
                print(f"[{log_tag}] 官方客户端连接成功")

        return ws_client.infer(payload)
    except Exception as e:
        print(f"[{log_tag}] 官方客户端调用失败: {e}")
        # 连接异常时清空单例，下一轮自动重连
        ws_client = None
        return None


def _resolve_ws_config(args):
    """解析 WebSocket 配置，仅使用命令行参数。"""
    log_tag = args.model_name.upper().strip() if args.model_name else "PI05"

    # 直接使用命令行参数，无回退机制
    ws_url = args.ws_url
    ws_timeout = args.ws_timeout

    return ws_url, ws_timeout, log_tag


def _extract_action_from_pi05_response(response, action_dim):
    """解析官方返回动作序列，取首步作为当前控制输出。"""
    if not isinstance(response, dict):
        return np.zeros((action_dim,), dtype=np.float32)

    if "actions" in response:
        actions_np = np.asarray(response["actions"], dtype=np.float32)
        if actions_np.ndim == 2 and actions_np.shape[0] > 0:
            action_np = actions_np[0].reshape(-1)
        else:
            action_np = actions_np.reshape(-1)
    elif "action" in response:
        action_np = np.asarray(response["action"], dtype=np.float32).reshape(-1)
    else:
        return np.zeros((action_dim,), dtype=np.float32)

    if action_np.size < action_dim:
        out = np.zeros((action_dim,), dtype=np.float32)
        out[:action_np.size] = action_np
        return out
    if action_np.size > action_dim:
        return action_np[:action_dim]
    return action_np


def _extract_action_sequence_from_pi05_response(response, action_dim):
    """Extract full action sequence and expand each step to action_dim."""
    if not isinstance(response, dict):
        return []

    if "actions" in response:
        seq = np.asarray(response["actions"], dtype=np.float32)
    elif "action" in response:
        seq = np.asarray(response["action"], dtype=np.float32)
    else:
        return []

    if seq.ndim == 1:
        seq = seq.reshape(1, -1)
    elif seq.ndim > 2:
        seq = seq.reshape(seq.shape[0], -1)

    out_seq = []
    for step in seq:
        step = np.asarray(step, dtype=np.float32).reshape(-1)
        if step.size < action_dim:
            out = np.zeros((action_dim,), dtype=np.float32)
            out[:step.size] = step
            out_seq.append(out)
        elif step.size > action_dim:
            out_seq.append(step[:action_dim])
        else:
            out_seq.append(step)

    return out_seq


def ros_process(args, config, meta_queue, connected_event, start_event, shm_ready_event):
    setup_loader(ROOT)
    #初始化 ROS
    rclpy.init()

    data = load_yaml(args.data)

    #RosOperator是一个 ROS2 节点包装器，负责：1.订阅传感器话题,接收相机图像、关节状态等 2.发布控制命令 发送机器人动作指令
    ros_operator = RosOperator(args, data, in_collect=False)

    def _spin_loop(node):
        while rclpy.ok():
            try:
                rclpy.spin_once(node, timeout_sec=0.001)
            except Exception:
                break
    #3 创建并启动循环线程
    spin_thread = threading.Thread(target=_spin_loop, args=(ros_operator,), daemon=True)
    spin_thread.start()

    if args.use_base:
        signal.signal(signal.SIGINT, partial(signal_handler, ros_operator=ros_operator))
    # 初始化机器人姿态并通知主进程
    init_robot(ros_operator, args.use_base, connected_event, start_event)

    rate = Rate(args.frame_rate)

    #数据形状提取循环
    while rclpy.ok():
        obs = ros_operator.get_observation()     #获取观察数据
        if obs:
            shapes = {"images": {}, "states": {}, "dtypes": {}}

            for cam in args.camera_names:
                img = obs["images"][cam]
                shapes["images"][cam] = img.shape
                shapes["dtypes"][cam] = img.dtype
            shapes["states"]["qpos"] = obs["qpos"].shape
            shapes["states"]["qvel"] = obs["qvel"].shape
            shapes["states"]["effort"] = obs["effort"].shape
            shapes["states"]["robot_base"] = obs["robot_base"].shape
            shapes["states"]["base_velocity"] = obs["base_velocity"].shape

            meta_queue.put(shapes)

            break

        rate.sleep()

    # 创建共享内存
    shm_name_dict = meta_queue.get()

    cleanup_shm(shm_name_dict.values())
    shm_dict = create_shm_dict(config, shm_name_dict, shapes, shapes["dtypes"])
    shm_ready_event.set()

    rate = Rate(args.frame_rate)
    while rclpy.ok():
        obs = ros_operator.get_observation()
        if not obs:
            rate.sleep()

            continue

        # 写入共享内存
        for cam in args.camera_names:
            shm, shape, dtype = shm_dict[cam]
            np_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            np_array[:] = obs["images"][cam]
        for state_key in shapes["states"]:
            shm, shape, dtype = shm_dict[state_key]
            np_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            np_array[:] = obs[state_key]

        # 读取动作并执行
        shm, shape, dtype = shm_dict["action"]
        action = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()
        gripper_idx = [6, 13]

        # In fixed-body-height mode, keep sending base command every cycle
        # even if action is all zeros (e.g. before first valid inference output).
        if args.use_base and args.fixed_body_height >= 0:
            fixed_h = float(args.fixed_body_height)
            action_base = np.zeros((10,), dtype=np.float32)
            action_base[3] = fixed_h
            ros_operator.set_robot_base_target(action_base)

        if np.any(action):  # 确保动作不全是 0
            gripper_gate = args.gripper_gate

            left_action = action[:gripper_idx[0] + 1]  # 取8维度
            if gripper_gate != -1:
                left_action[gripper_idx[0]] = apply_gripper_gate(left_action[gripper_idx[0]], gripper_gate)

            right_action = action[gripper_idx[0] + 1:gripper_idx[1] + 1]
            if gripper_gate != -1:
                right_action[gripper_idx[0]] = apply_gripper_gate(left_action[gripper_idx[0]], gripper_gate)

            ros_operator.follow_arm_publish(left_action, right_action)

            if args.use_base and args.fixed_body_height < 0:
                action_base = action[gripper_idx[1] + 1:gripper_idx[1] + 1 + 10].copy()
                ros_operator.set_robot_base_target(action_base)

        rate.sleep()

    try:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass

    spin_thread.join(timeout=0.2)
    for shm, _, _ in shm_dict.values():
        shm.close()
        shm.unlink()


def inference_process(args, config, shm_dict, shapes, ros_proc):
    # WebSocket service config (CLI > env > default)
    ws_url, ws_timeout, log_tag = _resolve_ws_config(args)

    action_dim = config['policy_config']['action_dim']
    action = np.zeros((action_dim,), dtype=np.float32)
    first_inference = True
    exec_rate = Rate(args.frame_rate)

    print(f"[{log_tag}] WebSocket inference enabled: {ws_url}")

    while ros_proc.is_alive():
        timestep = 0

        while timestep < args.max_publish_step and ros_proc.is_alive():
            obs_dict = {
                "images": {},
                "qpos": None,
                "qvel": None,
                "effort": None,
                "robot_base": None,
                "base_velocity": None,
            }

            # Keep the original shared-memory observation read path unchanged.
            for cam in args.camera_names:
                shm, shape, dtype = shm_dict[cam]
                obs_dict["images"][cam] = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()
            for state_key in shapes["states"]:
                shm, shape, dtype = shm_dict[state_key]
                obs_dict[state_key] = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()

            # Build request payload for PI05 server.
            request_payload = _build_pi05_request(
                obs_dict=obs_dict,
                timestep=timestep,
                camera_names=config['camera_names'],
                task_prompt=args.task_prompt,
                log_tag=log_tag,
            )

            # Call PI05 service (placeholder protocol; customize response schema later).
            response = _call_pi05_server(
                ws_url=ws_url,
                timeout_sec=ws_timeout,
                payload=request_payload,
                log_tag=log_tag,
            )

            # 服务端偶发失败时不直接退出，保留下轮重试。
            if response is None:
                print(f"[安全保护] {log_tag} 服务端无返回，本轮跳过并等待下一轮重试。")
                timestep += 1
                continue

            # One-time manual safety confirmation on the first successful PI05 inference.
            if first_inference:
                qpos = np.asarray(obs_dict.get("qpos", np.zeros((14,), dtype=np.float32)), dtype=np.float32).reshape(-1)
                if qpos.size < 14:
                    qpos_full = np.zeros((14,), dtype=np.float32)
                    qpos_full[:qpos.size] = qpos
                    qpos = qpos_full
                else:
                    qpos = qpos[:14]

                if not isinstance(response, dict) or response.get("actions", None) is None:
                    print(f"[安全保护] {log_tag} 返回缺少 actions(50,14)，程序安全退出，不执行机器人动作。")
                    return

                actions_seq = np.asarray(response["actions"], dtype=np.float32)
                if actions_seq.ndim != 2 or actions_seq.shape[1] != 14:
                    print(f"[安全保护] {log_tag} 返回 actions 形状异常: {actions_seq.shape}，期望 (N,14)。程序安全退出。")
                    return

                print("\n" + "=" * 72)
                print(f"[首次安全确认] 已获取 {log_tag} 动作结果，暂停下发机器人")
                print("=" * 72)
                print("当前机器人14维关节状态 qpos:")
                print("[" + ", ".join([f"{x:8.4f}" for x in qpos]) + "]")

                print(f"\n{log_tag} 返回动作序列 actions 前30步(每步14维):")
                num_steps = min(30, actions_seq.shape[0])
                for i in range(num_steps):
                    print(f"Step {i:2d}: [" + ", ".join([f"{x:8.4f}" for x in actions_seq[i]]) + "]")

                user_input = input("\n【安全确认】请检查动作数值是否正常，输入 y 并回车继续执行: ").strip().lower()
                if user_input != 'y':
                    print("[安全保护] 用户未确认，程序安全退出，不执行机器人动作。")
                    return

                first_inference = False
                print("[安全确认] 已确认，开始下发动作并进入正常实时推理。\n")

            # Execute the full predicted sequence step-by-step before next obs request.
            action_seq = _extract_action_sequence_from_pi05_response(response, action_dim)
            if len(action_seq) == 0:
                print(f"[安全保护] {log_tag} 返回动作为空，本轮跳过并等待下一轮重试。")
                timestep += 1
                continue

            for step_action in action_seq:
                if timestep >= args.max_publish_step or (not ros_proc.is_alive()):
                    break

                action = step_action
                robot_action(action, shm_dict)
                timestep += 1
                exec_rate.sleep()

        # Keep original safe-stop behavior for base related action slots.
        if args.use_base and action_dim > 19:
            action[16] = 0
            action[17] = 0
            action[19] = 0

        robot_action(action, shm_dict)


def parse_args(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_publish_step', type=int, default=10000, help='max publish step')

    # 配置文件
    parser.add_argument('--data', type=str,
                        default=Path.joinpath(ROOT, 'data/config.yaml'),
                        help='config file')

    # 推理设置
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--model_name', type=str, default='PI05',
                        help='model/server display name for logs, e.g. PI05/LINGBOT/OPENPI')
    parser.add_argument('--ws_url', type=str, default='ws://10.60.43.33:8000',
                        help='websocket endpoint (required for remote inference)')
    parser.add_argument('--ws_timeout', type=float, default=3.0,
                        help='websocket timeout in seconds (default: 3.0)')
    parser.add_argument('--task_prompt', type=str,
                        default='Clear the junk and items off the desktop.',
                        help='prompt sent to policy server')
    parser.add_argument('--action_dim', type=int, default=14,
                        help='action dimension expected by robot control loop, default 14')
    parser.add_argument('--state_dim', type=int, default=14,
                        help='state dimension used for metadata/config, default 14')

    # 摄像头设置
    parser.add_argument('--camera_names', nargs='+', type=str,
                        choices=['head', 'left_wrist', 'right_wrist', ],
                        default=['head', 'left_wrist', 'right_wrist'],
                        help='camera names to use')
    parser.add_argument('--use_depth_image', action='store_true',
                        help='enable depth image subscriptions (default: off)')

    # 机器人设置
    parser.add_argument('--use_base', action='store_true', help='use robot base')
    parser.add_argument('--fixed_body_height', type=float, default=-1.0,
                        help='>=0 to lock chassis and wheel motion, and set fixed body height (raw controller unit)')
    parser.add_argument('--record', choices=['Distance', 'Speed'], default='Distance',
                        help='record data')
    parser.add_argument('--frame_rate', type=int, default=60, help='frame rate')

    parser.add_argument('--gripper_gate', type=float, default=-1, help='gripper gate threshold')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(args):

    #双进程架构，使用多进程同步对象来协调 ROS 进程和推理进程
    meta_queue = mp.Queue()                  #进程间通信队列 用于传递元数据（数据形状、共享内存配置等）在主进程和 ROS 进程之间。
    connected_event = mp.Event()           #确保 ROS 已初始化完成，主进程才继续执行。
    start_event = mp.Event()             #用户确认机制：等待用户按回车键后，ROS 进程才开始运行主循环。
    shm_ready_event = mp.Event()         #确保 共享内存已创建完成，主进程才能连接

    # 获取模型config
    config = get_model_config(args)

    # 启动ROS进程
    ros_proc = mp.Process(target=ros_process, args=(args, config, meta_queue,
                                                    connected_event, start_event, shm_ready_event))
    ros_proc.start()
    # 等待 ROS 连接
    connected_event.wait()
    input("Enter any key to continue :")
    #通知 ROS 可以启动
    start_event.set()

    # # 从 ROS 获取形状 等待meta信息
    shapes = meta_queue.get()
    # 生成共享内存配置
    shm_name_dict = make_shm_name_dict(args, shapes)

    meta_queue.put(shm_name_dict) #发送给 ROS 进程共享内存配置，ROS 进程创建共享内存后会设置 shm_ready_event，主进程等待后再连接共享内存。

    shm_ready_event.wait()   #阻塞直到 ROS 创建完成
    #连接共享内存
    shm_dict = connect_shm_dict(shm_name_dict, shapes, shapes["dtypes"], config)

    # 推理
    try:
        inference_process(args, config, shm_dict, shapes, ros_proc)
    except KeyboardInterrupt:
        pass
    finally:
        for shm, _, _ in shm_dict.values():
            shm.close()
            shm.unlink()
        ros_proc.terminate()
        ros_proc.join()


if __name__ == '__main__':
    args = parse_args()
    main(args)
