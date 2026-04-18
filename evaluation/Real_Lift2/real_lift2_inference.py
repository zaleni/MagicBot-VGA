#!/usr/bin/env python

from __future__ import annotations

import argparse
import collections
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
from functools import partial
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

import numpy as np
import yaml

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
SRC_ROOT = ROOT / "src"

for candidate in [THIS_DIR, ROOT, SRC_ROOT]:
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

try:
    import rclpy
except Exception as exc:
    rclpy = None
    RCLPY_IMPORT_ERROR = exc
else:
    RCLPY_IMPORT_ERROR = None

try:
    from utils.utils import set_seed
    from utils.ros_operator import RosOperator, Rate
    from utils.setup_loader import setup_loader
except Exception as exc:
    set_seed = None
    RosOperator = None
    Rate = None
    setup_loader = None
    REAL_ROBOT_IMPORT_ERROR = exc
else:
    REAL_ROBOT_IMPORT_ERROR = None

try:
    from .real_lift2_remote_client import RealLift2RemoteClient
except ImportError:
    from real_lift2_remote_client import RealLift2RemoteClient


obs_dict = collections.OrderedDict()
np.set_printoptions(linewidth=200, suppress=True)


def ensure_runtime_available() -> None:
    if rclpy is None:
        raise ImportError(
            "rclpy is not available. Please run this script in the same robot runtime environment."
        ) from RCLPY_IMPORT_ERROR
    if RosOperator is None or Rate is None or setup_loader is None:
        raise ImportError(
            "Robot deployment helpers (utils.ros_operator / utils.setup_loader / utils.utils) are not available. "
            "Please run this script in the same robot runtime environment."
        ) from REAL_ROBOT_IMPORT_ERROR


def load_yaml(yaml_file: str | Path) -> dict[str, Any] | None:
    try:
        with open(yaml_file, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: file not found - {yaml_file}")
        return None
    except yaml.YAMLError as exc:
        print(f"Error: failed to parse YAML file - {exc}")
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
        size = int(np.prod(shape) * np.dtype(dtypes[cam]).itemsize)
        shm = SharedMemory(name=shm_name_dict[cam], create=True, size=size)
        shm_dict[cam] = (shm, shape, dtypes[cam])

    for state_key, shape in shapes["states"].items():
        size = int(np.prod(shape) * np.dtype(np.float32).itemsize)
        shm = SharedMemory(name=shm_name_dict[state_key], create=True, size=size)
        shm_dict[state_key] = (shm, shape, np.float32)

    action_shape = (config["policy_config"]["action_dim"],)
    size = int(np.prod(action_shape) * np.dtype(np.float32).itemsize)
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

    action_shape = (config["policy_config"]["action_dim"],)
    shm = SharedMemory(name=shm_name_dict["action"], create=False)
    shm_dict["action"] = (shm, action_shape, np.float32)
    return shm_dict


def robot_action(action, shm_dict):
    shm, shape, dtype = shm_dict["action"]
    np_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    np_array[:] = action


def get_model_config(args):
    if set_seed is not None:
        set_seed(args.seed)
    else:
        np.random.seed(args.seed)

    config = {
        "episode_len": args.max_publish_step,
        "policy_class": "cubev2_remote",
        "policy_config": {
            "action_dim": args.action_dim,
            "states_dim": args.state_dim,
        },
        "camera_names": args.camera_names,
    }
    return config


def init_robot(ros_operator, use_base, connected_event, start_event):
    init0 = [0, 0, 0, 0, 0, 0, 4]
    init1 = [0, 0, 0, 0, 0, 0, 0]

    ros_operator.follow_arm_publish_continuous(init0, init0)

    connected_event.set()
    start_event.wait()

    ros_operator.follow_arm_publish_continuous(init1, init1)
    if use_base:
        ros_operator.start_base_control_thread()


def publish_safe_stop_base(ros_operator, safe_stop_body_height: float | None, frame_rate: int, publish_steps: int) -> None:
    if safe_stop_body_height is None:
        return

    try:
        action_base = np.zeros((10,), dtype=np.float32)
        action_base[3] = float(safe_stop_body_height)
        print(
            f"[SafeStop] Publishing base target height={safe_stop_body_height} "
            f"for {publish_steps} steps before shutdown."
        )
        for _ in range(max(1, int(publish_steps))):
            ros_operator.set_robot_base_target(action_base)
            time.sleep(max(1.0 / max(1, int(frame_rate)), 0.01))
    except Exception as exc:
        print(f"[SafeStop] Failed to publish safe-stop base target: {exc}")


def signal_handler(_signal, _frame, ros_operator, use_base: bool, safe_stop_body_height: float | None, safe_stop_publish_steps: int, frame_rate: int):
    print("Caught shutdown signal")
    if use_base:
        publish_safe_stop_base(
            ros_operator=ros_operator,
            safe_stop_body_height=safe_stop_body_height,
            frame_rate=frame_rate,
            publish_steps=safe_stop_publish_steps,
        )
    ros_operator.base_enable = False
    ros_operator.robot_base_shutdown()
    base_thread = getattr(ros_operator, "base_control_thread", None)
    if base_thread is not None:
        base_thread.join(timeout=2.0)
    sys.exit(0)


def cleanup_shm(names):
    for name in names:
        try:
            shm = SharedMemory(name=name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass


def extract_action_sequence(response, action_dim):
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
            out[: step.size] = step
            out_seq.append(out)
        elif step.size > action_dim:
            out_seq.append(step[:action_dim])
        else:
            out_seq.append(step)
    return out_seq


def ros_process(args, config, meta_queue, connected_event, start_event, shm_ready_event):
    ensure_runtime_available()
    setup_loader(ROOT)

    rclpy.init()
    data = load_yaml(args.data)
    ros_operator = RosOperator(args, data, in_collect=False)

    def _spin_loop(node):
        while rclpy.ok():
            try:
                rclpy.spin_once(node, timeout_sec=0.001)
            except Exception:
                break

    spin_thread = threading.Thread(target=_spin_loop, args=(ros_operator,), daemon=True)
    spin_thread.start()

    signal.signal(
        signal.SIGINT,
        partial(
            signal_handler,
            ros_operator=ros_operator,
            use_base=args.use_base,
            safe_stop_body_height=args.safe_stop_body_height,
            safe_stop_publish_steps=args.safe_stop_publish_steps,
            frame_rate=args.frame_rate,
        ),
    )
    signal.signal(
        signal.SIGTERM,
        partial(
            signal_handler,
            ros_operator=ros_operator,
            use_base=args.use_base,
            safe_stop_body_height=args.safe_stop_body_height,
            safe_stop_publish_steps=args.safe_stop_publish_steps,
            frame_rate=args.frame_rate,
        ),
    )

    init_robot(ros_operator, args.use_base, connected_event, start_event)

    rate = Rate(args.frame_rate)
    while rclpy.ok():
        obs = ros_operator.get_observation()
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

        for cam in args.camera_names:
            shm, shape, dtype = shm_dict[cam]
            np_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            np_array[:] = obs["images"][cam]
        for state_key in shapes["states"]:
            shm, shape, dtype = shm_dict[state_key]
            np_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            np_array[:] = obs[state_key]

        shm, shape, dtype = shm_dict["action"]
        action = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()
        gripper_idx = [6, 13]

        if args.use_base and args.fixed_body_height >= 0:
            fixed_h = float(args.fixed_body_height)
            action_base = np.zeros((10,), dtype=np.float32)
            action_base[3] = fixed_h
            ros_operator.set_robot_base_target(action_base)

        if np.any(action):
            left_action = action[: gripper_idx[0] + 1].copy()
            right_action = action[gripper_idx[0] + 1 : gripper_idx[1] + 1].copy()

            if args.gripper_gate != -1:
                left_action[gripper_idx[0]] = 0 if left_action[gripper_idx[0]] < args.gripper_gate else 5
                right_action[gripper_idx[0]] = 0 if right_action[gripper_idx[0]] < args.gripper_gate else 5

            ros_operator.follow_arm_publish(left_action, right_action)

            if args.use_base and args.fixed_body_height < 0 and action.shape[0] > gripper_idx[1] + 10:
                action_base = action[gripper_idx[1] + 1 : gripper_idx[1] + 1 + 10].copy()
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
    ws_url = args.ws_url or os.getenv("REAL_LIFT2_WS_URL", "ws://127.0.0.1:8000")
    action_dim = config["policy_config"]["action_dim"]
    action = np.zeros((action_dim,), dtype=np.float32)
    first_inference = True
    exec_rate = Rate(args.frame_rate)

    def build_client():
        return RealLift2RemoteClient(
            host=ws_url,
            prompt=args.prompt,
            image_history_interval=args.image_history_interval,
            state_dim=args.state_dim,
            max_history=args.image_history_interval + 1,
        )

    client = build_client()
    print(f"[MagicBot] WebSocket inference enabled: {ws_url}")
    try:
        print(f"[MagicBot] server metadata keys: {list(client.metadata.keys())}")
    except Exception:
        pass

    while ros_proc.is_alive():
        timestep = 0
        client.reset()

        while timestep < args.max_publish_step and ros_proc.is_alive():
            obs_dict = {
                "images": {},
                "qpos": None,
                "qvel": None,
                "effort": None,
                "robot_base": None,
                "base_velocity": None,
            }

            for cam in args.camera_names:
                shm, shape, dtype = shm_dict[cam]
                obs_dict["images"][cam] = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()
            for state_key in shapes["states"]:
                shm, shape, dtype = shm_dict[state_key]
                obs_dict[state_key] = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()

            try:
                response = client.infer_step(
                    images=obs_dict["images"],
                    qpos=obs_dict.get("qpos", np.zeros((args.state_dim,), dtype=np.float32)),
                    timestep=timestep,
                    prompt=args.prompt,
                )
            except Exception as exc:
                print(f"[MagicBot] remote inference failed: {exc}")
                try:
                    client = build_client()
                    client.reset()
                except Exception as reconnect_exc:
                    print(f"[MagicBot] reconnect failed: {reconnect_exc}")
                timestep += 1
                continue

            if response is None:
                print("[SafeGuard] MagicBot server returned nothing, skip this cycle.")
                timestep += 1
                continue

            if first_inference:
                qpos = np.asarray(obs_dict.get("qpos", np.zeros((args.state_dim,), dtype=np.float32)), dtype=np.float32).reshape(-1)
                qpos_full = np.zeros((args.state_dim,), dtype=np.float32)
                qpos_full[: min(args.state_dim, qpos.size)] = qpos[: args.state_dim]

                if not isinstance(response, dict) or response.get("actions", None) is None:
                    print("[SafeGuard] MagicBot response is missing `actions`, aborting before robot execution.")
                    return

                actions_seq = np.asarray(response["actions"], dtype=np.float32)
                if actions_seq.ndim != 2 or actions_seq.shape[1] != action_dim:
                    print(
                        f"[SafeGuard] MagicBot returned unexpected action shape: {actions_seq.shape}, "
                        f"expected (N, {action_dim})."
                    )
                    return

                print("\n" + "=" * 72)
                print("[First Safety Check] Received first MagicBot action chunk, pausing before execution")
                print("=" * 72)
                print("Current qpos:")
                print("[" + ", ".join([f"{x:8.4f}" for x in qpos_full]) + "]")

                print("\nPredicted actions (first up to 30 steps):")
                num_steps = min(30, actions_seq.shape[0])
                for idx in range(num_steps):
                    print(f"Step {idx:2d}: [" + ", ".join([f"{x:8.4f}" for x in actions_seq[idx]]) + "]")

                user_input = input("\n[Safety Confirm] Inspect the actions and input y to continue: ").strip().lower()
                if user_input != "y":
                    print("[SafeGuard] User did not confirm, exiting without robot execution.")
                    return

                first_inference = False
                print("[Safety Confirmed] Starting normal real-time execution.\n")

            action_seq = extract_action_sequence(response, action_dim)
            if len(action_seq) == 0:
                print("[SafeGuard] MagicBot returned an empty action sequence, skip this cycle.")
                timestep += 1
                continue

            for step_action in action_seq:
                if timestep >= args.max_publish_step or (not ros_proc.is_alive()):
                    break

                action = step_action
                robot_action(action, shm_dict)
                timestep += 1
                exec_rate.sleep()

        if args.use_base and action_dim > 19:
            action[16] = 0
            action[17] = 0
            action[19] = 0

        robot_action(action, shm_dict)


def parse_args(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_publish_step", type=int, default=10000, help="Max publish step.")
    parser.add_argument("--data", type=str, default=str(ROOT / "data" / "config.yaml"), help="Robot config YAML.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--camera_names",
        nargs="+",
        type=str,
        choices=["head", "left_wrist", "right_wrist"],
        default=["head", "left_wrist", "right_wrist"],
        help="Camera names to use.",
    )
    parser.add_argument("--use_base", action="store_true", help="Use robot base.")
    parser.add_argument(
        "--fixed_body_height",
        type=float,
        default=-1.0,
        help=">=0 to lock chassis and wheel motion, and set fixed body height.",
    )
    parser.add_argument("--record", choices=["Distance", "Speed"], default="Distance", help="Record mode.")
    parser.add_argument("--frame_rate", type=int, default=60, help="Control frame rate.")
    parser.add_argument("--gripper_gate", type=float, default=-1, help="Optional gripper threshold.")
    parser.add_argument("--prompt", type=str, default="Clear the junk and items off the desktop.")
    parser.add_argument("--ws_url", type=str, default="", help="MagicBot websocket URL.")
    parser.add_argument("--image_history_interval", type=int, default=15, help="History interval in frames.")
    parser.add_argument("--state_dim", type=int, default=14, help="State dimension.")
    parser.add_argument("--action_dim", type=int, default=14, help="Action dimension.")
    parser.add_argument(
        "--safe_stop_body_height",
        type=float,
        default=None,
        help="Optional base height target to publish for a few cycles before shutdown. Set to 0 to lower before stop.",
    )
    parser.add_argument(
        "--safe_stop_publish_steps",
        type=int,
        default=30,
        help="Number of cycles to publish the safe-stop base target before shutdown.",
    )

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(args):
    ensure_runtime_available()
    meta_queue = mp.Queue()

    connected_event = mp.Event()
    start_event = mp.Event()
    shm_ready_event = mp.Event()

    config = get_model_config(args)

    ros_proc = mp.Process(
        target=ros_process,
        args=(args, config, meta_queue, connected_event, start_event, shm_ready_event),
    )
    ros_proc.start()

    connected_event.wait()
    input("Enter any key to continue :")
    start_event.set()

    shapes = meta_queue.get()
    shm_name_dict = make_shm_name_dict(args, shapes)
    meta_queue.put(shm_name_dict)
    shm_ready_event.wait()

    shm_dict = connect_shm_dict(shm_name_dict, shapes, shapes["dtypes"], config)

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


if __name__ == "__main__":
    main(parse_args())
