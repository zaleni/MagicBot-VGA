#!/usr/bin/env python

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import select
import sys
import threading
import time
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
SRC_ROOT = ROOT / "src"

for candidate in [THIS_DIR, ROOT, SRC_ROOT]:
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

try:
    from .real_lift2_remote_client import RealLift2RemoteClient
    from .real_lift2_runtime import (
        Rate,
        ROBOT_RUNTIME_ROOT,
        connect_shm_dict,
        ensure_runtime_available,
        get_model_config,
        make_shm_name_dict,
        request_graceful_ros_shutdown,
        robot_action,
        ros_process,
    )
except ImportError:
    from real_lift2_remote_client import RealLift2RemoteClient
    from real_lift2_runtime import (
        Rate,
        ROBOT_RUNTIME_ROOT,
        connect_shm_dict,
        ensure_runtime_available,
        get_model_config,
        make_shm_name_dict,
        request_graceful_ros_shutdown,
        robot_action,
        ros_process,
    )


np.set_printoptions(linewidth=200, suppress=True)


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


def read_observation_snapshot(args, shm_dict, shapes):
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

    return obs_dict


def compute_prefetch_lead_steps(frame_rate: int, action_seq_len: int, round_trip_ms: float | None, min_lead_steps: int) -> int:
    lead_steps = max(1, int(min_lead_steps))
    if round_trip_ms is not None:
        inferred_steps = int(np.ceil(float(round_trip_ms) * max(1, frame_rate) / 1000.0)) + 1
        lead_steps = max(lead_steps, inferred_steps)
    return min(lead_steps, max(1, action_seq_len - 1))


def print_timing_log(args, *, chunk_idx: int, action_seq_len: int, response: dict | None):
    if args.log_timing_every <= 0:
        return
    if not (chunk_idx <= 3 or chunk_idx % args.log_timing_every == 0):
        return

    server_timing = response.get("server_timing", {}) if isinstance(response, dict) else {}
    client_timing = response.get("client_timing", {}) if isinstance(response, dict) else {}
    infer_ms = server_timing.get("infer_ms")
    round_trip_ms = client_timing.get("round_trip_ms")
    chunk_budget_ms = 1000.0 * action_seq_len / max(1, args.frame_rate)

    message = f"[Timing] mode={args.inference_mode} chunk={chunk_idx} horizon={action_seq_len} budget={chunk_budget_ms:.1f}ms"
    if infer_ms is not None:
        message += f" server_infer={float(infer_ms):.1f}ms"
    if round_trip_ms is not None:
        message += f" round_trip={float(round_trip_ms):.1f}ms"
        if float(round_trip_ms) > chunk_budget_ms:
            message += "  <- slower than chunk budget, likely to cause stop-go motion"
    print(message)


def write_zero_action(shm_dict, action_dim: int, reason: str | None = None) -> np.ndarray:
    action = np.zeros((action_dim,), dtype=np.float32)
    robot_action(action, shm_dict)
    if reason:
        print(reason)
    return action


def set_manual_home_command(manual_home_command, enabled: bool) -> None:
    with manual_home_command.get_lock():
        manual_home_command.value = 1 if enabled else 0


def print_manual_home_help() -> None:
    print(
        "[Manual Home] Controls:\n"
        "  - Press Enter once: home both arms and pause the current rollout.\n"
        "  - Press Enter again: start a brand-new rollout from timestep 0.\n"
        "  - Base height will stay unchanged during manual home if fixed height is enabled.\n"
    )


def poll_manual_console_command(manual_home_active: bool) -> str | None:
    if not sys.stdin or not sys.stdin.isatty():
        return None

    try:
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    except Exception:
        return None

    if not ready:
        return None

    try:
        line = sys.stdin.readline()
    except Exception:
        return None

    if line == "":
        return None

    command = line.strip().lower()
    if command == "":
        if manual_home_active:
            return "resume"
        return "home"
    if manual_home_active and command == "h":
        return "resume"
    return command or None


def maybe_enter_manual_home_pause(args, ros_proc, shm_dict, manual_home_command, action_dim: int) -> bool:
    command = poll_manual_console_command(manual_home_active=False)
    if command != "home":
        return False

    set_manual_home_command(manual_home_command, True)
    robot_action(np.zeros((action_dim,), dtype=np.float32), shm_dict)
    print(
        "\n[Manual Home] Enter detected. Homing both arms now while keeping the base height unchanged.\n"
        "[Manual Home] Current rollout is now paused.\n"
        "[Manual Home] After the arms settle and you finish resetting the scene, press Enter again.\n"
        "[Manual Home] That second Enter will start a brand-new rollout from timestep 0.\n"
    )

    while ros_proc.is_alive():
        resume_command = poll_manual_console_command(manual_home_active=True)
        if resume_command == "resume":
            set_manual_home_command(manual_home_command, False)
            robot_action(np.zeros((action_dim,), dtype=np.float32), shm_dict)
            print(
                "[Manual Home] Second Enter detected.\n"
                "[Manual Home] Leaving home-pause state and starting a fresh rollout now.\n"
            )
            return True
        if resume_command not in (None, "home"):
            print("[Manual Home] Currently paused at home pose. Press Enter to resume.")
        time.sleep(0.05)

    set_manual_home_command(manual_home_command, False)
    return True


def maybe_run_first_safety_check(args, response, obs_dict, action_dim):
    qpos = np.asarray(
        obs_dict.get("qpos", np.zeros((args.state_dim,), dtype=np.float32)), dtype=np.float32
    ).reshape(-1)
    qpos_full = np.zeros((args.state_dim,), dtype=np.float32)
    qpos_full[: min(args.state_dim, qpos.size)] = qpos[: args.state_dim]

    if not isinstance(response, dict) or response.get("actions", None) is None:
        print("[SafeGuard] MagicBot response is missing `actions`, aborting before robot execution.")
        return False

    actions_seq = np.asarray(response["actions"], dtype=np.float32)
    if actions_seq.ndim != 2 or actions_seq.shape[1] != action_dim:
        print(
            f"[SafeGuard] MagicBot returned unexpected action shape: {actions_seq.shape}, "
            f"expected (N, {action_dim})."
        )
        return False

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
        return False

    print("[Safety Confirmed] Starting normal real-time execution.\n")
    return True


class AsyncChunkPrefetcher:
    """Background websocket inference worker that can prefetch the next action chunk."""

    def __init__(self, build_client):
        self._build_client = build_client
        self._request_queue: queue.Queue[dict | None] = queue.Queue(maxsize=1)
        self._result_queue: queue.Queue[dict] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._inflight = False
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    def _worker_loop(self):
        client = self._build_client()
        while not self._stop_event.is_set():
            try:
                task = self._request_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if task is None:
                return

            try:
                response = client.infer_step(
                    images=task["images"],
                    qpos=task["qpos"],
                    timestep=task["timestep"],
                    prompt=task["prompt"],
                )
                result = {
                    "ok": True,
                    "response": response,
                    "timestep": task["timestep"],
                }
            except Exception as exc:
                result = {
                    "ok": False,
                    "error": exc,
                    "timestep": task["timestep"],
                }
                try:
                    client = self._build_client()
                    client.reset()
                except Exception as reconnect_exc:
                    result["reconnect_error"] = reconnect_exc

            while not self._stop_event.is_set():
                try:
                    self._result_queue.put(result, timeout=0.1)
                    break
                except queue.Full:
                    try:
                        self._result_queue.get_nowait()
                    except queue.Empty:
                        pass

    def submit(self, *, images, qpos, timestep: int, prompt: str) -> bool:
        with self._lock:
            if self._inflight or self._stop_event.is_set():
                return False
            self._inflight = True

        task = {
            "images": images,
            "qpos": np.asarray(qpos, dtype=np.float32).reshape(-1),
            "timestep": int(timestep),
            "prompt": prompt,
        }

        try:
            self._request_queue.put_nowait(task)
            return True
        except queue.Full:
            with self._lock:
                self._inflight = False
            return False

    def has_inflight(self) -> bool:
        with self._lock:
            return self._inflight

    def poll(self):
        try:
            result = self._result_queue.get_nowait()
        except queue.Empty:
            return None
        with self._lock:
            self._inflight = False
        return result

    def wait(self, timeout: float | None = None):
        try:
            result = self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        with self._lock:
            self._inflight = False
        return result

    def stop(self):
        self._stop_event.set()
        try:
            self._request_queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=1.0)


def inference_process(args, config, shm_dict, shapes, ros_proc, manual_home_command):
    ws_url = args.ws_url or os.getenv("REAL_LIFT2_WS_URL", "ws://127.0.0.1:8000")
    action_dim = config["policy_config"]["action_dim"]
    action = np.zeros((action_dim,), dtype=np.float32)
    robot_action(action, shm_dict)
    first_inference = True
    exec_rate = Rate(args.frame_rate)
    chunk_idx = 0

    def build_client():
        return RealLift2RemoteClient(
            host=ws_url,
            prompt=args.prompt,
            image_history_interval=args.image_history_interval,
            state_dim=args.state_dim,
            max_history=args.image_history_interval + 1,
        )

    metadata_client = build_client()
    print(f"[MagicBot] WebSocket inference enabled: {ws_url}")
    try:
        print(f"[MagicBot] server metadata keys: {list(metadata_client.metadata.keys())}")
    except Exception:
        pass
    finally:
        metadata_client.close()
    print_manual_home_help()

    while ros_proc.is_alive():
        timestep = 0
        episode_restart_requested = False
        if args.inference_mode == "async":
            prefetcher = AsyncChunkPrefetcher(build_client)
            current_response = None
            current_obs = None
            next_response = None
            last_round_trip_ms = None

            try:
                while timestep < args.max_publish_step and ros_proc.is_alive():
                    if maybe_enter_manual_home_pause(args, ros_proc, shm_dict, manual_home_command, action_dim):
                        episode_restart_requested = True
                        break

                    if current_response is None:
                        if next_response is not None:
                            current_response = next_response
                            next_response = None
                        else:
                            if not prefetcher.has_inflight():
                                current_obs = read_observation_snapshot(args, shm_dict, shapes)
                                submitted = prefetcher.submit(
                                    images=current_obs["images"],
                                    qpos=current_obs.get("qpos", np.zeros((args.state_dim,), dtype=np.float32)),
                                    timestep=timestep,
                                    prompt=args.prompt,
                                )
                                if not submitted:
                                    time.sleep(0.002)
                                    continue

                            result = prefetcher.wait(timeout=5.0)
                            if result is None:
                                action = write_zero_action(
                                    shm_dict,
                                    action_dim,
                                    "[SafeGuard] Waiting for prefetched chunk timed out, forcing zero action for this cycle.",
                                )
                                continue

                            if not result.get("ok", False):
                                action = write_zero_action(
                                    shm_dict,
                                    action_dim,
                                    f"[SafeGuard] Remote inference failed, forcing zero action for this cycle: {result['error']}",
                                )
                                reconnect_exc = result.get("reconnect_error")
                                if reconnect_exc is not None:
                                    print(f"[MagicBot] reconnect failed: {reconnect_exc}")
                                timestep += 1
                                continue

                            current_response = result.get("response")

                    response = current_response
                    current_response = None

                    if response is None:
                        action = write_zero_action(
                            shm_dict,
                            action_dim,
                            "[SafeGuard] MagicBot server returned nothing, forcing zero action for this cycle.",
                        )
                        timestep += 1
                        continue

                    if first_inference:
                        if current_obs is None:
                            current_obs = read_observation_snapshot(args, shm_dict, shapes)
                        if not maybe_run_first_safety_check(args, response, current_obs, action_dim):
                            write_zero_action(shm_dict, action_dim, "[SafeGuard] First safety check failed, zeroing action buffer.")
                            return
                        first_inference = False

                    action_seq = extract_action_sequence(response, action_dim)
                    if len(action_seq) == 0:
                        action = write_zero_action(
                            shm_dict,
                            action_dim,
                            "[SafeGuard] MagicBot returned an empty action sequence, forcing zero action for this cycle.",
                        )
                        timestep += 1
                        continue

                    chunk_idx += 1
                    if isinstance(response, dict):
                        client_timing = response.get("client_timing", {})
                        if client_timing is not None:
                            last_round_trip_ms = client_timing.get("round_trip_ms", last_round_trip_ms)

                    print_timing_log(args, chunk_idx=chunk_idx, action_seq_len=len(action_seq), response=response)

                    prefetch_lead_steps = compute_prefetch_lead_steps(
                        frame_rate=args.frame_rate,
                        action_seq_len=len(action_seq),
                        round_trip_ms=last_round_trip_ms,
                        min_lead_steps=args.prefetch_lead_steps,
                    )

                    for step_idx, step_action in enumerate(action_seq):
                        if timestep >= args.max_publish_step or (not ros_proc.is_alive()):
                            break

                        if maybe_enter_manual_home_pause(args, ros_proc, shm_dict, manual_home_command, action_dim):
                            episode_restart_requested = True
                            break

                        remaining_steps = len(action_seq) - step_idx - 1
                        if (
                            next_response is None
                            and not prefetcher.has_inflight()
                            and remaining_steps <= prefetch_lead_steps
                            and timestep + 1 < args.max_publish_step
                        ):
                            next_obs = read_observation_snapshot(args, shm_dict, shapes)
                            prefetcher.submit(
                                images=next_obs["images"],
                                qpos=next_obs.get("qpos", np.zeros((args.state_dim,), dtype=np.float32)),
                                timestep=timestep,
                                prompt=args.prompt,
                            )

                        maybe_result = prefetcher.poll()
                        if maybe_result is not None:
                            if maybe_result.get("ok", False):
                                next_response = maybe_result.get("response")
                            else:
                                print(f"[MagicBot] background prefetch failed: {maybe_result['error']}")
                                reconnect_exc = maybe_result.get("reconnect_error")
                                if reconnect_exc is not None:
                                    print(f"[MagicBot] reconnect failed: {reconnect_exc}")

                        action = step_action
                        robot_action(action, shm_dict)
                        timestep += 1
                        exec_rate.sleep()
            finally:
                prefetcher.stop()
        else:
            client = build_client()
            client.reset()
            while timestep < args.max_publish_step and ros_proc.is_alive():
                if maybe_enter_manual_home_pause(args, ros_proc, shm_dict, manual_home_command, action_dim):
                    episode_restart_requested = True
                    break

                obs_dict = read_observation_snapshot(args, shm_dict, shapes)

                try:
                    response = client.infer_step(
                        images=obs_dict["images"],
                        qpos=obs_dict.get("qpos", np.zeros((args.state_dim,), dtype=np.float32)),
                        timestep=timestep,
                        prompt=args.prompt,
                    )
                except Exception as exc:
                    action = write_zero_action(
                        shm_dict,
                        action_dim,
                        f"[SafeGuard] Remote inference failed, forcing zero action for this cycle: {exc}",
                    )
                    try:
                        client.close()
                    except Exception:
                        pass
                    try:
                        client = build_client()
                        client.reset()
                    except Exception as reconnect_exc:
                        print(f"[MagicBot] reconnect failed: {reconnect_exc}")
                    timestep += 1
                    continue

                if response is None:
                    action = write_zero_action(
                        shm_dict,
                        action_dim,
                        "[SafeGuard] MagicBot server returned nothing, forcing zero action for this cycle.",
                    )
                    timestep += 1
                    continue

                if first_inference:
                    if not maybe_run_first_safety_check(args, response, obs_dict, action_dim):
                        write_zero_action(shm_dict, action_dim, "[SafeGuard] First safety check failed, zeroing action buffer.")
                        return
                    first_inference = False

                action_seq = extract_action_sequence(response, action_dim)
                if len(action_seq) == 0:
                    action = write_zero_action(
                        shm_dict,
                        action_dim,
                        "[SafeGuard] MagicBot returned an empty action sequence, forcing zero action for this cycle.",
                    )
                    timestep += 1
                    continue

                chunk_idx += 1
                print_timing_log(args, chunk_idx=chunk_idx, action_seq_len=len(action_seq), response=response)

                for step_action in action_seq:
                    if timestep >= args.max_publish_step or (not ros_proc.is_alive()):
                        break

                    if maybe_enter_manual_home_pause(args, ros_proc, shm_dict, manual_home_command, action_dim):
                        episode_restart_requested = True
                        break

                    action = step_action
                    robot_action(action, shm_dict)
                    timestep += 1
                    exec_rate.sleep()

            client.close()

        if episode_restart_requested:
            action = np.zeros((action_dim,), dtype=np.float32)
            first_inference = True
            chunk_idx = 0
            print(
                "[Manual Home] Rollout state has been reset. "
                "The next chunk will be treated as a fresh start.\n"
                "[Manual Home] You will see the first-chunk confirmation again before execution."
            )

        if args.use_base and action_dim > 19:
            action[16] = 0
            action[17] = 0
            action[19] = 0

        robot_action(action, shm_dict)


def parse_args(known=False):
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_publish_step", type=int, default=10000, help="Max publish step.")
    parser.add_argument(
        "--data",
        type=str,
        default=str(ROBOT_RUNTIME_ROOT / "data" / "config.yaml"),
        help="Robot config YAML.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--camera_names",
        nargs="+",
        type=str,
        choices=["head", "left_wrist", "right_wrist"],
        default=["head", "left_wrist", "right_wrist"],
        help="Camera names to use.",
    )
    parser.add_argument("--use_depth_image", action="store_true", help="Enable depth image subscriptions.")
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
        "--inference_mode",
        choices=["sync", "async"],
        default="sync",
        help="`sync` uses the original chunk-by-chunk blocking inference loop. `async` enables background prefetch.",
    )
    parser.add_argument(
        "--prefetch_lead_steps",
        type=int,
        default=10,
        help="Minimum number of remaining control steps before we start prefetching the next action chunk.",
    )
    parser.add_argument(
        "--log_timing_every",
        type=int,
        default=5,
        help="Print server/client timing every N chunks. The first 3 chunks are always logged.",
    )
    parser.add_argument(
        "--safe_stop_home_arms",
        action="store_true",
        help="Before lowering the base on shutdown, first publish an all-zero home pose to both arms.",
    )
    parser.add_argument(
        "--safe_stop_home_publish_steps",
        type=int,
        default=180,
        help="Number of cycles to publish the arm-home pose before lowering the base.",
    )
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
    manual_home_command = mp.Value("i", 0)

    ros_proc = mp.Process(
        target=ros_process,
        args=(args, config, meta_queue, connected_event, start_event, shm_ready_event, manual_home_command),
    )
    ros_proc.start()

    if not connected_event.wait(timeout=15.0):
        request_graceful_ros_shutdown(ros_proc, args)
        raise RuntimeError("ROS process did not reach the connected state in time.")

    input("Enter any key to continue :")
    start_event.set()

    try:
        shapes = meta_queue.get(timeout=20.0)
    except queue.Empty as exc:
        request_graceful_ros_shutdown(ros_proc, args)
        raise RuntimeError("Timed out waiting for ROS observation metadata.") from exc

    if isinstance(shapes, dict) and "error" in shapes:
        request_graceful_ros_shutdown(ros_proc, args)
        raise RuntimeError(str(shapes["error"]))

    shm_name_dict = make_shm_name_dict(args, shapes)
    meta_queue.put(shm_name_dict)
    if not shm_ready_event.wait(timeout=10.0):
        request_graceful_ros_shutdown(ros_proc, args)
        raise RuntimeError("Timed out waiting for shared-memory setup.")

    shm_dict = connect_shm_dict(shm_name_dict, shapes, shapes["dtypes"], config)

    try:
        inference_process(args, config, shm_dict, shapes, ros_proc, manual_home_command)
    except KeyboardInterrupt:
        print("[Shutdown] KeyboardInterrupt received in parent process, requesting graceful ROS shutdown...")
    finally:
        request_graceful_ros_shutdown(ros_proc, args)
        for shm, _, _ in shm_dict.values():
            try:
                shm.close()
            except FileNotFoundError:
                pass
            try:
                shm.unlink()
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    main(parse_args())
