"""
Minimal example: convert dataset to the LeRobot format.

CLI Example (using the *arrange_flowers* task as an example):
    python convert_libero_to_lerobot.py \
        --repo-name arrange_flowers_repo \
        --raw-dataset /path/to/arrange_flowers \
        --frame-interval 1 \

Notes:
- If you plan to push to the Hugging Face Hub later, handle that outside this script.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def create_lerobot_dataset(
    repo_name: str,
    robot_type: str,
    fps: float,
    height: int,
    width: int,
) -> LeRobotDataset:
    """
    Create a LeRobot dataset with custom feature schema
    """
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type=robot_type,
        fps=fps,
        features={
            "global_image": {
                "dtype": "image",
                "shape": (height, width, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (height, width, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,), # for joint_positions and gripper width
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,), # for ee_positions and gripper width
                "names": ["actions"],
            },
        },
        image_writer_threads=32,
        image_writer_processes=16,
    )
    return dataset


def process_episode_dir(
    episode_path: Path,
    dataset: LeRobotDataset,
    frame_interval: int,
    prompt: str,
) -> None:
    """
    Process a single episode directory and append frames to the given dataset.

    episode_path : Path
        Episode directory containing `states/states.jsonl` and `videos/*.mp4`.
    dataset : LeRobotDataset
        Target dataset to which frames are added.
    frame_interval : int
        Sampling stride (>=1).
    prompt : str
        Language instruction of this episode.
    """
    # Modify if your dataset consists of bimanual data.
    states_path = episode_path / "states" / "states.jsonl" 
    videos_dir = episode_path / "videos"

    ep_states = load_jsonl(states_path)

    # adjust them to match your dataset’s actual naming.
    wrist_video = cv2.VideoCapture(str(videos_dir / "cam_arm_rgb.mp4")) 
    global_video = cv2.VideoCapture(str(videos_dir / "cam_global_rgb.mp4"))

    wrist_frames_count  = int(wrist_video.get(cv2.CAP_PROP_FRAME_COUNT))
    global_frames_count = int(global_video.get(cv2.CAP_PROP_FRAME_COUNT))
    n_states = len(ep_states)

    # assert all lengths match 
    assert (
        n_states == wrist_frames_count == global_frames_count
    ), (
        f"Mismatch in episode {episode_path.name}: "
        f"states={n_states}, wrist={wrist_frames_count}, "
        f"global={global_frames_count}"
    )

    # write frames to the episode of lerobot dataset
    for idx in range(frame_interval, n_states, frame_interval):
        # Build pose
        pose = np.concatenate(
            (np.asarray(ep_states[idx]["ee_positions"]), [ep_states[idx]["gripper_width"]])
        )
        last_pose = np.concatenate(
            (np.asarray(ep_states[idx - frame_interval]["joint_positions"]),
             [ep_states[idx - frame_interval]["gripper_width"]])
        )

        # Read frames && BGR -> RGB
        # Resize as needed, but update the LeRobot feature shape accordingly.
        _, wrist_image  = wrist_video.read()
        _, global_image = global_video.read()
        
        wrist_image  = cv2.cvtColor(wrist_image,  cv2.COLOR_BGR2RGB)
        global_image = cv2.cvtColor(global_image, cv2.COLOR_BGR2RGB)

        dataset.add_frame(
            {
                "global_image": global_image,
                "wrist_image": wrist_image,
                "state": last_pose.astype(np.float32, copy=False),
                "actions": pose.astype(np.float32, copy=False),
                "task": prompt,
            }
        )

    wrist_video.release()
    global_video.release()
    dataset.save_episode()


def main(
    repo_name: str,
    raw_dataset: Path,
    frame_interval: int = 1,
    overwrite_repo: bool = False,
) -> None:
    """
    Convert a dataset directory into LeRobot format.

    repo_name : str
        Output repo/dataset name (saved under $LEROBOT_HOME / repo_name).
    raw_dataset : Path
        Path to the raw dataset root directory.
    frame_interval : int, default=1
        Sample every N frames (kept identical).
    overwrite_repo : bool, default=False
        If True, remove the existing dataset directory before writing.
    """
    assert frame_interval >= 1, "frame_interval must be >= 1"

    # overwrite repo
    dst_dir = HF_LEROBOT_HOME / repo_name
    if overwrite_repo and dst_dir.exists():
        print(f"removing existing dataset at {dst_dir}")
        shutil.rmtree(dst_dir)

    # Load task_infos
    task_info_path = raw_dataset / "meta" / "task_info.json"
    with task_info_path.open("r", encoding="utf-8") as f:
        task_info = json.load(f)

    robot_type = task_info["task_desc"]["task_tag"][2]  # "ARX5"
    video_info = task_info["video_info"]
    video_info["width"]  = 640  # TODO: derive from task_info or actual videos
    video_info["height"] = 480
    fps = float(video_info["fps"])

    prompt = task_info["task_desc"]["prompt"]

    # Create dataset, define feature in the form you need.
    # - proprio is stored in `state` and actions in `action`
    # - LeRobot assumes that dtype of image data is `image`
    dataset = create_lerobot_dataset(
        repo_name=repo_name,
        robot_type=robot_type,
        fps=fps,
        height=video_info["height"],
        width=video_info["width"],
    )

    # populate the dataset to lerobot dataset
    data_root = raw_dataset / "data"
    for episode_path in data_root.iterdir():
        if not episode_path.is_dir():
            continue
        print(f"Processing episode: {episode_path.name}")
        process_episode_dir(
            episode_path=episode_path,
            dataset=dataset,
            frame_interval=frame_interval,
            prompt=prompt,
        )
    
    dataset.consolidate(run_compute_stats=False)
    print("Done. Dataset saved to: {dst_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a custom dataset to LeRobot format."
    )
    parser.add_argument(
        "--repo-name",
        required=True,
        help="Name of the output dataset (under $LEROBOT_HOME).",
    )
    parser.add_argument(
        "--raw-dataset",
        required=True,
        type=str,
        help="Path to the raw dataset root.",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=1,
        help="Sample every N frames. Default: 1",
    )
    parser.add_argument(
        "--overwrite-repo",
        action="store_true",
        help="Remove existing output directory if it exists.",
    )
    args = parser.parse_args()
    
    main(
        repo_name=args.repo_name,
        raw_dataset=Path(args.raw_dataset),
        frame_interval=args.frame_interval,
        overwrite_repo=args.overwrite_repo,
    )
