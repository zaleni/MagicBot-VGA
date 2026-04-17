#!/usr/bin/env python
"""Convert real-robot HDF5 episodes into a local LeRobot v3.0 dataset.

This script is intended for CubeV2-style finetuning where the source data does
not already follow an existing converter schema such as LIBERO. It uses the
official LeRobotDataset v3.0 writer so the output can be consumed by the normal
training pipeline in this repository.

The current defaults target the user-provided schema:

- one HDF5 file per episode
- state from ``observations/qpos``
- action from ``action``
- images from ``observations/images/{head,left_wrist,right_wrist}``

Example:

python util_scripts/convert_real_hdf5_to_lerobot.py ^
  --input-root /path/to/datasets ^
  --output-dir /path/to/output/my_real_robot_lerobot ^
  --task "pick up the object" ^
  --frame-stride 2 ^
  --fps 15
"""

from __future__ import annotations

import argparse
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import h5py
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError("h5py is required for HDF5 conversion.") from exc

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from real_hdf5_utils import (
    DEFAULT_INSTRUCTION,
    IMAGE_LAYOUTS,
    build_camera_source_keys,
    decode_image_frame,
    derive_instruction,
    infer_episode_spec,
    load_h5_array,
    make_vector_names,
    validate_episode_lengths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert real-robot HDF5 episodes into a local LeRobot v3.0 dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory containing one or more episode_*.hdf5 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for the LeRobot dataset. Must not already exist.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Logical dataset name stored in LeRobot metadata. Defaults to output directory name.",
    )
    parser.add_argument(
        "--episode-glob",
        type=str,
        default="*.hdf5",
        help="Glob used to find episode files recursively under input-root.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Convert at most this many episodes after filtering.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Effective FPS written into LeRobot metadata after any frame subsampling.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Keep one frame every N frames. Example: source 60Hz -> target 30Hz uses --frame-stride 2 --fps 30.",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="real_lift2",
        help=(
            "robot_type written into meta/info.json. "
            "For the current 14D real dual-arm setup (left arm + left gripper + right arm + right gripper), "
            "real_lift2 is the recommended default."
        ),
    )
    parser.add_argument(
        "--image-layout",
        type=str,
        choices=sorted(IMAGE_LAYOUTS.keys()),
        default="head_left_right",
        help="Output image-key layout used in the LeRobot dataset.",
    )
    parser.add_argument(
        "--state-key",
        type=str,
        default="observations/qpos",
        help="HDF5 dataset path used as observation.state.",
    )
    parser.add_argument(
        "--action-key",
        type=str,
        default="action",
        help="HDF5 dataset path used as action.",
    )
    parser.add_argument(
        "--head-key",
        type=str,
        default="observations/images/head",
        help="HDF5 dataset path used as the head camera stream.",
    )
    parser.add_argument(
        "--left-key",
        type=str,
        default="observations/images/left_wrist",
        help="HDF5 dataset path used as the left/wrist camera stream.",
    )
    parser.add_argument(
        "--right-key",
        type=str,
        default="observations/images/right_wrist",
        help="HDF5 dataset path used as the right camera stream.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_INSTRUCTION,
        help="Task/instruction text written for every episode.",
    )
    parser.add_argument(
        "--task-attr",
        type=str,
        default="task",
        help="Root HDF5 attribute used to source task text when --task is not provided.",
    )
    parser.add_argument(
        "--use-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store visual features as videos instead of images in the LeRobot dataset.",
    )
    parser.add_argument(
        "--image-writer-processes",
        type=int,
        default=0,
        help="Async LeRobot image-writer processes.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=8,
        help="Async LeRobot image-writer threads.",
    )
    parser.add_argument(
        "--batch-encoding-size",
        type=int,
        default=1,
        help=(
            "Number of episodes to accumulate before LeRobot batch video encoding. "
            "For multi-camera real-robot conversion, 1 is usually the safest default and "
            "lets LeRobot parallelize encoding across cameras per episode."
        ),
    )
    parser.add_argument(
        "--raw-image-shape",
        type=int,
        nargs=3,
        metavar=("HEIGHT", "WIDTH", "CHANNELS"),
        default=None,
        help=(
            "Optional fallback shape for raw uncompressed image arrays that cannot be decoded as JPEG/PNG bytes. "
            "Expected order is H W C."
        ),
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Emit a progress log after every N converted episodes.",
    )
    parser.add_argument(
        "--validate-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the converted dataset back through LeRobotDataset and validate a sample after conversion.",
    )
    return parser.parse_args()


def iter_progress(items: Iterable, **kwargs):
    if tqdm is None:
        return items
    return tqdm(items, **kwargs)


def build_features(
    state_dim: int,
    action_dim: int,
    image_shapes: OrderedDict[str, tuple[int, int, int]],
    image_layout: str,
    use_videos: bool,
) -> dict[str, dict]:
    image_dtype = "video" if use_videos else "image"
    features: dict[str, dict] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": make_vector_names("state", state_dim),
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": make_vector_names("action", action_dim),
        },
    }

    for camera_name, output_key in IMAGE_LAYOUTS[image_layout].items():
        height, width, channels = image_shapes[camera_name]
        features[output_key] = {
            "dtype": image_dtype,
            "shape": (height, width, channels),
            "names": ["height", "width", "channels"],
        }

    return features


def discover_episode_files(input_root: Path, episode_glob: str, max_episodes: int | None) -> list[Path]:
    paths = sorted(input_root.rglob(episode_glob))
    paths = [path for path in paths if path.is_file()]
    if max_episodes is not None:
        paths = paths[:max_episodes]
    return paths


def create_dataset(
    output_dir: Path,
    repo_id: str,
    fps: int,
    robot_type: str,
    features: dict[str, dict],
    use_videos: bool,
    image_writer_processes: int,
    image_writer_threads: int,
    batch_encoding_size: int,
) -> LeRobotDataset:
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=output_dir,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=use_videos,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
        batch_encoding_size=batch_encoding_size,
    )


def convert_episode(
    dataset: LeRobotDataset,
    episode_path: Path,
    args: argparse.Namespace,
) -> int:
    camera_source_keys = build_camera_source_keys(
        args.image_layout,
        head_key=args.head_key,
        left_key=args.left_key,
        right_key=args.right_key,
    )
    output_image_keys = IMAGE_LAYOUTS[args.image_layout]
    raw_image_shape = tuple(args.raw_image_shape) if args.raw_image_shape else None

    with h5py.File(episode_path, "r") as h5_file:
        state_array = load_h5_array(h5_file, args.state_key).astype(np.float32, copy=False)
        action_array = load_h5_array(h5_file, args.action_key).astype(np.float32, copy=False)
        image_arrays = OrderedDict(
            (camera_name, load_h5_array(h5_file, source_key))
            for camera_name, source_key in camera_source_keys.items()
        )
        episode_len = validate_episode_lengths(episode_path, state_array, action_array, image_arrays)
        task_text = derive_instruction(
            h5_file,
            episode_path,
            input_root=args.input_root,
            instruction=args.task,
            task_attr=args.task_attr,
        )

        kept_frames = 0
        for frame_idx in range(0, episode_len, args.frame_stride):
            frame = {
                "observation.state": state_array[frame_idx],
                "action": action_array[frame_idx],
                "task": task_text,
            }
            for camera_name, output_key in output_image_keys.items():
                decoded = decode_image_frame(image_arrays[camera_name][frame_idx], raw_image_shape)
                frame[output_key] = decoded
            dataset.add_frame(frame)
            kept_frames += 1

    dataset.save_episode()
    return kept_frames


def validate_output_dataset(
    output_dir: Path,
    repo_id: str,
    fps: int,
    robot_type: str,
    use_videos: bool,
) -> None:
    dataset = LeRobotDataset(repo_id, root=output_dir, episodes=[0])

    if dataset.meta.robot_type != robot_type:
        raise RuntimeError(
            f"Validation failed: robot_type mismatch. Expected {robot_type}, got {dataset.meta.robot_type}."
        )
    if dataset.fps != fps:
        raise RuntimeError(f"Validation failed: fps mismatch. Expected {fps}, got {dataset.fps}.")
    if dataset.meta.total_episodes <= 0 or dataset.meta.total_frames <= 0:
        raise RuntimeError("Validation failed: converted dataset is empty.")
    if use_videos and len(dataset.meta.video_keys) == 0:
        raise RuntimeError("Validation failed: expected video-backed cameras, but no video keys were found.")
    if not use_videos and len(dataset.meta.image_keys) == 0:
        raise RuntimeError("Validation failed: expected image-backed cameras, but no image keys were found.")

    sample = dataset[0]
    required_keys = {"observation.state", "action", "task", "robot_type", *dataset.meta.camera_keys}
    missing = sorted(key for key in required_keys if key not in sample)
    if missing:
        raise RuntimeError(f"Validation failed: sample 0 is missing keys: {missing}")

    logging.info(
        "Validated LeRobot dataset: episodes=%s frames=%s cameras=%s storage=%s",
        dataset.meta.total_episodes,
        dataset.meta.total_frames,
        dataset.meta.camera_keys,
        "video" if use_videos else "image",
    )


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    if not args.input_root.is_dir():
        raise NotADirectoryError(f"Input root does not exist or is not a directory: {args.input_root}")

    episode_paths = discover_episode_files(args.input_root, args.episode_glob, args.max_episodes)
    if not episode_paths:
        raise FileNotFoundError(
            f"No HDF5 episodes matching {args.episode_glob!r} were found under {args.input_root}"
        )

    repo_id = args.repo_id or args.output_dir.name
    spec = infer_episode_spec(
        episode_paths[0],
        state_key=args.state_key,
        action_key=args.action_key,
        camera_source_keys=build_camera_source_keys(
            args.image_layout,
            head_key=args.head_key,
            left_key=args.left_key,
            right_key=args.right_key,
        ),
        raw_image_shape=tuple(args.raw_image_shape) if args.raw_image_shape else None,
    )
    features = build_features(
        spec.state_dim,
        spec.action_dim,
        spec.image_shapes,
        args.image_layout,
        args.use_videos,
    )

    logging.info("Creating LeRobot dataset at %s", args.output_dir)
    logging.info("repo_id=%s robot_type=%s fps=%s", repo_id, args.robot_type, args.fps)
    logging.info("frame_stride=%s", args.frame_stride)
    logging.info(
        "state_dim=%s action_dim=%s image_shapes=%s",
        spec.state_dim,
        spec.action_dim,
        dict(spec.image_shapes),
    )

    dataset = create_dataset(
        output_dir=args.output_dir,
        repo_id=repo_id,
        fps=args.fps,
        robot_type=args.robot_type,
        features=features,
        use_videos=args.use_videos,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        batch_encoding_size=args.batch_encoding_size,
    )

    total_frames = 0
    try:
        for episode_idx, episode_path in enumerate(
            iter_progress(episode_paths, desc="Converting episodes", unit="episode"),
            start=1,
        ):
            episode_frames = convert_episode(dataset, episode_path, args)
            total_frames += episode_frames
            if episode_idx % args.log_every == 0 or episode_idx == len(episode_paths):
                logging.info(
                    "Converted %s/%s episodes (%s total frames). Latest: %s",
                    episode_idx,
                    len(episode_paths),
                    total_frames,
                    episode_path,
                )

        if args.use_videos and dataset.batch_encoding_size > 1 and dataset.episodes_since_last_encoding > 0:
            start_ep = dataset.num_episodes - dataset.episodes_since_last_encoding
            logging.info(
                "Flushing final partial video batch for episodes %s to %s",
                start_ep,
                dataset.num_episodes - 1,
            )
            dataset._batch_save_episode_video(start_ep, dataset.num_episodes)
            dataset.episodes_since_last_encoding = 0
    except Exception:
        try:
            dataset.finalize()
        except Exception as finalize_exc:  # pragma: no cover - best effort cleanup
            logging.warning("Failed to finalize partially converted dataset after error: %s", finalize_exc)
        raise

    dataset.finalize()

    if args.validate_output:
        validate_output_dataset(
            output_dir=args.output_dir,
            repo_id=repo_id,
            fps=args.fps,
            robot_type=args.robot_type,
            use_videos=args.use_videos,
        )

    logging.info(
        "Finished conversion: %s episodes, %s frames written to %s",
        len(episode_paths),
        total_frames,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
