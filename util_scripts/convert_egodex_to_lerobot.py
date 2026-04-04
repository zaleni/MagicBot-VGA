#!/usr/bin/env python
"""Convert raw EgoDex MP4+HDF5 episodes into a LeRobot v3 dataset for CubeV2.

This converter is intentionally tailored to the current CubeV2 training design:

- keep the single egocentric RGB stream as `observation.image`
- keep the language task as `task`
- write dummy zero `observation.state` / `action` placeholders

The resulting dataset is meant to be mixed with robot datasets after conversion.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError("opencv-python-headless is required for EgoDex conversion.") from exc

try:
    import h5py
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError("h5py is required for EgoDex conversion.") from exc

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

from lerobot.datasets.lerobot_dataset import LeRobotDataset

DEFAULT_FPS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw EgoDex MP4+HDF5 episodes into a local LeRobot v3 dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Root directory containing EgoDex task folders and paired *.hdf5/*.mp4 files.",
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
        "--robot-type",
        type=str,
        default="egodex_v",
        help="robot_type written into meta/info.json.",
    )
    parser.add_argument(
        "--dummy-dim",
        type=int,
        default=1,
        help="Dimension of the dummy state/action placeholder vectors.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="Source EgoDex video FPS before frame subsampling.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Keep one frame every N frames. Effective dataset FPS becomes fps / frame_stride.",
    )
    parser.add_argument(
        "--task-regex",
        type=str,
        default=None,
        help="Optional regex applied to the relative task path for filtering episodes.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Convert at most this many paired episodes after filtering.",
    )
    parser.add_argument(
        "--max-frames-per-episode",
        type=int,
        default=None,
        help="Optional cap on converted frames per episode after stride.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Emit an info log after every N converted episodes.",
    )
    return parser.parse_args()


def normalize_h5_attr(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return normalize_h5_attr(value.item())
    return str(value)


def select_task_description(h5_root: h5py.File, fallback_task: str) -> str:
    attrs = h5_root.attrs
    if "llm_description" not in attrs:
        return fallback_task

    llm_type = normalize_h5_attr(attrs.get("llm_type", ""))
    if llm_type == "reversible":
        which = normalize_h5_attr(attrs.get("which_llm_description", "1"))
        if which == "2" and "llm_description2" in attrs:
            return normalize_h5_attr(attrs["llm_description2"])
    return normalize_h5_attr(attrs["llm_description"])


def discover_episode_pairs(
    input_root: Path,
    *,
    task_regex: str | None,
    max_episodes: int | None,
) -> list[tuple[Path, Path, str]]:
    pattern = re.compile(task_regex) if task_regex else None
    pairs: list[tuple[Path, Path, str]] = []

    for h5_path in sorted(input_root.rglob("*.hdf5")):
        mp4_path = h5_path.with_suffix(".mp4")
        if not mp4_path.is_file():
            logging.warning("Skipping %s because paired MP4 is missing.", h5_path)
            continue

        task_rel = str(h5_path.parent.relative_to(input_root)).replace("\\", "/")
        if pattern and not pattern.search(task_rel):
            continue

        pairs.append((h5_path, mp4_path, task_rel))
        if max_episodes is not None and len(pairs) >= max_episodes:
            break

    return pairs


def get_video_shape(mp4_path: Path) -> tuple[int, int]:
    capture = cv2.VideoCapture(str(mp4_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {mp4_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    if width <= 0 or height <= 0:
        raise RuntimeError(f"Could not read frame size from video: {mp4_path}")

    return height, width


def build_features(height: int, width: int, dummy_dim: int) -> dict[str, dict]:
    dummy_names = [f"dummy_{i}" for i in range(dummy_dim)]
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (dummy_dim,),
            "names": dummy_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (dummy_dim,),
            "names": dummy_names,
        },
        "observation.image": {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
    }


def iter_progress(items: Iterable, **kwargs):
    if tqdm is None:
        return items
    return tqdm(items, **kwargs)


def convert_episode(
    dataset: LeRobotDataset,
    h5_path: Path,
    mp4_path: Path,
    task_rel: str,
    *,
    frame_stride: int,
    max_frames_per_episode: int | None,
    dummy_state: np.ndarray,
) -> tuple[int, str]:
    with h5py.File(h5_path, "r") as root:
        episode_task = select_task_description(root, fallback_task=task_rel)
        episode_len = int(root["/transforms/camera"].shape[0])

    if episode_len <= 0:
        raise ValueError(f"Episode has zero frames: {h5_path}")

    capture = cv2.VideoCapture(str(mp4_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {mp4_path}")

    frame_index = 0
    output_frames = 0
    while frame_index < episode_len:
        ok, frame_bgr = capture.read()
        if not ok:
            logging.warning(
                "Video ended early for %s at source frame %d (expected at least %d frames).",
                mp4_path,
                frame_index,
                episode_len,
            )
            break

        if frame_index % frame_stride != 0:
            frame_index += 1
            continue

        if max_frames_per_episode is not None and output_frames >= max_frames_per_episode:
            break

        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame = {
            "observation.image": image_rgb,
            "observation.state": dummy_state.copy(),
            "action": dummy_state.copy(),
            "task": episode_task,
        }
        dataset.add_frame(frame)
        output_frames += 1
        frame_index += 1

    capture.release()

    if output_frames == 0:
        raise ValueError(f"No frames were converted for episode: {h5_path}")

    dataset.save_episode()
    return output_frames, episode_task


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be >= 1")
    if args.dummy_dim <= 0:
        raise ValueError("--dummy-dim must be >= 1")
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    if not args.input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")

    pairs = discover_episode_pairs(
        args.input_root,
        task_regex=args.task_regex,
        max_episodes=args.max_episodes,
    )
    if not pairs:
        raise RuntimeError(f"No paired EgoDex episodes found under {args.input_root}")

    first_h, first_w = get_video_shape(pairs[0][1])
    features = build_features(first_h, first_w, args.dummy_dim)
    effective_fps = args.fps / args.frame_stride
    if abs(effective_fps - round(effective_fps)) > 1e-6:
        raise ValueError(
            f"fps/frame_stride must be an integer for LeRobot metadata, got {args.fps}/{args.frame_stride}"
        )

    repo_id = args.repo_id or args.output_dir.name
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(round(effective_fps)),
        features=features,
        root=args.output_dir,
        robot_type=args.robot_type,
        use_videos=True,
        batch_encoding_size=1,
    )

    dummy_state = np.zeros((args.dummy_dim,), dtype=np.float32)
    total_frames = 0
    total_episodes = 0
    converted_tasks: set[str] = set()
    try:
        for idx, (h5_path, mp4_path, task_rel) in enumerate(
            iter_progress(pairs, desc="Converting EgoDex episodes"), start=1
        ):
            frames_written, task_name = convert_episode(
                dataset,
                h5_path,
                mp4_path,
                task_rel,
                frame_stride=args.frame_stride,
                max_frames_per_episode=args.max_frames_per_episode,
                dummy_state=dummy_state,
            )
            total_frames += frames_written
            total_episodes += 1
            converted_tasks.add(task_name)

            if idx % args.log_every == 0:
                logging.info(
                    "Converted %d episodes / %d frames so far into %s",
                    total_episodes,
                    total_frames,
                    args.output_dir,
                )
    except Exception:
        dataset.finalize()
        raise

    dataset.finalize()
    logging.info(
        "Finished conversion: %d episodes, %d frames, %d unique task strings -> %s",
        total_episodes,
        total_frames,
        len(converted_tasks),
        args.output_dir,
    )


if __name__ == "__main__":
    main()
