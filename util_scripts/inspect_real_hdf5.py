#!/usr/bin/env python
"""Inspect real-robot HDF5 episodes before conversion to LeRobot v3.0.

The goal is to make dataset conversion and later CubeV2 finetuning easier by
using the exact same HDF5 reading and image-decoding logic as the converter.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import h5py
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError("h5py is required for HDF5 inspection.") from exc

from real_hdf5_utils import (
    DEFAULT_INSTRUCTION,
    IMAGE_LAYOUTS,
    build_camera_source_keys,
    collect_h5_tree_lines,
    decode_image_frame,
    derive_instruction,
    h5_attr_to_text,
    infer_episode_spec,
    iter_h5_datasets,
    load_h5_array,
    resolve_sample_indices,
    validate_episode_lengths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect real-robot HDF5 episodes thoroughly before LeRobot conversion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="A single HDF5 episode path or a directory containing episode HDF5 files.",
    )
    parser.add_argument(
        "--episode-glob",
        type=str,
        default="*.hdf5",
        help="Glob used to discover episodes when input_path is a directory.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=1,
        help="Maximum number of files to inspect when input_path is a directory.",
    )
    parser.add_argument(
        "--image-layout",
        type=str,
        choices=sorted(IMAGE_LAYOUTS.keys()),
        default="head_left_right",
        help="Expected output image layout for later conversion.",
    )
    parser.add_argument(
        "--state-key",
        type=str,
        default="observations/qpos",
        help="HDF5 dataset path treated as observation.state.",
    )
    parser.add_argument(
        "--action-key",
        type=str,
        default="action",
        help="HDF5 dataset path treated as action.",
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
        help="HDF5 dataset path used as the left camera stream.",
    )
    parser.add_argument(
        "--right-key",
        type=str,
        default="observations/images/right_wrist",
        help="HDF5 dataset path used as the right camera stream.",
    )
    parser.add_argument(
        "--task-attr",
        type=str,
        default="task",
        help="Root HDF5 attribute inspected for task text.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=DEFAULT_INSTRUCTION,
        help="Instruction/task text that will be written during conversion.",
    )
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="+",
        default=[0, -1],
        help="Frame indices to decode for each camera. Negative values index from the end.",
    )
    parser.add_argument(
        "--raw-image-shape",
        type=int,
        nargs=3,
        metavar=("HEIGHT", "WIDTH", "CHANNELS"),
        default=None,
        help="Fallback H W C shape for flattened raw image arrays.",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="Optional directory to save decoded preview PNG files.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional JSON file path to save the inspection report.",
    )
    parser.add_argument(
        "--vector-preview-dim",
        type=int,
        default=8,
        help="How many values from the first frame to preview for numeric vectors.",
    )
    return parser.parse_args()


def discover_files(input_path: Path, episode_glob: str, max_files: int | None) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    files = sorted(path for path in input_path.rglob(episode_glob) if path.is_file())
    if max_files is not None:
        files = files[:max_files]
    return files


def to_jsonable(value: Any):
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def summarize_numeric_array(array: np.ndarray, preview_dim: int) -> dict[str, Any]:
    summary = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
    if array.size == 0:
        return summary

    summary.update(
        {
            "min": float(np.min(array)),
            "max": float(np.max(array)),
            "mean": float(np.mean(array)),
        }
    )

    if array.ndim >= 2:
        preview = np.asarray(array[0]).reshape(-1)[:preview_dim]
        summary["frame0_preview"] = preview.astype(np.float64).tolist()
    else:
        preview = np.asarray(array).reshape(-1)[:preview_dim]
        summary["preview"] = preview.astype(np.float64).tolist()
    return summary


def summarize_action_state_alignment(state_array: np.ndarray, action_array: np.ndarray) -> dict[str, Any]:
    if state_array.ndim != 2 or action_array.ndim != 2 or state_array.shape[1] != action_array.shape[1]:
        return {
            "available": False,
            "reason": "state/action are not aligned 2D arrays with the same feature dimension",
        }

    same_step = np.abs(action_array - state_array)
    summary: dict[str, Any] = {
        "available": True,
        "same_step_mean_abs": float(same_step.mean()),
        "same_step_max_abs": float(same_step.max()),
    }

    if state_array.shape[0] > 1 and action_array.shape[0] > 1:
        next_state = state_array[1:]
        curr_action = action_array[:-1]
        next_step = np.abs(curr_action - next_state)
        summary["next_state_mean_abs"] = float(next_step.mean())
        summary["next_state_max_abs"] = float(next_step.max())

    return summary


def summarize_image_array(
    image_array: np.ndarray,
    *,
    sample_indices: list[int],
    raw_image_shape: tuple[int, int, int] | None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "raw_shape": list(image_array.shape),
        "raw_dtype": str(image_array.dtype),
    }

    if image_array.ndim == 2 and image_array.dtype == np.uint8:
        byte_lengths = np.count_nonzero(image_array, axis=1)
        summary["encoded_nonzero_bytes"] = {
            "min": int(byte_lengths.min()),
            "max": int(byte_lengths.max()),
            "mean": float(byte_lengths.mean()),
        }

    decoded_samples = []
    for idx in sample_indices:
        decoded = decode_image_frame(image_array[idx], raw_image_shape)
        decoded_samples.append(
            {
                "frame_index": int(idx),
                "decoded_shape": list(decoded.shape),
                "decoded_dtype": str(decoded.dtype),
                "min": int(decoded.min()),
                "max": int(decoded.max()),
                "mean": float(decoded.mean()),
            }
        )
    summary["decoded_samples"] = decoded_samples
    return summary


def summarize_time_aligned_datasets(
    h5_file: h5py.File,
    *,
    episode_len: int,
    preview_dim: int,
    exclude_paths: set[str],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path, dataset in iter_h5_datasets(h5_file):
        if path in exclude_paths:
            continue
        if len(dataset.shape) == 0 or dataset.shape[0] != episode_len:
            continue
        if not np.issubdtype(dataset.dtype, np.number):
            continue
        array = np.asarray(dataset)
        summaries.append(
            {
                "path": path,
                **summarize_numeric_array(array, preview_dim),
            }
        )
    return summaries


def save_preview_images(
    preview_dir: Path,
    episode_path: Path,
    image_arrays: dict[str, np.ndarray],
    *,
    sample_indices: list[int],
    raw_image_shape: tuple[int, int, int] | None,
) -> list[str]:
    preview_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []
    episode_stem = episode_path.stem

    for camera_name, image_array in image_arrays.items():
        for idx in sample_indices:
            decoded = decode_image_frame(image_array[idx], raw_image_shape)
            out_path = preview_dir / f"{episode_stem}_{camera_name}_frame{idx}.png"
            Image.fromarray(decoded).save(out_path)
            saved_paths.append(str(out_path))
    return saved_paths


def print_report(report: dict[str, Any]) -> None:
    print(f"File: {report['file']}")
    print(f"Instruction: {report['instruction']}")
    print(f"Root keys: {report['root_keys']}")
    print("  @attrs:")
    for key, value in report["root_attrs"].items():
        print(f"    - {key}: {value!r}")

    print("\nTree:")
    for line in report["tree_lines"]:
        print(f"  {line}")

    print("\nSelected Fields:")
    print(f"  - state_key: {report['selected_keys']['state_key']}")
    print(f"  - action_key: {report['selected_keys']['action_key']}")
    for camera_name, source_key in report["selected_keys"]["camera_keys"].items():
        print(f"  - {camera_name}_key: {source_key}")

    print("\nEpisode Summary:")
    print(f"  - frame_count: {report['episode_summary']['frame_count']}")
    print(f"  - state_dim: {report['episode_summary']['state_dim']}")
    print(f"  - action_dim: {report['episode_summary']['action_dim']}")

    print("\nState Summary:")
    for key, value in report["state_summary"].items():
        print(f"  - {key}: {value}")

    print("\nAction Summary:")
    for key, value in report["action_summary"].items():
        print(f"  - {key}: {value}")

    print("\nAction-State Alignment:")
    for key, value in report["action_state_alignment"].items():
        print(f"  - {key}: {value}")

    print("\nCamera Summary:")
    for camera_name, camera_summary in report["image_summaries"].items():
        print(f"  - {camera_name}:")
        print(f"    raw_shape: {camera_summary['raw_shape']}")
        print(f"    raw_dtype: {camera_summary['raw_dtype']}")
        if "encoded_nonzero_bytes" in camera_summary:
            print(f"    encoded_nonzero_bytes: {camera_summary['encoded_nonzero_bytes']}")
        for sample in camera_summary["decoded_samples"]:
            print(f"    decoded_sample: {sample}")

    print("\nOther Time-Aligned Numeric Datasets:")
    for item in report["aligned_numeric_datasets"]:
        print(
            f"  - {item['path']}: shape={item['shape']} dtype={item['dtype']} "
            f"min={item.get('min')} max={item.get('max')} mean={item.get('mean')}"
        )

    print("\nSuggested Converter Command:")
    print(report["suggested_converter_command"])

    if report.get("saved_previews"):
        print("\nSaved Preview Images:")
        for path in report["saved_previews"]:
            print(f"  - {path}")


def inspect_file(file_path: Path, args: argparse.Namespace, input_root: Path) -> dict[str, Any]:
    raw_image_shape = tuple(args.raw_image_shape) if args.raw_image_shape else None
    camera_source_keys = build_camera_source_keys(
        args.image_layout,
        head_key=args.head_key,
        left_key=args.left_key,
        right_key=args.right_key,
    )

    with h5py.File(file_path, "r") as h5_file:
        state_array = load_h5_array(h5_file, args.state_key)
        action_array = load_h5_array(h5_file, args.action_key)
        image_arrays = {
            camera_name: load_h5_array(h5_file, source_key)
            for camera_name, source_key in camera_source_keys.items()
        }
        episode_len = validate_episode_lengths(file_path, state_array, action_array, image_arrays)
        spec = infer_episode_spec(
            file_path,
            state_key=args.state_key,
            action_key=args.action_key,
            camera_source_keys=camera_source_keys,
            raw_image_shape=raw_image_shape,
        )
        sample_indices = resolve_sample_indices(episode_len, args.sample_indices)
        instruction = derive_instruction(
            h5_file,
            file_path,
            input_root=input_root,
            instruction=args.instruction,
            task_attr=args.task_attr,
        )

        exclude_paths = {
            args.state_key if args.state_key.startswith("/") else f"/{args.state_key}",
            args.action_key if args.action_key.startswith("/") else f"/{args.action_key}",
        }
        exclude_paths.update(
            source_key if source_key.startswith("/") else f"/{source_key}"
            for source_key in camera_source_keys.values()
        )

        root_attrs = {key: h5_attr_to_text(value) for key, value in h5_file.attrs.items()}
        report: dict[str, Any] = {
            "file": str(file_path),
            "instruction": instruction,
            "root_keys": list(h5_file.keys()),
            "root_attrs": root_attrs,
            "tree_lines": collect_h5_tree_lines(h5_file),
            "selected_keys": {
                "state_key": args.state_key,
                "action_key": args.action_key,
                "camera_keys": dict(camera_source_keys),
            },
            "episode_summary": {
                "frame_count": episode_len,
                "state_dim": spec.state_dim,
                "action_dim": spec.action_dim,
                "image_shapes": {k: list(v) for k, v in spec.image_shapes.items()},
            },
            "state_summary": summarize_numeric_array(np.asarray(state_array), args.vector_preview_dim),
            "action_summary": summarize_numeric_array(np.asarray(action_array), args.vector_preview_dim),
            "action_state_alignment": summarize_action_state_alignment(
                np.asarray(state_array),
                np.asarray(action_array),
            ),
            "image_summaries": {
                camera_name: summarize_image_array(
                    np.asarray(image_array),
                    sample_indices=sample_indices,
                    raw_image_shape=raw_image_shape,
                )
                for camera_name, image_array in image_arrays.items()
            },
            "aligned_numeric_datasets": summarize_time_aligned_datasets(
                h5_file,
                episode_len=episode_len,
                preview_dim=args.vector_preview_dim,
                exclude_paths=exclude_paths,
            ),
        }

    suggested = [
        "python util_scripts/convert_real_hdf5_to_lerobot.py",
        f'--input-root "{input_root}"',
        f'--output-dir "{input_root / "lerobot_v30"}"',
        '--robot-type "real_lift2"',
        f'--task "{instruction}"',
        f'--image-layout "{args.image_layout}"',
        f"--fps 15",
        f'--state-key "{args.state_key}"',
        f'--action-key "{args.action_key}"',
        f'--head-key "{args.head_key}"',
        f'--left-key "{args.left_key}"',
    ]
    if args.image_layout == "head_left_right":
        suggested.append(f'--right-key "{args.right_key}"')
    if raw_image_shape is not None:
        suggested.append(
            "--raw-image-shape "
            + " ".join(str(v) for v in raw_image_shape)
        )
    report["suggested_converter_command"] = " \\\n  ".join(suggested)

    if args.preview_dir is not None:
        report["saved_previews"] = save_preview_images(
            args.preview_dir,
            file_path,
            image_arrays,
            sample_indices=sample_indices,
            raw_image_shape=raw_image_shape,
        )

    return report


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    files = discover_files(args.input_path, args.episode_glob, args.max_files)
    if not files:
        raise FileNotFoundError(f"No files found for {args.input_path}")

    input_root = args.input_path if args.input_path.is_dir() else args.input_path.parent
    reports = [inspect_file(path, args, input_root) for path in files]

    for idx, report in enumerate(reports, start=1):
        if len(reports) > 1:
            print(f"\n========== Report {idx}/{len(reports)} ==========\n")
        print_report(report)

    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        with args.report_json.open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(reports if len(reports) > 1 else reports[0]), f, indent=2, ensure_ascii=False)
        logging.info("Wrote JSON report to %s", args.report_json)


if __name__ == "__main__":
    main()
