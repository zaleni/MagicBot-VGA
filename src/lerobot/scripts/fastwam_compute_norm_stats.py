#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lerobot.policies.MagicBot_R0.configuration_fastwam import MagicBotR0DatasetConfig
from lerobot.policies.MagicBot_R0.dataset_fastwam import build_fastwam_dataset


def _json_arg(value: str, name: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be valid JSON: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute FastWAM/MagicBot_R0 normalization stats in the format expected by FastWAMProcessor."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--repo-id-file", type=str, default=None, help="Text file with one local dataset path per line.")
    source.add_argument("--dataset-dirs", type=str, nargs="+", default=None, help="One or more local LeRobot dataset paths.")

    parser.add_argument("--output-path", type=str, required=True, help="Exact output stats JSON path.")
    parser.add_argument("--action-mode", type=str, choices=["abs", "delta"], default="abs")
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--action-video-freq-ratio", type=int, default=4)
    parser.add_argument("--video-size", type=str, default="[384,320]", help="JSON list, e.g. '[384,320]'.")
    parser.add_argument("--concat-multi-camera", type=str, default="robotwin")
    parser.add_argument("--video-backend", type=str, default=None)

    parser.add_argument("--image-keys", type=str, default='["head","left","right"]')
    parser.add_argument("--image-raw-shapes", type=str, default="[[3,480,640],[3,480,640],[3,480,640]]")
    parser.add_argument("--image-shapes", type=str, default="[[3,240,320],[3,240,320],[3,240,320]]")
    parser.add_argument("--action-keys", type=str, default='["default"]')
    parser.add_argument("--action-raw-shapes", type=str, default="[14]")
    parser.add_argument("--action-shapes", type=str, default="[14]")
    parser.add_argument("--state-keys", type=str, default='["default"]')
    parser.add_argument("--state-raw-shapes", type=str, default="[14]")
    parser.add_argument("--state-shapes", type=str, default="[14]")

    parser.add_argument("--processor-num-output-cameras", type=int, default=3)
    parser.add_argument("--processor-action-output-dim", type=int, default=14)
    parser.add_argument("--processor-proprio-output-dim", type=int, default=14)
    parser.add_argument("--processor-norm-default-mode", type=str, default="z-score")
    parser.add_argument("--processor-use-stepwise-action-norm", action="store_true")
    parser.add_argument(
        "--processor-delta-action-dim-mask",
        type=str,
        default='{"default":[true,true,true,true,true,true,false,true,true,true,true,true,true,false]}',
        help="JSON mask. Only used when --action-mode=delta.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)

    cfg_kwargs: dict[str, Any] = {
        "repo_id": "multidata_from_file",
        "repo_id_file": args.repo_id_file,
        "dataset_dirs": args.dataset_dirs or [],
        "action_mode": args.action_mode,
        "image_keys": _json_arg(args.image_keys, "--image-keys"),
        "image_raw_shapes": _json_arg(args.image_raw_shapes, "--image-raw-shapes"),
        "image_shapes": _json_arg(args.image_shapes, "--image-shapes"),
        "action_keys": _json_arg(args.action_keys, "--action-keys"),
        "action_raw_shapes": _json_arg(args.action_raw_shapes, "--action-raw-shapes"),
        "action_shapes": _json_arg(args.action_shapes, "--action-shapes"),
        "state_keys": _json_arg(args.state_keys, "--state-keys"),
        "state_raw_shapes": _json_arg(args.state_raw_shapes, "--state-raw-shapes"),
        "state_shapes": _json_arg(args.state_shapes, "--state-shapes"),
        "num_frames": args.num_frames,
        "action_video_freq_ratio": args.action_video_freq_ratio,
        "video_size": tuple(_json_arg(args.video_size, "--video-size")),
        "concat_multi_camera": args.concat_multi_camera,
        "processor_num_output_cameras": args.processor_num_output_cameras,
        "processor_action_output_dim": args.processor_action_output_dim,
        "processor_proprio_output_dim": args.processor_proprio_output_dim,
        "processor_norm_default_mode": args.processor_norm_default_mode,
        "processor_use_stepwise_action_norm": args.processor_use_stepwise_action_norm,
        "processor_delta_action_dim_mask": _json_arg(
            args.processor_delta_action_dim_mask,
            "--processor-delta-action-dim-mask",
        ),
        "normalization_stats_path": str(output_path),
        "text_embedding_cache_dir": None,
        "return_future_3d_images": False,
    }
    if args.video_backend is not None:
        cfg_kwargs["video_backend"] = args.video_backend

    cfg = MagicBotR0DatasetConfig(**cfg_kwargs)
    dataset = build_fastwam_dataset(cfg, stats_cache_path=str(output_path))

    print("---------- done ----------")
    print(f"action_mode: {args.action_mode}")
    print(f"norm_default_mode: {args.processor_norm_default_mode}")
    print(f"num_episodes: {dataset.num_episodes}")
    print(f"num_frames: {dataset.num_frames}")
    print(f"stats_path: {output_path}")


if __name__ == "__main__":
    main()
