#!/usr/bin/env python
"""Repack a CubeV2 checkpoint/pretrained_model directory without frozen external modules.

This tool removes `model.cosmos.*` and `model.da3_teacher.*` tensors from an existing
`model.safetensors` file and writes a smaller `pretrained_model` directory suitable for
Hub upload. It copies every other artifact from the source directory unchanged.
"""
# python util_scripts/repack_cubev2_checkpoint.py \
#   --src /path/to/checkpoints/300000 \
#   --dst /path/to/checkpoints/300000_slim/pretrained_model

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from safetensors import safe_open
from safetensors.torch import save_file

EXCLUDED_PREFIXES = (
    "model.cosmos.",
    "model.da3_teacher.",
)
PRETRAINED_MODEL_DIR = "pretrained_model"


def format_big_number(num: int, precision: int = 0) -> str:
    suffixes = ["", "K", "M", "B", "T", "Q"]
    divisor = 1000.0

    value = float(num)
    for suffix in suffixes:
        if abs(value) < divisor:
            return f"{value:.{precision}f}{suffix}"
        value /= divisor

    return f"{value:.{precision}f}Q"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repack a CubeV2 checkpoint by dropping frozen external-module weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Source checkpoint dir or pretrained_model dir.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="Output pretrained_model dir. Defaults to a sibling '<name>_slim' directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the destination directory if it already exists.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print dropped tensor keys.",
    )
    return parser.parse_args()


def resolve_pretrained_dir(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    if (candidate / SAFETENSORS_SINGLE_FILE).is_file():
        return candidate

    nested_candidate = candidate / PRETRAINED_MODEL_DIR
    if (nested_candidate / SAFETENSORS_SINGLE_FILE).is_file():
        return nested_candidate

    raise FileNotFoundError(
        f"Could not find {SAFETENSORS_SINGLE_FILE!r} under {candidate} or {nested_candidate}."
    )


def default_output_dir(src_pretrained_dir: Path) -> Path:
    return src_pretrained_dir.parent / f"{src_pretrained_dir.name}_slim"


def should_exclude(key: str) -> bool:
    return any(key.startswith(prefix) for prefix in EXCLUDED_PREFIXES)


def copy_auxiliary_files(src_pretrained_dir: Path, dst_pretrained_dir: Path) -> None:
    for item in src_pretrained_dir.iterdir():
        if item.name == SAFETENSORS_SINGLE_FILE:
            continue

        dst_item = dst_pretrained_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst_item, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst_item)


def filter_model_file(src_model_file: Path) -> tuple[dict, dict[str, str] | None, list[str], int, int]:
    kept_tensors = {}
    dropped_keys: list[str] = []
    total_params = 0
    dropped_params = 0

    with safe_open(str(src_model_file), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            total_params += tensor.numel()
            if should_exclude(key):
                dropped_keys.append(key)
                dropped_params += tensor.numel()
                continue
            kept_tensors[key] = tensor

    filtered_metadata = {key: value for key, value in metadata.items() if not should_exclude(key)} or None
    return kept_tensors, filtered_metadata, dropped_keys, total_params, dropped_params


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    src_pretrained_dir = resolve_pretrained_dir(args.src)
    dst_pretrained_dir = args.dst.expanduser().resolve() if args.dst else default_output_dir(src_pretrained_dir)

    if src_pretrained_dir == dst_pretrained_dir:
        raise ValueError("Source and destination must be different directories.")

    if dst_pretrained_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Destination already exists: {dst_pretrained_dir}. Pass --overwrite to replace it."
            )
        shutil.rmtree(dst_pretrained_dir)

    dst_pretrained_dir.mkdir(parents=True, exist_ok=True)

    src_model_file = src_pretrained_dir / SAFETENSORS_SINGLE_FILE
    dst_model_file = dst_pretrained_dir / SAFETENSORS_SINGLE_FILE

    logging.info(f"Source pretrained_model: {src_pretrained_dir}")
    logging.info(f"Destination pretrained_model: {dst_pretrained_dir}")
    logging.info(f"Excluded prefixes: {', '.join(EXCLUDED_PREFIXES)}")

    kept_tensors, filtered_metadata, dropped_keys, total_params, dropped_params = filter_model_file(src_model_file)
    save_file(kept_tensors, str(dst_model_file), metadata=filtered_metadata)
    copy_auxiliary_files(src_pretrained_dir, dst_pretrained_dir)

    kept_params = total_params - dropped_params
    src_size_mb = src_model_file.stat().st_size / (1024 * 1024)
    dst_size_mb = dst_model_file.stat().st_size / (1024 * 1024)

    logging.info("")
    logging.info("Repack summary:")
    logging.info(f"  Total params in source file : {total_params} ({format_big_number(total_params)})")
    logging.info(f"  Dropped params              : {dropped_params} ({format_big_number(dropped_params)})")
    logging.info(f"  Kept params                : {kept_params} ({format_big_number(kept_params)})")
    logging.info(f"  Dropped tensor keys        : {len(dropped_keys)}")
    logging.info(f"  Source model.safetensors   : {src_size_mb:.2f} MB")
    logging.info(f"  Output model.safetensors   : {dst_size_mb:.2f} MB")

    if args.verbose and dropped_keys:
        logging.info("")
        logging.info("Dropped keys:")
        for key in dropped_keys:
            logging.info(f"  {key}")


if __name__ == "__main__":
    main()
