#!/usr/bin/env python
"""Patch tiny action std values in a normalization stats JSON.

This is a temporary guard for delta-action datasets where some robot joints are
inactive in a specific dataset split. Tiny std values make z-score
normalization amplify numerical noise and can explode action loss.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path


def parse_dims(value: str | None) -> set[int] | None:
    if value is None or value.strip() == "":
        return None
    dims: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            dims.update(range(int(start), int(end) + 1))
        else:
            dims.add(int(part))
    return dims


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clamp tiny action std values in a stats.json file."
    )
    parser.add_argument("stats_path", type=Path)
    parser.add_argument("--key", default="action", help="Stats key to patch.")
    parser.add_argument(
        "--min-std",
        type=float,
        default=1e-3,
        help="Patch dims whose std is smaller than this threshold.",
    )
    parser.add_argument(
        "--replace-std",
        type=float,
        default=1.0,
        help="Replacement std for patched dims.",
    )
    parser.add_argument(
        "--dims",
        default=None,
        help="Optional dims to consider, e.g. '7-13' or '7,8,13'. Defaults to all dims.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write a .bak file before patching.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats_path = args.stats_path.expanduser()
    if not stats_path.is_file():
        raise FileNotFoundError(f"stats file not found: {stats_path}")

    with stats_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if args.key not in payload or "std" not in payload[args.key]:
        raise KeyError(f"{stats_path} does not contain {args.key!r}.std")

    std = payload[args.key]["std"]
    if not isinstance(std, list):
        raise TypeError(f"{args.key}.std must be a list, got {type(std).__name__}")

    selected_dims = parse_dims(args.dims)
    changed: list[tuple[int, float, float]] = []
    for idx, value in enumerate(std):
        if selected_dims is not None and idx not in selected_dims:
            continue
        value_f = float(value)
        if (not math.isfinite(value_f)) or abs(value_f) < args.min_std:
            std[idx] = float(args.replace_std)
            changed.append((idx, value_f, float(args.replace_std)))

    if not changed:
        print(f"No std values below {args.min_std} in {stats_path}")
        return

    if not args.no_backup:
        backup_path = stats_path.with_suffix(stats_path.suffix + ".bak")
        shutil.copy2(stats_path, backup_path)
        print(f"Backup written: {backup_path}")

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Patched {len(changed)} dims in {stats_path}:")
    for idx, old, new in changed:
        print(f"  dim{idx}: {old:g} -> {new:g}")


if __name__ == "__main__":
    main()
