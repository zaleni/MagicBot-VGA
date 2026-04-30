#!/usr/bin/env python
"""Inspect vector columns in LeRobot parquet data.

This is meant for quick checks such as whether observation.state/action are
all zeros. It reads only the requested parquet columns and never decodes images
or videos.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


@dataclass
class ColumnStats:
    rows: int = 0
    elems: int = 0
    nonzero_elems: int = 0
    max_abs: float = 0.0
    min_value: float | None = None
    max_value: float | None = None
    first_nonzero_file: str | None = None
    first_nonzero_row: int | None = None
    first_nonzero_value: list[float] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check LeRobot vector columns for all-zero values.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python util_scripts/inspect_lerobot_vectors.py /path/to/lerobot_repo
  python util_scripts/inspect_lerobot_vectors.py /path/to/lerobot_repo --keys observation.state action --atol 1e-8
""",
    )
    parser.add_argument("dataset_dir", help="LeRobot dataset root directory.")
    parser.add_argument(
        "--keys",
        nargs="+",
        default=["observation.state", "action"],
        help="Vector columns to inspect.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="Treat values with abs(value) <= atol as zero.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on parquet files to read.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on rows per column across all files.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=16,
        help="Number of flattened values to print for the first nonzero row.",
    )
    return parser.parse_args()


def to_numpy(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == object:
        arr = np.asarray(value, dtype=np.float64)
    return arr.astype(np.float64, copy=False).reshape(-1)


def update_stats(
    stats: ColumnStats,
    values: pd.Series,
    parquet_path: Path,
    row_offset: int,
    atol: float,
    preview: int,
    max_rows: int | None,
) -> int:
    rows_used = 0
    for local_row, value in enumerate(values.tolist()):
        if max_rows is not None and stats.rows >= max_rows:
            break

        arr = to_numpy(value)
        if arr.size == 0:
            continue

        abs_arr = np.abs(arr)
        nonzero_mask = abs_arr > atol
        nonzero_count = int(nonzero_mask.sum())

        stats.rows += 1
        stats.elems += int(arr.size)
        stats.nonzero_elems += nonzero_count
        stats.max_abs = max(stats.max_abs, float(abs_arr.max()))
        cur_min = float(arr.min())
        cur_max = float(arr.max())
        stats.min_value = cur_min if stats.min_value is None else min(stats.min_value, cur_min)
        stats.max_value = cur_max if stats.max_value is None else max(stats.max_value, cur_max)

        if nonzero_count > 0 and stats.first_nonzero_file is None:
            stats.first_nonzero_file = str(parquet_path)
            stats.first_nonzero_row = row_offset + local_row
            stats.first_nonzero_value = arr[:preview].tolist()

        rows_used += 1
    return rows_used


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_dir)
    data_dir = root / "data"
    parquet_files = sorted(data_dir.glob("*/*.parquet"))
    if args.max_files is not None:
        parquet_files = parquet_files[: args.max_files]
    if not parquet_files:
        raise SystemExit(f"No parquet files found under {data_dir}")

    stats_by_key = {key: ColumnStats() for key in args.keys}
    missing_by_key: dict[str, int] = {key: 0 for key in args.keys}
    global_row_offset = 0

    for parquet_path in parquet_files:
        metadata = pq.read_metadata(parquet_path)
        available = set(pq.read_schema(parquet_path).names)
        columns = [key for key in args.keys if key in available]
        for key in args.keys:
            if key not in available:
                missing_by_key[key] += 1
        if not columns:
            global_row_offset += int(metadata.num_rows)
            continue

        df = pd.read_parquet(parquet_path, columns=columns, engine="pyarrow")
        for key in columns:
            update_stats(
                stats=stats_by_key[key],
                values=df[key],
                parquet_path=parquet_path,
                row_offset=global_row_offset,
                atol=float(args.atol),
                preview=int(args.preview),
                max_rows=args.max_rows,
            )
        global_row_offset += int(metadata.num_rows)

    print(f"dataset: {root}")
    print(f"files_read: {len(parquet_files)}")
    print(f"atol: {args.atol}")
    for key, stats in stats_by_key.items():
        all_zero = stats.nonzero_elems == 0
        print(f"\n[{key}]")
        print(f"rows: {stats.rows}")
        print(f"elems: {stats.elems}")
        print(f"nonzero_elems: {stats.nonzero_elems}")
        print(f"all_zero: {all_zero}")
        print(f"max_abs: {stats.max_abs}")
        print(f"min: {stats.min_value}")
        print(f"max: {stats.max_value}")
        if missing_by_key[key] > 0:
            print(f"missing_in_files: {missing_by_key[key]}")
        if stats.first_nonzero_file is not None:
            print(f"first_nonzero_file: {stats.first_nonzero_file}")
            print(f"first_nonzero_row: {stats.first_nonzero_row}")
            print(f"first_nonzero_value_preview: {stats.first_nonzero_value}")


if __name__ == "__main__":
    main()
