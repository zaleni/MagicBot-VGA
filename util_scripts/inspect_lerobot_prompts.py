#!/usr/bin/env python
"""Print prompts/tasks stored in LeRobot dataset metadata.

LeRobot v3.0 stores natural-language prompts in ``meta/tasks.parquet``.
This script reads that metadata directly, so it does not instantiate a
LeRobotDataset or decode images/videos.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


TASK_TEXT_COLUMNS = (
    "task",
    "prompt",
    "instruction",
    "language_instruction",
    "natural_language_instruction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect prompt/task strings in LeRobot v3.0 dataset metadata.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python util_scripts/inspect_lerobot_prompts.py /path/to/lerobot_dataset
  python util_scripts/inspect_lerobot_prompts.py --repo-id-file outputs/MagicBot_R0/_stats_repo_id_files/chunk32/aloha.txt
  python util_scripts/inspect_lerobot_prompts.py /path/to/datasets_root --recursive --show-episodes
""",
    )
    parser.add_argument("dataset_dirs", nargs="*", help="LeRobot dataset root directories.")
    parser.add_argument(
        "--repo-id-file",
        action="append",
        default=[],
        help="Text file containing one dataset root per line. Can be passed multiple times.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Treat positional paths as parent folders and find */meta/tasks.parquet under them.",
    )
    parser.add_argument("--limit", type=int, default=50, help="Max tasks to print per dataset.")
    parser.add_argument(
        "--show-episodes",
        action="store_true",
        help="Also print a few episode-level task lists from meta/episodes.",
    )
    parser.add_argument(
        "--episode-limit",
        type=int,
        default=10,
        help="Max episodes to print per dataset when --show-episodes is set.",
    )
    return parser.parse_args()


def read_repo_id_file(path: str | Path) -> list[Path]:
    roots: list[Path] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            roots.append(Path(line))
    return roots


def is_dataset_root(path: Path) -> bool:
    return (path / "meta" / "tasks.parquet").exists() or (path / "meta" / "tasks.jsonl").exists()


def expand_dataset_roots(paths: Iterable[Path], recursive: bool) -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()

    def add(root: Path) -> None:
        key = root.resolve() if root.exists() else root
        if key not in seen:
            roots.append(root)
            seen.add(key)

    for path in paths:
        if recursive:
            if is_dataset_root(path):
                add(path)
            for task_file in sorted(path.rglob("meta/tasks.parquet")):
                add(task_file.parent.parent)
            for task_file in sorted(path.rglob("meta/tasks.jsonl")):
                add(task_file.parent.parent)
        else:
            add(path)
    return roots


def safe_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def value_is_present(value: object) -> bool:
    try:
        missing = pd.isna(value)
    except TypeError:
        return True
    if isinstance(missing, bool):
        return not missing
    return True


def task_text_from_row(index: object, row: pd.Series) -> str:
    for column in TASK_TEXT_COLUMNS:
        if column in row and value_is_present(row[column]):
            return str(row[column])
    if isinstance(index, str) and index:
        return index
    return str(index)


def load_v30_tasks(root: Path) -> list[tuple[int, str]]:
    df = pd.read_parquet(root / "meta" / "tasks.parquet")
    tasks: list[tuple[int, str]] = []
    for pos, (index, row) in enumerate(df.iterrows()):
        task_index = safe_int(row.get("task_index", pos), pos)
        task_text = task_text_from_row(index, row)
        tasks.append((task_index, task_text))
    return sorted(tasks, key=lambda item: item[0])


def load_legacy_tasks(root: Path) -> list[tuple[int, str]]:
    tasks: list[tuple[int, str]] = []
    with (root / "meta" / "tasks.jsonl").open("r", encoding="utf-8") as f:
        for pos, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            task_index = safe_int(item.get("task_index", pos), pos)
            task_text = str(item.get("task", item.get("prompt", "")))
            tasks.append((task_index, task_text))
    return sorted(tasks, key=lambda item: item[0])


def load_tasks(root: Path) -> list[tuple[int, str]]:
    if (root / "meta" / "tasks.parquet").exists():
        return load_v30_tasks(root)
    if (root / "meta" / "tasks.jsonl").exists():
        return load_legacy_tasks(root)
    raise FileNotFoundError(f"No meta/tasks.parquet or meta/tasks.jsonl found under {root}")


def format_task_value(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return "; ".join(str(item) for item in value)
    return str(value)


def iter_episode_rows(root: Path, limit: int) -> Iterable[dict[str, object]]:
    episodes_dir = root / "meta" / "episodes"
    count = 0
    for parquet_path in sorted(episodes_dir.glob("*/*.parquet")):
        df = pd.read_parquet(parquet_path)
        for _, row in df.iterrows():
            yield row.to_dict()
            count += 1
            if count >= limit:
                return


def print_dataset(root: Path, limit: int, show_episodes: bool, episode_limit: int) -> None:
    print(f"\n=== {root} ===")
    try:
        tasks = load_tasks(root)
    except Exception as exc:  # noqa: BLE001 - this is a CLI inspector; keep going across many roots.
        print(f"ERROR: {exc}")
        return

    print(f"tasks: {len(tasks)}")
    for task_index, task_text in tasks[:limit]:
        print(f"[{task_index}] {task_text}")
    if len(tasks) > limit:
        print(f"... ({len(tasks) - limit} more)")

    if not show_episodes:
        return

    if not (root / "meta" / "episodes").exists():
        print("episodes: meta/episodes not found")
        return

    print(f"episodes (first {episode_limit}):")
    for row in iter_episode_rows(root, episode_limit):
        episode_index = row.get("episode_index", "?")
        if "tasks" in row:
            task_value = format_task_value(row["tasks"])
        elif "task_index" in row:
            task_value = f"task_index={row['task_index']}"
        else:
            task_value = "<no tasks/task_index column>"
        print(f"  episode {episode_index}: {task_value}")


def main() -> None:
    args = parse_args()
    roots = [Path(path) for path in args.dataset_dirs]
    for repo_id_file in args.repo_id_file:
        roots.extend(read_repo_id_file(repo_id_file))

    roots = expand_dataset_roots(roots, recursive=args.recursive)
    if not roots:
        raise SystemExit("No dataset roots provided.")

    for root in roots:
        print_dataset(root, args.limit, args.show_episodes, args.episode_limit)


if __name__ == "__main__":
    main()
