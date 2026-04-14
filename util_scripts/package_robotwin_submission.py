#!/usr/bin/env python

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
from pathlib import Path


DEFAULT_TASKS = [
    "hanging_mug",
    "move_stapler_pad",
    "place_fan",
    "handover_mic",
    "open_microwave",
    "place_can_basket",
    "place_dual_shoes",
    "stack_blocks_three",
    "move_can_pot",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
]

COMPETITION_NAME = "CVPR 2026 RoboTwin Track Leaderboard"
COMPETITION_URL = "https://huggingface.co/spaces/open-gigaai/CVPR-2026-RoboTwin-Track-LeaderBoard"

TASK_ALIASES = {
    "hugging_mug": "hanging_mug",
    "hanging_mug": "hanging_mug",
    "block_ranking_size": "blocks_ranking_size",
    "blocks_ranking_size": "blocks_ranking_size",
}

VIDEO_RE = re.compile(r"^(success|failure)_(\d+)\.mp4$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Package selected RoboTwin randomized-eval videos into a submission_package directory."
        )
    )
    parser.add_argument(
        "--run",
        required=True,
        help=(
            "Path to the randomized evaluation run directory, or directly to its summary.txt / summary.json."
        ),
    )
    parser.add_argument(
        "--dst",
        default="submission_package",
        help="Destination submission directory. Default: ./submission_package",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=DEFAULT_TASKS,
        help=(
            "Task names to export. Defaults to the 11-task submission set. "
            "Aliases like hugging_mug and block_ranking_size are accepted."
        ),
    )
    parser.add_argument(
        "--policy-dir",
        default=None,
        help=(
            "Optional local directory to copy into submission_package/ as Your_Policy. "
            "Useful if you also want to bundle your deploy policy folder."
        ),
    )
    parser.add_argument(
        "--policy-name",
        default="Your_Policy",
        help="Destination folder name for --policy-dir. Default: Your_Policy",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination directory if it already exists.",
    )
    return parser.parse_args()


def normalize_task_name(task_name: str) -> str:
    normalized = TASK_ALIASES.get(task_name, task_name)
    return normalized


def resolve_run_output_path(run_arg: str) -> Path:
    run_path = Path(run_arg).expanduser().resolve()
    if run_path.is_file() and run_path.name in {"summary.txt", "summary.json"}:
        return run_path.parent
    return run_path


def load_task_names(repo_root: Path) -> list[str]:
    inference_path = repo_root / "evaluation" / "RoboTwin" / "inference.py"
    source = inference_path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(inference_path))

    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "TASK_NAMES":
                value = ast.literal_eval(node.value)
                if not isinstance(value, list):
                    raise TypeError(f"TASK_NAMES in {inference_path} is not a list")
                return [str(item) for item in value]

    raise RuntimeError(f"Failed to find TASK_NAMES in {inference_path}")


def collect_episode_videos(task_dir: Path) -> list[tuple[int, str, Path]]:
    indexed_videos: list[tuple[int, str, Path]] = []
    for path in task_dir.glob("*.mp4"):
        match = VIDEO_RE.match(path.name)
        if match is None:
            continue
        indexed_videos.append((int(match.group(2)), match.group(1), path))
    indexed_videos.sort(key=lambda item: item[0])
    return indexed_videos


def prepare_destination(dst_root: Path, overwrite: bool) -> None:
    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination already exists: {dst_root}. Use --overwrite to replace it."
            )
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)


def copy_policy_dir(policy_dir: Path, dst_root: Path, policy_name: str) -> None:
    if not policy_dir.is_dir():
        raise FileNotFoundError(f"--policy-dir does not exist or is not a directory: {policy_dir}")
    shutil.copytree(policy_dir, dst_root / policy_name)


def load_task_summary(
    task_name: str,
    task_idx: int,
    task_dir: Path,
    episode_videos: list[tuple[int, str, Path]],
) -> dict:
    summary_path = task_dir / "summary.json"
    fallback_success_count = sum(1 for _, status, _ in episode_videos if status == "success")
    fallback_test_num = len(episode_videos)

    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        success_count = int(payload.get("success_count", fallback_success_count))
        test_num = int(payload.get("test_num", fallback_test_num))
    else:
        success_count = fallback_success_count
        test_num = fallback_test_num

    success_rate = round((success_count / test_num) * 100, 2) if test_num else 0.0
    return {
        "task_idx": task_idx,
        "task_name": task_name,
        "success_count": success_count,
        "test_num": test_num,
        "success_rate": success_rate,
    }


def write_selected_task_summary(dst_root: Path, run_output_path: Path, task_summaries: list[dict]) -> None:
    completed_tasks = len(task_summaries)
    total_success = sum(item["success_count"] for item in task_summaries)
    total_tests = sum(item["test_num"] for item in task_summaries)
    avg_task_success_rate = round(
        sum(item["success_rate"] for item in task_summaries) / completed_tasks,
        2,
    ) if completed_tasks else 0.0
    overall_episode_success_rate = round((total_success / total_tests) * 100, 2) if total_tests else 0.0

    payload = {
        "competition_name": COMPETITION_NAME,
        "competition_url": COMPETITION_URL,
        "run_output_path": str(run_output_path),
        "completed_tasks": completed_tasks,
        "avg_task_success_rate": avg_task_success_rate,
        "overall_episode_success_rate": overall_episode_success_rate,
        "total_success": total_success,
        "total_tests": total_tests,
        "tasks": task_summaries,
    }

    (dst_root / "selected_task_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"competition_name: {COMPETITION_NAME}",
        f"competition_url: {COMPETITION_URL}",
        f"run_output_path: {run_output_path}",
        f"completed_tasks: {completed_tasks}",
        f"avg_task_success_rate: {avg_task_success_rate:.2f}%",
        f"overall_episode_success_rate: {overall_episode_success_rate:.2f}%",
        f"total_success: {total_success}",
        f"total_tests: {total_tests}",
        "",
        "per_task:",
    ]
    for item in task_summaries:
        lines.append(
            f"{item['task_idx']:02d} {item['task_name']}: "
            f"{item['success_rate']:.2f}% ({item['success_count']}/{item['test_num']})"
        )

    (dst_root / "selected_task_summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def package_task(task_name: str, task_idx: int, run_output_path: Path, dst_root: Path) -> dict:
    src_task_dir = run_output_path / "tasks" / f"task_{task_idx:02d}"
    if not src_task_dir.is_dir():
        raise FileNotFoundError(f"Task output directory not found: {src_task_dir}")

    episode_videos = collect_episode_videos(src_task_dir)
    if not episode_videos:
        raise FileNotFoundError(f"No replay videos found in {src_task_dir}")

    dst_task_dir = dst_root / task_name
    dst_task_dir.mkdir(parents=True, exist_ok=True)

    for episode_idx, (_, _, src_video) in enumerate(episode_videos):
        shutil.copy2(src_video, dst_task_dir / f"episode{episode_idx}.mp4")

    summary = load_task_summary(task_name, task_idx, src_task_dir, episode_videos)
    summary["episode_count"] = len(episode_videos)
    return summary


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    run_output_path = resolve_run_output_path(args.run)
    if not run_output_path.is_dir():
        raise FileNotFoundError(f"Run output directory not found: {run_output_path}")

    task_names = load_task_names(repo_root)
    task_to_idx = {name: idx for idx, name in enumerate(task_names)}

    requested_tasks = list(dict.fromkeys(normalize_task_name(task) for task in args.tasks))
    unknown_tasks = [task for task in requested_tasks if task not in task_to_idx]
    if unknown_tasks:
        raise ValueError(
            f"Unknown task names: {unknown_tasks}. Valid names come from evaluation/RoboTwin/inference.py"
        )
    requested_tasks.sort(key=lambda task: task_to_idx[task])

    dst_root = Path(args.dst).expanduser().resolve()
    prepare_destination(dst_root, overwrite=args.overwrite)

    task_summaries: list[dict] = []
    packaged_rows: list[str] = []
    for task_name in requested_tasks:
        summary = package_task(task_name, task_to_idx[task_name], run_output_path, dst_root)
        task_summaries.append(summary)
        packaged_rows.append(
            f"{summary['task_idx']:02d}\t{task_name}\t{summary['episode_count']}\t"
            f"{summary['success_rate']:.2f}%\t{summary['success_count']}/{summary['test_num']}"
        )

    if args.policy_dir is not None:
        copy_policy_dir(Path(args.policy_dir).expanduser().resolve(), dst_root, args.policy_name)

    manifest_lines = [
        f"run_output_path: {run_output_path}",
        f"task_count: {len(requested_tasks)}",
        f"competition_name: {COMPETITION_NAME}",
        f"competition_url: {COMPETITION_URL}",
        "",
        "task_idx\ttask_name\tepisode_count\tsuccess_rate\tsuccess_count/test_num",
        *packaged_rows,
    ]
    (dst_root / "package_manifest.txt").write_text(
        "\n".join(manifest_lines) + "\n",
        encoding="utf-8",
    )
    write_selected_task_summary(dst_root, run_output_path, task_summaries)

    print(f"Packaged {len(requested_tasks)} tasks into: {dst_root}")
    for row in packaged_rows:
        print(row.replace("\t", " | "))
    if task_summaries:
        avg_task_success_rate = round(
            sum(item["success_rate"] for item in task_summaries) / len(task_summaries),
            2,
        )
        total_success = sum(item["success_count"] for item in task_summaries)
        total_tests = sum(item["test_num"] for item in task_summaries)
        overall_episode_success_rate = round((total_success / total_tests) * 100, 2) if total_tests else 0.0
        print(f"avg_task_success_rate | {avg_task_success_rate:.2f}%")
        print(f"overall_episode_success_rate | {overall_episode_success_rate:.2f}%")
    if args.policy_dir is not None:
        print(f"Included policy folder: {dst_root / args.policy_name}")


if __name__ == "__main__":
    main()
