#!/usr/bin/env python
"""Backfill missing MagicBot_R0 checkpoint stats.json files.

This utility copies a known-good ``pretrained_model/stats.json`` into checkpoint
directories where that file is missing or empty. It is intended for pretraining
runs where resumed checkpoints may not have re-saved dataset stats.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a valid pretrained_model/stats.json to checkpoint directories "
            "that are missing stats.json or have an empty stats payload."
        )
    )
    parser.add_argument(
        "run_or_checkpoints_dir",
        type=Path,
        help=(
            "Run directory containing checkpoints/, or the checkpoints directory itself. "
            "Example: outputs/MagicBot_R0/<run-name>"
        ),
    )
    parser.add_argument(
        "--source-stats",
        type=Path,
        default=None,
        help=(
            "Explicit stats.json to copy from. If omitted, the first valid "
            "checkpoint stats.json is used."
        ),
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help=(
            "train_config.json to use when no valid source stats.json exists. "
            "Defaults to the first numeric checkpoint's pretrained_model/train_config.json."
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help=(
            "Project root used to resolve relative paths in train_config.json. "
            "Defaults to current directory, or the parent of outputs/ inferred from the run path."
        ),
    )
    parser.add_argument(
        "--no-generate-from-config",
        action="store_true",
        help="Do not generate stats from train_config.json when no valid source stats.json exists.",
    )
    parser.add_argument(
        "--require-key",
        default="",
        help=(
            "Require this top-level JSON key for a stats file to be considered valid. "
            "Use --require-key '' to accept any non-empty JSON object."
        ),
    )
    parser.add_argument(
        "--replace-invalid",
        action="store_true",
        help="Also replace invalid/non-JSON stats.json files. By default they are skipped.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write files. Without this flag the script only prints a dry-run plan.",
    )
    return parser.parse_args()


def resolve_checkpoints_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "checkpoints").is_dir():
        return path / "checkpoints"
    if path.name == "checkpoints" and path.is_dir():
        return path
    raise FileNotFoundError(f"Expected a run dir with checkpoints/ or a checkpoints dir, got: {path}")


def checkpoint_dirs(checkpoints_dir: Path) -> list[Path]:
    dirs = []
    for child in checkpoints_dir.iterdir():
        if not child.is_dir() or child.is_symlink():
            continue
        if not child.name.isdigit():
            continue
        if (child / "pretrained_model").is_dir():
            dirs.append(child)
    return sorted(dirs, key=lambda p: int(p.name))


def infer_run_dir(path: Path, checkpoints_dir: Path) -> Path:
    if checkpoints_dir.name == "checkpoints":
        return checkpoints_dir.parent
    if (path / "checkpoints").is_dir():
        return path
    return checkpoints_dir.parent


def infer_project_root(run_dir: Path, explicit_root: Path | None) -> Path | None:
    if explicit_root is not None:
        return explicit_root.expanduser().resolve()
    for parent in [run_dir, *run_dir.parents]:
        if parent.name == "outputs":
            return parent.parent
        if (parent / "launch" / "magicbot_r0").is_dir() and (parent / "src" / "lerobot").is_dir():
            return parent
    return Path.cwd().resolve()


def resolve_path(path_value: str | Path, *, project_root: Path | None, base_dir: Path | None = None) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    candidates = []
    if project_root is not None:
        candidates.append(project_root / path)
    if base_dir is not None:
        candidates.append(base_dir / path)
    candidates.append(Path.cwd() / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def to_jsonable(value: Any) -> Any:
    try:
        import numpy as np
        import torch
    except Exception:
        np = None
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    tmp.replace(path)


def is_valid_stats(path: Path, required_key: str | None) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    payload = load_json(path)
    if not isinstance(payload, dict) or len(payload) == 0:
        return False
    if required_key and required_key not in payload:
        return False
    if required_key and isinstance(payload.get(required_key), dict) and len(payload[required_key]) == 0:
        return False
    return True


def stats_status(path: Path, required_key: str | None) -> str:
    if not path.exists():
        return "missing"
    if not path.is_file():
        return "not_file"
    if path.stat().st_size == 0:
        return "empty"
    try:
        payload = load_json(path)
    except Exception:
        return "invalid"
    if not isinstance(payload, dict) or len(payload) == 0:
        return "empty"
    if required_key and required_key not in payload:
        return f"missing_key:{required_key}"
    if required_key and isinstance(payload.get(required_key), dict) and len(payload[required_key]) == 0:
        return f"empty_key:{required_key}"
    return "ok"


def find_source_stats(
    checkpoints: list[Path],
    explicit_source: Path | None,
    required_key: str | None,
) -> Path:
    if explicit_source is not None:
        source = explicit_source.expanduser().resolve()
        if not is_valid_stats(source, required_key):
            raise ValueError(f"Explicit --source-stats is not a valid stats file: {source}")
        return source

    candidates = [ckpt / "pretrained_model" / "stats.json" for ckpt in checkpoints]
    for candidate in candidates:
        try:
            if is_valid_stats(candidate, required_key):
                return candidate
        except Exception:
            continue
    raise FileNotFoundError(
        "No valid source stats.json found. Pass --source-stats /path/to/stats.json explicitly."
    )


def find_config_path(
    checkpoints: list[Path],
    explicit_config: Path | None,
    *,
    project_root: Path | None,
    run_dir: Path,
) -> Path:
    if explicit_config is not None:
        candidate = resolve_path(explicit_config, project_root=project_root, base_dir=run_dir)
        if not candidate.is_file():
            raise FileNotFoundError(f"--config-path does not exist: {candidate}")
        return candidate

    for ckpt in checkpoints:
        candidate = ckpt / "pretrained_model" / "train_config.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("No train_config.json found in numeric checkpoint pretrained_model directories.")


def resolve_dataset_dirs(dataset_cfg: dict[str, Any], *, project_root: Path | None, config_dir: Path) -> list[Path]:
    dataset_dirs = dataset_cfg.get("dataset_dirs") or []
    if dataset_dirs:
        return [
            resolve_path(dataset_dir, project_root=project_root, base_dir=config_dir)
            for dataset_dir in dataset_dirs
            if str(dataset_dir).strip()
        ]

    repo_id_file = dataset_cfg.get("repo_id_file")
    if not repo_id_file:
        raise ValueError("train_config dataset has neither dataset_dirs nor repo_id_file.")
    repo_id_file_path = resolve_path(repo_id_file, project_root=project_root, base_dir=config_dir)
    if not repo_id_file_path.is_file():
        raise FileNotFoundError(f"dataset.repo_id_file does not exist: {repo_id_file_path}")

    out = []
    with repo_id_file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value:
                out.append(resolve_path(value, project_root=project_root, base_dir=repo_id_file_path.parent))
    if not out:
        raise ValueError(f"dataset.repo_id_file is empty: {repo_id_file_path}")
    return out


def load_external_stats_payload(path: Path) -> dict[str, Any]:
    from lerobot.policies.MagicBot_R0.core.data.lerobot.utils.normalizer import (
        load_dataset_stats_from_json,
    )

    return load_dataset_stats_from_json(str(path))


def generate_stats_from_config(config_path: Path, *, project_root: Path | None) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.transforms.constants import infer_embodiment_variant

    cfg = load_json(config_path)
    dataset_cfg = cfg.get("dataset")
    if not isinstance(dataset_cfg, dict):
        raise ValueError(f"train_config.json has no dataset object: {config_path}")

    external_stats_root = dataset_cfg.get("external_stats_root")
    if not external_stats_root:
        raise ValueError("dataset.external_stats_root is required to generate pretrain checkpoint stats.")
    external_stats_root = resolve_path(external_stats_root, project_root=project_root, base_dir=config_path.parent)
    action_mode = str(dataset_cfg.get("action_mode", "abs"))
    dataset_dirs = resolve_dataset_dirs(dataset_cfg, project_root=project_root, config_dir=config_path.parent)

    stats_by_key: dict[str, Any] = {}
    stats_source_by_key: dict[str, Path] = {}
    cache: dict[tuple[str, str], tuple[Path, dict[str, Any]]] = {}

    print(f"Generating checkpoint stats from {len(dataset_dirs)} dataset dir(s)...")
    for index, dataset_dir in enumerate(dataset_dirs, start=1):
        meta = LeRobotDatasetMetadata(repo_id=str(Path(dataset_dir)), root=Path(dataset_dir))
        robot_type = str(meta.robot_type)
        resolved_robot_type = str(infer_embodiment_variant(robot_type, meta.features))
        cache_key = (robot_type, resolved_robot_type)

        if cache_key in cache:
            stats_path, payload = cache[cache_key]
        else:
            candidates = [
                external_stats_root / resolved_robot_type / action_mode / "stats.json",
                external_stats_root / robot_type / action_mode / "stats.json",
            ]
            stats_path = next((candidate for candidate in candidates if candidate.is_file()), None)
            if stats_path is None:
                tried = "\n".join(f"  - {candidate}" for candidate in candidates)
                raise FileNotFoundError(
                    f"Missing external stats for dataset {dataset_dir} "
                    f"(robot_type={robot_type}, resolved={resolved_robot_type}). Tried:\n{tried}"
                )
            payload = load_external_stats_payload(stats_path)
            cache[cache_key] = (stats_path, payload)

        for key in (resolved_robot_type, robot_type):
            if not key:
                continue
            if key in stats_by_key:
                if stats_source_by_key[key] != stats_path:
                    print(
                        f"warning: stats key {key} already came from {stats_source_by_key[key]}; "
                        f"keeping it and ignoring duplicate source {stats_path}"
                    )
                continue
            stats_by_key[key] = payload
            stats_source_by_key[key] = stats_path

        if index % 100 == 0:
            print(f"  processed {index}/{len(dataset_dirs)} dataset dirs, stats_keys={len(stats_by_key)}")

    if not stats_by_key:
        raise ValueError("Generated stats payload is empty.")
    print(
        "Generated stats keys: "
        + ", ".join(f"{key}<-{stats_source_by_key[key]}" for key in sorted(stats_by_key))
    )
    return to_jsonable(stats_by_key)


def copy_stats(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    shutil.copy2(source, tmp)
    tmp.replace(target)


def main() -> None:
    args = parse_args()
    required_key = args.require_key or None
    checkpoints_dir = resolve_checkpoints_dir(args.run_or_checkpoints_dir)
    run_dir = infer_run_dir(args.run_or_checkpoints_dir.expanduser().resolve(), checkpoints_dir)
    project_root = infer_project_root(run_dir, args.project_root)
    checkpoints = checkpoint_dirs(checkpoints_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No numeric checkpoint directories found in: {checkpoints_dir}")

    source_stats: Path | None = None
    generated_stats: dict[str, Any] | None = None
    try:
        source_stats = find_source_stats(checkpoints, args.source_stats, required_key)
    except FileNotFoundError:
        if args.no_generate_from_config:
            raise
        config_path = find_config_path(
            checkpoints,
            args.config_path,
            project_root=project_root,
            run_dir=run_dir,
        )
        generated_stats = generate_stats_from_config(config_path, project_root=project_root)
    except ValueError:
        if args.source_stats is not None or args.no_generate_from_config:
            raise
        config_path = find_config_path(
            checkpoints,
            args.config_path,
            project_root=project_root,
            run_dir=run_dir,
        )
        generated_stats = generate_stats_from_config(config_path, project_root=project_root)

    targets: list[tuple[Path, str]] = []
    ok_count = 0
    skipped_count = 0

    for ckpt in checkpoints:
        stats_path = ckpt / "pretrained_model" / "stats.json"
        if source_stats is not None and stats_path.resolve() == source_stats.resolve():
            ok_count += 1
            continue
        status = stats_status(stats_path, required_key)
        if status == "ok":
            ok_count += 1
            continue
        if status == "invalid" and not args.replace_invalid:
            skipped_count += 1
            print(f"skip invalid stats: {stats_path}")
            continue
        if status == "not_file":
            skipped_count += 1
            print(f"skip non-file stats path: {stats_path}")
            continue
        targets.append((stats_path, status))

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] checkpoints_dir={checkpoints_dir}")
    print(f"[{mode}] project_root={project_root}")
    if source_stats is not None:
        print(f"[{mode}] source_stats={source_stats}")
    else:
        print(f"[{mode}] source_stats=<generated from train_config.json>")
    print(f"[{mode}] valid_existing={ok_count}, to_backfill={len(targets)}, skipped={skipped_count}")
    for target, status in targets:
        source_label = str(source_stats) if source_stats is not None else "<generated stats>"
        print(f"{'copy' if args.apply else 'would copy'} ({status}): {source_label} -> {target}")
        if args.apply:
            if source_stats is not None:
                copy_stats(source_stats, target)
            else:
                write_json(target, generated_stats)

    if not args.apply and targets:
        print("\nDry run only. Re-run with --apply to write these files.")


if __name__ == "__main__":
    main()
