#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import tqdm

from lerobot.datasets.utils import write_json
from lerobot.policies.cubev2.robochallenge_raw_dataset import (
    ROBOCHALLENGE_W1_DELTA_MASK,
    discover_robochallenge_w1_episodes,
    get_robochallenge_raw_spec,
    load_robochallenge_w1_state_array,
    resolve_robochallenge_raw_task_names,
    resolve_robochallenge_raw_task_weights,
)
from lerobot.utils.constants import ACTION, OBS_STATE


class RunningStats:
    def __init__(self):
        self._count = 0
        self._mass = 0.0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None

    def update(self, batch, *, sample_weight: float = 1.0) -> None:
        sample_weight = float(sample_weight)
        if sample_weight <= 0.0:
            return
        batch = torch.as_tensor(batch, dtype=torch.float32)
        if batch.ndim == 1:
            batch = batch[:, None]
        if batch.ndim > 1:
            batch = batch.reshape(-1, batch.shape[-1])
        if batch.numel() == 0:
            return

        count = int(batch.shape[0])
        mass = count * sample_weight
        mean = batch.mean(dim=0)
        mean_sq = (batch * batch).mean(dim=0)
        min_v = batch.min(dim=0).values
        max_v = batch.max(dim=0).values

        if self._mass == 0.0:
            self._count = count
            self._mass = mass
            self._mean = mean
            self._mean_of_squares = mean_sq
            self._min = min_v
            self._max = max_v
            return

        total = self._mass + mass
        old_w = self._mass / total
        new_w = mass / total
        self._mean = old_w * self._mean + new_w * mean
        self._mean_of_squares = old_w * self._mean_of_squares + new_w * mean_sq
        self._min = torch.minimum(self._min, min_v)
        self._max = torch.maximum(self._max, max_v)
        self._count += count
        self._mass = total

    def get_statistics(self) -> dict:
        if self._mass == 0.0:
            raise ValueError("No values were added to RunningStats.")
        var = self._mean_of_squares - self._mean * self._mean
        std = torch.sqrt(torch.clamp(var, min=0.0))
        return {
            "min": self._min.tolist(),
            "max": self._max.tolist(),
            "mean": self._mean.tolist(),
            "std": std.tolist(),
            "count": [int(self._count)],
        }


def build_parser(
    *,
    description: str = "Compute CubeV2 stats for RoboChallenge raw dual-arm data.",
    default_embodiment: str = "DOS-W1",
    default_task_preset: str = "table30v2_w1",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--raw-root", required=True, help="RoboChallenge raw root or one task directory.")
    parser.add_argument("--output-path", required=True, help="Output stats.json path.")
    parser.add_argument("--action-mode", choices=["abs", "delta"], default="delta")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--frame-interval", type=int, default=1)
    parser.add_argument("--embodiment", default=default_embodiment)
    parser.add_argument("--task-preset", default=default_task_preset)
    parser.add_argument("--weighted-task-stats", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--regular-task-weight", type=float, default=1.0)
    parser.add_argument("--extra-task-weight", type=float, default=0.8)
    parser.add_argument("--task-regex", default=None)
    parser.add_argument("--state-cache-dir", default=None)
    parser.add_argument("--block-size", type=int, default=4096)
    return parser


def parse_args():
    return build_parser().parse_args()


def _sample_arrays(states: np.ndarray, sample_count: int, frame_interval: int) -> tuple[np.ndarray, np.ndarray]:
    current_indices = np.arange(sample_count, dtype=np.int64) * int(frame_interval)
    target_indices = (np.arange(sample_count, dtype=np.int64) + 1) * int(frame_interval)
    return states[current_indices], states[target_indices]


def _update_action_stats(
    stats: RunningStats,
    state_rows: np.ndarray,
    action_rows: np.ndarray,
    *,
    action_mode: str,
    chunk_size: int,
    block_size: int,
    sample_weight: float = 1.0,
) -> None:
    num_starts = int(action_rows.shape[0])
    if num_starts <= 0:
        return

    mask = ROBOCHALLENGE_W1_DELTA_MASK.astype(np.float32)
    offsets = np.arange(chunk_size, dtype=np.int64)
    for start in range(0, num_starts, block_size):
        end = min(start + block_size, num_starts)
        starts = np.arange(start, end, dtype=np.int64)
        chunk_indices = np.minimum(starts[:, None] + offsets[None, :], num_starts - 1)
        action_chunk = action_rows[chunk_indices]
        if action_mode == "delta":
            state = state_rows[starts]
            action_chunk = action_chunk - (state * mask)[:, None, :]
        stats.update(action_chunk, sample_weight=sample_weight)


def compute_stats(args) -> None:
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if args.frame_interval <= 0:
        raise ValueError("--frame-interval must be positive")
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")

    episodes = discover_robochallenge_w1_episodes(
        args.raw_root,
        embodiment=args.embodiment,
        frame_interval=args.frame_interval,
        task_regex=args.task_regex,
        task_names=resolve_robochallenge_raw_task_names(args.task_preset),
    )
    task_sampling_weights = (
        resolve_robochallenge_raw_task_weights(
            args.task_preset,
            regular_task_weight=args.regular_task_weight,
            extra_task_weight=args.extra_task_weight,
        )
        if args.weighted_task_stats
        else None
    )
    task_sample_counts = {}
    for record in episodes:
        task_sample_counts[record.task_name] = task_sample_counts.get(record.task_name, 0) + record.sample_count

    state_stats = RunningStats()
    action_stats = RunningStats()
    total_samples = 0

    spec = get_robochallenge_raw_spec(args.embodiment, args.task_preset)

    for record in tqdm.tqdm(episodes, desc=f"Computing {spec.robot_type} raw stats"):
        states = load_robochallenge_w1_state_array(record, args.state_cache_dir)
        sample_count = record.sample_count
        state_rows, action_rows = _sample_arrays(states, sample_count, args.frame_interval)

        if task_sampling_weights is None:
            state_sample_weight = 1.0
            action_sample_weight = 1.0
        else:
            task_weight = float(task_sampling_weights.get(record.task_name, 0.0))
            task_sample_count = int(task_sample_counts[record.task_name])
            state_sample_weight = task_weight / task_sample_count
            action_sample_weight = state_sample_weight / args.chunk_size

        state_stats.update(state_rows, sample_weight=state_sample_weight)
        total_samples += int(sample_count)

        _update_action_stats(
            action_stats,
            state_rows,
            action_rows,
            action_mode=args.action_mode,
            chunk_size=args.chunk_size,
            block_size=args.block_size,
            sample_weight=action_sample_weight,
        )

    output = {
        OBS_STATE: state_stats.get_statistics(),
        ACTION: action_stats.get_statistics(),
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, output_path)

    print("---------- done ----------")
    print(f"episodes: {len(episodes)}")
    print(f"samples: {total_samples}")
    print(f"action_mode: {args.action_mode}")
    print(f"chunk_size: {args.chunk_size}")
    print(f"embodiment: {spec.robot_type}")
    print(f"task_preset: {args.task_preset}")
    print(f"weighted_task_stats: {args.weighted_task_stats}")
    print(f"output: {output_path}")


if __name__ == "__main__":
    compute_stats(parse_args())
