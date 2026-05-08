#!/usr/bin/env python

from __future__ import annotations

from compute_norm_stats_robochallenge_raw_w1 import build_parser, compute_stats


def parse_args():
    return build_parser(
        description="Compute CubeV2 stats for RoboChallenge raw ALOHA data.",
        default_embodiment="ALOHA",
        default_task_preset="table30v2_aloha",
    ).parse_args()


if __name__ == "__main__":
    compute_stats(parse_args())
