from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


DEFAULT_CAMERA_MAP = {
    "head": "cam_high",
    "left_wrist": "cam_left_wrist",
    "right_wrist": "cam_right_wrist",
}


def to_chw_uint8(image: Any) -> np.ndarray:
    image_np = np.asarray(image)
    if image_np.ndim == 3 and image_np.shape[-1] == 3:
        image_np = np.transpose(image_np, (2, 0, 1))
    elif image_np.ndim == 3 and image_np.shape[0] == 3:
        pass
    else:
        raise ValueError(f"Invalid image shape: {image_np.shape}")

    if image_np.dtype != np.uint8:
        if np.issubdtype(image_np.dtype, np.floating):
            scale = 255.0 if float(np.nanmax(image_np)) <= 1.5 else 1.0
            image_np = np.clip(image_np * scale, 0.0, 255.0).astype(np.uint8)
        else:
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(image_np)


def build_history_stack(history_frames: list[np.ndarray], image_history_interval: int) -> np.ndarray:
    if not history_frames:
        raise ValueError("history_frames must contain at least one frame.")

    past_idx = max(len(history_frames) - image_history_interval - 1, 0)
    current = history_frames[-1]
    past = history_frames[past_idx]
    return np.stack([past, current], axis=0)


def build_cubev2_request(
    *,
    qpos: np.ndarray,
    image_histories: Mapping[str, list[np.ndarray]],
    prompt: str,
    timestep: int,
    image_history_interval: int = 15,
    state_dim: int = 14,
    camera_name_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if camera_name_map is None:
        camera_name_map = DEFAULT_CAMERA_MAP

    state = np.zeros((state_dim,), dtype=np.float32)
    qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
    state[: min(state_dim, qpos.size)] = qpos[:state_dim]

    images: dict[str, np.ndarray] = {}
    for local_name, remote_name in camera_name_map.items():
        history = image_histories.get(local_name, [])
        if not history:
            blank = np.zeros((3, 480, 640), dtype=np.uint8)
            images[remote_name] = np.stack([blank, blank], axis=0)
            continue

        chw_history = [to_chw_uint8(frame) for frame in history]
        images[remote_name] = build_history_stack(chw_history, image_history_interval=image_history_interval)

    return {
        "images": images,
        "state": state,
        "prompt": prompt,
        "timestep": int(timestep),
        "reset": bool(timestep == 0),
    }
