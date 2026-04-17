from __future__ import annotations

import io
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError

try:
    import h5py
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError("h5py is required for HDF5 utilities.") from exc


DEFAULT_INSTRUCTION = "Clear the junk and items off the desktop."

IMAGE_LAYOUTS: dict[str, OrderedDict[str, str]] = {
    "head_left_right": OrderedDict(
        [
            ("head", "observation.images.head"),
            ("left", "observation.images.left"),
            ("right", "observation.images.right"),
        ]
    ),
    "head_left": OrderedDict(
        [
            ("head", "observation.images.head"),
            ("left", "observation.images.left"),
        ]
    ),
    "libero_2cam": OrderedDict(
        [
            ("head", "observation.images.image"),
            ("left", "observation.images.wrist_image"),
        ]
    ),
}


@dataclass
class EpisodeSpec:
    state_dim: int
    action_dim: int
    image_shapes: OrderedDict[str, tuple[int, int, int]]


def build_camera_source_keys(
    image_layout: str,
    *,
    head_key: str,
    left_key: str,
    right_key: str,
) -> OrderedDict[str, str]:
    camera_keys = OrderedDict([("head", head_key), ("left", left_key)])
    if image_layout == "head_left_right":
        camera_keys["right"] = right_key
    return camera_keys


def iter_h5_datasets(group: h5py.Group, prefix: str = "") -> Iterable[tuple[str, h5py.Dataset]]:
    for key, value in group.items():
        path = f"{prefix}/{key}" if prefix else f"/{key}"
        if isinstance(value, h5py.Dataset):
            yield path, value
        elif isinstance(value, h5py.Group):
            yield from iter_h5_datasets(value, path)


def collect_h5_tree_lines(group: h5py.Group, prefix: str = "") -> list[str]:
    lines: list[str] = []
    for key, value in group.items():
        path = f"{prefix}/{key}" if prefix else f"/{key}"
        if isinstance(value, h5py.Group):
            lines.append(f"[G] {path}")
            lines.extend(collect_h5_tree_lines(value, path))
        else:
            lines.append(f"[D] {path}  shape={tuple(value.shape)}  dtype={value.dtype}")
    return lines


def h5_attr_to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return h5_attr_to_text(value.item())
    return str(value).strip()


def derive_instruction(
    h5_file: h5py.File,
    episode_path: Path,
    *,
    input_root: Path,
    instruction: str | None,
    task_attr: str,
) -> str:
    if instruction:
        return instruction

    attr_text = h5_attr_to_text(h5_file.attrs.get(task_attr))
    if attr_text:
        return attr_text

    rel_parent = episode_path.parent.relative_to(input_root)
    if str(rel_parent) != ".":
        return str(rel_parent).replace("\\", " / ")

    stem = re.sub(r"[_\-]+", " ", episode_path.stem).strip()
    return stem or DEFAULT_INSTRUCTION


def load_h5_array(h5_file: h5py.File, key: str) -> np.ndarray:
    if key not in h5_file:
        raise KeyError(f"HDF5 key not found: {key}")
    return np.asarray(h5_file[key])


def decode_image_frame(raw_frame, raw_image_shape: tuple[int, int, int] | None) -> np.ndarray:
    """Decode one frame into a uint8 HWC RGB array."""
    if isinstance(raw_frame, (bytes, bytearray, np.bytes_)):
        byte_candidates = [bytes(raw_frame)]
        frame = None
    elif isinstance(raw_frame, np.ndarray):
        frame = raw_frame
        byte_candidates = []
    else:
        frame = np.asarray(raw_frame)
        byte_candidates = []

    if frame is not None and frame.ndim == 3:
        if frame.shape[-1] in (1, 3, 4):
            image = frame
        elif frame.shape[0] in (1, 3, 4):
            image = np.transpose(frame, (1, 2, 0))
        else:
            raise ValueError(f"Unsupported 3D image shape: {frame.shape}")
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        if image.shape[-1] == 4:
            image = image[..., :3]
        return np.ascontiguousarray(image.astype(np.uint8, copy=False))

    if frame is not None and frame.ndim == 1 and frame.dtype == np.uint8:
        byte_candidates.extend([frame.tobytes(), np.trim_zeros(frame, trim="b").tobytes()])
    elif frame is not None and frame.ndim == 0 and frame.dtype.kind in {"S", "O", "U"}:
        byte_candidates.append(frame.item())

    for candidate in byte_candidates:
        if not candidate:
            continue
        if isinstance(candidate, str):
            candidate = candidate.encode("utf-8", errors="ignore")
        try:
            with Image.open(io.BytesIO(candidate)) as image:
                return np.asarray(image.convert("RGB"), dtype=np.uint8)
        except (UnidentifiedImageError, OSError, TypeError):
            pass

    if frame is not None and raw_image_shape is not None and frame.size == int(np.prod(raw_image_shape)):
        h, w, c = raw_image_shape
        return frame.reshape(h, w, c).astype(np.uint8, copy=False)

    raise ValueError(
        "Failed to decode image frame. "
        f"Shape={None if frame is None else frame.shape}, "
        f"dtype={None if frame is None else frame.dtype}. "
        "If frames are stored as raw flattened arrays, pass --raw-image-shape H W C."
    )


def validate_episode_lengths(
    episode_path: Path,
    state_array: np.ndarray,
    action_array: np.ndarray,
    image_arrays: OrderedDict[str, np.ndarray],
) -> int:
    lengths = {
        "state": int(state_array.shape[0]),
        "action": int(action_array.shape[0]),
        **{camera_name: int(camera_array.shape[0]) for camera_name, camera_array in image_arrays.items()},
    }
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(f"Mismatched frame counts in {episode_path}: {lengths}")
    return next(iter(unique_lengths))


def infer_episode_spec(
    episode_path: Path,
    *,
    state_key: str,
    action_key: str,
    camera_source_keys: OrderedDict[str, str],
    raw_image_shape: tuple[int, int, int] | None,
) -> EpisodeSpec:
    with h5py.File(episode_path, "r") as h5_file:
        state_array = load_h5_array(h5_file, state_key)
        action_array = load_h5_array(h5_file, action_key)
        if state_array.ndim != 2:
            raise ValueError(f"Expected 2D state array at {state_key}, got {state_array.shape}")
        if action_array.ndim != 2:
            raise ValueError(f"Expected 2D action array at {action_key}, got {action_array.shape}")

        image_shapes: OrderedDict[str, tuple[int, int, int]] = OrderedDict()
        for camera_name, source_key in camera_source_keys.items():
            image_array = load_h5_array(h5_file, source_key)
            if image_array.ndim < 2:
                raise ValueError(f"Expected image dataset with time dimension at {source_key}, got {image_array.shape}")
            decoded = decode_image_frame(image_array[0], raw_image_shape)
            image_shapes[camera_name] = tuple(int(v) for v in decoded.shape)

    return EpisodeSpec(
        state_dim=int(state_array.shape[1]),
        action_dim=int(action_array.shape[1]),
        image_shapes=image_shapes,
    )


def resolve_sample_indices(frame_count: int, indices: Iterable[int]) -> list[int]:
    resolved: list[int] = []
    for idx in indices:
        resolved_idx = frame_count + idx if idx < 0 else idx
        if 0 <= resolved_idx < frame_count and resolved_idx not in resolved:
            resolved.append(resolved_idx)
    return resolved


def make_vector_names(prefix: str, dim: int) -> list[str]:
    return [f"{prefix}_{idx}" for idx in range(dim)]
