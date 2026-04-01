#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


def resolve_da3_import(code_root: str | None) -> type:
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as first_exc:
        if code_root is None:
            raise ImportError(
                "Failed to import DepthAnything3. Install the `depth_anything_3` package or set "
                "`policy.da3_code_root` to a standalone DA3 repository root or its `src` directory."
            ) from first_exc

        candidate = Path(code_root).expanduser().resolve()
        candidate_src = candidate / "src" if (candidate / "src").exists() else candidate
        candidate_src_str = str(candidate_src)
        if not candidate_src.exists():
            raise FileNotFoundError(
                f"policy.da3_code_root={code_root!r} does not exist or does not contain a `src` directory"
            )
        if candidate_src_str not in sys.path:
            sys.path.append(candidate_src_str)
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as second_exc:
            raise ImportError(
                "Failed to import DepthAnything3 from the configured `policy.da3_code_root`. "
                "Expected a standalone DA3 checkout or installed `depth_anything_3` package."
            ) from second_exc

    return DepthAnything3


class DA3BackboneTeacher(nn.Module):
    def __init__(
        self,
        model_path_or_name: str,
        process_res: int = 504,
        dtype: torch.dtype = torch.bfloat16,
        teacher_layers: tuple[int, ...] | None = None,
        code_root: str | None = None,
    ):
        super().__init__()
        DepthAnything3 = resolve_da3_import(code_root)

        self.wrapper = DepthAnything3.from_pretrained(model_path_or_name)
        self.model = self.wrapper.model
        self.out_layers = tuple(int(layer_idx) for layer_idx in self.wrapper.config.net.out_layers)
        self.teacher_layers = self.out_layers if teacher_layers is None else tuple(int(layer_idx) for layer_idx in teacher_layers)
        missing_layers = [layer_idx for layer_idx in self.teacher_layers if layer_idx not in self.out_layers]
        if missing_layers:
            raise ValueError(
                f"Requested DA3 layers {self.teacher_layers} are not available from teacher backbone layers {self.out_layers}. "
                f"Missing: {missing_layers}"
            )
        self.process_res = process_res
        self._dtype = dtype

        self.model.to(dtype=dtype)
        self.model.eval()
        self.model.requires_grad_(False)

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=dtype).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=dtype).view(1, 1, 3, 1, 1),
            persistent=False,
        )

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        model_param = next(self.model.parameters())
        x = images.to(device=model_param.device, dtype=model_param.dtype)
        if x.max() > 1.0:
            x = x / 255.0
        x = (x - self.mean.to(device=x.device, dtype=x.dtype)) / self.std.to(device=x.device, dtype=x.dtype)

        bsize, num_views, channels, height, width = x.shape
        x = x.view(bsize * num_views, channels, height, width)
        x = F.interpolate(
            x,
            size=(self.process_res, self.process_res),
            mode="bilinear",
            align_corners=False,
        )
        x = x.view(bsize, num_views, channels, self.process_res, self.process_res)

        features_tuple, _ = self.model.backbone(x)
        layer_outputs = {layer_idx: layer_out for layer_idx, layer_out in zip(self.out_layers, features_tuple, strict=False)}
        teacher_features = []
        for layer_idx in self.teacher_layers:
            layer_out = layer_outputs[layer_idx]
            patch_tokens = layer_out[0]
            teacher_features.append(patch_tokens.reshape(bsize, -1, patch_tokens.shape[-1]))
        return teacher_features
