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

from dataclasses import dataclass, field, replace

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.policies.cubev2.da3_teacher import resolve_da3_backbone_defaults
from lerobot.policies.cubev2.transform_internvla_a1 import (
    Qwen3_VLProcessorTransformFn,
    UnifyCubeV2InputsTransformFn,
)
from lerobot.transforms.core import *
from lerobot.utils.constants import OBS_IMAGES


@DatasetConfig.register_subclass("cubev2")
@dataclass
class CubeV2DatasetConfig(DatasetConfig):
    height: int = 224
    width: int = 224
    max_state_dim: int = 32
    max_action_dim: int = 32
    qwen3_vl_processor_path: str = "Qwen/Qwen3-VL-2B-Instruct"

    data_transforms: TransformGroup = field(
        default_factory=lambda: TransformGroup(
            inputs=[
                DeltaActionTransformFn(),
                ResizeImagesWithPadFn(
                    height=CubeV2DatasetConfig.height,
                    width=CubeV2DatasetConfig.width,
                ),
                RemapImageKeyTransformFn(),
                NormalizeTransformFn(),
                ComposeFieldsTransform(),
                PadStateAndActionTransformFn(
                    max_state_dim=CubeV2DatasetConfig.max_state_dim,
                    max_action_dim=CubeV2DatasetConfig.max_action_dim,
                ),
                Qwen3_VLProcessorTransformFn(),
                UnifyCubeV2InputsTransformFn(),
            ],
            outputs=[],
        )
    )

    def __post_init__(self):
        super().__post_init__()
        inputs = list(self.data_transforms.inputs)
        for idx, transform in enumerate(inputs):
            if isinstance(transform, Qwen3_VLProcessorTransformFn):
                inputs[idx] = replace(
                    transform,
                    pretrained_model_name_or_path=self.qwen3_vl_processor_path,
                )
        has_delta = any(isinstance(t, DeltaActionTransformFn) for t in inputs)
        if self.action_mode == "delta":
            if not has_delta:
                inputs = [DeltaActionTransformFn(), *inputs]
                self.data_transforms = replace(self.data_transforms, inputs=inputs)
        else:
            if has_delta:
                inputs = [t for t in inputs if not isinstance(t, DeltaActionTransformFn)]
                self.data_transforms = replace(self.data_transforms, inputs=inputs)


@PreTrainedConfig.register_subclass("cubev2")
@dataclass
class CubeV2Config(PreTrainedConfig):
    qwen3_vl_variant: str = "qwen3_vl_2b"
    action_expert_variant: str = "qwen3_600m"
    qwen3_vl_pretrained_path: str = "Qwen/Qwen3-VL-2B-Instruct"
    dtype: str = "bfloat16"

    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    max_state_dim: int = 32
    max_action_dim: int = 32

    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    image_resolution: tuple[int, int] = (224, 224)
    empty_cameras: int = 0

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    gradient_checkpointing: bool = False
    compile_model: bool = False
    compile_mode: str = "max-autotune"
    device: str | None = None

    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    tokenizer_max_length: int = 48

    freeze_vision_encoder: bool = False
    train_expert_only: bool = False
    train_vlm_only: bool = False

    scale_factor: int = 8
    lambda_gen: float = 0.01
    cosmos_tokenizer_path_or_name: str = "nvidia/Cosmos-Tokenizer-CI8x8"

    enable_3d_queries: bool = False
    num_3d_query_tokens: int = 1296  # compressed future-3D bottleneck queries
    da3_alignment_mode: str = "query_decoder"
    da3_query_resampler_layers: int = 1  # kept for config compatibility; fixed to 1
    da3_query_resampler_ff_mult: int = 1  # kept for config compatibility; fixed to 1
    query_layer_indices: tuple[int, ...] = (13, 19, 23, 27)
    da3_variant: str = "auto"
    da3_teacher_layers: tuple[int, ...] | None = None
    da3_query_dim: int | None = None
    da3_tokens_per_view: int = 1296
    da3_num_views: int = 3
    lambda_3d: float = 0.05
    da3_model_path_or_name: str = "depth-anything/DA3-GIANT-1.1"
    da3_model_name: str | None = None
    da3_code_root: str | None = None
    da3_teacher_process_res: int = 504
    da3_layer_weights: tuple[float, ...] = (1.0, 1.2, 1.4, 1.6)
    future_query_init_std: float = 0.02
    log_da3_teacher_timing: bool = False

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        if self.enable_3d_queries and self.num_3d_query_tokens <= 0:
            raise ValueError("num_3d_query_tokens must be positive when 3D queries are enabled")

        if self.enable_3d_queries and self.num_3d_query_tokens % self.da3_num_views != 0:
            raise ValueError(
                "num_3d_query_tokens must be divisible by da3_num_views for view-aware 3D alignment"
            )

        if self.da3_alignment_mode not in {"query_decoder", "upsample"}:
            raise ValueError(
                "da3_alignment_mode must be one of: 'query_decoder', 'upsample'"
            )
        if self.da3_query_resampler_layers != 1:
            raise ValueError("da3_query_resampler_layers is fixed to 1 in the current CubeV2 query decoder")
        if self.da3_query_resampler_ff_mult != 1:
            raise ValueError("da3_query_resampler_ff_mult is fixed to 1 in the current CubeV2 query decoder")

        if self.da3_model_name is not None:
            self.da3_model_path_or_name = self.da3_model_name

        da3_defaults = resolve_da3_backbone_defaults(
            self.da3_model_path_or_name,
            self.da3_variant,
        )
        if self.da3_teacher_layers is None:
            self.da3_teacher_layers = tuple(int(layer_idx) for layer_idx in da3_defaults["teacher_layers"])
        if self.da3_query_dim is None:
            self.da3_query_dim = int(da3_defaults["query_dim"])

        if len(self.query_layer_indices) != len(self.da3_teacher_layers):
            raise ValueError("query_layer_indices and da3_teacher_layers must have the same length")

        if len(self.query_layer_indices) != len(self.da3_layer_weights):
            raise ValueError("da3_layer_weights must align with query_layer_indices")

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            self.input_features[key] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),
            )

        if "observation.state" not in self.input_features:
            self.input_features["observation.state"] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),
            )

        if "action" not in self.output_features:
            self.output_features["action"] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def image_delta_indices(self) -> list | None:
        return [-15, 0, 15]


QwenA1Config = CubeV2Config
QwenA1DatasetConfig = CubeV2DatasetConfig
