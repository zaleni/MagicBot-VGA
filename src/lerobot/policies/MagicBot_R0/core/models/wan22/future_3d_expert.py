from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wan_video_dit import GateModule, SelfAttention, precompute_freqs_cis, sinusoidal_embedding_1d


class Future3DPerceiverFeedForward(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim, bias=False),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Future3DPerceiverAttention(nn.Module):
    def __init__(self, dim: int, dim_head: int, heads: int):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.memory_norm = nn.LayerNorm(dim)
        self.query_norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, width = x.shape
        x = x.view(batch_size, seq_len, self.heads, width // self.heads)
        return x.transpose(1, 2)

    def forward(self, memory: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        memory = self.memory_norm(memory)
        queries = self.query_norm(queries)

        q = self._reshape_heads(self.to_q(queries))
        k, v = self.to_kv(memory).chunk(2, dim=-1)
        k = self._reshape_heads(k)
        v = self._reshape_heads(v)

        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(queries.shape[0], queries.shape[1], -1)
        return self.to_out(attended)


class Future3DPerceiverResampler(nn.Module):
    def __init__(self, dim: int, num_heads: int, output_dim: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        dim_head = dim // num_heads
        self.attn = Future3DPerceiverAttention(dim=dim, dim_head=dim_head, heads=num_heads)
        self.ff = Future3DPerceiverFeedForward(dim=dim)
        self.output_norm = nn.LayerNorm(dim)
        self.output_proj = nn.Linear(dim, output_dim)

    def forward(self, output_queries: torch.Tensor, messenger_tokens: torch.Tensor) -> torch.Tensor:
        latents = output_queries
        latents = latents + self.attn(messenger_tokens, latents)
        latents = latents + self.ff(latents)
        return self.output_proj(self.output_norm(latents))


class Future3DBlock(nn.Module):
    """DiT-compatible block without text cross-attention.

    MoT only requires each expert block to expose `self_attn`, `norm1`, `norm2`,
    `ffn`, `modulation`, and `gate`. Keeping this block cross-attn-free makes the
    future-3D branch a compact auxiliary expert instead of another full DiT.
    """

    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_head_dim = attn_head_dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_dim) / hidden_dim**0.5)
        self.gate = GateModule()


class Future3DExpert(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 768,
        ffn_dim: int = 3072,
        text_dim: int = 4096,
        freq_dim: int = 256,
        eps: float = 1e-6,
        num_heads: int = 24,
        attn_head_dim: int = 128,
        num_layers: int = 30,
        num_query_tokens: int = 288,
        da3_num_views: int = 2,
        da3_tokens_per_view: int = 1296,
        da3_query_dim: int = 2048,
        query_layer_indices: tuple[int, ...] = (13, 19, 23, 27),
        future_query_init_std: float = 0.02,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        del text_dim
        if num_query_tokens <= 0:
            raise ValueError(f"`num_query_tokens` must be positive, got {num_query_tokens}")
        if da3_num_views <= 0:
            raise ValueError(f"`da3_num_views` must be positive, got {da3_num_views}")
        if num_query_tokens % da3_num_views != 0:
            raise ValueError(
                f"`num_query_tokens` ({num_query_tokens}) must be divisible by da3_num_views ({da3_num_views})"
            )

        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.num_layers = num_layers
        self.num_query_tokens = num_query_tokens
        self.da3_num_views = da3_num_views
        self.da3_tokens_per_view = da3_tokens_per_view
        self.da3_query_dim = da3_query_dim
        self.query_layer_indices = tuple(int(idx) for idx in query_layer_indices)
        self.query_tokens_per_view = num_query_tokens // da3_num_views

        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_dim) * future_query_init_std)
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        self.blocks = nn.ModuleList(
            [
                Future3DBlock(
                    hidden_dim=hidden_dim,
                    attn_head_dim=attn_head_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.freqs = precompute_freqs_cis(attn_head_dim, end=max(1024, num_query_tokens))
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in self.query_layer_indices])
        self.output_queries = nn.Parameter(
            torch.randn(1, da3_tokens_per_view, hidden_dim) * future_query_init_std
        )
        self.output_decoder = Future3DPerceiverResampler(
            dim=hidden_dim,
            num_heads=num_heads,
            output_dim=da3_query_dim,
        )
        self.use_gradient_checkpointing = use_gradient_checkpointing

    def pre_dit(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        timestep: torch.Tensor | None = None,
    ) -> Dict[str, Any]:
        if timestep is None:
            timestep = torch.zeros((batch_size,), device=device, dtype=dtype)
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B], got shape {tuple(timestep.shape)}")
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("During training, future-3D timestep length must match batch_size.")
            timestep = timestep.expand(batch_size)
        if timestep.shape[0] != batch_size:
            raise ValueError(f"`timestep` length must be {batch_size}, got {timestep.shape[0]}")

        tokens = self.query_tokens.expand(batch_size, -1, -1).to(device=device, dtype=dtype)
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))
        freqs = self.freqs[: self.num_query_tokens].view(self.num_query_tokens, 1, -1).to(device)
        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": None,
            "context_mask": None,
            "meta": {
                "batch_size": batch_size,
                "seq_len": self.num_query_tokens,
            },
        }

    def project_query_layers(self, layer_tokens: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        if len(layer_tokens) != len(self.query_layer_indices):
            raise ValueError(
                f"Expected {len(self.query_layer_indices)} future-3D layers, got {len(layer_tokens)}"
            )

        projected = []
        for layer_norm, tokens in zip(self.layer_norms, layer_tokens, strict=True):
            tokens = layer_norm(tokens)
            batch_size = tokens.shape[0]
            messenger = tokens.view(batch_size, self.da3_num_views, self.query_tokens_per_view, self.hidden_dim)
            messenger = messenger.reshape(batch_size * self.da3_num_views, self.query_tokens_per_view, self.hidden_dim)

            output_queries = self.output_queries.expand(batch_size, self.da3_tokens_per_view, -1)
            output_queries = output_queries[:, None, :, :].expand(
                batch_size, self.da3_num_views, self.da3_tokens_per_view, self.hidden_dim
            )
            output_queries = output_queries.reshape(
                batch_size * self.da3_num_views, self.da3_tokens_per_view, self.hidden_dim
            )

            decoded = self.output_decoder(output_queries, messenger)
            decoded = decoded.view(batch_size, self.da3_num_views * self.da3_tokens_per_view, self.da3_query_dim)
            projected.append(decoded)
        return projected
