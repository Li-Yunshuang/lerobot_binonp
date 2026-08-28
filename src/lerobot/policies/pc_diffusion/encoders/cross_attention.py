#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Joint observation/goal encoder with bidirectional cross-attention.

The baseline encodes each cloud independently to a 256-d vector and concatenates the two, so the
network has to infer "where is the object relative to where it should be" from two separately
max-pooled summaries. A max-pool keeps, per channel, the single most extreme point -- which point
that is differs between the two clouds, so the correspondence the task actually needs is destroyed
before the two ever meet.

Here the clouds meet *before* pooling: observation points attend to goal points and vice versa, so
a point can pick up "the goal surface nearest me is 4 cm that way" while it is still a point.

Deliberately matched to the baseline it replaces:

* The output is ``2 * out_dim``, exactly what the two independent encoders contributed together,
  so ``global_cond_dim`` -- and therefore the U-Net parameter count -- is unchanged. The arm
  measures the cross-attention, not extra capacity downstream.
* The per-point MLP is **shared** between the clouds, with a learned type embedding marking which
  cloud a token came from. Both are geometry in the same frame; giving them separate encoders
  would let the model drift into two incompatible representations, which is the defect this is
  meant to fix.

Attention is computed with ``F.scaled_dot_product_attention`` so the 1024x512 score matrix is
never materialised -- at batch 64 with n_obs_steps 2 that would be ~1 GB per layer per direction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class _MultiheadCrossAttention(nn.Module):
    """Pre-norm multi-head cross-attention, ``x`` attending to ``ctx``."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm_x = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        # Zero-init the output projection so the block starts as identity: at step 0 this encoder
        # is exactly the baseline plus a no-op, which keeps early training from being dominated by
        # untrained attention noise.
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def _heads(self, t: Tensor) -> Tensor:
        b, n, _ = t.shape
        return t.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: Tensor, ctx: Tensor) -> Tensor:
        xn, cn = self.norm_x(x), self.norm_ctx(ctx)
        q, k, v = self._heads(self.to_q(xn)), self._heads(self.to_k(cn)), self._heads(self.to_v(cn))
        out = F.scaled_dot_product_attention(q, k, v)  # (B, H, N, head_dim)
        out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        return x + self.to_out(out)


class _FeedForward(nn.Module):
    """Pre-norm position-wise FFN with a residual connection."""

    def __init__(self, dim: int, mult: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class CrossAttentionPointEncoder(nn.Module):
    """Encode an observation cloud and a goal cloud jointly.

    Args:
        in_channels: per-point channels (3 for xyz).
        out_dim: width contributed *per cloud*; ``forward`` returns ``2 * out_dim``.
        hidden_dim: token width inside the attention stack.
        num_heads: attention heads.
        num_layers: bidirectional cross-attention blocks.
        point_hidden_dims: per-point MLP widths before attention.
        use_layernorm: LayerNorm inside the per-point MLP, as in the baseline encoder.
    """

    def __init__(
        self,
        *,
        in_channels: int = 3,
        out_dim: int = 256,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        point_hidden_dims: tuple[int, ...] = (64, 128),
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim

        layers: list[nn.Module] = []
        prev = in_channels
        for width in point_hidden_dims:
            layers.append(nn.Linear(prev, width))
            if use_layernorm:
                layers.append(nn.LayerNorm(width))
            layers.append(nn.ReLU(inplace=True))
            prev = width
        layers.append(nn.Linear(prev, hidden_dim))
        self.point_mlp = nn.Sequential(*layers)

        # Marks whether a token came from the observation or the goal cloud. Needed because the
        # per-point MLP is shared, so the two are otherwise indistinguishable.
        self.type_emb = nn.Parameter(torch.zeros(2, hidden_dim))
        nn.init.normal_(self.type_emb, std=0.02)

        self.obs_attn = nn.ModuleList()
        self.goal_attn = nn.ModuleList()
        self.obs_ff = nn.ModuleList()
        self.goal_ff = nn.ModuleList()
        for _ in range(num_layers):
            self.obs_attn.append(_MultiheadCrossAttention(hidden_dim, num_heads))
            self.goal_attn.append(_MultiheadCrossAttention(hidden_dim, num_heads))
            self.obs_ff.append(_FeedForward(hidden_dim))
            self.goal_ff.append(_FeedForward(hidden_dim))

        self.obs_out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, out_dim))
        self.goal_out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, out_dim))

    @property
    def feature_dim(self) -> int:
        """Total width contributed to the global conditioning vector."""
        return 2 * self.out_dim

    def forward(self, obs_pc: Tensor, goal_pc: Tensor) -> Tensor:
        """Encode both clouds jointly.

        Args:
            obs_pc: ``(B, N, C)`` observation cloud.
            goal_pc: ``(B, M, C)`` goal cloud. ``M`` need not equal ``N``.

        Returns:
            ``(B, 2 * out_dim)`` -- the observation summary concatenated with the goal summary.
        """
        for name, pc in (("obs_pc", obs_pc), ("goal_pc", goal_pc)):
            if pc.ndim != 3:
                raise ValueError(f"Expected (B, N, C) for {name}, got shape {tuple(pc.shape)}")
        if obs_pc.shape[0] != goal_pc.shape[0]:
            raise ValueError(
                f"Batch mismatch: obs_pc {obs_pc.shape[0]} vs goal_pc {goal_pc.shape[0]}"
            )

        o = self.point_mlp(obs_pc) + self.type_emb[0]
        g = self.point_mlp(goal_pc) + self.type_emb[1]

        # Both directions read the *same* inputs within a layer, so neither gets a stale view.
        for oa, ga, off, gff in zip(
            self.obs_attn, self.goal_attn, self.obs_ff, self.goal_ff, strict=True
        ):
            o_new = oa(o, g)
            g_new = ga(g, o)
            o, g = off(o_new), gff(g_new)

        o = self.obs_out(o.max(dim=1).values)
        g = self.goal_out(g.max(dim=1).values)
        return torch.cat([o, g], dim=-1)
