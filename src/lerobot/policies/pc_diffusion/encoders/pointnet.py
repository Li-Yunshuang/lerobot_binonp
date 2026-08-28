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

"""PointNet-style max-pool encoder -- the baseline control.

This is the DP3 encoder (Ze et al., RSS 2024): a per-point MLP followed by a global max-pool.
It is deliberately the *first* encoder rather than the intended final one: DP3 reports that this
simple design matches or beats hierarchical encoders (PointNet++, PointNeXt, Point Transformer)
for visuomotor policy learning, so any heavier architecture added later should be measured
against it rather than assumed better.

Note it is permutation-invariant and has no local neighbourhood aggregation -- a single global
max-pool is the only mixing between points. That is exactly the limitation a stronger encoder
would target.
"""

from __future__ import annotations

from torch import Tensor, nn

from .base import PointCloudEncoder
from .registry import register_pc_encoder


@register_pc_encoder("pointnet_maxpool")
class PointNetMaxPoolEncoder(PointCloudEncoder):
    """Per-point MLP -> global max-pool -> projection.

    Args:
        num_points: points per cloud (unused by the math; kept for interface symmetry).
        in_channels: per-point channel count (3 for xyz).
        out_dim: width of the returned feature vector.
        hidden_dims: per-point MLP widths.
        use_layernorm: LayerNorm after each per-point linear. DP3 finds this matters; without it
            the max-pool tends to be dominated by a few extreme coordinates.
        final_norm: ``"layernorm"`` or ``"none"`` on the projection output.
    """

    def __init__(
        self,
        *,
        num_points: int,
        in_channels: int,
        out_dim: int,
        hidden_dims: tuple[int, ...] = (64, 128, 256),
        use_layernorm: bool = True,
        final_norm: str = "layernorm",
        **kwargs,
    ) -> None:
        super().__init__(num_points=num_points, in_channels=in_channels, out_dim=out_dim)

        layers: list[nn.Module] = []
        prev = in_channels
        for width in hidden_dims:
            layers.append(nn.Linear(prev, width))
            if use_layernorm:
                layers.append(nn.LayerNorm(width))
            layers.append(nn.ReLU(inplace=True))
            prev = width
        self.point_mlp = nn.Sequential(*layers)

        proj: list[nn.Module] = [nn.Linear(prev, out_dim)]
        if final_norm == "layernorm":
            proj.append(nn.LayerNorm(out_dim))
        elif final_norm != "none":
            raise ValueError(f"final_norm must be 'layernorm' or 'none', got {final_norm!r}")
        self.projection = nn.Sequential(*proj)

    def forward(self, pc: Tensor) -> Tensor:
        if pc.ndim != 3:
            raise ValueError(f"Expected (B, N, C) point cloud, got shape {tuple(pc.shape)}")
        x = self.point_mlp(pc)  # (B, N, hidden)
        x = x.max(dim=1).values  # (B, hidden) -- global max-pool over points
        return self.projection(x)  # (B, out_dim)
