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

"""Interface every point-cloud encoder must implement.

Keeping this surface tiny is the point: swapping architectures should mean adding one file under
``encoders/`` and changing ``--policy.pc_encoder=<name>``, with nothing else in the policy aware
of which encoder is in use.
"""

from __future__ import annotations

import abc

from torch import Tensor, nn


class PointCloudEncoder(nn.Module, abc.ABC):
    """Maps a batch of point clouds to a flat feature vector.

    Implementations receive already-normalized clouds (the policy's preprocessor pipeline handles
    normalization) with the batch and observation-step dimensions already flattened together, so
    they only ever see a plain ``(B, N, C)`` tensor.
    """

    def __init__(self, *, num_points: int, in_channels: int, out_dim: int, **kwargs) -> None:
        super().__init__()
        self.num_points = num_points
        self.in_channels = in_channels
        self.out_dim = out_dim

    @property
    def feature_dim(self) -> int:
        """Width of the vector this encoder contributes to the diffusion global conditioning."""
        return self.out_dim

    @abc.abstractmethod
    def forward(self, pc: Tensor) -> Tensor:
        """Encode a batch of clouds.

        Args:
            pc: ``(B, N, C)`` float tensor. ``N`` is ``num_points``, ``C`` is ``in_channels``.

        Returns:
            ``(B, feature_dim)``.
        """
        raise NotImplementedError
