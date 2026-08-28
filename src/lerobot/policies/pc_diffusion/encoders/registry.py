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

"""Name -> point-cloud-encoder registry.

Encoder choice is a plain string in the policy config, so it round-trips through
``config.json`` and the Hub with no special handling, and a new architecture becomes available by
importing its module (which runs ``@register_pc_encoder``).
"""

from __future__ import annotations

from collections.abc import Callable

from .base import PointCloudEncoder

_PC_ENCODERS: dict[str, type[PointCloudEncoder]] = {}


def register_pc_encoder(name: str) -> Callable[[type[PointCloudEncoder]], type[PointCloudEncoder]]:
    """Class decorator registering a :class:`PointCloudEncoder` under ``name``."""

    def _register(cls: type[PointCloudEncoder]) -> type[PointCloudEncoder]:
        if name in _PC_ENCODERS and _PC_ENCODERS[name] is not cls:
            raise ValueError(f"Point-cloud encoder '{name}' is already registered to {_PC_ENCODERS[name]}.")
        if not issubclass(cls, PointCloudEncoder):
            raise TypeError(f"{cls.__name__} must subclass PointCloudEncoder to be registered.")
        _PC_ENCODERS[name] = cls
        cls._registry_name = name  # type: ignore[attr-defined]
        return cls

    return _register


def available_pc_encoders() -> list[str]:
    return sorted(_PC_ENCODERS)


def make_pc_encoder(
    name: str, *, num_points: int, in_channels: int, out_dim: int, **kwargs
) -> PointCloudEncoder:
    """Instantiate a registered encoder.

    Raises:
        ValueError: if ``name`` is not registered, listing what is.
    """
    if name not in _PC_ENCODERS:
        raise ValueError(
            f"Unknown point-cloud encoder '{name}'. Available: {available_pc_encoders()}. "
            "Register a new one with @register_pc_encoder in lerobot.policies.pc_diffusion.encoders."
        )
    return _PC_ENCODERS[name](num_points=num_points, in_channels=in_channels, out_dim=out_dim, **kwargs)
