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

"""Point-cloud encoders for `pc_diffusion`.

Importing a module here is what registers its encoder, so every built-in must be imported below.
"""

from .base import PointCloudEncoder as PointCloudEncoder
from .pointnet import PointNetMaxPoolEncoder as PointNetMaxPoolEncoder
from .registry import available_pc_encoders as available_pc_encoders
from .registry import make_pc_encoder as make_pc_encoder
from .registry import register_pc_encoder as register_pc_encoder

__all__ = [
    "PointCloudEncoder",
    "PointNetMaxPoolEncoder",
    "available_pc_encoders",
    "make_pc_encoder",
    "register_pc_encoder",
]
