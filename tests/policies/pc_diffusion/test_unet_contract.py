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

"""Pin the contract `pc_diffusion` relies on when importing the image policy's U-Net.

`pc_diffusion` deliberately reuses `DiffusionConditionalUnet1d` and `_make_noise_scheduler`
rather than copying ~200 lines, on the grounds that they are modality-agnostic: they read only a
handful of config fields and see a flat conditioning vector. That is a cross-package dependency
no other policy in the repo has, so these tests exist to make an upstream signature or
field-name change fail loudly here instead of silently at train time.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.diffusion.modeling_diffusion import (  # noqa: E402
    DiffusionConditionalUnet1d,
    _make_noise_scheduler,
)
from lerobot.policies.pc_diffusion.configuration_pc_diffusion import PCDiffusionConfig  # noqa: E402

ACTION_DIM = 14
GLOBAL_COND_DIM = 96


def _config() -> PCDiffusionConfig:
    cfg = PCDiffusionConfig(
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,
        down_dims=(64, 128),
        diffusion_step_embed_dim=32,
        device="cpu",
        push_to_hub=False,
    )
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
    return cfg


def test_unet_accepts_pc_diffusion_config():
    """The U-Net must be constructible from PCDiffusionConfig, not just DiffusionConfig."""
    unet = DiffusionConditionalUnet1d(_config(), global_cond_dim=GLOBAL_COND_DIM)
    batch, horizon = 2, 16
    sample = torch.randn(batch, horizon, ACTION_DIM)
    timesteps = torch.zeros(batch, dtype=torch.long)
    out = unet(sample, timesteps, global_cond=torch.randn(batch, GLOBAL_COND_DIM))
    assert out.shape == sample.shape


def test_unet_reads_only_known_config_fields():
    """Guard the exact field names the U-Net pulls off the config.

    If upstream renames one of these, PCDiffusionConfig stops satisfying the duck-typed
    contract; failing here names the field instead of raising a bare AttributeError later.
    """
    cfg = _config()
    for field in (
        "diffusion_step_embed_dim",
        "down_dims",
        "kernel_size",
        "n_groups",
        "use_film_scale_modulation",
        "gradient_checkpointing",
    ):
        assert hasattr(cfg, field), f"PCDiffusionConfig is missing U-Net field {field!r}"
    assert cfg.action_feature is not None and cfg.action_feature.shape[0] == ACTION_DIM


@pytest.mark.parametrize("name", ["DDPM", "DDIM"])
def test_noise_scheduler_factory(name):
    sched = _make_noise_scheduler(
        name,
        num_train_timesteps=100,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        clip_sample_range=1.0,
        prediction_type="epsilon",
    )
    assert sched.config.num_train_timesteps == 100
