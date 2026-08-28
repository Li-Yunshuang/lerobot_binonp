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

"""Cross-attention observation/goal encoder.

The load-bearing test here is `test_unet_parameter_count_unchanged`: the arm is only a clean
measurement of cross-attention if the conditioning width -- and therefore the U-Net -- is
identical to the baseline. If that ever drifts, the comparison silently becomes a capacity test.
"""

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pc_diffusion.configuration_pc_diffusion import PCDiffusionConfig
from lerobot.policies.pc_diffusion.encoders.cross_attention import CrossAttentionPointEncoder
from lerobot.policies.pc_diffusion.modeling_pc_diffusion import PCDiffusionModel


def _config(**overrides) -> PCDiffusionConfig:
    cfg = PCDiffusionConfig(
        horizon=16,
        n_action_steps=8,
        n_obs_steps=2,
        num_objects=1,
        aux_residual_weight=0.1,
        aux_predict_rotation=False,
        goal_conditioning="points",
        use_task_onehot=False,
        device="cpu",
        **overrides,
    )
    cfg.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "observation.point_cloud": PolicyFeature(type=FeatureType.POINT_CLOUD, shape=(128, 3)),
        "observation.goal_point_cloud": PolicyFeature(type=FeatureType.POINT_CLOUD, shape=(64, 3)),
    }
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(14,))}
    return cfg


def _batch(b: int = 2, s: int = 2) -> dict[str, torch.Tensor]:
    return {
        "observation.state": torch.randn(b, s, 14),
        "observation.point_cloud": torch.randn(b, s, 128, 3),
        "observation.goal_point_cloud": torch.randn(b, s, 64, 3),
    }


def test_encoder_output_width_matches_two_independent_encoders():
    """Contributes 2*out_dim -- exactly what the pair of separate encoders contributed."""
    enc = CrossAttentionPointEncoder(out_dim=256, hidden_dim=64, num_heads=2, num_layers=1)
    assert enc.feature_dim == 512
    out = enc(torch.randn(3, 128, 3), torch.randn(3, 64, 3))
    assert out.shape == (3, 512)


def test_handles_differently_sized_clouds():
    """The observation cloud is 1024 points and the goal cloud 512; they must not be assumed equal."""
    enc = CrossAttentionPointEncoder(out_dim=32, hidden_dim=64, num_heads=2, num_layers=1)
    assert enc(torch.randn(2, 200, 3), torch.randn(2, 17, 3)).shape == (2, 64)


def test_goal_cloud_actually_influences_the_observation_summary():
    """Guards against the goal branch being silently dropped -- the whole point of the arm."""
    torch.manual_seed(0)
    enc = CrossAttentionPointEncoder(out_dim=16, hidden_dim=32, num_heads=2, num_layers=1)
    # Zero-init output projections make the block start as identity, so perturb them first;
    # otherwise this would pass trivially at init and fail to catch a disconnected goal path.
    for p in enc.parameters():
        if p.requires_grad and p.dim() > 1:
            torch.nn.init.normal_(p, std=0.05)
    obs = torch.randn(1, 40, 3)
    a = enc(obs, torch.randn(1, 20, 3))
    b = enc(obs, torch.randn(1, 20, 3))
    assert not torch.allclose(a[:, :16], b[:, :16], atol=1e-6), (
        "observation half of the summary is independent of the goal cloud"
    )


def test_starts_as_identity_at_init():
    """Zero-init output projections: at step 0 attention contributes nothing."""
    enc = CrossAttentionPointEncoder(out_dim=16, hidden_dim=32, num_heads=2, num_layers=2)
    obs = torch.randn(2, 30, 3)
    with torch.no_grad():
        tokens = enc.point_mlp(obs) + enc.type_emb[0]
        expected = enc.obs_out(tokens.max(dim=1).values)
        got = enc(obs, torch.randn(2, 12, 3))[:, :16]
    torch.testing.assert_close(got, expected)


def test_unet_parameter_count_unchanged():
    """The arm must measure cross-attention, not extra downstream capacity."""
    base = PCDiffusionModel(_config(pc_cross_attention=False))
    cross = PCDiffusionModel(_config(pc_cross_attention=True))

    assert base.global_cond_dim == cross.global_cond_dim
    n_base = sum(p.numel() for p in base.unet.parameters())
    n_cross = sum(p.numel() for p in cross.unet.parameters())
    assert n_base == n_cross, f"U-Net changed size: {n_base} vs {n_cross}"

    # The aux head reads the same conditioning vector, so it must be unchanged too.
    a_base = sum(p.numel() for p in base.aux_head.parameters())
    a_cross = sum(p.numel() for p in cross.aux_head.parameters())
    assert a_base == a_cross


def test_forward_and_backward_with_cross_attention():
    model = PCDiffusionModel(_config(pc_cross_attention=True))
    batch = _batch()
    batch["action"] = torch.randn(2, 16, 14)
    batch["action_is_pad"] = torch.zeros(2, 16, dtype=torch.bool)
    # Legacy pose schema, as shipped by push_pc1024_poses -- the aux head needs one of the two.
    eye = torch.eye(4).expand(2, 2, 1, 4, 4).clone()
    batch["observation.object_poses"] = eye
    batch["observation.goal_object_poses"] = eye.clone()
    loss, out = model.compute_loss(batch)
    assert loss.isfinite()
    loss.backward()
    grads = [
        p.grad for p in model.cross_encoder.parameters() if p.grad is not None and p.grad.any()
    ]
    assert grads, "no gradient reached the cross-attention encoder"
    assert "aux_residual_loss" in out


def test_separate_encoders_are_not_built_when_cross_attention_is_on():
    model = PCDiffusionModel(_config(pc_cross_attention=True))
    assert model.cross_encoder is not None
    assert model.pc_encoder is None and model.goal_encoder is None


def test_requires_a_goal_cloud():
    cfg = _config(pc_cross_attention=True)
    cfg.goal_conditioning = "vector"
    cfg.goal_feature_key = "observation.goal"
    cfg.input_features["observation.goal"] = PolicyFeature(type=FeatureType.STATE, shape=(3,))
    with pytest.raises(ValueError, match="requires goal_conditioning to include points"):
        PCDiffusionModel(cfg)
