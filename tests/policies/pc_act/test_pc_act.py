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

"""ACT on the point-cloud stack: composition, factory resolution, training and inference."""

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class
from lerobot.policies.pc_act.configuration_pc_act import PCACTConfig
from lerobot.policies.pc_act.modeling_pc_act import PCACTPolicy


def _config() -> PCACTConfig:
    cfg = PCACTConfig(device="cpu", chunk_size=16, n_action_steps=8)
    cfg.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "observation.point_cloud": PolicyFeature(type=FeatureType.POINT_CLOUD, shape=(64, 3)),
        "observation.goal_point_cloud": PolicyFeature(type=FeatureType.POINT_CLOUD, shape=(32, 3)),
        "observation.goal_transform": PolicyFeature(type=FeatureType.STATE, shape=(9,)),
    }
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(14,))}
    return cfg


def _batch(b: int = 2) -> dict[str, torch.Tensor]:
    return {
        "observation.state": torch.randn(b, 14),
        "observation.point_cloud": torch.randn(b, 64, 3),
        "observation.goal_point_cloud": torch.randn(b, 32, 3),
        "observation.goal_transform": torch.randn(b, 9),
        "action": torch.randn(b, 16, 14),
        "action_is_pad": torch.zeros(b, 16, dtype=torch.bool),
    }


def test_factory_resolves_pc_act():
    assert get_policy_class("pc_act") is PCACTPolicy


def test_env_token_width_is_encoders_plus_goal_vector():
    cfg = _config()
    assert cfg.env_state_feature.shape[0] == 256 + 256 + 9
    assert cfg.image_features == {}


def test_forward_backward_and_gradients_reach_encoders():
    pol = PCACTPolicy(_config())
    loss, d = pol.forward(_batch())
    assert loss.isfinite() and "l1_loss" in d and "kld_loss" in d
    loss.backward()
    for enc in (pol.pc_encoder, pol.goal_encoder):
        assert any(p.grad is not None and p.grad.any() for p in enc.parameters())


def test_select_action_queues_a_chunk():
    pol = PCACTPolicy(_config())
    obs = {k: v for k, v in _batch().items() if k.startswith("observation")}
    a1 = pol.select_action(obs)
    assert a1.shape == (2, 14)
    # 7 more pops from the queue without re-inference
    for _ in range(7):
        assert pol.select_action(obs).shape == (2, 14)


def test_action_space_metadata_defaults_absolute():
    assert _config().action_space == "absolute_joint"


def test_pose_label_keys_are_stripped_from_inputs():
    from lerobot.configs.types import FeatureType, PolicyFeature

    cfg = _config()
    cfg.input_features["observation.object_pose"] = PolicyFeature(type=FeatureType.STATE, shape=(7,))
    cfg.input_features["observation.pose_valid"] = PolicyFeature(type=FeatureType.STATE, shape=(1,))
    cfg.validate_features()
    assert "observation.object_pose" not in cfg.input_features
    assert "observation.pose_valid" not in cfg.input_features


def test_cross_attention_variant_trains_and_keeps_token_width():
    import torch

    cfg = _config()
    cfg.pc_cross_attention = True
    pol = PCACTPolicy(cfg)
    assert pol.pc_encoder is None and pol.cross_encoder is not None
    assert cfg.env_state_feature.shape[0] == 256 + 256 + 9
    loss, _ = pol.forward(_batch())
    loss.backward()
    assert any(p.grad is not None and p.grad.any() for p in pol.cross_encoder.parameters())
    obs = {k: v for k, v in _batch().items() if k.startswith("observation")}
    assert pol.select_action(obs).shape == (2, 14)
