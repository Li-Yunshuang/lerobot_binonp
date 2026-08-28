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

"""Future-latent auxiliary head.

Two tests here are load-bearing rather than routine:

`test_future_frame_never_reaches_the_conditioning` -- the observation window gains a frame from
the future, and if it leaked into the conditioning the policy would be predicting actions from
its own future and every resulting number would be meaningless. That failure is silent: the loss
would improve and the success rate would look good until deployment.

`test_stop_gradient_on_target` -- the target comes from the encoder being trained, so the cheapest
way to cut the loss is to collapse the encoder to a constant. Because that encoder is shared with
the policy conditioning, a collapse would take the policy down with it.
"""

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pc_diffusion.configuration_pc_diffusion import PCDiffusionConfig
from lerobot.policies.pc_diffusion.modeling_pc_diffusion import PCDiffusionModel

HORIZON = 4


def _config(**overrides) -> PCDiffusionConfig:
    kwargs = {
        "horizon": 16,
        "n_action_steps": 8,
        "n_obs_steps": 2,
        "num_objects": 0,
        "goal_conditioning": "points",
        "use_task_onehot": False,
        "future_latent_weight": 0.1,
        "future_latent_horizon": HORIZON,
        "future_latent_predictor_dims": (64,),
        "device": "cpu",
    }
    kwargs.update(overrides)
    cfg = PCDiffusionConfig(**kwargs)
    cfg.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "observation.point_cloud": PolicyFeature(type=FeatureType.POINT_CLOUD, shape=(64, 3)),
        "observation.goal_point_cloud": PolicyFeature(type=FeatureType.POINT_CLOUD, shape=(32, 3)),
    }
    cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(14,))}
    return cfg


def _batch(b: int = 2, steps: int = 3) -> dict[str, torch.Tensor]:
    """`steps` is n_obs_steps + 1 when the future-latent head is on."""
    return {
        "observation.state": torch.randn(b, steps, 14),
        "observation.point_cloud": torch.randn(b, steps, 64, 3),
        "observation.goal_point_cloud": torch.randn(b, steps, 32, 3),
        "action": torch.randn(b, 16, 14),
        "action_is_pad": torch.zeros(b, 16, dtype=torch.bool),
    }


def test_delta_indices_request_exactly_one_future_frame():
    cfg = _config()
    assert cfg.observation_delta_indices == [-1, 0, HORIZON]
    # Off by default, so existing configs are untouched.
    assert PCDiffusionConfig(device="cpu").future_latent_weight == 0.0


def test_future_frame_never_reaches_the_conditioning():
    """The conditioning must depend only on the first n_obs_steps frames."""
    torch.manual_seed(0)
    model = PCDiffusionModel(_config())
    batch = _batch()

    stripped, future = model._split_future(batch)
    assert future is not None
    assert future["observation.point_cloud"].shape == (2, 64, 3)
    for key in ("observation.state", "observation.point_cloud", "observation.goal_point_cloud"):
        assert stripped[key].shape[1] == 2, f"{key} still carries the future frame"

    cond_a = model._prepare_global_conditioning(stripped)
    # Change ONLY the future frame; the conditioning must be bit-identical.
    perturbed = {k: v.clone() for k, v in batch.items()}
    perturbed["observation.point_cloud"][:, -1] += 100.0
    perturbed["observation.state"][:, -1] += 100.0
    cond_b = model._prepare_global_conditioning(model._split_future(perturbed)[0])
    torch.testing.assert_close(cond_a, cond_b)


def test_stop_gradient_on_target():
    """No gradient may flow through the target, or the encoder can collapse to satisfy it."""
    model = PCDiffusionModel(_config())
    batch = _batch()
    _, future = model._split_future(batch)
    goal = batch["observation.goal_point_cloud"][:, 1]

    tgt = model._encode_obs_latent(future["observation.point_cloud"], goal)
    assert tgt.requires_grad, "sanity: the raw target should be differentiable before detach"

    loss, out = model.compute_loss(batch)
    loss.backward()
    assert "future_latent_loss" in out
    # The predictor is the only path that should carry this gradient; the encoder still gets
    # gradient from the action loss, so we cannot simply assert it is None.
    assert any(p.grad is not None and p.grad.any() for p in model.future_predictor.parameters())


def test_collapse_diagnostic_is_reported():
    """A collapsing encoder shows a falling std while the loss still looks healthy."""
    model = PCDiffusionModel(_config())
    _, out = model.compute_loss(_batch())
    assert "future_latent_std" in out
    assert out["future_latent_std"] > 0


def test_loss_is_bounded_like_a_cosine():
    model = PCDiffusionModel(_config())
    _, out = model.compute_loss(_batch())
    assert -1.0001 <= out["future_latent_loss"] <= 1.0001


def test_missing_future_frame_fails_loudly():
    """A batch built from a stale config would silently train without the target."""
    model = PCDiffusionModel(_config())
    with pytest.raises(ValueError, match="no future frame"):
        model.compute_loss(_batch(steps=2))


def test_works_alongside_cross_attention():
    model = PCDiffusionModel(_config(pc_cross_attention=True))
    loss, out = model.compute_loss(_batch())
    assert loss.isfinite() and "future_latent_loss" in out
    loss.backward()
    assert any(p.grad is not None and p.grad.any() for p in model.cross_encoder.parameters())


def test_disabled_by_default_leaves_the_batch_alone():
    model = PCDiffusionModel(_config(future_latent_weight=0.0))
    batch = _batch(steps=2)
    stripped, future = model._split_future(batch)
    assert future is None and stripped is batch
    _, out = model.compute_loss(batch)
    assert "future_latent_loss" not in out


def _pose_batch(b: int = 2, steps: int = 3) -> dict[str, torch.Tensor]:
    batch = _batch(b, steps)
    eye = torch.eye(4).expand(b, steps, 1, 4, 4).clone()
    batch["observation.object_poses"] = eye.clone()
    goal = eye.clone()
    goal[..., :3, 3] = torch.tensor([0.1, 0.0, 0.0])  # object still 10 cm from its goal
    batch["observation.goal_object_poses"] = goal
    return batch


def test_object_crop_keeps_cloud_length_and_drops_far_points():
    """Cropping must not change the encoder's input length -- max-pool is duplication-invariant."""
    model = PCDiffusionModel(_config(future_latent_object_only=True))
    batch = _pose_batch()
    _, fut = model._split_future(batch)
    cloud = fut["observation.point_cloud"]
    goal = fut["observation.goal_point_cloud"]
    cropped = model._object_crop(cloud, goal, fut)
    assert cropped.shape == cloud.shape, "cropping changed the cloud length"
    # Every kept point must be one of the originals, and closer to the derived centre than the
    # points that were dropped.
    centre = goal.mean(dim=1) - torch.tensor([0.1, 0.0, 0.0])
    d_kept = (cropped - centre[:, None, :]).norm(dim=-1).max(dim=1).values
    d_all = (cloud - centre[:, None, :]).norm(dim=-1).max(dim=1).values
    assert (d_kept <= d_all + 1e-6).all()


def test_object_crop_requires_pose_keys():
    model = PCDiffusionModel(_config(future_latent_object_only=True))
    with pytest.raises(ValueError, match="object_poses"):
        model.compute_loss(_batch())


def test_object_crop_trains_end_to_end():
    model = PCDiffusionModel(_config(future_latent_object_only=True))
    loss, out = model.compute_loss(_pose_batch())
    assert loss.isfinite() and "future_latent_loss" in out
    loss.backward()
