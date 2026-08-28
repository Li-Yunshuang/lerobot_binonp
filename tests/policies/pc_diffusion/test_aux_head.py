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

"""The auxiliary residual-pose head on `pc_diffusion`.

The head predicts the remaining current->goal object transform from the same conditioning vector
the U-Net sees. Three properties make or break the ablation it exists to serve, and each is
pinned here:

* it is **off by default**, so every checkpoint trained before it existed still loads and behaves
  identically;
* the auxiliary gradient actually reaches the **point-cloud encoder** -- if it did not, the head
  would be a free-floating regressor and the ablation would measure nothing;
* it is **training-only**, so eval (where no object tracker runs) never needs the pose labels.
"""

import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pc_diffusion.configuration_pc_diffusion import PCDiffusionConfig
from lerobot.policies.pc_diffusion.modeling_pc_diffusion import PCDiffusionPolicy

N_POINTS, N_GOAL, STATE_DIM, ACTION_DIM = 128, 64, 14, 14


def _features():
    return (
        {
            "observation.state": PolicyFeature(FeatureType.STATE, (STATE_DIM,)),
            "observation.point_cloud": PolicyFeature(FeatureType.POINT_CLOUD, (N_POINTS, 3)),
            "observation.goal_point_cloud": PolicyFeature(FeatureType.POINT_CLOUD, (N_GOAL, 3)),
        },
        {"action": PolicyFeature(FeatureType.ACTION, (ACTION_DIM,))},
    )


def _config(**kw):
    inp, out = _features()
    kw.setdefault("down_dims", (64, 128))
    kw.setdefault("pc_encoder_kwargs", {"hidden_dims": (16, 32)})
    return PCDiffusionConfig(input_features=inp, output_features=out, device="cpu", **kw)


def _batch(policy, cur_xy, goal_xy, batch_size=2, with_poses=True):
    cfg = policy.config
    t = cfg.n_obs_steps
    batch = {
        "observation.state": torch.randn(batch_size, t, STATE_DIM),
        "observation.point_cloud": torch.randn(batch_size, t, N_POINTS, 3),
        "observation.goal_point_cloud": torch.randn(batch_size, t, N_GOAL, 3),
        "action": torch.randn(batch_size, cfg.horizon, ACTION_DIM),
    }
    if with_poses:
        for key, (dx, dy) in (
            (cfg.object_poses_key, cur_xy),
            (cfg.goal_object_poses_key, goal_xy),
        ):
            m = torch.eye(4).repeat(batch_size, t, cfg.num_objects or 1, 1, 1)
            m[..., 0, 3], m[..., 1, 3] = dx, dy
            batch[key] = m
    return batch


def test_head_is_off_by_default():
    """A default config must build the pre-existing model exactly, head absent."""
    policy = PCDiffusionPolicy(_config())
    assert policy.config.num_objects == 0
    assert policy.config.uses_aux_head is False
    assert policy.diffusion.aux_head is None

    loss, out = policy.forward(_batch(policy, (0.0, 0.0), (0.1, 0.0), with_poses=False))
    assert "aux_residual_loss" not in out
    assert torch.isfinite(loss)


def test_head_adds_auxiliary_loss():
    policy = PCDiffusionPolicy(_config(num_objects=1, aux_residual_weight=0.1))
    assert policy.diffusion.aux_head is not None

    loss, out = policy.forward(_batch(policy, (0.05, 0.0), (0.17, 0.0)))
    assert "aux_residual_loss" in out and out["aux_residual_loss"] > 0
    # The reported total must be the weighted sum, not the action loss alone.
    assert loss.item() == pytest.approx(
        out["action_loss"] + 0.1 * out["aux_residual_loss"], rel=1e-5
    )


def test_auxiliary_target_is_the_remaining_displacement():
    """cur=(0.05, 0) and goal=(0.17, 0) must give a target of (0.12, 0, 0), not (0.17, 0, 0).

    Regressing the goal's absolute position instead of the *residual* would still train and still
    look plausible in the loss curve, so this asserts the transform composition directly.
    """
    from lerobot.policies.pcd_diffusion.modeling_pcd_diffusion import rigid_inverse

    policy = PCDiffusionPolicy(_config(num_objects=1))
    batch = _batch(policy, (0.05, 0.0), (0.17, 0.0))
    cur = batch[policy.config.object_poses_key][:, -1]
    goal = batch[policy.config.goal_object_poses_key][:, -1]
    residual = rigid_inverse(cur) @ goal
    torch.testing.assert_close(
        residual[..., :3, 3], torch.tensor([0.12, 0.0, 0.0]).expand_as(residual[..., :3, 3])
    )


def test_auxiliary_gradient_reaches_the_point_cloud_encoder():
    """Without this, the head learns in isolation and the ablation measures nothing."""
    policy = PCDiffusionPolicy(_config(num_objects=1, aux_residual_weight=1.0))
    policy.train()
    batch = _batch(policy, (0.05, 0.0), (0.17, 0.0))

    # Isolate the auxiliary path: backprop the aux term only, so any encoder gradient must have
    # come through it rather than through the diffusion loss.
    global_cond = policy.diffusion._prepare_global_conditioning(batch)
    cur = batch[policy.config.object_poses_key][:, -1]
    goal = batch[policy.config.goal_object_poses_key][:, -1]
    from lerobot.policies.pcd_diffusion.modeling_pcd_diffusion import rigid_inverse

    tgt = (rigid_inverse(cur) @ goal)[..., :3, 3]
    torch.nn.functional.mse_loss(policy.diffusion.aux_head(global_cond), tgt).backward()

    grad = sum(
        float(p.grad.norm()) for p in policy.diffusion.pc_encoder.parameters() if p.grad is not None
    )
    assert grad > 0, "auxiliary loss does not reach the point-cloud encoder"


def test_pose_keys_are_labels_not_inputs():
    """They must be stripped from input_features: normalising a rigid transform is meaningless,
    and demanding them at inference would break the eval bridge, where no tracker runs."""
    cfg = _config(num_objects=1)
    assert cfg.object_poses_key not in (cfg.input_features or {})
    assert cfg.goal_object_poses_key not in (cfg.input_features or {})


def test_inference_needs_no_pose_labels():
    policy = PCDiffusionPolicy(_config(num_objects=1, num_inference_steps=2))
    policy.eval()
    policy.reset()
    obs = {
        "observation.state": torch.randn(2, STATE_DIM),
        "observation.point_cloud": torch.randn(2, N_POINTS, 3),
        "observation.goal_point_cloud": torch.randn(2, N_GOAL, 3),
    }
    with torch.no_grad():
        action = policy.select_action(obs)
    assert action.shape == (2, ACTION_DIM)


def test_missing_pose_labels_raise_rather_than_silently_skip():
    """Silently training without the auxiliary term would make the ablation a no-op."""
    policy = PCDiffusionPolicy(_config(num_objects=1))
    with pytest.raises(ValueError, match="residual-pose head"):
        policy.forward(_batch(policy, (0.0, 0.0), (0.1, 0.0), with_poses=False))


# --------------------------------------------------------------------------------------
# Backbone swap and the v2 (sim + real) schema
# --------------------------------------------------------------------------------------


def _v2_features():
    return (
        {
            "observation.state": PolicyFeature(FeatureType.STATE, (STATE_DIM,)),
            "observation.point_cloud": PolicyFeature(FeatureType.POINT_CLOUD, (N_POINTS, 3)),
            "observation.goal_point_cloud": PolicyFeature(FeatureType.POINT_CLOUD, (N_GOAL, 3)),
            "observation.goal_pose": PolicyFeature(FeatureType.STATE, (9,)),
            "observation.task_onehot": PolicyFeature(FeatureType.STATE, (3,)),
            "observation.object_pose": PolicyFeature(FeatureType.STATE, (7,)),
            "observation.pose_valid": PolicyFeature(FeatureType.STATE, (1,)),
        },
        {"action": PolicyFeature(FeatureType.ACTION, (ACTION_DIM,))},
    )


def _v2_config(**kw):
    inp, out = _v2_features()
    kw.setdefault("goal_conditioning", "both")
    kw.setdefault("down_dims", (64, 128))
    kw.setdefault("hidden_dim", 64)
    kw.setdefault("num_layers", 2)
    kw.setdefault("pc_encoder_kwargs", {"hidden_dims": (16, 32)})
    return PCDiffusionConfig(input_features=inp, output_features=out, device="cpu", **kw)


def _v2_batch(policy, batch_size=4, valid=1.0):
    cfg = policy.config
    t = cfg.n_obs_steps
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(batch_size, t, 1)
    rot6d = torch.tensor([1.0, 0, 0, 0, 1, 0]).repeat(batch_size, t, 1)
    v = valid if torch.is_tensor(valid) else torch.full((batch_size, t, 1), float(valid))
    return {
        "observation.state": torch.randn(batch_size, t, STATE_DIM),
        "observation.point_cloud": torch.randn(batch_size, t, N_POINTS, 3),
        "observation.goal_point_cloud": torch.randn(batch_size, t, N_GOAL, 3),
        "observation.goal_pose": torch.cat([torch.randn(batch_size, t, 3), rot6d], dim=-1),
        "observation.task_onehot": torch.tensor([1.0, 0, 0]).repeat(batch_size, t, 1),
        "observation.object_pose": torch.cat([torch.randn(batch_size, t, 3), quat], dim=-1),
        "observation.pose_valid": v,
        "action": torch.randn(batch_size, cfg.horizon, ACTION_DIM),
    }


@pytest.mark.parametrize("backbone", ["unet", "dit"])
def test_both_backbones_train_and_infer(backbone):
    policy = PCDiffusionPolicy(_v2_config(backbone=backbone, num_objects=1, num_inference_steps=2))
    policy.train()
    loss, out = policy.forward(_v2_batch(policy))
    assert torch.isfinite(loss) and "aux_residual_loss" in out
    loss.backward()

    policy.eval()
    policy.reset()
    with torch.no_grad():
        action = policy.select_action(
            {
                "observation.state": torch.randn(2, STATE_DIM),
                "observation.point_cloud": torch.randn(2, N_POINTS, 3),
                "observation.goal_point_cloud": torch.randn(2, N_GOAL, 3),
                "observation.goal_pose": torch.cat(
                    [torch.randn(2, 3), torch.tensor([1.0, 0, 0, 0, 1, 0]).repeat(2, 1)], dim=-1
                ),
                "observation.task_onehot": torch.tensor([1.0, 0, 0]).repeat(2, 1),
            }
        )
    assert action.shape == (2, ACTION_DIM)


def test_backbones_see_an_identical_conditioning_vector():
    """The whole point of the swap: only the denoiser differs.

    If the two backbones were fed different conditioning the comparison would measure the input
    stack as well as the architecture -- the confound that made the earlier two-policy comparison
    uninterpretable.
    """
    unet = PCDiffusionPolicy(_v2_config(backbone="unet", num_objects=1))
    dit = PCDiffusionPolicy(_v2_config(backbone="dit", num_objects=1))
    assert unet.diffusion.global_cond_dim == dit.diffusion.global_cond_dim


def test_auxiliary_loss_is_masked_for_sources_without_pose_truth():
    """Real-world data has no object pose. Those samples must not contribute an auxiliary target,
    and a batch with none at all must not produce NaN."""
    policy = PCDiffusionPolicy(_v2_config(backbone="unet", num_objects=1))
    policy.train()

    valid = torch.ones(4, policy.config.n_obs_steps, 1)
    valid[2:] = 0.0
    _, out = policy.forward(_v2_batch(policy, valid=valid))
    assert out["aux_supervised_frac"] == pytest.approx(0.5)

    _, none = policy.forward(_v2_batch(policy, valid=0.0))
    assert none["aux_supervised_frac"] == pytest.approx(0.0)
    assert none["aux_residual_loss"] == none["aux_residual_loss"], "NaN with no supervised sample"


def test_goal_pose_and_task_onehot_reach_the_conditioning_vector():
    """A wider conditioning vector is the observable consequence of adding these inputs."""
    base = PCDiffusionPolicy(_v2_config(goal_conditioning="points", use_task_onehot=False))
    with_goal = PCDiffusionPolicy(_v2_config(goal_conditioning="both", use_task_onehot=False))
    with_task = PCDiffusionPolicy(_v2_config(goal_conditioning="both", use_task_onehot=True))
    assert with_goal.diffusion.global_cond_dim == base.diffusion.global_cond_dim + 9
    assert with_task.diffusion.global_cond_dim == with_goal.diffusion.global_cond_dim + 3
