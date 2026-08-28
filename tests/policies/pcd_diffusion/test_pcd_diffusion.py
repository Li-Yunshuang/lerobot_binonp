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

"""Unit tests for the `pcd_diffusion` policy."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.pcd_diffusion.configuration_pcd_diffusion import PcdDiffusionConfig  # noqa: E402
from lerobot.policies.pcd_diffusion.modeling_pcd_diffusion import (  # noqa: E402
    PcdDiffusionPolicy,
    matrix_to_rot6d,
    rigid_inverse,
    rot6d_to_matrix,
)

N_PTS, M_PTS, STATE_DIM, ACTION_DIM, K = 96, 64, 14, 14, 2
HORIZON, N_OBS, N_ACT = 8, 2, 4


def make_config(**kw) -> PcdDiffusionConfig:
    kw.setdefault("n_points", 32)
    cfg = PcdDiffusionConfig(
        n_obs_steps=N_OBS, horizon=HORIZON, n_action_steps=N_ACT,
        perceiver_num_latents=2, perceiver_dim=32, perceiver_depth=1,
        perceiver_heads=2, state_mlp_dims=(32,), hidden_dim=32, num_layers=2, num_heads=2,
        fourier_bands=2, num_inference_steps=2, device="cpu", push_to_hub=False, **kw,
    )
    cfg.input_features = {
        "observation.state": PolicyFeature(FeatureType.STATE, (STATE_DIM,)),
        "observation.point_cloud": PolicyFeature(FeatureType.POINT_CLOUD, (N_PTS, 3)),
        "observation.goal_point_cloud": PolicyFeature(FeatureType.POINT_CLOUD, (M_PTS, 3)),
        "observation.object_poses": PolicyFeature(FeatureType.STATE, (K, 4, 4)),
        "observation.goal_object_poses": PolicyFeature(FeatureType.STATE, (K, 4, 4)),
    }
    cfg.output_features = {"action": PolicyFeature(FeatureType.ACTION, (ACTION_DIM,))}
    return cfg


def make_batch(b=2, with_poses=True) -> dict:
    batch = {
        "observation.state": torch.randn(b, N_OBS, STATE_DIM),
        "observation.point_cloud": torch.randn(b, N_OBS, N_PTS, 3) * 0.2,
        "observation.goal_point_cloud": torch.randn(b, N_OBS, M_PTS, 3) * 0.2,
        "action": torch.randn(b, HORIZON, ACTION_DIM),
        "action_is_pad": torch.zeros(b, HORIZON, dtype=torch.bool),
    }
    if with_poses:
        eye = torch.eye(4).expand(b, N_OBS, K, 4, 4).clone()
        batch["observation.object_poses"] = eye
        batch["observation.goal_object_poses"] = eye.clone()
    return batch


# ---- rotation helpers ------------------------------------------------------------------


def test_rot6d_round_trip():
    q = torch.linalg.qr(torch.randn(5, 3, 3))[0]
    r = q * torch.det(q).sign().reshape(-1, 1, 1)  # ensure proper rotations
    assert torch.allclose(rot6d_to_matrix(matrix_to_rot6d(r)), r, atol=1e-5)


def test_rigid_inverse():
    t = torch.eye(4).repeat(4, 1, 1)
    q = torch.linalg.qr(torch.randn(4, 3, 3))[0]
    t[:, :3, :3] = q * torch.det(q).sign().reshape(-1, 1, 1)
    t[:, :3, 3] = torch.randn(4, 3)
    assert torch.allclose(rigid_inverse(t) @ t, torch.eye(4).expand(4, 4, 4), atol=1e-5)


# ---- policy ----------------------------------------------------------------------------


def test_pose_keys_stripped_from_inputs():
    """Pose keys are training labels; they must not become inference inputs."""
    policy = PcdDiffusionPolicy(make_config(num_objects=K))
    assert "observation.object_poses" not in policy.config.input_features
    assert "observation.goal_object_poses" not in policy.config.input_features
    assert "observation.point_cloud" in policy.config.input_features


def test_forward_and_backward():
    policy = PcdDiffusionPolicy(make_config(num_objects=K))
    loss, out = policy.forward(make_batch())
    assert torch.isfinite(loss) and loss.ndim == 0
    assert "action_loss" in out and "aux_residual_loss" in out
    loss.backward()
    grads = [p.grad for p in policy.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "no gradients populated"
    assert all(torch.isfinite(g).all() for g in grads)


def test_aux_head_disabled_without_objects():
    policy = PcdDiffusionPolicy(make_config(num_objects=0))
    loss, out = policy.forward(make_batch(with_poses=False))
    assert torch.isfinite(loss)
    assert "aux_residual_loss" not in out


def test_aux_missing_pose_keys_raises_clearly():
    policy = PcdDiffusionPolicy(make_config(num_objects=K))
    with pytest.raises(ValueError, match="object_poses"):
        policy.forward(make_batch(with_poses=False))


def test_predict_action_chunk_shape():
    policy = PcdDiffusionPolicy(make_config())
    actions = policy.predict_action_chunk(make_batch(b=3))
    assert actions.shape == (3, N_ACT, ACTION_DIM)


def test_select_action_replans_every_n_action_steps():
    policy = PcdDiffusionPolicy(make_config())
    policy.reset()
    obs = {k: v[:, -1] for k, v in make_batch(b=1).items() if k.startswith("observation.")}
    calls = []
    original = policy.model.generate_actions

    def counting(*a, **kw):
        calls.append(1)
        return original(*a, **kw)

    policy.model.generate_actions = counting
    for _ in range(2 * N_ACT):
        a = policy.select_action({k: v.clone() for k, v in obs.items()})
        assert a.shape == (1, ACTION_DIM)
    assert len(calls) == 2, f"expected 2 re-plans over {2 * N_ACT} calls, got {len(calls)}"


def test_resampling_both_directions():
    """n_points above and below the dataset's N must both work."""
    for n_points in (N_PTS // 2, N_PTS * 2):
        policy = PcdDiffusionPolicy(make_config(n_points=n_points))
        loss, _ = policy.forward(make_batch())
        assert torch.isfinite(loss)


def test_save_load_round_trip(tmp_path):
    policy = PcdDiffusionPolicy(make_config())
    policy.eval()
    batch = make_batch(b=1)
    noise = torch.randn(1, HORIZON, ACTION_DIM)
    before = policy.predict_action_chunk(batch, noise=noise)

    policy.save_pretrained(tmp_path)
    policy.config._save_pretrained(tmp_path)
    reloaded = PcdDiffusionPolicy.from_pretrained(tmp_path, config=policy.config)
    reloaded.eval()
    after = reloaded.predict_action_chunk(batch, noise=noise)
    assert torch.allclose(before, after, atol=1e-5)


def test_shared_frame_normalization():
    """Translating obs and goal together must leave the model's view unchanged."""
    policy = PcdDiffusionPolicy(make_config())
    batch = make_batch(b=2)
    ctx_a, _ = policy.model._encode(batch)
    shift = torch.tensor([0.5, -0.3, 0.2])
    shifted = dict(batch)
    shifted["observation.point_cloud"] = batch["observation.point_cloud"] + shift
    shifted["observation.goal_point_cloud"] = batch["observation.goal_point_cloud"] + shift
    torch.manual_seed(0)
    ctx_a, _ = policy.model._encode(batch)
    torch.manual_seed(0)
    ctx_b, _ = policy.model._encode(shifted)
    assert torch.allclose(ctx_a, ctx_b, atol=1e-4), "encoder is not translation-invariant"
