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

"""Point-cloud conditioned diffusion policy.

Structure mirrors `lerobot.policies.diffusion` -- the queue/chunking logic and the training loss
are the same algorithm -- but the observation encoder is a point-cloud encoder chosen from a
registry, and goal conditioning is a first-class input.

The 1-D conditional U-Net, its residual blocks and the noise-scheduler factory are **imported**
from the image policy rather than copied: they only ever see a flat `(B, global_cond_dim)`
conditioning vector and an action trajectory, so nothing about them is image-specific. A contract
test (`tests/policies/pc_diffusion/test_unet_contract.py`) pins that assumption so an upstream
signature change fails loudly instead of silently at train time.
"""

from collections import deque
from typing import ClassVar

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionConditionalUnet1d,
    _make_noise_scheduler,
)
# The residual-pose head and its rigid-transform helpers are shared with `pcd_diffusion` rather
# than duplicated, so the two policies' auxiliary task is provably the same one.
from lerobot.policies.multi_task_dit.modeling_multi_task_dit import DiffusionTransformer
from lerobot.policies.pcd_diffusion.modeling_pcd_diffusion import (
    ResidualPoseHead,
    matrix_to_rot6d,
    rigid_inverse,
    rot6d_to_matrix,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import require_package

from .configuration_pc_diffusion import PCDiffusionConfig
from .encoders import make_pc_encoder


class PCDiffusionPolicy(PreTrainedPolicy):
    """Diffusion policy over point-cloud observations."""

    config_class = PCDiffusionConfig
    name = "pc_diffusion"

    # FSDP2 wraps at residual-block granularity, matching the image diffusion policy.
    _fsdp_wrap_modules: ClassVar[list[str] | None] = ["DiffusionConditionalResidualBlock1d"]

    def __init__(self, config: PCDiffusionConfig, **kwargs):
        require_package("diffusers", extra="pc_diffusion")
        super().__init__(config)
        config.validate_features()
        self.config = config

        self._queues = None
        self.diffusion = PCDiffusionModel(config)
        self.reset()

    def get_optim_params(self) -> dict:
        return self.diffusion.parameters()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`.

        Every key the model reads must be present here: `populate_queues` only fills keys that
        already exist, so a missing entry silently starves the model of that input at inference.
        """
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            self.config.pc_feature_key: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        for key in self.config.extra_state_keys:
            self._queues[key] = deque(maxlen=self.config.n_obs_steps)
        if self.config.goal_conditioning in ("points", "both"):
            self._queues[self.config.goal_pc_feature_key] = deque(maxlen=self.config.n_obs_steps)
        if self.config.goal_conditioning in ("vector", "both"):
            self._queues[self.config.goal_feature_key] = deque(maxlen=self.config.n_obs_steps)
        if self.config.use_task_onehot:
            self._queues[self.config.task_feature_key] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Predict a chunk of actions.

        Two modes, matching the image policy: online (queues populated by `select_action`) stacks
        the queued observations along a new time axis; offline (dataloader batch) uses the batch
        as-is, since it already carries the `n_obs_steps` axis from `delta_timestamps`.
        """
        queues_populated = any(len(q) > 0 for q in self._queues.values())
        if queues_populated:
            batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        return self.diffusion.generate_actions(batch, noise=noise)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action, caching an observation history and an action chunk."""
        if ACTION in batch:
            batch = {k: v for k, v in batch.items() if k != ACTION}

        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        """Run the batch through the model and compute the training loss."""
        loss, out = self.diffusion.compute_loss(batch)
        return loss, out


def _matrix_from_quat(q: Tensor) -> Tensor:
    """(B, 4) wxyz -> (B, 3, 3). Matches the numpy encoder in `pc_common/pc_ops.py`."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(q.shape[0], 3, 3)


def _se3(pos: Tensor, rot: Tensor) -> Tensor:
    """(B,3) + (B,3,3) -> (B,4,4)."""
    m = torch.eye(4, device=pos.device, dtype=pos.dtype).repeat(pos.shape[0], 1, 1)
    m[:, :3, :3] = rot
    m[:, :3, 3] = pos
    return m


class PCDiffusionModel(nn.Module):
    def __init__(self, config: PCDiffusionConfig):
        super().__init__()
        self.config = config

        # ---- global conditioning width -------------------------------------------------
        global_cond_dim = config.robot_state_feature.shape[0]
        for key in config.extra_state_keys:
            global_cond_dim += config.input_features[key].shape[0]

        pc_ft = config.observation_pc_feature
        num_points, in_channels = int(pc_ft.shape[0]), int(pc_ft.shape[1])
        self.pc_encoder = make_pc_encoder(
            config.pc_encoder,
            num_points=num_points,
            in_channels=in_channels,
            out_dim=config.pc_feature_dim,
            **config.pc_encoder_kwargs,
        )
        global_cond_dim += self.pc_encoder.feature_dim

        self.goal_encoder = None
        if config.goal_conditioning in ("points", "both"):
            goal_ft = config.goal_pc_feature
            self.goal_encoder = make_pc_encoder(
                config.goal_encoder or config.pc_encoder,
                num_points=int(goal_ft.shape[0]),
                in_channels=int(goal_ft.shape[1]),
                out_dim=config.goal_feature_dim,
                **(config.goal_encoder_kwargs or config.pc_encoder_kwargs),
            )
            global_cond_dim += self.goal_encoder.feature_dim
        if config.goal_conditioning in ("vector", "both"):
            global_cond_dim += config.input_features[config.goal_feature_key].shape[0]
        if config.use_task_onehot and config.task_feature_key in config.input_features:
            global_cond_dim += config.input_features[config.task_feature_key].shape[0]

        self.global_cond_dim = global_cond_dim

        # Buffers so isotropic rescaling travels with the checkpoint.
        self.register_buffer("_pc_center", torch.tensor(config.pc_center, dtype=torch.float32))

        # Auxiliary head: reads the same flat conditioning vector the U-Net does, so whatever it
        # needs to solve the residual-pose task must be present in the encoder's output. Training
        # only -- `conditional_sample` never touches it, so inference cost is unchanged.
        self.aux_head = None
        if config.uses_aux_head:
            self.aux_head = ResidualPoseHead(
                in_dim=global_cond_dim * config.n_obs_steps,
                num_objects=config.num_objects,
                hidden_dims=config.aux_head_dims,
                predict_rotation=config.aux_predict_rotation,
            )

        cond_dim = global_cond_dim * config.n_obs_steps
        if config.backbone == "dit":
            # Same contract as the U-Net: (x, timestep, flat conditioning) -> eps-hat.
            self.unet = DiffusionTransformer(config, conditioning_dim=cond_dim)
        else:
            self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=cond_dim)
        if config.compile_model:
            self.unet = torch.compile(self.unet, mode=config.compile_mode)

        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )
        self.num_inference_steps = (
            config.num_inference_steps
            if config.num_inference_steps is not None
            else self.noise_scheduler.config.num_train_timesteps
        )

    # ---- conditioning ------------------------------------------------------------------

    def _encode_cloud(self, encoder, cloud: Tensor, batch_size: int, n_obs_steps: int) -> Tensor:
        """Encode a `(B, S, N, C)` cloud stack to `(B, S, D)`."""
        if self.config.pc_isotropic_rescale:
            cloud = cloud.clone()
            cloud[..., :3] = (cloud[..., :3] - self._pc_center) / self.config.pc_scale
        feats = encoder(einops.rearrange(cloud, "b s n c -> (b s) n c"))
        return einops.rearrange(feats, "(b s) d -> b s d", b=batch_size, s=n_obs_steps)

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Concatenate state, point-cloud and goal features into `(B, global_cond_dim)`.

        This is the only modality-aware part of the model; everything downstream sees a flat
        vector.
        """
        b, s = batch[OBS_STATE].shape[:2]
        feats = [batch[OBS_STATE]]
        feats.extend(batch[key] for key in self.config.extra_state_keys)

        feats.append(self._encode_cloud(self.pc_encoder, batch[self.config.pc_feature_key], b, s))

        if self.config.goal_conditioning in ("points", "both"):
            feats.append(
                self._encode_cloud(self.goal_encoder, batch[self.config.goal_pc_feature_key], b, s)
            )
        if self.config.goal_conditioning in ("vector", "both"):
            feats.append(self._broadcast(batch[self.config.goal_feature_key], b, s))
        if self.config.use_task_onehot and self.config.task_feature_key in batch:
            feats.append(self._broadcast(batch[self.config.task_feature_key], b, s))

        return torch.cat(feats, dim=-1).flatten(start_dim=1)

    def _aux_targets(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Current and goal object poses as (B, K, 4, 4), from whichever schema the batch carries.

        The v2 datasets ship `observation.object_pose` (xyz + wxyz quaternion) per frame and
        `observation.goal_pose` (xyz + rot6d) per episode. The older pose-augmented dataset shipped
        two pre-built (K, 4, 4) matrices; both are accepted so checkpoints and datasets from either
        generation still train.
        """
        cfg = self.config

        def last(x: Tensor) -> Tensor:
            return x[:, -1] if x.ndim == 3 else x

        if cfg.object_poses_key in batch and cfg.goal_object_poses_key in batch:
            cur, goal = batch[cfg.object_poses_key], batch[cfg.goal_object_poses_key]
            return (cur[:, -1] if cur.ndim == 5 else cur, goal[:, -1] if goal.ndim == 5 else goal)

        obj = last(batch[cfg.object_pose_key])       # (B, 7) xyz + quat wxyz
        gp = last(batch[cfg.goal_feature_key])       # (B, 9) xyz + rot6d
        cur = _se3(obj[:, :3], _matrix_from_quat(obj[:, 3:]))
        goal = _se3(gp[:, :3], rot6d_to_matrix(gp[:, 3:]))
        return cur[:, None], goal[:, None]           # K = 1 object

    @staticmethod
    def _broadcast(x: Tensor, b: int, s: int) -> Tensor:
        """Give an episode-constant vector the (B, n_obs_steps, D) shape the concat expects.

        Delta timestamps normally supply the time axis, but a feature can arrive without it --
        at inference the queue holds single steps, and a scalar-shaped feature loses the axis
        entirely -- so normalise here rather than assuming.
        """
        if x.ndim == 1:
            x = x[:, None]
        if x.ndim == 2:
            x = x[:, None, :].expand(b, s, x.shape[-1])
        return x

    # ---- inference ---------------------------------------------------------------------

    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        sample = (
            noise
            if noise is not None
            else torch.randn(
                size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
                dtype=dtype,
                device=device,
                generator=generator,
            )
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            model_output = self.unet(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                global_cond,
            )
            sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample
        return sample

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        if n_obs_steps != self.config.n_obs_steps:
            raise ValueError(f"Expected {self.config.n_obs_steps} observation steps, got {n_obs_steps}.")

        global_cond = self._prepare_global_conditioning(batch)

        # Optionally average K independent samples for the same observation.
        #
        # DDIM with eta=0 is deterministic given the initial noise, so every difference between two
        # samples of one observation comes from `x_T`. The demonstrator here is a deterministic
        # script, so p(action | observation) is unimodal and its conditional *mean* is the
        # minimum-MSE estimate -- there is no multimodality that averaging would blur together.
        # Measured across samplers, roughly 35-40% of position-error variance moves with the
        # sampler, so this targets that component without retraining.
        #
        # For a multimodal task (several valid strategies) this would be actively wrong: the mean
        # of two good trajectories need not be a good trajectory. Keep `num_samples=1` there.
        k = max(1, int(getattr(self.config, "num_samples", 1) or 1))
        if k == 1 or noise is not None:
            actions = self.conditional_sample(batch_size, global_cond=global_cond, noise=noise)
        else:
            cond = global_cond.repeat_interleave(k, dim=0)
            samples = self.conditional_sample(batch_size * k, global_cond=cond)
            actions = samples.reshape(batch_size, k, *samples.shape[1:]).mean(dim=1)

        start = n_obs_steps - 1
        return actions[:, start : start + self.config.n_action_steps]

    # ---- training ----------------------------------------------------------------------

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        missing = {OBS_STATE, ACTION, self.config.pc_feature_key} - set(batch)
        if missing:
            raise ValueError(f"Batch is missing required keys: {sorted(missing)}")

        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon = batch[ACTION].shape[1]
        if horizon != self.config.horizon:
            raise ValueError(f"Expected action horizon {self.config.horizon}, got {horizon}.")
        if n_obs_steps != self.config.n_obs_steps:
            raise ValueError(f"Expected {self.config.n_obs_steps} observation steps, got {n_obs_steps}.")

        global_cond = self._prepare_global_conditioning(batch)

        trajectory = batch[ACTION]
        eps = torch.randn(trajectory.shape, device=trajectory.device, dtype=trajectory.dtype)
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        # Positional: the U-Net names the third argument `global_cond`, the DiT names it
        # `conditioning_vec`. Calling positionally is what keeps the swap a one-word config change.
        pred = self.unet(noisy_trajectory, timesteps, global_cond)
        target = eps if self.config.prediction_type == "epsilon" else trajectory

        loss = F.mse_loss(pred, target, reduction="none")

        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "do_mask_loss_for_padding=True requires 'action_is_pad' in the batch."
                )
            mask = (~batch["action_is_pad"]).unsqueeze(-1)
            num_valid = mask.sum() * loss.shape[-1]
            action_loss = (loss * mask).sum() / num_valid.clamp_min(1)
        else:
            action_loss = loss.mean()

        out: dict[str, float] = {"action_loss": action_loss.item()}
        total = action_loss

        if self.aux_head is not None:
            cfg = self.config
            # Either schema satisfies the head: the v2 datasets ship a per-frame object pose
            # plus the episode's goal pose, the older pose-augmented dataset shipped two 4x4s.
            v2 = cfg.object_pose_key in batch and cfg.goal_feature_key in batch
            legacy = cfg.object_poses_key in batch and cfg.goal_object_poses_key in batch
            if not (v2 or legacy):
                raise ValueError(
                    f"num_objects={cfg.num_objects} enables the residual-pose head, which needs "
                    f"either ('{cfg.object_pose_key}', '{cfg.goal_feature_key}') or "
                    f"('{cfg.object_poses_key}', '{cfg.goal_object_poses_key}') in the batch. "
                    "Port with examples/port_datasets/port_isaaclab_pointcloud_push.py, or set "
                    "--policy.num_objects=0."
                )
            cur, goal_pose = self._aux_targets(batch)
            # Observation keys carry an n_obs_steps window; the residual is defined at the most
            # recent step, which is the one the action chunk starts from.
            if cur.ndim == 5:
                cur = cur[:, -1]
            if goal_pose.ndim == 5:
                goal_pose = goal_pose[:, -1]
            residual = rigid_inverse(cur) @ goal_pose  # (B, K, 4, 4)
            tgt = residual[..., :3, 3]
            if cfg.aux_predict_rotation:
                tgt = torch.cat([tgt, matrix_to_rot6d(residual[..., :3, :3])], dim=-1)
            per = F.mse_loss(self.aux_head(global_cond), tgt, reduction="none").mean(dim=(1, 2))

            # Real-world sources carry no object-pose ground truth, so their auxiliary target is
            # meaningless. Mask those samples rather than dropping the head: the batch still
            # trains the encoder on real clouds through the action loss, which is the whole point
            # of co-training, while the auxiliary term sees only supervised samples.
            valid = batch.get(cfg.pose_valid_key)
            if valid is not None:
                v = valid.reshape(valid.shape[0], -1)[:, -1].to(per.dtype)
                aux_loss = (per * v).sum() / v.sum().clamp_min(1.0)
                out["aux_supervised_frac"] = float(v.mean())
            else:
                aux_loss = per.mean()
            out["aux_residual_loss"] = aux_loss.item()
            total = total + cfg.aux_residual_weight * aux_loss

        return total, out
