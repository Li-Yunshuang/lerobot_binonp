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

"""Goal-conditioned point-cloud diffusion policy: Perceiver encoder + DiT action head.

Architecture (see `configuration_pcd_diffusion.py` for the rationale and the paper it follows)::

    observation.point_cloud (B,T,N,3) ─┐
                                       ├─ shared PerceiverPointEncoder ─→ obs latents  (B,T,L,D)
    observation.goal_point_cloud (B,M,3)┘   (+ learned obs/goal type emb)  goal latents (B,L,D)

    observation.state (B,T,S) ─ MLP ─→ proprio token (B,1,D)

      context = [obs latents | goal latents | proprio] ─ Linear(D→H) ─┐
                                                                      ├─ DiT self-attention
                                     noisy action tokens (B,horizon,H)┘   AdaLN-Zero(timestep)
                                                                      └─→ eps-hat on the
                                                                          action-token slice
"""

import math
from collections import deque
from typing import ClassVar

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.diffusion.modeling_diffusion import _make_noise_scheduler
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import require_package

from .configuration_pcd_diffusion import PcdDiffusionConfig

# --------------------------------------------------------------------------------------
# Rotation helpers (torch, on-device). `lerobot/utils/rotation.py` is numpy-only.
# --------------------------------------------------------------------------------------


def matrix_to_rot6d(matrix: Tensor) -> Tensor:
    """(..., 3, 3) -> (..., 6): the first two columns, per Zhou et al. 2019.

    A 6-D parameterisation avoids the double cover of quaternions and the discontinuities of
    Euler angles, both of which make a regression target ill-posed.

    The transpose matters: reshaping the (..., 3, 2) column slice directly would interleave the
    two columns row-major, and `rot6d_to_matrix` expects them contiguous.
    """
    return matrix[..., :, :2].transpose(-1, -2).reshape(*matrix.shape[:-2], 6)


def rot6d_to_matrix(rot6d: Tensor) -> Tensor:
    """(..., 6) -> (..., 3, 3) by Gram-Schmidt on the two 3-vectors."""
    a1, a2 = rot6d[..., :3], rot6d[..., 3:]
    b1 = F.normalize(a1, dim=-1, eps=1e-8)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def rigid_inverse(transform: Tensor) -> Tensor:
    """Inverse of a (..., 4, 4) rigid transform, without a general matrix inverse."""
    rot = transform[..., :3, :3]
    trans = transform[..., :3, 3]
    rot_t = rot.transpose(-1, -2)
    out = torch.zeros_like(transform)
    out[..., :3, :3] = rot_t
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", rot_t, trans)
    out[..., 3, 3] = 1.0
    return out


# --------------------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------------------


class FourierFeatures(nn.Module):
    """Lift XYZ to sin/cos bands at geometrically spaced frequencies, concatenated with raw XYZ.

    Raw coordinates alone are a poor input to an MLP for fine spatial discrimination; the bands
    give the network high-frequency spatial detail without increasing point count.
    """

    def __init__(self, num_bands: int = 8, in_dim: int = 3) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.in_dim = in_dim
        self.register_buffer("freqs", 2.0 ** torch.arange(num_bands, dtype=torch.float32) * math.pi)

    @property
    def out_dim(self) -> int:
        return self.in_dim * (1 + 2 * self.num_bands)

    def forward(self, xyz: Tensor) -> Tensor:
        scaled = xyz.unsqueeze(-1) * self.freqs  # (..., 3, bands)
        feats = torch.cat([scaled.sin(), scaled.cos()], dim=-1)  # (..., 3, 2*bands)
        return torch.cat([xyz, feats.flatten(start_dim=-2)], dim=-1)


class PerceiverPointEncoder(nn.Module):
    """Perceiver-style encoder: a small set of learned latents cross-attend to the points.

    Cost is linear in point count (the quadratic attention is over `num_latents` only), which is
    what makes attending over hundreds of points affordable. The same weights encode the
    observation and goal clouds; a learned type embedding added to the latents tells them apart.
    """

    def __init__(
        self,
        *,
        num_latents: int = 4,
        dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        fourier_bands: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fourier = FourierFeatures(fourier_bands)
        self.point_proj = nn.Sequential(
            nn.Linear(self.fourier.out_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)
        self.type_emb = nn.Parameter(torch.zeros(2, dim))  # 0 = observation, 1 = goal

        self.self_attn = nn.ModuleList(
            [nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout) for _ in range(depth)]
        )
        self.cross_attn = nn.ModuleList(
            [nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout) for _ in range(depth)]
        )
        self.norm_self = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.norm_cross_q = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.norm_cross_kv = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.norm_ffn = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, dim * 4), nn.GELU(approximate="tanh"), nn.Linear(dim * 4, dim)
                )
                for _ in range(depth)
            ]
        )
        self.out_norm = nn.LayerNorm(dim)
        self.dim = dim
        self.num_latents = num_latents

    def forward(self, points: Tensor, *, is_goal: bool = False) -> Tensor:
        """(B, N, 3) -> (B, num_latents, dim)."""
        b = points.shape[0]
        kv = self.point_proj(self.fourier(points))
        latents = self.latents.unsqueeze(0).expand(b, -1, -1) + self.type_emb[int(is_goal)]

        for i in range(len(self.self_attn)):
            q = self.norm_self[i](latents)
            latents = latents + self.self_attn[i](q, q, q, need_weights=False)[0]
            q = self.norm_cross_q[i](latents)
            k = self.norm_cross_kv[i](kv)
            latents = latents + self.cross_attn[i](q, k, k, need_weights=False)[0]
            latents = latents + self.ffn[i](self.norm_ffn[i](latents))

        return self.out_norm(latents)


class SinusoidalPosEmb(nn.Module):
    """Standard sinusoidal embedding of the diffusion timestep."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class DiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero conditioning, as in DiT.

    The zero-initialised gates make each block start as the identity, so training begins from a
    well-behaved function and the conditioning is learned rather than fought.
    """

    def __init__(self, hidden: int, heads: int, cond_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden * 4), nn.GELU(approximate="tanh"), nn.Linear(hidden * 4, hidden)
        )
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * hidden, bias=True))
        nn.init.zeros_(self.ada_ln[-1].weight)
        nn.init.zeros_(self.ada_ln[-1].bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.ada_ln(cond).chunk(6, dim=-1)
        h = modulate(self.norm1(x), shift_a.unsqueeze(1), scale_a.unsqueeze(1))
        x = x + gate_a.unsqueeze(1) * self.attn(h, h, h, need_weights=False)[0]
        h = modulate(self.norm2(x), shift_m.unsqueeze(1), scale_m.unsqueeze(1))
        return x + gate_m.unsqueeze(1) * self.mlp(h)


class DiffusionActionHead(nn.Module):
    """DiT over [context tokens ; noisy action tokens]; predicts on the action slice only.

    Keeping the context as *tokens* rather than pooling it into one vector follows the paper's
    "encoder tokens concatenated with the proprioception token", and lets each action step attend
    to whichever latent is relevant.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        horizon: int,
        hidden: int,
        num_layers: int,
        num_heads: int,
        context_dim: int,
        max_context_tokens: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.action_in = nn.Linear(action_dim, hidden)
        self.action_pos = nn.Parameter(torch.randn(horizon, hidden) * 0.02)
        self.context_proj = nn.Linear(context_dim, hidden)
        self.context_type = nn.Parameter(torch.randn(max_context_tokens, hidden) * 0.02)

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(hidden), nn.Linear(hidden, hidden * 4), nn.SiLU(), nn.Linear(hidden * 4, hidden)
        )
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden, num_heads, hidden, dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden, bias=True))
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)
        self.action_out = nn.Linear(hidden, action_dim)
        nn.init.zeros_(self.action_out.weight)
        nn.init.zeros_(self.action_out.bias)

    def forward(self, noisy_actions: Tensor, timesteps: Tensor, context: Tensor) -> Tensor:
        """(B, horizon, A), (B,), (B, C, context_dim) -> (B, horizon, A)."""
        b, horizon, _ = noisy_actions.shape
        cond = self.time_emb(timesteps.float())

        ctx = self.context_proj(context) + self.context_type[: context.shape[1]]
        act = self.action_in(noisy_actions) + self.action_pos[:horizon]
        x = torch.cat([ctx, act], dim=1)

        for block in self.blocks:
            x = block(x, cond)

        x = x[:, ctx.shape[1] :]  # action-token slice
        shift, scale = self.final_ada(cond).chunk(2, dim=-1)
        x = modulate(self.final_norm(x), shift.unsqueeze(1), scale.unsqueeze(1))
        return self.action_out(x)


class ResidualPoseHead(nn.Module):
    """Predicts the remaining current->goal rigid transform per object, as 3-D t + 6-D rotation.

    This is the auxiliary supervision: it forces the latents to actually localise objects rather
    than shortcut off proprioception, and doubles as a readable progress signal at eval time
    (progress corresponds to a shrinking residual).
    """

    def __init__(
        self, *, in_dim: int, num_objects: int, hidden_dims: tuple[int, ...], predict_rotation: bool = True
    ) -> None:
        super().__init__()
        self.out_per_object = 9 if predict_rotation else 3
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU(approximate="tanh")]
            prev = h
        layers.append(nn.Linear(prev, num_objects * self.out_per_object))
        self.net = nn.Sequential(*layers)
        self.num_objects = num_objects

    def forward(self, feats: Tensor) -> Tensor:
        """(B, in_dim) -> (B, K, 9) with rotation, else (B, K, 3)."""
        return self.net(feats).reshape(feats.shape[0], self.num_objects, self.out_per_object)


# --------------------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------------------


class PcdDiffusionPolicy(PreTrainedPolicy):
    """Goal-conditioned point-cloud diffusion policy (Perceiver + DiT)."""

    config_class = PcdDiffusionConfig
    name = "pcd_diffusion"

    _fsdp_wrap_modules: ClassVar[list[str] | None] = ["DiTBlock"]

    def __init__(self, config: PcdDiffusionConfig, **kwargs):
        require_package("diffusers", extra="pcd_diffusion")
        super().__init__(config)
        config.validate_features()
        self.config = config
        self._queues = None
        self.model = PcdDiffusionModel(config)
        self.reset()

    def get_optim_params(self) -> dict:
        return self.model.parameters()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`."""
        self._queues = {
            self.config.state_key: deque(maxlen=self.config.n_obs_steps),
            self.config.pointcloud_key: deque(maxlen=self.config.n_obs_steps),
            self.config.goal_pointcloud_key: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        queues_populated = any(len(q) > 0 for q in self._queues.values())
        if queues_populated:
            batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        return self.model.generate_actions(batch, noise=noise)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        if ACTION in batch:
            batch = {k: v for k, v in batch.items() if k != ACTION}
        self._queues = populate_queues(self._queues, batch)
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))
        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        return self.model.compute_loss(batch)


class PcdDiffusionModel(nn.Module):
    def __init__(self, config: PcdDiffusionConfig):
        super().__init__()
        self.config = config

        self.encoder = PerceiverPointEncoder(
            num_latents=config.perceiver_num_latents,
            dim=config.perceiver_dim,
            depth=config.perceiver_depth,
            heads=config.perceiver_heads,
            fourier_bands=config.fourier_bands,
            dropout=config.dropout,
        )

        state_dim = config.robot_state_feature.shape[0]
        layers: list[nn.Module] = []
        prev = state_dim * config.n_obs_steps
        for h in config.state_mlp_dims:
            layers += [nn.Linear(prev, h), nn.GELU(approximate="tanh")]
            prev = h
        layers.append(nn.Linear(prev, config.perceiver_dim))
        self.state_mlp = nn.Sequential(*layers)

        # context = obs latents (T x L) + goal latents (L) + 1 proprio token
        n_context = config.n_obs_steps * config.perceiver_num_latents + config.perceiver_num_latents + 1
        self.head = DiffusionActionHead(
            action_dim=config.action_feature.shape[0],
            horizon=config.horizon,
            hidden=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            context_dim=config.perceiver_dim,
            max_context_tokens=n_context,
            dropout=config.dropout,
        )

        self.aux_head = None
        if config.num_objects > 0:
            self.aux_head = ResidualPoseHead(
                in_dim=config.perceiver_dim,
                num_objects=config.num_objects,
                hidden_dims=config.aux_head_dims,
                predict_rotation=config.aux_predict_rotation,
            )

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

    # ---- point-cloud preprocessing (in-model) ------------------------------------------

    def _resample(self, points: Tensor) -> Tensor:
        """(B, N, 3) -> (B, n_points, 3)."""
        b, n, _ = points.shape
        k = self.config.n_points
        if n == k:
            return points
        if self.config.resample_mode == "fps":
            idx = _farthest_point_sample(points, k)
        elif n > k:
            idx = torch.argsort(torch.rand(b, n, device=points.device), dim=1)[:, :k]
        else:  # fewer points than requested: sample with replacement
            idx = torch.randint(0, n, (b, k), device=points.device)
        return torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))

    def _normalize_clouds(self, obs_pc: Tensor, goal_pc: Tensor) -> tuple[Tensor, Tensor]:
        """Centre both clouds on the *current* observation centroid, then scale.

        Both clouds must share one frame. Centring the goal on its own centroid would express it
        in a different origin from the observation and destroy the very displacement the policy
        needs to read. The centroid of the most recent observation step is used for the whole
        sample so the frame is also consistent across the observation history.

        Args:
            obs_pc: (B, T, N, 3)
            goal_pc: (B, M, 3)
        """
        centroid = obs_pc[:, -1].mean(dim=1)  # (B, 3)
        obs = obs_pc - centroid[:, None, None, :]
        goal = goal_pc - centroid[:, None, :]

        if self.config.pointcloud_scale is not None:
            scale = obs.new_tensor(self.config.pointcloud_scale).reshape(1)
        else:
            # Per-sample scale: the current cloud's max radius about its own centroid.
            scale = obs[:, -1].norm(dim=-1).amax(dim=1).clamp_min(1e-6)  # (B,)
        return obs / scale[:, None, None, None], goal / scale[:, None, None]

    def _encode(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Returns (context tokens (B, C, D), pooled cloud latents (B, D))."""
        cfg = self.config
        obs_pc = batch[cfg.pointcloud_key]  # (B, T, N, 3)
        goal_pc = batch[cfg.goal_pointcloud_key]
        if goal_pc.ndim == 4:  # static within an episode -> take the current step
            goal_pc = goal_pc[:, -1]

        obs_pc, goal_pc = self._normalize_clouds(obs_pc, goal_pc)
        b, t = obs_pc.shape[:2]

        obs_flat = self._resample(einops.rearrange(obs_pc, "b t n c -> (b t) n c"))
        obs_lat = self.encoder(obs_flat, is_goal=False)  # (B*T, L, D)
        obs_lat = einops.rearrange(obs_lat, "(b t) l d -> b (t l) d", b=b, t=t)
        goal_lat = self.encoder(self._resample(goal_pc), is_goal=True)  # (B, L, D)

        state = batch[cfg.state_key].flatten(start_dim=1)  # (B, T*S)
        proprio = self.state_mlp(state).unsqueeze(1)  # (B, 1, D)

        context = torch.cat([obs_lat, goal_lat, proprio], dim=1)
        pooled = torch.cat([obs_lat, goal_lat], dim=1).mean(dim=1)
        return context, pooled

    # ---- inference ---------------------------------------------------------------------

    def conditional_sample(
        self, batch_size: int, context: Tensor, generator=None, noise: Tensor | None = None
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)
        sample = (
            noise
            if noise is not None
            else torch.randn(
                (batch_size, self.config.horizon, self.config.action_feature.shape[0]),
                dtype=dtype,
                device=device,
                generator=generator,
            )
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            model_out = self.head(
                sample, torch.full(sample.shape[:1], t, dtype=torch.long, device=device), context
            )
            sample = self.noise_scheduler.step(model_out, t, sample, generator=generator).prev_sample
        return sample

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        b, t = batch[self.config.state_key].shape[:2]
        if t != self.config.n_obs_steps:
            raise ValueError(f"Expected {self.config.n_obs_steps} observation steps, got {t}.")
        context, _ = self._encode(batch)
        actions = self.conditional_sample(b, context, noise=noise)
        start = t - 1
        return actions[:, start : start + self.config.n_action_steps]

    # ---- training ----------------------------------------------------------------------

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        cfg = self.config
        missing = {cfg.state_key, ACTION, cfg.pointcloud_key, cfg.goal_pointcloud_key} - set(batch)
        if missing:
            raise ValueError(f"Batch is missing required keys: {sorted(missing)}")

        trajectory = batch[ACTION]
        if trajectory.shape[1] != cfg.horizon:
            raise ValueError(f"Expected action horizon {cfg.horizon}, got {trajectory.shape[1]}.")

        context, pooled = self._encode(batch)

        eps = torch.randn(trajectory.shape, device=trajectory.device, dtype=trajectory.dtype)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy = self.noise_scheduler.add_noise(trajectory, eps, timesteps)
        pred = self.head(noisy, timesteps, context)
        target = eps if cfg.prediction_type == "epsilon" else trajectory

        action_loss = F.mse_loss(pred, target, reduction="none")
        if cfg.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError("do_mask_loss_for_padding=True requires 'action_is_pad' in the batch.")
            mask = (~batch["action_is_pad"]).unsqueeze(-1)
            action_loss = (action_loss * mask).sum() / (mask.sum() * action_loss.shape[-1]).clamp_min(1)
        else:
            action_loss = action_loss.mean()

        out = {"action_loss": action_loss.item()}
        loss = action_loss

        if self.aux_head is not None and cfg.aux_residual_weight > 0:
            for key in (cfg.object_poses_key, cfg.goal_object_poses_key):
                if key not in batch:
                    raise ValueError(
                        f"aux_residual_weight={cfg.aux_residual_weight} and num_objects="
                        f"{cfg.num_objects} require '{key}' of shape (B, T, {cfg.num_objects}, 4, 4) "
                        "in the batch. Set num_objects=0 for datasets without object poses."
                    )
            cur = batch[cfg.object_poses_key]
            goal = batch[cfg.goal_object_poses_key]
            if cur.ndim == 5:
                cur = cur[:, -1]
            if goal.ndim == 5:
                goal = goal[:, -1]
            residual = rigid_inverse(cur) @ goal  # (B, K, 4, 4)
            tgt = residual[..., :3, 3]
            if cfg.aux_predict_rotation:
                tgt = torch.cat([tgt, matrix_to_rot6d(residual[..., :3, :3])], dim=-1)
            aux_loss = F.mse_loss(self.aux_head(pooled), tgt)
            out["aux_residual_loss"] = aux_loss.item()
            loss = loss + cfg.aux_residual_weight * aux_loss

        return loss, out


def _farthest_point_sample(points: Tensor, k: int) -> Tensor:
    """Farthest-point sampling indices, (B, N, 3) -> (B, k). Pure torch, no custom CUDA ops."""
    b, n, _ = points.shape
    device = points.device
    idx = torch.zeros(b, k, dtype=torch.long, device=device)
    dist = torch.full((b, n), float("inf"), device=device)
    far = torch.randint(0, n, (b,), dtype=torch.long, device=device)
    arange = torch.arange(b, device=device)
    for i in range(k):
        idx[:, i] = far
        centroid = points[arange, far].unsqueeze(1)
        dist = torch.minimum(dist, (points - centroid).pow(2).sum(-1))
        far = dist.argmax(dim=1)
    return idx
