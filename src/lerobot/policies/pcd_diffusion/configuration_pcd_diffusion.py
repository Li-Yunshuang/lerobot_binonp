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

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_GOAL_POINT_CLOUD, OBS_POINT_CLOUD, OBS_STATE


@PreTrainedConfig.register_subclass("pcd_diffusion")
@dataclass
class PcdDiffusionConfig(PreTrainedConfig):
    """Goal-conditioned point-cloud diffusion policy: Perceiver encoder + DiT action head.

    Follows the policy-learning recipe of arXiv:2606.13677 (Mana) -- a Perceiver encoder over
    point clouds feeding a transformer diffusion head -- with three deliberate divergences:

    1. **Goal conditioning.** The goal point cloud is encoded by the *same* Perceiver weights as
       the observation cloud (distinguished by a learned type embedding), so the model conditions
       on a target scene configuration rather than an implicit task identity.
    2. **Auxiliary residual-pose loss.** A head predicts the remaining rigid transform between
       each object's current and goal pose, forcing the latents to localise objects instead of
       shortcutting off proprioception. Requires per-frame object-pose labels; set
       `num_objects=0` (the default) when the dataset has none.
    3. **Action chunking.** The paper predicts one action; LeRobot's whole inference path assumes
       chunks. `horizon=1, n_action_steps=1` recovers the paper's behaviour.

    Point clouds are normalised **inside the model**, not by the processor: centring both clouds
    on the *observation* cloud's centroid keeps them in a shared frame, which a per-element
    MIN_MAX over an (N, 3) array cannot do (it would also make normalisation depend on point
    index order). `normalization_mapping["POINT_CLOUD"] = IDENTITY` is what leaves them alone.
    """

    # --- I/O structure ---
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8
    drop_n_last_frames: int | None = None

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            # Clouds are normalised in-model; see the class docstring.
            "POINT_CLOUD": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
            "VISUAL": NormalizationMode.IDENTITY,
        }
    )

    # --- dataset keys (all configurable; defaults match the IsaacLab push dataset) ---
    pointcloud_key: str = OBS_POINT_CLOUD
    goal_pointcloud_key: str = OBS_GOAL_POINT_CLOUD
    object_poses_key: str = "observation.object_poses"
    goal_object_poses_key: str = "observation.goal_object_poses"
    state_key: str = OBS_STATE

    # --- point-cloud preprocessing (in-model) ---
    n_points: int = 512
    resample_mode: str = "random"  # "random" | "fps"
    # None -> divide by the observation cloud's own max radius (per-sample scale normalisation).
    pointcloud_scale: float | None = None
    fourier_bands: int = 8

    # --- Perceiver encoder (paper values) ---
    perceiver_num_latents: int = 4
    perceiver_dim: int = 128
    perceiver_depth: int = 4
    perceiver_heads: int = 4

    # --- proprioception ---
    state_mlp_dims: tuple[int, ...] = (512, 256, 256)

    # --- DiT action head (paper values) ---
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 2
    dropout: float = 0.0

    # --- auxiliary residual-pose head ---
    # Number of objects with pose labels. 0 disables the head entirely and removes the
    # requirement for the two pose keys.
    num_objects: int = 0
    aux_residual_weight: float = 0.1
    # Predict the 6-D rotation of the residual transform as well as its translation. Set False
    # when the pose labels carry no reliable orientation -- regressing against a constant
    # (identity) rotation target teaches nothing and dilutes the translation signal.
    aux_predict_rotation: bool = True
    aux_head_dims: tuple[int, ...] = (256, 256)

    # --- noise scheduler ---
    noise_scheduler_type: str = "DDIM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0
    num_inference_steps: int | None = 10

    do_mask_loss_for_padding: bool = False

    # --- training presets (paper values) ---
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 100_000
    scheduler_decay_lr: float = 1e-6

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.drop_n_last_frames is None:
            self.drop_n_last_frames = self.horizon - self.n_action_steps - self.n_obs_steps + 1

        if self.n_action_steps > self.horizon - self.n_obs_steps + 1:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) must be <= horizon - n_obs_steps + 1 "
                f"({self.horizon - self.n_obs_steps + 1})."
            )
        if self.prediction_type not in ("epsilon", "sample"):
            raise ValueError(f"prediction_type must be 'epsilon' or 'sample', got {self.prediction_type!r}")
        if self.noise_scheduler_type not in ("DDPM", "DDIM"):
            raise ValueError(
                f"noise_scheduler_type must be 'DDPM' or 'DDIM', got {self.noise_scheduler_type!r}"
            )
        if self.resample_mode not in ("random", "fps"):
            raise ValueError(f"resample_mode must be 'random' or 'fps', got {self.resample_mode!r}")
        if self.num_objects < 0:
            raise ValueError(f"num_objects must be >= 0, got {self.num_objects}")
        if self.hidden_dim % self.num_heads:
            raise ValueError(f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})")
        if self.perceiver_dim % self.perceiver_heads:
            raise ValueError(
                f"perceiver_dim ({self.perceiver_dim}) must be divisible by "
                f"perceiver_heads ({self.perceiver_heads})"
            )

    @property
    def uses_aux_head(self) -> bool:
        return self.num_objects > 0 and self.aux_residual_weight > 0

    def validate_features(self) -> None:
        if self.pointcloud_key not in (self.input_features or {}):
            raise ValueError(
                f"pcd_diffusion requires a point cloud at '{self.pointcloud_key}'. "
                f"Inputs present: {sorted(self.input_features or {})}."
            )
        if self.goal_pointcloud_key not in (self.input_features or {}):
            raise ValueError(
                f"pcd_diffusion requires a goal point cloud at '{self.goal_pointcloud_key}'."
            )
        if self.robot_state_feature is None:
            raise ValueError(f"pcd_diffusion requires '{OBS_STATE}' among the inputs.")
        if self.action_feature is None:
            raise ValueError("pcd_diffusion requires 'action' among the outputs.")

        # The pose keys are *training labels*, not inference inputs, but LeRobot types every
        # non-image observation.* key as an input feature. Strip them so they are neither
        # normalised nor expected at inference time.
        for key in (self.object_poses_key, self.goal_object_poses_key):
            if self.input_features and key in self.input_features:
                self.input_features.pop(key)

        if self.uses_aux_head and self.num_objects <= 0:
            raise ValueError("aux_residual_weight > 0 requires num_objects > 0.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
        )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
