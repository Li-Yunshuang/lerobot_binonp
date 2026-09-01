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
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.utils.constants import OBS_GOAL_POINT_CLOUD, OBS_POINT_CLOUD


@PreTrainedConfig.register_subclass("pc_act")
@dataclass
class PCACTConfig(PreTrainedConfig):
    """ACT (Action Chunking Transformer, CVAE + L1) on the point-cloud input stack.

    Same inputs as the `pc_diffusion` baseline family -- observation point cloud, goal point
    cloud, and the commanded `goal_transform` vector -- but the denoiser is replaced by the
    stock ACT model: the two clouds are encoded by PointNetMaxPool encoders, concatenated with
    the goal vector into a single environment-state token, and fed to ACT's transformer
    alongside the robot-state token and the CVAE latent. The ACT module itself is reused
    verbatim from `lerobot.policies.act`; only the input composition is new.

    Deliberately kept to ACT's own conventions where they differ from the diffusion stack:
    MEAN_STD normalisation for state/action (ACT's native recipe; L1 regression has no
    unit-variance assumption, so the MIN_MAX/SNR issue that killed delta actions for diffusion
    does not apply), single observation step, L1 + KL objective.
    """

    # --- input / output structure ---
    n_obs_steps: int = 1  # ACT is single-observation-step by construction
    chunk_size: int = 64  # matched to the diffusion baseline's horizon
    n_action_steps: int = 32  # matched to the diffusion baseline's executed prefix

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            # Clouds bypass stats-based normalisation; the shared isotropic workspace map below
            # is applied inside the model, exactly as in pc_diffusion.
            "POINT_CLOUD": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
            "VISUAL": NormalizationMode.IDENTITY,
        }
    )

    # --- point-cloud inputs (same conventions as pc_diffusion) ---
    pc_encoder: str = "pointnet_maxpool"
    pc_feature_dim: int = 256
    goal_feature_dim: int = 256
    pc_feature_key: str = OBS_POINT_CLOUD
    goal_pc_feature_key: str = OBS_GOAL_POINT_CLOUD
    goal_feature_key: str = "observation.goal_transform"
    # The capture workspace mapped isotropically into [-1, 1]; see pc_diffusion's config for the
    # full rationale. Keep these in lockstep with that policy.
    pc_isotropic_rescale: bool = True
    pc_center: tuple[float, float, float] = (0.0, 0.0, 0.285)
    pc_scale: float = 0.40

    # --- action semantics (contract metadata, same as pc_diffusion) ---
    action_space: str = "absolute_joint"

    # --- ACT architecture (stock lerobot ACT defaults) ---
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    # Original ACT effectively uses one decoder layer (upstream bug reproduced deliberately).
    n_decoder_layers: int = 1
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4
    temporal_ensemble_coeff: float | None = None
    dropout: float = 0.1
    kl_weight: float = 10.0

    # --- training preset ---
    # ACT's reference lr is 1e-5 at batch size 8; this stack trains at batch 64, so the default
    # is linearly scaled. Override with --policy.optimizer_lr to run the reference value.
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4

    def __post_init__(self):
        super().__post_init__()
        if self.n_obs_steps != 1:
            raise ValueError(f"pc_act is single-observation-step; got n_obs_steps={self.n_obs_steps}")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) must be <= chunk_size ({self.chunk_size})"
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError("temporal ensembling requires n_action_steps == 1")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> None:
        return None

    # ---- features the inner ACT module reads off this config -------------------------------
    # ACT tokenises [latent, robot_state, env_state, image pixels]. Images are absent; the
    # env-state token is the composed cloud/goal embedding, whose width is fixed by the encoder
    # dims rather than by any dataset feature -- hence the override of the base property.

    @property
    def image_features(self) -> dict:
        return {}

    @property
    def env_state_feature(self) -> PolicyFeature:
        goal_dim = (
            self.input_features[self.goal_feature_key].shape[0]
            if self.input_features and self.goal_feature_key in self.input_features
            else 9
        )
        return PolicyFeature(
            type=FeatureType.STATE,
            shape=(self.pc_feature_dim + self.goal_feature_dim + goal_dim,),
        )

    def validate_features(self) -> None:
        # Same strip as pc_diffusion: pose labels are training bookkeeping, not inference
        # inputs, but LeRobot types every observation.* key as an input feature. Removing them
        # here keeps them out of the checkpoint's `expects` contract at eval time.
        for key in ("observation.object_pose", "observation.pose_valid",
                    "observation.object_poses", "observation.goal_object_poses"):
            if self.input_features and key in self.input_features:
                self.input_features.pop(key)
        for key, what in (
            (self.pc_feature_key, "observation point cloud"),
            (self.goal_pc_feature_key, "goal point cloud"),
            (self.goal_feature_key, "goal transform vector"),
        ):
            if key not in (self.input_features or {}):
                raise ValueError(f"pc_act requires the {what} ('{key}') among input_features")
        if self.robot_state_feature is None:
            raise ValueError("pc_act requires observation.state among input_features")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
