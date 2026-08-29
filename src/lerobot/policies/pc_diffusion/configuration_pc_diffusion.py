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
from typing import Any

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig
from lerobot.utils.constants import OBS_GOAL_POINT_CLOUD, OBS_POINT_CLOUD, OBS_STATE


@PreTrainedConfig.register_subclass("pc_diffusion")
@dataclass
class PCDiffusionConfig(PreTrainedConfig):
    """Diffusion policy conditioned on 3D point clouds instead of RGB images.

    Reuses the 1-D conditional U-Net and DDPM/DDIM machinery from the image-based `diffusion`
    policy verbatim -- those are modality-agnostic, seeing only a flat `(B, global_cond_dim)`
    conditioning vector -- and replaces the RGB encoder with a swappable point-cloud encoder
    selected by `pc_encoder`.

    Goal conditioning is first-class because the target task (pushing an object to a commanded
    pose) is not learnable without it: the object's start pose is fixed, so the goal is the only
    thing distinguishing two episodes of the same object.

    Args:
        n_obs_steps: Observation steps fed to the encoder (the current step plus history).
        horizon: Length of the action trajectory the diffusion model generates.
        n_action_steps: How many of those actions are executed before re-planning. Must satisfy
            `n_action_steps <= horizon - n_obs_steps + 1`.
        pc_encoder: Registered encoder name; see `available_pc_encoders()`.
        goal_conditioning: How the goal reaches the network -- "none", "points" (a goal point
            cloud through its own encoder), or "vector" (a small goal vector concatenated into
            the global conditioning).
    """

    # --- input / output structure ---
    n_obs_steps: int = 2
    horizon: int = 64
    n_action_steps: int = 32
    # Number of trailing frames per episode the sampler skips, so a sampled window always has a
    # full action horizon. `None` derives the correct value from horizon/n_action_steps/n_obs_steps.
    drop_n_last_frames: int | None = None

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            # Per-axis min/max over the whole dataset, matching DP3's "limits" normalizer.
            # Note this is anisotropic (x spans ~1.2 m, z ~0.5 m); set POINT_CLOUD to IDENTITY
            # and use `pc_isotropic_rescale` instead if an encoder needs metric geometry.
            "POINT_CLOUD": NormalizationMode.MIN_MAX,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
            "VISUAL": NormalizationMode.IDENTITY,
        }
    )

    # --- point-cloud encoder (swappable) ---
    pc_encoder: str = "pointnet_maxpool"
    pc_encoder_kwargs: dict[str, Any] = field(default_factory=dict)
    pc_feature_dim: int = 256
    pc_feature_key: str = OBS_POINT_CLOUD
    # Applied inside the model, after the processor pipeline. Only meaningful with
    # POINT_CLOUD -> IDENTITY.
    #
    # These map the *capture workspace* -- `_PCD_CROP`, x +/-0.40, y +/-0.30, z -0.03..0.60 --
    # isotropically into [-1, 1]: centre is the box centre, scale its largest half-extent.
    # Preferred over per-key MIN_MAX for four reasons:
    #   * both clouds get the SAME map, so "where the object is vs where it should be" is a
    #     comparison in one frame. Per-key MIN_MAX gave the observation cloud the arms' extent
    #     and the goal cloud the object's, a 1.83x stretch on x.
    #   * isotropic, so a cube stays a cube. MIN_MAX scales each axis independently and hands
    #     the encoder distorted geometry.
    #   * independent of dataset statistics, so adding rotate/flip data cannot shift it, and
    #     re-porting cannot silently change what the numbers mean.
    #   * the same three constants apply on hardware -- no need to carry sim statistics across
    #     or recompute them from real captures.
    # An object spans ~0.46 m in a 0.80 m box, so it occupies ~38% of the range rather than
    # filling it. That is deliberate: the unused range is what encodes *where in the workspace*
    # the object sits.
    pc_isotropic_rescale: bool = False
    pc_center: tuple[float, float, float] = (0.0, 0.0, 0.285)
    pc_scale: float = 0.40

    # --- joint observation/goal encoding ---
    # Replaces the two independent encoders with one that lets the clouds cross-attend before
    # pooling. Contributes `2 * pc_feature_dim`, i.e. exactly what the pair contributed, so
    # global_cond_dim and the U-Net parameter count are unchanged. Requires goal_conditioning to
    # include points -- there is no goal cloud to attend to otherwise.
    pc_cross_attention: bool = False
    cross_attn_hidden_dim: int = 256
    cross_attn_num_heads: int = 4
    cross_attn_num_layers: int = 2

    # --- goal conditioning ---
    goal_conditioning: str = "points"
    goal_pc_feature_key: str = OBS_GOAL_POINT_CLOUD
    goal_feature_key: str = "observation.goal_pose"
    # Extra conditioning available in the v2 datasets. Both are per-frame features that are
    # constant within an episode; `task_onehot` is what makes one generalist policy able to tell
    # push from rotate from flip.
    task_feature_key: str = "observation.task_onehot"
    use_task_onehot: bool = True
    # Auxiliary-head label validity, 0 for sources with no object-pose ground truth (real-world
    # data). Lets one schema serve sim and real, with the auxiliary loss masked per sample.
    pose_valid_key: str = "observation.pose_valid"
    object_pose_key: str = "observation.object_pose"
    # None -> same architecture as `pc_encoder`, with its own (untied) weights.
    goal_encoder: str | None = None
    goal_encoder_kwargs: dict[str, Any] = field(default_factory=dict)
    goal_feature_dim: int = 256

    # --- extra proprioception concatenated into the global conditioning ---
    extra_state_keys: tuple[str, ...] = ()

    # --- auxiliary residual-pose head ---
    # Predicts the remaining current->goal object transform from the same conditioning vector the
    # U-Net sees. It adds no inference cost (the head is dropped at sampling time) and exists to
    # force the point-cloud encoder to actually localise the object relative to the goal rather
    # than shortcutting off proprioception. Off by default so existing checkpoints are unaffected.
    object_poses_key: str = "observation.object_poses"
    goal_object_poses_key: str = "observation.goal_object_poses"
    num_objects: int = 0
    aux_residual_weight: float = 0.1
    # The offline tracker recovers translation only -- the pose labels carry an identity rotation
    # -- so predicting rotation would regress against a constant. Keep False for that dataset.
    aux_predict_rotation: bool = False
    aux_head_dims: tuple[int, ...] = (256, 256)

    # --- action space (metadata) ---
    # "absolute_joint": actions are joint targets, commanded as-is. "delta_joint": actions are
    # action-minus-state at each frame, and the consumer must add the live joint position back
    # (command = jp + d). The model does not branch on this -- it exists so the checkpoint
    # declares what its outputs MEAN, the server advertises it, and the evaluator refuses a
    # mismatch instead of silently commanding near-zero motion.
    action_space: str = "absolute_joint"

    # --- denoiser backbone ---------------------------------------------------------
    # "unet"  -> DiffusionConditionalUnet1d, from the image `diffusion` policy
    # "dit"   -> DiffusionTransformer, from the `multi_task_dit` policy
    # Both take (noisy_action, timestep, flat_conditioning) and return eps-hat, so swapping them
    # holds the input stack, the heads and the objective fixed. That is the point: a comparison
    # between two policies with different encoders and 6x the parameters measures the whole
    # stack, not the architecture.
    backbone: str = "unet"

    # --- DiT (field names must match what DiffusionTransformer reads) ---
    # Sized to match the U-Net's parameter count: 271.0 M against 279.2 M, within 3%. The
    # multi_task_dit defaults (512 x 8) give only 59 M, and a backbone comparison at a 4.7x
    # parameter difference measures capacity, not architecture -- which is exactly what made the
    # earlier pc_diffusion-vs-pcd_diffusion result uninterpretable. Re-check this if the
    # conditioning width changes: adaLN modulation scales with it, so the match will drift.
    hidden_dim: int = 1024
    num_layers: int = 13
    num_heads: int = 8
    dropout: float = 0.1
    use_positional_encoding: bool = False
    timestep_embed_dim: int = 256
    use_rope: bool = True
    rope_base: float = 10000.0

    # --- U-Net (field names must match what DiffusionConditionalUnet1d reads) ---
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True
    gradient_checkpointing: bool = False

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
    # Inference-only: average this many independent samples per observation. 1 keeps the exact
    # single-sample path. Sound here because the scripted demonstrator makes p(action|obs)
    # unimodal, so the conditional mean is the minimum-MSE estimate; it would be wrong for a
    # genuinely multimodal task, where averaging two good trajectories can give a bad one.
    num_samples: int = 1

    compile_model: bool = False
    compile_mode: str = "reduce-overhead"
    do_mask_loss_for_padding: bool = False

    # --- training presets ---
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.drop_n_last_frames is None:
            # A window starting at the last valid index must still contain a full horizon.
            self.drop_n_last_frames = self.horizon - self.n_action_steps - self.n_obs_steps + 1

        if self.n_action_steps > self.horizon - self.n_obs_steps + 1:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) must be <= horizon - n_obs_steps + 1 "
                f"({self.horizon - self.n_obs_steps + 1})."
            )
        # The U-Net halves the temporal dimension once per down_dims level.
        downsample = 2 ** len(self.down_dims)
        if self.horizon % downsample != 0:
            raise ValueError(
                f"horizon ({self.horizon}) must be divisible by {downsample} "
                f"(2 ** len(down_dims), down_dims={self.down_dims})."
            )
        if self.prediction_type not in ("epsilon", "sample"):
            raise ValueError(f"prediction_type must be 'epsilon' or 'sample', got {self.prediction_type!r}")
        if self.noise_scheduler_type not in ("DDPM", "DDIM"):
            raise ValueError(f"noise_scheduler_type must be 'DDPM' or 'DDIM', got {self.noise_scheduler_type!r}")
        if self.backbone not in ("unet", "dit"):
            raise ValueError(f"backbone must be 'unet' or 'dit', got {self.backbone!r}")
        if self.goal_conditioning not in ("none", "points", "vector", "both"):
            raise ValueError(
                "goal_conditioning must be 'none', 'points', 'vector' or 'both', got "
                f"{self.goal_conditioning!r}"
            )

    # --- feature helpers -------------------------------------------------------------

    @property
    def observation_pc_feature(self):
        """The observed scene cloud (excludes the goal cloud)."""
        return (self.input_features or {}).get(self.pc_feature_key)

    @property
    def goal_pc_feature(self):
        return (self.input_features or {}).get(self.goal_pc_feature_key)

    # --- future-latent auxiliary head ---
    # Predicts the encoder's own latent for the observation `future_latent_horizon` frames
    # ahead, i.e. a latent forward model. Unlike the residual-pose head -- whose target's
    # variance collapsed 10x under the commanded-goal convention, leaving aux_residual_loss
    # at 0.000 on every recent run -- this target carries real variance: the scene cloud's
    # centroid moves a median 33 mm over 32 frames.
    #
    # The target is produced by the SAME encoder being trained, so the loss is minimised by a
    # constant encoder output. That failure would not merely trivialise the auxiliary task --
    # the encoder is shared with the policy conditioning, so a collapse destroys the policy's
    # observation features. Two guards, both on by default:
    #   * `future_latent_stop_grad` detaches the target, so no gradient rewards shrinking it.
    #   * `future_latent_predictor_dims` puts an asymmetric predictor on the online side only
    #     (SimSiam, Chen & He 2021), which prevents collapse without a momentum encoder.
    # `future_latent_std` is logged every step; if it trends to zero, the representation is
    # collapsing regardless of what the loss says.
    future_latent_weight: float = 0.0
    future_latent_horizon: int = 32
    future_latent_stop_grad: bool = True
    future_latent_predictor_dims: tuple[int, ...] = (512, 512)
    # Restrict the target to the object rather than the whole scene. The cloud contains arms as
    # well as the object, and the object is stationary outside the push phase, so most of the
    # frame-to-frame change is arm motion -- which the policy already predicts from
    # proprioception. Left unmasked, the cheapest way to satisfy this head is to model the robot,
    # teaching the encoder nothing about object dynamics.
    #
    # The dataset carries no per-point labels (segmentation exists only in the simulator at
    # collection time), so the object region is derived: its centroid is the goal cloud's
    # centroid minus the remaining displacement, both of which are stored. Validated on
    # push_pc1024_poses -- the K nearest points to that centroid span 86-88% of the object's
    # true extent, stably across an episode.
    future_latent_object_only: bool = False
    future_latent_object_points: int = 256

    @property
    def uses_aux_head(self) -> bool:
        return self.num_objects > 0 and self.aux_residual_weight > 0

    @property
    def uses_future_latent(self) -> bool:
        return self.future_latent_weight > 0 and self.future_latent_horizon > 0

    def validate_features(self) -> None:
        if self.observation_pc_feature is None:
            available = sorted(self.point_cloud_features)
            raise ValueError(
                f"pc_diffusion requires a point-cloud input at '{self.pc_feature_key}'. "
                f"Point-cloud features found: {available}."
            )
        if self.robot_state_feature is None:
            raise ValueError(f"pc_diffusion requires '{OBS_STATE}' among the inputs.")
        if self.action_feature is None:
            raise ValueError("pc_diffusion requires 'action' among the outputs.")

        if self.goal_conditioning in ("points", "both") and self.goal_pc_feature is None:
            raise ValueError(
                f"goal_conditioning='points' requires a point cloud at '{self.goal_pc_feature_key}'. "
                "Port the dataset with a goal cloud, or set --policy.goal_conditioning=none."
            )
        if self.goal_conditioning in ("vector", "both") and self.goal_feature_key not in (
            self.input_features or {}
        ):
            raise ValueError(
                f"goal_conditioning='vector' requires '{self.goal_feature_key}' among the inputs."
            )

        for key in self.extra_state_keys:
            if key not in (self.input_features or {}):
                raise ValueError(f"extra_state_keys entry '{key}' is not among the dataset inputs.")

        # The pose keys are *training labels*, not inference inputs, but LeRobot types every
        # non-image observation.* key as an input feature. Strip them so they are neither
        # normalised (mean/std over a rigid transform is meaningless) nor demanded at eval time,
        # where no tracker is running.
        for key in (self.object_poses_key, self.goal_object_poses_key,
                    self.object_pose_key, self.pose_valid_key):
            if self.input_features and key in self.input_features:
                self.input_features.pop(key)
        if not self.use_task_onehot and self.input_features:
            self.input_features.pop(self.task_feature_key, None)

        # num_objects is the switch: it stays 0 unless the dataset carries pose labels, so the
        # default config (and every checkpoint trained before the head existed) is aux-free.
        if self.num_objects < 0:
            raise ValueError(f"num_objects must be >= 0, got {self.num_objects}")

        for key, ft in self.point_cloud_features.items():
            if ft.type is not FeatureType.POINT_CLOUD or len(ft.shape) != 2:
                raise ValueError(f"'{key}' must be a rank-2 (num_points, channels) point cloud, got {ft}.")

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        idx = list(range(1 - self.n_obs_steps, 1))
        if self.uses_future_latent:
            # One extra, non-contiguous frame: the future-latent target. A contiguous range
            # would make the dataloader fetch `horizon` extra frames of every observation key
            # -- 32 extra 1024-point clouds per sample -- to use exactly one of them.
            idx.append(self.future_latent_horizon)
        return idx

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
