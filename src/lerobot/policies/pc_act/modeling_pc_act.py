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

"""ACT on the point-cloud input stack.

The stock `lerobot.policies.act.ACT` module is reused verbatim -- CVAE, transformer encoder /
decoder, chunk queries, L1 + KL objective. What is new is only the input composition: the
observation and goal point clouds are encoded by the same PointNetMaxPool encoders the diffusion
stack uses, concatenated with the commanded `goal_transform`, and handed to ACT as its
environment-state token. The encoders train end-to-end through ACT's loss.
"""

from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.policies.act.modeling_act import ACT
from lerobot.policies.pc_diffusion.encoders import make_pc_encoder
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_STATE

from .configuration_pc_act import PCACTConfig


class PCACTPolicy(PreTrainedPolicy):
    """ACT policy over point-cloud observations."""

    config_class = PCACTConfig
    name = "pc_act"

    def __init__(self, config: PCACTConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        pc_ft = config.input_features[config.pc_feature_key]
        goal_ft = config.input_features[config.goal_pc_feature_key]
        self.pc_encoder = make_pc_encoder(
            config.pc_encoder,
            num_points=int(pc_ft.shape[0]),
            in_channels=int(pc_ft.shape[1]),
            out_dim=config.pc_feature_dim,
        )
        self.goal_encoder = make_pc_encoder(
            config.pc_encoder,
            num_points=int(goal_ft.shape[0]),
            in_channels=int(goal_ft.shape[1]),
            out_dim=config.goal_feature_dim,
        )
        self.register_buffer("_pc_center", torch.tensor(config.pc_center, dtype=torch.float32))

        # The stock ACT model, configured through this policy's own config: image_features is
        # empty and env_state_feature reports the composed embedding width, so ACT builds
        # exactly [latent, robot_state, env_state] tokens and no vision backbone.
        self.model = ACT(config)
        self.reset()

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    # ---- input composition -------------------------------------------------------------

    def _rescale(self, cloud: Tensor) -> Tensor:
        if not self.config.pc_isotropic_rescale:
            return cloud
        cloud = cloud.clone()
        cloud[..., :3] = (cloud[..., :3] - self._pc_center) / self.config.pc_scale
        return cloud

    def _compose(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Encode clouds and build ACT's environment-state token. `(B, N, C)` clouds in."""
        pc = self._rescale(batch[self.config.pc_feature_key])
        goal = self._rescale(batch[self.config.goal_pc_feature_key])
        env = torch.cat(
            [self.pc_encoder(pc), self.goal_encoder(goal), batch[self.config.goal_feature_key]],
            dim=-1,
        )
        out = {OBS_STATE: batch[OBS_STATE], OBS_ENV_STATE: env}
        for k in (ACTION, "action_is_pad"):
            if k in batch:
                out[k] = batch[k]
        return out

    # ---- inference ---------------------------------------------------------------------

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        actions, _ = self.model(self._compose(batch))
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    # ---- training ----------------------------------------------------------------------

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        composed = self._compose(batch)
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(composed)

        abs_err = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
        valid = ~batch["action_is_pad"].unsqueeze(-1)
        num_valid = valid.sum() * abs_err.shape[-1]
        l1_loss = (abs_err * valid).sum() / num_valid.clamp_min(1)

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae and log_sigma_x2_hat is not None:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp()))
                .sum(-1)
                .mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss
        return loss, loss_dict
