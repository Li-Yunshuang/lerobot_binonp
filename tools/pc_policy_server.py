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

"""Serve a trained `pc_diffusion` checkpoint to the IsaacLab evaluation process.

LeRobot requires Python >= 3.12 / numpy >= 2 while IsaacSim ships CPython 3.11 / numpy 1.26, so
the simulator cannot import the policy. Rather than reimplement normalization on the simulator
side -- which is exactly how eval silently drifts from training -- the policy stays here, behind
a socket, and runs *the same* `make_pre_post_processors` pipeline objects that training used.

The `hello` handshake reports the point counts and preprocessing contract this checkpoint expects
so the client can assert its own preprocessing matches before a single rollout runs.

Usage::

    python tools/pc_policy_server.py --checkpoint outputs/train/pc_v0/checkpoints/last/pretrained_model
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION

DEFAULT_PC_COMMON = "/home/samsung/3D_Bimanual_repo/scripts_isaaclab/irregular/pc_common"

logger = logging.getLogger("pc_policy_server")


def build_policy(checkpoint: Path, device: str, dataset_root: Path | None, repo_id: str | None,
                 n_action_steps: int | None = None, num_inference_steps: int | None = None,
                 num_samples: int | None = None):
    """Load the checkpoint plus its saved processor pipelines.

    Features come from the training dataset's metadata when it is available, which keeps the
    feature dict (and therefore the normalization keys) identical to training.
    """
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    cfg.pretrained_path = checkpoint
    cfg.device = device
    if n_action_steps is not None:
        limit = cfg.horizon - cfg.n_obs_steps + 1
        if not 1 <= n_action_steps <= limit:
            raise SystemExit(f"n_action_steps must be in [1, {limit}] for horizon={cfg.horizon}")
        logger.info("overriding n_action_steps %d -> %d", cfg.n_action_steps, n_action_steps)
        cfg.n_action_steps = n_action_steps
    if num_inference_steps is not None:
        logger.info("overriding num_inference_steps %s -> %d", cfg.num_inference_steps, num_inference_steps)
        cfg.num_inference_steps = num_inference_steps
    if num_samples is not None:
        if num_samples < 1:
            raise SystemExit(f"num_samples must be >= 1, got {num_samples}")
        logger.info("overriding num_samples %s -> %d", getattr(cfg, "num_samples", 1), num_samples)
        cfg.num_samples = num_samples

    ds_meta = None
    if dataset_root is not None:
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

        ds_meta = LeRobotDatasetMetadata(repo_id or "local/dataset", root=dataset_root)

    policy = make_policy(cfg=cfg, ds_meta=ds_meta)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return cfg, policy, preprocessor, postprocessor


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--socket", type=str, default="/tmp/pc_policy.sock")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dataset_root", type=Path, default=None)
    p.add_argument("--repo_id", type=str, default=None)
    p.add_argument("--pc_common", type=str, default=DEFAULT_PC_COMMON)
    # Inference-time only: the model always predicts `horizon` actions and this decides how many
    # are executed before re-planning. Overridable without retraining, which makes the
    # open-loop-horizon question answerable in an evaluation rather than a training run.
    p.add_argument("--n_action_steps", type=int, default=None,
                   help="override the checkpoint's action-chunk execution length")
    p.add_argument("--num_inference_steps", type=int, default=None,
                   help="override the number of denoising steps at sampling time")
    p.add_argument("--num_samples", type=int, default=None,
                   help="average this many samples per observation (inference-only; 1 = unchanged)")
    args = p.parse_args()

    sys.path.insert(0, args.pc_common)
    from pc_protocol import pack_arrays, recv_msg, send_msg, unpack_arrays

    cfg, policy, preprocessor, postprocessor = build_policy(
        args.checkpoint, args.device, args.dataset_root, args.repo_id,
        n_action_steps=args.n_action_steps, num_inference_steps=args.num_inference_steps,
        num_samples=args.num_samples,
    )

    # Read the cloud features generically: pc_diffusion and pcd_diffusion name their config
    # fields differently, but both type their clouds as FeatureType.POINT_CLOUD.
    pc_key = getattr(cfg, "pc_feature_key", None) or getattr(cfg, "pointcloud_key", None)
    goal_key = getattr(cfg, "goal_pc_feature_key", None) or getattr(cfg, "goal_pointcloud_key", None)
    inputs = cfg.input_features or {}
    pc_ft = inputs.get(pc_key)
    goal_ft = inputs.get(goal_key)
    if pc_ft is None:
        raise SystemExit(
            f"Checkpoint config {type(cfg).__name__} exposes no observation point cloud "
            f"(looked for {pc_key!r} among {sorted(inputs)})."
        )
    contract = {}
    prep_path = args.dataset_root / "meta" / "preprocessing.json" if args.dataset_root else None
    if prep_path and prep_path.exists():
        contract = json.loads(prep_path.read_text())

    expects = [k for k in (cfg.input_features or {})]
    hello = {
        "op": "hello",
        "policy": cfg.type,
        # Encoder name differs per policy; report whatever the config calls it.
        "pc_encoder": getattr(cfg, "pc_encoder", None) or f"{cfg.type}:builtin",
        # The dataset-side point count, i.e. what the client must produce. A policy that
        # resamples internally (pcd_diffusion's n_points) still expects the dataset shape here.
        "num_points": int(pc_ft.shape[0]),
        "in_channels": int(pc_ft.shape[1]),
        "goal_num_points": int(goal_ft.shape[0]) if goal_ft is not None else 0,
        "goal_conditioning": getattr(cfg, "goal_conditioning", "points" if goal_ft is not None else "none"),
        "n_obs_steps": cfg.n_obs_steps,
        "n_action_steps": cfg.n_action_steps,
        "num_samples": int(getattr(cfg, "num_samples", 1) or 1),
        "expects": expects,
        "preprocessing": contract,
    }
    logger.info("checkpoint contract: %s", json.dumps({k: v for k, v in hello.items() if k != "preprocessing"}))

    if os.path.exists(args.socket):
        os.unlink(args.socket)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(args.socket)
    srv.listen(1)
    logger.info("listening on %s", args.socket)
    print(f"PC_POLICY_SERVER_READY {args.socket}", flush=True)

    try:
        while True:
            conn, _ = srv.accept()
            logger.info("client connected")
            n_act = 0
            t_total = 0.0
            try:
                while True:
                    header, payload = recv_msg(conn)
                    op = header.get("op")
                    if op == "hello":
                        send_msg(conn, {**hello, "payload_bytes": 0})
                    elif op == "reset":
                        policy.reset()
                        send_msg(conn, {"op": "reset_ok", "payload_bytes": 0})
                    elif op == "act":
                        t0 = time.perf_counter()
                        arrays = unpack_arrays(header, payload)
                        batch = {k: torch.from_numpy(np.array(v, copy=True)) for k, v in arrays.items()}
                        batch = preprocessor(batch)
                        with torch.inference_mode():
                            action = policy.select_action(batch)
                        action = postprocessor(action)
                        act_np = action.detach().to("cpu").numpy().astype(np.float32)
                        dt = (time.perf_counter() - t0) * 1000
                        n_act += 1
                        t_total += dt
                        h, pay = pack_arrays({ACTION: act_np}, [ACTION])
                        send_msg(conn, {"op": "act_ok", "latency_ms": round(dt, 2), **h}, pay)
                    elif op == "bye":
                        send_msg(conn, {"op": "bye_ok", "payload_bytes": 0})
                        break
                    else:
                        send_msg(conn, {"op": "error", "message": f"unknown op {op!r}", "payload_bytes": 0})
            except ConnectionError as exc:
                logger.info("client disconnected: %s", exc)
            finally:
                if n_act:
                    logger.info("served %d act calls, mean %.1f ms", n_act, t_total / n_act)
                conn.close()
    except KeyboardInterrupt:
        logger.info("shutting down")
    finally:
        srv.close()
        if os.path.exists(args.socket):
            os.unlink(args.socket)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
