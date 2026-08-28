#!/usr/bin/env python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License"); see the repo LICENSE.

"""Add object-pose labels to the ported push dataset, for the auxiliary residual head.

Consumes the per-frame tracks from `track_push_object_poses.py` and writes two new features:

    observation.object_poses       (1, 4, 4)  current object pose
    observation.goal_object_poses  (1, 4, 4)  goal object pose

Both are expressed relative to the object's t=0 position, which is the origin of the track, so
`inv(T_cur) @ T_goal` is exactly the remaining displacement -- the progress signal the auxiliary
head regresses.

**Rotation is identity in both.** The tracker recovers translation only; a yaw estimate from a
partial 2-view cloud would be far noisier than the translation. Train with
`--policy.aux_predict_rotation=false` so the head regresses the 3 translation components instead
of being fit against a constant rotation target.

Episodes whose track failed the end-point cross-check fall back to a straight-line interpolation
from origin to goal, so the episode set stays identical to the other training runs -- changing it
would confound the ablation with a change of data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lerobot.datasets.dataset_tools import add_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_root", type=Path, required=True)
    ap.add_argument("--repo_id", type=str, default="local/push_pc1024")
    ap.add_argument("--tracks", type=Path, required=True)
    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument("--out_repo_id", type=str, default="local/push_pc1024_poses")
    args = ap.parse_args()

    ds = LeRobotDataset(args.repo_id, root=args.dataset_root)
    rows = [json.loads(x) for x in (args.dataset_root / "meta" / "episode_objects.jsonl").read_text().splitlines() if x.strip()]
    by_new = {r["episode_index"]: r for r in rows}

    tracks: dict[int, dict] = {}
    for line in args.tracks.read_text().splitlines():
        if line.strip():
            t = json.loads(line)
            tracks[t["episode_index"]] = t  # keyed by SOURCE episode index

    n_frames = ds.meta.total_frames
    cur = np.zeros((n_frames, 1, 4, 4), dtype=np.float32)
    goal = np.zeros((n_frames, 1, 4, 4), dtype=np.float32)
    cur[:, 0] = np.eye(4, dtype=np.float32)
    goal[:, 0] = np.eye(4, dtype=np.float32)

    eps = ds.meta.episodes
    n_fallback = 0
    for new_ep, row in sorted(by_new.items()):
        start = int(eps["dataset_from_index"][new_ep])
        end = int(eps["dataset_to_index"][new_ep])
        length = end - start
        g = np.array([row["goal_dx"], row["goal_dy"]], dtype=np.float32)

        t = tracks.get(row["src_episode_index"])
        if t is not None and t.get("residual_ok") and t.get("xy") and len(t["xy"]) >= length:
            xy = np.asarray(t["xy"][:length], dtype=np.float32)
        else:
            # Straight-line fallback: keeps the episode set identical across runs.
            n_fallback += 1
            xy = np.linspace(np.zeros(2, np.float32), g, length).astype(np.float32)

        cur[start:end, 0, 0, 3] = xy[:, 0]
        cur[start:end, 0, 1, 3] = xy[:, 1]
        goal[start:end, 0, 0, 3] = g[0]
        goal[start:end, 0, 1, 3] = g[1]

    print(f"built pose labels for {len(by_new)} episodes / {n_frames} frames "
          f"({n_fallback} used the straight-line fallback)", flush=True)
    resid = goal[:, 0, :3, 3] - cur[:, 0, :3, 3]
    d = np.linalg.norm(resid, axis=-1)
    print(f"  residual-to-goal: start median {np.median(d[::310]):.3f} m, overall median {np.median(d):.3f} m", flush=True)

    info = {"dtype": "float32", "shape": [1, 4, 4], "names": None}
    add_features(
        ds,
        {"observation.object_poses": (cur, dict(info)), "observation.goal_object_poses": (goal, dict(info))},
        output_dir=str(args.out_root),
        repo_id=args.out_repo_id,
    )
    print(f"ADD_FEATURES_DONE -> {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
