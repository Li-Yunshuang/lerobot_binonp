#!/usr/bin/env python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations.

"""Recover a per-frame object trajectory from the source point clouds.

The collector never wrote object poses, so the auxiliary residual-to-goal head has no labels.
This reconstructs the object's planar translation for every frame by isolating its points and
registering them against the frame-0 object cloud, warm-starting each frame from the previous
estimate (the object moves ~2 mm per frame, so a small local search suffices).

**Translation only.** Yaw is not recovered: a 3-DoF search costs ~12x more, and a rotation label
derived from a partial 2-view cloud would be far noisier than the translation. Downstream, set
`aux_predict_rotation=False` so the head regresses only the 3 translation components rather than
being trained against a degenerate rotation target.

Writes `<src>/../object_tracks.jsonl` (never inside the source dataset):
    {"episode_index": i, "xy": [[x,y], ...310], "residual_ok": true, "end_err_m": 0.004}

`end_err_m` cross-checks the final tracked position against the independently ICP-recovered goal
displacement; a large value means the track drifted and the episode should be dropped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, "/home/samsung/3D_Bimanual_repo/scripts_isaaclab/irregular/pc_common")


def load_pcd(path: Path) -> np.ndarray:
    col = pq.read_table(path, columns=["observation.point_cloud"]).column("observation.point_cloud")
    arr = col.combine_chunks()
    while hasattr(arr, "values"):
        arr = arr.values
    return arr.to_numpy(zero_copy_only=False).astype(np.float32).reshape(len(col), -1, 3)


def track_episode(job: tuple[int, str], goal: dict[int, list[float]], max_src: int = 300) -> dict:
    import pc_ops
    from scipy.spatial import cKDTree

    ep, path = job
    try:
        pcd = load_pcd(Path(path))
        obj0 = pc_ops.isolate_object(pcd[0])
        if len(obj0) < 50:
            return {"episode_index": ep, "residual_ok": False, "reason": "few_t0_points"}
        top = float(obj0[:, 2].max())
        rng = np.random.default_rng(ep)
        src = obj0[rng.permutation(len(obj0))[:max_src]]

        est = np.zeros(2)
        xy = []
        for t in range(pcd.shape[0]):
            o = pc_ops.isolate_object(pcd[t], xy_halfspan=0.38, z_max=top + 0.03)
            if len(o) >= 50:
                tree = cKDTree(o)
                best = est.copy()
                for step in (0.005, 0.001):
                    offs = np.arange(-3, 4) * step
                    cands = [(best[0] + i, best[1] + j) for i in offs for j in offs]
                    sc = [tree.query(src + np.array([dx, dy, 0.0], np.float32))[0].mean() for dx, dy in cands]
                    best = np.array(cands[int(np.argmin(sc))])
                est = best
            xy.append(est.copy())
        xy = np.asarray(xy)
        g = np.asarray(goal.get(ep, [np.nan, np.nan]))
        end_err = float(np.linalg.norm(xy[-1] - g)) if np.isfinite(g).all() else float("nan")
        return {
            "episode_index": ep,
            "xy": xy.round(5).tolist(),
            "residual_ok": bool(np.isfinite(end_err) and end_err < 0.03),
            "end_err_m": end_err,
        }
    except Exception as exc:  # noqa: BLE001 - one bad episode must not kill the run
        return {"episode_index": ep, "residual_ok": False, "reason": f"{type(exc).__name__}:{exc}"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_root", type=Path, required=True, help="ported v3.0 dataset (for the goals)")
    ap.add_argument("--src_root", type=Path, required=True, help="v2.1 source (read-only)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = [json.loads(x) for x in (args.dataset_root / "meta" / "episode_objects.jsonl").read_text().splitlines() if x.strip()]
    info = json.loads((args.src_root / "meta" / "info.json").read_text())
    chunks = int(info["chunks_size"])

    goal = {r["src_episode_index"]: [r["goal_dx"], r["goal_dy"]] for r in rows}
    jobs = [
        (r["src_episode_index"],
         str(args.src_root / "data" / f"chunk-{r['src_episode_index'] // chunks:03d}" /
             f"episode_{r['src_episode_index']:06d}.parquet"))
        for r in rows
    ]
    if args.limit:
        jobs = jobs[: args.limit]

    done = set()
    if args.out.exists():
        done = {json.loads(x)["episode_index"] for x in args.out.read_text().splitlines() if x.strip()}
    jobs = [j for j in jobs if j[0] not in done]
    print(f"tracking {len(jobs)} episodes ({len(done)} already done)", flush=True)

    t0 = time.time()
    n_ok = 0
    with open(args.out, "a") as f, ProcessPoolExecutor(args.num_workers) as ex:
        for i, res in enumerate(ex.map(partial(track_episode, goal=goal), jobs, chunksize=1), 1):
            f.write(json.dumps(res) + "\n")
            f.flush()
            n_ok += bool(res.get("residual_ok"))
            if i % 100 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(jobs)}  ok={n_ok}  {el/i:.2f}s/ep  eta {(len(jobs)-i)*el/i/60:.0f} min", flush=True)
    print(f"TRACK_DONE ok={n_ok}/{len(jobs)} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
