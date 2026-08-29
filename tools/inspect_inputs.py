#!/usr/bin/env python3
"""Render every policy input of a ported primitive dataset as a single HTML page.

    python3 tools/inspect_inputs.py --dataset_root /home/samsung/data/push_objonly_delta \
        --out /tmp/inputs.html

Picks four objects spread across the shape range (thinness = min/max extent of the goal
cloud, computed from the data itself -- no external file), extracts one successful episode
each, and writes a self-contained page: animated point clouds in the normalised [-1, 1]
workspace box, goal cloud, the goal_transform arrow, joint/action/velocity traces, the
object path, and the scalar fields. One scrubber drives everything.

This is the pre-training review step of docs/primitive_dataset_pipeline.md: run it on every
newly ported dataset (rotate, flip, ...) BEFORE spending GPU on it, and look at the page with
the checklist in that doc.

Reading the parquet shards takes a couple of minutes for a ~2000-episode dataset.
"""

from __future__ import annotations

import argparse
import base64
import collections
import glob
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# Must match PCDiffusionConfig.pc_center / pc_scale (the capture-workspace box mapped
# isotropically into [-1, 1]). If those defaults change, change these with them.
CENTER = np.array([0.0, 0.0, 0.285], np.float32)
SCALE = np.float32(0.40)

COLS = [
    "observation.point_cloud", "observation.goal_point_cloud", "observation.state",
    "observation.velocity", "observation.object_pose", "observation.goal_pose",
    "observation.goal_transform", "observation.task_onehot", "observation.pose_valid",
    "action", "episode_index",
]


def projector():
    """The fixed 3/4 view used across all campaign visualisations."""
    az, el = np.radians(28), np.radians(24)
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(el), -np.sin(el)], [0, np.sin(el), np.cos(el)]])
    return (rz.T @ rx.T).astype(np.float32)


def pick_objects(root: Path, n: int = 4) -> dict[int, str]:
    """One successful episode per object, spread across the thinness range."""
    eps = [json.loads(x) for x in open(root / "meta" / "episode_objects.jsonl") if x.strip()]
    first: dict[str, int] = {}
    for e in eps:
        name = os.path.basename(e["object"])
        if name not in first and e.get("success", True):
            first[name] = e["episode_index"]

    # Thinness per object from the goal cloud of its first episode (the goal cloud is the
    # object's own geometry, so min/max extent ratio is object shape, not workspace).
    want = set(first.values())
    thin: dict[int, float] = {}
    for f in sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True)):
        try:
            ei = np.array(pq.read_table(f, columns=["episode_index"])["episode_index"].to_pylist())
        except Exception:
            continue  # shard still being written
        hit = want & set(ei.tolist())
        if not hit:
            continue
        t = pq.read_table(f, columns=["observation.goal_point_cloud", "episode_index"])
        gc = np.stack(t["observation.goal_point_cloud"].to_pylist()).astype(np.float32)
        gc = gc.reshape(len(ei), -1, 3)
        for ep in hit:
            g = gc[ei == ep][0]
            ext = np.sort(g.max(0) - g.min(0))
            thin[ep] = float(ext[0] / max(ext[2], 1e-6))
    by_ep = {v: k for k, v in first.items()}
    order = sorted(thin, key=lambda e: thin[e])
    if len(order) <= n:
        picks = order
    else:
        picks = [order[round(i * (len(order) - 1) / (n - 1))] for i in range(n)]
    return {e: by_ep[e] for e in picks}, {e: thin[e] for e in picks}


def extract(root: Path, pick: dict[int, str], thin: dict[int, float]) -> dict:
    R = projector()
    out: dict[str, dict] = {}
    for f in sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True)):
        try:
            ei = np.array(pq.read_table(f, columns=["episode_index"])["episode_index"].to_pylist())
        except Exception:
            continue
        hit = set(pick) & set(ei.tolist())
        if not hit:
            continue
        cols = [c for c in COLS if c in pq.read_schema(f).names]
        t = pq.read_table(f, columns=cols)
        for ep in hit:
            m = ei == ep
            def g(k, shape):
                a = np.stack(t[k].to_pylist()).astype(np.float32)[m]
                return a.reshape(int(m.sum()), *shape)
            pc = (g("observation.point_cloud", (-1, 3)) - CENTER) / SCALE
            gc = (g("observation.goal_point_cloud", (-1, 3)) - CENTER) / SCALE
            st = g("observation.state", (-1,))
            ac = g("action", (-1,))
            ve = g("observation.velocity", (-1,)) if "observation.velocity" in cols else st * 0
            op = g("observation.object_pose", (7,))
            gt9 = (g("observation.goal_transform", (9,))[0].tolist()
                   if "observation.goal_transform" in cols else None)
            T = pc.shape[0]

            corners = np.array(
                [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], np.float32) @ R
            view = np.stack([corners[:, 0], -corners[:, 2]], -1)
            lo, hi = view.min(0), view.max(0)

            def pk(a):
                q = a @ R
                xy = np.stack([q[..., 0], -q[..., 2]], -1)
                return np.clip((xy - lo) / (hi - lo) * 248 + 4, 0, 255).astype(np.uint8)

            step = max(1, T // 80)
            frames = [pk(pc[i]).tobytes() for i in range(0, T, step)]
            out[pick[ep]] = {
                "ep": int(ep), "T": T, "n": len(frames), "pts": pc.shape[1], "step": int(step),
                "thin": round(thin[ep], 2),
                "xy": base64.b64encode(b"".join(frames)).decode(),
                "goal": base64.b64encode(pk(gc[0]).tobytes()).decode(),
                "box": pk(np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
                                   np.float32)).tolist(),
                "arrow": [pk(pc[0].mean(0, keepdims=True))[0].tolist(),
                          pk(gc[0].mean(0, keepdims=True))[0].tolist()],
                "state": np.round(st[::step], 4).tolist(),
                "action": np.round(ac[::step], 4).tolist(),
                "vel": np.round(ve[::step], 4).tolist(),
                "objxy": np.round(op[::step, :2] - op[0, :2], 4).tolist(),
                "goalTf": gt9,
                "pcNorm": [np.round(pc.reshape(-1, 3).min(0), 2).tolist(),
                           np.round(pc.reshape(-1, 3).max(0), 2).tolist()],
                "gcNorm": [np.round(gc[0].min(0), 2).tolist(),
                           np.round(gc[0].max(0), 2).tolist()],
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset_root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--num_objects", type=int, default=4)
    args = ap.parse_args()

    info = json.loads((args.dataset_root / "meta" / "info.json").read_text())
    pick, thin = pick_objects(args.dataset_root, args.num_objects)
    print(f"picked: {[(v, round(thin[k], 2)) for k, v in pick.items()]}")
    data = extract(args.dataset_root, pick, thin)
    if not data:
        raise SystemExit("no episodes extracted -- is the dataset still being ported?")

    meta = {
        "name": args.dataset_root.name,
        "episodes": info["total_episodes"],
        "frames": info.get("total_frames", 0),
        "fps": info.get("fps"),
        "features": sorted(k for k in info["features"]
                           if k.startswith("observation") or k == "action"),
    }
    tpl = (Path(__file__).parent / "inspect_inputs_template.html").read_text()
    page = tpl.replace("__META__", json.dumps(meta)).replace(
        "__DATA__", json.dumps(data, separators=(",", ":"),
                               default=lambda o: o.item() if hasattr(o, "item") else float(o)))
    args.out.write_text(page)
    print(f"wrote {args.out}  ({len(page) // 1024} KB, {len(data)} objects)")


if __name__ == "__main__":
    main()
