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

"""Port the real-world bimanual push recordings (LeRobot v2.1 + depth) to v3.0.

Sibling of ``port_isaaclab_pointcloud_push.py``. That script ports the *simulated* push
dataset; this one ports the *real* recordings so the two can be aggregated into one
co-training set. ``aggregate_datasets`` requires identical ``fps``, ``robot_type`` and
feature dicts, so this script emits exactly the ``push_v2`` schema -- same keys, same
shapes, same dtypes, ``fps=10``, ``robot_type='trossen_aloha_bimanual'`` -- even where the
real data has nothing to put in a field.

What the real source has and does not have
------------------------------------------
Source: ``real_world_push_data/{irregular_push, regular_push/depth_real}``, LeRobot v2.1,
20 episodes each, 30 fps, ``robot_type='trossen_ai_stationary'``.

  * ``observation.state`` / ``action`` (14) -- present, radians, same joint layout.
  * ``observation.velocity``               -- ABSENT. Reconstructed by central differencing
                                              the 30 fps state, then decimated. This is a
                                              numerical derivative of a quantised encoder
                                              signal, not a measured velocity.
  * point cloud                            -- ABSENT. Built here from the two depth streams
                                              (``data_depth/``, 480x640 uint16).
  * object pose                            -- ABSENT, and unrecoverable: no motion capture,
                                              no marker, no mesh registration. Hence
                                              ``observation.pose_valid = 0`` for every real
                                              frame and ``observation.object_pose`` a
                                              per-episode constant that the auxiliary loss
                                              must mask.
  * goal                                   -- ABSENT. Recovered by registering the object's
                                              cloud at the first frame onto its cloud at the
                                              last frame (3 DoF: yaw + xy). See
                                              ``register_xy_yaw`` for the error characteristics.

Frames and calibration
----------------------
Clouds are built with :mod:`depth_to_cloud` and preprocessed with :mod:`pc_ops` -- the same
``crop_and_resample`` / ``resample_to`` / ``pose_to_vec9`` / ``transform_points`` the sim
port and the IsaacLab eval client use. That is the single most important property here: one
crop-and-resample implementation, so training and evaluation cannot drift apart.

The 90-degree ``calibworld_to_mjworld`` correction in
``scripts/data_collection/pointcloud_utils.py`` is **not** applied. The IsaacLab collector
feeds ``transform_camera_to_world`` straight into its camera offset and subtracts the env
origin, so the sim's env-local frame *is* the calibration world frame. Applying the rotation
here would put every real cloud a quarter turn away from every sim cloud. See
``depth_to_cloud`` for the full argument.

The rig calibration is measured from the data rather than trusted (``calibrate_rig``):
``camera_extrinsics.json`` is only correct for one of the two cameras on these recordings.
Run with ``--recalibrate`` to redo it; the result is cached next to the destination.

Usage::

    python examples/port_datasets/port_real_world_push.py \
        --src_root .../real_world_push_data/irregular_push \
        --src_root .../real_world_push_data/regular_push/depth_real \
        --dst_root /home/samsung/data/push_real_v1 \
        --repo_id local/push_real_v1 --num_workers 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# `pc_ops` and `depth_to_cloud` live in the IsaacLab repo so the Python 3.11 eval/inference
# clients can import the exact same files. Injected on sys.path rather than copied.
DEFAULT_PC_COMMON = "/home/samsung/3D_Bimanual_repo/scripts_isaaclab/irregular/pc_common"
DEFAULT_EXTRINSICS = "/home/samsung/3D_Bimanual_repo/scripts/data_collection/camera_extrinsics.json"

logger = logging.getLogger(__name__)

STATE_JOINT_NAMES = (
    [f"left_joint_{i}" for i in range(6)]
    + ["left_gripper"]
    + [f"right_joint_{i}" for i in range(6)]
    + ["right_gripper"]
)

# Must match push_v2 exactly or `aggregate_datasets` refuses the merge.
TARGET_ROBOT_TYPE = "trossen_aloha_bimanual"
TARGET_FPS = 10
# The source task string is "Real world perception", which names no primitive and would make
# `pc_ops.task_onehot` raise. Real episodes ARE pushes, and the merged dataset should carry
# one task, so they are re-labelled with the sim dataset's task string verbatim.
TARGET_TASK = "Push the object to the goal position."

DEPTH_KEYS = {"cam_high": "observation.depth.cam_high", "cam_low": "observation.depth.cam_low"}
# Which calibration entry describes which recorded stream. Established empirically, not from
# the file: `calibrate_rig` fits the tabletop through each candidate extrinsic and keeps the
# assignment whose table comes out horizontal. Measured on these recordings, cam_high through
# the "right" entry gives a 0.95 deg tilt and puts the tabletop at 21.9 mm (the calibration
# recorded 20.0 mm); every other pairing is off by 4.4-5.6 deg. See --verify_assignment.
CAMERA_ASSIGNMENT = {"cam_high": "right", "cam_low": "left"}


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


@dataclass
class PortConfig:
    src_roots: list[Path]
    dst_root: Path
    repo_id: str = "local/push_real_v1"
    num_points: int = 1024
    goal_num_points: int = 512
    crop_x: tuple[float, float] = (-0.61, 0.61)
    crop_y: tuple[float, float] = (-0.38, 0.38)
    crop_z: tuple[float, float] = (-0.03, 0.50)
    # 30 fps source, 10 fps target. Integer decimation, no resampling of the signal.
    decimate: int = 3
    cameras: tuple[str, ...] = ("cam_high", "cam_low")
    extrinsics_json: str = DEFAULT_EXTRINSICS
    pc_common: str = DEFAULT_PC_COMMON
    # Tabletop removal. The sim clouds are object + arms with everything static already
    # dropped by the collector's mesh/skeleton segmentation; a raw real cloud is ~86%
    # tabletop inside the same crop. A height cut is the closest equivalent that needs
    # neither a mesh nor a pose. Set to a negative value to keep the table.
    table_cut_m: float = 0.015
    # Object isolation for goal recovery. Measured on these recordings the object occupies
    # z in [0.015, 0.14] and the parked arms / overhead frame sit above z = 0.25, with an
    # empty band between, so this band isolates the object cleanly at t=0.
    obj_z_max: float = 0.16
    obj_xy_halfspan: tuple[float, float] = (0.35, 0.30)
    # The object has moved by the last frame, so the box that isolates it there must be
    # wider than the one used at t=0 -- same reason the sim port has obj_last_xy_halfspan.
    # Too tight and the registration is handed a truncated target and slides the whole
    # object onto whatever fragment survived, which is exactly what the residual gate below
    # is there to catch.
    obj_last_xy_halfspan: tuple[float, float] = (0.52, 0.36)
    cluster_grid_m: float = 0.03
    # Goal-recovery gates, same spirit as the sim port's.
    max_reg_residual_m: float = 0.030
    max_goal_dist_m: float = 0.45
    max_goal_yaw_deg: float = 60.0
    min_object_points: int = 200
    # Rig calibration
    calib_episodes: int = 8
    recalibrate: bool = False
    verify_assignment: bool = False
    # Bookkeeping
    max_episodes: int | None = None
    num_workers: int = 3
    seed: int = 0
    data_files_size_in_mb: int = 200
    depth_stride: int = 1

    def crop(self) -> dict[str, tuple[float, float]]:
        return {"x": self.crop_x, "y": self.crop_y, "z": self.crop_z}


def _import_pc_common(pc_common: str):
    if pc_common not in sys.path:
        sys.path.insert(0, pc_common)
    import depth_to_cloud
    import pc_ops

    return pc_ops, depth_to_cloud


# --------------------------------------------------------------------------------------
# Source reading (read-only, raw pyarrow)
# --------------------------------------------------------------------------------------


@dataclass
class SourceSnapshot:
    """Frozen view of one v2.1 source dataset, captured once at startup."""

    root: Path
    name: str
    total_episodes: int
    chunks_size: int
    fps: float
    robot_type: str
    episode_lengths: dict[int, int] = field(default_factory=dict)


def freeze_source_snapshot(src_root: Path) -> SourceSnapshot:
    info = json.loads((src_root / "meta" / "info.json").read_text())
    if not str(info.get("codebase_version", "")).startswith("v2"):
        raise ValueError(f"{src_root}: expected a v2.x source, got {info.get('codebase_version')!r}")
    if "observation.depth.cam_high" not in (info.get("depth_features") or {}):
        raise ValueError(f"{src_root}: no depth_features in meta/info.json; nothing to unproject")

    lengths: dict[int, int] = {}
    with open(src_root / "meta" / "episodes.jsonl") as f:
        for line in f:
            row = json.loads(line)
            lengths[int(row["episode_index"])] = int(row["length"])

    return SourceSnapshot(
        root=src_root,
        name=src_root.name if src_root.name != "depth_real" else src_root.parent.name,
        total_episodes=int(info["total_episodes"]),
        chunks_size=int(info["chunks_size"]),
        fps=float(info["fps"]),
        robot_type=str(info.get("robot_type", "?")),
        episode_lengths=lengths,
    )


def episode_paths(snap: SourceSnapshot, ep_idx: int) -> tuple[Path, Path]:
    chunk = f"chunk-{ep_idx // snap.chunks_size:03d}"
    name = f"episode_{ep_idx:06d}.parquet"
    return snap.root / "data" / chunk / name, snap.root / "data_depth" / chunk / name


def _flat_values(col) -> np.ndarray:
    """Fully flatten a (possibly chunked, possibly nested) Arrow list column to 1-D numpy.

    ``ChunkedArray`` has no ``.values``; combining chunks first is what makes the descent
    terminate on the leaf buffer instead of returning one object per row.
    """
    arr = col.combine_chunks() if hasattr(col, "combine_chunks") else col
    while hasattr(arr, "values"):
        arr = arr.values
    return arr.to_numpy(zero_copy_only=False)


def read_state_action(path: Path, expected_len: int) -> dict[str, np.ndarray]:
    """Read the 30 fps state/action table. Returns ``(T, 14)`` float32 arrays."""
    table = pq.read_table(path, columns=["observation.state", "action"])
    t = table.num_rows
    if t != expected_len:
        raise ValueError(f"{path.name}: {t} rows but metadata says {expected_len}")
    return {
        key: _flat_values(table.column(key)).astype(np.float32, copy=False).reshape(t, 14)
        for key in ("observation.state", "action")
    }


def iter_depth_batches(path: Path, keys, batch_size: int = 12):
    """Yield ``(start_row, {cam: (n, 480, 640) uint16})`` batches from a depth parquet.

    Streaming matters: one episode's depth parquet is ~120 MB compressed and ~360 MB
    decompressed, and several workers hold one each.
    """
    cols = [DEPTH_KEYS[k] for k in keys]
    start = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=cols):
        n = batch.num_rows
        out = {}
        for cam in keys:
            out[cam] = _flat_values(batch.column(DEPTH_KEYS[cam])).reshape(n, 480, 640)
        yield start, out
        start += n


def read_depth_frames(path: Path, keys, want: set[int]) -> dict[int, dict]:
    """Read only the listed frame indices out of a depth parquet."""
    out: dict[int, dict] = {}
    if not want:
        return out
    last = max(want)
    for start, batch in iter_depth_batches(path, keys):
        n = next(iter(batch.values())).shape[0]
        for i in range(n):
            if start + i in want:
                out[start + i] = {cam: batch[cam][i].copy() for cam in keys}
        if start + n > last:
            break
    return out


def central_difference(x: np.ndarray, dt: float) -> np.ndarray:
    """d/dt of a ``(T, D)`` signal, central inside, one-sided at the ends.

    ``observation.velocity`` is a real feature of the sim dataset and the merged schema must
    have it. The real recordings never stored joint velocity, so this is the honest
    reconstruction -- and it is a difference of encoder readings, so it is noisier than the
    simulator's analytic velocity. Differentiating at the source's 30 fps and decimating
    afterwards keeps the noise a third of what differentiating the 10 fps signal would give.
    """
    x = np.asarray(x, dtype=np.float64)
    v = np.empty_like(x)
    if x.shape[0] == 1:
        return np.zeros_like(x, dtype=np.float64)
    v[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    v[0] = (x[1] - x[0]) / dt
    v[-1] = (x[-1] - x[-2]) / dt
    return v


# --------------------------------------------------------------------------------------
# Rig calibration
#
# `camera_extrinsics.json` is a hand-eye calibration from 2026-04-29 with a tabletop refine
# that put the table plane at 20.0 mm with a 0.02-0.05 deg residual tilt. On THESE recordings
# only one of the two entries still describes its camera:
#
#   cam_high through the "right" entry -> tabletop tilt 0.95 deg, plane at 21.9 mm.  OK.
#   cam_low  through the "left"  entry -> tabletop tilt 5.55 deg, plane at 97.3 mm.  NOT OK.
#
# The left camera moved between calibration and capture. Rather than bake a stale extrinsic
# into the training clouds, the transform is measured from the recordings themselves:
#   1. fit each camera's tabletop and level it onto z = 0 (fixes tilt + height, 3 DoF);
#   2. register the levelled cam_low cloud onto the levelled cam_high cloud in-plane
#      (yaw + xy, 3 DoF), on the first frame of many episodes, and take the median.
# The rig is rigid, so one transform serves every episode; the spread across episodes is the
# honest uncertainty and is written into the calibration file.
#
# One residual geometry difference survives this and cannot be fixed here: the calibration was
# refined against a tabletop at z = +20 mm, while the IsaacLab scene's tabletop is at z = 0.
# Levelling the real cloud onto z = 0 therefore leaves the real cameras sitting ~22 mm lower
# above their table than the sim cameras sit above theirs. That slightly changes grazing
# angles and self-occlusion; it is far below the second camera's own ~15 mm registration
# uncertainty, so it is recorded rather than corrected.
# --------------------------------------------------------------------------------------


def _occupancy(pts: np.ndarray, lim: float, res: float) -> np.ndarray:
    n = int(2 * lim / res)
    ix = ((pts[:, 0] + lim) / res).astype(np.int64)
    iy = ((pts[:, 1] + lim) / res).astype(np.int64)
    m = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    g = np.zeros((n, n), dtype=np.float32)
    np.add.at(g, (iy[m], ix[m]), 1.0)
    return np.minimum(g, 1.0)


def _best_shift(g_src: np.ndarray, g_dst: np.ndarray, res: float) -> tuple[float, float, float]:
    """Translation maximising 2-D overlap, via FFT cross-correlation (global, not local)."""
    cc = np.fft.irfft2(np.fft.rfft2(g_dst) * np.conj(np.fft.rfft2(g_src)), g_dst.shape)
    k = np.unravel_index(int(np.argmax(cc)), cc.shape)
    n = g_dst.shape[0]
    sy = k[0] - n if k[0] > n // 2 else k[0]
    sx = k[1] - n if k[1] > n // 2 else k[1]
    return sx * res, sy * res, float(cc[k])


def register_xy_yaw(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    yaw_range_deg: float = 180.0,
    yaw_step_deg: float = 3.0,
    icp_iters: int = 40,
    trim: float = 0.7,
    max_src_points: int = 6000,
    seed: int = 0,
    dtc=None,
) -> tuple[float, np.ndarray, float]:
    """Rigid planar transform (yaw, dx, dy) carrying ``src`` onto ``dst``.

    Two stages, because either alone fails here:

      * a **global** stage -- for each candidate yaw, the best translation is read off an
        FFT cross-correlation of top-down occupancy grids. This cannot get stuck in a local
        minimum, which a gradient/ICP-only method does whenever the two clouds are more than
        a few centimetres apart (they routinely are: a push moves the object ~20-40 cm).
      * a **local** stage -- trimmed point-to-point ICP on the 2-D projection, which refines
        the 1 cm grid answer to sub-centimetre. Trimming to the best 70% of correspondences
        is what makes it survive the arm entering the target cloud.

    Both clouds are partial two-view captures, so as the object moves the cameras see
    different faces of it. That is a floor on the achievable residual, not a bug: the sim
    port measured 5.7-8.0 mm for its 2-DoF version and this one measures ~10-20 mm on real
    data, where the second camera's extrinsic is itself only good to ~1.5 cm.

    Returns:
        ``(yaw_rad, (dx, dy) float32, mean trimmed nearest-neighbour residual in metres)``.
    """
    from scipy.spatial import cKDTree

    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)
    if src.shape[0] < 50 or dst.shape[0] < 50:
        return 0.0, np.zeros(2, dtype=np.float32), float("inf")

    rng = np.random.default_rng(seed)
    if src.shape[0] > max_src_points:
        src = src[rng.permutation(src.shape[0])[:max_src_points]]

    lim, res = 1.0, 0.01
    g_dst = _occupancy(dst, lim, res)
    best = (-1.0, 0.0, 0.0, 0.0)
    for deg in np.arange(-yaw_range_deg, yaw_range_deg, yaw_step_deg):
        rot = dtc.yaw_translation(np.radians(float(deg)))
        g_src = _occupancy(dtc.apply_transform(src, rot), lim, res)
        dx, dy, score = _best_shift(g_src, g_dst, res)
        if score > best[0]:
            best = (score, float(deg), dx, dy)

    yaw = float(np.radians(best[1]))
    t = np.array([best[2], best[3]], dtype=np.float64)
    tree = cKDTree(dst)
    resid = float("inf")
    src2 = src[:, :2].astype(np.float64)
    for _ in range(int(icp_iters)):
        moved = dtc.apply_transform(src, dtc.yaw_translation(yaw, t[0], t[1]))
        dist, idx = tree.query(moved, workers=-1)
        keep = np.argsort(dist)[: max(3, int(trim * dist.shape[0]))]
        a = src2[keep]
        b = dst[idx[keep]][:, :2].astype(np.float64)
        ca, cb = a.mean(axis=0), b.mean(axis=0)
        h = (a - ca).T @ (b - cb)
        u, _, vt = np.linalg.svd(h)
        r = vt.T @ u.T
        if np.linalg.det(r) < 0:
            vt[-1] *= -1.0
            r = vt.T @ u.T
        yaw = float(np.arctan2(r[1, 0], r[0, 0]))
        t = cb - r @ ca
        resid = float(dist[keep].mean())
    return yaw, t.astype(np.float32), resid


def calibrate_rig(cfg: PortConfig, snaps: list[SourceSnapshot]) -> dict:
    """Measure the per-camera table-frame extrinsics from the recordings themselves."""
    pc_ops, dtc = _import_pc_common(cfg.pc_common)
    raw = dtc.load_extrinsics_json(cfg.extrinsics_json)

    jobs = []
    for snap in snaps:
        for ep in range(min(cfg.calib_episodes, snap.total_episodes)):
            if ep in snap.episode_lengths:
                jobs.append((snap, ep))
    if not jobs:
        raise RuntimeError("no episodes available to calibrate the rig")

    # --- optionally re-derive which extrinsic belongs to which stream ------------------
    assignment = dict(CAMERA_ASSIGNMENT)
    assignment_report = None
    if cfg.verify_assignment:
        snap, ep = jobs[0]
        _, depth_path = episode_paths(snap, ep)
        frames = read_depth_frames(depth_path, tuple(DEPTH_KEYS), {0})[0]
        assignment_report = {}
        for cam in DEPTH_KEYS:
            scores = {}
            for arm in ("left", "right"):
                world = dtc.depth_to_world(frames[cam], raw[arm])
                sel = world[dtc.workspace_mask(world)]
                normal, centroid, frac = dtc.fit_plane_ransac(sel, seed=1)
                scores[arm] = {
                    "tilt_deg": dtc.tilt_deg(normal),
                    "plane_z_at_origin_m": dtc.plane_z_at_origin(normal, centroid),
                    "inlier_fraction": float(frac),
                }
            assignment_report[cam] = scores
        # A *matching*, not a per-camera argmin. Picking each camera's best extrinsic
        # independently is degenerate on these recordings: only one entry still levels its
        # camera's tabletop, so both streams pick the same one. Scoring whole permutations
        # forces a one-to-one mapping and lets the residual tilt speak for itself.
        cams_ordered = list(DEPTH_KEYS)
        arms = ("left", "right")
        perms = [dict(zip(cams_ordered, arms)), dict(zip(cams_ordered, reversed(arms)))]
        assignment = min(
            perms, key=lambda p: sum(assignment_report[c][a]["tilt_deg"] for c, a in p.items())
        )
        for cam, arm in assignment.items():
            tilt = assignment_report[cam][arm]["tilt_deg"]
            if tilt > 2.0:
                logger.warning(
                    "%s <- extrinsic %r leaves the tabletop tilted %.2f deg (z=%.4f m): that "
                    "calibration entry no longer describes this camera. Its 3 remaining "
                    "degrees of freedom are recovered by registration against the anchor "
                    "camera below; treat the fused cloud accordingly.",
                    cam,
                    arm,
                    tilt,
                    assignment_report[cam][arm]["plane_z_at_origin_m"],
                )
        logger.info("verified camera assignment: %s", assignment)

    # --- tabletop plane per camera, averaged over episodes -----------------------------
    planes: dict[str, list] = {cam: [] for cam in DEPTH_KEYS}
    for snap, ep in jobs:
        _, depth_path = episode_paths(snap, ep)
        frames = read_depth_frames(depth_path, tuple(DEPTH_KEYS), {0})[0]
        for cam in DEPTH_KEYS:
            world = dtc.depth_to_world(frames[cam], raw[assignment[cam]])
            sel = world[dtc.workspace_mask(world)]
            normal, centroid, frac = dtc.fit_plane_ransac(sel, seed=1)
            planes[cam].append(
                np.array(
                    [*normal, dtc.plane_z_at_origin(normal, centroid), frac], dtype=np.float64
                )
            )

    level: dict[str, np.ndarray] = {}
    plane_report: dict[str, dict] = {}
    for cam, rows in planes.items():
        a = np.stack(rows)
        n = a[:, :3].mean(axis=0)
        n = n / np.linalg.norm(n)
        z0 = float(a[:, 3].mean())
        level[cam] = dtc.level_transform(n, np.array([0.0, 0.0, z0]))
        plane_report[cam] = {
            "extrinsic_entry": assignment[cam],
            "serial": raw[assignment[cam]].name,
            "normal": n.tolist(),
            "tilt_deg": dtc.tilt_deg(n),
            "plane_z_at_origin_m": z0,
            "plane_z_std_m": float(a[:, 3].std()),
            "inlier_fraction": float(a[:, 4].mean()),
            "n_frames": int(a.shape[0]),
        }

    cams = {cam: raw[assignment[cam]].with_transform(dtc.compose(level[cam], raw[assignment[cam]].T))
            for cam in DEPTH_KEYS}
    for cam in cams:
        cams[cam].name = cam

    # --- in-plane refinement of every camera against the anchor -----------------------
    # The anchor is the camera whose own calibration already levels the table; every other
    # camera is brought onto it.
    anchor = min(plane_report, key=lambda c: plane_report[c]["tilt_deg"])
    refine_report: dict[str, dict] = {anchor: {"anchor": True}}
    for cam in DEPTH_KEYS:
        if cam == anchor:
            continue
        rows = []
        for snap, ep in jobs:
            _, depth_path = episode_paths(snap, ep)
            frames = read_depth_frames(depth_path, tuple(DEPTH_KEYS), {0})[0]
            a = dtc.depth_to_world(frames[anchor], cams[anchor])
            b = dtc.depth_to_world(frames[cam], cams[cam])
            a = a[(a[:, 2] > 0.02) & (a[:, 2] < 0.5) & (np.abs(a[:, 0]) < 0.8) & (np.abs(a[:, 1]) < 0.8)]
            b = b[(b[:, 2] > 0.02) & (b[:, 2] < 0.5) & (np.abs(b[:, 0]) < 0.8) & (np.abs(b[:, 1]) < 0.8)]
            yaw, t, resid = register_xy_yaw(b, a, max_src_points=8000, dtc=dtc)
            rows.append([np.degrees(yaw), t[0], t[1], resid])
        a = np.stack(rows)
        med = np.median(a, axis=0)
        mad = np.median(np.abs(a - med), axis=0)
        cams[cam] = cams[cam].with_transform(
            dtc.compose(dtc.yaw_translation(np.radians(med[0]), med[1], med[2]), cams[cam].T)
        )
        refine_report[cam] = {
            "anchor": False,
            "against": anchor,
            "yaw_deg": float(med[0]),
            "tx_m": float(med[1]),
            "ty_m": float(med[2]),
            "icp_residual_m": float(med[3]),
            "mad_yaw_deg": float(mad[0]),
            "mad_tx_m": float(mad[1]),
            "mad_ty_m": float(mad[2]),
            "n_episodes": int(a.shape[0]),
            "per_episode": a.tolist(),
        }

    # --- depth-unit assertion ----------------------------------------------------------
    snap, ep = jobs[0]
    _, depth_path = episode_paths(snap, ep)
    frames = read_depth_frames(depth_path, tuple(DEPTH_KEYS), {0})[0]
    depth_checks = {
        cam: dtc.verify_depth_scale(frames[cam], cams[cam], expected_table_z=0.0, tol=0.01)
        for cam in DEPTH_KEYS
    }
    for cam, chk in depth_checks.items():
        if not chk.get("ok"):
            raise RuntimeError(
                f"{cam}: tabletop landed at {chk.get('plane_z_at_origin_m')} m after calibration, "
                f"not 0. Depth scale or extrinsic is wrong: {chk}"
            )

    calib = dtc.RigCalibration(
        cameras=cams,
        provenance={
            "extrinsics_json": str(cfg.extrinsics_json),
            "assignment": assignment,
            "assignment_search": assignment_report,
            "table_planes": plane_report,
            "in_plane_refinement": refine_report,
            "depth_scale_m": dtc.DEPTH_SCALE_M,
            "depth_range_m": list(dtc.DEFAULT_DEPTH_RANGE),
            "post_calibration_check": depth_checks,
            "sources": [str(s.root) for s in snaps],
            "note": (
                "Transforms map camera-optical -> table frame (tabletop z=0, +Z up), the same "
                "frame pc_ops documents. NO calibworld_to_mjworld 90 deg rotation is applied."
            ),
        },
    )
    return calib.to_dict()


# --------------------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------------------


def isolate_real_object(
    pc: np.ndarray, cfg: PortConfig, pc_ops, *, halfspan: tuple[float, float] | None = None
) -> np.ndarray:
    """Object-only points from a real table-frame cloud.

    Real clouds are the whole scene: tabletop, object, arms, the table's aluminium frame and
    whatever is behind it. Measured on these recordings the object lives in a clean height
    band -- points from 15 mm to ~140 mm above the table, with nothing at all between 140 mm
    and 250 mm, above which sit the parked arms and the overhead mount. Cutting that band
    and keeping the largest connected component on a coarse xy grid isolates the object
    without any pose or mesh, which is all that is available here.
    """
    p = np.asarray(pc, dtype=np.float32)
    hx, hy = halfspan if halfspan is not None else cfg.obj_xy_halfspan
    m = (
        (p[:, 2] > max(cfg.table_cut_m, 0.0))
        & (p[:, 2] < cfg.obj_z_max)
        & (np.abs(p[:, 0]) < hx)
        & (np.abs(p[:, 1]) < hy)
    )
    return pc_ops.largest_xy_cluster(p[m], grid=cfg.cluster_grid_m)


def build_episode_buffer(job: tuple, cfg: PortConfig, calib: dict) -> dict:
    """Read, unproject, preprocess and goal-label one source episode. Runs in a worker."""
    src_root_str, src_name, ep_idx, expected_len, chunks_size, src_fps = job
    pc_ops, dtc = _import_pc_common(cfg.pc_common)
    rig = dtc.RigCalibration.from_dict(calib)
    cams = {c: rig.cameras[c] for c in cfg.cameras}

    diag: dict = {"source": src_name, "src_episode_index": ep_idx}
    src_root = Path(src_root_str)
    chunk = f"chunk-{ep_idx // chunks_size:03d}"
    name = f"episode_{ep_idx:06d}.parquet"
    data_path = src_root / "data" / chunk / name
    depth_path = src_root / "data_depth" / chunk / name

    try:
        if not data_path.exists() or not depth_path.exists():
            return {"_skip": "missing_parquet", **diag}
        n_data = pq.read_metadata(data_path).num_rows
        n_depth = pq.read_metadata(depth_path).num_rows
        if n_data != expected_len:
            return {"_skip": f"row_count_mismatch:data:{n_data}!={expected_len}", **diag}
        if n_depth != expected_len:
            return {"_skip": f"row_count_mismatch:depth:{n_depth}!={expected_len}", **diag}
        arrays = read_state_action(data_path, expected_len)
    except Exception as exc:  # noqa: BLE001 - one bad episode must not kill the run
        return {"_skip": f"read_error:{type(exc).__name__}:{exc}", **diag}

    keep = np.arange(0, expected_len, cfg.decimate, dtype=np.int64)
    t = int(keep.shape[0])
    if t < 4:
        return {"_skip": f"too_short_after_decimation:{t}", **diag}
    keep_set = set(int(i) for i in keep)
    rng = np.random.default_rng(cfg.seed + abs(hash((src_name, ep_idx))) % 10_000)

    crop = cfg.crop()
    obs_pc = np.empty((t, cfg.num_points, 3), dtype=np.float32)
    n_raw = np.empty(t, dtype=np.int64)
    n_after_cut = np.empty(t, dtype=np.int64)
    n_in_crop = np.empty(t, dtype=np.int64)
    # Points inside the crop *before* the tabletop is removed. This is the honest
    # denominator for "how much of what the policy would see is tabletop", and it is the
    # number that quantifies the real-vs-sim content mismatch: the sim clouds contain no
    # table at all, so anything this measures has no counterpart on the sim side.
    n_in_crop_uncut = np.empty(t, dtype=np.int64)
    first_cloud = None
    last_cloud = None

    pos = 0
    try:
        for start, batch in iter_depth_batches(depth_path, tuple(cfg.cameras)):
            n = next(iter(batch.values())).shape[0]
            for i in range(n):
                row = start + i
                if row not in keep_set:
                    continue
                cloud = dtc.fuse(
                    [
                        dtc.depth_to_world(batch[cam][i], cams[cam], stride=cfg.depth_stride)
                        for cam in cfg.cameras
                    ]
                )
                n_raw[pos] = cloud.shape[0]
                n_in_crop_uncut[pos] = int(pc_ops.crop_mask(cloud, crop).sum())
                if cfg.table_cut_m >= 0.0:
                    cloud = dtc.drop_table(cloud, cfg.table_cut_m)
                n_after_cut[pos] = cloud.shape[0]
                obs_pc[pos], n_in_crop[pos] = pc_ops.crop_and_resample(
                    cloud, crop, cfg.num_points, rng
                )
                if pos == 0:
                    first_cloud = cloud
                last_cloud = cloud
                pos += 1
    except Exception as exc:  # noqa: BLE001
        return {"_skip": f"depth_error:{type(exc).__name__}:{exc}", **diag}
    if pos != t:
        return {"_skip": f"depth_frames_missing:{pos}!={t}", **diag}

    diag.update(
        n_frames=t,
        raw_points_mean=float(n_raw.mean()),
        after_table_cut_mean=float(n_after_cut.mean()),
        in_crop_uncut_mean=float(n_in_crop_uncut.mean()),
        in_crop_mean=float(n_in_crop.mean()),
        in_crop_min=int(n_in_crop.min()),
        frames_below_num_points=int((n_in_crop < cfg.num_points).sum()),
        # Fraction of the cropped cloud that is tabletop, i.e. the share of the observation
        # that has no counterpart anywhere in the simulated dataset.
        tabletop_fraction_of_crop=float(
            1.0 - n_in_crop.mean() / max(n_in_crop_uncut.mean(), 1.0)
        ),
    )

    # ---- goal recovery ---------------------------------------------------------------
    # No object pose exists for real data, so the goal is measured from the clouds: isolate
    # the object at the first kept frame and at the last, and register the first onto the
    # last. Every recorded episode is a completed human demonstration, so the object's final
    # pose IS the goal the demonstrator was aiming at -- to within however far they stopped
    # short, which is unobservable here.
    obj_t0 = isolate_real_object(first_cloud, cfg, pc_ops)
    if obj_t0.shape[0] < cfg.min_object_points:
        return {"_skip": f"few_t0_object_points:{obj_t0.shape[0]}", **diag}

    # Two candidate target clouds, and the registration itself picks between them. A tight
    # isolation box can truncate an object that travelled far; a wide one can swallow an arm
    # that ended up next to it. Which failure occurs depends on the episode, so rather than
    # guess, register against both and keep whichever fits better -- the residual is exactly
    # the quantity that distinguishes "found the object" from "found something else".
    candidates = []
    for tag, halfspan in (
        ("tight", cfg.obj_xy_halfspan),
        ("wide", cfg.obj_last_xy_halfspan),
    ):
        obj = isolate_real_object(last_cloud, cfg, pc_ops, halfspan=halfspan)
        if obj.shape[0] < cfg.min_object_points:
            continue
        yaw_c, delta_c, resid_c = register_xy_yaw(
            obj_t0,
            obj,
            yaw_range_deg=cfg.max_goal_yaw_deg,
            seed=cfg.seed + ep_idx,
            dtc=dtc,
        )
        candidates.append((resid_c, tag, obj, yaw_c, delta_c))
    if not candidates:
        return {"_skip": "few_last_object_points", **diag}
    residual, isolation_used, obj_tn, yaw, delta = min(candidates, key=lambda c: c[0])
    diag["goal_isolation_box"] = isolation_used
    diag["goal_candidate_residuals_m"] = {c[1]: c[0] for c in candidates}
    goal_dist = float(np.hypot(delta[0], delta[1]))
    yaw_deg = float(np.degrees(yaw))
    # The object cloud's own centroid is the only origin available for its "pose"; there is
    # no body frame to refer to. That is fine because the same point is used as the source
    # pose and as the rotation centre of `transform_points`, so the goal cloud it produces is
    # exactly the registered cloud. It does mean `observation.goal_pose[:3]` is a centroid,
    # not a body origin -- consistent within this dataset, and offset by an unknown constant
    # from the sim's body-origin convention. See the README note in the module docstring.
    centroid = obj_t0.mean(axis=0).astype(np.float64)
    object_top = float(obj_t0[:, 2].max())
    pose0 = np.array(
        [centroid[0], centroid[1], object_top, 1.0, 0.0, 0.0, 0.0], dtype=np.float32
    )
    half = 0.5 * yaw
    pose_goal = np.array(
        [
            centroid[0] + delta[0],
            centroid[1] + delta[1],
            object_top,
            np.cos(half),
            0.0,
            0.0,
            np.sin(half),
        ],
        dtype=np.float32,
    )
    diag.update(
        n_obj_t0=int(obj_t0.shape[0]),
        n_obj_last=int(obj_tn.shape[0]),
        object_top_m=object_top,
        object_centroid=centroid.tolist(),
        goal_dx=float(delta[0]),
        goal_dy=float(delta[1]),
        goal_dist_m=goal_dist,
        goal_yaw_deg=yaw_deg,
        reg_residual_m=residual,
        goal_source="registration_3dof",
    )
    if not np.isfinite(residual) or residual > cfg.max_reg_residual_m:
        return {"_skip": f"reg_residual:{residual:.4f}", **diag}
    if goal_dist > cfg.max_goal_dist_m:
        return {"_skip": f"goal_dist:{goal_dist:.4f}", **diag}

    goal_pts = pc_ops.transform_points(obj_t0, pose0[:3], pose0[3:], pose_goal[:3], pose_goal[3:])
    goal_pc = pc_ops.resample_to(goal_pts, cfg.goal_num_points, rng)
    goal_vec9 = pc_ops.pose_to_vec9(pose_goal[:3], pose_goal[3:])
    onehot = pc_ops.task_onehot(TARGET_TASK)

    state30 = arrays["observation.state"]
    vel30 = central_difference(state30, 1.0 / float(src_fps))
    buf: dict = {
        "size": t,
        "task": [TARGET_TASK] * t,
        "timestamp": (np.arange(t, dtype=np.float32) / float(TARGET_FPS)).astype(np.float32),
        "frame_index": np.arange(t, dtype=np.int64),
        "index": np.zeros(t, dtype=np.int64),
        "task_index": np.zeros(t, dtype=np.int64),
        "observation.state": state30[keep],
        "observation.velocity": vel30[keep].astype(np.float32),
        "observation.point_cloud": obs_pc,
        "observation.goal_point_cloud": np.repeat(goal_pc[None], t, axis=0),
        "observation.goal_pose": np.tile(goal_vec9, (t, 1)).astype(np.float32),
        "observation.task_onehot": np.tile(onehot, (t, 1)).astype(np.float32),
        # Label only, and there is no label here. `pose0` is the measured initial object
        # centroid: a per-episode constant that is inside the sim's value range, so it cannot
        # stretch the merged dataset's MIN_MAX statistics the way a sentinel like -1 would.
        # `pose_valid = 0` is what actually tells the auxiliary loss to ignore it.
        "observation.object_pose": np.tile(pose0, (t, 1)).astype(np.float32),
        "observation.pose_valid": np.zeros((t, 1), dtype=np.float32),
        "action": arrays["action"][keep],
    }
    return {"_buffer": buf, **diag}


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def build_features(cfg: PortConfig) -> dict:
    pc_ops, _ = _import_pc_common(cfg.pc_common)
    return {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": STATE_JOINT_NAMES},
        "observation.point_cloud": {
            "dtype": "float32",
            "shape": (cfg.num_points, 3),
            "names": ["x", "y", "z"],
        },
        "observation.goal_point_cloud": {
            "dtype": "float32",
            "shape": (cfg.goal_num_points, 3),
            "names": ["x", "y", "z"],
        },
        "observation.goal_pose": {
            "dtype": "float32",
            "shape": (9,),
            "names": ["x", "y", "z", "r00", "r10", "r20", "r01", "r11", "r21"],
        },
        "observation.task_onehot": {
            "dtype": "float32",
            "shape": (len(pc_ops.TASK_NAMES),),
            "names": list(pc_ops.TASK_NAMES),
        },
        "observation.object_pose": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "qw", "qx", "qy", "qz"],
        },
        "observation.pose_valid": {"dtype": "float32", "shape": (1,), "names": ["valid"]},
        "action": {"dtype": "float32", "shape": (14,), "names": STATE_JOINT_NAMES},
        "observation.velocity": {"dtype": "float32", "shape": (14,), "names": STATE_JOINT_NAMES},
    }


def main(cfg: PortConfig) -> None:
    srcs = [p.resolve() for p in cfg.src_roots]
    dst = cfg.dst_root.resolve()
    # Safety rails. The source directories are irreplaceable recordings; nothing here may
    # write into or near them, and no LeRobotDataset is ever rooted at one (LeRobotDataset
    # writes meta/ on open, and the v2.1->v3.0 converter would move the directory outright).
    for src in srcs:
        if dst == src or src in dst.parents or dst in src.parents:
            raise ValueError(f"dst_root must be outside every src_root (src={src}, dst={dst})")
        if not (src / "meta" / "info.json").exists():
            raise FileNotFoundError(f"{src} does not look like a LeRobot dataset")

    pc_ops, dtc = _import_pc_common(cfg.pc_common)

    snaps = [freeze_source_snapshot(s) for s in srcs]
    for snap in snaps:
        logger.info(
            "source %s: %d episodes @ %.1f fps, robot_type=%s -> decimate x%d to %d fps",
            snap.name,
            snap.total_episodes,
            snap.fps,
            snap.robot_type,
            cfg.decimate,
            TARGET_FPS,
        )
        if abs(snap.fps / cfg.decimate - TARGET_FPS) > 1e-6:
            raise ValueError(
                f"{snap.name}: {snap.fps} fps / {cfg.decimate} != {TARGET_FPS} fps; "
                "the merged dataset requires an exact match"
            )

    dst.parent.mkdir(parents=True, exist_ok=True)
    calib_path = dst.parent / f"{dst.name}_rig_calibration.json"
    if calib_path.exists() and not cfg.recalibrate:
        calib = json.loads(calib_path.read_text())
        logger.info("loaded rig calibration from %s", calib_path)
    else:
        logger.info("calibrating the rig from %d episodes per source ...", cfg.calib_episodes)
        calib = calibrate_rig(cfg, snaps)
        calib_path.write_text(json.dumps(calib, indent=2, default=_json_default))
        logger.info("wrote rig calibration to %s", calib_path)
    for cam, rep in calib["provenance"]["table_planes"].items():
        logger.info(
            "  %s <- extrinsic %r (serial %s): tabletop tilt %.2f deg at z=%.4f m",
            cam,
            rep["extrinsic_entry"],
            rep["serial"],
            rep["tilt_deg"],
            rep["plane_z_at_origin_m"],
        )
    for cam, rep in calib["provenance"]["in_plane_refinement"].items():
        if not rep.get("anchor"):
            logger.info(
                "  %s refined onto %s: yaw %+.2f deg, t=(%+.3f, %+.3f) m, "
                "ICP residual %.1f mm, MAD (%.2f deg, %.1f mm, %.1f mm) over %d episodes",
                cam,
                rep["against"],
                rep["yaw_deg"],
                rep["tx_m"],
                rep["ty_m"],
                1000 * rep["icp_residual_m"],
                rep["mad_yaw_deg"],
                1000 * rep["mad_tx_m"],
                1000 * rep["mad_ty_m"],
                rep["n_episodes"],
            )

    features = build_features(cfg)
    if (dst / "meta" / "info.json").exists():
        ds = LeRobotDataset.resume(cfg.repo_id, root=dst)
        logger.info("resuming: %d episodes already written", ds.meta.total_episodes)
    else:
        ds = LeRobotDataset.create(
            cfg.repo_id,
            fps=TARGET_FPS,
            features=features,
            root=dst,
            robot_type=TARGET_ROBOT_TYPE,
            use_videos=False,
            data_files_size_in_mb=cfg.data_files_size_in_mb,
        )

    diag_path = dst / "meta" / "port_diagnostics.jsonl"
    map_path = dst / "meta" / "episode_sources.jsonl"
    done: set = set()
    if map_path.exists():
        with open(map_path) as f:
            for line in f:
                row = json.loads(line)
                done.add((row["source"], row["src_episode_index"]))

    jobs = []
    for snap in snaps:
        n_eps = (
            snap.total_episodes
            if cfg.max_episodes is None
            else min(snap.total_episodes, cfg.max_episodes)
        )
        for ep in range(n_eps):
            if ep in snap.episode_lengths and (snap.name, ep) not in done:
                jobs.append(
                    (
                        str(snap.root),
                        snap.name,
                        ep,
                        snap.episode_lengths[ep],
                        snap.chunks_size,
                        snap.fps,
                    )
                )
    logger.info("porting %d episodes (%d already done)", len(jobs), len(done))

    t_start = time.time()
    n_written = n_skipped = 0
    worker = partial(build_episode_buffer, cfg=cfg, calib=calib)
    with (
        open(diag_path, "a") as diag_f,
        open(map_path, "a") as map_f,
        ProcessPoolExecutor(cfg.num_workers) as ex,
    ):
        for result in ex.map(worker, jobs, chunksize=1):
            tag = (result["source"], result["src_episode_index"])
            if "_buffer" not in result:
                n_skipped += 1
                diag_f.write(json.dumps(result, default=_json_default) + "\n")
                diag_f.flush()
                logger.warning("skip %s ep %d: %s", tag[0], tag[1], result.get("_skip"))
                continue
            buf = result.pop("_buffer")
            new_ep_idx = int(ds.meta.total_episodes)
            buf["episode_index"] = new_ep_idx
            ds.save_episode(episode_data=buf)
            map_f.write(
                json.dumps(
                    {
                        "episode_index": new_ep_idx,
                        "source": tag[0],
                        "src_episode_index": tag[1],
                        "goal_dx": result["goal_dx"],
                        "goal_dy": result["goal_dy"],
                        "goal_dist_m": result["goal_dist_m"],
                        "goal_yaw_deg": result["goal_yaw_deg"],
                        "reg_residual_m": result["reg_residual_m"],
                        "n_frames": result["n_frames"],
                    },
                    default=_json_default,
                )
                + "\n"
            )
            map_f.flush()
            diag_f.write(json.dumps(result, default=_json_default) + "\n")
            diag_f.flush()
            n_written += 1
            elapsed = time.time() - t_start
            logger.info(
                "%d/%d written (%d skipped) | %.1f s/ep | goal %.3f m %+.1f deg, resid %.1f mm",
                n_written,
                len(jobs),
                n_skipped,
                elapsed / max(n_written, 1),
                result["goal_dist_m"],
                result["goal_yaw_deg"],
                1000 * result["reg_residual_m"],
            )

    ds.finalize()
    describe = pc_ops.describe(cfg.crop(), cfg.num_points, cfg.goal_num_points)
    describe.update(
        source="real_world",
        table_cut_m=cfg.table_cut_m,
        decimate=cfg.decimate,
        cameras=list(cfg.cameras),
        depth_scale_m=dtc.DEPTH_SCALE_M,
        depth_range_m=list(dtc.DEFAULT_DEPTH_RANGE),
        rig_calibration=str(calib_path),
        goal_source="registration_3dof",
        pose_valid="always 0: real recordings have no object-pose ground truth",
    )
    (dst / "meta" / "preprocessing.json").write_text(json.dumps(describe, indent=2))

    # Row-count verification: what the source metadata promised, decimated, must equal what
    # the destination holds. A silent short-write here would only surface as a training bug.
    expected_rows = 0
    with open(map_path) as f:
        for line in f:
            expected_rows += int(json.loads(line)["n_frames"])
    info = json.loads((dst / "meta" / "info.json").read_text())
    if int(info["total_frames"]) != expected_rows:
        raise RuntimeError(
            f"destination holds {info['total_frames']} frames, expected {expected_rows}"
        )
    logger.info(
        "done: %d episodes / %d frames written, %d skipped, %.1f min",
        n_written,
        expected_rows,
        n_skipped,
        (time.time() - t_start) / 60,
    )


def parse_args() -> PortConfig:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--src_root", type=Path, action="append", required=True)
    p.add_argument("--dst_root", type=Path, required=True)
    p.add_argument("--repo_id", type=str, default="local/push_real_v1")
    p.add_argument("--num_points", type=int, default=1024)
    p.add_argument("--goal_num_points", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=3)
    p.add_argument("--max_episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--decimate", type=int, default=3)
    p.add_argument(
        "--cameras",
        type=str,
        default="cam_high,cam_low",
        help="comma list; 'cam_high' alone uses only the camera whose calibration is verified",
    )
    p.add_argument("--table_cut_m", type=float, default=0.015)
    p.add_argument("--calib_episodes", type=int, default=8)
    p.add_argument("--recalibrate", action="store_true")
    p.add_argument("--verify_assignment", action="store_true")
    p.add_argument("--depth_stride", type=int, default=1)
    p.add_argument("--extrinsics_json", type=str, default=DEFAULT_EXTRINSICS)
    p.add_argument("--pc_common", type=str, default=DEFAULT_PC_COMMON)
    a = p.parse_args()
    return PortConfig(
        src_roots=list(a.src_root),
        dst_root=a.dst_root,
        repo_id=a.repo_id,
        num_points=a.num_points,
        goal_num_points=a.goal_num_points,
        num_workers=a.num_workers,
        max_episodes=a.max_episodes,
        seed=a.seed,
        decimate=a.decimate,
        cameras=tuple(c.strip() for c in a.cameras.split(",") if c.strip()),
        table_cut_m=a.table_cut_m,
        calib_episodes=a.calib_episodes,
        recalibrate=a.recalibrate,
        verify_assignment=a.verify_assignment,
        depth_stride=a.depth_stride,
        extrinsics_json=a.extrinsics_json,
        pc_common=a.pc_common,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(parse_args())
