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

"""Port the IsaacLab bimanual push point-cloud dataset from LeRobot v2.1 to v3.0.

The source dataset (``collect_push_lerobot.py``) stores a 4096-point cloud per frame but records
**no goal**, and the goal is the only thing that varies between episodes of the same object (the
object spawn pose is fixed). Without it the task is not learnable. This script therefore does
three things at once:

1. **Recovers the goal.** Only successful rollouts are saved, so the object ends at the goal
   within the recorded ``pos_err_m``. We isolate the object in the first and last frames and
   register them with a 2-D translation search; that translation is the goal displacement.
2. **Preprocesses the clouds.** Crops to the table workspace and resamples to a fixed point
   count via :mod:`pc_ops` -- the same module the IsaacLab eval client uses, so training and
   evaluation preprocessing cannot drift apart.
3. **Writes a fresh v3.0 dataset.** The in-place converter
   (``convert_dataset_v21_to_v30.py``) is deliberately NOT used: it ``shutil.move``s the source
   directory, and it cannot change the point count.

The source directory is opened strictly read-only.

Usage::

    python examples/port_datasets/port_isaaclab_pointcloud_push.py \
        --src_root /home/samsung/3D_Bimanual_repo/scripts_isaaclab/irregular/dataset_collection/push_data \
        --dst_root /home/samsung/data/push_pc1024 \
        --repo_id local/push_pc1024 \
        --num_points 1024 --num_workers 6
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
from scipy.spatial import cKDTree
import pyarrow.parquet as pq

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# `pc_ops` is the shared crop/resample contract, and it lives in the IsaacLab repo so the eval
# client (Python 3.11) can import the exact same file. Injected on sys.path rather than copied.
DEFAULT_PC_COMMON = "/home/samsung/3D_Bimanual_repo/scripts_isaaclab/irregular/pc_common"

logger = logging.getLogger(__name__)

STATE_JOINT_NAMES = (
    [f"left_joint_{i}" for i in range(6)]
    + ["left_gripper"]
    + [f"right_joint_{i}" for i in range(6)]
    + ["right_gripper"]
)

ORIENTATION_PREFIXES = (
    "neg_x_down__",
    "pos_x_down__",
    "neg_y_down__",
    "pos_y_down__",
    "neg_z_down__",
    "pos_z_down__",
)


def _json_default(obj):
    """Let json.dumps handle numpy scalars/arrays that leak into the diagnostics records."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def base_mesh_name(object_path: str) -> str:
    """Strip the orientation prefix so every pose variant of a mesh shares one grouping key.

    ``irregular_objects/rotated/pos_x_down__Foo`` and ``irregular_objects/round/Foo`` are the
    same physical mesh. Splitting train/test on the raw name would leak a mesh across the split.
    """
    name = object_path.split("/")[-1]
    for prefix in ORIENTATION_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


@dataclass
class PortConfig:
    src_root: Path
    dst_root: Path
    repo_id: str = "local/push_pc1024"
    num_points: int = 1024
    goal_num_points: int = 512
    crop_x: tuple[float, float] = (-0.61, 0.61)
    crop_y: tuple[float, float] = (-0.38, 0.38)
    crop_z: tuple[float, float] = (-0.03, 0.50)
    keep_velocity: bool = True
    # "commanded" (default) puts the goal orientation at the object's START orientation, matching
    # both what the task asks on a push and what the evaluator commands. "achieved" uses the
    # recorded final orientation, which bakes the demonstrator's residual error into the goal.
    goal_orientation: str = "commanded"
    max_episodes: int | None = None
    num_workers: int = 6
    seed: int = 0
    data_files_size_in_mb: int = 200
    pc_common: str = DEFAULT_PC_COMMON
    # Goal-recovery gates. Registration residual is a mean nearest-neighbour distance; measured
    # 5.7-8.0 mm on healthy episodes (that is just the cloud's own point spacing). The
    # displacement bound comes from goal_push_dist_m = 0.20 plus slack for the demo's own error.
    max_reg_residual_m: float = 0.015
    max_goal_dist_m: float = 0.22
    # Object isolation. The spawn pose is fixed, so a generous box plus a connected-component
    # pass separates the object from the two static arm bases (which sit at |x| > 0.40).
    obj_xy_halfspan: float = 0.30
    obj_last_xy_halfspan: float = 0.38
    obj_top_margin_m: float = 0.03
    cluster_grid_m: float = 0.03
    # Object-only observation, matching what a segmentation-based pipeline delivers on
    # hardware: segment the object in RGB, project onto the registered depth, keep those points.
    # Here the mask is exact (mesh proximity) rather than predicted. Threshold matches the
    # collector's own `--obj_seg_thresh`, so the retained set is what the simulator calls the
    # object. The policy still knows its arm configuration from `observation.state`.
    drop_arm_points: bool = True
    obj_seg_thresh_m: float = 0.015
    # "absolute": store the recorded joint targets unchanged. "delta": store
    # action[t] - state[t] -- the commanded target relative to the measured joints at that
    # frame. At execution the consumer adds the live joint position back (command = jp + d),
    # so the semantic is "move by this much from wherever you are". Checkpoints trained on
    # deltas carry action_space="delta_joint" and the evaluator refuses to run them as
    # absolute -- a silent mismatch would command near-zero motion and score ~0.
    action_mode: str = "absolute"

    def crop(self) -> dict[str, tuple[float, float]]:
        return {"x": self.crop_x, "y": self.crop_y, "z": self.crop_z}


# --------------------------------------------------------------------------------------
# Source reading (read-only, raw pyarrow)
# --------------------------------------------------------------------------------------


@dataclass
class SourceSnapshot:
    """Frozen view of the source dataset, captured once at startup.

    The collector flushes ``meta/`` only when a chunk finalises, so during a live run the data
    directory can hold episodes that ``meta/`` does not yet describe. Freezing the metadata and
    only porting what it lists is what makes this safe to run against a dataset still being
    written, and reproducible if re-run.
    """

    total_episodes: int
    chunks_size: int
    fps: float
    episode_lengths: dict[int, int]
    tasks: dict[int, str]
    objects: dict[int, dict]


def freeze_source_snapshot(src_root: Path) -> SourceSnapshot:
    info = json.loads((src_root / "meta" / "info.json").read_text())
    if not str(info.get("codebase_version", "")).startswith("v2"):
        raise ValueError(f"Expected a v2.x source dataset, got {info.get('codebase_version')!r}")

    lengths: dict[int, int] = {}
    with open(src_root / "meta" / "episodes.jsonl") as f:
        for line in f:
            row = json.loads(line)
            lengths[row["episode_index"]] = row["length"]

    tasks: dict[int, str] = {}
    with open(src_root / "meta" / "tasks.jsonl") as f:
        for line in f:
            row = json.loads(line)
            tasks[row["task_index"]] = row["task"]

    objects: dict[int, dict] = {}
    obj_path = src_root / "meta" / "episode_objects.jsonl"
    if obj_path.exists():
        with open(obj_path) as f:
            for line in f:
                row = json.loads(line)
                objects[row["episode_index"]] = row

    return SourceSnapshot(
        total_episodes=int(info["total_episodes"]),
        chunks_size=int(info["chunks_size"]),
        fps=float(info["fps"]),
        episode_lengths=lengths,
        tasks=tasks,
        objects=objects,
    )


def episode_parquet_path(src_root: Path, ep_idx: int, chunks_size: int) -> Path:
    return src_root / "data" / f"chunk-{ep_idx // chunks_size:03d}" / f"episode_{ep_idx:06d}.parquet"


def _flat_values(col) -> np.ndarray:
    """Fully flatten a (possibly nested) Arrow list column to a 1-D numpy buffer.

    Going through the flat values buffer instead of ``to_pylist()`` is a ~250x difference on
    this data (2 ms vs 550 ms per episode), which is the whole reason this port is fast enough
    to run repeatedly.
    """
    arr = col.combine_chunks()
    while hasattr(arr, "values"):
        arr = arr.values
    return arr.to_numpy(zero_copy_only=False)


def _sanitise_poses(pose_seq: np.ndarray, pc_ops) -> np.ndarray:
    """Replace implausible poses with the most recent plausible one (forward fill).

    Keeps the array free of outliers that would otherwise define the dataset's MIN_MAX range --
    a single object that fell through the floor stretched the goal-pose z span from ~1 cm to
    5.2 m, which normalised every real value to a constant.
    """
    ok = pc_ops.pose_is_plausible(pose_seq)
    if ok.all():
        return pose_seq
    out = pose_seq.copy()
    last = None
    for i in range(len(out)):
        if ok[i]:
            last = out[i].copy()
        elif last is not None:
            out[i] = last
    if last is None:
        return out
    # Any leading implausible frames take the first plausible pose.
    first = int(np.nonzero(ok)[0][0])
    out[:first] = pose_seq[first]
    return out


def load_episode_arrays(path: Path, expected_len: int, keep_velocity: bool) -> dict[str, np.ndarray]:
    """Read one source episode. Returns state/velocity/action (T,14) and point_cloud (T,N,3).

    ``observation.object_pose`` (T,7) is read when the collection recorded it. Collections made
    before the collector was patched do not have it, and fall back to registration-based goal
    recovery -- see `port_episode`.
    """
    cols = ["observation.state", "observation.point_cloud", "action"]
    if keep_velocity:
        cols.append("observation.velocity")
    have_pose = "observation.object_pose" in pq.read_schema(path).names
    if have_pose:
        cols.append("observation.object_pose")
    table = pq.read_table(path, columns=cols)
    t = table.num_rows
    if t != expected_len:
        raise ValueError(f"{path.name}: {t} rows but metadata says {expected_len}")

    out: dict[str, np.ndarray] = {}
    for key in ("observation.state", "action", *(("observation.velocity",) if keep_velocity else ())):
        out[key] = _flat_values(table.column(key)).astype(np.float32, copy=False).reshape(t, 14)
    pcd = _flat_values(table.column("observation.point_cloud")).astype(np.float32, copy=False)
    out["observation.point_cloud"] = pcd.reshape(t, -1, 3)
    if have_pose:
        out["observation.object_pose"] = (
            _flat_values(table.column("observation.object_pose"))
            .astype(np.float32, copy=False).reshape(t, 7)
        )
    return out


# --------------------------------------------------------------------------------------
# Goal recovery
# --------------------------------------------------------------------------------------


def register_xy(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    steps: tuple[float, ...] = (0.02, 0.005, 0.001),
    half_width: int = 6,
    max_src_points: int = 500,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Estimate the 2-D translation taking ``src`` onto ``dst`` by coarse-to-fine search.

    Goal orientation is identity and the object's initial yaw is fixed, so the transform is a
    pure translation -- a 2-DoF problem, which is why a small grid search beats full ICP here
    and cannot fall into a rotational local minimum.

    Deliberately NOT a centroid difference: the two frames are partial 2-view captures of the
    object from different positions, so they see different faces. Measured, the centroid is off
    by 11-38 mm while this search lands within the cloud's own ~6-8 mm point spacing.

    Returns:
        ``((dx, dy) float32, mean nearest-neighbour residual in metres)``.
    """
    from scipy.spatial import cKDTree

    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)
    if src.shape[0] < 20 or dst.shape[0] < 20:
        return np.zeros(2, dtype=np.float32), float("inf")

    if src.shape[0] > max_src_points:
        rng = np.random.default_rng(seed)
        src = src[rng.permutation(src.shape[0])[:max_src_points]]

    tree = cKDTree(dst)
    best = (dst[:, :2].mean(0) - src[:, :2].mean(0)).astype(np.float64)
    best_score = float("inf")
    offsets = np.arange(-half_width, half_width + 1)
    for step in steps:
        cands = np.stack(
            np.meshgrid(best[0] + offsets * step, best[1] + offsets * step, indexing="ij"), axis=-1
        ).reshape(-1, 2)
        scores = np.empty(cands.shape[0])
        for i, (dx, dy) in enumerate(cands):
            shifted = src + np.array([dx, dy, 0.0], dtype=np.float32)
            scores[i] = tree.query(shifted, workers=1)[0].mean()
        k = int(np.argmin(scores))
        best = cands[k]
        best_score = float(scores[k])
    return best.astype(np.float32), best_score


# --------------------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------------------


def _keep_object_points(
    pcd: np.ndarray, obj_t0: np.ndarray, gt_pose: np.ndarray | None, thresh: float, pc_ops
) -> np.ndarray:
    """Keep points near the object, dropping the arms. `(T, N, 3) -> (T, N, 3)`.

    The object's frame-0 surface is re-posed onto each frame with the recorded pose, and points
    within `thresh` of it are kept; the rest are moved far outside the crop so the existing
    resampler discards them, which avoids a ragged intermediate.

    Needs per-frame ground-truth pose. Real-world sources have none and must use the
    forward-kinematics arm mask directly -- which is what the evaluator and the robot do.
    """
    if gt_pose is None:
        raise ValueError(
            "drop_arm_points=True needs per-frame observation.object_pose in the source. "
            "Re-port with --drop_arm_points=false, or apply FK arm removal at capture time."
        )
    out = pcd.copy()
    far = np.array([1e3, 1e3, 1e3], dtype=pcd.dtype)
    base_p, base_q = gt_pose[0, :3], gt_pose[0, 3:]
    for i in range(pcd.shape[0]):
        surf = pc_ops.transform_points(obj_t0, base_p, base_q, gt_pose[i, :3], gt_pose[i, 3:])
        # KD-tree, not a dense distance matrix: the brute-force form is ~12 M distances per
        # frame and 647 k frames across the dataset, which is days of work for the same answer.
        keep = cKDTree(surf).query(pcd[i], distance_upper_bound=thresh)[0] <= thresh
        out[i][~keep] = far
    return out


def build_episode_buffer(ep_idx: int, cfg: PortConfig, snap: SourceSnapshot) -> dict | None:
    """Read, preprocess and goal-label one source episode. Runs in a worker process."""
    if cfg.pc_common not in sys.path:
        sys.path.insert(0, cfg.pc_common)
    import pc_ops

    path = episode_parquet_path(cfg.src_root, ep_idx, snap.chunks_size)
    expected = snap.episode_lengths[ep_idx]
    diag: dict = {"src_episode_index": ep_idx}
    try:
        if not path.exists():
            return {"_skip": "missing_parquet", **diag}
        if pq.read_metadata(path).num_rows != expected:
            return {"_skip": "row_count_mismatch", **diag}
        arrays = load_episode_arrays(path, expected, cfg.keep_velocity)
    except Exception as exc:  # noqa: BLE001 - one bad episode must not kill the run
        return {"_skip": f"read_error:{type(exc).__name__}:{exc}", **diag}

    pcd = arrays["observation.point_cloud"]
    t = pcd.shape[0]
    rng = np.random.default_rng(cfg.seed + ep_idx)

    # ---- goal recovery -------------------------------------------------------------
    # Two paths. When the collector recorded `observation.object_pose`, the goal is simply the
    # object's pose on the final frame -- only successful episodes are saved, so the object ends
    # at the goal within the logged error (median 8.6 mm / 0.044 rad). That is exact, carries
    # orientation, and needs no gates. The registration path below is kept for collections made
    # before the collector was patched, and for real-world data, where no pose is available; it
    # recovers translation only.
    gt_pose = arrays.get("observation.object_pose")

    obj_t0 = pc_ops.isolate_object(
        pcd[0],
        xy_center=pc_ops.OBJECT_SPAWN_XY,
        xy_halfspan=cfg.obj_xy_halfspan,
        z_max=1.0,
        grid=cfg.cluster_grid_m,
    )
    if obj_t0.shape[0] < 50:
        return {"_skip": f"few_t0_object_points:{obj_t0.shape[0]}", **diag}
    object_top = float(obj_t0[:, 2].max())

    obj_last = pc_ops.isolate_object(
        pcd[t - 1],
        xy_center=pc_ops.OBJECT_SPAWN_XY,
        xy_halfspan=cfg.obj_last_xy_halfspan,
        z_max=object_top + cfg.obj_top_margin_m,
        grid=cfg.cluster_grid_m,
    )
    if obj_last.shape[0] < 50:
        return {"_skip": f"few_last_object_points:{obj_last.shape[0]}", **diag}

    if gt_pose is not None:
        # The goal is the object's *last plausible* pose, not simply the last frame. Success is
        # latched at the end of the push phase; the object is occasionally knocked off the table
        # during the arm lift that follows, and those frames put it metres below the floor. Taking
        # them as the goal corrupts the goal vector, the goal cloud and the auxiliary target for
        # the whole episode -- and, through MIN_MAX statistics, the normalisation of every other
        # episode too.
        gi = pc_ops.last_plausible_index(gt_pose)
        if gi < 0:
            return {"_skip": "no_plausible_pose", **diag}
        pose0, pose_goal = gt_pose[0], gt_pose[gi].copy()
        if cfg.goal_orientation == "commanded":
            # The goal ORIENTATION is the one the task commands, not the one the demonstrator
            # happened to reach. On a push the command is "do not rotate it", so the goal
            # orientation is the start orientation -- which is also exactly what the evaluator
            # sets `cmd.target_quat` to. Using the achieved orientation instead trains the policy
            # on the demonstrator's residual error (median 2.7 deg, p90 8.5 deg) as though it were
            # the target, and mismatches evaluation, where the goal cloud carries no rotation.
            pose_goal[3:] = pose0[3:]
        n_bad = int((~pc_ops.pose_is_plausible(gt_pose)).sum())
        diag.update(goal_frame=gi, n_implausible_frames=n_bad)
        delta = (pose_goal[:3] - pose0[:3]).astype(np.float64)
        residual = 0.0
    else:
        delta, residual = register_xy(obj_t0, obj_last, seed=cfg.seed + ep_idx)
        # No rotation is recoverable from the 2-DoF registration; the object is spawned upright
        # and the demonstration's own residual yaw is discarded.
        pose0 = np.array([*pc_ops.OBJECT_SPAWN_XY, object_top, 1.0, 0.0, 0.0, 0.0], np.float32)
        pose_goal = pose0.copy()
        pose_goal[:2] += delta[:2]
    goal_dist = float(np.hypot(delta[0], delta[1]))
    diag.update(
        n_obj_t0=int(obj_t0.shape[0]),
        n_obj_last=int(obj_last.shape[0]),
        object_top_m=object_top,
        goal_dx=float(delta[0]),
        goal_dy=float(delta[1]),
        goal_dist_m=goal_dist,
        reg_residual_m=residual,
        goal_source="ground_truth" if gt_pose is not None else "registration",
    )
    # Gates apply only to the estimated path; ground-truth poses need no rejection, which is
    # why this port keeps every episode the older one had to drop.
    if gt_pose is None:
        if not np.isfinite(residual) or residual > cfg.max_reg_residual_m:
            return {"_skip": f"reg_residual:{residual:.4f}", **diag}
        if goal_dist > cfg.max_goal_dist_m:
            return {"_skip": f"goal_dist:{goal_dist:.4f}", **diag}

    # ---- clouds --------------------------------------------------------------------
    crop = cfg.crop()
    if cfg.drop_arm_points:
        # Self-filtering, matching what runs on hardware. The captured cloud is already
        # table-free (the collector keeps only arm and object returns, and the real perception
        # pipeline likewise returns no table surface), so removing the robot's own body leaves
        # the object.
        #
        # On the robot this is done from joint encoders: forward kinematics gives link poses and
        # points within a radius of that skeleton are the arm. Offline there are no link poses in
        # the source -- only the 14 joint angles -- so the *complement* is computed instead: the
        # object's frame-0 surface is re-posed onto every frame with the recorded pose, and points
        # near it are kept. On a table-free cloud the two are the same set, since the object is
        # all that remains once the arms are gone.
        pcd = _keep_object_points(pcd, obj_t0, gt_pose, cfg.obj_seg_thresh_m, pc_ops)
    obs_pc, counts = pc_ops.crop_and_resample_batch(pcd, crop, cfg.num_points, rng)
    # Re-pose the t=0 object points onto the goal. With ground truth this carries rotation too,
    # which is what makes rotate/flip representable; the estimated path degenerates to a shift.
    goal_pts = pc_ops.transform_points(obj_t0, pose0[:3], pose0[3:], pose_goal[:3], pose_goal[3:])
    goal_pc = pc_ops.resample_to(goal_pts, cfg.goal_num_points, rng)
    diag.update(
        surviving_min=int(counts.min()),
        surviving_mean=float(counts.mean()),
        n_below_target=int((counts < cfg.num_points).sum()),
    )

    task = snap.tasks.get(0, "Push the object to the goal position.")
    onehot = pc_ops.task_onehot(task)
    goal_vec9 = pc_ops.pose_to_vec9(pose_goal[:3], pose_goal[3:])
    # The commanded transformation, initial pose -> goal pose: [dt_world (3), rot6d(dR) (6)].
    # This is the goal *as the robot receives it* -- known at deployment by construction (it is
    # the command), constant within an episode, and needing no online pose tracking, unlike the
    # current->goal residual. On push dR is identity and dz is 0, so 7 of 9 dims are constant;
    # the MIN_MAX normalizer maps a zero-span dim stably to -1 (normalize_processor.py:374), so
    # they sit inert until rotate/flip data gives them variance -- there, dR *is* the task.
    d_rot = pc_ops.quat_to_matrix(pose_goal[3:]) @ pc_ops.quat_to_matrix(pose0[3:]).T
    goal_transform = np.concatenate(
        [
            (pose_goal[:3] - pose0[:3]).astype(np.float32),
            pc_ops.matrix_to_rot6d(d_rot).astype(np.float32),
        ]
    )
    buf: dict = {
        "size": t,
        "task": [task] * t,
        "timestamp": (np.arange(t, dtype=np.float32) / snap.fps).astype(np.float32),
        "frame_index": np.arange(t, dtype=np.int64),
        "index": np.zeros(t, dtype=np.int64),
        "task_index": np.zeros(t, dtype=np.int64),
        "observation.state": arrays["observation.state"],
        "observation.point_cloud": obs_pc,
        # Constant within an episode; stored per frame because LeRobot features are per frame.
        "observation.goal_point_cloud": np.repeat(goal_pc[None], t, axis=0),
        # Goal *pose*, absolute in the env frame, constant within an episode. This is the
        # policy's goal vector; the residual to it is what the auxiliary head predicts, so the
        # current pose below is a label and never an input.
        "observation.goal_pose": np.tile(goal_vec9, (t, 1)),
        # Relative encoding of the same goal; see the comment where it is built.
        "observation.goal_transform": np.tile(goal_transform, (t, 1)),
        "observation.task_onehot": np.tile(onehot, (t, 1)),
        # Label only. `pose_valid` is 0 for sources without ground truth (real-world data), which
        # lets one schema serve both and the auxiliary loss mask per sample.
        # Implausible frames are replaced by the last plausible pose so no 5-metre outlier can
        # reach the dataset statistics; `pose_valid` marks them so the loss ignores them anyway.
        "observation.object_pose": (
            _sanitise_poses(gt_pose, pc_ops) if gt_pose is not None else np.tile(pose0, (t, 1))
        ).astype(np.float32),
        # Per-frame validity: 0 where the recorded pose is not plausible, so the auxiliary loss
        # masks those frames instead of regressing a 5-metre residual. The same flag is 0 for
        # whole sources that have no ground truth at all, such as real-world data.
        "observation.pose_valid": (
            pc_ops.pose_is_plausible(gt_pose).astype(np.float32)[:, None]
            if gt_pose is not None
            else np.zeros((t, 1), dtype=np.float32)
        ),
        "action": (
            arrays["action"] - arrays["observation.state"]
            if cfg.action_mode == "delta"
            else arrays["action"]
        ),
    }
    if cfg.keep_velocity:
        buf["observation.velocity"] = arrays["observation.velocity"]
    return {"_buffer": buf, **diag}


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def build_features(cfg: PortConfig) -> dict:
    # Same sys.path injection as the worker: `pc_ops` lives in the IsaacLab repo so both
    # interpreters share one definition of the task list and the pose encoding.
    if cfg.pc_common not in sys.path:
        sys.path.insert(0, cfg.pc_common)
    import pc_ops

    feats = {
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
        "observation.goal_transform": {
            "dtype": "float32",
            "shape": (9,),
            "names": ["dx", "dy", "dz", "r00", "r10", "r20", "r01", "r11", "r21"],
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
    }
    if cfg.keep_velocity:
        feats["observation.velocity"] = {
            "dtype": "float32",
            "shape": (14,),
            "names": STATE_JOINT_NAMES,
        }
    return feats


def main(cfg: PortConfig) -> None:
    src = cfg.src_root.resolve()
    dst = cfg.dst_root.resolve()
    # The source may still be receiving writes from the collector. Never write near it.
    if dst == src or src in dst.parents or dst in src.parents:
        raise ValueError(f"dst_root must be outside src_root (src={src}, dst={dst})")

    sys.path.insert(0, cfg.pc_common)
    import pc_ops

    snap = freeze_source_snapshot(src)
    n_eps = snap.total_episodes if cfg.max_episodes is None else min(snap.total_episodes, cfg.max_episodes)
    logger.info("source: %d episodes @ %.1f fps (frozen snapshot)", n_eps, snap.fps)

    features = build_features(cfg)
    if (dst / "meta" / "info.json").exists():
        ds = LeRobotDataset.resume(cfg.repo_id, root=dst)
        logger.info("resuming: %d episodes already written", ds.meta.total_episodes)
    else:
        # `create` makes the dataset dir itself and errors if it already exists; only ensure the parent.
        dst.parent.mkdir(parents=True, exist_ok=True)
        ds = LeRobotDataset.create(
            cfg.repo_id,
            fps=int(snap.fps),
            features=features,
            root=dst,
            robot_type="trossen_aloha_bimanual",
            use_videos=False,
            data_files_size_in_mb=cfg.data_files_size_in_mb,
        )

    diag_path = dst / "meta" / "port_diagnostics.jsonl"
    map_path = dst / "meta" / "episode_objects.jsonl"
    done_src = set()
    if map_path.exists():
        with open(map_path) as f:
            for line in f:
                done_src.add(json.loads(line)["src_episode_index"])
    todo = [i for i in range(n_eps) if i in snap.episode_lengths and i not in done_src]
    logger.info("porting %d episodes (%d already done)", len(todo), len(done_src))

    t_start = time.time()
    n_written = 0
    n_skipped = 0
    worker = partial(build_episode_buffer, cfg=cfg, snap=snap)
    with (
        open(diag_path, "a") as diag_f,
        open(map_path, "a") as map_f,
        ProcessPoolExecutor(cfg.num_workers) as ex,
    ):
        for result in ex.map(worker, todo, chunksize=1):
            src_idx = result["src_episode_index"]
            if "_buffer" not in result:
                n_skipped += 1
                diag_f.write(json.dumps({k: v for k, v in result.items() if k != "_buffer"}, default=_json_default) + "\n")
                diag_f.flush()
                logger.warning("skip src ep %d: %s", src_idx, result.get("_skip"))
                continue
            buf = result.pop("_buffer")
            # Capture before save_episode: the writer overwrites this key in-place with a
            # per-frame array, so reading it back afterwards yields a list, not the index.
            new_ep_idx = int(ds.meta.total_episodes)
            buf["episode_index"] = new_ep_idx
            ds.save_episode(episode_data=buf)
            obj = snap.objects.get(src_idx, {})
            map_f.write(
                json.dumps(
                    {
                        "episode_index": new_ep_idx,
                        "src_episode_index": src_idx,
                        "object": obj.get("object"),
                        "base_mesh": base_mesh_name(obj["object"]) if obj.get("object") else None,
                        "group": obj["object"].split("/")[1] if obj.get("object") else None,
                        "success": obj.get("success"),
                        "pos_err_m": obj.get("pos_err_m"),
                        "ori_err_rad": obj.get("ori_err_rad"),
                        "goal_dx": result["goal_dx"],
                        "goal_dy": result["goal_dy"],
                        "goal_dist_m": result["goal_dist_m"],
                        "reg_residual_m": result["reg_residual_m"],
                    },
                    default=_json_default,
                )
                + "\n"
            )
            map_f.flush()
            diag_f.write(json.dumps({k: v for k, v in result.items() if k != "_buffer"}) + "\n")
            diag_f.flush()
            n_written += 1
            if n_written % 25 == 0:
                el = time.time() - t_start
                logger.info(
                    "%d/%d written (%d skipped) | %.2f s/ep | eta %.1f min",
                    n_written,
                    len(todo),
                    n_skipped,
                    el / n_written,
                    (len(todo) - n_written) * el / n_written / 60,
                )

    ds.finalize()
    (dst / "meta" / "preprocessing.json").write_text(
        json.dumps(pc_ops.describe(cfg.crop(), cfg.num_points, cfg.goal_num_points), indent=2)
    )
    logger.info(
        "done: %d episodes written, %d skipped, %.1f min",
        n_written,
        n_skipped,
        (time.time() - t_start) / 60,
    )


def parse_args() -> PortConfig:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src_root", type=Path, required=True)
    p.add_argument("--dst_root", type=Path, required=True)
    p.add_argument("--repo_id", type=str, default="local/push_pc1024")
    p.add_argument("--num_points", type=int, default=1024)
    p.add_argument("--goal_num_points", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--max_episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_velocity", action="store_true")
    p.add_argument("--goal_orientation", choices=("commanded", "achieved"), default="commanded",
                   help="'commanded' sets the goal orientation to the object's START orientation, "
                        "which is what the task actually asks for on a push ('move it there without "
                        "rotating it') and what the simulator commands at evaluation time. "
                        "'achieved' uses the object's final recorded orientation, which bakes the "
                        "demonstrator's own residual error into the goal and mismatches evaluation.")
    p.add_argument("--pc_common", type=str, default=DEFAULT_PC_COMMON)
    a = p.parse_args()
    return PortConfig(
        src_root=a.src_root,
        dst_root=a.dst_root,
        repo_id=a.repo_id,
        num_points=a.num_points,
        goal_num_points=a.goal_num_points,
        num_workers=a.num_workers,
        max_episodes=a.max_episodes,
        seed=a.seed,
        keep_velocity=not a.no_velocity,
        goal_orientation=a.goal_orientation,
        pc_common=a.pc_common,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main(parse_args())
