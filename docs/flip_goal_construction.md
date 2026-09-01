# Flip goal construction: goal point cloud and goal vector

Written 2026-08-31, for replicating the training-time goal exactly on hardware.

The flip policy is conditioned on two goal inputs -- `observation.goal_point_cloud` (512, 3) and
`observation.goal_transform` (9,) -- both derived from a single **commanded goal pose**. If the
deployment computes that pose even slightly differently from the porter that built the training
labels, the policy is conditioned on one thing and asked to achieve another, and the failure is
silent: no error, just a lower success rate that looks like a bad policy.

This document is the specification. The reference implementation is
`eval_flip_policy.flip_command_pose` (3D_Bimanual_repo), deliberately placed **above** the
IsaacLab import so it can be imported with numpy alone -- no simulator. Copy that function rather
than reimplementing from this prose.

Related: [`flip_benchmark_protocol.md`](./flip_benchmark_protocol.md) (how flips are scored),
[`primitive_dataset_pipeline.md`](./primitive_dataset_pipeline.md) (the general porting stack).

---

## 0. Frame and units

Everything is in the **single-scene frame** (in simulation: env-local, i.e. the env origin
subtracted -- a pure translation, so orientations are identical in world and env-local).

| anchor | value |
|---|---|
| table top surface | z = 0 |
| object spawn | (x, y) = (0.0, -0.019) |
| robot bases | x = +-0.4575 |
| units | metres, radians |
| quaternions | **wxyz**, Isaac convention |

## 1. What the policy expects

| key | shape | normalisation | you supply |
|---|---|---|---|
| `observation.point_cloud` | (1024, 3) f32 | IDENTITY | raw metres |
| `observation.goal_point_cloud` | (512, 3) f32 | IDENTITY | raw metres |
| `observation.goal_transform` | (9,) f32 | **MIN_MAX** | raw values |
| `observation.state` | (14,) f32 | MIN_MAX | raw joint radians |

**Send raw values for every one of them.** Two separate normalisations happen downstream and
double-applying either will silently wreck the inputs:

* MIN_MAX for `goal_transform` / `state` / `action` is applied **server-side**
  (`pc_policy_server.py` runs `preprocessor(batch)`) from the statistics of the dataset the
  checkpoint trained on. Point the server at that dataset root, not at a new one.
* The isotropic cloud rescale `(p - [0, 0, 0.285]) / 0.40` is applied **inside the model**
  (`modeling_pc_diffusion._rescale`), identically to both clouds. Clouds go in as metres.

## 2. The commanded goal pose

A flip is not commanded as a place to put the object. The command is:

> tip it `flip_angle_deg` (default 90) about the horizontal axis **v(d)** of its own flip frame,
> pivoting on the edge it would tip over.

`d` is one of `+x / -x / +y / -y`, read in the **object** frame -- so the axis turns with the
object's spawn yaw and `+x` tips the same physical face over however the object is rotated on the
table.

Given the object's initial pose `pose0 = [pos0 (3), quat0 (4, wxyz)]` and its initial observed
object-only cloud `obj_t0`:

```python
W = {"+x": 0.0, "-x": pi, "+y": pi/2, "-y": -pi/2}

yaw = W[d] + yaw_z(quat0)                      # world-Z yaw of the object; SEE THE FLOAT64 NOTE
u   = ( cos(yaw), sin(yaw), 0.0)               # travel axis: the top moves this way
v   = (-sin(yaw), cos(yaw), 0.0)               # rotation axis: horizontal, frozen during the tip

z_min = obj_t0[:, 2].min()                     # table contact
foot  = obj_t0[obj_t0[:, 2] <= z_min + 0.005]  # resting footprint, 5 mm band
if len(foot) < 3: foot = obj_t0                # degenerate capture: fall back to the whole cloud
pivot = foot[argmax(foot @ u)]                 # the +u extreme of that footprint ...
pivot[2] = z_min                               # ... dropped to table height

R = rotation_matrix(axis=v, angle=radians(flip_angle_deg))
goal_pos  = pivot + R @ (pos0 - pivot)         # rotate the object ABOUT THE PIVOT EDGE
goal_quat = quat(axis=v, angle=...) (x) quat0  # LEFT-multiply: a world-frame rotation
```

`yaw_z(q) = atan2(2(wz + xy), 1 - 2(y^2 + z^2))`.

### Why pivot geometry and not the achieved final pose

The goal must be *known at deployment by construction* -- nothing can know where the object will
come to rest before it is pushed. Push gets away with using the achieved final position because
its demonstrator was driven to a commanded goal and lands a median 8.6 mm from it. The flip
collection never commanded a position, and substituting the achieved pose differs from the pivot
prediction by a **median 36 mm, p90 86 mm** (n=246) -- larger than push's entire 30 mm position
gate. Training on it would also make the goal uncomputable on hardware.

Per-axis correlation between the pivot prediction and the achieved displacement is 0.971 / 0.974
/ 0.979, so the model is structurally right; it slightly over-predicts magnitude (0.218 vs
0.195 m mean) because a real flip slides a little rather than pivoting perfectly about the edge.

## 3. `observation.goal_point_cloud` (512, 3)

The **initial object cloud, re-posed onto the goal**:

```python
obj_t0 = isolate_object(cloud_t0, xy_center=(0.0, -0.019), xy_halfspan=0.30,
                        z_max=1.0, grid=0.03)          # object-only points at t=0
goal_pts = transform_points(obj_t0, pos0, quat0, goal_pos, goal_quat)
goal_cloud = resample_to(goal_pts, 512, rng)
```

`transform_points` maps world -> object frame using `pose0`, then object frame -> world using the
goal pose. `resample_to` draws without replacement when there are enough points and with
replacement when there are not.

Three properties that are easy to get wrong:

1. **Built once, at t=0, and held constant for the whole episode.** It is not recomputed per
   frame.
2. **Not cropped.** Unlike the observation cloud, goal points outside the workspace box are kept.
   They are faithful, not a bug -- the training data was built the same way, so cropping here
   would change the distribution.
3. **It is a partial two-view observation.** Rotating it 90 deg exposes surfaces the cameras
   never saw, so its occlusion statistics genuinely differ from the observation cloud's. That is
   inherent to the representation and training saw exactly the same thing. Do not try to "fix" it
   by completing the shape.

## 4. `observation.goal_transform` (9,) -- the goal vector

```
[dx, dy, dz, r00, r10, r20, r01, r11, r21]

dt   = goal_pos - pos0                       # world frame, metres
rot6d = first two COLUMNS of dR = R_goal @ R_0^T, flattened column-major
```

```python
dR    = quat_to_matrix(goal_quat) @ quat_to_matrix(quat0).T
rot6d = swapaxes(dR[..., :, :2], -1, -2).reshape(6)      # COLUMNS, not rows
goal_transform = concatenate([goal_pos - pos0, rot6d])
```

Constant for the whole episode.

**The column/row trap.** `dR[..., :, :2].reshape(6)` interleaves row-major and produces a
different, still-6-dimensional, entirely plausible-looking, wrong encoding. The dataset's own
feature names spell the correct order out: `r00, r10, r20` is the first column, `r01, r11, r21`
the second.

**The float64 trap.** Compute `yaw_z` in float64. An earlier version of the evaluator computed it
in float32 and diverged from the porter by up to 0.04 deg -- small, but it meant training and
evaluation were building different goals. After the fix the two agree **bitwise, 0 ULP over 3000
randomised cases**.

Sanity ranges from the training set (2453 episodes):

| dim | min | max | | dim | min | max |
|---|---|---|---|---|---|---|
| dx | -0.306 | +0.329 | | r00 | 0.0 | 1.0 |
| dy | -0.301 | +0.303 | | r10 | -0.5 | +0.5 |
| dz | -0.223 | +0.127 | | r20 | -1.0 | +1.0 |
| | | | | r01 | -0.5 | +0.5 |
| | | | | r11 | 0.0 | 1.0 |
| | | | | r21 | -1.0 | +1.0 |

If your `rot6d` comes out near `[1,0,0,0,1,0]` (identity) you have built push's goal, not a
flip's -- that is exactly the failure mode the porter's original `--goal_orientation commanded`
mode produced on flip data, and it deletes the entire task signal.

## 5. `observation.point_cloud` (1024, 3)

Per frame, at 10 Hz: object-only points (arms and table removed), cropped to
`x +-0.61, y +-0.38, z -0.03..0.50`, resampled to 1024.

In simulation 8.9% of frames had fewer than 1024 object points -- the two arms occlude the object
during the press -- so the resampler duplicates. Expect the same on hardware; it is not an error
condition.

## 6. Runtime

```
n_obs_steps      2       the server keeps the history; send ONE frame per call
horizon          64
n_action_steps   32
inference        DDIM-10
rate             10 Hz   (50 Hz control / capture_decim 5)
action_space     absolute_joint
```

Call `reset()` at episode start to clear the observation queue, then send observations at 10 Hz.
Actions are **absolute joint targets** -- do not add them to the current joint positions. The
14-D vector expands to 16 as `[L arm x6, L grip, L grip, R arm x6, R grip, R grip]`.

## 7. Verifying your implementation

The cheapest end-to-end check, and it exercises the column-major rot6d, the float64 yaw and the
pivot geometry in one shot:

1. Pick an episode from `/home/samsung/data/flip_objonly`.
2. Its commanded direction is in
   `.../flip_data/meta/episode_flip.jsonl` (`flip_direction`, `flip_angle_deg`).
3. Feed your pipeline that episode's t=0 cloud and pose.
4. Compare your `goal_transform` against the stored one. It should match to float32.

If only the rotation dims disagree, suspect the column/row order. If everything disagrees by a
little, suspect float32 yaw. If `dz` is near zero, suspect that you built the goal without the
pivot (a flip about the object's own centre has almost no height change; a flip about its edge
does).

## 8. Reference implementations

| what | where |
|---|---|
| goal pose (deployment-shaped, numpy only) | `eval_flip_policy.flip_command_pose` |
| goal pose (training labels) | `port_isaaclab_pointcloud_push.flip_command_pose` |
| cloud ops | `pc_common/pc_ops.py`: `isolate_object`, `transform_points`, `resample_to`, `matrix_to_rot6d` |
| per-episode command record | `flip_data/meta/episode_flip.jsonl` |

The two `flip_command_pose` implementations are required to stay bitwise identical. Re-run the
0-ULP equivalence check after touching either.
