# Primitive dataset pipeline: cleaning, porting, and input review

How a raw IsaacLab collection becomes a training dataset, and how to review every input
before spending GPU on it. Written for reuse: **rotate and flip data go through exactly this
procedure** when they arrive. The push-specific checklist at the end says what changes.

The porting code is `examples/port_datasets/port_isaaclab_pointcloud_push.py`; the review
tool is `tools/inspect_inputs.py`. Evaluation-side counterparts live in
`eval_push_policy.py` (IsaacLab repo) and are noted where train/eval symmetry matters.

---

## 0. Source contract

The collector (`dataset_collection/collect_push_lerobot.py`) must have produced, per frame:

| field | shape | notes |
|---|---|---|
| `observation.point_cloud` | 4096×3 | capture crop `x ±0.40, y ±0.30, z −0.03…0.60`; **arm ∪ object only** — the collector keeps `arm_mask \| obj_mask`, so the table is already absent |
| `observation.object_pose` | 7 | ground-truth pose (xyz + wxyz quat). Required for cleaning steps 2, 4, 6 |
| `observation.state` / `observation.velocity` | 14 | joint positions / velocities, both arms |
| `action` | 14 | commanded joint targets |

Plus `meta/attempts.jsonl` (all attempts including failures — the only record of demonstrator
success rates; the dataset itself stores successes only).

## 1. The port command

```bash
python examples/port_datasets/port_isaaclab_pointcloud_push.py \
  --src_root <collection>/push_data \
  --dst_root /home/samsung/data/<name> \
  --repo_id local/<name> \
  --action_mode delta \
  --num_points 1024 --num_workers 6
# ~4 minutes for ~2100 episodes. Defaults: --goal_orientation commanded, arm removal ON.
```

Deterministic given `--seed` (default 0): re-running reproduces the dataset bit-for-bit.

## 2. Cleaning steps, in port order, with the failure each one exists to prevent

**2.1 Pose sanitisation** (`_sanitise_poses`, `pc_ops.pose_is_plausible`). Implausible poses
(fallen objects recorded at z ≈ −5 m) are forward-filled from the last plausible frame, and
`observation.pose_valid` marks them so auxiliary losses skip them.
*Why: three bad episodes once set the dataset min/max, stretching the goal-pose z-span from
~0.01 m to 5.24 m and collapsing the normalised goal signal to a constant.*

**2.2 Goal construction — commanded, never achieved** (`--goal_orientation commanded`).
The goal pose is the object's start pose plus the commanded transform; the goal **orientation
is the spawn orientation** on push. The goal cloud is the isolated t=0 object cloud re-posed by
that transform (`pc_ops.transform_points`), 512 points. It is **not cropped** to the workspace —
deployment builds it the same way (transform of the observed initial cloud), so out-of-box goal
points are faithful, not a bug.
*Why: "achieved" goals bake the demonstrator's residual error (median ~3°) into the target and
mismatch what the evaluator commands.*

**2.3 Object isolation at t=0** (`pc_ops.isolate_object`): spawn-centred box + connected
components separates the object from the static arm bases. Episodes with <50 object points are
skipped (0 of 2115 on push).

**2.4 Object-only observation** (`drop_arm_points`, default ON). Per frame, keep points within
15 mm (`obj_seg_thresh_m`, = the collector's own threshold) of the t=0 object surface re-posed by
the recorded pose (KD-tree; a dense distance matrix here costs days over 647k frames). Kept
count on push: 1960–3048 of 4096, always above the 1024 resample target, so no duplication.
*Why: with arms in the cloud the object was ~29% of the points; success correlated with object
shape (r=+0.61) while the expert showed no such correlation — and a skeleton-radius arm filter
was measured deleting 7.6% of object points exactly at contact. Keeping the object directly is
also what deployment does: segment RGB, project onto registered depth. The evaluator's capture
mirrors this (`keep = obj_mask`; `--keep_arm_points` restores the old format).*

**2.5 Crop + resample to 1024 points** (`pc_ops.crop_and_resample_batch`, shared crop constants
with the evaluator — the eval contract check verifies they match at startup).

**2.6 Goal transform** (`observation.goal_transform`, 9): `[Δt_world (3), rot6d(R_goal·R₀ᵀ) (6)]`
— the *commanded* initial→goal transformation, constant per episode, known at deployment by
construction. On push, ΔR is identity and Δz≈0, so 7 of 9 dims are constant; the MIN_MAX
normaliser maps zero-span dims stably to −1 (`normalize_processor.py`), and they gain variance
automatically on rotate/flip. The evaluator computes the identical quantity from the command.
*Why relative rather than the current→goal residual: the residual needs online pose tracking the
robot does not have. Why rot6d: avoids the quaternion double cover; note `matrix_to_rot6d` takes
the first two COLUMNS — a row-major reshape produces a different, plausible-looking, wrong
encoding.*

**2.7 Delta actions** (`--action_mode delta`): store `action[t] − state[t]` — the commanded
target relative to the measured joints. |δ| mean ≈ 0.008 rad vs 0.586 for absolute; round-trip
verified to float32 rounding. Gripper deltas are ~constant → zero-span guard → the inverse maps
them back exactly, grippers hold. The checkpoint carries `action_space="delta_joint"`, the
policy server advertises it, and **the evaluator adds the live joint position back and refuses
unknown values** — a silent mismatch would command near-zero motion and score ~0.

**2.8 Normalisation at training time** (flags, not port steps — see the training command):
`POINT_CLOUD → IDENTITY` plus `pc_isotropic_rescale` with `pc_center=(0,0,0.285)`,
`pc_scale=0.40` — the capture workspace mapped isotropically into [−1, 1], **one map for both
clouds** (obs/goal stretch 1.00 by construction, shape-preserving, independent of dataset
statistics, same three constants on hardware). STATE and ACTION stay per-dim MIN_MAX.
*Why: per-key MIN_MAX once gave the observation cloud the arms' extent and the goal cloud the
object's — a 1.83× stretch on x between the two things the network must compare.*

## 3. Verification gates — run before any training

Automated (the training chain runs these and aborts on failure):

```python
i = json.load(open(f"{root}/meta/info.json")); s = json.load(open(f"{root}/meta/stats.json"))
assert i["total_episodes"] == EXPECTED
assert "observation.goal_transform" in i["features"]
assert abs(np.array(s["action"]["max"])).max() < 0.5   # delta-scaled, not absolute
```

Manual spot-checks that have each caught a real problem once:
- object fraction of the observation cloud ≈ 100% (compare a frame against the goal-cloud AABB
  at the recorded pose);
- unique points per frame = 1024 (resampler not duplicating);
- all values finite; per-key stats spans sane (a 5 m z-span means step 2.1 failed);
- `port_diagnostics.jsonl` skip count (push: 0 of 2115).

## 4. Input review — `tools/inspect_inputs.py`

```bash
python3 tools/inspect_inputs.py --dataset_root /home/samsung/data/<name> --out inputs.html
```

Self-contained page, no session state: picks objects spread across the shape range (thinness
computed from the goal cloud itself), extracts one successful episode each, and renders every
field on one scrubber — animated observation cloud and goal cloud inside the [−1, 1] workspace
box, the goal_transform arrow (start centroid → goal centroid), joint/action/velocity traces,
object path, scalars, and the field-encoding table. Publish as an artifact or open locally.

What to look for:
- **Cloud**: clean object surface, no arm structure, shape held while translating; both clouds
  inside (or goal slightly outside — see 2.2) the wireframe box.
- **Arrow**: matches the object path's direction and length; for rotate/flip, check the rot6d
  numbers on the card are non-identity.
- **State/action traces**: smooth approach → push → return; actions in delta mode are
  small-magnitude and centred, not posture-sized.
- **Velocity**: marked "not read" — confirm nothing you expect the model to see is grey in the
  encoding table.

## 5. Rotate / flip checklist — what changes, what must not

- **Same collector contract** (§0), including per-frame `observation.object_pose` — without it
  steps 2.4, 2.6, 2.7 degrade or fail loudly.
- **Goal stays commanded** (2.2). For rotate/flip the commanded transform carries real rotation:
  `goal_transform`'s rot6d dims stop being constant and start carrying the task. Verify on the
  inspector page that they are non-identity and vary across episodes.
- **Flip adds Δz**: check the goal cloud sits at the flipped height and Δz in the transform is
  non-zero.
- **MIN_MAX on rot6d dims** becomes per-dim scaling of a rotation encoding — monotonic and
  usable as conditioning, but not a valid rot6d after normalisation; fine for input, never
  decode it back.
- **`task_onehot`** must name the primitive (`pc_ops.TASK_NAMES`); turn `use_task_onehot` on
  only when training one policy across primitives.
- **Auxiliary rotation** (`aux_predict_rotation`) was degenerate on push (commanded goals make
  the residual-rotation target near-constant; its loss logs 0.000). On rotate/flip the residual
  rotation *is* the task — re-evaluate the head there, as its own arm.
- **Frozen eval list per primitive**: build the object list once (demonstrator-competence
  screen, then freeze the file, per `docs/push_benchmark_protocol.md` §1) and never regenerate
  it from statistics — that silently reshuffled 10 of 21 objects once.
- **Success gates per primitive**: position/orientation tolerances must be set per task; push's
  50 mm / 0.20 rad do not automatically transfer to a flip.

## 6. Provenance of the current datasets

| dataset | actions | clouds | goal_transform | use |
|---|---|---|---|---|
| `push_pc1024_poses` | absolute | arms + object | no | historical baselines (aux_v1) |
| `push_objonly` | absolute | object only | yes | absolute-action arms |
| `push_objonly_delta` | **delta** | object only | yes | **current training** |

All three port from the same collection with the same seed; differences are exactly the flags
above.
