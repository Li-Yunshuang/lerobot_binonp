> **SUPERSEDED (2026-08-31).** Kept as historical record of the pre-cleaning campaign.
> Current process: [`primitives_training_playbook.md`](./primitives_training_playbook.md). Current baseline: [`baseline_push_dit.md`](./baseline_push_dit.md).

# Point-cloud push policies: data → training → simulated evaluation

End-to-end runbook for the IsaacLab bimanual push task: turning raw simulator demonstrations into
a trained goal-conditioned point-cloud policy, and scoring it back in the simulator.

Everything here has been run; the numbers quoted are measured, not estimated.


> **Current experiment plan:** [`push_experiment_plan.md`](push_experiment_plan.md) — the
> live plan for improving success rate (diagnosis, experiment queue, and the levers already
> measured as dead). This file documents the *pipeline*; that one documents *what to run next*.

---

## The shape of the problem

Two facts drive every design decision below.

**The work spans two Python interpreters that cannot be merged.** LeRobot requires Python ≥ 3.12 /
numpy ≥ 2; IsaacSim ships CPython 3.11 / numpy 1.26. So the simulator cannot import the policy, and
the policy process cannot import the simulator. Evaluation therefore runs as two processes talking
over a local socket (Stage 5).

**The raw dataset does not record the goal.** The object's start pose is fixed, so at *t=0* the
observation is identical across every episode — the goal is the only thing that distinguishes them,
and it was never written to disk. It has to be recovered offline (Stage 1), and without it the task
is not learnable at all.

### Where things live

| | |
|---|---|
| **`lerobot_binonp`** (this repo) | port scripts, policies, training, policy server |
| **`3D_Bimanual_repo`** | collection, the shared preprocessing module, the eval harness |
| `…/irregular/pc_common/pc_ops.py` | **single source of truth** for crop/resample, imported by *both* interpreters |
| `…/irregular/eval/eval_push_policy.py` | the eval harness (IsaacLab side) |
| `tools/pc_policy_server.py` | the policy server (LeRobot side) |

`pc_ops.py` being shared is the single most important correctness property in this pipeline.
Train/eval preprocessing skew is silent and fatal: it produces a policy that trains fine and then
scores near zero, with nothing in the logs to say why. Keep that file numpy-1.26/2.x portable
(`np.asarray`, explicit dtypes, no `np.float_`).

---

## Stage 0 — environments

```bash
LEROBOT_PY=/home/samsung/miniforge3/envs/lerobot/bin/python     # 3.12, numpy 2.x
ISAAC_PY=/home/samsung/miniforge3/envs/isaaclab/bin/python      # 3.11, numpy 1.26
```

Set `HF_LEROBOT_HOME` and `HF_DATASETS_CACHE` somewhere with room — the datasets below are ~9 GB
each and the pipeline writes two of them.

---

## Stage 1 — port the raw dataset, and recover the goal

The collector writes LeRobot **v2.1**. This repo is `CODEBASE_VERSION = "v3.0"` and
`check_version_compatibility` hard-raises across a major version gap, so the data must be ported
before anything can read it.

> **Do not run `convert_dataset_v21_to_v30.py` at a live collection directory.** It begins with
> `shutil.move(root, root_old)`, which renames the directory out from under a running collector.
> The port script below instead reads the source strictly read-only and writes to a fresh root,
> asserting at entry that destination ≠ source and that neither is nested inside the other.

```bash
$LEROBOT_PY examples/port_datasets/port_isaaclab_pointcloud_push.py \
  --src_root /home/samsung/3D_Bimanual_repo/scripts_isaaclab/irregular/dataset_collection/push_data \
  --dst_root /home/samsung/data/push_pc1024 \
  --repo_id local/push_pc1024 \
  --num_points 1024 --goal_num_points 512 --num_workers 6
```

**How the goal is recovered.** Only successful episodes are saved, so the object *ends at the goal*
— within the recorded position error (median 7.8 mm, max 26 mm, all under the 30 mm success
threshold). By the final frame the arms have lifted clear, so the object is cleanly isolable. The
script isolates the object at *t=0* and at the last frame, then runs a coarse-to-fine 2-DoF
registration to recover `(dx, dy)`.

Registration, not centroid difference: the two partial 2-view clouds see different faces of the
object, so centroids disagree by 11–38 mm while registration converges to the cloud's own point
spacing.

Episodes are rejected if the residual exceeds 15 mm or the displacement exceeds 0.22 m. Measured
over the full run: **2088 of 2106 episodes ported** (18 rejected), residual **median 6.8 mm, p99
13.8 mm**, 647,280 frames, 8.9 GB.

The goal is written as `observation.goal_point_cloud` — the *t=0* object points translated by
`(dx, dy)` — plus a 3-float `observation.goal`, so goal-cloud vs goal-vector is a config flag
rather than a re-port. It is built by the **same construction** at training and at evaluation
time, which is what keeps the two sides comparable.

The script also writes `meta/preprocessing.json` (the crop box, point counts and channel list). The
policy server replays this to the eval client in a handshake so a mismatch fails at startup rather
than as a mysteriously low success rate.

**Cross-checks worth repeating if you change anything.** The collector's own log reports
`goal_dist: min=8.7cm mean=17.4cm max=20.0cm`, matching the recovered 0.087–0.198 m. Independently,
per-frame tracking (Stage 3) agrees with the one-shot registration to a **median of 0.8 mm** across
2088 episodes.

---

## Stage 2 — held-out object split

Generalization to unseen object geometry is the point of a 3D policy here, so the split is by
**object**, decided once, and used by both training and evaluation.

```bash
$LEROBOT_PY examples/port_datasets/make_push_object_split.py \
  --dataset_root /home/samsung/data/push_pc1024 \
  --src_root /home/samsung/3D_Bimanual_repo/scripts_isaaclab/irregular/dataset_collection/push_data \
  --test_frac 0.2
```

Three things make this less trivial than a random episode split:

- **Orientation variants leak.** The same base mesh appears under several orientations
  (`Foo`, `pos_x_down__Foo`, …). Splitting on the entry name puts identical geometry on both sides
  and badly flatters the held-out number. The grouping key is the **prefix-stripped base mesh**.
- **Groups are uneven** (round / non-convex / rotated / frustum), so the draw is stratified by
  group — a plain random 20% can miss a category entirely.
- **Some objects are infeasible for the scripted expert** (0% success). A policy failing there
  measures the demonstrator, not the policy, so they are kept out of the held-out set and reported
  separately.

Result: **61 base meshes → 48 train / 1619 episodes, 13 held-out / 469 episodes.**

Outputs land in `<dataset>/splits/`. Note `objects_*_paths.txt` as well as `objects_*.txt`: the
simulator spawns assets by **path**, while the split reasons about base-mesh **names**.

---

## Stage 3 — object pose labels (only for the auxiliary head)

Skip this unless training with `--policy.num_objects=1`. It produces a second dataset carrying
per-frame object poses, used as the auxiliary regression target.

```bash
# 1. Track the object through every episode (~22.5 ms/frame).
$LEROBOT_PY examples/port_datasets/track_push_object_poses.py \
  --dataset_root /home/samsung/data/push_pc1024 \
  --src_root /home/samsung/3D_Bimanual_repo/scripts_isaaclab/irregular/dataset_collection/push_data \
  --out /home/samsung/data/push_pc1024/meta/object_tracks.jsonl

# 2. Attach them as features (writes a NEW dataset; ~5 min, ~4 GB peak RSS, +8 GB disk).
$LEROBOT_PY examples/port_datasets/add_object_pose_features.py \
  --dataset_root /home/samsung/data/push_pc1024 --repo_id local/push_pc1024 \
  --tracks /home/samsung/data/push_pc1024/meta/object_tracks.jsonl \
  --out_root /home/samsung/data/push_pc1024_poses --out_repo_id local/push_pc1024_poses

# 3. The split and object metadata are not copied by add_features — bring them across.
cp -r /home/samsung/data/push_pc1024/splits /home/samsung/data/push_pc1024_poses/
cp /home/samsung/data/push_pc1024/meta/episode_objects.jsonl /home/samsung/data/push_pc1024_poses/meta/
```

**The tracker recovers translation only** — a yaw estimate from a partial 2-view cloud would be far
noisier than the translation — so the pose labels carry an identity rotation. Train with
`--policy.aux_predict_rotation=false`, or the head regresses against a constant.

Sanity check the result: the residual-to-goal should shrink over an episode (measured: **0.189 m at
frame 0 → 0.0013 m at frame 309**), and its overall median should be ≈ 0.108 m.

`add_features` uses pandas, which expands each nested point cloud roughly 19× — a 199 MB parquet
file becomes ~3.7 GB in memory. It processes one file at a time, so peak RSS stays ~4 GB, but do
not parallelise it.

---

## Stage 4 — train

```bash
$LEROBOT_PY -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/push_pc1024 \
  --dataset.root=/home/samsung/data/push_pc1024 \
  --dataset.episodes="$(cat /home/samsung/data/push_pc1024/splits/episodes_train.json)" \
  --policy.type=pc_diffusion --policy.device=cuda --policy.push_to_hub=false \
  --batch_size=64 --steps=100000 --num_workers=6 --seed=1000 \
  --output_dir=/home/samsung/data/runs/pc_diffusion_v1 --job_name=pc_diffusion_v1 \
  --wandb.enable=true --wandb.project=pc_diffusion_push
```

Add `--policy.num_objects=1 --policy.aux_residual_weight=0.1 --policy.aux_predict_rotation=false`
(and point at `push_pc1024_poses`) to enable the auxiliary residual-pose head. See the
[`pc_diffusion` README](../src/lerobot/policies/pc_diffusion/README.md).

`--policy.push_to_hub=false` is required; otherwise `cfg.validate()` fails at startup demanding a
hub `repo_id`.

**Do not use `--dataset.eval_split` for the object holdout.** `make_train_eval_datasets` takes the
last `ceil(n × eval_split)` episodes *per task*, and this dataset has exactly one task — so it
would split on collection order, not on object. Pass the explicit episode list instead.

Throughput ≈ 12.5 steps/s → ~2 h 15 m for 100k steps (RTX 4090, batch 64).

### Memory budget — read this before raising `num_workers`

Measured on a 31.3 GiB host:

| item | size |
|---|---|
| dataloader worker RSS (measured in situ) | 1.52 GB × 6 = 9.1 GB |
| prefetch buffers (39.7 KB/sample × 64 × 2) | 5 MB/worker |
| training parent | ~4 GB |
| desktop reserve | ~8 GB |
| **total / headroom** | **~21 GB / ~10 GB** |

Launch through a wrapper that sets `oom_score_adj` so the kernel kills training rather than your
editor — children inherit it:

```bash
#!/bin/bash
echo 400 > /proc/self/oom_score_adj
exec "$LEROBOT_PY" -m lerobot.scripts.lerobot_train "$@"
```

> **The dataloader fast path is load-bearing.** `dataset_reader._take_numeric_column` gathers each
> column straight from Arrow, **taking rows from the chunk that owns them**. Calling `.take()` on
> the whole `ChunkedArray` instead combines every chunk first and materialises the entire column —
> ~8 GB for a 647k-frame point-cloud column, *in every worker*. That is a host OOM that takes the
> desktop session with it, and reducing worker count does not save you (5 × 7.36 GB is still 37 GB).
> Watch `data_s` in the training log: **~0.001 s** is the fast path, **0.17 s** means it fell back.

---

## Stage 5 — evaluate in IsaacLab

Two processes. Start the policy server first, then the harness.

```bash
# LeRobot side — holds the policy and its trained processor pipelines.
$LEROBOT_PY tools/pc_policy_server.py \
  --checkpoint /home/samsung/data/runs/pc_diffusion_v1/checkpoints/100000/pretrained_model \
  --dataset_root /home/samsung/data/push_pc1024 --repo_id local/push_pc1024 \
  --socket /tmp/pc_policy.sock
# wait for: PC_POLICY_SERVER_READY /tmp/pc_policy.sock

# IsaacLab side — held-out objects.
cd /home/samsung/3D_Bimanual_repo/scripts_isaaclab/irregular/eval
OMNI_KIT_ACCEPT_EULA=YES $ISAAC_PY eval_push_policy.py \
  --object_list /home/samsung/data/push_pc1024/splits/objects_test_paths.txt \
  --num_envs 21 --episodes_per_object 6 \
  --socket /tmp/pc_policy.sock \
  --output /home/samsung/data/runs/pc_diffusion_v1/eval_heldout.jsonl
```

Repeat with `objects_indomain_sample_paths.txt` for the in-domain number. The server runs the
*same* `make_pre_post_processors` pipeline objects training used, so normalization cannot drift.

Success is `pos_err_xy < 0.03 m` **and** `ori_err < 0.15 rad`.

### Recording rollout videos

```bash
  --record_video 6 --video_every 10 --video_dir <run>/videos_goal
```

The goal is drawn into the frames as a transparent green ghost of the object's own mesh, posed at
the goal, plus a 3 cm tolerance ring. It is **rasterised in post-processing, never spawned into the
scene** — a marker prim is real geometry the depth capture cameras bake into the point cloud, which
would change what the policy sees and invalidate the very rollout being visualised. (The collector
stopped spawning its own goal ghost mesh for exactly this reason.)
`eval/test_goal_overlay.py` pins the projection geometry without booting the simulator.

### Things that will bite you

| symptom | cause | fix |
|---|---|---|
| success collapses to 0% after batch 1 | `Articulation.reset()` does not restore joints, and the collector disables `reset_robot_joints` | call `rsp._drive_robot_home(...)` before `rsp._reset_env(...)` each batch |
| orientation error stuck high | `cmd.target_quat` left at identity | set it to the object's spawn orientation after reset |
| eval never exits, 130% CPU, 5.4 GB VRAM held | `simulation_app.close()` hangs | `os._exit(...)` instead; do not call `close()` |
| depth cameras produce nothing | `enable_cameras` derives from the video flags | pass `--tiled_video` even when not recording |
| only the first N objects evaluated | `env_obj_idx = [i % n_objects]` silently truncates | `num_envs` must be ≥ number of objects (the harness now hard-errors) |
| hangs waiting on a licence prompt | Isaac EULA | `OMNI_KIT_ACCEPT_EULA=YES` (this accepts the NVIDIA licence — read it first) |

---

## Stage 6 — compare

```bash
$LEROBOT_PY tools/compare_push_policies.py \
  --runs /home/samsung/data/runs/pc_diffusion_v1 /home/samsung/data/runs/pcd_diffusion_v1
```

Reports Wilson score intervals, difference-of-proportions CIs, expert-normalised scores,
difficulty stratification, and per-object breakdowns.

Point `--src_root` at the collection the policies were **trained on**. It defaults to
`push_data_old`; a newer collection's attempt log describes a different object set and would
silently produce the wrong expert reference.

### How to report a result

Four rules, each of which corrects a way these numbers mislead:

1. **Never quote a global expert average.** The all-object figure is 70.2%, but it includes four
   objects the expert never once solves. On the objects actually evaluated the expert scores
   **74.8% held-out** and **80.7% in-domain**. Always compare against the expert on the *same
   objects*, which is what the tool does.
2. **Exclude objects the expert never solves.** A policy failing where the demonstrator also fails
   is scoring the demonstrator. Only successful episodes are saved, so such objects contribute no
   training data at all. The tool prints the exclusion rather than applying it silently. (As of the
   current split, none of the four appear in either eval set, so nothing is dropped — but that is a
   property of the split to be re-checked, not a standing guarantee.)
3. **Prefer the expert-normalised score.** Excluding at 0% is a hard threshold, and an object the
   expert solves 10% of the time is nearly as unfair as one it never solves. Policy successes ÷
   expert successes over the same objects handles the whole range smoothly, with exclusion as its
   degenerate case.
4. **Normalise before comparing splits.** The in-domain object set is intrinsically easier for the
   expert (80.7% vs 74.8%), so a raw in-domain − held-out gap conflates object difficulty with
   memorisation. Normalising shrinks the measured gap from **+5.5 pp to +2.5 pp** — that is, most
   of the apparent generalization gap is the object set, not the policy.

Report three numbers, never one: in-domain, held-out, and the expert on each of those same sets.

### Results

Evaluation: 21 held-out objects × 6 episodes = 126; 19 in-domain objects × 6 = 114. No
expert-infeasible objects are present in either set.

| policy | params | in-domain | held-out | in-domain (norm.) | held-out (norm.) |
|---|---:|---:|---:|---:|---:|
| scripted expert (same objects) | — | 80.7% | 74.8% | 100% | 100% |
| `pc_diffusion` | 278.15 M | 49.1% (56/114) | 43.7% (55/126) | 60.9% | 58.4% |
| `pc_diffusion` + aux head | 278.49 M | **56.1%** (64/114) | **50.0%** (63/126) | **69.6%** | **66.9%** |
| `pcd_diffusion` | 47.33 M | 10.5% (12/114) | 9.5% (12/126) | 13.0% | 12.7% |

The auxiliary head is **+6.3 pp held-out / +7.0 pp in-domain**, but the 95% CIs are [−5.9, +18.6]
and [−5.9, +19.9] — not distinguishable from zero. At n=126 the interval around 43.7% is roughly
±8.7pp, and comparing two arms of that size resolves about ±12pp: enough for the
`pc_diffusion`/`pcd_diffusion` gap, not for this ablation. Treat it as suggestive and unreplicated.

## Reproducing from scratch

```
Stage 1  port + goal recovery      ~25 min    → push_pc1024 (8.9 GB)
Stage 2  object split              seconds
Stage 3  pose labels (optional)    ~15 min    → push_pc1024_poses (8.0 GB)
Stage 4  train                     ~2 h 15 m  → checkpoints/100000
Stage 5  eval (×2 splits)          ~1 h
Stage 6  compare                   seconds
```
