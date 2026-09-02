# Primitives training playbook

The standard process for training and evaluating point-cloud diffusion policies on a
manipulation primitive (push today; rotate and flip next), written to be followed on a machine
that has never seen this campaign. Everything here is the corrected-stack procedure as of
2026-08-31; older documents that disagree carry a superseded banner pointing back here.

**The standard configurations** (all action-head-only — no auxiliary prediction). The full 2×2
over the two goal pathways, n=276 each, paired goals, corrected harness:

| config | goal cloud | goal vector | cross-attention | primary / strict |
|---|---|---|---|---|
| **reference: cross-attention** | ✓ | ✓ | ✓ | **83.3% / 72.8%** |
| **cross-attention, no vector** | ✓ | — | ✓ | **82.2% / 72.5%** |
| no cross-attention (baseline) | ✓ | ✓ | — | 79.3% / 67.4% |
| neither | ✓ | — | — | 80.8% / 68.8% |

Two effects, no interaction: **cross-attention is worth ~+5 pp at the strict gate in both columns**
(+5.4 pp p=0.036 with the vector, +5.1 pp p=0.065 without); **the goal vector is worth ~0 in both
rows** (−0.4 pp p=1.0 inside the full method). On push, the goal *cloud* carries the whole task.
Caveat before generalising: push's commanded rotation is ~identity, so the vector's six rotation
dims were near-constant here — on **rotate/flip they carry the task**, and the vector must be
re-ablated there rather than dropped by inheritance. For push deployment, cross-attention without
the vector is the simplest recipe at full strength.

---

## 0. Setup

| | |
|---|---|
| policy repo | `lerobot_binonp`, branch `campaign/pc-diffusion-push` |
| sim/eval repo | `3D_Bimanual_repo`, branch `data-collection`, at or after `6baaae8` (the eval settle fix) |
| python envs | **two, they cannot merge**: `lerobot` (py3.12) for training/porting, `isaaclab` (py3.11) for collection/evaluation |
| install | `uv sync --locked --extra all` in the lerobot repo; IsaacLab per its own docs |
| GPU | 24 GB is enough for every config here (peak ~13 GB train, ~15 GB eval) |
| disk | ~9 GB per ported dataset, ~27 GB transient per training run (pruned to ~2.2 GB after eval) |

## 1. Data: collect → clean → port → inspect

Authoritative detail: [`primitive_dataset_pipeline.md`](./primitive_dataset_pipeline.md) — every
cleaning step with the failure it prevents, the verification gates, and the rotate/flip
checklist. The short form:

**1.1 Collection contract** (what the collector must record per frame): 4096-pt capture cloud
(arm ∪ object, table-free), `observation.object_pose` (7, ground truth), state/velocity/action
(14), plus `meta/attempts.jsonl` including failures.

**1.2 Port** (cleaning happens here):

```bash
python examples/port_datasets/port_isaaclab_pointcloud_push.py \
  --src_root <collection_dir> --dst_root /data/<name> --repo_id local/<name> \
  --num_points 1024 --num_workers 6
# defaults: --goal_orientation commanded, object-only clouds (arm removal), absolute actions
```

What the port does, in order: sanitises implausible poses (fallen objects) and flags them via
`pose_valid`; builds the goal from the **commanded** transform, never the achieved one; isolates
the object at t=0; keeps only object-surface points per frame (the deployable equivalent of
RGB-segmentation → depth); crops and resamples to 1024; writes `observation.goal_transform`
(9,) = `[Δt_world, rot6d(R_goal·R₀ᵀ)]` — the command itself, world frame; writes absolute joint
actions. (Do **not** use `--action_mode delta` with MIN_MAX normalisation: measured 9.1% vs
76.8% — the delta distribution's outliers crush typical actions to ~13% of the normalised
range.)

**1.3 Verify** (the chains run this automatically; on a new machine run it once by hand):

```python
i = json.load(open(f"{root}/meta/info.json"));  s = json.load(open(f"{root}/meta/stats.json"))
assert i["total_episodes"] == EXPECTED and "observation.goal_transform" in i["features"]
assert abs(np.array(s["action"]["max"])).max() > 0.5     # absolute actions, not delta
```

**1.4 Inspect before spending GPU**:

```bash
python3 tools/inspect_inputs.py --dataset_root /data/<name> --out inputs.html
```

One self-contained page: animated clouds in the normalised workspace box, the goal_transform
arrow, joint/action traces, object path, per-field encoding table. Review checklist is §4 of the
pipeline doc. For rotate/flip specifically: confirm the rot6d dims of `goal_transform` are
non-identity and vary across episodes, and (flip) that Δz is non-zero.

## 2. Training: the three standard configurations

Common base — every run uses exactly this, plus the per-config flags below:

```bash
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/<name> --dataset.root=/data/<name> \
  --policy.type=pc_diffusion --policy.device=cuda --policy.push_to_hub=false \
  --policy.backbone=dit \
  --policy.normalization_mapping='{"POINT_CLOUD":"IDENTITY","STATE":"MIN_MAX","ACTION":"MIN_MAX","VISUAL":"IDENTITY"}' \
  --policy.pc_isotropic_rescale=true \
  --policy.use_task_onehot=false \
  --policy.num_objects=0 \
  --policy.action_space=absolute_joint \
  --ema.enable=true --batch_size=64 --steps=100000 --num_workers=6 --seed=1000 \
  --log_freq=200 --save_freq=20000 \
  --output_dir=/data/runs/<run_name> --job_name=<run_name>
```

`num_objects=0` is what makes these action-head-only. `pc_isotropic_rescale` maps both clouds by
the shared capture-workspace transform `(p − [0,0,0.285]) / 0.40` — one frame, isotropic,
data-independent. `output_dir` must not pre-exist. Train on **all** episodes (no `--dataset.episodes`).

**2.1 Reference — cross-attention on current + goal cloud, with goal vector:**

```bash
  --policy.goal_conditioning=both \
  --policy.goal_feature_key=observation.goal_transform \
  --policy.pc_cross_attention=true
```

The two clouds pass through one joint encoder whose points cross-attend bidirectionally before
max-pooling (`encoders/cross_attention.py`; zero-initialised output projections, shared
per-point MLP with a type embedding, output width identical to the two independent encoders so
the denoiser is unchanged). The goal vector is concatenated into the conditioning alongside.

**2.2 Ablation — without cross-attention** (independent PointNetMaxPool per cloud):

```bash
  --policy.goal_conditioning=both \
  --policy.goal_feature_key=observation.goal_transform
```

**2.3 Ablation — without goal vector** (goal enters as a point cloud only). Combine
`--policy.goal_conditioning=points` with §2.1's cross-attention flag for the strong no-vector
config, or with §2.2 (no cross-attention) for the both-off cell:

```bash
  --policy.goal_conditioning=points
```

Timing at 24 GB: no-xattn ~3¾ h (7.3 step/s); with cross-attention ~6¾ h (4.2 step/s). Each run
saves `pretrained_model` and `pretrained_model_ema`; **evaluate the EMA weights**. After
evaluation, prune `checkpoints/0{2,4,6,8}0000` and `checkpoints/100000/training_state`.

Model shape (reference config): 2-frame conditioning
[state 14 ‖ obs-cloud 256 ‖ goal-cloud 256 ‖ goal_transform 9] × 2 → adaLN-Zero into a 13×1024
RoPE DiT over 64 action tokens; ε-prediction, DDIM-10 at inference; 64-step chunk, 32 executed.

## 2b. Benchmark baselines (other method families, same inputs and protocol)

Both are pure configurations -- no code beyond what the repo carries. Use the §2 common base
command with `--policy.type`/flags below; evaluate identically to §3. Push results (n=276):

**ACT** (Zhao et al. 2023; `pc_act` policy -- stock ACT head on our encoders): 63.0% / 42.0%.
Cross-attention (`--policy.pc_cross_attention=true`) does nothing for it (63.0% / 40.6%) --
the correspondence gain is diffusion-head-specific.

```bash
  --policy.type=pc_act            # chunk 64 / execute 32, MEAN_STD, lr 1e-4 are its defaults
```

**DP3** (Ze et al. 2024, adapted): 51.8% / 25.0%. Faithful pieces: 64-d compact encoders, DP
UNet1D, sample prediction, DDIM 100/10, To=2. Adaptations: goal cloud through a second 64-d
encoder (the task is goal-conditioned; original DP3 has no goal input), H=8/Ta=6 instead of
4/3 (our UNet needs horizon % 8 == 0), state concatenated raw, our standard budget.

```bash
  --policy.type=pc_diffusion --policy.backbone=unet --policy.down_dims='[256,512,1024]' \
  --policy.horizon=8 --policy.n_action_steps=6 \
  --policy.pc_feature_dim=64 --policy.prediction_type=sample \
  --policy.normalization_mapping='{"POINT_CLOUD":"IDENTITY","STATE":"MIN_MAX","ACTION":"MIN_MAX","VISUAL":"IDENTITY"}' \
  --policy.pc_isotropic_rescale=true --policy.goal_conditioning=points \
  --policy.use_task_onehot=false --policy.num_objects=0 --policy.action_space=absolute_joint
```

## 3. Evaluation

Protocol (frozen): [`push_benchmark_protocol.md`](./push_benchmark_protocol.md). Harness must be
at/after `6baaae8` — earlier harnesses sample goal clouds before physics settles (up to ~9 cm
high), which depressed strict-gate scores ~5 pp for every model.

```bash
RECORD_VIDEO=6 tools/eval_ckpt_video.sh \
  <run>/checkpoints/100000/pretrained_model_ema \
  /data/<name> local/<name> \
  <frozen object list> <run>/eval_e23_ema_fixed.jsonl 12 1
python3 tools/scorecard.py -o docs/push_scorecard.md   # also refreshes docs/push_experiments.md
```

The harness verifies its contract against the checkpoint at startup (schema, `action_space`) and
aborts on mismatch. Goals are deterministic in (seed, object list): two runs on the same list are
paired slot-for-slot — report paired differences, not two independent rates. Every rollout
records per-step error traces, so success-at-any-horizon and ever-in-goal are computable
after the fact.

**For rotate/flip**: the evaluation harness is push-specific today (`eval_push_policy.py` — goal
command, success gates, ghost). Training transfers as-is; evaluation needs a per-primitive
variant: its own goal command term, its own frozen object list (build once from demonstrator
competence, then freeze the file — never regenerate), and its own success gates (push's
50 mm / 0.20 rad do not automatically transfer).

## 4. Results live in two auto-generated files

- [`push_scorecard.md`](./push_scorecard.md) — grouped comparisons with pairing verification.
- [`push_experiments.md`](./push_experiments.md) — one row per evaluation: full setup
  (stack/backbone/actions/heads/normalisation) and all metrics (three gates with CI, per-axis
  gates, ever-in-goal, mean/median/std/p90 both axes).

Never edit either by hand; rerun `tools/scorecard.py`.

## 5. Traps that have each cost a run

- `output_dir` pre-exists → `FileExistsError` at step 0.
- `num_objects` defaults to 0 (no aux head); older docs said 1 — for these configs 0 is correct.
- `use_task_onehot` defaults to **true**; these single-primitive configs need `false`.
- delta actions + MIN_MAX (see §1.2). If you must use deltas, that requires MEAN_STD — untested.
- Editing a bash chain script **while it runs** corrupts it (bash reads by byte offset). Python
  files are safe to edit while a process runs; shell scripts are not.
- Process-wait loops keyed on command-line *strings* can match stale shells and hang forever;
  wait on PIDs.
- The dataset root passed to the eval server must be the one the checkpoint **trained on** — it
  carries the normalisation statistics.
