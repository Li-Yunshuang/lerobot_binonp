# Training recipes: `aux_v1` and `aux_v1 + point-cloud prediction`

Two runnable configurations for `pc_diffusion` on the push primitive, written so they can be
launched on a machine that has never seen this campaign.

- **A. `aux_v1`** — the current best push policy. 78.6% under the
  [benchmark protocol](./push_benchmark_protocol.md).
- **B. `aux_v1 + future-latent`** — same recipe with the residual-pose head **replaced** by a
  head that predicts the encoder's own latent for the observation 32 frames ahead.

Evaluate with [`push_benchmark_protocol.md`](./push_benchmark_protocol.md); results collect in
[`push_scorecard.md`](./push_scorecard.md).

---

## 0. Read this first — three defaults that will silently ruin a run

`aux_v1` was trained before several config fields existed. Re-running it with today's defaults
does **not** reproduce it. All three are already pinned in `tools/train_aux_recipe.sh`; they are
listed here because anyone writing their own command will hit them.

| field | today's default | what `aux_v1` needs | what goes wrong |
|---|---|---|---|
| `num_objects` | `0` | `1` (config A) | the auxiliary head is not built at all — silently, no error |
| `use_task_onehot` | `True` | `false` | the field postdates `aux_v1`; leaving it on changes the conditioning width |
| `output_dir` | — | must **not** exist | `lerobot_train` raises `FileExistsError` if it does; do not `mkdir` it |

A fourth, specific to config B: **the dataloader must be built from the same config object as the
model.** Enabling `future_latent_weight` changes `observation_delta_indices`, which is what makes
the dataloader fetch the future frame. Constructing them separately gives a batch with no future
frame, and the model raises rather than training silently without a target.

## 1. Environment and data

```bash
uv sync --locked --extra all     # or the project's usual install
git lfs install && git lfs pull
```

Dataset: `push_pc1024_poses` — 2088 episodes, 10 fps, LeRobot v2.1. Required keys:

| key | shape | A | B |
|---|---|---|---|
| `observation.state` | (14,) | ✓ | ✓ |
| `observation.point_cloud` | (1024, 3) | ✓ | ✓ |
| `observation.goal_point_cloud` | (512, 3) | ✓ | ✓ |
| `observation.object_poses` | (1, 4, 4) | ✓ | ✓ |
| `observation.goal_object_poses` | (1, 4, 4) | ✓ | ✓ |
| `action` | (14,) | ✓ | ✓ |

Both configs need the two pose keys: A supervises the residual-pose head with them, B uses them to
locate the object for its masked target (§3). A dataset without them can still run **B** with
`--policy.future_latent_object_only=false`, at the cost described in §3.

`aux_v1` trained on a **1619-episode subset**, not all 2088. To reproduce it exactly, carry
`aux_v1_episodes.json` across and pass it as `EPS_FILE`. Training on all 2088 is a reasonable
choice, but it is then a different run and should not be compared to the 78.6% figure directly.

## 2. Config A — `aux_v1`

```bash
EPS_FILE=/path/to/aux_v1_episodes.json \
tools/train_aux_recipe.sh aux_v1_repro
```

Or explicitly, if not using the runner:

```bash
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/push_pc1024_poses \
  --dataset.root=/path/to/push_pc1024_poses \
  --dataset.episodes="$(cat aux_v1_episodes.json)" \
  --policy.type=pc_diffusion --policy.device=cuda --policy.push_to_hub=false \
  --policy.goal_conditioning=points \
  --policy.use_task_onehot=false \
  --policy.num_objects=1 \
  --policy.aux_residual_weight=0.1 \
  --policy.aux_predict_rotation=false \
  --batch_size=64 --steps=100000 --num_workers=6 --seed=1000 \
  --log_freq=200 --save_freq=20000 \
  --output_dir=/path/to/runs/aux_v1_repro --job_name=aux_v1_repro
```

Everything else is a current default: UNet backbone, `down_dims=[512,1024,2048]`, `horizon=64`,
`n_action_steps=32`, `n_obs_steps=2`, DDIM with 10 inference steps, epsilon prediction, MIN_MAX
normalization, `pointnet_maxpool` at `pc_feature_dim=256`.

**≈278.5 M parameters, ~2.5 h on one 24 GB GPU** (~11 step/s), ~27 GB of checkpoints at
`save_freq=20000`.

**Expect `aux_residual_loss` to sit at 0.000.** That is not a bug in the run — under the
commanded-goal convention the residual target's variance collapsed ~10x and the head has nothing
to learn. It is the reason config B exists. Measured head-to-head against an otherwise identical
no-head model on 336 paired rollouts, the residual head is worth **−0.3 pp (p=1.000)**: on push it
is inert. Keep it enabled for continuity into rotate/flip, where residual rotation *is* the task
and the target will not be degenerate.

## 3. Config B — `aux_v1 + future-latent prediction`

```bash
EPS_FILE=/path/to/aux_v1_episodes.json \
tools/train_aux_recipe.sh aux_v1_futlat \
  --policy.num_objects=0 \
  --policy.future_latent_weight=0.1 \
  --policy.future_latent_horizon=32 \
  --policy.future_latent_object_only=true \
  --ema.enable=true
```

`num_objects=0` **replaces** the residual-pose head rather than stacking both, so the single
variable versus config A is the head swap.

**What the head does.** From the same conditioning vector the U-Net sees, an asymmetric predictor
regresses the encoder's own latent for the observation 32 frames ahead. Loss is negative cosine
similarity against that latent.

**Why horizon 32 rather than 1.** The cloud contains arms as well as the object, and the object is
stationary outside the push phase, so at 10 fps a t+1 target is dominated by arm motion the policy
already predicts from proprioception. Measured centroid shift: **11.2 mm at 1 frame, 33.2 mm at
32**. 32 also matches the action chunk the policy commits to.

**Why the target is object-masked.** Unmasked, the cheapest way to satisfy this head is to model
the robot. The dataset has no per-point labels — segmentation exists only inside the simulator at
collection time, and the port reads an already-merged cloud — so the object region is derived from
stored poses: the goal cloud is the object's geometry at the goal, and
`goal_object_poses − object_poses` is the displacement still to come, so subtracting it from the
goal cloud's centroid locates the object now. Validated on `push_pc1024_poses`: the K nearest
points to that centroid span **86–88% of the object's true extent**, stably across an episode. Kept
points are tiled back to the original cloud length, which a max-pool is invariant to, so the
encoder never sees an out-of-distribution input size.

### Watch `future_latent_std`, not the loss

The target comes from the **same encoder being trained**, so the loss is minimised by a constant
encoder output. That encoder is shared with the policy conditioning, so a collapse would not merely
trivialise the auxiliary task — it would destroy the policy's observation features while the loss
looked healthy.

Three guards are on by default: the target is detached (`future_latent_stop_grad`), the predictor
is asymmetric (online side only, SimSiam), and the loss is a scale-invariant cosine.

**`future_latent_std` is logged every step and is the number to watch.** If it trends toward zero
the representation is collapsing — kill the run rather than waiting for the evaluation. A healthy
run holds it roughly flat.

```bash
grep -oE "future_latent_std:[0-9.]+" train_aux_v1_futlat.log | tail -20
```

### Honest expectation

This has not been evaluated yet. Seven levers have been tested on this policy and **all seven came
back null** within the ±6 pp resolution of one evaluation: backbone architecture, normalization
frame, EMA, sample averaging, goal coverage, the residual-pose head, and observation↔goal
cross-attention. Every UNet variant lands between 75.4% and 80.4%, and they fail on the *same*
objects — `Dell_Ink_Cartridge` is 0/60 across all of them.

The mechanism here is sound, and unlike the residual-pose head the target carries real variance.
But the base rate says treat a small positive as noise until it replicates. §6 of the benchmark
protocol has the specific traps.

## 4. Verify before spending the GPU

```bash
pytest tests/policies/pc_diffusion -q          # 35 tests, CPU, ~25 s
```

Then confirm the flags actually reached the process — a typo in a `--policy.*` flag is accepted
silently by some shells and wastes the whole run:

```bash
tr '\0' '\n' < /proc/$(pgrep -f lerobot_train | head -1)/cmdline | grep -E "policy\.|ema\."
```

For config B, confirm in the first 200 steps that `future_latent_loss` appears in the log at all.
If it is missing, the dataloader was not built from the model's config and there is no future frame
in the batch.

## 5. Checkpoints and disk

`save_freq=20000` writes 5 checkpoints. With `--ema.enable=true` each is ~5.4 GB
(1.1 model + 1.1 EMA + 3.2 optimizer state), so **~27 GB per run**. After evaluating step 100000,
the intermediates and `training_state` can go, leaving ~2.2 GB:

```bash
rm -rf runs/<name>/checkpoints/0{2,4,6,8}0000 runs/<name>/checkpoints/100000/training_state
```

With EMA the run produces **two** models, `pretrained_model` and `pretrained_model_ema`. EMA is
passive — it never touches the training trajectory — so evaluating both gives a paired A/B at
identical seed and data order for free. Measured on an earlier arm: **+2.1 pp (p=0.40)**, i.e.
cheap and mildly positive, not a lever.
