# Improving push-policy success — current plan

Supersedes the original planning document. The earlier blockers (goal not recorded, sim/real
schema alignment, held-out object split, live data collection on this machine) are all resolved or
out of scope, and are not repeated here.

## Where things stand

| | |
|---|---|
| best result | **43.5%** (DiT) on the curated 14-object held-out set |
| demonstrator ceiling | 91.9% on that set, 70.5% over all objects |
| training now | `push_dit_v3_all` — DiT 271 M, all 2115 episodes, commanded goal orientation |
| its aux head | **yes** — `num_objects=1`, `aux_residual_weight=0.1`, `aux_predict_rotation=True`, `aux_supervised_frac=1.000` (verified from the run's own log) |
| Tier 0 | **exhausted** — chunk length, denoising steps and the aux rotation term all measured null |

## Ground rules (set by the user)

- **Train on all data.** No held-out object split. Evaluation is **in-distribution**: every object
  is in training, and the simulator draws a *fresh random goal per episode*, so rollouts are scored
  on task instances the policy has never seen. 28-object screening set, 12 rollouts, n=336,
  paired resolution ~±7.5 pp. The 97-object full set exists for confirming finalists.
- **Goal construction**, all primitives alike: take the object's point cloud at the *initial* state
  and apply the commanded goal transformation. Never the achieved final pose — that bakes in the
  demonstrator's residual error (median 2.7°) and mismatches evaluation. This is what `push_v3`
  does (`--goal_orientation commanded`), and it is what rotate and flip must do too.
- **No sim + real co-training** until the user raises it again. The real-world port exists and is
  schema-compatible, but it is parked.
- **No data collection on this machine.** Rotate and flip datasets arrive from elsewhere; the user
  will say when. The schema contract they must meet: LeRobot v2.1, 10 fps,
  `trossen_aloha_bimanual`, `observation.object_pose` (7) per frame, same capture parameters,
  task string naming the primitive.
- **Freeze the split and the eval set.** Regenerating the object split from expert statistics
  silently reshuffled 10 of 21 objects between campaigns. Choosing an eval set to maximise the
  number is worth **+10.6 pp of pure selection bias**. Both are now copied verbatim, never
  regenerated.

## Diagnosis — why the Tier 0 nulls were predictable

1. **The demonstrator is open-loop on a fixed clock.** `record_mesh_push.py:84-90` hard-codes the
   phase schedule (200 approach / 600 push / 100 retreat). Median position error over the dataset
   goes 161 mm (frame 0) → 3.2 mm (frame 200) → 0 mm (frame 309), and the arms return to home.
   **There is no terminal servoing behaviour in the data to learn.** Accuracy is decided at contact
   selection (~frame 100-140). Every closed-loop-flavoured lever was therefore doomed; stop
   spending arms on that axis.
2. **Position is the binding gate.** Perfect orientation caps success at **60.1%**; perfect
   position caps it at **69.0%**.
3. **The gates differ in character.** Across samplers on one checkpoint, `corr(ori_err)=0.85` but
   `corr(pos_err)=0.57-0.62`. Orientation is ~85% determined by the weights — no sampling change
   can touch it. Position carries ~35-40% variance from the initial diffusion noise, **recoverable
   without retraining**.
4. **Sensitivity is steep**: a 10% reduction in error magnitude is worth **~+8 pp**.
5. **Two verified defects.**
   - The observation and goal clouds are MIN_MAX-normalised with *different* affine maps — y-span
     differs by **1.66×**, z-centres by **169 mm**. The policy's core "observed vs goal" comparison
     is being made across mismatched anisotropic frames.
   - The geometry pathway is **0.218 M parameters, 0.08% of the model**, while adaLN modulation
     alone is 106.5 M — a **489× misallocation**. The model spends vastly more capacity turning the
     observation vector into modulation coefficients than turning 3072 numbers of geometry into
     that vector.

## Phase 0 — evaluation protocol (do first, costs ~nothing)

- **Per-step error traces.** `eval_push_policy.py::_goal_live_state` (~line 1138) already computes
  `pos_err`/`ori_err` per env for the video HUD, but only for recorded envs. Extend to all envs
  every 10 steps and append to the results record (~8 lines). Every later eval then yields
  error-vs-time for free — the artefact that separates "stopped short" from "overshot".
- **Paired continuous margins as the powered endpoint.** Goals are bit-reproducible across runs.
  Report McNemar on success as the headline and paired log-`pos_err`/log-`ori_err` as the
  secondary; at n=336 the continuous endpoint resolves ~±10% on position, roughly twice the
  binary's power for small effects.

## Phase 1 — evaluate `push_dit_v3_all` (the run training now)

Only this checkpoint. The superseded ones sit on the old goal convention and the split-trained
protocol, so anything learned there would need re-confirming on the current design anyway.

- **E1. Baseline (K=1)** with traces, run at `--steps 3100` and scored at both 1550 and 3100. Gives
  the headline number, the error-vs-time curve, and "does more time help" in one run. ~2 h.
- **E2. Sample-averaged (K=8).** Same checkpoint, same goals — exactly paired with E1. The expert
  is deterministic, so `p(action|obs)` is unimodal and the conditional mean is the right estimator;
  DDIM with eta=0 is deterministic given `x_T`, so all sampling spread comes from the initial
  noise. Predicted ~0.83× position error → **+5 to +10 pp**, and predicted *not* to move
  orientation — an asymmetry that falsifies the decomposition if it fails. ~6 lines in
  `generate_actions` plus a `--num_samples` server flag. ~2 h.
- **D4. Offline probes**, no rollouts, ~1 h: split the aux readout into mm and degrees on held-out
  episodes and re-fit it from the point-cloud feature *alone*; per-phase action error vs frame
  index; bias-vs-variance split at fixed conditioning. These gate A3 and Phase 3.

**Postponed at the user's request:** multi-attempt rollouts (a second 310-step episode from the
displaced object). Likely a large gain, but it changes the task protocol.

## Phase 2 — training arms, one variable each (~5.5 h per arm)

**A1. Shared isotropic point-cloud frame** — config only, top priority.
`--policy.normalization_mapping='{"POINT_CLOUD":"IDENTITY","STATE":"MIN_MAX","ACTION":"MIN_MAX","VISUAL":"IDENTITY"}' --policy.pc_isotropic_rescale=true`
Fixes defect 5(a); `pc_isotropic_rescale` already applies one shared transform to both clouds in
`_encode_cloud` (`modeling_pc_diffusion.py:242-248`). Conditioning width unchanged, so the DiT
parameter count is unchanged — a genuinely clean single variable. Expect +2 to +8 pp.

**A2. EMA** — config only, self-paired. `--ema.enable=true`. Standard for diffusion policies and
never enabled on any `push_v2`/`push_v3` run. `lerobot_train.py:824-834` saves *both*
`pretrained_model` and `pretrained_model_ema`, so **one run yields a perfectly paired A/B** at
identical seed and data order. Point the server at `pretrained_model_ema` for the second eval
(`run_arm_indist.sh` hardcodes `pretrained_model`). Expect +2 to +5 pp.

**A3. Fat point encoder at fixed conditioning width** — config only.
`--policy.pc_encoder_kwargs='{"hidden_dims":[256,512,1024]}'` and the same for `goal_encoder_kwargs`.
Grows the geometry pathway ~13× while leaving `pc_feature_dim=256`, hence `cond_dim`, hence the DiT
parameter count **exactly unchanged**. This is the clean version of the bottleneck test that
`pc_feature_dim=512` cannot be (+30% params, confounds capacity). Gate on D4's pc-only probe.

**A4. Action normalisation scale** — config only, two mechanically coupled flags.
`ACTION:MEAN_STD` plus `--policy.clip_sample_range=8.0`. Normalised action std is ~0.30, i.e. 9% of
the unit variance diffusion assumes, mis-centring the SNR schedule. **Leaving
`clip_sample_range=1.0` would clip most of the action distribution and produce an uninterpretable
null** — gripper dims reach |z|=17.

**A5. Encoder inductive bias** — ~30 lines. Register an encoder wrapping the existing
`FourierFeatures` (`pcd_diffusion/modeling_pcd_diffusion.py:98`) + PointNet MLP + `cat(max, mean)`
pooling, keeping `out_dim=256`. Run after A3 so capacity and inductive bias read separately.

## Phase 3 — observation↔goal cross-attention (~4 h coding). **Run A1 first.**

This is the structurally right fix for the task's core computation — "where is the object relative
to where it should be" — and most of it already exists.

Today each cloud is independently max-pooled to 256 dims and the two summaries are concatenated;
the network must infer their relationship from two separately-compressed vectors. The DiT's *only*
observation pathway is one adaLN shift/scale/gate per block, identical across all 64 action tokens,
and its self-attention runs over action tokens alone
(`multi_task_dit/modeling_multi_task_dit.py:615-629`).

`pcd_diffusion` already implements the alternative:
- `PerceiverPointEncoder` (line 121) — learned latents cross-attend to points, **shared weights**
  for both clouds with a learned `type_emb` (line 145-146) distinguishing observation from goal.
- `DiffusionActionHead` (line 232) — concatenates `[obs latents ; goal latents ; proprio ; action]`
  and runs **joint self-attention**, so observation latents attend to goal latents directly. That
  *is* the obs↔goal cross-attention, realised as self-attention over a concatenated sequence.

Port that path into `pc_diffusion` **at the current 271 M budget**. The old 9.5% `pcd_diffusion`
result is no evidence against it: 4 latents, 128 dim, 47 M params, different input stack.

**Ordering matters: A1 must land first.** Cross-attention between the two clouds is only meaningful
once they share a metric frame. Under today's normalisation the y-span differs by 1.66× and the
z-centres by 169 mm, so attention would be learning correspondences across a distorted map.

## Real-Time Chunking — a deployment concern, not a simulation lever

RTC exists in the repo (`policies/rtc/`, `rollout/inference/rtc.py`, 602 lines; used by `evo1` and
`molmoact2` via an `rtc_config` field). It runs a background thread producing chunks asynchronously
while the control loop polls `get_action`, with a `LatencyTracker` and soft-masking of the
already-committed prefix.

**It should not change the simulation number.** Our evaluation loop is synchronous — `client.act()`
blocks, the policy answers in 3-5 ms, and the simulator waits — so there is no latency to hide.
RTC's benefit is on real hardware, where the control loop runs at a fixed rate and inference time
is real.

The one sim-relevant angle is that RTC also smooths chunk boundaries, which could in principle
allow more frequent replanning without the discontinuity penalty we measured (`n_action_steps=8`
was −8.3 pp). But finding 1 says the demonstrator is open-loop with no terminal servoing, so more
frequent replanning has nothing to exploit. **Recommendation: adopt RTC when deploying to hardware,
not as a simulation experiment.** Integration also needs an `rtc_config` field on
`PCDiffusionConfig` plus `init_rtc_processor()`, and it is built around the `lerobot-rollout`
harness rather than the IsaacLab evaluator.

## Dead levers — measured or derived; do not spend arms on these

| Lever | Why not |
|---|---|
| `QUANTILES` for ACTION | Arithmetic no-op: q01≈min, q99≈max on all 14 dims (span shrink 0.97-0.9997) |
| `pc_feature_dim` 256→512 | +30% DiT params, so it measures capacity; use A3 instead |
| `do_mask_loss_for_padding` | Padded windows sit entirely inside the deterministic retract, where action std has collapsed to 0.010 |
| `extra_state_keys=[velocity]` | `action[t]≈state[t+1]` and n_obs_steps=2 already give a finite difference; bundle it, don't spend an arm |
| `n_obs_steps` > 2 | Quasi-static task; +15% DiT params at 3 |
| chunk length / denoising steps / aux rotation | Already measured null, and explained by the open-loop demonstrator |

## Two ceilings to keep in view

- **The teacher's own orientation margin is thin**: final orientation error in the training data is
  median 0.048 rad but **p90 0.142 rad** against a 0.15 rad gate. Perfect imitation gives an
  orientation gate of ~0.90, not 1.0.
- **80% stays out of reach** in this scope: it needs ~0.5× on both error distributions, i.e. ~1.5×
  the teacher's own accuracy. Honest near-term target is **low-to-mid 60s**.

## Verification

- After each arm: `tools/compare_push_policies.py --runs <new> <baseline>` for the paired McNemar
  and interval, plus the paired continuous margins from Phase 0.
- A1 pre-check: encode one batch and print per-axis min/max of both clouds to confirm they land in
  the same range under `pc_isotropic_rescale`.
- A3 pre-check: assert `global_cond_dim` and total DiT parameter count are unchanged vs
  `push_dit_v3_all`, so the arm is genuinely single-variable.
- E2 falsification: sample averaging must cut `pos_err` materially while leaving `ori_err` roughly
  unchanged. If it moves both, the sampler/weights decomposition is wrong and Phase 2 priorities
  need revisiting. If it moves neither, the position-variance estimate was wrong.
- Sample-averaging correctness: `--num_samples=1` must be bit-identical to the current path, and
  wall-clock should rise well under K× (it batches K copies of one conditioning vector).
- Regression guard: `pytest tests/policies/pc_diffusion -q` (16 tests) after any code change.

## Immediate sequence

1. `push_dit_v3_all` finishes training (~18:00); the halt watcher stops it before evaluation.
2. Implement Phase 0 traces (~8 lines) and K-sample averaging (~6 lines + server flag); confirm
   `--num_samples=1` is bit-identical to today's behaviour.
3. Run E1 (baseline, traces, `--steps 3100`) then E2 (K=8) — paired, same goals. ~4 h.
4. Run the D4 offline probes.
5. Choose the first training arm from A1 / A2 / A3 from what E1/E2/D4 say, not by guess.
6. A1 (shared frame) before Phase 3 (obs↔goal cross-attention), since the latter is only meaningful
   once both clouds share a metric frame.

## Open question worth settling on the current run

`aux_residual_loss` is already **0.000** on `push_dit_v3_all`. Under the commanded-goal convention
the residual-rotation target's variance collapsed 10× (0.00796 → 0.00079), so six of the head's
nine outputs are now near-constant on push. The head is therefore close to inert here — but it
should become genuinely informative on rotate and flip, where residual rotation *is* the task.
Suggest making `aux_predict_rotation` a per-task setting rather than a global one when those
datasets arrive, and re-testing the head's value then rather than on push.
