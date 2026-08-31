# Baseline: `push_dit_objabs`

The reference method for the push primitive, frozen 2026-08-29. Modules built to improve
success rates are measured **against this**, on the protocol below, changing one thing at a
time. This document is the map; the linked docs carry the depth. The end-to-end procedure for a
new machine (rotate/flip included) is [`primitives_training_playbook.md`](./primitives_training_playbook.md).

| | |
|---|---|
| checkpoint | `/home/samsung/data/runs/push_dit_objabs/checkpoints/100000/pretrained_model_ema` |
| code | lerobot `campaign/pc-diffusion-push`, 3D_Bimanual_repo `6baaae8` |
| dataset | `push_objonly` (2115 episodes, all trained on) |
| primary score | **see `push_scorecard.md`** — corrected-harness number (`eval_e23_ema_fixed`); the earlier 76.8% was measured with the floating-goal-cloud eval bug and is superseded |

## 1. Dataset pipeline

Full procedure and rationale: [`primitive_dataset_pipeline.md`](./primitive_dataset_pipeline.md).
The one-command review tool: `tools/inspect_inputs.py`. Summary of what the baseline trains on:

- **Object-only observation clouds** (1024 pts): arms removed via object-surface segmentation —
  the deployable equivalent of RGB-segmentation → depth projection on hardware.
- **Goal cloud** (512 pts): the isolated t=0 object re-posed by the *commanded* transform.
- **`goal_transform`** (9): `[Δt_world, rot6d(R_goal·R₀ᵀ)]`, initial→goal — the command itself,
  so it exists at deployment with no pose tracking.
- **Absolute joint actions** (14). Delta actions (`action − state`) were tried and failed
  catastrophically (9.1% vs 76.8%, paired): MIN_MAX over the delta distribution crushed typical
  actions to ~13% of the normalised range. Deltas remain viable only with MEAN_STD normalisation
  — untested.
- Pose sanitisation, verification gates, and the rotate/flip checklist are in the pipeline doc.

## 2. Model

270.5 M parameters. Details confirmed by instantiation, not description:

| stage | |
|---|---|
| encoders | 2 × PointNetMaxPool (per-point MLP 64→128→256 + LayerNorm, global max-pool, projection), untied, 256-d each; clouds first pass the shared workspace map `(p − [0,0,0.285]) / 0.40` |
| conditioning | concat [state 14, obs cloud 256, goal cloud 256, goal_transform 9] × 2 obs steps = **1070** |
| denoiser | **DiT**: 64 action tokens (linear 14→1024), 13 layers, 8 heads, RoPE; conditioning + 256-d timestep embed drive **adaLN-Zero** per block; ε-prediction, `squaredcos_cap_v2`, 100 train / 10 DDIM inference steps |
| output | 64×14 chunk, execute 32; `action_space="absolute_joint"` in the checkpoint contract |
| heads | **action head only** — no auxiliary prediction (`num_objects=0`). Auxiliary heads return as separate arms so their effect is attributable |
| normalisation | `POINT_CLOUD: IDENTITY` + isotropic workspace rescale; `STATE`/`ACTION`: MIN_MAX |

## 3. Training recipe

```bash
python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/push_objonly --dataset.root=/home/samsung/data/push_objonly \
  --policy.type=pc_diffusion --policy.device=cuda --policy.push_to_hub=false \
  --policy.backbone=dit \
  --policy.normalization_mapping='{"POINT_CLOUD":"IDENTITY","STATE":"MIN_MAX","ACTION":"MIN_MAX","VISUAL":"IDENTITY"}' \
  --policy.pc_isotropic_rescale=true \
  --policy.goal_conditioning=both --policy.goal_feature_key=observation.goal_transform \
  --policy.use_task_onehot=false --policy.num_objects=0 \
  --policy.action_space=absolute_joint \
  --ema.enable=true --batch_size=64 --steps=100000 --num_workers=6 --seed=1000 \
  --save_freq=20000 --output_dir=<runs>/<name> --job_name=<name>
```

~3¾ h on one RTX 4090-class GPU (7.3 step/s), ~27 GB of checkpoints at peak (prune intermediates
+ `training_state` after evaluating; ~2.2 GB remain). `output_dir` must not pre-exist. Evaluate
the **EMA** weights; the run saves both.

Trap list (each has burned a run): `num_objects` defaults to 0 but older recipes needed 1;
`use_task_onehot` defaults to true; a delta-action checkpoint evaluated as absolute (or vice
versa) is prevented by the `action_space` contract — never bypass it.

## 4. Evaluation

Protocol (frozen): [`push_benchmark_protocol.md`](./push_benchmark_protocol.md) — 23-object list,
fresh goals per episode, 12 rollouts/object = 276, primary gate 50 mm / 0.20 rad, strict
30 mm / 0.15 rad alongside, paired stats via `goal_dist_m` verification, scorecard via
`tools/scorecard.py`.

```bash
RECORD_VIDEO=6 tools/eval_ckpt_video.sh <ckpt>/pretrained_model_ema \
  /home/samsung/data/push_objonly local/push_objonly \
  /home/samsung/data/push_v3/splits/objects_e23_paths.txt <run>/eval_e23_ema.jsonl 12 1
```

The harness (3D_Bimanual_repo `6baaae8`) now also provides:
- **settled batch starts** — physics steps before any read; pre-`6baaae8` evals sampled goal
  clouds mid-air (up to ~9 cm high), an eval-only skew for every historical number;
- **per-step error traces** (`trace_pos_m`/`trace_ori_rad`, 1 s resolution) — success at any
  horizon from one run;
- **solid-mesh goal ghost** in videos, alignment-verified (`--ghost_selftest`);
- contract checks that fail at startup on any schema/action-space mismatch.

## 5. Known headroom, for module design

- **Stopping**: ~8% of rollouts enter the goal region and drift out (traces show it at every
  gate). The demonstrator never demonstrates reactive holding — its stop is a schedule. This is
  the most concentrated single bucket.
- **Flat objects** remain the hard tail (shape↔success r = +0.61 on the old stack; re-measure on
  this one). The expert solves them (~94%), so it is a learning gap, not a task limit.
- **Old-stack ablations do not transfer**: every pre-2026-08-29 null (cross-attention, EMA
  magnitude, K-sample averaging, auxiliary heads, goal replay) was measured on arms-in-cloud,
  mismatched-normalisation, absolute-action inputs — all are open questions on this stack.

## 6. Comparison rules

New modules: train on `push_objonly` with the recipe above ± the module, evaluate on the e23
protocol, report the paired difference (goals are identical given the seed and list). Same-run
comparisons (EMA vs non-EMA) resolve ~±7 pp at n=276; cross-run single differences under ~10 pp
deserve a replicate before belief. Numbers from before the harness fix are not comparable to
numbers after it.
