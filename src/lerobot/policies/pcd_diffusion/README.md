# `pcd_diffusion` — goal-conditioned point-cloud diffusion policy

Perceiver encoder over point clouds feeding a transformer (DiT) diffusion head, following the
policy-learning recipe of [arXiv:2606.13677 (Mana)](https://arxiv.org/abs/2606.13677), with three
deliberate divergences: goal conditioning, an auxiliary residual-pose loss, and action chunking.

For the full data → training → evaluation workflow, see
[`docs/push_pointcloud_pipeline.md`](../../../../docs/push_pointcloud_pipeline.md).

## Dataset contract

All key names are config fields; these are the defaults. Everything is `float32`.

| key | shape | role |
| --- | --- | --- |
| `observation.point_cloud` | `(N, 3)` | scene cloud, XYZ |
| `observation.goal_point_cloud` | `(M, 3)` | target-configuration cloud |
| `observation.state` | `(S,)` | proprioception |
| `observation.object_poses` | `(K, 4, 4)` | **label only** — current object poses |
| `observation.goal_object_poses` | `(K, 4, 4)` | **label only** — goal object poses |
| `action` | `(A,)` | |

The two pose keys are training labels, not inference inputs. LeRobot types every non-image
`observation.*` key as an input feature, so `validate_features()` strips them from
`input_features` — otherwise they would be normalised and expected at inference time.

Set `num_objects=0` (the default) when the dataset has no pose labels; that disables the
auxiliary head entirely and removes the requirement for those two keys.

## Architecture

```
observation.point_cloud (B,T,N,3) ─┐
                                   ├─ shared PerceiverPointEncoder ─→ obs latents  (B,T,L,D)
observation.goal_point_cloud (B,M,3)┘  (+ learned obs/goal type emb)  goal latents (B,L,D)

observation.state (B,T,S) ─ MLP[512,256,256] ─→ proprio token (B,1,D)

  context = [obs latents | goal latents | proprio] ─ Linear(D→H) ─┐
                                                                   ├─ DiT self-attention
                                  noisy action tokens (B,horizon,H)┘  AdaLN-Zero(timestep)
                                                                   └─→ eps-hat, action slice
```

The Perceiver's cost is linear in point count — the quadratic attention is over `num_latents`
only — which is what makes attending over hundreds of points affordable. The same encoder
weights process both clouds; a learned type embedding distinguishes them.

Context is kept as *tokens* rather than pooled into one vector, so each action step can attend to
whichever latent is relevant.

### The scene bottleneck is `perceiver_num_latents × perceiver_dim`

Every bit of spatial information reaches the decoder through that many floats per timestep. The
paper's `4 × 128 = 512` was tuned for a single articulated tool; a scene with two 7-DoF arms plus
an object needs more. Raising `perceiver_num_latents` is by far the cheapest way to widen it
(4→16 costs ~0.02M parameters and quadruples bandwidth), and is usually a better first move than
deepening or widening the DiT.

## Point-cloud normalisation happens in the model, not the processor

`make_pcd_diffusion_pre_post_processors` drops the cloud and pose keys from `dataset_stats`, and
`normalization_mapping["POINT_CLOUD"] = IDENTITY`. Both, deliberately: a per-element MIN_MAX over
an `(N, 3)` array is silently wrong — it makes normalisation depend on point *index order* — and
nothing downstream would flag it.

The model then centres **both** clouds on the *observation* cloud's centroid. This is the part
that matters: centring the goal on its own centroid would express it in a different origin from
the observation and destroy the very displacement the policy needs to read. Scale is either
`pointcloud_scale` or, when that is `None`, the observation cloud's own max radius.

## Losses

```
loss = action_loss + aux_residual_weight * aux_residual_loss
```

- `action_loss` — MSE on the diffusion target (`epsilon` by default), optionally padding-masked.
- `aux_residual_loss` — MSE on the 3-D translation and 6-D rotation of `inv(T_cur) @ T_goal`.
  The 6-D rotation parameterisation (Zhou et al.) avoids the double cover of quaternions and the
  discontinuities of Euler angles, both of which make a regression target ill-posed.

The auxiliary head forces the latents to actually localise objects rather than shortcut off
proprioception, and doubles as a progress readout at eval time: progress corresponds to a
shrinking residual.

## Usage

```bash
lerobot-train --policy.type=pcd_diffusion \
  --dataset.repo_id=local/push_pc1024 --dataset.root=/path/to/dataset \
  --policy.perceiver_num_latents=16 --policy.perceiver_dim=256 \
  --policy.hidden_dim=512 --policy.num_layers=8 \
  --batch_size=64 --steps=100000 --ema.enable=true
```

`horizon=1, n_action_steps=1` recovers the paper's single-action behaviour.

## Tests

`tests/policies/pcd_diffusion/test_pcd_diffusion.py` covers the rotation round-trip, rigid
inverse, pose-key stripping, forward/backward, the aux head on and off, chunk shapes, re-planning
cadence, resampling in both directions, save/load determinism, and translation invariance of the
shared-frame normalisation.
