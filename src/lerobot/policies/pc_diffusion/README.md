# `pc_diffusion` — point-cloud diffusion policy

A DP3-style policy: a swappable point-cloud encoder feeding the **1-D conditional U-Net and
DDPM/DDIM machinery from the image-based [`diffusion`](../diffusion) policy, reused verbatim**.
Those components are modality-agnostic — they only ever see a flat `(B, global_cond_dim)`
conditioning vector — so the only thing that changes is how that vector is produced.

For the full data → training → evaluation workflow, see
[`docs/push_pointcloud_pipeline.md`](../../../../docs/push_pointcloud_pipeline.md).

## Dataset contract

All key names are config fields; these are the defaults. Everything is `float32`.

| key | shape | role |
| --- | --- | --- |
| `observation.point_cloud` | `(N, 3)` | scene cloud, XYZ |
| `observation.goal_point_cloud` | `(M, 3)` | target-configuration cloud |
| `observation.state` | `(S,)` | proprioception |
| `action` | `(A,)` | |
| `observation.object_poses` | `(K, 4, 4)` | **label only**, auxiliary head |
| `observation.goal_object_poses` | `(K, 4, 4)` | **label only**, auxiliary head |

Point clouds are typed `FeatureType.POINT_CLOUD` (not `STATE`), so they get their own
`NormalizationMode` instead of inheriting proprioception's.

> The default is `POINT_CLOUD: MIN_MAX`, matching DP3's "limits" normalizer, and that is what the
> trained baseline used. Be aware it is **anisotropic** — per-axis min/max over a workspace where x
> spans ~1.2 m and z ~0.5 m — so it does not preserve metric distance. If an encoder needs metric
> geometry, set `POINT_CLOUD` to `IDENTITY` and use `pc_isotropic_rescale` instead, which rescales
> inside the model and so travels with the checkpoint.

## Architecture

```
observation.point_cloud      (B,T,N,3) ─ pc_encoder   ─┐
observation.goal_point_cloud (B,T,M,3) ─ goal_encoder ─┤
observation.state            (B,T,S)   ───────────────┼─ concat ─ flatten ─→ global_cond
extra_state_keys                       ───────────────┘                        (B, D·T)
                                                                                   │
                       noisy action trajectory (B,horizon,A) ─ DiffusionConditionalUnet1d ─→ eps-hat
```

`_prepare_global_conditioning` is the only modality-aware code in the model; everything downstream
is modality-agnostic. Goal conditioning is a first-class config field
(`"none" | "points" | "vector"`) because the target task is **not learnable without it**: the
object's start pose is fixed, so at *t=0* the observation is identical across every episode and the
goal is the only thing that varies.

The cross-package U-Net import is pinned by
[`tests/policies/pc_diffusion/test_unet_contract.py`](../../../../tests/policies/pc_diffusion/test_unet_contract.py),
so an upstream signature change fails in CI rather than silently at train time.

## Swapping the encoder

```python
@register_pc_encoder("my_encoder")
class MyEncoder(PointCloudEncoder):
    def __init__(self, *, num_points: int, in_channels: int, out_dim: int, **kwargs): ...
    @property
    def feature_dim(self) -> int: ...
    def forward(self, pc: Tensor) -> Tensor:   # (B, N, C) -> (B, feature_dim)
```

Select with `--policy.pc_encoder=my_encoder`. Ships with `pointnet_maxpool` (per-point MLP →
max-pool → linear), which is the **control**, not the proposal: DP3's own claim is that this simple
encoder matches or beats hierarchical ones for visuomotor policy learning, so "an MLP is too weak"
is a hypothesis to test rather than a settled fact. The registry exists to make testing it cheap.

> Off-the-shelf pretrained PointNet++ weights are a poor fit here. ModelNet/ShapeNet pretraining
> deliberately builds translation- and scale-invariant features, but in this task **absolute
> position is the signal** — where the object is, and where the goal is, in a fixed table frame.
> Any pretrained encoder needs an explicit position pathway kept alongside it.

## Auxiliary residual-pose head

Optional. Predicts the remaining current→goal object transform from the same conditioning vector
the U-Net sees, forcing the encoder to localise the object relative to the goal rather than
shortcutting off proprioception.

```bash
--policy.num_objects=1 \
--policy.aux_residual_weight=0.1 \
--policy.aux_predict_rotation=false
```

- **Off by default** (`num_objects=0`), so checkpoints trained before it existed load unchanged.
- **Training only** — `conditional_sample` never touches it, so inference cost and the eval bridge
  are unaffected and no pose labels are needed at eval time.
- **Target is the residual**, `rigid_inverse(cur) @ goal`, not the absolute goal pose. Regressing
  the goal instead would still train and still look plausible in the loss curve, so
  [`test_aux_head.py`](../../../../tests/policies/pc_diffusion/test_aux_head.py) asserts the
  composition directly.
- `aux_predict_rotation=false` for the push dataset: the offline tracker recovers translation only,
  so the labels carry an identity rotation and predicting it would regress against a constant.
- The head and its rigid-transform helpers are **imported from `pcd_diffusion`** rather than
  duplicated, so the two policies' auxiliary task is provably the same one.

Requires a dataset built by
[`add_object_pose_features.py`](../../../../examples/port_datasets/add_object_pose_features.py);
missing labels raise rather than silently skipping the term, which would make the ablation a no-op.

### Measured behaviour

On the push dataset the auxiliary loss **saturates within ~600 of 100,000 steps**, explaining ~90%
of residual variance (a constant predictor gets MSE 0.00492; the head reaches < 0.0005). After that
the term contributes ~0.1% of the gradient at weight 0.1, so training is effectively identical to
the baseline from there on.

Read that as a result about the *task*, not a bug: whatever is needed to predict "how far and which
way to the goal" is already present in the conditioning vector almost immediately. Caveat —
`global_cond` includes proprioception, and arm pose correlates with push progress within an
episode, so some of that 90% may come from the arm rather than the cloud. The auxiliary gradient
does reach the point-cloud encoder (asserted by test), but the full 90% cannot be attributed to it.

## Config notes

| field | default | note |
| --- | --- | --- |
| `n_obs_steps` / `horizon` / `n_action_steps` | 2 / 64 / 32 | action chunking |
| `pc_encoder` | `pointnet_maxpool` | registry name |
| `goal_conditioning` | `points` | `none` / `points` / `vector` |
| `down_dims` | `(512, 1024, 2048)` | U-Net; **99.9% of the 278 M parameters** |
| `num_objects` | `0` | > 0 enables the auxiliary head |

`down_dims` is where essentially all the capacity lives — that is why this policy has 278 M
parameters against `pcd_diffusion`'s 47 M, despite the encoder being far simpler.

## Tests

```bash
pytest tests/policies/pc_diffusion/ -q     # 11 tests: U-Net contract + auxiliary head
```
