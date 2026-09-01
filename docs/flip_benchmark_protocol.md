# Flip benchmark protocol

Frozen 2026-08-31. The reference for comparing methods on the bimanual **flip** primitive.
Anything reported as a flip success rate should state that it follows this protocol, or say
exactly how it deviates. The push equivalent is
[`push_benchmark_protocol.md`](./push_benchmark_protocol.md); this document exists because
`primitives_training_playbook.md` section 3 requires a per-primitive variant, and because three
parts of the push protocol do not transfer.

Harness: `scripts_isaaclab/irregular/eval/eval_flip_policy.py` (3D_Bimanual_repo).

---

## 1. Object set

**121 (asset, flip_direction) pairs**, listed in
`/home/samsung/data/flip_v1/splits/flip_e75_pairs.txt`.

That file is the specification. Do not regenerate it. The push campaign regenerated its set from
success rates and silently reshuffled 10 of 21 objects between runs.

**The unit is the pair, not the asset.** The same object flips differently in different
directions -- `Lenovo_Yoga_2_11` is flipped 95% of the time in one direction and 0% in another --
so an asset-level list would average over task instances of wildly different difficulty.

**Provenance.** All 146 pairs in `flip_list.json` were attempted 20 times each by the scripted
demonstrator during the 2026-08-30/31 collection (2920 attempts, `meta/attempts.jsonl`). A pair
is included iff the demonstrator succeeded on **>= 75%** of its attempts: 121 pairs over 45 of
the 52 assets. The 25 exclusions are listed in the file itself with their rates.

The screen is on **demonstrator competence, measured before any policy existed** -- not on policy
scores. This is the property push's set lacks: there, five objects were dropped by their scores
after the fact, inflating every subsequent number by roughly +9.5 pp. Numbers under this protocol
carry no such inflation, and are therefore **not comparable to push numbers** in either
direction.

The excluded pairs are dominated by thin, near-planar boxes (`Lenovo_Yoga_2_11`,
`Markings_Letter_Holder`, `Dell_Ink_Cartridge`, `BlueBlack_Nintendo_3DSXL`, the Crayola chalk
boxes) -- the same geometry class push identified. A rule stated over geometry rather than over
scores would plausibly select approximately this set, but that rule has not been established.

## 2. Goals

A flip is **not** commanded as a place to put the object. The command is *"tip it
`--flip_angle_deg` (default 90) about the horizontal axis v(d) of its own flip frame"*, where d
is one of `+x/-x/+y/-y` read in the OBJECT frame -- so the axis turns with the object's spawn
yaw and "+x" tips the same physical face over however the object is turned.

The goal pose is that rotation applied **about the object's pivot edge**: the +u extreme of its
resting footprint (points within 5 mm of its lowest point), at table height.

| | |
|---|---|
| rotation | `--flip_angle_deg`, default 90 deg, about v(d) |
| axis | horizontal, at yaw `W(d) + spawn_yaw`; derived from the observed t=0 cloud |
| translation | implied by rotating about the pivot edge -- not commanded independently |
| task variation | spawn yaw, drawn U[-180, 180] deg per episode, as in collection |

**Why pivot geometry rather than the achieved final pose.** The goal must be *known at
deployment by construction* (`primitive_dataset_pipeline.md` 2.6). Push can use the achieved
position because its demonstrator was driven to a commanded goal and lands a median 8.6 mm from
it, so achieved ~= commanded. The flip collection never commanded a position -- the object ends
wherever physics puts it -- and substituting the achieved pose is off by a **median 36 mm, p90
86 mm** from the pivot prediction (n=246), larger than push's entire 30 mm position gate. Since
nothing can know the achieved pose before the object is pushed, training on it would also make
the goal uncomputable on hardware.

**Train/eval identity is enforced, not assumed.** `eval_flip_policy.flip_command_pose` and
`port_isaaclab_pointcloud_push.flip_command_pose` must produce the same pose. They were verified
**bitwise identical (0 ULP) over 3000 randomised cases**. Re-run that check after touching
either. An earlier version differed by up to 0.04 deg purely because the evaluator computed the
spawn yaw in float32.

**Reproducibility.** `--seed` (default **1234**) drives two things: `env_cfg.seed`, and the
`numpy` generator that draws the spawn yaws and resamples the clouds. Goals therefore depend
only on (seed, pair list) -- never on the policy or the checkpoint -- so two runs over the same
list draw identical yaws and are paired slot-for-slot. Every rollout records `spawn_yaw_deg`, so
comparing that field **verifies** pairing rather than assuming it.

**Goals are deterministic; OUTCOMES ARE NOT.** The seed fixes the spawn yaws, and two runs
draw identical ones -- verified: two runs at `num_envs=3, seed=1234` produced spawn yaws equal to
15 decimal places. But the policy is a **diffusion** model and samples its initial noise at
inference (`modeling_pc_diffusion.py`, `torch.randn(..., generator=generator)` with
`generator=None`), and the policy server exposes no seed. The same checkpoint on the same goal
therefore gives different rollouts: in those two runs `Dell_Ink_Cartridge -x` scored 91.8 deg in
one and 0.0 deg in the other -- a clean flip and a total failure, same spawn yaw.

Consequences:

* A paired comparison between two checkpoints is paired **on goals only**. Policy sampling noise
  is not removed by pairing and does not cancel; it has to be averaged down with rollouts.
* Per-rollout results are not reproducible. Do not treat a single rollout, or a handful, as a
  property of the policy -- the 3-env clips in `flip_video_tests/` are illustrations, not
  measurements.
* Aggregate rates over ~320 rollouts are stable enough to compare (the per-batch spread was
  80/68/78/78/78/.../78%), but a difference of a few points between two runs of the *same*
  checkpoint is expected, not evidence of a change.
* If reproducible rollouts are ever needed, the policy server would have to accept a seed and
  thread a `torch.Generator` into `conditional_sample`. It does not today.

**Sampling is not perfectly uniform per pair.** Envs are filled by
`slot = (b * num_envs + i) % len(pairs)`, so when `num_envs` does not divide 121 the indexing
wraps and a few pairs receive one extra rollout. The 2026-08-31 run covered all 121 pairs with
2 or 3 rollouts each (320 total, not 242). It is deterministic and identical across runs, so
pairing still holds; it just means per-pair counts differ by one.

Per-rollout records carry: `success`, `on_axis_deg`, `on_axis_err_deg`, `on_axis_hold_deg`,
`settled`, `face_start` / `face_hold` / `face_final`, `face_residual_deg`, `pos_err_m`,
`spawn_yaw_deg`, `flip_direction`, `object`, and a per-step `on_axis_trace_deg`, so
success-at-any-horizon is computable after the fact without re-running.

## 3. Success gates

Push's 50 mm / 0.20 rad do not transfer. A flip either landed on the intended face or it did
not. Both gates below are the collector's own, so the policy is scored by the same rule that
selected its training demonstrations:

1. **On-axis rotation.** `|rotation about v(d) - flip_angle_deg| < --max_angle_error_deg`
   (default 10 deg). Rotation **about the flip axis**, not the geodesic magnitude: an object
   that rolled 90 deg sideways or yawed 90 deg on the spot is axis-blind-identical to a clean
   flip and must not score as one. Verified: the metric returns 90.0000 deg on an exact flip and
   0.0000 deg on a 90 deg roll about the wrong axis.
2. **Settled.** The object rests on the same face after the arms retreat as it did while they
   were still pressing. This rejects a flip that topples back the moment it is released --
   the collector logs that case as `TOPPLED-ON-RETREAT`.

Verbatim, from `eval_flip_policy.py`:

```python
err     = np.abs(on_axis_final - args.flip_angle_deg)   # commanded 90 deg
settled = face_final == face_hold                       # face after retreat == face before
succ    = (err < args.max_angle_error_deg) & settled    # tolerance 10 deg
```

| parameter | default | meaning |
|---|---|---|
| `--flip_angle_deg` | 90.0 | commanded rotation about v(d) |
| `--max_angle_error_deg` | 10.0 | on-axis tolerance; the collector's own gate |
| `--retreat_steps` | 150 | arms driven home before the settled check |
| `--steps` | 1750 | control steps per rollout (= collection's 350 captures x decim 5) |
| `--spawn_yaw_range_deg` | -180 180 | task variation, drawn per episode |

**Position does not gate.** It is recorded as `pos_err_m` and should be reported as a
diagnostic, but the collector's definition of a successful flip has no position term, and
gating on one would score the policy against a criterion its demonstrations were never selected
under.

**A known property of the data.** Because gate 1 is one-dimensional, the demonstrations
themselves contain substantial off-axis wobble: measured against the commanded goal, the full
geodesic error is median 6.65 deg, p90 21.3 deg, with 110 of 2453 episodes (4.5%) beyond 30 deg.
The policy is trained on those. If a future run adds a geodesic gate, it is measuring something
the demonstrator was never asked to do.

## 4. Running it

```bash
# 1. policy server (lerobot env), pointed at the dataset the checkpoint TRAINED on --
#    it carries the normalisation statistics
python tools/pc_policy_server.py --checkpoint <run>/checkpoints/100000/pretrained_model_ema \
    --dataset_root /home/samsung/data/flip_objonly --repo_id local/flip_objonly

# 2. harness (isaaclab env)
python scripts_isaaclab/irregular/eval/eval_flip_policy.py \
    --object_list /home/samsung/data/flip_v1/splits/flip_e75_pairs.txt \
    --episodes_per_pair 2 --num_envs 40 --seed 1234 \
    --record_video 6 --env_spacing 25 --output <run>/eval_flip_e75.jsonl
```

The harness verifies the checkpoint contract at startup (input keys, point counts, crop box,
`action_space`) and aborts on mismatch rather than producing a quietly wrong number.

Every rollout records a per-step `on_axis_trace_deg`, so success-at-any-horizon and
ever-on-goal are computable after the fact without re-running.

## 5. Recording

`--record_video N` writes mp4s for the first N envs, with the object's own mesh drawn
translucent at the commanded goal pose plus a HUD laid out identically to push's
(`eval/goal_overlay.py` holds the shared renderer, so the two cannot drift). Rows: status |
object | face state | ORI DEG bar | STEP bar -- push's POS MM row is replaced by the face state,
because that is flip's second gate. Push's 3 cm position-tolerance ring is suppressed: flip has
no position gate and drawing it would advertise a criterion the policy is not scored against.

`--env_spacing 25` is needed for recording. At the scene default of 2.5 m the neighbouring rigs
sit inside the camera frustum and appear in every clip; their ground tiles are also what produces
the colour banding on the floor. Envs are physically independent, so spacing changes nothing but
the view.

`--ghost_selftest` renders one still per env and exits, without connecting to the policy server
-- a ~1 minute check of framing and overlay that is safe to run while a benchmark holds the
socket.

## 6. First result -- 2026-08-31

`flip_dit_xattn`, DiT + cross-attention + goal vector, absolute joint actions, EMA weights at
100k steps, evaluated on the frozen 121-pair list with seed 1234.

| | |
|---|---|
| **success** | **245/320 (76.6%)** |
| on-axis gate alone | 246/320 (76.9%) |
| settled gate alone | 271/320 (84.7%) |
| flipped but toppled on release | 1 |
| settled but never reached the angle | 26 |

Per direction: `+x` 85% (94/111), `-x` 83% (78/94), `+y` 67% (39/58), `-y` 60% (34/57).

Per-batch rates were 80/68/78/78/78/…/78%, a tight spread with no batch far off.

**Reading these numbers.**

* The binding gate is the **angle, not the settle**: of the 75 failures, 26 held a face but never
  reached 90 deg, and exactly 1 flipped and then toppled. The policy's difficulty is completing
  the rotation, not keeping the object there.
* **The +y / -y deficit is real and unexplained.** It tracks training volume (`+x` 848 episodes
  vs `+y` 451), but it also tracks arm assignment: `_FLIP_TOP_ARM_IS_LEFT` breaks the +-y tie for
  role balance rather than by geometry, so the side contact can be assigned to the arm further
  from it. Volume and kinematics are confounded here; distinguishing them needs a targeted run,
  not a re-reading of this table.
* **Not comparable to push's 83.3%.** Push's object set had its five worst objects removed by
  their scores after the fact, worth roughly +9.5 pp; this list was screened on demonstrator
  competence before any policy existed. Neither number should be quoted against the other.
* `pos_err_m` is a diagnostic and does not gate: median 0.073 m, p90 0.609 m. **44 of 320
  rollouts exceed 0.5 m**, i.e. the object left the table; the max is 33.9 m. Those are counted
  as failures by the angle gate anyway, but the magnitude matters for hardware -- a policy that
  occasionally throws the object is benign in simulation and is not benign on a real robot.
  Worth understanding before deployment.

**Provenance of the harness.** This is the first end-to-end run of `eval_flip_policy.py`. The
goal construction is verified bitwise against the porter (0 ULP, 3000 cases) and the on-axis
metric is unit-checked (90.0000 deg on an exact flip, 0.0000 deg on a 90 deg roll about the wrong
axis), but the harness as a whole has produced exactly one result. Per
`push_scorecard.md`, a single run can land far off with no config difference: re-run a surprising
arm before acting on it.
