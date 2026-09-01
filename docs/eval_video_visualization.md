# Evaluation video visualisation

How evaluation rollouts are rendered: framing, the goal overlay, and the HUD. Written to be
shared across primitives -- push and rotate should pull from this rather than each growing their
own conventions, because the whole point of the overlay is that two clips from different
primitives can be read the same way.

Reference implementation: `scripts_isaaclab/irregular/eval/eval_flip_policy.py` (3D_Bimanual_repo).
Shared drawing primitives: `scripts_isaaclab/irregular/eval/goal_overlay.py`.

---

## 0. Where the code lives

`goal_overlay.py` holds everything that is primitive-agnostic: the ghost rasteriser, shading,
stroke/blend layers, the pixel font and the HUD primitives (`_hud_text`, `_hud_bar`). It was
lifted verbatim out of `eval_push_policy.py` so a harness can draw overlays **without importing
that module** -- importing it boots IsaacLab and runs its module-level argument parser.

What a goal MEANS stays in the harness. Push draws a position tolerance ring; flip draws a
commanded rotation and a face state. Keep primitive semantics out of `goal_overlay.py`.

`_build_goal_ghosts` needs only `.target_pos` / `.target_quat` off its `cmd` argument, so a
harness with no command term can pass a plain `types.SimpleNamespace` carrying those two tensors.
**They must be WORLD coordinates** -- if the harness works in env-local poses (as flip does), add
the env origin before handing them over.

## 1. Scene framing

### 1.1 Env spacing

```
--env_spacing 25          # scene default is 2.5 (PushSceneCfg)
```

At the 2.5 m default the neighbouring rigs sit inside the camera frustum and appear in every
clip. Envs are physically independent, so spacing changes the view and nothing else -- no
physics, no scoring.

Note the side effect: `/World/plane` is only 10 x 10 m, so at large spacing the outer envs sit
beyond it and have no floor beneath them. That is harmless for evaluation (an object that leaves
the table has already failed) but it means "object falls forever" instead of "object lands".

### 1.2 The grey ground

**Symptom:** a grey slab and its shadow fill the lower third of the frame, and widening
`--env_spacing` does not remove it.

**Cause:** `/World/plane` (`src/isaaclab_push/env_cfg.py`) is a *single global* prim, not one per
env -- a 10 x 10 x 0.1 m cuboid at z = -1.1 with `diffuse_color=(0.3, 0.3, 0.35)`. Spacing moves
the robots apart; it never moves the floor.

**Fix:**

```python
env_cfg.scene.plane.spawn.visible = False      # --hide_ground, default on when recording
```

`SpawnerCfg.visible` suppresses rendering only. **Keep the collision.** That slab is what catches
objects knocked off the table -- 44 of 320 rollouts in the first flip benchmark did exactly that
-- so deleting the asset would change the dynamics and the footage would no longer match the
scored run.

Measured effect of hiding it (identical scene, one env): table and object brightness unchanged to
**0.1 of a grey level**; only the background regions the slab used to occupy get lighter. It does
not relight the scene, and the bounce-light loss one might expect does not materialise.

### 1.3 Camera

`record_frustum_push._make_tiled_camera_cfg(hz)` places a per-env camera looking at
`(0, 0, hz)`, where `hz` is the tallest object's half-height in the batch. Presets:

| view | eye |
|---|---|
| `front` (default) | (0.0, -1.75, 1.2) |
| `side` | (2.0, 0.0, 1.1) |
| `iso` | (1.3, -1.5, 1.2) |

```
--camera_view front|side|iso
--camera_dist 1.0            # scales the eye about the look-at: <1 moves in, >1 pulls back
```

`hz` is batch-dependent, so the same object framed in a 3-env run and a 40-env run can differ
slightly. That is framing only, not a scoring difference -- but do not read two clips from
different batch sizes as pixel-comparable.

## 2. The goal ghost

The object's own mesh, posed at the commanded goal, rasterised once per batch (the goal and the
camera are both static within an episode) and reused for every frame. Translucent, tinted by the
success predicate: **blue pending, green on goal**. The switch is deliberately binary -- a
continuous colour ramp would imply the metric resolves differences it does not; the HUD numerals
carry the magnitude.

### 2.1 The tint trap -- read this before reusing `_draw_goal_ghost`

`goal_overlay._draw_goal_ghost` derives its tint from push's gates:

```python
pos_ok = ori_ok = False
if live is not None:
    pos_ok = live["pos_err_m"]  < SUCCESS_POS_M      # 3 cm
    ori_ok = live["ori_err_rad"] < SUCCESS_ORI_RAD   # 0.15 rad
key = "ongoal" if (pos_ok and ori_ok) else "pending"
```

A primitive without those two terms has nothing sensible to put in `live`, and passing
`live=None` -- the obvious thing to do -- pins `pos_ok = ori_ok = False`, so **the ghost renders
"pending" for the entire episode no matter how well the rollout goes**. It fails silently: the
HUD still turns green from the harness's own predicate, the ghost does not, and the clip looks
deliberately styled rather than broken. It survived several rounds of review here before anyone
noticed the contradiction between a green `ON GOAL` banner and a blue mask.

Flip therefore draws the ghost itself and supplies its own predicate
(`eval_flip_policy._draw_flip_ghost`):

```python
layers = ghost.get("layers")
ovl._blend_layer(frame, layers, layers["rgb"]["ongoal" if ok else "pending"])
```

**Any primitive whose gates are not push's must do the same.** Rotate currently carries its own
copy of `_draw_goal_ghost` with the same structure and should be checked for this specifically.

Two rules that fall out of it:

* **Tint and HUD status must read the same variable.** If they can disagree, one of them is
  lying, and the ghost is the one nobody checks.
* **The predicate should follow what is decidable at that moment.** Flip tints on the on-axis
  gate during the rollout, then on the *full* criterion (on-axis AND face held) during the
  retreat -- so an object that topples on release turns the mask back to blue while the HUD reads
  `TOPPLED`. Watching a success being lost is the point of recording the retreat at all.

Verified on a real rollout: the mask switches to green on the same frame the error crosses the
threshold (frame 119 of 238, `ORI DEG 32.9 -> 8.7` against a 10.0 gate).

A ghost carries the full goal *pose*, which is why it is worth the rasterisation: no flat marker
can show a commanded orientation, and for rotate and flip the orientation IS the task.

`coverage px per env` is printed per batch. A zero means the ghost is off-screen or behind the
camera -- treat it as a failed render, not an empty goal.

**Per-primitive:** push additionally draws a tolerance ring and centre dot in the table plane
(`_goal_marks`) for its 3 cm position gate. Flip suppresses them (`ghost["marks"] = ()`) because
it has no position gate and drawing them would advertise a criterion the policy is not scored
against. Rotate should decide the same way: draw the marks only if position actually gates.

## 3. The HUD

One layout, shared. Constants live in `goal_overlay.py`, so changing one moves every primitive's
HUD together.

```
position   top-left (5, 5)              _HUD_PAD
size       172 x 58 px                  _HUD_W, _HUD_H
scrim      rgb(12,17,23) @ 66%          _HUD_BG, _HUD_SCRIM
line pitch 10 px                        _HUD_LINE
bar        x=51, 56 x 5 px              _HUD_BAR_X/W/H
tick       at 60% of bar width          _HUD_TICK_FRAC
value col  x=112                        _HUD_VAL_X
labels     rgb(150,166,182) dim         _HUD_DIM
values     green on-goal / blue pending _GHOST_RGB_ONGOAL / _PENDING
```

Five rows. Rows 1-2 and row 5 are fixed across primitives; rows 3-4 carry the primitive's own
success terms:

| row | push | flip | rotate (suggested) |
|---|---|---|---|
| 1 | `ON GOAL` / `PENDING` | same | same |
| 2 | object name | same | same |
| 3 | `POS MM` bar | `FACE` state | `POS MM` bar, if position gates |
| 4 | `ORI DEG` bar | `ORI DEG` bar (on-axis) | `ORI DEG` bar (about the commanded axis) |
| 5 | `STEP` bar | `STEP <dir>` bar | `STEP` bar |

Bars are `value / threshold` with the threshold marked at 60% of the width, so over-threshold is
visible rather than clipped. The numeral is always printed beside the bar because the fill stops
growing at 2x threshold.

**Bar semantics must match the scored metric.** Flip's `ORI DEG` is on-axis error against the
10 deg gate -- the same quantity in the results JSONL -- not a geodesic angle. If the HUD and the
scorecard disagree, the video is lying.

**Direction labelling.** From one fixed camera two flip directions can look nearly identical, and
the ghost alone cannot disambiguate them. Flip therefore prints the commanded direction on the
STEP row. Rotate has the same problem and should do the same.

**Context-dependent rows are allowed and useful.** Flip's `FACE` reads `SAME` / `CHANGED` against
the start face during the rollout, then switches during the retreat to compare against the
pre-retreat face and reads `STABLE` / `TOPPLED`. That makes the second success gate visible: you
can watch an object flip and then topple as the arms release, and see the verdict change with it.

## 4. Recommended flags

```bash
--record_video 6 --video_every 8 --env_spacing 25 --hide_ground
```

`--video_every` trades smoothness against size; 8 gives ~220 frames for a 1750-step rollout at
15 fps. Filenames encode the outcome so a directory listing is already a report:

```
b0_env3_<object>_<dir>_OK_90deg.mp4
b0_env0_<object>_<dir>_noflip_0deg.mp4
b0_env1_<object>_<dir>_nosettle_88deg.mp4
```

**Recording continues through the retreat.** A clip that stops at the last policy step cannot
show why an episode was scored a failure, because the settle gate is decided during the retreat.

## 5. Verifying a render without a rollout

```bash
--ghost_selftest        # one still per env, then exit
```

Renders the overlay immediately after the goal is built and exits -- about a minute versus ten
for a full batch. It does **not** connect to the policy server, so it is safe to run while a
benchmark holds the socket.

To prove the overlay is not touching something it should not (a recurring question about the
ground), render the same scene with and without `--no_ghost` and diff:

```python
d = np.abs(a.astype(int) - b.astype(int)).sum(-1)
(d > 32).sum()          # threshold matters
```

Use a threshold. Two separate process runs differ by +-1-2 RGB units everywhere from renderer
non-determinism, so a threshold of 0 reports ~35% of pixels changed and proves nothing. At >32
the flip overlay's footprint was 7080 px confined to the ghost silhouette, with zero ground and
zero HUD pixels.

## 6. Adopting this in push / rotate

1. Import `goal_overlay` instead of keeping a private copy of the renderer. Push currently owns
   the original; moving it to the shared module and importing it back is the change that stops
   the two drifting.
2. Add `--env_spacing` and `--hide_ground`; both are scene-config one-liners and neither affects
   physics or scoring.
3. Keep rows 1, 2 and 5 of the HUD identical. Put the primitive's own gates in rows 3-4, and make
   each bar report the same quantity the scorecard does.
4. If the primitive has no position gate, suppress `marks`.
5. **Check the ghost tint (section 2.1).** If the harness calls `_draw_goal_ghost(..., live=None)`
   its ghost is permanently blue. Rotate has its own copy of that function and is very likely
   affected.
6. Verify with `--ghost_selftest` before committing to a long recording run.

## 7. Known duplication

As of 2026-08-31 the renderer exists in **three** places: `eval_push_policy.py` (the original),
`eval_rotate_policy.py` (its own copy, lines ~522-800), and `goal_overlay.py` (the extraction).
Flip is the only harness importing the shared module.

This was deliberate at the time -- push had already produced the scorecard numbers and rewiring
it mid-campaign risked perturbing a working harness -- but it is debt, and the tint trap in
section 2.1 is exactly the kind of bug that now has to be fixed three times. Consolidating push
and rotate onto `goal_overlay.py` is the change that stops them drifting.
