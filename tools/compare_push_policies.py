#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare push policies against each other and against the scripted expert.

Reports success rates with Wilson intervals rather than bare point estimates: at ~200 rollouts
the binomial noise is several percentage points, which is the same size as many differences worth
arguing about. Differences between policies are reported with a confidence interval too, so a gap
that is inside the noise reads as inside the noise.

The expert baseline is restricted to the *same objects* the policy was evaluated on -- comparing
a held-out policy number against the expert's all-object average would flatter or punish it for
reasons unrelated to the policy. (The all-object average is 70.2%, but it includes four objects
the expert never once solves; on the objects actually evaluated the expert scores 74.8% held-out
and 80.7% in-domain, so the all-object figure understates the demonstrator.)

**Objects the scripted expert never solves are excluded.** A policy failing where the demonstrator
also fails measures the demonstrator, not the policy. The exclusion is reported explicitly rather
than applied silently, so a change in the object set cannot pass unnoticed.

Exclusion is the degenerate case of a better idea, so both are reported: an object the expert
solves 10% of the time is nearly as unfair as one it never solves, and the **expert-normalised**
score (policy successes / expert successes over the same objects) handles the whole range
smoothly. It also matters for the generalization gap -- the in-domain object set is intrinsically
easier for the expert than the held-out set (80.7% vs 74.8%), so most of the raw in-domain
advantage is object difficulty rather than memorisation, and only the normalised gap separates
the two.

Usage:
    python tools/compare_push_policies.py --runs pc_diffusion_v1 pcd_diffusion_v1
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

ORIENTATION_PREFIXES = (
    "neg_x_down__", "pos_x_down__", "neg_y_down__",
    "pos_y_down__", "neg_z_down__", "pos_z_down__",
)


def base_mesh(obj: str) -> str:
    name = obj.split("/")[-1]
    for p in ORIENTATION_PREFIXES:
        if name.startswith(p):
            return name[len(p):]
    return name


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval. Normal approximation breaks down near 0 and 1; this does not."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def paired_diff(rows_a: list[dict], rows_b: list[dict], z: float = 1.96):
    """Paired comparison of two runs over the same rollout slots.

    Evaluation goals are bit-reproducible across runs: the same (batch, env) slot gets the same
    object and the same goal every time. That makes A/B comparisons *paired*, and pairing is free
    statistical power -- an unpaired test throws away the fact that both policies faced identical
    task instances. On the DiT-vs-UNet comparison it narrows the interval from [-2.7, +18.2] to
    [-0.8, +16.2] on exactly the same data.

    Returns ``(diff_pp, lo, hi, mcnemar_chi2, n_paired, only_a, only_b)`` or None if the two runs
    do not share slots -- which would mean the seeds or object lists differ and pairing is invalid.
    """
    ka = {(r["batch"], r["env"]): r for r in rows_a}
    kb = {(r["batch"], r["env"]): r for r in rows_b}
    shared = sorted(set(ka) & set(kb))
    if not shared:
        return None
    # Pairing is only valid if the slots really are the same task instance.
    if any(ka[s]["object"] != kb[s]["object"] for s in shared):
        return None
    only_a = sum(1 for s in shared if ka[s]["success"] and not kb[s]["success"])
    only_b = sum(1 for s in shared if kb[s]["success"] and not ka[s]["success"])
    n = len(shared)
    d = (only_a - only_b) / n
    se = math.sqrt(only_a + only_b) / n
    disc = only_a + only_b
    chi2 = ((abs(only_a - only_b) - 1) ** 2) / disc if disc else 0.0
    return 100 * d, 100 * (d - z * se), 100 * (d + z * se), chi2, n, only_a, only_b


def diff_ci(k1: int, n1: int, k2: int, n2: int, z: float = 1.96) -> tuple[float, float, float]:
    """Difference of two proportions with a normal-approximation interval."""
    if n1 == 0 or n2 == 0:
        return 0.0, 0.0, 0.0
    p1, p2 = k1 / n1, k2 / n2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return p1 - p2, (p1 - p2) - z * se, (p1 - p2) + z * se


def load_eval(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def expert_rates(src_root: Path) -> dict[str, tuple[int, int]]:
    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    path = src_root / "meta" / "attempts.jsonl"
    if not path.exists():
        return {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("object"):
            continue
        stats[r["object"]][1] += 1
        if r.get("success"):
            stats[r["object"]][0] += 1
    return {k: tuple(v) for k, v in stats.items()}


def summarise(rows: list[dict]) -> tuple[int, int]:
    return sum(bool(r["success"]) for r in rows), len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs_dir", type=Path, default=Path("/home/samsung/data/runs"))
    ap.add_argument("--runs", nargs="+", default=["pc_diffusion_v1", "pcd_diffusion_v1"])
    ap.add_argument("--src_root", type=Path,
                    default=Path("/home/samsung/3D_Bimanual_repo/scripts_isaaclab/irregular/"
                                 "dataset_collection/push_data_old"),
                    help="collection the policies were TRAINED on -- not a newer one, whose "
                         "attempt log describes different objects")
    args = ap.parse_args()

    exp = expert_rates(args.src_root)
    splits = ["heldout", "indomain"]
    data: dict[str, dict[str, list[dict]]] = {}
    for run in args.runs:
        data[run] = {s: load_eval(args.runs_dir / run / f"eval_{s}.jsonl") for s in splits}

    # --- expert-infeasible objects -------------------------------------------------------
    # Drop rollouts on objects the scripted expert never solves: the policy saw no successful
    # demonstration of them, so a failure there scores the demonstrator, not the policy.
    infeasible = {o for o, (k, n) in exp.items() if n > 0 and k == 0}
    evaluated = {r["object"] for run in args.runs for sp in splits for r in data[run][sp]}
    unknown = sorted(o for o in evaluated if o not in exp)
    hit = sorted(evaluated & infeasible)
    print("=" * 78)
    print("EXPERT-INFEASIBLE OBJECTS")
    print("=" * 78)
    print(f"objects the expert never solves, over the whole asset set : {len(infeasible)}")
    for o in sorted(infeasible):
        print(f"    0/{exp[o][1]:<3d}  {o}")
    print(f"of those, present in these evaluations                     : {len(hit)}")
    for o in hit:
        print(f"    EXCLUDED  {o}")
    if not hit:
        print("    (none -- the split already keeps them out, so no rollouts are dropped)")
    if unknown:
        print(f"WARNING: {len(unknown)} evaluated object(s) absent from the attempt log; "
              "no expert rate available, so they are kept but unnormalised:")
        for o in unknown:
            print(f"    {o}")
    if hit:
        for run in args.runs:
            for sp in splits:
                data[run][sp] = [r for r in data[run][sp] if r["object"] not in infeasible]
    print()

    print("=" * 78)
    print("SUCCESS RATE  (pos_err_xy < 0.03 m AND ori_err < 0.15 rad)")
    print("=" * 78)
    print(f"{'run':22s} {'split':10s} {'policy':>20s} {'expert (same objs)':>22s}")
    for run in args.runs:
        for s in splits:
            rows = data[run][s]
            if not rows:
                print(f"{run:22s} {s:10s} {'(no results yet)':>20s}")
                continue
            k, n = summarise(rows)
            p, lo, hi = wilson(k, n)
            objs = {r["object"] for r in rows}
            ek = sum(exp.get(o, (0, 0))[0] for o in objs)
            en = sum(exp.get(o, (0, 0))[1] for o in objs)
            ep, elo, ehi = wilson(ek, en)
            print(f"{run:22s} {s:10s} {k:4d}/{n:<4d} {100*p:5.1f}% "
                  f"[{100*lo:4.1f},{100*hi:4.1f}]   {ek:4d}/{en:<4d} {100*ep:5.1f}% [{100*elo:4.1f},{100*ehi:4.1f}]")

    # --- expert-normalised score ---------------------------------------------------------
    # Fraction of the demonstrator's own performance the policy reaches, on the same objects.
    # This is the smooth version of excluding infeasible objects, and it is the only form in
    # which the two splits are comparable: they differ in intrinsic difficulty.
    print("\n" + "=" * 78)
    print("EXPERT-NORMALISED  (policy successes / expert successes, same objects)")
    print("=" * 78)
    print(f"{'run':22s} {'held-out':>12s} {'in-domain':>12s} {'gap (raw)':>12s} {'gap (norm)':>12s}")
    for run in args.runs:
        norm, raw = {}, {}
        for sp in splits:
            rows = data[run][sp]
            if not rows:
                continue
            k, n = summarise(rows)
            objs = {r["object"] for r in rows}
            ek = sum(exp.get(o, (0, 0))[0] for o in objs)
            en = sum(exp.get(o, (0, 0))[1] for o in objs)
            raw[sp] = 100 * k / n
            norm[sp] = 100 * (k / n) / (ek / en) if ek else float("nan")
        if len(norm) == 2:
            print(f"{run:22s} {norm['heldout']:11.1f}% {norm['indomain']:11.1f}% "
                  f"{raw['indomain'] - raw['heldout']:+11.1f} {norm['indomain'] - norm['heldout']:+11.1f}")

    # --- stratified by how hard the object is for the expert -------------------------------
    print("\n" + "=" * 78)
    print("BY EXPERT DIFFICULTY  (both splits pooled)")
    print("=" * 78)
    bands = [("expert <50%", 0.0, 0.5), ("expert 50-85%", 0.5, 0.85), ("expert >=85%", 0.85, 1.01)]
    print(f"{'run':22s} " + " ".join(f"{b[0]:>18s}" for b in bands))
    for run in args.runs:
        cells = []
        for _, lo, hi in bands:
            k = n = 0
            for sp in splits:
                for r in data[run][sp]:
                    e = exp.get(r["object"])
                    if not e or not e[1]:
                        continue
                    if lo <= e[0] / e[1] < hi:
                        n += 1
                        k += bool(r["success"])
            cells.append(f"{k:3d}/{n:<3d}={100 * k / n:5.1f}%" if n else f"{'n/a':>13s}")
        print(f"{run:22s} " + " ".join(f"{c:>18s}" for c in cells))

    # --- paired policy-vs-policy ----------------------------------------------------------
    if len(args.runs) >= 2:
        a, b = args.runs[0], args.runs[1]
        print("\n" + "=" * 78)
        print(f"PAIRED DIFFERENCE  ({a} minus {b})  -- same objects, same goals, slot by slot")
        print("=" * 78)
        for s_ in splits:
            res = paired_diff(data[a][s_], data[b][s_])
            if res is None:
                print(f"  {s_:10s} not pairable (different slots or object lists)")
                continue
            d, lo, hi, chi2, n, oa, ob = res
            sig = "SIGNIFICANT" if chi2 > 3.84 else "not significant"
            print(f"  {s_:10s} {d:+6.1f} pp  95% CI [{lo:+6.1f},{hi:+6.1f}]   n={n:<4d} "
                  f"only-{a[:10]} {oa:3d}  only-{b[:10]} {ob:3d}   McNemar chi2 {chi2:5.2f} -> {sig}")

    # policy-vs-policy, per split
    if len(args.runs) >= 2:
        a, b = args.runs[0], args.runs[1]
        print("\n" + "=" * 78)
        print(f"DIFFERENCE  ({a} minus {b})")
        print("=" * 78)
        for s in splits:
            ra, rb = data[a][s], data[b][s]
            if not ra or not rb:
                print(f"  {s:10s} (incomplete)")
                continue
            ka, na = summarise(ra)
            kb, nb = summarise(rb)
            d, lo, hi = diff_ci(ka, na, kb, nb)
            verdict = "significant" if lo > 0 or hi < 0 else "NOT distinguishable at 95%"
            print(f"  {s:10s} {100*d:+6.1f} pp  95% CI [{100*lo:+6.1f},{100*hi:+6.1f}]   -> {verdict}")

    # generalization gap within each run
    print("\n" + "=" * 78)
    print("GENERALIZATION GAP  (in-domain minus held-out, same run)")
    print("=" * 78)
    for run in args.runs:
        ri, rh = data[run]["indomain"], data[run]["heldout"]
        if not ri or not rh:
            print(f"  {run:22s} (incomplete)")
            continue
        ki, ni = summarise(ri)
        kh, nh = summarise(rh)
        d, lo, hi = diff_ci(ki, ni, kh, nh)
        print(f"  {run:22s} {100*d:+6.1f} pp  95% CI [{100*lo:+6.1f},{100*hi:+6.1f}]")

    # per-object detail on the held-out split
    print("\n" + "=" * 78)
    print("HELD-OUT, PER OBJECT (worst first)   policy | expert")
    print("=" * 78)
    for run in args.runs:
        rows = data[run]["heldout"]
        if not rows:
            continue
        print(f"\n-- {run}")
        per: dict[str, list[int]] = defaultdict(list)
        for r in rows:
            per[r["object"]].append(int(r["success"]))
        for obj, v in sorted(per.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
            ek, en = exp.get(obj, (0, 0))
            estr = f"{ek:3d}/{en:<3d} {100*ek/en:5.1f}%" if en else "    n/a    "
            print(f"   {base_mesh(obj):<58s} {sum(v):3d}/{len(v):<3d} {100*sum(v)/len(v):5.1f}%  | {estr}")

    # error magnitudes: how close did failures get?
    print("\n" + "=" * 78)
    print("ERROR DISTRIBUTION on held-out (metres / radians)")
    print("=" * 78)
    for run in args.runs:
        rows = data[run]["heldout"]
        if not rows:
            continue
        pos = sorted(r["pos_err_m"] for r in rows)
        ori = sorted(r["ori_err_rad"] for r in rows)
        gd = sorted(r["goal_dist_m"] for r in rows)

        def q(v, f):
            return v[min(int(f * len(v)), len(v) - 1)]

        near = sum(1 for r in rows if r["pos_err_m"] < 0.06)
        print(f"  {run}")
        print(f"    pos_err  p25 {q(pos,.25):.3f}  median {q(pos,.5):.3f}  p75 {q(pos,.75):.3f}  (threshold 0.030)")
        print(f"    ori_err  p25 {q(ori,.25):.3f}  median {q(ori,.5):.3f}  p75 {q(ori,.75):.3f}  (threshold 0.150)")
        print(f"    goal_dist median {q(gd,.5):.3f}  -- pos_err at/above this means the object never moved")
        print(f"    within 2x the position threshold: {near}/{len(rows)} ({100*near/len(rows):.1f}%)")


if __name__ == "__main__":
    main()
