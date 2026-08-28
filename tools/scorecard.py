#!/usr/bin/env python3
"""Regenerate the push-policy scorecard from the eval files on disk.

Every number here is derived from `<run>/eval_*.jsonl`, so re-running this after a new
evaluation picks it up with no editing. Results are grouped by eval set, because a success
rate is only meaningful next to others measured on the same objects and rollout count --
the same checkpoint scores ~10 pp apart on `heldout` and `indomain`.

    python3 tools/scorecard.py                      # print
    python3 tools/scorecard.py -o docs/push_scorecard.md

Success thresholds are 3 cm / 0.15 rad, verbatim from the collector.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
from collections import defaultdict
from glob import glob
from math import comb

RUNS = "/home/samsung/data/runs"
DATASETS = "/home/samsung/data"
POS_THRESH_M = 0.03
ORI_THRESH_RAD = 0.15
DEG = 180.0 / math.pi

# The reference each eval set is compared against, when both are present.
BASELINES = {"e75": "pc_diffusion_aux_v1 (K=1)"}

# Filename suffixes that vary the *sampler or weights*, not the object set. Stripping them
# groups `eval_e75.jsonl` and `eval_e75_k1.jsonl` together, so models measured on the same
# objects and goals sit in one table and can be paired.
VARIANTS = {
    "k1": "K=1",
    "k8": "K=8",
    "ema": "EMA",
    "traingoals": "train goals",
    "ns50": "50 steps",
    "ns100": "100 steps",
    "nact8": "chunk 8",
    "nact16": "chunk 16",
}


def split_eval(name: str) -> tuple[str, str]:
    """`eval_e75_k8` -> ("e75", "K=8"). Unknown suffixes stay part of the set name."""
    parts = name.removeprefix("eval_").split("_")
    if len(parts) > 1 and parts[-1] in VARIANTS:
        return "_".join(parts[:-1]), VARIANTS[parts[-1]]
    return "_".join(parts), "K=1"


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson interval. Beats the normal approximation at the sample sizes here."""
    if not n:
        return (0.0, 0.0)
    p, z = k / n, 1.96
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (centre - half), 100 * (centre + half))


def mcnemar(n01: int, n10: int) -> float:
    """Exact two-sided McNemar. n01/n10 are the discordant pair counts."""
    n = n01 + n10
    if not n:
        return 1.0
    tail = sum(comb(n, i) for i in range(max(n01, n10), n + 1))
    return min(1.0, 2 * tail / 2**n)


def load(path: str) -> dict[tuple, dict]:
    """Index rollouts by slot. goal_dist_m fingerprints the goal, so pairing is checkable."""
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[(r["batch"], r["env"], r["object"])] = r
    return out


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    succ = sum(bool(r["success"]) for r in rows)
    pos = [1000 * r["pos_err_m"] for r in rows]
    ori = [DEG * r["ori_err_rad"] for r in rows]
    lo, hi = wilson(succ, n)
    return {
        "n": n,
        "success": 100 * succ / n,
        "ci": (lo, hi),
        "pos_mean": st.mean(pos),
        "pos_median": st.median(pos),
        "ori_mean": st.mean(ori),
        "ori_median": st.median(ori),
        # Ceilings: what the rate would be if the other axis were always perfect.
        "pos_gate": 100 * sum(r["pos_err_m"] <= POS_THRESH_M for r in rows) / n,
        "ori_gate": 100 * sum(r["ori_err_rad"] <= ORI_THRESH_RAD for r in rows) / n,
    }


def recipe(run: str) -> dict:
    """Fingerprint the run from its newest checkpoint config."""
    cks = sorted(glob(f"{RUNS}/{run}/checkpoints/*/pretrained_model/config.json"))
    if not cks:
        return {}
    cfg = json.load(open(cks[-1]))
    train_path = cks[-1].replace("config.json", "train_config.json")
    data, episodes = "?", "?"
    if os.path.exists(train_path):
        tc = json.load(open(train_path))
        ds = tc.get("dataset") or {}
        data = (ds.get("repo_id") or "?").split("/")[-1]
        eps = ds.get("episodes")
        episodes = len(eps) if isinstance(eps, list) else "all"
    return {
        "backbone": cfg.get("backbone", "unet"),
        "goal_cond": cfg.get("goal_conditioning", "?"),
        "aux": f"n={cfg.get('num_objects', 0)},rot={str(cfg.get('aux_predict_rotation', False))[0]}",
        "onehot": str(cfg.get("use_task_onehot", False))[0],
        "iso": str(cfg.get("pc_isotropic_rescale", False))[0],
        "data": data,
        "episodes": episodes,
    }


def teacher(eval_name: str) -> str:
    """Demonstrator error on the objects an eval set uses, for scale.

    The dataset stores only successful demonstrations, so this is the teacher's error
    *conditional on success* -- its true distribution is wider. Not a success rate.
    """
    lst = f"{DATASETS}/push_v3/splits/objects_e75_screen_paths.txt"
    src = f"{DATASETS}/push_v3/meta/episode_objects.jsonl"
    if "e75" not in eval_name or not (os.path.exists(lst) and os.path.exists(src)):
        return ""
    want = {os.path.splitext(os.path.basename(x.strip()))[0] for x in open(lst) if x.strip()}
    pos, ori = [], []
    for line in open(src):
        if not line.strip():
            continue
        r = json.loads(line)
        if os.path.splitext(os.path.basename(r["object"]))[0] in want:
            pos.append(1000 * r["pos_err_m"])
            ori.append(DEG * r["ori_err_rad"])
    if not pos:
        return ""
    return (
        f"\nDemonstrator on these objects (successful demos only, n={len(pos)}): "
        f"**{st.mean(pos):.1f} mm** / **{st.mean(ori):.1f}°** mean, "
        f"{st.median(pos):.1f} mm / {st.median(ori):.1f}° median.\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", help="write markdown here instead of stdout")
    args = ap.parse_args()

    by_eval: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted(glob(f"{RUNS}/*/eval_*.jsonl")):
        run = path.split("/")[-2]
        eval_set, variant = split_eval(os.path.splitext(os.path.basename(path))[0])
        rows = [json.loads(x) for x in open(path) if x.strip()]
        if rows:
            label = f"{run} ({variant})"
            by_eval[eval_set][label] = {"stats": summarise(rows), "slots": path, "run": run}

    out: list[str] = [
        "# Push policy scorecard",
        "",
        "Generated by `tools/scorecard.py` from the eval files on disk; re-run it after any",
        "new evaluation. Success requires **both** gates: position ≤ 3 cm and orientation",
        "≤ 0.15 rad (8.6°).",
        "",
        "`pos`/`ori` are mean errors in mm and degrees. `pos gate`/`ori gate` are the ceilings",
        "each axis would allow if the other were perfect — the lower one is what binds.",
        "",
    ]

    for name in sorted(by_eval, key=lambda s: (("e75" not in s), s)):
        entries = by_eval[name]
        out += [f"## `{name}`  ({len(entries)} result{'s' if len(entries) > 1 else ''})", ""]
        out += [
            "| model | backbone | goal | aux | data | eps | n | success | 95% CI | pos | ori | pos gate | ori gate |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for run in sorted(entries, key=lambda r: -entries[r]["stats"]["success"]):
            s, rc = entries[run]["stats"], recipe(entries[run]["run"])
            out.append(
                f"| `{run}` | {rc.get('backbone', '?')} | {rc.get('goal_cond', '?')} "
                f"| {rc.get('aux', '?')} | {rc.get('data', '?')} | {rc.get('episodes', '?')} "
                f"| {s['n']} | **{s['success']:.1f}%** | {s['ci'][0]:.1f}–{s['ci'][1]:.1f} "
                f"| {s['pos_mean']:.1f} | {s['ori_mean']:.1f} "
                f"| {s['pos_gate']:.1f}% | {s['ori_gate']:.1f}% |"
            )
        out.append("")

        base = BASELINES.get(name)
        others = [r for r in entries if r != base]
        if base in entries and others:
            out += [
                f"Paired against `{base}`. Slots are keyed by (batch, env, object) and goals are",
                "verified identical via `goal_dist_m`, so McNemar applies only where marked paired.",
                "",
                "| vs baseline | shared | paired? | difference | 95% CI | McNemar p |",
                "|---|---|---|---|---|---|",
            ]
            A = load(entries[base]["slots"])
            for run in others:
                B = load(entries[run]["slots"])
                ks = sorted(set(A) & set(B))
                if not ks:
                    continue
                bad = sum(abs(A[k]["goal_dist_m"] - B[k]["goal_dist_m"]) > 1e-6 for k in ks)
                n01 = sum(1 for k in ks if B[k]["success"] and not A[k]["success"])
                n10 = sum(1 for k in ks if A[k]["success"] and not B[k]["success"])
                diff = 100 * (n01 - n10) / len(ks)
                se = math.sqrt(max(n01 + n10 - (n01 - n10) ** 2 / len(ks), 0)) / len(ks)
                p = mcnemar(n01, n10)
                mark = "yes" if not bad else f"**NO** ({bad} differ)"
                pcell = f"{p:.2g}" if not bad else "n/a"
                out.append(
                    f"| `{run}` | {len(ks)} | {mark} | {diff:+.1f} pp "
                    f"| {100 * (diff / 100 - 1.96 * se):+.1f}–{100 * (diff / 100 + 1.96 * se):+.1f} | {pcell} |"
                )
            out.append("")
        out.append(teacher(name))

    out += [
        "## Reading these numbers",
        "",
        "- **Compare only within an eval-set section.** The same checkpoint scores several points",
        "  apart across `heldout`, `indomain` and `e75`.",
        "- **Cross-run differences under ~15 pp are not trustworthy.** `pc_diffusion_aux_v1` and",
        "  `push_unet_oldata` share config, seed, episode list and architecture, yet differ by",
        "  14.3 pp paired (p=0.003) — they were trained two days apart from untracked code.",
        "  Treat that as the noise floor for any comparison between two separate training runs.",
        "- **Same-run comparisons are exempt.** EMA vs non-EMA weights from one run, or K=1 vs K=8",
        "  on one checkpoint, share the training trajectory and resolve to ~±7.5 pp at n=336.",
        "- `pc_diffusion_aux_v1` cannot be retrained; the code that produced it is gone. It remains",
        "  valid to evaluate, deploy and fine-tune from, but it cannot serve as an ablation baseline.",
        "",
    ]

    text = "\n".join(out)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as fh:
            fh.write(text)
        print(f"wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
