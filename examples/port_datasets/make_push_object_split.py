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

"""Build a held-out **object** split for the push dataset.

Three things make this less trivial than a random episode split:

1. **Orientation variants leak.** The object set contains the same base mesh in several
   orientations (``Foo``, ``pos_x_down__Foo``, ...). Splitting on the entry name would put the
   same geometry in both train and test and inflate the held-out number. We group on the
   prefix-stripped base mesh so all variants land on the same side.
2. **Groups are very uneven** (round / non-convex / rotated / frustum), so the draw is stratified
   by group -- otherwise a random 20% can miss a whole category.
3. **Some objects are infeasible for the scripted expert** (0% success), so they contribute no
   training data and a policy failing on them measures the demonstrator, not the policy. Those
   are kept out of the held-out set and reported separately.

Writes into ``<dataset>/splits/``::

    objects_train.txt / objects_test.txt     base mesh names
    episodes_train.json / episodes_test.json episode indices for --dataset.episodes
    split_report.md                          what went where, and why

Usage::

    python examples/port_datasets/make_push_object_split.py \
        --dataset_root /home/samsung/data/push_pc1024 \
        --src_root /home/samsung/.../push_data \
        --test_frac 0.2
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ORIENTATION_PREFIXES = (
    "neg_x_down__",
    "pos_x_down__",
    "neg_y_down__",
    "pos_y_down__",
    "neg_z_down__",
    "pos_z_down__",
)


def base_mesh_name(object_path: str) -> str:
    name = object_path.split("/")[-1]
    for prefix in ORIENTATION_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def expert_success_by_path(src_root: Path) -> dict[str, tuple[int, int]]:
    """Per *asset path* ``(successes, attempts)``.

    Deliberately finer-grained than the base-mesh grouping used for the split. The simulator
    spawns a specific orientation variant, and difficulty depends strongly on it -- a frustum on
    its wide face is solved every time, the same mesh on its small face almost never. Evaluation
    curation therefore has to reason per path, even though leakage prevention reasons per mesh.
    """
    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    path = src_root / "meta" / "attempts.jsonl"
    if not path.exists():
        return {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            obj = row.get("object")
            if not obj:
                continue
            stats[obj][1] += 1
            if row.get("success"):
                stats[obj][0] += 1
    return {k: (v[0], v[1]) for k, v in stats.items()}


def expert_success_by_base_mesh(src_root: Path) -> dict[str, tuple[int, int]]:
    """Per base mesh ``(successes, attempts)`` from the collector's attempt log.

    Uses ``attempts.jsonl`` rather than ``episode_objects.jsonl`` because the former records
    failed rollouts too, which is what makes a real success *rate* available.
    """
    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    path = src_root / "meta" / "attempts.jsonl"
    if not path.exists():
        return {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            obj = row.get("object")
            if not obj:
                continue
            key = base_mesh_name(obj)
            stats[key][1] += 1
            if row.get("success"):
                stats[key][0] += 1
    return {k: (v[0], v[1]) for k, v in stats.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_root", type=Path, required=True)
    p.add_argument("--src_root", type=Path, default=None, help="v2.1 source, for expert success rates")
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_min_expert", type=float, default=0.75,
                   help="curated evaluation sets keep only objects the scripted expert solves at "
                        "least this often; 0 disables curation")
    p.add_argument("--indomain_eval_objects", type=int, default=0,
                   help="how many training objects to sample for the in-domain evaluation set; "
                        "0 matches the curated held-out count, which keeps the two comparable")
    args = p.parse_args()

    root: Path = args.dataset_root
    rows = [json.loads(line) for line in open(root / "meta" / "episode_objects.jsonl")]
    if not rows:
        raise SystemExit(f"No episode_objects.jsonl rows under {root}")

    by_mesh: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_mesh[r["base_mesh"]].append(r)

    # A base mesh's group is well defined: no mesh spans more than one group in this asset set.
    mesh_group = {m: rs[0]["group"] for m, rs in by_mesh.items()}
    expert = expert_success_by_base_mesh(args.src_root) if args.src_root else {}

    def rate(mesh: str) -> float:
        s, n = expert.get(mesh, (0, 0))
        return s / n if n else 1.0

    infeasible = sorted(m for m in by_mesh if expert.get(m, (1, 1))[0] == 0)
    eligible = [m for m in by_mesh if m not in infeasible]

    # Stratify by group, and within a group order by expert success so the held-out draw spans
    # the difficulty range instead of clustering at one end.
    rng = random.Random(args.seed)
    test_meshes: list[str] = []
    for group in sorted({mesh_group[m] for m in eligible}):
        members = sorted(m for m in eligible if mesh_group[m] == group)
        rng.shuffle(members)
        members.sort(key=rate)
        k = max(1, round(len(members) * args.test_frac)) if len(members) > 1 else 0
        if k:
            # Even spacing across the difficulty-sorted list.
            step = len(members) / k
            test_meshes.extend(members[min(int(i * step + step / 2), len(members) - 1)] for i in range(k))
    test_meshes = sorted(set(test_meshes))
    train_meshes = sorted(m for m in by_mesh if m not in test_meshes)

    overlap = set(train_meshes) & set(test_meshes)
    if overlap:
        raise SystemExit(f"Split is not disjoint: {sorted(overlap)}")

    train_eps = sorted(r["episode_index"] for m in train_meshes for r in by_mesh[m])
    test_eps = sorted(r["episode_index"] for m in test_meshes for r in by_mesh[m])

    out = root / "splits"
    out.mkdir(exist_ok=True)
    (out / "objects_train.txt").write_text("\n".join(train_meshes) + "\n")
    (out / "objects_test.txt").write_text("\n".join(test_meshes) + "\n")

    # The simulator spawns assets by full path, and one base mesh can appear under several
    # orientation variants (each its own asset). Emit those too so the eval runner does not have
    # to guess which group directory a bare mesh name lives in.
    def asset_paths(meshes: list[str]) -> list[str]:
        return sorted({r["object"] for m in meshes for r in by_mesh[m] if r.get("object")})

    (out / "objects_train_paths.txt").write_text("\n".join(asset_paths(train_meshes)) + "\n")
    (out / "objects_test_paths.txt").write_text("\n".join(asset_paths(test_meshes)) + "\n")
    (out / "episodes_train.json").write_text(json.dumps(train_eps))
    (out / "episodes_test.json").write_text(json.dumps(test_eps))

    # ---- curated evaluation sets ------------------------------------------------------
    # Failures on objects the demonstrator itself cannot do measure the demonstrator, and since
    # only successful episodes are recorded those objects contribute little training data either.
    # Curating raises the reachable ceiling: on the full set the expert scores ~70%, so 80%
    # absolute would need a policy to beat its own teacher.
    #
    # The cost is sample size, and it has to be paid back in rollouts per object -- a smaller
    # object set at the same rollout count resolves *less*, not more.
    by_path = expert_success_by_path(args.src_root) if args.src_root else {}
    curated_note = ""
    if by_path and args.eval_min_expert > 0:
        def rate_of(path: str) -> float:
            k, n = by_path.get(path, (0, 0))
            return k / n if n else 0.0

        held_curated = sorted(p for p in asset_paths(test_meshes) if rate_of(p) >= args.eval_min_expert)
        train_pool = sorted(p for p in asset_paths(train_meshes) if rate_of(p) >= args.eval_min_expert)

        # Match the in-domain count to the held-out count so the two numbers carry the same
        # statistical weight, and stratify by group so it is not accidentally all one shape.
        n_ind = args.indomain_eval_objects or len(held_curated)
        rng2 = random.Random(args.seed + 1)
        by_group: dict[str, list[str]] = defaultdict(list)
        for pth in train_pool:
            by_group[pth.split("/")[1]].append(pth)
        ind_curated: list[str] = []
        groups_cycle = sorted(by_group)
        for g in groups_cycle:
            rng2.shuffle(by_group[g])
        i = 0
        while len(ind_curated) < min(n_ind, len(train_pool)):
            g = groups_cycle[i % len(groups_cycle)]
            if by_group[g]:
                ind_curated.append(by_group[g].pop())
            i += 1
        ind_curated.sort()

        (out / "objects_test_curated_paths.txt").write_text("\n".join(held_curated) + "\n")
        (out / "objects_indomain_curated_paths.txt").write_text("\n".join(ind_curated) + "\n")

        def er(paths: list[str]) -> str:
            k = sum(by_path[p][0] for p in paths)
            n = sum(by_path[p][1] for p in paths)
            return f"{100 * k / n:.1f}%" if n else "n/a"

        curated_note = (
            f"\n## Curated evaluation sets (expert >= {100 * args.eval_min_expert:.0f}%)\n\n"
            f"- held-out: **{len(held_curated)}** of {len(asset_paths(test_meshes))} asset paths, "
            f"expert {er(held_curated)}\n"
            f"- in-domain: **{len(ind_curated)}** paths sampled from {len(train_pool)} eligible "
            f"training paths, expert {er(ind_curated)}\n"
            f"- dropped from held-out: "
            f"{', '.join(p.split('/')[-1] for p in asset_paths(test_meshes) if p not in held_curated)}\n"
        )
        print(f"curated eval: held-out {len(held_curated)} paths ({er(held_curated)}), "
              f"in-domain {len(ind_curated)} paths ({er(ind_curated)})")

    lines = [
        "# Push dataset object split",
        "",
        f"- grouping key: base mesh (orientation prefix stripped) -- {len(by_mesh)} meshes over {len(rows)} episodes",
        f"- seed {args.seed}, target held-out fraction {args.test_frac}",
        f"- **train**: {len(train_meshes)} meshes / {len(train_eps)} episodes",
        f"- **held-out**: {len(test_meshes)} meshes / {len(test_eps)} episodes",
        "",
        "## Held-out meshes",
        "",
        "| mesh | group | episodes | expert success |",
        "|---|---|---|---|",
    ]
    for m in test_meshes:
        s, n = expert.get(m, (0, 0))
        lines.append(f"| {m} | {mesh_group[m]} | {len(by_mesh[m])} | {s}/{n} |" if n else f"| {m} | {mesh_group[m]} | {len(by_mesh[m])} | n/a |")
    lines += ["", "## Per-group balance", "", "| group | train meshes | held-out meshes |", "|---|---|---|"]
    for g in sorted({v for v in mesh_group.values()}):
        lines.append(
            f"| {g} | {sum(1 for m in train_meshes if mesh_group[m] == g)} "
            f"| {sum(1 for m in test_meshes if mesh_group[m] == g)} |"
        )
    if infeasible:
        lines += [
            "",
            "## Expert-infeasible (0% scripted-expert success, kept out of the held-out set)",
            "",
            ", ".join(infeasible),
        ]
    (out / "split_report.md").write_text("\n".join(lines) + "\n" + curated_note)

    print(f"train: {len(train_meshes)} meshes / {len(train_eps)} episodes")
    print(f"test : {len(test_meshes)} meshes / {len(test_eps)} episodes")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
