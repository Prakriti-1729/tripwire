"""Build data/*.jsonl: plain prompts, wrapped test set, augmentation set, XSTest.

Record schema: {id, text, label (1=harmful, 0=benign), source, behavior_id, template, split}
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import random
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tripwire.common import DATA, write_json, write_jsonl  # noqa: E402
from tripwire.data import assign_splits, dedup  # noqa: E402
from tripwire.templates import FAMILIES, GROUP_B, group_of, wrap  # noqa: E402

URLS = {
    "harmbench": "https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv",
    "advbench": "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
    "xstest": "https://raw.githubusercontent.com/paul-rottger/exaggerated-safety/main/xstest_prompts.csv",
}
log_lines: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    log_lines.append(msg)


def fetch_csv(name: str) -> pd.DataFrame | None:
    raw = DATA / "raw" / Path(URLS[name]).name
    try:
        if not raw.exists():
            raw.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(URLS[name], timeout=60) as r:
                raw.write_bytes(r.read())
        return pd.read_csv(io.BytesIO(raw.read_bytes()))
    except Exception as e:  # documented fallback: drop the source and continue
        log(f"WARNING: source {name} unavailable ({e!r}); dropping it")
        return None


def main(args):
    rng = random.Random(args.seed)
    from datasets import load_dataset

    harmful, benign = [], []
    # --- JailbreakBench (harmful + topic-matched benign, paired by Index -> same group/split)
    try:
        jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
        for r in jbb["harmful"]:
            harmful.append(dict(text=r["Goal"], source="jbb", group=f"jbb-{r['Index']}"))
        for r in jbb["benign"]:
            benign.append(dict(text=r["Goal"], source="jbb_benign", group=f"jbb-{r['Index']}"))
        log(f"JBB: {len(jbb['harmful'])} harmful, {len(jbb['benign'])} benign")
    except Exception as e:
        log(f"WARNING: JBB unavailable ({e!r}); dropping it")
    # --- HarmBench (standard functional category only)
    hb = fetch_csv("harmbench")
    if hb is not None:
        hb = hb[hb["FunctionalCategory"] == "standard"]
        for _, r in hb.iterrows():
            harmful.append(dict(text=r["Behavior"], source="harmbench", group=f"hb-{r['BehaviorID']}"))
        log(f"HarmBench standard: {len(hb)}")
    # --- AdvBench
    ab = fetch_csv("advbench")
    if ab is not None:
        for i, r in ab.iterrows():
            harmful.append(dict(text=r["goal"], source="advbench", group=f"adv-{i}"))
        log(f"AdvBench: {len(ab)}")

    harmful, ne, nn = dedup(harmful, args.near_thresh)
    log(f"Harmful after dedup: {len(harmful)} (removed {ne} exact, {nn} near-duplicates @ cos>={args.near_thresh})")

    # --- Alpaca: instruction-only, ~1.5x harmful count
    alp = load_dataset("tatsu-lab/alpaca", split="train")
    pool = [r["instruction"] for r in alp if not r["input"].strip() and 15 <= len(r["instruction"]) <= 400]
    rng.shuffle(pool)
    n_alp = int(round(1.5 * len(harmful)))
    for i, t in enumerate(pool[: int(n_alp * 1.1)]):
        benign.append(dict(text=t, source="alpaca", group=f"alp-{i}"))
    benign, ne, nn = dedup(benign, args.near_thresh)
    # keep all JBB benign, trim alpaca to target
    jb = [b for b in benign if b["source"] == "jbb_benign"]
    al = [b for b in benign if b["source"] == "alpaca"][:n_alp]
    benign = jb + al
    log(f"Benign: {len(jb)} JBB-benign + {len(al)} Alpaca = {len(benign)} (dedup removed {ne} exact, {nn} near)")

    # --- splits by behavior group
    rows = [dict(r, label=1) for r in harmful] + [dict(r, label=0) for r in benign]
    split_of = assign_splits([r["group"] for r in rows], [r["label"] for r in rows], seed=args.seed)
    plain = []
    for i, r in enumerate(rows):
        plain.append(dict(id=f"p{i:05d}", text=r["text"], label=r["label"], source=r["source"],
                          behavior_id=r["group"], template="none", split=split_of[r["group"]]))
    if args.tiny:
        by = collections.defaultdict(list)
        for r in plain:
            by[(r["split"], r["label"])].append(r)
        plain = [r for k in sorted(by) for r in by[k][: args.tiny]]
    write_jsonl(DATA / "plain.jsonl", plain)

    # --- wrapped TEST set: every test harmful AND benign prompt x every template
    wt = [dict(id=f"{r['id']}-{f}", text=wrap(r["text"], f), label=r["label"], source=r["source"],
               behavior_id=r["behavior_id"], template=f, split="test")
          for r in plain if r["split"] == "test" for f in FAMILIES]
    write_jsonl(DATA / "wrapped_test.jsonl", wt)
    # --- augmentation set (for the B-augmented probe variant): train/val prompts x group-B templates
    wa = [dict(id=f"{r['id']}-{f}", text=wrap(r["text"], f), label=r["label"], source=r["source"],
               behavior_id=r["behavior_id"], template=f, split=r["split"])
          for r in plain if r["split"] in ("train", "val") for f in GROUP_B]
    write_jsonl(DATA / "wrapped_aug.jsonl", wa)

    # --- XSTest (eval only): safe prompts label 0, unsafe contrast prompts label 1
    xs = fetch_csv("xstest")
    xrows = []
    if xs is not None:
        for _, r in xs.iterrows():
            xrows.append(dict(id=f"xs{int(r['id']):04d}", text=r["prompt"], label=int(r["label"] == "unsafe"),
                              source=f"xstest_{r['type']}", behavior_id=f"xs-{r['id']}", template="none",
                              split="xstest"))
        if args.tiny:
            xrows = xrows[: args.tiny * 2] + xrows[-args.tiny:]
    write_jsonl(DATA / "xstest.jsonl", xrows)

    # --- manifest (tiny, committed)
    cnt = collections.Counter((r["split"], r["label"]) for r in plain)
    manifest = {
        "plain": {f"{s}_{'harmful' if y else 'benign'}": c for (s, y), c in sorted(cnt.items())},
        "plain_by_source": dict(collections.Counter(r["source"] for r in plain)),
        "wrapped_test": len(wt), "wrapped_aug": len(wa),
        "xstest": dict(collections.Counter("unsafe" if r["label"] else "safe" for r in xrows)),
        "families": {f: group_of(f) for f in FAMILIES},
        "log": log_lines,
    }
    write_json(DATA / "manifest.json", manifest)
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--near-thresh", type=float, default=0.85)
    ap.add_argument("--tiny", type=int, default=0, help="keep N per (split,label) for a smoke run")
    main(ap.parse_args())
