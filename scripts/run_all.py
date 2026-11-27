"""Reproduce everything from scratch:  python scripts/run_all.py [--model ...] [--also-small]

data -> activations -> probes -> generations (RQ3) -> response judge -> guard -> evaluate -> latency -> figures -> report.
Each step logs to logs/<step>.log; the run stops at the first failing step.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def step(name, args, log):
    t = time.time()
    print(f"== {name}: {' '.join(args)}  (log: {log})", flush=True)
    with open(log, "w") as f:
        r = subprocess.run([PY, *args], stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"step {name} failed (exit {r.returncode}); see {log}")
    print(f"   done in {time.time() - t:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--also-small", action="store_true", help="also run Qwen2.5-1.5B-Instruct (scale comparison)")
    ap.add_argument("--skip-data", action="store_true")
    a = ap.parse_args()
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    if not a.skip_data:
        step("build_data", ["scripts/build_data.py"], logs / "build_data.log")
    models = [(a.model, True)] + ([("Qwen/Qwen2.5-1.5B-Instruct", False)] if a.also_small else [])
    for m, primary in models:
        tag = m.split("/")[-1]
        step(f"extract[{tag}]", ["scripts/extract_acts.py", "--model", m], logs / f"extract_{tag}.log")
        step(f"train[{tag}]", ["scripts/train_probes.py", "--model", m], logs / f"train_{tag}.log")
        step(f"generate[{tag}]", ["scripts/generate.py", "--model", m], logs / f"generate_{tag}.log")
        step(f"judge[{tag}]", ["scripts/judge.py", "--model", m], logs / f"judge_{tag}.log")
        step(f"guard[{tag}]", ["scripts/guard.py", "--model", m], logs / f"guard_{tag}.log")
        step(f"evaluate[{tag}]", ["scripts/evaluate.py", "--model", m], logs / f"evaluate_{tag}.log")
        step(f"latency[{tag}]", ["scripts/benchmark_latency.py", "--model", m], logs / f"latency_{tag}.log")
        step(f"figures[{tag}]", ["scripts/make_figures.py", "--model", m] + (["--primary"] if primary else []),
             logs / f"figures_{tag}.log")
    step("report", ["scripts/make_report.py", "--model", a.model], logs / "report.log")


if __name__ == "__main__":
    main()
