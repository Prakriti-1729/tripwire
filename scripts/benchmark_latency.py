"""RQ5 latency: batch size 1, prompts of ~64/256/1024 tokens, interleaved conditions, 300 runs (default) after
10 warmups, median + p90 (ms):
  (a) model prefill alone
  (b) model prefill + Tripwire probe (hook at chosen layer + dot product, score copied to host)
  (c) guard model forward pass (its own formatted prompt for the same user text)
  (d) early abort: Tripwire fires at layer L and the remaining layers are skipped (time to layer L)
Writes results/<slug>/latency.json
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tripwire.common import DEFAULT_MODEL, results_dir, write_json  # noqa: E402
from tripwire.model import encode_one, load_model  # noqa: E402
from tripwire.probe import LinearProbe, Tripwire, TripwireAbort  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from guard import load_guard  # noqa: E402

FILLER = ("The history of bread spans thousands of years, from flatbreads baked on hot stones to the sourdough "
          "loaves of modern bakeries. Bakers learned to control fermentation, flour quality and oven heat. ")


def gpu_status():
    """Utilisation/memory of OUR GPU (matched by UUID) and number of other compute processes on it."""
    try:
        uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid)
        q = subprocess.run(["nvidia-smi", "--query-gpu=uuid,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        for line in q.strip().splitlines():
            u, util, used, tot = [x.strip() for x in line.split(",")]
            if u == uuid:
                apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                                      capture_output=True, text=True, timeout=20).stdout
                n = sum(1 for l in apps.splitlines() if l.split(",")[0].strip() == uuid)
                return {"util_pct": int(util), "mem_used_mib": int(used), "mem_total_mib": int(tot),
                        "n_compute_procs_incl_ours": n}
    except Exception as e:
        return {"error": repr(e)}
    return {"error": "gpu not matched"}


def make_text(tok, target):
    """User text such that the full chat-formatted prompt is ~target tokens."""
    overhead = len(encode_one(tok, "x")[0]) - 1
    ids = tok(FILLER * (target // 20 + 2), add_special_tokens=False)["input_ids"][: max(8, target - overhead)]
    text = tok.decode(ids) + " Summarize this."
    return text


def timed(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) * 1000


def stats(xs):
    xs = np.asarray(xs)
    return {"median_ms": round(float(np.median(xs)), 3), "p90_ms": round(float(np.percentile(xs, 90)), 3),
            "min_ms": round(float(xs.min()), 3), "p10_ms": round(float(np.percentile(xs, 10)), 3),
            "mean_ms": round(float(xs.mean()), 3), "n": len(xs)}


@torch.no_grad()
def main(a):
    rd = results_dir(a.model)
    probe = LinearProbe.load(rd / "probe")
    model, tok = load_model(a.model)
    guard = None if a.no_guard else load_guard()
    guard2 = None
    if a.guard2 and not a.no_guard:
        guard2 = load_guard([a.guard2])
    status_before = gpu_status()
    res = {"model": a.model, "layer": probe.layer, "n_layers": model.config.num_hidden_layers,
           "pooling": probe.pooling, "guard": guard.name if guard else None,
           "guard2": guard2.name if guard2 else None, "runs": a.runs, "warmup": a.warmup,
           "batch_size": 1, "gpu": torch.cuda.get_device_name(0), "gpu_status_before": status_before, "lengths": {}}
    for target in a.lengths:
        text = make_text(tok, target)
        ids, um = encode_one(tok, text)
        x = torch.tensor([ids], device="cuda")
        umask = torch.tensor([um], device="cuda", dtype=torch.float32)

        def run_a():
            model(input_ids=x, logits_to_keep=1).logits[:, -1].argmax(-1).item()

        tw = Tripwire(model, probe)
        tw.umask = umask
        tw.remove()

        def run_b():
            tw.handle = model.model.layers[probe.layer - 1].register_forward_hook(tw._hook)
            try:
                out = model(input_ids=x, logits_to_keep=1)
                tw.last_score.item()
                out.logits[:, -1].argmax(-1).item()
            finally:
                tw.handle.remove()

        ab = LinearProbe(probe.layer, probe.pooling, probe.method, probe.w, probe.b, threshold=-1e30)
        twa = Tripwire(model, ab, early_abort=True)
        twa.umask = umask
        twa.remove()

        def run_d():
            twa.handle = model.model.layers[probe.layer - 1].register_forward_hook(twa._hook)
            try:
                model(input_ids=x, logits_to_keep=1)
                raise RuntimeError("early abort did not fire")
            except TripwireAbort as e:
                e.score.item()
            finally:
                twa.handle.remove()

        conds = {"a_prefill": run_a, "b_prefill_plus_probe": run_b, "d_early_abort_at_L": run_d}
        if guard is not None:
            gx = torch.tensor([guard.encode(text)], device="cuda")

            def run_c():
                guard.model(input_ids=gx, logits_to_keep=1).logits[:, -1].float().argmax(-1).item()
            conds["c_guard_forward"] = run_c
        if guard2 is not None:
            gx2 = torch.tensor([guard2.encode(text)], device="cuda")

            def run_c2():
                guard2.model(input_ids=gx2, logits_to_keep=1).logits[:, -1].float().argmax(-1).item()
            conds["c2_guard2_forward"] = run_c2
        times = {k: [] for k in conds}
        for _ in range(a.warmup):
            for f in conds.values():
                f()
        for _ in range(a.runs):
            for k, f in conds.items():  # interleaved to share noise from other users' jobs
                times[k].append(timed(f))
        r = {k: stats(v) for k, v in times.items()}
        r["prompt_tokens"] = len(ids)
        if guard is not None:
            r["guard_prompt_tokens"] = int(gx.shape[1])
        pa = r["a_prefill"]["median_ms"]
        r["probe_added_ms"] = round(r["b_prefill_plus_probe"]["median_ms"] - pa, 3)
        # least-contention estimate: difference of the 10th percentiles
        r["probe_added_ms_p10"] = round(r["b_prefill_plus_probe"]["p10_ms"] - r["a_prefill"]["p10_ms"], 3)
        r["probe_overhead_pct"] = round(100 * r["probe_added_ms"] / pa, 2)
        r["early_abort_saving_pct"] = round(100 * (1 - r["d_early_abort_at_L"]["median_ms"] / pa), 1)
        if guard is not None:
            r["guard_added_ms"] = r["c_guard_forward"]["median_ms"]
            r["guard_overhead_pct"] = round(100 * r["guard_added_ms"] / pa, 1)
        if guard2 is not None:
            r["guard2_added_ms"] = r["c2_guard2_forward"]["median_ms"]
            r["guard2_overhead_pct"] = round(100 * r["guard2_added_ms"] / pa, 1)
        res["lengths"][str(target)] = r
        print(target, {k: (v["median_ms"] if isinstance(v, dict) else v) for k, v in r.items()}, flush=True)
    # isolated probe op on a realistic hidden state (256 tokens): pooling + dot product + host copy
    T, D = 256, model.config.hidden_size
    h = torch.randn(1, T, D, device="cuda", dtype=next(model.parameters()).dtype)
    um = torch.ones(1, T, device="cuda")
    w = torch.tensor(probe.w, device="cuda")

    def op():
        x = h.float()
        v = x[:, -1, :] if probe.pooling == "last" else (x * um[..., None]).sum(1) / um.sum(1, keepdim=True)
        (v @ w + probe.b).item()
    for _ in range(50):
        op()
    res["probe_op_only"] = dict(stats([timed(op) for _ in range(1000)]), dim=D,
                                note="each run includes a host sync (.item()); on a time-sliced shared GPU the sync wait dominates")

    def op_nosync():
        x = h.float()
        v = x[:, -1, :] if probe.pooling == "last" else (x * um[..., None]).sum(1) / um.sum(1, keepdim=True)
        return v @ w + probe.b
    am = []
    for _ in range(20):
        am.append(timed(lambda: [op_nosync() for _ in range(1000)]) / 1000)
    res["probe_op_amortized"] = dict(stats(am), note="GPU time per probe op, 1000 ops per sync (20 repeats)")
    res["gpu_status_after"] = gpu_status()
    write_json(rd / "latency.json", res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--lengths", type=int, nargs="+", default=[64, 256, 1024])
    ap.add_argument("--runs", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--no-guard", action="store_true")
    ap.add_argument("--guard2", default="ibm-granite/granite-guardian-3.1-2b",
                    help="second (dense) guard, timed for latency only; '' to skip")
    main(ap.parse_args())
