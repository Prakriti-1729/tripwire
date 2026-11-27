"""Prefill-only activation extraction for every layer -> acts/<model_slug>/<dataset>_{last,mean}.npy"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tripwire.common import ACTS, DATA, DATASETS, DEFAULT_MODEL, read_jsonl, slug  # noqa: E402
from tripwire.model import extract, load_model  # noqa: E402


def main(a):
    free, total = torch.cuda.mem_get_info()
    print(f"GPU free {free/1e9:.1f} GB / {total/1e9:.1f} GB", flush=True)
    out = ACTS / slug(a.model)
    out.mkdir(parents=True, exist_ok=True)
    model, tok = load_model(a.model)
    print(f"model loaded: {a.model}, layers={model.config.num_hidden_layers}, d={model.config.hidden_size}, "
          f"mem used {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
    for name in a.datasets:
        rows = read_jsonl(DATA / f"{name}.jsonl")
        ids = [r["id"] for r in rows]
        idf = out / f"{name}_ids.json"
        if not a.force and idf.exists() and json.loads(idf.read_text()) == ids:
            print(f"{name}: up to date ({len(ids)}), skipping", flush=True)
            continue
        t0 = time.time()
        bs = a.batch_size
        while True:
            try:
                last, mean = extract(model, tok, [r["text"] for r in rows], batch_size=bs,
                                     max_tokens_per_batch=a.max_tokens)
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                a.max_tokens //= 2
                print(f"OOM -> batch_size {bs}, max_tokens {a.max_tokens}", flush=True)
        np.save(out / f"{name}_last.npy", last)
        np.save(out / f"{name}_mean.npy", mean)
        idf.write_text(json.dumps(ids))
        print(f"{name}: {last.shape} in {time.time()-t0:.0f}s, peak mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB",
              flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args())
