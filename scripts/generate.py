"""RQ3 model-behaviour baseline: greedy 64-token completions for TEST harmful prompts (plain + wrapped),
classified with the keyword refusal detector.

Completions -> outputs/<slug>/generations.jsonl (git-ignored, never committed).
Flags only  -> results/<slug>/refusals.json  ({id: 0/1}) — no completion text.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tripwire.common import DATA, DEFAULT_MODEL, OUTPUTS, read_jsonl, results_dir, slug, write_json, write_jsonl  # noqa: E402
from tripwire.model import collate, encode_one, load_model, position_ids  # noqa: E402
from tripwire.refusal import is_refusal  # noqa: E402


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens=64, batch_size=64):
    enc = [encode_one(tok, p) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: -len(enc[i][0]))
    out = [None] * len(prompts)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        ids, attn, _ = collate([enc[i] for i in idx], tok.pad_token_id)
        g = model.generate(input_ids=ids, attention_mask=attn, max_new_tokens=max_new_tokens, do_sample=False,
                           temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(g[:, ids.shape[1]:], skip_special_tokens=True)
        for i, t in zip(idx, texts):
            out[i] = t
        print(f"  {min(s + batch_size, len(order))}/{len(order)}", flush=True)
    return out


def main(a):
    rows = [r for r in read_jsonl(DATA / "plain.jsonl") if r["split"] == "test" and r["label"] == 1]
    rows += [r for r in read_jsonl(DATA / "wrapped_test.jsonl") if r["label"] == 1]
    if a.benign_too:
        rows += [r for r in read_jsonl(DATA / "plain.jsonl") if r["split"] == "test" and r["label"] == 0]
        rows += [r for r in read_jsonl(DATA / "wrapped_test.jsonl") if r["label"] == 0]
    print(f"GPU free {torch.cuda.mem_get_info()[0]/1e9:.1f} GB; generating for {len(rows)} prompts", flush=True)
    model, tok = load_model(a.model)
    t0 = time.time()
    comps = generate(model, tok, [r["text"] for r in rows], a.max_new_tokens, a.batch_size)
    flags = {r["id"]: int(is_refusal(c)) for r, c in zip(rows, comps)}
    write_jsonl(OUTPUTS / slug(a.model) / "generations.jsonl",
                [dict(id=r["id"], template=r["template"], label=r["label"], refusal=flags[r["id"]], completion=c)
                 for r, c in zip(rows, comps)])
    write_json(results_dir(a.model) / "refusals.json", flags)
    print(f"done in {time.time()-t0:.0f}s; refusal rate {sum(flags.values())/len(flags):.3f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--benign-too", action="store_true")
    main(ap.parse_args())
