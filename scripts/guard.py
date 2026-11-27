"""Guard-model baseline: score test prompts with an open guard model.

Granite Guardian (per its model card): apply the chat template with guardian_config={"risk_name": "harm"}
and add_generation_prompt=True; the model answers "Yes" (risky) or "No". We read the first answer token's
logits in ONE forward pass: score = logit("Yes") - logit("No"); the model's own decision is score > 0.

Outputs results/<model_slug>/guard_scores.json  ({"guard": name, "scores": {id: float}})
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tripwire.common import DATA, DEFAULT_MODEL, read_jsonl, results_dir, write_json  # noqa: E402

CANDIDATES = ["ibm-granite/granite-guardian-3.2-3b-a800m", "ibm-granite/granite-guardian-3.1-2b"]


class Guard:
    def __init__(self, name: str, device="cuda"):
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map=device).eval()
        self.yes = self.tok.encode("Yes", add_special_tokens=False)
        self.no = self.tok.encode("No", add_special_tokens=False)
        assert len(self.yes) == 1 and len(self.no) == 1, (self.yes, self.no)

    def encode(self, prompt: str) -> list[int]:
        out = self.tok.apply_chat_template([{"role": "user", "content": prompt}], guardian_config={"risk_name": "harm"},
                                           add_generation_prompt=True, tokenize=True)
        return list(out["input_ids"]) if hasattr(out, "keys") else list(out)

    def encode_pair(self, prompt: str, response: str) -> list[int]:
        """Judge the ASSISTANT message (response) for harm, in the context of the user prompt."""
        msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        out = self.tok.apply_chat_template(msgs, guardian_config={"risk_name": "harm"}, add_generation_prompt=True,
                                           tokenize=True)
        return list(out["input_ids"]) if hasattr(out, "keys") else list(out)

    @torch.no_grad()
    def score_ids(self, batch: list[list[int]]) -> torch.Tensor:
        L = max(map(len, batch))
        pad = self.tok.pad_token_id
        ids = torch.tensor([[pad] * (L - len(x)) + x for x in batch], device=self.model.device)
        attn = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in batch], device=self.model.device)
        pos = (attn.cumsum(-1) - 1).clamp_min(0)
        logits = self.model(input_ids=ids, attention_mask=attn, position_ids=pos, logits_to_keep=1).logits[:, -1].float()
        return logits[:, self.yes[0]] - logits[:, self.no[0]]

    def score(self, prompts: list[str], batch_size=32, responses: list[str] | None = None) -> list[float]:
        enc = [self.encode(p) for p in prompts] if responses is None else \
            [self.encode_pair(p, r) for p, r in zip(prompts, responses)]
        order = sorted(range(len(enc)), key=lambda i: -len(enc[i]))
        out = [0.0] * len(enc)
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            for i, v in zip(idx, self.score_ids([enc[i] for i in idx]).tolist()):
                out[i] = v
        return out


def load_guard(names=CANDIDATES):
    for n in names:
        try:
            g = Guard(n)
            g.score(["How do I bake bread?"])
            return g
        except Exception as e:
            print(f"guard {n} failed: {e!r}", flush=True)
    return None


def main(a):
    g = load_guard([a.guard] if a.guard else CANDIDATES)
    rd = results_dir(a.model)
    if g is None:
        write_json(rd / "guard_scores.json", {"guard": None, "scores": {}})
        print("no guard model available; skipped", flush=True)
        return
    rows = [r for r in read_jsonl(DATA / "plain.jsonl") if r["split"] in ("val", "test")]
    rows += read_jsonl(DATA / "wrapped_test.jsonl") + read_jsonl(DATA / "xstest.jsonl")
    t0 = time.time()
    sc = g.score([r["text"] for r in rows], a.batch_size)
    write_json(rd / "guard_scores.json", {"guard": g.name, "risk_name": "harm", "decision": "score > 0 (Yes over No)",
                                          "scores": {r["id"]: round(s, 4) for r, s in zip(rows, sc)}})
    print(f"guard {g.name}: scored {len(rows)} prompts in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, help="primary model (results folder)")
    ap.add_argument("--guard", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    main(ap.parse_args())
