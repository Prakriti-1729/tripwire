"""Runtime used by the demos: prefill -> Tripwire verdict -> (if safe) stream a greedy answer.

Optional per-token mode: during decoding the same probe is applied to each new token's hidden state at
layer L and generation stops if it fires (experimental: the probe was trained on prompt representations).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .common import DEFAULT_MODEL, results_dir
from .model import encode_one, load_model
from .probe import LinearProbe, Tripwire


@dataclass
class Verdict:
    score: float
    fired: bool
    layer: int
    threshold: float
    prefill_ms: float


class TripwireServer:
    def __init__(self, model_name: str = DEFAULT_MODEL, probe_path: str | Path | None = None):
        self.model, self.tok = load_model(model_name)
        self.probe = LinearProbe.load(probe_path or results_dir(model_name) / "probe")
        self.tw = Tripwire(self.model, self.probe)
        self.check("Hello!")  # warm up CUDA kernels so the first real verdict's timing is representative

    @torch.no_grad()
    def _prefill(self, prompt: str):
        ids, um = encode_one(self.tok, prompt)
        x = torch.tensor([ids], device=self.model.device)
        self.tw.umask = torch.tensor([um], device=self.model.device, dtype=torch.float32)
        self.tw.per_token = False
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = self.model(input_ids=x, use_cache=True, logits_to_keep=1)
        score = float(self.tw.last_score[0])
        ms = (time.perf_counter() - t) * 1000
        v = Verdict(score, score > self.probe.threshold, self.probe.layer, self.probe.threshold, ms)
        return v, out

    def check(self, prompt: str) -> Verdict:
        return self._prefill(prompt)[0]

    @torch.no_grad()
    def respond(self, prompt: str, max_new_tokens: int = 256, per_token: bool = False):
        """Yields ("verdict", Verdict), then ("text", full_text_so_far)..., then ("stop", reason)."""
        v, out = self._prefill(prompt)
        yield "verdict", v
        if v.fired:
            yield "stop", f"blocked before the first token: Tripwire triggered at layer {v.layer} (score {v.score:.2f})"
            return
        past = out.past_key_values
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        gen = []
        self.tw.per_token = True
        eos = {self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")}
        for step in range(max_new_tokens):
            tid = int(nxt[0, 0])
            if tid in eos:
                yield "stop", "end of answer"
                return
            gen.append(tid)
            yield "text", self.tok.decode(gen, skip_special_tokens=True)
            out = self.model(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            s = float(self.tw.last_score[0])
            if per_token and s > self.probe.threshold:
                yield "stop", f"stopped mid-stream at token {step + 1}: per-token Tripwire fired (score {s:.2f})"
                return
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        yield "stop", "max tokens reached"
