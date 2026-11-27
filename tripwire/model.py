"""Model loading, chat-template tokenisation with user-token masks, and batched prefill extraction.

Layer convention: ``hidden_states[i]`` for i = 0..N, where 0 is the embedding output and i >= 1 is the
residual stream after decoder block i-1 (HF ``output_hidden_states`` convention). In HF the last entry
(i = N) has the final RMSNorm applied; :class:`tripwire.probe.Tripwire` reproduces that when hooking.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(name: str, dtype: str | None = None, device: str = "cuda"):
    if dtype is None:
        dtype = "float16" if "1.5B" in name else "bfloat16"
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype), device_map=device)
    model.eval()
    return model, tok


def chat_text(tok, prompt: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True)


def encode_one(tok, prompt: str) -> tuple[list[int], list[int]]:
    """Token ids of the chat-formatted prompt and a 0/1 mask of the tokens inside the user message."""
    text = chat_text(tok, prompt)
    start = text.rfind(prompt)
    assert start >= 0, "user prompt not found in chat-formatted text"
    end = start + len(prompt)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    mask = [int(s < end and e > start) for s, e in enc["offset_mapping"]]
    assert sum(mask) > 0
    return enc["input_ids"], mask


def collate(batch: list[tuple[list[int], list[int]]], pad_id: int, device="cuda"):
    """Left-pad a list of (ids, user_mask)."""
    L = max(len(ids) for ids, _ in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), L), dtype=torch.long)
    umask = torch.zeros((len(batch), L), dtype=torch.float32)
    for i, (x, m) in enumerate(batch):
        n = len(x)
        ids[i, L - n:] = torch.tensor(x)
        attn[i, L - n:] = 1
        umask[i, L - n:] = torch.tensor(m, dtype=torch.float32)
    return ids.to(device), attn.to(device), umask.to(device)


def position_ids(attn: torch.Tensor) -> torch.Tensor:
    return (attn.cumsum(-1) - 1).clamp_min(0)


def pool(hidden: torch.Tensor, umask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """hidden: (B, T, D) left-padded. Returns (last, mean-over-user-tokens), each (B, D) float32."""
    h = hidden.float()
    last = h[:, -1, :]  # left padding => final position is always the last real token
    m = umask.unsqueeze(-1)
    mean = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
    return last, mean


@torch.no_grad()
def extract(model, tok, prompts: list[str], batch_size: int = 32, max_tokens_per_batch: int = 32768,
            progress: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Prefill-only extraction for every layer. Returns (last, mean) arrays of shape (n, N+1, D), fp16."""
    from tqdm import tqdm

    enc = [encode_one(tok, p) for p in prompts]
    order = np.argsort([len(e[0]) for e in enc])[::-1]
    n_layers = model.config.num_hidden_layers + 1
    D = model.config.hidden_size
    out_last = np.zeros((len(prompts), n_layers, D), np.float16)
    out_mean = np.zeros((len(prompts), n_layers, D), np.float16)
    i = 0
    pbar = tqdm(total=len(prompts), disable=not progress, mininterval=10)
    while i < len(order):
        L = len(enc[order[i]][0])
        bs = max(1, min(batch_size, max_tokens_per_batch // L))
        idx = order[i:i + bs]
        ids, attn, um = collate([enc[j] for j in idx], tok.pad_token_id)
        out = model(input_ids=ids, attention_mask=attn, position_ids=position_ids(attn),
                    output_hidden_states=True, logits_to_keep=1)
        for li, h in enumerate(out.hidden_states):
            last, mean = pool(h, um)
            out_last[idx, li] = last.cpu().numpy().astype(np.float16)
            out_mean[idx, li] = mean.cpu().numpy().astype(np.float16)
        i += bs
        pbar.update(len(idx))
    pbar.close()
    return out_last, out_mean
