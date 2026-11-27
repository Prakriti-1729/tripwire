"""Dedup + behavior-level split utilities."""
from __future__ import annotations

import random
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip(".!? ")


def dedup(rows: list[dict], near_thresh: float = 0.85) -> tuple[list[dict], int, int]:
    """Exact dedup on normalized text, then greedy near-dup removal (char 3-5gram TF-IDF cosine).

    Keeps the first occurrence, so put the preferred source first. Returns (kept, n_exact, n_near).
    """
    seen, out = set(), []
    for r in rows:
        k = normalize(r["text"])
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    n_exact = len(rows) - len(out)
    if len(out) < 2:
        return out, n_exact, 0
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform([normalize(r["text"]) for r in out])
    S = (X @ X.T).toarray()
    keep = np.ones(len(out), bool)
    for i in range(len(out)):
        if not keep[i]:
            continue
        dup = np.where(S[i, i + 1:] >= near_thresh)[0] + i + 1
        keep[dup] = False
    kept = [r for r, k in zip(out, keep) if k]
    return kept, n_exact, len(out) - len(kept)


def assign_splits(groups: list[str], labels: list[int], seed: int = 0,
                  fracs=(0.60, 0.15, 0.25)) -> dict[str, str]:
    """Split unique *groups* (behaviors) into train/val/test, stratified by label.

    A group may contain several prompts (e.g. a topic-matched JBB harmful/benign pair); all of its
    prompts and all of their wrapped variants inherit the group's split.
    """
    g_label: dict[str, int] = {}
    for g, y in zip(groups, labels):
        g_label[g] = max(g_label.get(g, 0), y)
    rng = random.Random(seed)
    out = {}
    for y in (0, 1):
        gs = sorted(g for g, l in g_label.items() if l == y)
        rng.shuffle(gs)
        n = len(gs)
        n_tr, n_va = round(fracs[0] * n), round(fracs[1] * n)
        for i, g in enumerate(gs):
            out[g] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    return out
