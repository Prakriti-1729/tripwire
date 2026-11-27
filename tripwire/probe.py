"""Linear probes and the runtime Tripwire (forward hook at one layer + one dot product)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch


@dataclass
class LinearProbe:
    """score(x) = w . x + b  (StandardScaler folded into w, b for LR; raw projection for diff-of-means)."""
    layer: int
    pooling: str  # "last" | "mean"
    method: str  # "lr" | "dom"
    w: np.ndarray
    b: float
    threshold: float = 0.0
    meta: dict = field(default_factory=dict)

    def score(self, X: np.ndarray) -> np.ndarray:
        return X.astype(np.float32) @ self.w.astype(np.float32) + self.b

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix(".npy"), self.w.astype(np.float32))
        d = {k: v for k, v in asdict(self).items() if k != "w"}
        path.with_suffix(".json").write_text(json.dumps(d, indent=2))

    @classmethod
    def load(cls, path) -> "LinearProbe":
        path = Path(path)
        d = json.loads(path.with_suffix(".json").read_text())
        return cls(w=np.load(path.with_suffix(".npy")), **d)


def fit_lr(Xtr, ytr, Xva, yva, Cs=(1e-3, 1e-2, 1e-1, 1.0), layer=0, pooling="last"):
    """StandardScaler + LogisticRegression, C tuned on val AUROC; scaler folded into (w, b)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    Xtr = Xtr.astype(np.float32)
    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-6
    Ztr = (Xtr - mu) / sd
    Zva = (Xva.astype(np.float32) - mu) / sd
    best = None
    for C in Cs:
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(Ztr, ytr)
        auc = roc_auc_score(yva, clf.decision_function(Zva))
        if best is None or auc > best[0] + 1e-9:
            best = (auc, C, clf)
    auc, C, clf = best
    w = clf.coef_[0] / sd
    b = float(clf.intercept_[0] - (clf.coef_[0] * mu / sd).sum())
    return LinearProbe(layer, pooling, "lr", w.astype(np.float32), b, meta={"C": C, "val_auroc": auc})


def fit_dom(Xtr, ytr, layer=0, pooling="last"):
    """Difference-of-means direction (unit norm); score = projection, centred on the class midpoint."""
    Xtr = Xtr.astype(np.float32)
    m1, m0 = Xtr[ytr == 1].mean(0), Xtr[ytr == 0].mean(0)
    d = m1 - m0
    d = d / (np.linalg.norm(d) + 1e-8)
    b = -float(d @ (m1 + m0) / 2)
    return LinearProbe(layer, pooling, "dom", d.astype(np.float32), b)


def threshold_at_fpr(benign_scores: np.ndarray, fpr: float = 0.01) -> float:
    """Smallest threshold whose FPR (score > t) on these benign scores is <= fpr."""
    s = np.sort(np.asarray(benign_scores, np.float64))
    k = int(np.floor(fpr * len(s)))  # number of benign allowed above threshold
    return float(s[len(s) - 1 - k]) if k < len(s) else float(s[0])


class Tripwire:
    """Attach a probe to a live HF model via a forward hook on decoder block ``layer-1``.

    After a forward pass, ``self.last_score`` holds the probe score for each sequence in the batch.
    With ``early_abort=True`` the hook raises :class:`TripwireAbort` once the probe fires, so the
    remaining layers are never computed.
    """

    def __init__(self, model, probe: LinearProbe, early_abort: bool = False):
        self.model, self.probe, self.early_abort = model, probe, early_abort
        dev = next(model.parameters()).device
        self.w = torch.tensor(probe.w, device=dev, dtype=torch.float32)
        self.b = float(probe.b)
        self.umask = None  # set before forward for mean pooling: (B, T) float
        self.last_score = None
        self.per_token = False  # if True, score the last position regardless of pooling (decode steps)
        inner = model.model
        n = model.config.num_hidden_layers
        assert 1 <= probe.layer <= n, "probe on embedding layer (0) is not supported at runtime"
        self.final_norm = inner.norm if probe.layer == n else None
        self.handle = inner.layers[probe.layer - 1].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if self.final_norm is not None:
            h = self.final_norm(h)
        h = h.float()
        if self.probe.pooling == "last" or self.per_token or self.umask is None:
            x = h[:, -1, :]
        else:
            m = self.umask.unsqueeze(-1).to(h.dtype)
            x = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
        self.last_score = x @ self.w + self.b
        if self.early_abort and bool((self.last_score > self.probe.threshold).any()):
            raise TripwireAbort(self.last_score)

    def remove(self):
        self.handle.remove()


class TripwireAbort(Exception):
    def __init__(self, score):
        super().__init__("tripwire fired")
        self.score = score
