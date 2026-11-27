"""Train probes for every layer x pooling (LR + diff-of-means) on PLAIN train prompts; pick the best by
plain-val AUROC; set the threshold at 1% FPR on val benign. Also trains the TF-IDF text baseline and
the group-B-augmented probe variant.

Outputs (results/<slug>/): layer_sweep.json, selection.json, probe.{npy,json}, probe_aug.{npy,json},
probe_dom.{npy,json}, tfidf.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tripwire.common import ACTS, DATA, DEFAULT_MODEL, read_jsonl, results_dir, slug, write_json  # noqa: E402
from tripwire.probe import fit_dom, fit_lr, threshold_at_fpr  # noqa: E402

POOLINGS = ("last", "mean")


def load_acts(model, name, pooling, mmap=True):
    d = ACTS / slug(model)
    ids = json.loads((d / f"{name}_ids.json").read_text())
    rows = {r["id"]: r for r in read_jsonl(DATA / f"{name}.jsonl")}
    assert set(ids) == set(rows), f"{name}: acts out of sync with data; re-run extract_acts.py"
    X = np.load(d / f"{name}_{pooling}.npy", mmap_mode="r" if mmap else None)
    return X, [rows[i] for i in ids]


def tpr_at(scores_pos, thr):
    return float((np.asarray(scores_pos) > thr).mean())


def sweep_layer(layer, pooling, Xtr, ytr, Xva, yva, Xte, yte, Xw, yw, jbb_te, xs_X, xs_y):
    out = {}
    for method in ("lr", "dom"):
        p = fit_lr(Xtr, ytr, Xva, yva, layer=layer, pooling=pooling) if method == "lr" else fit_dom(Xtr, ytr, layer, pooling)
        sva = p.score(Xva)
        thr = threshold_at_fpr(sva[yva == 0], 0.01)
        out[method] = {
            "val_auroc": float(roc_auc_score(yva, sva)),
            "val_tpr_at_1fpr": tpr_at(sva[yva == 1], thr),
            "test_auroc": float(roc_auc_score(yte, p.score(Xte))),
            "wrapped_test_auroc": float(roc_auc_score(yw, p.score(Xw))),
            # topic-matched subset: JBB harmful vs JBB benign test behaviors (same topics, harder)
            "test_auroc_jbb_matched": float(roc_auc_score(yte[jbb_te], p.score(Xte[jbb_te]))),
            "xstest_auroc": float(roc_auc_score(xs_y, p.score(xs_X))),
            "C": p.meta.get("C"),
        }
    return layer, pooling, out


def main(a):
    t0 = time.time()
    rd = results_dir(a.model)
    sweep = {}
    cache = {}
    for pooling in POOLINGS:
        X, rows = load_acts(a.model, "plain", pooling, mmap=False)
        Xw, wrows = load_acts(a.model, "wrapped_test", pooling, mmap=False)
        y = np.array([r["label"] for r in rows])
        sp = np.array([r["split"] for r in rows])
        yw = np.array([r["label"] for r in wrows])
        tr, va, te = sp == "train", sp == "val", sp == "test"
        jbb_te = np.array([r["source"] in ("jbb", "jbb_benign") for r in rows])[te]
        Xx, xrows = load_acts(a.model, "xstest", pooling, mmap=False)
        yx = np.array([r["label"] for r in xrows])
        cache[pooling] = (X, y, sp)
        n_layers = X.shape[1]
        jobs = (delayed(sweep_layer)(l, pooling, X[tr, l], y[tr], X[va, l], y[va], X[te, l], y[te], Xw[:, l], yw,
                                      jbb_te, Xx[:, l], yx)
                for l in range(n_layers))
        for l, pl, res in Parallel(n_jobs=a.jobs, verbose=0)(jobs):
            sweep[f"{pl}/{l}"] = res
            print(f"{pl:4s} L{l:02d}  " + "  ".join(
                f"{m}: val {r['val_auroc']:.4f} test {r['test_auroc']:.4f} wrapped {r['wrapped_test_auroc']:.4f}"
                for m, r in res.items()), flush=True)
        del Xw
    write_json(rd / "layer_sweep.json", sweep)

    # ---- selection: plain-val AUROC only (rounded to 4 dp), then val TPR@1%FPR, then earlier layer
    cands = []
    for key, res in sweep.items():
        pl, l = key.split("/")
        if int(l) == 0:
            continue  # embedding layer: not a residual-stream readout of the prompt's meaning
        for m, r in res.items():
            cands.append((round(r["val_auroc"], 4), r["val_tpr_at_1fpr"], -int(l), pl, m))
    cands.sort(reverse=True)
    _, _, negl, pooling, method = cands[0]
    layer = -negl
    X, y, sp = cache[pooling]
    tr, va = sp == "train", sp == "val"
    if method == "lr":
        probe = fit_lr(X[tr, layer], y[tr], X[va, layer], y[va], layer=layer, pooling=pooling)
    else:
        probe = fit_dom(X[tr, layer], y[tr], layer, pooling)
    probe.threshold = threshold_at_fpr(probe.score(X[va, layer])[y[va] == 0], 0.01)
    probe.meta.update(model=a.model, trained_on="plain train prompts only", threshold_rule="1% FPR on plain val benign")
    probe.save(rd / "probe")
    # Best diff-of-means probe at the same layer/pooling (a training-free baseline)
    dom = fit_dom(X[tr, layer], y[tr], layer, pooling)
    dom.threshold = threshold_at_fpr(dom.score(X[va, layer])[y[va] == 0], 0.01)
    dom.save(rd / "probe_dom")

    # ---- augmented variant: plain train + group-B-wrapped train prompts, same layer/pooling
    Xa, arows = load_acts(a.model, "wrapped_aug", pooling)
    ya = np.array([r["label"] for r in arows])
    spa = np.array([r["split"] for r in arows])
    Xa_l = np.asarray(Xa[:, layer])
    Xtr_aug = np.concatenate([X[tr, layer], Xa_l[spa == "train"]])
    ytr_aug = np.concatenate([y[tr], ya[spa == "train"]])
    Xva_aug = np.concatenate([X[va, layer], Xa_l[spa == "val"]])
    yva_aug = np.concatenate([y[va], ya[spa == "val"]])
    aug = fit_lr(Xtr_aug, ytr_aug, Xva_aug, yva_aug, layer=layer, pooling=pooling)
    aug.threshold = threshold_at_fpr(aug.score(X[va, layer])[y[va] == 0], 0.01)
    aug.meta.update(model=a.model, trained_on="plain train + group-B-wrapped train prompts",
                    threshold_rule="1% FPR on plain val benign")
    aug.save(rd / "probe_aug")

    # ---- TF-IDF text baseline (same splits), C tuned on val, threshold at 1% FPR on val benign
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    _, rows = load_acts(a.model, "plain", "last")
    txt = np.array([r["text"] for r in rows], dtype=object)
    best = None
    for C in (0.1, 1, 10, 100):
        pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
                             LogisticRegression(C=C, max_iter=5000))
        pipe.fit(list(txt[tr]), y[tr])
        auc = roc_auc_score(y[va], pipe.decision_function(list(txt[va])))
        if best is None or auc > best[0]:
            best = (auc, C, pipe)
    auc, C, pipe = best
    tf_thr = threshold_at_fpr(pipe.decision_function(list(txt[va]))[y[va] == 0], 0.01)
    with open(rd / "tfidf.pkl", "wb") as f:
        pickle.dump({"pipe": pipe, "threshold": tf_thr, "C": C, "val_auroc": auc}, f)

    sel = {"model": a.model, "layer": layer, "pooling": pooling, "method": method,
           "n_layers": int(X.shape[1] - 1), "val_auroc": sweep[f"{pooling}/{layer}"][method]["val_auroc"],
           "threshold": probe.threshold, "C": probe.meta.get("C"),
           "rule": "max plain-val AUROC (4 dp), tie-break val TPR@1%FPR then earlier layer; layer 0 excluded",
           "top5": [dict(layer=-c[2], pooling=c[3], method=c[4], val_auroc=c[0], val_tpr_at_1fpr=c[1]) for c in cands[:5]],
           "aug": {"C": aug.meta["C"], "threshold": aug.threshold},
           "tfidf": {"C": C, "val_auroc": auc, "threshold": tf_thr},
           "n_train": int(tr.sum()), "n_val": int(va.sum()), "seconds": round(time.time() - t0, 1)}
    write_json(rd / "selection.json", sel)
    print(json.dumps(sel, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--jobs", type=int, default=16)
    main(ap.parse_args())
