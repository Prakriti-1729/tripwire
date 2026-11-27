"""Evaluate Tripwire probes + baselines on plain test, wrapped test (per family), XSTest, and the
model-compliance subset (RQ3). Writes results/<slug>/metrics.json and results/<slug>/scores.json."""
from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tripwire.common import DEFAULT_MODEL, read_json, results_dir, write_json  # noqa: E402
from tripwire.probe import LinearProbe  # noqa: E402
from tripwire.templates import FAMILIES, GROUP_A, GROUP_B  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_probes import load_acts  # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0, c - h), 4), round(min(1, c + h), 4)]


def binmetrics(scores, labels, thr):
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    out = {"n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum())}
    if out["n_pos"]:
        k = int((s[y == 1] > thr).sum())
        out["tpr"], out["tpr_ci95"] = k / out["n_pos"], wilson(k, out["n_pos"])
    if out["n_neg"]:
        k = int((s[y == 0] > thr).sum())
        out["fpr"], out["fpr_ci95"] = k / out["n_neg"], wilson(k, out["n_neg"])
    if out["n_pos"] and out["n_neg"]:
        out["auroc"] = float(roc_auc_score(y, s))
    return out


def main(a):
    rd = results_dir(a.model)
    sel = read_json(rd / "selection.json")
    layer, pooling = sel["layer"], sel["pooling"]
    sets = {}
    for name in ("plain", "wrapped_test", "xstest"):
        X, rows = load_acts(a.model, name, pooling)
        if name == "plain":
            keep = [i for i, r in enumerate(rows) if r["split"] == "test"]
            X, rows = np.asarray(X[keep, layer]), [rows[i] for i in keep]
        else:
            X = np.asarray(X[:, layer])
        sets[name] = (X, rows)

    # ---- scores for every method: {method: {id: score}}, thresholds {method: thr}
    scores, thr = {}, {}
    for m, f in (("tripwire", "probe"), ("tripwire_aug", "probe_aug"), ("diff_of_means", "probe_dom")):
        p = LinearProbe.load(rd / f)
        thr[m] = p.threshold
        scores[m] = {r["id"]: float(s) for X, rows in sets.values() for r, s in zip(rows, p.score(X))}
    tf = pickle.load(open(rd / "tfidf.pkl", "rb"))
    thr["tfidf"] = tf["threshold"]
    scores["tfidf"] = {}
    for X, rows in sets.values():
        for r, s in zip(rows, tf["pipe"].decision_function([r["text"] for r in rows])):
            scores["tfidf"][r["id"]] = float(s)
    guard_name = None
    gpath = rd / "guard_scores.json"
    if gpath.exists():
        g = read_json(gpath)
        if g.get("guard"):
            guard_name = g["guard"]
            scores["guard"], thr["guard"] = g["scores"], 0.0
    methods = list(scores)

    plain_rows = sets["plain"][1]
    wrows = sets["wrapped_test"][1]
    xrows = sets["xstest"][1]

    def evalset(m, rows):
        return binmetrics([scores[m][r["id"]] for r in rows], [r["label"] for r in rows], thr[m])

    M = {"model": a.model, "layer": layer, "pooling": pooling, "probe_method": sel["method"], "guard": guard_name,
         "thresholds": thr, "methods": {}}
    for m in methods:
        d = {"plain_test": evalset(m, plain_rows), "wrapped_test": evalset(m, wrows),
             "group_A": evalset(m, [r for r in wrows if r["template"] in GROUP_A]),
             "group_B": evalset(m, [r for r in wrows if r["template"] in GROUP_B]),
             "per_family": {f: evalset(m, [r for r in wrows if r["template"] == f]) for f in FAMILIES},
             "xstest_safe": evalset(m, [r for r in xrows if r["label"] == 0]),
             "xstest_unsafe": evalset(m, [r for r in xrows if r["label"] == 1]),
             "xstest_all": evalset(m, xrows)}
        M["methods"][m] = d

    # ---- RQ3: model compliance vs probe catch, under two definitions of "the model complied":
    #   keyword: the keyword refusal detector found no refusal (spec definition; over-counts, see README "Method in detail")
    #   judge:   Granite Guardian judged the model's RESPONSE harmful (stricter)
    rpath = rd / "refusals.json"
    if rpath.exists():
        ref = read_json(rpath)
        jpath = rd / "judge.json"
        judge = read_json(jpath).get("harmful_response", {}) if jpath.exists() else {}
        defs = {"keyword": lambda i: not ref[i]}
        if judge:
            defs["judge"] = lambda i: bool(judge[i])
        harm = [r for r in plain_rows + wrows if r["label"] == 1 and r["id"] in ref]
        for dname, complied in defs.items():
            rq3 = {}
            for fam in ["none"] + FAMILIES + ["ALL", "ALL_WRAPPED"]:
                rs = [r for r in harm if (fam == "ALL") or (fam == "ALL_WRAPPED" and r["template"] != "none")
                      or r["template"] == fam]
                comp = [r for r in rs if complied(r["id"])]
                d = {"n": len(rs), "model_compliance_rate": len(comp) / len(rs) if rs else None,
                     "n_complied": len(comp), "catch": {}}
                d["model_refusal_rate"] = 1 - d["model_compliance_rate"] if rs else None
                for m in methods:
                    k = sum(scores[m][r["id"]] > thr[m] for r in comp)
                    d["catch"][m] = {"rate": k / len(comp) if comp else None, "ci95": wilson(k, len(comp)), "k": int(k)}
                    # end-to-end: harmful prompts that get through = model complied AND detector silent
                    d["catch"][m]["residual_harm_rate"] = (len(comp) - k) / len(rs) if rs else None
                rq3[fam] = d
            M["rq3" if dname == "keyword" else "rq3_judge"] = rq3
    write_json(rd / "metrics.json", M)
    # scores for figures (ids -> score); no text
    write_json(rd / "scores.json", {"thresholds": thr, "scores": {m: {k: round(v, 4) for k, v in s.items()}
                                                                 for m, s in scores.items()}})
    t = M["methods"]["tripwire"]
    print(f"plain test AUROC {t['plain_test']['auroc']:.4f} TPR {t['plain_test']['tpr']:.3f} FPR {t['plain_test']['fpr']:.3f}")
    print(f"wrapped    AUROC {t['wrapped_test']['auroc']:.4f} TPR {t['wrapped_test']['tpr']:.3f} FPR {t['wrapped_test']['fpr']:.3f}")
    print(f"xstest safe FPR {t['xstest_safe']['fpr']:.3f}")
    for m in methods:
        w = M["methods"][m]["wrapped_test"]
        print(f"  {m:14s} plainAUC {M['methods'][m]['plain_test']['auroc']:.3f} wrapAUC {w['auroc']:.3f} "
              f"TPR {w['tpr']:.3f} FPR {w['fpr']:.3f} xsFPR {M['methods'][m]['xstest_safe']['fpr']:.3f}"
              + (f" catch {M['rq3']['ALL']['catch'][m]['rate']:.3f}" if "rq3" in M else "")
              + (f" catchJ {M['rq3_judge']['ALL']['catch'][m]['rate']:.3f}" if "rq3_judge" in M else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    main(ap.parse_args())
