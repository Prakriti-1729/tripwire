"""Figures 1-5 -> results/figures/ (primary model) or results/<slug>/figures/ (others)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tripwire.common import DATA, DEFAULT_MODEL, RESULTS, read_json, read_jsonl, results_dir  # noqa: E402
from tripwire.templates import FAMILIES, group_of  # noqa: E402

# Validated categorical palette (fixed order), text inks, recessive grid
C = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
NAMES = {"tripwire": "Tripwire probe", "tfidf": "TF-IDF text baseline", "guard": "Guard model",
         "tripwire_aug": "Tripwire (+B-aug)", "diff_of_means": "Diff-of-means"}
COLOR = {"tripwire": C[0], "tfidf": C[1], "guard": C[2], "tripwire_aug": C[3], "diff_of_means": C[4]}

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 150, "savefig.bbox": "tight",
})


def fam_label(f):
    return f"{f}\n(group {group_of(f)})"


def bars(ax, groups, series, ylabel, ylim=(0, 1.05)):
    """series: list of (name, color, values). Thin bars with a 2px surface gap."""
    n = len(series)
    x = np.arange(len(groups))
    w = 0.8 / n
    for i, (name, col, vals) in enumerate(series):
        ax.bar(x - 0.4 + w * (i + 0.5), vals, w, color=col, label=name, edgecolor="white", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace("refusal_suppression", "refusal_\nsuppression") for g in groups], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)


def main(a):
    rd = results_dir(a.model)
    out = RESULTS / "figures" if a.primary else rd / "figures"
    out.mkdir(parents=True, exist_ok=True)
    M = read_json(rd / "metrics.json")
    sweep = read_json(rd / "layer_sweep.json")
    sel = read_json(rd / "selection.json")
    methods = [m for m in ("tripwire", "tfidf", "guard", "tripwire_aug") if m in M["methods"]]
    model_short = a.model.split("/")[-1]

    # ---- 1. AUROC vs layer
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, pl in zip(axes, ("last", "mean")):
        L = sorted(int(k.split("/")[1]) for k in sweep if k.startswith(pl + "/"))
        get = lambda m, key: [sweep[f"{pl}/{l}"][m][key] for l in L]  # noqa: E731
        ax.plot(L, get("lr", "test_auroc"), color=C[0], lw=2, label="LR, plain test")
        ax.plot(L, get("dom", "test_auroc"), color=C[1], lw=2, label="Diff-of-means, plain test")
        ax.plot(L, get("lr", "wrapped_test_auroc"), color=C[2], lw=2, label="LR, wrapped test (unseen templates)")
        ax.plot(L, get("dom", "wrapped_test_auroc"), color=C[2], lw=2, ls="--", label="Diff-of-means, wrapped test")
        if "test_auroc_jbb_matched" in sweep[f"{pl}/{L[0]}"]["lr"]:
            ax.plot(L, get("lr", "test_auroc_jbb_matched"), color=C[4], lw=2,
                    label="LR, topic-matched JBB test (harmful vs benign)")
        if pl == sel["pooling"]:
            ax.axvline(sel["layer"], color=INK2, lw=1, ls=":")
            ax.annotate(f"chosen: layer {sel['layer']}", (sel["layer"], 0.52), xytext=(4, 0),
                        textcoords="offset points", color=INK2, fontsize=8)
        ax.set_title(f"{pl}-token pooling" if pl == "last" else "mean over user tokens", fontsize=10)
        ax.set_xlabel("layer (0 = embeddings)")
        ax.set_ylim(0.5, 1.01)
    axes[0].set_ylabel("AUROC")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"RQ1/RQ2 - where does 'harmful request' become linearly decodable? ({model_short})", x=0.01,
                 ha="left", fontsize=11)
    fig.savefig(out / "fig1_auroc_vs_layer.png")
    plt.close(fig)

    # ---- 2. Per-family TPR / FPR
    fams = ["none"] + FAMILIES
    labels = ["plain\n(no template)"] + [fam_label(f) for f in FAMILIES]
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True)
    for ax, key, ylab in ((axes[0], "tpr", "TPR on harmful"), (axes[1], "fpr", "FPR on benign")):
        series = []
        for m in methods:
            d = M["methods"][m]
            vals = [d["plain_test"][key]] + [d["per_family"][f][key] for f in FAMILIES]
            series.append((NAMES[m], COLOR[m], vals))
        bars(ax, labels, series, ylab)
    axes[0].set_title("harmful prompts: higher is better", loc="left", fontsize=10, color=INK2)
    axes[1].set_title("same templates wrapped around BENIGN prompts (the control): lower is better", loc="left",
                      fontsize=10, color=INK2)
    fig.suptitle("RQ2 - detection per jailbreak family (Tripwire/TF-IDF threshold fixed at 1% FPR on plain val; "
                 "guard: its own Yes/No)", x=0.01, y=1.0, ha="left", fontsize=11)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=len(methods), loc="upper left", bbox_to_anchor=(0.01, 0.975), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig2_per_family_tpr_fpr.png")
    plt.close(fig)

    # ---- 3. Model compliance vs probe catch (two definitions of "complied")
    if "rq3" in M:
        fams3 = ["none"] + FAMILIES
        lab3 = ["plain"] + [fam_label(f) for f in FAMILIES]
        panels = [("rq3", "complied = keyword refusal detector found no refusal (spec definition; over-counts)")]
        if "rq3_judge" in M:
            panels.append(("rq3_judge", "complied = Granite Guardian judged the model's response harmful (stricter)"))
        fig, axes = plt.subplots(len(panels), 1, figsize=(13, 3.6 * len(panels)), sharex=True, squeeze=False)
        for ax, (key, title) in zip(axes[:, 0], panels):
            R = M[key]
            series = [("Model complied (jailbreak success rate)", C[7], [R[f]["model_compliance_rate"] for f in fams3]),
                      ("Tripwire catch rate on complied prompts", C[0],
                       [R[f]["catch"]["tripwire"]["rate"] or 0 for f in fams3])]
            if "guard" in M["methods"]:
                series.append(("Guard catch rate on complied prompts", C[2],
                               [R[f]["catch"]["guard"]["rate"] or 0 for f in fams3]))
            bars(ax, lab3, series, "rate", ylim=(0, 1.12))
            for i, f in enumerate(fams3):
                ax.annotate(f"n={R[f]['n_complied']}", (i, 1.04), ha="center", fontsize=7, color=INK2)
            ax.set_title(title, loc="left", fontsize=9, color=INK2)
        fig.suptitle("RQ3 - of the harmful test prompts the model went along with, how many does the probe flag? "
                     "(n = complied prompts; per family)", x=0.01, ha="left", fontsize=11)
        axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.1), ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "fig3_compliance_vs_catch.png")
        plt.close(fig)

    # ---- 4. Score histograms
    S = read_json(rd / "scores.json")
    s, thr = S["scores"]["tripwire"], S["thresholds"]["tripwire"]
    plain = {r["id"]: r for r in read_jsonl(DATA / "plain.jsonl") if r["split"] == "test"}
    wrapped = {r["id"]: r for r in read_jsonl(DATA / "wrapped_test.jsonl")}
    xs = {r["id"]: r for r in read_jsonl(DATA / "xstest.jsonl")}
    groups = [("benign, plain", C[0], [s[i] for i, r in plain.items() if r["label"] == 0]),
              ("benign, wrapped", C[1], [s[i] for i, r in wrapped.items() if r["label"] == 0]),
              ("harmful, plain", C[2], [s[i] for i, r in plain.items() if r["label"] == 1]),
              ("harmful, wrapped", C[3], [s[i] for i, r in wrapped.items() if r["label"] == 1]),
              ("XSTest safe", C[4], [s[i] for i, r in xs.items() if r["label"] == 0])]
    allv = np.concatenate([g[2] for g in groups])
    lo, hi = np.percentile(allv, 0.5), np.percentile(allv, 99.5)
    bins = np.linspace(lo, hi, 60)
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, col, v in groups:
        ax.hist(np.clip(v, lo, hi), bins=bins, density=True, histtype="step", lw=2, color=col, label=f"{name} (n={len(v)})")
    ax.axvline(thr, color=INK, lw=1, ls="--")
    ax.annotate("threshold (1% FPR on val)", (thr, ax.get_ylim()[1] * 0.92), xytext=(4, 0), textcoords="offset points",
                fontsize=8, color=INK)
    ax.set_xlabel(f"Tripwire score (layer {sel['layer']}, {sel['pooling']} pooling, {sel['method']}); "
                  f"tails clipped to the 0.5-99.5th percentile")
    ax.set_ylabel("density")
    ax.set_title("Score distributions", loc="left", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.savefig(out / "fig4_score_histograms.png")
    plt.close(fig)

    # ---- 5. Latency
    lp = rd / "latency.json"
    if lp.exists():
        Lt = read_json(lp)
        lens = list(Lt["lengths"])
        conds = [("a_prefill", "model prefill", C[0]), ("b_prefill_plus_probe", "prefill + Tripwire probe", C[1]),
                 ("d_early_abort_at_L", f"Tripwire early abort (stop after layer {Lt['layer']})", C[2]),
                 ("c_guard_forward", f"guard forward ({(Lt['guard'] or '').split('/')[-1]}, MoE)", C[3]),
                 ("c2_guard2_forward", f"guard forward ({(Lt.get('guard2') or '').split('/')[-1]}, dense)", C[4])]
        conds = [c for c in conds if c[0] in Lt["lengths"][lens[0]]]
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(lens))
        w = 0.8 / len(conds)
        for i, (k, name, col) in enumerate(conds):
            med = [Lt["lengths"][l][k]["median_ms"] for l in lens]
            p90 = [Lt["lengths"][l][k]["p90_ms"] for l in lens]
            xi = x - 0.4 + w * (i + 0.5)
            ax.bar(xi, med, w, color=col, label=name, edgecolor="white", linewidth=1.0)
            ax.errorbar(xi, med, yerr=[np.zeros(len(med)), np.array(p90) - np.array(med)], fmt="none",
                        ecolor=INK2, elinewidth=1, capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"~{l} tokens\n(actual {Lt['lengths'][l]['prompt_tokens']})" for l in lens])
        ax.set_ylabel("ms (bar = median, whisker = p90)")
        gs = Lt.get("gpu_status_before", {})
        ax.set_title(f"RQ5 - latency at batch size 1 on a shared GPU (other load at start: {gs.get('util_pct', '?')}% util)",
                     loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
        ax.legend(fontsize=8)
        fig.savefig(out / "fig5_latency.png")
        plt.close(fig)
    print("figures ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--primary", action="store_true", help="write to results/figures/")
    main(ap.parse_args())
