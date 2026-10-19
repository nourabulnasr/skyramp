"""Render README figures from the committed eval_results.json (latest run) -- no hand-typed numbers."""
import json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import ROOT

def main():
    hist = json.loads((ROOT / "eval_results.json").read_text())
    r = hist[0]   # first_run: models + baseline conformal variants (ground truth)
    M = r["models"]
    names = ["persistence", "smart_persistence", "lgbm_pv", "nn_pv", "lgbm_pv_img", "cnn_img_pv", "cnn_img_only"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.arange(len(names)); w = 0.38
    ax[0].bar(x - w / 2, [M[n]["rmse_nonramp"] for n in names], w, label="non-ramp minutes", color="#9aa5b1")
    ax[0].bar(x + w / 2, [M[n]["rmse_ramp"] for n in names], w, label=f"ramp minutes (n={r['n_test_ramp']})", color="#d9480f")
    ax[0].set_xticks(x); ax[0].set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7.5)
    ax[0].set_ylabel("15-min-ahead RMSE (kW), test blocks"); ax[0].legend(frameon=False)
    ax[0].set_title("Forecast error by regime"); ax[0].spines[["top", "right"]].set_visible(False)
    cb = dict(r["claim_B"]["variants"])
    for h in hist[1:]:                       # fix attempts appended later
        if "claim_B" in h:
            cb["FIX: " + h["claim_B"]["variant"]] = h["claim_B"]
    labs = list(cb); xs = np.arange(len(labs))
    ax[1].bar(xs - w, [cb[k]["coverage_all"] for k in labs], w, label="all", color="#495057")
    ax[1].bar(xs, [cb[k]["coverage_nonramp"] for k in labs], w, label="non-ramp", color="#9aa5b1")
    ax[1].bar(xs + w, [cb[k]["coverage_ramp"] for k in labs], w, label="ramp", color="#d9480f")
    ax[1].axhline(0.9, color="k", ls="--", lw=1); ax[1].text(len(labs) - 0.5, 0.905, "target 0.90", ha="right", fontsize=8)
    ax[1].set_xticks(xs); ax[1].set_xticklabels([k.replace("/", "\n") for k in labs], fontsize=6)
    ax[1].set_ylim(0.0, 1.05); ax[1].set_ylabel("empirical coverage of 90% interval")
    ax[1].set_title("Split-conformal coverage by regime"); ax[1].legend(frameon=False, loc="upper left", ncol=3)
    ax[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = ROOT / "results"; out.mkdir(exist_ok=True)
    fig.savefig(out / "results.png", dpi=140)
    print("wrote", out / "results.png")

if __name__ == "__main__":
    main()
