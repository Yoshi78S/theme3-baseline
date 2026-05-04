"""Plot inference time vs recommendation performance from benchmark_results.csv.

Produces one PNG per (dataset, metric) pair, e.g.:
  - inference_vs_perf_LastFM_HR10.png
  - inference_vs_perf_Beauty_NDCG10.png
  ...
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


MODEL_ORDER = [
    "GRU4Rec", "Caser", "SASRec", "BERT4Rec",
    "FMLPRec", "DuoRec", "FEARec", "LRURec",
    "Mamba4Rec", "ICSRec", "ICLRec", "SIGMA",
    "BSARec",
]
MODEL_COLORS = {
    "GRU4Rec": "#1f77b4", "Caser": "#ff7f0e", "SASRec": "#2ca02c",
    "BERT4Rec": "#d62728", "FMLPRec": "#9467bd", "DuoRec": "#8c564b",
    "FEARec": "#e377c2", "LRURec": "#7f7f7f", "Mamba4Rec": "#bcbd22",
    "ICSRec": "#aec7e8", "ICLRec": "#ffbb78", "SIGMA": "#c49c94",
    "BSARec": "#17becf",
}
MODEL_MARKERS = {
    "GRU4Rec": "o", "Caser": "s", "SASRec": "^", "BERT4Rec": "D",
    "FMLPRec": "v", "DuoRec": "P", "FEARec": "X", "LRURec": "<",
    "Mamba4Rec": ">", "ICSRec": "p", "ICLRec": "h", "SIGMA": "d",
    "BSARec": "*",
}
DATASET_ORDER = ["LastFM", "Beauty", "ML-1M"]


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def plot_one(rows, dataset, metric, out_path,
             xlabel="Inference time on test set (s)"):
    pts = []
    for r in rows:
        if r["dataset"] != dataset:
            continue
        try:
            t = float(r["test_sec"])
            y = float(r[metric])
        except (ValueError, TypeError, KeyError):
            continue
        pts.append((r["model"], t, y))
    if not pts:
        print(f"[plot] skip {dataset} {metric} (no rows)")
        return

    pts.sort(key=lambda x: MODEL_ORDER.index(x[0]) if x[0] in MODEL_ORDER else 999)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for model, t, y in pts:
        ax.scatter(
            t, y,
            s=220 if model == "BSARec" else 140,
            color=MODEL_COLORS.get(model, "gray"),
            marker=MODEL_MARKERS.get(model, "o"),
            edgecolor="black", linewidth=0.8,
            label=model, zorder=3,
        )
        ax.annotate(
            model, (t, y),
            xytext=(7, 5), textcoords="offset points",
            fontsize=10, alpha=0.9,
        )
    ax.set_title(f"{dataset}: {metric} vs inference time")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ts = [t for _, t, _ in pts]
    if ts and max(ts) / max(min(ts), 1e-9) > 15:
        ax.set_xscale("log")
        ax.set_xlabel(xlabel + " (log scale)")

    # legend
    handles, labels = [], []
    for m in MODEL_ORDER:
        if m in MODEL_COLORS:
            handles.append(plt.Line2D(
                [0], [0], marker=MODEL_MARKERS[m], color="w",
                markerfacecolor=MODEL_COLORS[m], markeredgecolor="black",
                markersize=10 if m != "BSARec" else 14, label=m,
            ))
            labels.append(m)
    ax.legend(handles, labels, loc="best", frameon=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved {out_path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="benchmark_results.csv")
    p.add_argument("--outdir", default=".")
    args = p.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    rows = load(args.csv)

    datasets = [d for d in DATASET_ORDER if any(r["dataset"] == d for r in rows)]
    for ds in datasets:
        for metric in ["HR@10", "NDCG@10", "HR@20", "NDCG@20"]:
            fname = f"inference_vs_perf_{ds}_{metric.replace('@','')}.png"
            out = os.path.join(args.outdir, fname)
            plot_one(rows, ds, metric, out)


if __name__ == "__main__":
    main()
