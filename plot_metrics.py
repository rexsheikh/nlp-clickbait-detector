#!/usr/bin/env python3
import os
import json
from collections import defaultdict
from datetime import datetime
import argparse

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description="Plot consolidated metrics from metrics.jsonl")
    ap.add_argument("--metrics", default="metrics.jsonl", help="Path to JSONL metrics file")
    ap.add_argument("--outdir", default="plots", help="Directory to write plots")
    ap.add_argument("--dpi", type=int, default=150, help="Figure save DPI")
    return ap.parse_args()


def load_records(path):
    recs = []
    if not os.path.exists(path):
        print(f"Metrics file not found: {path}")
        return recs
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                # normalize run_id to sortable timestamp if present
                rid = rec.get("run_id")
                try:
                    rec["_run_dt"] = datetime.fromisoformat(rid.replace("Z", "+00:00")) if isinstance(rid, str) else datetime.min
                except Exception:
                    rec["_run_dt"] = datetime.min
                recs.append(rec)
            except json.JSONDecodeError:
                continue
    return recs


def latest_by_key(records, key_fields, predicate=lambda r: True):
    """
    Pick the latest (by run_id timestamp) record for each unique key tuple.
    """
    best = {}
    for r in records:
        if not predicate(r):
            continue
        key = tuple(r.get(k) for k in key_fields)
        prev = best.get(key)
        if prev is None or r.get("_run_dt", datetime.min) >= prev.get("_run_dt", datetime.min):
            best[key] = r
    return list(best.values())


def plot_grouped_bars(dataset, rows, outdir, dpi=150):
    """
    rows: list of dicts with keys: model, accuracy, precision_pos1, recall_pos1, f1_pos1
    """
    if not rows:
        return
    # Order models for consistent plotting
    order = ["NLTK_NB", "MNB", "LogReg"]
    rows_sorted = sorted(rows, key=lambda r: order.index(r["model"]) if r["model"] in order else 999)
    labels = [r["model"] for r in rows_sorted]
    metrics = ["accuracy", "precision_pos1", "recall_pos1", "f1_pos1"]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]

    x = np.arange(len(labels))
    width = 0.15
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, m in enumerate(metrics):
        vals = [r.get("metrics", {}).get(m) for r in rows_sorted]
        vals = [v if isinstance(v, (int, float)) and v is not None else 0.0 for v in vals]
        ax.bar(x + (i - 1.5) * width, vals, width, label=m, color=colors[i])

    ax.set_title(f"{dataset} - Holdout Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.legend(loc="lower right", ncols=2)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(outdir, f"bars_{dataset.lower()}.png")
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"Saved: {out_path}")


def plot_cv_best_bars(dataset, rows, outdir, dpi=150):
    """
    rows: list of dicts with keys: model, metrics.f1_pos1 (CV best)
    """
    if not rows:
        return
    order = ["NLTK_NB", "MNB", "LogReg"]
    rows_sorted = sorted(rows, key=lambda r: order.index(r["model"]) if r["model"] in order else 999)
    labels = [r["model"] for r in rows_sorted]
    vals = [r.get("metrics", {}).get("f1_pos1") or 0.0 for r in rows_sorted]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(x, vals, color="#4c78a8", width=0.5)
    ax.set_title(f"{dataset} - CV Best F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1 (pos=1)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(outdir, f"cvbest_{dataset.lower()}.png")
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"Saved: {out_path}")


def plot_confusion_heatmaps(dataset, rows, outdir, dpi=150):
    """
    rows: list of dicts with keys: model, confusion (2x2)
    Output one image with subplots for each model that has confusion.
    """
    rows = [r for r in rows if isinstance(r.get("confusion"), list)]
    if not rows:
        return
    order = ["NLTK_NB", "MNB", "LogReg"]
    rows_sorted = sorted(rows, key=lambda r: order.index(r["model"]) if r["model"] in order else 999)
    n = len(rows_sorted)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, r in enumerate(rows_sorted):
        ax = axes[i]
        mat = np.array(r["confusion"])
        im = ax.imshow(mat, cmap="Blues")
        for (ii, jj), val in np.ndenumerate(mat):
            ax.text(jj, ii, int(val), ha="center", va="center", color="black")
        ax.set_title(f"{dataset} - {r['model']}")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred 0", "pred 1"]); ax.set_yticklabels(["true 0", "true 1"])
    # hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    out_path = os.path.join(outdir, f"cm_{dataset.lower()}.png")
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    records = load_records(args.metrics)
    if not records:
        print("No records to plot.")
        return

    # Group by dataset
    datasets = sorted({r.get("dataset") for r in records if r.get("dataset")})
    for ds in datasets:
        # Latest holdout per (dataset, model, variant) -> then reduce to latest per (dataset, model)
        holdout_latest = latest_by_key(
            records,
            key_fields=["dataset", "model", "variant"],
            predicate=lambda r: r.get("dataset") == ds and r.get("split") == "holdout"
        )
        # pick latest per (dataset, model)
        holdout_latest = latest_by_key(
            holdout_latest,
            key_fields=["dataset", "model"],
            predicate=lambda r: True
        )
        # Plot grouped bars for holdout metrics
        plot_grouped_bars(
            dataset=ds,
            rows=[{"model": r["model"], "metrics": r.get("metrics", {})} for r in holdout_latest],
            outdir=args.outdir,
            dpi=args.dpi
        )

        # Plot confusion heatmaps for holdout (one subplot per model)
        plot_confusion_heatmaps(
            dataset=ds,
            rows=[{"model": r["model"], "confusion": r.get("confusion")} for r in holdout_latest],
            outdir=args.outdir,
            dpi=args.dpi
        )

        # CV-best per (dataset, model) with best_f1
        cvbest_latest = latest_by_key(
            records,
            key_fields=["dataset", "model"],
            predicate=lambda r: r.get("dataset") == ds and r.get("split") == "cv-best"
        )
        plot_cv_best_bars(
            dataset=ds,
            rows=[{"model": r["model"], "metrics": r.get("metrics", {})} for r in cvbest_latest],
            outdir=args.outdir,
            dpi=args.dpi
        )

    print("Done.")


if __name__ == "__main__":
    main()
