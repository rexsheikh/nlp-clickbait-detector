#!/usr/bin/env python3
"""Parse raw metric dumps and produce consolidated summaries/plots."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required to run this script. Install it via 'pip install matplotlib'."
    ) from exc

import numpy as np
import pandas as pd

BRACKET_RE = re.compile(r"\[([^\]]+)\]")
METRIC_RE = re.compile(
    r"Acc=(?P<acc>[0-9.]+)\s+Prec\(pos=1\)=(?P<prec>[0-9.]+)\s+Rec\(pos=1\)=(?P<rec>[0-9.]+)"
)
MACRO_RE = re.compile(
    r"Macro:\s+prec=(?P<macro_prec>[0-9.]+)\s+rec=(?P<macro_rec>[0-9.]+)\s+f1=(?P<macro_f1>[0-9.]+)"
)
ROC_RE = re.compile(r"ROC-AUC:\s+(?P<roc_auc>[0-9.]+)")
PR_RE = re.compile(r"PR-AUC:\s+(?P<pr_auc>[0-9.]+)")


def feature_bucket_from_header(line: str) -> str | None:
    """Infer feature bucket (baseline/applied) from a section header."""
    header = line.strip("= ").lower()
    if "baseline" in header:
        return "baseline"
    if "additional" in header or "feature" in header:
        return "applied"
    return None


def parse_metrics_from_file(path: Path) -> List[Dict[str, object]]:
    """Parse one *Raw.txt file and returns metric dicts."""
    feature_bucket = None
    records: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("==="):
            bucket = feature_bucket_from_header(line)
            if bucket:
                feature_bucket = bucket
            continue

        if feature_bucket is None:
            continue

        if "Acc=" not in line and "Macro:" not in line and "ROC-AUC" not in line and "PR-AUC" not in line:
            continue

        bracket_chunks = BRACKET_RE.findall(line)
        if len(bracket_chunks) < 2:
            continue

        dataset = bracket_chunks[0]
        model_label = bracket_chunks[1]
        config = " | ".join(bracket_chunks[2:]) if len(bracket_chunks) > 2 else ""
        key = (dataset, model_label, feature_bucket, config, path.name)

        record = records.setdefault(
            key,
            {
                "dataset": dataset,
                "model_family": model_label,
                "feature_set": feature_bucket,
                "config": config,
                "source_file": path.name,
            },
        )

        if "Acc=" in line:
            metric_match = METRIC_RE.search(line)
            if metric_match:
                record["accuracy"] = float(metric_match.group("acc"))
                record["precision"] = float(metric_match.group("prec"))
                record["recall"] = float(metric_match.group("rec"))
        elif "Macro:" in line:
            macro_match = MACRO_RE.search(line)
            if macro_match:
                record["macro_precision"] = float(macro_match.group("macro_prec"))
                record["macro_recall"] = float(macro_match.group("macro_rec"))
                record["macro_f1"] = float(macro_match.group("macro_f1"))
        elif "ROC-AUC" in line:
            roc_match = ROC_RE.search(line)
            if roc_match:
                record["roc_auc"] = float(roc_match.group("roc_auc"))
        elif "PR-AUC" in line:
            pr_match = PR_RE.search(line)
            if pr_match:
                record["pr_auc"] = float(pr_match.group("pr_auc"))

    return list(records.values())


def build_dataframe(raw_glob: str) -> pd.DataFrame:
    """Collect metrics from all matching files."""
    all_records: List[Dict[str, object]] = []
    for path in sorted(Path(".").glob(raw_glob)):
        if path.is_file():
            all_records.extend(parse_metrics_from_file(path))

    if not all_records:
        raise SystemExit(f"No metrics parsed via glob '{raw_glob}'.")

    df = pd.DataFrame(all_records)
    df.sort_values(
        by=["dataset", "model_family", "feature_set", "config"],
        inplace=True,
        ignore_index=True,
    )
    return df


def plot_metric_grid(df: pd.DataFrame, output_path: Path) -> None:
    """Create a dataset x metric grid comparing baseline/applied features."""
    metrics = ["accuracy", "precision", "recall", "macro_f1"]
    datasets = sorted(df["dataset"].unique())
    feature_order = ["baseline", "applied"]

    fig, axes = plt.subplots(
        len(datasets),
        len(metrics),
        figsize=(4 * len(metrics), 3 * len(datasets)),
        squeeze=False,
    )

    for row, dataset in enumerate(datasets):
        subset = df[df["dataset"] == dataset]
        models = sorted(subset["model_family"].unique())
        x = np.arange(len(models))
        width = 0.35

        for col, metric in enumerate(metrics):
            ax = axes[row][col]
            for idx, feature_set in enumerate(feature_order):
                values = []
                for model in models:
                    match = subset[
                        (subset["model_family"] == model) & (subset["feature_set"] == feature_set)
                    ]
                    if not match.empty and pd.notna(match.iloc[0].get(metric)):
                        values.append(match.iloc[0][metric])
                    else:
                        values.append(np.nan)
                ax.bar(
                    x + idx * width,
                    values,
                    width=width,
                    label=feature_set if row == 0 and col == 0 else None,
                    alpha=0.9 if feature_set == "applied" else 0.7,
                )

            ax.set_title(f"{dataset} – {metric}")
            ax.set_xticks(x + width / 2)
            ax.set_xticklabels(models, rotation=45, ha="right")
            ax.set_ylim(0, 1.05)
            if col == 0:
                ax.set_ylabel("score")

    fig.tight_layout()
    if axes[0][0].legend_ is None:
        axes[0][0].legend(loc="lower right")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_f1_scatter(df: pd.DataFrame, output_path: Path, metric: str = "macro_f1") -> None:
    """Scatter plot of dataset vs. comprehensive metric across models."""
    if metric not in df.columns:
        raise SystemExit(f"Column '{metric}' not available for scatter plot.")

    data = df.dropna(subset=[metric]).copy()
    if data.empty:
        raise SystemExit(f"No rows with '{metric}' populated for scatter plotting.")

    datasets = sorted(data["dataset"].unique())
    models = sorted(data["model_family"].unique())
    feature_sets = sorted(data["feature_set"].unique())

    dataset_positions = {ds: idx for idx, ds in enumerate(datasets)}
    if len(feature_sets) == 1:
        offsets = {feature_sets[0]: 0.0}
    else:
        spread = np.linspace(-0.2, 0.2, len(feature_sets))
        offsets = {fs: spread[idx] for idx, fs in enumerate(feature_sets)}

    color_map = plt.cm.get_cmap("tab10", max(len(models), 3))
    model_colors = {model: color_map(i % color_map.N) for i, model in enumerate(models)}
    marker_cycle = ["o", "s", "^", "D", "P"]
    marker_map = {fs: marker_cycle[i % len(marker_cycle)] for i, fs in enumerate(feature_sets)}

    fig, ax = plt.subplots(figsize=(2 + 2.5 * len(datasets), 5))

    plotted_models = set()
    for _, row in data.iterrows():
        dataset = row["dataset"]
        model = row["model_family"]
        feature = row["feature_set"]
        x = dataset_positions[dataset] + offsets.get(feature, 0.0)
        y = row[metric]
        label = model if model not in plotted_models else None
        ax.scatter(
            x,
            y,
            color=model_colors[model],
            marker=marker_map.get(feature, "o"),
            s=80,
            edgecolor="black",
            linewidth=0.5,
            label=label,
        )
        plotted_models.add(model)

    ax.set_xticks(list(dataset_positions.values()))
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"{metric.replace('_', ' ').upper()} by Dataset and Model")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    legend1 = ax.legend(title="Model", loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.add_artist(legend1)
    feature_handles = [
        Line2D(
            [],
            [],
            color="black",
            marker=marker_map.get(fs, "o"),
            linestyle="None",
            markersize=8,
            label=fs,
        )
        for fs in feature_sets
    ]
    ax.legend(handles=feature_handles, title="Feature set", loc="lower left", bbox_to_anchor=(1.02, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_metric_scatter_by_model(
    df: pd.DataFrame,
    metrics: Sequence[str],
    output_dir: Path,
) -> None:
    """Scatter plots per model for requested metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_order = sorted(df["dataset"].unique())
    if not dataset_order:
        raise SystemExit("No datasets found in dataframe for scatter plotting.")
    dataset_positions = {ds: idx for idx, ds in enumerate(dataset_order)}

    feature_sets = sorted([fs for fs in df["feature_set"].dropna().unique()])
    if not feature_sets:
        feature_sets = ["default"]
    if len(feature_sets) == 1:
        offsets = {feature_sets[0]: 0.0}
    else:
        spread = np.linspace(-0.2, 0.2, len(feature_sets))
        offsets = {fs: spread[idx] for idx, fs in enumerate(feature_sets)}

    color_map = plt.cm.get_cmap("Dark2", max(len(feature_sets), 1))
    feature_colors = {fs: color_map(idx % color_map.N) for idx, fs in enumerate(feature_sets)}
    markers = ["o", "s", "^", "D", "P"]
    marker_map = {fs: markers[idx % len(markers)] for idx, fs in enumerate(feature_sets)}

    for metric in metrics:
        if metric not in df.columns:
            print(f"[scatter] Skipping metric '{metric}' (not present in dataframe).")
            continue
        metric_df = df.dropna(subset=[metric])
        if metric_df.empty:
            print(f"[scatter] Skipping metric '{metric}' (no non-null values).")
            continue

        models = sorted(metric_df["model_family"].unique())
        for model in models:
            subset = metric_df[metric_df["model_family"] == model]
            if subset.empty:
                continue

            fig, ax = plt.subplots(figsize=(2 + 2.5 * len(dataset_order), 4.5))
            for dataset in dataset_order:
                x_base = dataset_positions[dataset]
                for feature in feature_sets:
                    rows = subset[(subset["dataset"] == dataset) & (subset["feature_set"] == feature)]
                    if rows.empty:
                        continue
                    value = rows.iloc[0][metric]
                    if pd.isna(value):
                        continue
                    ax.scatter(
                        x_base + offsets.get(feature, 0.0),
                        value,
                        color=feature_colors.get(feature, "C0"),
                        marker=marker_map.get(feature, "o"),
                        s=90,
                        edgecolor="black",
                        linewidth=0.5,
                    )

            ax.set_xticks(list(dataset_positions.values()))
            ax.set_xticklabels(dataset_order)
            ax.set_ylim(0, 1.05)
            pretty_metric = metric.replace("_", " ").title()
            ax.set_ylabel(pretty_metric)
            ax.set_title(f"{model} – {pretty_metric}")
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)

            handles = [
                Line2D(
                    [],
                    [],
                    marker=marker_map.get(fs, "o"),
                    color=feature_colors.get(fs, "black"),
                    linestyle="None",
                    markersize=8,
                    label=fs,
                )
                for fs in feature_sets
            ]
            ax.legend(handles=handles, title="Feature set", loc="lower left", bbox_to_anchor=(1.02, 0))

            fig.tight_layout()
            out_path = output_dir / f"{metric}_{model}.png"
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved scatter plot for {model} ({metric}) to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate metric summary plots.")
    parser.add_argument("--raw-glob", default="*Raw.txt", help="Glob for raw metric files.")
    parser.add_argument(
        "--output-csv",
        default="metrics_summary.csv",
        help="Path to write the consolidated CSV.",
    )
    parser.add_argument(
        "--figure-path",
        default="plots/classical_metrics.png",
        help="Path to save the matplotlib figure.",
    )
    parser.add_argument(
        "--scatter-path",
        default="plots/f1_scatter.png",
        help="Path to save the dataset/model scatter plot.",
    )
    parser.add_argument(
        "--scatter-dir",
        default="plots/scatter_by_model",
        help="Directory to store per-model scatter plots.",
    )
    parser.add_argument(
        "--scatter-metrics",
        default="macro_f1,accuracy,precision,recall",
        help="Comma-separated list of metrics for per-model scatter plots.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the matplotlib window after saving the figure.",
    )
    args = parser.parse_args()

    df = build_dataframe(args.raw_glob)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote consolidated metrics to {output_csv}")
    print(df[["dataset", "model_family", "feature_set", "accuracy", "precision", "recall"]])

    plot_metric_grid(df, Path(args.figure_path))
    print(f"Saved figure to {args.figure_path}")
    plot_f1_scatter(df, Path(args.scatter_path))
    print(f"Saved scatter plot to {args.scatter_path}")
    metrics = [m.strip() for m in args.scatter_metrics.split(",") if m.strip()]
    if metrics:
        plot_metric_scatter_by_model(df, metrics, Path(args.scatter_dir))

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
