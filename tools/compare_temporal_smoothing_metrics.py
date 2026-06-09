#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


def read_key_value_summary(path: Path):
    df = pd.read_csv(path, header=None, names=["metric", "value"])

    row = {}

    for metric, value in zip(df["metric"], df["value"]):
        try:
            row[metric] = float(value)
        except ValueError:
            row[metric] = str(value)

    return row


def load_gonett_summaries(root: Path):
    paths = sorted(root.glob("*_summary.csv"))

    rows = []

    for path in paths:
        row = read_key_value_summary(path)

        name = path.name.replace("_full_comparison_summary.csv", "")

        row["method"] = "GONet+T"
        row["variant"] = "logit_lpred05_lsmooth05"
        row["split_side"] = name
        row["source_path"] = str(path)

        # Normalize column names to match EMA.
        row["smoothed_go_frames"] = row.get("gonet_t_go_frames")
        row["smoothed_jitter_mean_abs_delta"] = row.get("gonet_t_jitter_mean_abs_delta")

        rows.append(row)

    return rows


def load_ema_summaries(root: Path, method_name: str):
    paths = sorted(root.glob("alpha*/**/*_summary.csv"))

    rows = []

    for path in paths:
        row = read_key_value_summary(path)

        split_side = path.name.replace("_ema_summary.csv", "")

        row["method"] = method_name
        row["variant"] = path.parent.name
        row["split_side"] = split_side
        row["source_path"] = str(path)

        # Normalize column names to match GONet+T.
        row["mean_gonet_t_prob"] = row.get("mean_ema_prob")
        row["smoothed_go_frames"] = row.get("ema_go_frames")
        row["smoothed_jitter_mean_abs_delta"] = row.get("ema_jitter_mean_abs_delta")

        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gonett-root",
        default="outputs/gonet_t_full_inference",
        help="Folder containing GONet+T full inference summary CSVs.",
    )

    parser.add_argument(
        "--ema-prob-root",
        default="outputs/gonet_ema",
        help="Folder containing probability-space EMA summary CSVs.",
    )

    parser.add_argument(
        "--ema-logit-root",
        default="outputs/gonet_ema_logit",
        help="Folder containing logit-space EMA summary CSVs.",
    )

    parser.add_argument(
        "--output",
        default="outputs/temporal_smoothing_metrics_summary.csv",
    )

    args = parser.parse_args()

    rows = []

    rows.extend(load_gonett_summaries(Path(args.gonett_root)))
    rows.extend(load_ema_summaries(Path(args.ema_prob_root), "EMA-prob"))
    rows.extend(load_ema_summaries(Path(args.ema_logit_root), "EMA-logit"))

    if len(rows) == 0:
        raise RuntimeError("No summary files found.")

    df = pd.DataFrame(rows)

    # Keep only important columns if present.
    keep_cols = [
        "method",
        "variant",
        "split_side",
        "num_frames",
        "num_segments",
        "mean_vanilla_prob",
        "mean_gonet_t_prob",
        "mean_abs_difference",
        "vanilla_go_frames",
        "smoothed_go_frames",
        "decision_flip_count",
        "decision_flip_rate",
        "vanilla_jitter_mean_abs_delta",
        "smoothed_jitter_mean_abs_delta",
        "jitter_reduction_ratio",
        "source_path",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print()
    print(f"Saved full comparison table: {output_path}")

    print()
    print("Per split/side comparison:")
    display_cols = [
        "method",
        "variant",
        "split_side",
        "mean_abs_difference",
        "decision_flip_rate",
        "jitter_reduction_ratio",
        "vanilla_go_frames",
        "smoothed_go_frames",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    print(df[display_cols].to_string(index=False))

    print()
    print("Mean by method/variant:")
    grouped = (
        df.groupby(["method", "variant"])
        .agg(
            mean_abs_difference=("mean_abs_difference", "mean"),
            decision_flip_rate=("decision_flip_rate", "mean"),
            jitter_reduction_ratio=("jitter_reduction_ratio", "mean"),
        )
        .reset_index()
        .sort_values(["method", "variant"])
    )

    print(grouped.to_string(index=False))


if __name__ == "__main__":
    main()