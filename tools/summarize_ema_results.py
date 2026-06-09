#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import argparse

def read_summary(path):
    df = pd.read_csv(path, header=None, names=["metric", "value"])

    row = {}

    for metric, value in zip(df["metric"], df["value"]):
        try:
            row[metric] = float(value)
        except ValueError:
            row[metric] = str(value)

    row["summary_path"] = str(path)

    # Infer alpha directory and filename from path.
    row["alpha_dir"] = path.parent.name
    row["file"] = path.name

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs/gonet_ema")
    args = parser.parse_args()

    summary_paths = sorted(Path(args.root).glob("alpha*/**/*_summary.csv"))

    if len(summary_paths) == 0:
        raise RuntimeError("No EMA summary files found under outputs/gonet_ema")

    rows = [read_summary(p) for p in summary_paths]
    df = pd.DataFrame(rows)

    output_path = Path(args.root) / "ema_summary_all.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print()

    cols = [
        "alpha_dir",
        "file",
        "num_frames",
        "mean_abs_difference",
        "decision_flip_rate",
        "vanilla_jitter_mean_abs_delta",
        "ema_jitter_mean_abs_delta",
        "jitter_reduction_ratio",
        "vanilla_go_frames",
        "ema_go_frames",
    ]

    print(df[cols].to_string(index=False))

    print()
    print("Mean by alpha:")
    grouped = df.groupby("alpha_dir").agg({
        "mean_abs_difference": "mean",
        "decision_flip_rate": "mean",
        "jitter_reduction_ratio": "mean",
    }).reset_index()

    print(grouped.to_string(index=False))


if __name__ == "__main__":
    main()