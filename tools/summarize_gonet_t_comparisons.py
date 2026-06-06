#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


def summarize_one_csv(path: Path, threshold: float):
    df = pd.read_csv(path)

    required = [
        "segment_id",
        "frame_idx",
        "vanilla_prob",
        "gonet_t_prob",
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in {path}")

    df = df.sort_values("frame_idx").reset_index(drop=True)

    vanilla = df["vanilla_prob"].astype(float)
    gonet_t = df["gonet_t_prob"].astype(float)

    vanilla_go = vanilla >= threshold
    gonet_t_go = gonet_t >= threshold

    if len(df) > 1:
        vanilla_delta = vanilla.diff().abs().iloc[1:]
        gonet_t_delta = gonet_t.diff().abs().iloc[1:]

        vanilla_jitter = float(vanilla_delta.mean())
        gonet_t_jitter = float(gonet_t_delta.mean())

        vanilla_max_jump = float(vanilla_delta.max())
        gonet_t_max_jump = float(gonet_t_delta.max())
    else:
        vanilla_jitter = 0.0
        gonet_t_jitter = 0.0
        vanilla_max_jump = 0.0
        gonet_t_max_jump = 0.0

    decision_flip_count = int((vanilla_go != gonet_t_go).sum())

    return {
        "file": str(path),
        "segment_id": int(df.iloc[0]["segment_id"]),
        "num_frames": int(len(df)),

        "vanilla_mean": float(vanilla.mean()),
        "gonet_t_mean": float(gonet_t.mean()),
        "mean_abs_difference": float((vanilla - gonet_t).abs().mean()),

        "vanilla_go_frames": int(vanilla_go.sum()),
        "gonet_t_go_frames": int(gonet_t_go.sum()),
        "decision_flip_count": decision_flip_count,
        "decision_flip_rate": float(decision_flip_count / max(1, len(df))),

        "vanilla_jitter_mean_abs_delta": vanilla_jitter,
        "gonet_t_jitter_mean_abs_delta": gonet_t_jitter,
        "jitter_reduction_ratio": float(
            1.0 - (gonet_t_jitter / vanilla_jitter)
        ) if vanilla_jitter > 1e-12 else 0.0,

        "vanilla_max_jump": vanilla_max_jump,
        "gonet_t_max_jump": gonet_t_max_jump,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing *_comparison.csv files.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output summary CSV path. Default: input_dir/summary.csv",
    )

    parser.add_argument("--threshold", type=float, default=0.85)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    csv_paths = sorted(input_dir.glob("*_comparison.csv"))

    if len(csv_paths) == 0:
        raise RuntimeError(f"No *_comparison.csv files found in {input_dir}")

    rows = []

    for path in csv_paths:
        rows.append(summarize_one_csv(path, args.threshold))

    summary = pd.DataFrame(rows)

    if args.output is None:
        output_path = input_dir / "summary.csv"
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    print()
    print(f"Input dir: {input_dir}")
    print(f"Files summarized: {len(summary)}")
    print(f"Saved summary: {output_path}")

    print()
    print("Aggregate summary:")
    print(f"Total frames: {summary['num_frames'].sum()}")

    weighted = lambda col: (
        summary[col] * summary["num_frames"]
    ).sum() / summary["num_frames"].sum()

    print(f"Mean vanilla prob:        {weighted('vanilla_mean'):.4f}")
    print(f"Mean GONet+T prob:        {weighted('gonet_t_mean'):.4f}")
    print(f"Mean abs difference:      {weighted('mean_abs_difference'):.4f}")

    print()
    print(f"Vanilla GO frames:        {summary['vanilla_go_frames'].sum()}")
    print(f"GONet+T GO frames:        {summary['gonet_t_go_frames'].sum()}")
    print(f"Decision flips:           {summary['decision_flip_count'].sum()}")
    print(
        f"Decision flip rate:       "
        f"{summary['decision_flip_count'].sum() / summary['num_frames'].sum():.4f}"
    )

    print()
    print(f"Vanilla jitter:           {weighted('vanilla_jitter_mean_abs_delta'):.4f}")
    print(f"GONet+T jitter:           {weighted('gonet_t_jitter_mean_abs_delta'):.4f}")
    print(f"Jitter reduction:         {weighted('jitter_reduction_ratio'):.4f}")

    print()
    print("Top segments by decision flip rate:")
    cols = [
        "segment_id",
        "num_frames",
        "decision_flip_rate",
        "mean_abs_difference",
        "vanilla_jitter_mean_abs_delta",
        "gonet_t_jitter_mean_abs_delta",
        "jitter_reduction_ratio",
    ]
    print(
        summary.sort_values("decision_flip_rate", ascending=False)
        .head(10)[cols]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()