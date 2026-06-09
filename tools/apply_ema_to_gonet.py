#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd

import math


def prob_to_logit_scalar(p, eps=1e-4):
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))

def sigmoid_scalar(x):
    return 1.0 / (1.0 + math.exp(-float(x)))
    
def apply_ema_to_segment(probs, alpha, ema_space="prob", eps=1e-4):
    """
    Applies causal EMA to one temporal segment.

    ema_space="prob":
        EMA directly on probabilities.

    ema_space="logit":
        Convert prob -> logit, apply EMA in logit space,
        then convert back using sigmoid.
    """

    if len(probs) == 0:
        return []

    probs = [float(p) for p in probs]

    if ema_space == "prob":
        ema = [probs[0]]

        for p in probs[1:]:
            smoothed = alpha * p + (1.0 - alpha) * ema[-1]
            ema.append(smoothed)

        return ema

    elif ema_space == "logit":
        logits = [prob_to_logit_scalar(p, eps=eps) for p in probs]

        ema_logits = [logits[0]]

        for z in logits[1:]:
            smoothed_z = alpha * z + (1.0 - alpha) * ema_logits[-1]
            ema_logits.append(smoothed_z)

        ema_probs = [sigmoid_scalar(z) for z in ema_logits]

        return ema_probs

    else:
        raise ValueError(f"Unsupported ema_space: {ema_space}")


def compute_summary(df, threshold):
    vanilla = df["vanilla_prob"].astype(float)
    ema = df["ema_prob"].astype(float)

    vanilla_go = vanilla >= threshold
    ema_go = ema >= threshold

    decision_flips = vanilla_go != ema_go

    vanilla_deltas = []
    ema_deltas = []

    for _, sdf in df.groupby("segment_id"):
        sdf = sdf.sort_values("segment_local_index")

        if len(sdf) < 2:
            continue

        vanilla_deltas.extend(
            sdf["vanilla_prob"].astype(float).diff().abs().iloc[1:].tolist()
        )

        ema_deltas.extend(
            sdf["ema_prob"].astype(float).diff().abs().iloc[1:].tolist()
        )

    vanilla_jitter = sum(vanilla_deltas) / max(1, len(vanilla_deltas))
    ema_jitter = sum(ema_deltas) / max(1, len(ema_deltas))

    jitter_reduction = (
        1.0 - ema_jitter / vanilla_jitter
        if vanilla_jitter > 1e-12
        else 0.0
    )

    summary = {
        "num_frames": int(len(df)),
        "num_segments": int(df["segment_id"].nunique()),

        "alpha": float(df["ema_alpha"].iloc[0]),
        "ema_space": str(df["ema_space"].iloc[0]) if "ema_space" in df.columns else "prob",
        "mean_vanilla_prob": float(vanilla.mean()),
        "mean_ema_prob": float(ema.mean()),
        "mean_abs_difference": float((vanilla - ema).abs().mean()),

        "vanilla_go_frames": int(vanilla_go.sum()),
        "ema_go_frames": int(ema_go.sum()),

        "decision_flip_count": int(decision_flips.sum()),
        "decision_flip_rate": float(decision_flips.sum() / max(1, len(df))),

        "vanilla_jitter_mean_abs_delta": float(vanilla_jitter),
        "ema_jitter_mean_abs_delta": float(ema_jitter),
        "jitter_reduction_ratio": float(jitter_reduction),
    }

    return summary


def save_summary(summary, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for k, v in summary.items():
            f.write(f"{k},{v}\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pseudo-csv",
        required=True,
        help="Pseudo-label CSV containing vanilla GONet probabilities.",
    )

    parser.add_argument(
        "--output-csv",
        required=True,
        help="Output CSV with EMA probabilities.",
    )

    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Optional summary CSV. Default: output_csv with _summary.csv suffix.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="EMA alpha. Higher = less smoothing, lower = more smoothing.",
    )

    parser.add_argument(
        "--ema-space",
        default="prob",
        choices=["prob", "logit"],
        help="Apply EMA in probability space or logit space.",
    )

    parser.add_argument(
        "--logit-eps",
        type=float,
        default=1e-4,
        help="Clamp epsilon for logit-space EMA.",
    )

    parser.add_argument("--threshold", type=float, default=0.85)

    args = parser.parse_args()

    if not (0.0 < args.alpha <= 1.0):
        raise ValueError("--alpha must be in the range (0, 1].")

    pseudo_csv = Path(args.pseudo_csv)
    output_csv = Path(args.output_csv)

    if args.summary_csv is None:
        summary_csv = output_csv.with_name(output_csv.stem + "_summary.csv")
    else:
        summary_csv = Path(args.summary_csv)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pseudo_csv)

    df["segment_id"] = df["segment_id"].astype(int)
    df["segment_local_index"] = df["segment_local_index"].astype(int)
    df["building_id"] = df["building_id"].astype(int)
    df["frame_idx"] = df["frame_idx"].astype(int)
    df["prob_traversable"] = df["prob_traversable"].astype(float)

    output_rows = []

    for segment_id, sdf in df.groupby("segment_id"):
        sdf = sdf.sort_values("segment_local_index").reset_index(drop=True)

        vanilla_probs = sdf["prob_traversable"].astype(float).tolist()
        ema_probs = apply_ema_to_segment( vanilla_probs, alpha=args.alpha, ema_space=args.ema_space, eps=args.logit_eps, )

        for (_, row), ema_prob in zip(sdf.iterrows(), ema_probs):
            vanilla_prob = float(row["prob_traversable"])
            ema_prob = float(ema_prob)

            output_rows.append({
                "segment_id": int(row["segment_id"]),
                "segment_local_index": int(row["segment_local_index"]),
                "global_index": int(row["global_index"]) if "global_index" in row else -1,
                "building_id": int(row["building_id"]),
                "frame_idx": int(row["frame_idx"]),
                "side": row["side"],
                "filename": row["filename"],
                "path": row["path"],
                "vanilla_prob": vanilla_prob,
                "ema_prob": ema_prob,
                "ema_alpha": float(args.alpha),
                "ema_space": args.ema_space,
                "threshold": float(args.threshold),
                "vanilla_decision": "GO" if vanilla_prob >= args.threshold else "NO_GO",
                "ema_decision": "GO" if ema_prob >= args.threshold else "NO_GO",
                "abs_difference": abs(vanilla_prob - ema_prob),
            })

    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(output_csv, index=False)

    summary = compute_summary(out_df, threshold=args.threshold)
    save_summary(summary, summary_csv)

    print()
    print(f"Saved EMA CSV:     {output_csv}")
    print(f"Saved summary CSV: {summary_csv}")

    print()
    print("EMA summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()