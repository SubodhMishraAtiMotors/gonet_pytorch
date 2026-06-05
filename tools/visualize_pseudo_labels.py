#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_longest_segments(df, output_dir: Path, top_k: int, threshold: float):
    output_dir.mkdir(parents=True, exist_ok=True)

    segment_stats = (
        df.groupby("segment_id")
        .agg(
            building_id=("building_id", "first"),
            side=("side", "first"),
            start_frame_idx=("frame_idx", "min"),
            end_frame_idx=("frame_idx", "max"),
            num_frames=("frame_idx", "count"),
            mean_prob=("prob_traversable", "mean"),
            min_prob=("prob_traversable", "min"),
            max_prob=("prob_traversable", "max"),
        )
        .reset_index()
        .sort_values("num_frames", ascending=False)
    )

    stats_path = output_dir / "segment_probability_summary.csv"
    segment_stats.to_csv(stats_path, index=False)

    print(f"Saved segment summary: {stats_path}")
    print()
    print("Top longest segments:")
    print(segment_stats.head(top_k).to_string(index=False))

    for _, seg in segment_stats.head(top_k).iterrows():
        segment_id = int(seg["segment_id"])
        sdf = df[df["segment_id"] == segment_id].copy()
        sdf = sdf.sort_values("frame_idx")

        plt.figure(figsize=(12, 4))
        plt.plot(sdf["frame_idx"], sdf["prob_traversable"], marker=".", linewidth=1)
        plt.axhline(threshold, linestyle="--", label=f"threshold={threshold}")
        plt.ylim(-0.05, 1.05)
        plt.xlabel("Frame index")
        plt.ylabel("GONet traversability probability")
        plt.title(
            f"Segment {segment_id} | "
            f"build {int(seg['building_id'])} | side {seg['side']} | "
            f"frames {int(seg['start_frame_idx'])}-{int(seg['end_frame_idx'])} | "
            f"N={int(seg['num_frames'])}"
        )
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        out_path = output_dir / f"segment_{segment_id:05d}_prob_plot.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

        print(f"Saved plot: {out_path}")


def make_segment_video(
    df,
    segment_id: int,
    output_dir: Path,
    threshold: float,
    fps: float,
    scale: int,
):
    sdf = df[df["segment_id"] == segment_id].copy()

    if len(sdf) == 0:
        print(f"Segment {segment_id} not found.")
        return

    sdf = sdf.sort_values("frame_idx")

    output_dir.mkdir(parents=True, exist_ok=True)

    first_path = Path(sdf.iloc[0]["path"])
    first_img = cv2.imread(str(first_path), cv2.IMREAD_COLOR)

    if first_img is None:
        print(f"Could not read first image: {first_path}")
        return

    h, w = first_img.shape[:2]
    out_w = w * scale
    out_h = h * scale

    video_path = output_dir / f"segment_{segment_id:05d}_pseudo_video.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (out_w, out_h))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {video_path}")

    for _, row in sdf.iterrows():
        path = Path(row["path"])
        prob = float(row["prob_traversable"])
        frame_idx = int(row["frame_idx"])
        building_id = int(row["building_id"])
        side = row["side"]

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if img is None:
            print(f"Warning: could not read {path}")
            continue

        img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        decision = "GO" if prob >= threshold else "NO-GO"
        color = (0, 180, 0) if decision == "GO" else (0, 0, 220)

        # Header background
        cv2.rectangle(img, (0, 0), (out_w, 95), (0, 0, 0), -1)

        cv2.putText(
            img,
            f"GONet pseudo-label prob: {prob:.3f}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            img,
            f"Decision: {decision} | threshold={threshold:.2f}",
            (15, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            img,
            f"build={building_id}, side={side}, frame={frame_idx}, segment={segment_id}",
            (15, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        # Decision border
        cv2.rectangle(img, (0, 0), (out_w - 1, out_h - 1), color, 8)

        writer.write(img)

    writer.release()

    print(f"Saved segment video: {video_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pseudo-csv",
        required=True,
        help="Pseudo-label CSV produced by pseudo_label_unlabelled_manifest.py",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/gonet_t_visual_checks",
    )

    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--scale", type=int, default=4)

    parser.add_argument(
        "--make-videos",
        action="store_true",
        help="Create annotated videos for the top-k longest segments.",
    )

    args = parser.parse_args()

    pseudo_csv = Path(args.pseudo_csv)
    output_dir = Path(args.output_dir)

    df = pd.read_csv(pseudo_csv)

    # Ensure numeric types
    df["segment_id"] = df["segment_id"].astype(int)
    df["building_id"] = df["building_id"].astype(int)
    df["frame_idx"] = df["frame_idx"].astype(int)
    df["prob_traversable"] = df["prob_traversable"].astype(float)

    print(f"Loaded pseudo-labels: {pseudo_csv}")
    print(f"Rows: {len(df)}")
    print()
    print("Probability summary:")
    print(df["prob_traversable"].describe())

    plot_longest_segments(
        df=df,
        output_dir=output_dir,
        top_k=args.top_k,
        threshold=args.threshold,
    )

    if args.make_videos:
        segment_lengths = (
            df.groupby("segment_id")
            .size()
            .sort_values(ascending=False)
        )

        top_segments = list(segment_lengths.head(args.top_k).index)

        video_dir = output_dir / "videos"

        for segment_id in top_segments:
            make_segment_video(
                df=df,
                segment_id=int(segment_id),
                output_dir=video_dir,
                threshold=args.threshold,
                fps=args.fps,
                scale=args.scale,
            )


if __name__ == "__main__":
    main()