#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import cv2
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets.go_stanford import preprocess_gonet_image
from models.gonet import Generator, InvG, Discriminator
from models.gonet_temporal import (
    GONetTemporalFeatureReducer,
    GONetTemporalClassifier,
    GONetTemporalFull,
)


def load_backbone_from_gonet_checkpoint(checkpoint_path, device, nz=100, use_tanh=False):
    ckpt = torch.load(checkpoint_path, map_location=device)

    generator = Generator(nz=nz, use_tanh=use_tanh).to(device)
    invg = InvG(nz=nz).to(device)
    discriminator = Discriminator().to(device)

    generator.load_state_dict(ckpt["generator"])
    invg.load_state_dict(ckpt["invg"])
    discriminator.load_state_dict(ckpt["discriminator"])

    generator.eval()
    invg.eval()
    discriminator.eval()

    return generator, invg, discriminator


def load_gonet_t(
    gonet_checkpoint,
    gonet_t_checkpoint,
    device,
    nz=100,
    use_tanh=False,
    reduced_dim=10,
    hidden_dim=64,
    num_layers=1,
    dropout=0.0,
    bidirectional=False,
):
    generator, invg, discriminator = load_backbone_from_gonet_checkpoint(
        checkpoint_path=gonet_checkpoint,
        device=device,
        nz=nz,
        use_tanh=use_tanh,
    )

    feature_reducer = GONetTemporalFeatureReducer(
        reduced_dim=reduced_dim,
    ).to(device)

    temporal_classifier = GONetTemporalClassifier(
        input_dim=3 * reduced_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    ).to(device)

    model = GONetTemporalFull(
        generator=generator,
        invg=invg,
        discriminator=discriminator,
        feature_reducer=feature_reducer,
        temporal_classifier=temporal_classifier,
    ).to(device)

    model.freeze_gonet_backbone()

    ckpt = torch.load(gonet_t_checkpoint, map_location=device)

    model.feature_reducer.load_state_dict(ckpt["feature_reducer"])
    model.temporal_classifier.load_state_dict(ckpt["temporal_classifier"])

    model.eval()

    return model


def load_segment_images(segment_df, output_size=128):
    tensors = []
    original_images = []

    for _, row in segment_df.iterrows():
        path = Path(row["path"])
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise RuntimeError(f"Could not read image: {path}")

        tensor = preprocess_gonet_image(
            image_bgr=img_bgr,
            output_size=output_size,
            use_fisheye_mask=False,
            rotate_clockwise=False,
        )

        tensors.append(tensor)
        original_images.append(img_bgr)

    images = torch.stack(tensors, dim=0)  # [T, 3, 128, 128]

    return images, original_images


@torch.no_grad()
def infer_gonet_t_sequence(model, images, device, feature_chunk_size=64):
    """
    Runs GONet+T on a full segment.

    To avoid GPU memory spikes, GONet backbone feature extraction is done
    in chunks, but the LSTM still sees the full segment at once.
    """

    model.eval()

    images = images.to(device)
    total_frames = images.shape[0]

    feature_chunks = []

    for start in range(0, total_frames, feature_chunk_size):
        end = min(start + feature_chunk_size, total_frames)

        chunk = images[start:end].unsqueeze(0)  # [1, Tc, 3, 128, 128]

        features = model.extract_temporal_features(chunk)  # [1, Tc, 30]
        feature_chunks.append(features.cpu())

    temporal_features = torch.cat(feature_chunks, dim=1).to(device)  # [1, T, 30]

    lengths = torch.tensor([total_frames], dtype=torch.long, device=device)

    probs = model.temporal_classifier(
        temporal_features=temporal_features,
        lengths=lengths,
    )

    probs = probs.squeeze(0).squeeze(-1).detach().cpu().tolist()

    return probs


def plot_comparison(segment_df, temporal_probs, threshold, output_path):
    frame_idx = segment_df["frame_idx"].astype(int).tolist()
    vanilla_probs = segment_df["prob_traversable"].astype(float).tolist()

    segment_id = int(segment_df.iloc[0]["segment_id"])
    building_id = int(segment_df.iloc[0]["building_id"])
    side = segment_df.iloc[0]["side"]

    plt.figure(figsize=(13, 5))
    plt.plot(frame_idx, vanilla_probs, marker=".", linewidth=1.5, label="Vanilla GONet")
    plt.plot(frame_idx, temporal_probs, marker=".", linewidth=1.5, label="GONet+T")
    plt.axhline(threshold, linestyle="--", label=f"threshold={threshold:.2f}")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Frame index")
    plt.ylabel("Traversability probability")
    plt.title(
        f"Vanilla GONet vs GONet+T | "
        f"segment={segment_id}, build={building_id}, side={side}, N={len(segment_df)}"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def write_comparison_csv(segment_df, temporal_probs, threshold, output_path):
    rows = []

    for (_, row), temporal_prob in zip(segment_df.iterrows(), temporal_probs):
        vanilla_prob = float(row["prob_traversable"])

        rows.append({
            "segment_id": int(row["segment_id"]),
            "segment_local_index": int(row["segment_local_index"]),
            "building_id": int(row["building_id"]),
            "frame_idx": int(row["frame_idx"]),
            "side": row["side"],
            "filename": row["filename"],
            "path": row["path"],
            "vanilla_prob": vanilla_prob,
            "gonet_t_prob": float(temporal_prob),
            "threshold": float(threshold),
            "vanilla_decision": "GO" if vanilla_prob >= threshold else "NO_GO",
            "gonet_t_decision": "GO" if temporal_prob >= threshold else "NO_GO",
            "abs_difference": abs(float(temporal_prob) - vanilla_prob),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_comparison_overlay(
    image_bgr,
    vanilla_prob,
    temporal_prob,
    threshold,
    frame_idx,
    segment_id,
    building_id,
    side,
    filename,
    scale=4,
):
    h, w = image_bgr.shape[:2]

    if scale != 1:
        image_bgr = cv2.resize(
            image_bgr,
            (w * scale, h * scale),
            interpolation=cv2.INTER_NEAREST,
        )

    out_h, out_w = image_bgr.shape[:2]

    vanilla_decision = "GO" if vanilla_prob >= threshold else "NO-GO"
    temporal_decision = "GO" if temporal_prob >= threshold else "NO-GO"

    temporal_color = (0, 180, 0) if temporal_decision == "GO" else (0, 0, 220)
    vanilla_color = (0, 180, 0) if vanilla_decision == "GO" else (0, 0, 220)

    # Header
    cv2.rectangle(image_bgr, (0, 0), (out_w, 125), (0, 0, 0), -1)

    cv2.putText(
        image_bgr,
        f"Vanilla GONet: {vanilla_prob:.3f} | {vanilla_decision}",
        (15, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        vanilla_color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        image_bgr,
        f"GONet+T:       {temporal_prob:.3f} | {temporal_decision}",
        (15, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        temporal_color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        image_bgr,
        f"threshold={threshold:.2f} | build={building_id}, side={side}, segment={segment_id}, frame={frame_idx}",
        (15, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    # Bottom filename strip
    cv2.rectangle(image_bgr, (0, out_h - 35), (out_w, out_h), (0, 0, 0), -1)

    cv2.putText(
        image_bgr,
        filename,
        (15, out_h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Border uses GONet+T decision
    cv2.rectangle(
        image_bgr,
        (0, 0),
        (out_w - 1, out_h - 1),
        temporal_color,
        8,
    )

    return image_bgr


def make_comparison_video(
    segment_df,
    original_images,
    temporal_probs,
    threshold,
    output_path,
    fps=3.0,
    scale=4,
):
    first = original_images[0]
    h, w = first.shape[:2]
    out_w = w * scale
    out_h = h * scale

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")

    for image_bgr, (_, row), temporal_prob in zip(
        original_images,
        segment_df.iterrows(),
        temporal_probs,
    ):
        vanilla_prob = float(row["prob_traversable"])

        vis = draw_comparison_overlay(
            image_bgr=image_bgr.copy(),
            vanilla_prob=vanilla_prob,
            temporal_prob=float(temporal_prob),
            threshold=threshold,
            frame_idx=int(row["frame_idx"]),
            segment_id=int(row["segment_id"]),
            building_id=int(row["building_id"]),
            side=row["side"],
            filename=row["filename"],
            scale=scale,
        )

        writer.write(vis)

    writer.release()


def select_segments(df, segment_ids=None, top_k=5, min_length=1):
    segment_lengths = df.groupby("segment_id").size().sort_values(ascending=False)

    if segment_ids is not None and len(segment_ids) > 0:
        selected = [int(x) for x in segment_ids]
    else:
        segment_lengths = segment_lengths[segment_lengths >= min_length]
        selected = [int(x) for x in segment_lengths.head(top_k).index]

    return selected


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pseudo-csv",
        required=True,
        help="Pseudo-label CSV for one side, e.g. train_unlabel_L_pseudo.csv",
    )

    parser.add_argument(
        "--gonet-checkpoint",
        required=True,
        help="Vanilla GONet checkpoint, e.g. checkpoints/gonet_fl/fl_best.pt",
    )

    parser.add_argument(
        "--gonet-t-checkpoint",
        required=True,
        help="GONet+T checkpoint, e.g. checkpoints/gonet_t/gonet_t_latest.pt",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/gonet_t_compare",
    )

    parser.add_argument(
        "--segment-id",
        nargs="*",
        type=int,
        default=None,
        help="Specific segment IDs to visualize. If omitted, top longest segments are used.",
    )

    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-length", type=int, default=10)

    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--scale", type=int, default=4)

    parser.add_argument(
        "--feature-chunk-size",
        type=int,
        default=64,
        help="Backbone feature extraction chunk size for long segments.",
    )

    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--use-tanh", action="store_true")

    parser.add_argument("--reduced-dim", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bidirectional", action="store_true")

    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    pseudo_csv = Path(args.pseudo_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Pseudo CSV: {pseudo_csv}")
    print(f"Vanilla GONet checkpoint: {args.gonet_checkpoint}")
    print(f"GONet+T checkpoint:       {args.gonet_t_checkpoint}")

    df = pd.read_csv(pseudo_csv)

    df["segment_id"] = df["segment_id"].astype(int)
    df["segment_local_index"] = df["segment_local_index"].astype(int)
    df["building_id"] = df["building_id"].astype(int)
    df["frame_idx"] = df["frame_idx"].astype(int)
    df["prob_traversable"] = df["prob_traversable"].astype(float)

    selected_segments = select_segments(
        df=df,
        segment_ids=args.segment_id,
        top_k=args.top_k,
        min_length=args.min_length,
    )

    print()
    print("Selected segments:", selected_segments)

    model = load_gonet_t(
        gonet_checkpoint=Path(args.gonet_checkpoint),
        gonet_t_checkpoint=Path(args.gonet_t_checkpoint),
        device=device,
        nz=args.nz,
        use_tanh=args.use_tanh,
        reduced_dim=args.reduced_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    )

    for segment_id in selected_segments:
        segment_df = df[df["segment_id"] == segment_id].copy()

        if len(segment_df) == 0:
            print(f"Skipping missing segment: {segment_id}")
            continue

        segment_df = segment_df.sort_values("segment_local_index").reset_index(drop=True)

        first = segment_df.iloc[0]
        last = segment_df.iloc[-1]

        print()
        print(
            f"Segment {segment_id}: "
            f"build={int(first['building_id'])}, side={first['side']}, "
            f"frames={int(first['frame_idx'])}->{int(last['frame_idx'])}, "
            f"N={len(segment_df)}"
        )

        images, original_images = load_segment_images(segment_df)

        temporal_probs = infer_gonet_t_sequence(
            model=model,
            images=images,
            device=device,
            feature_chunk_size=args.feature_chunk_size,
        )

        base_name = (
            f"segment_{segment_id:05d}_"
            f"build{int(first['building_id'])}_"
            f"{first['side']}_"
            f"{int(first['frame_idx'])}_{int(last['frame_idx'])}"
        )

        plot_path = output_dir / f"{base_name}_plot.png"
        video_path = output_dir / f"{base_name}_video.mp4"
        csv_path = output_dir / f"{base_name}_comparison.csv"

        plot_comparison(
            segment_df=segment_df,
            temporal_probs=temporal_probs,
            threshold=args.threshold,
            output_path=plot_path,
        )

        make_comparison_video(
            segment_df=segment_df,
            original_images=original_images,
            temporal_probs=temporal_probs,
            threshold=args.threshold,
            output_path=video_path,
            fps=args.fps,
            scale=args.scale,
        )

        write_comparison_csv(
            segment_df=segment_df,
            temporal_probs=temporal_probs,
            threshold=args.threshold,
            output_path=csv_path,
        )

        vanilla = segment_df["prob_traversable"].astype(float).values
        temporal = pd.Series(temporal_probs).values

        print(f"Saved plot:  {plot_path}")
        print(f"Saved video: {video_path}")
        print(f"Saved CSV:   {csv_path}")
        print(f"Mean vanilla prob: {vanilla.mean():.4f}")
        print(f"Mean GONet+T prob: {temporal.mean():.4f}")
        print(f"Mean abs difference: {abs(vanilla - temporal).mean():.4f}")


if __name__ == "__main__":
    main()