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

import math

from datasets.go_stanford import preprocess_gonet_image
from models.gonet import Generator, InvG, Discriminator
from models.gonet_temporal import (
    GONetTemporalFeatureReducer,
    GONetTemporalClassifier,
    GONetTemporalFull,
)

def prob_to_logit_scalar(p, eps=1e-4):
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def sigmoid_scalar(x):
    return 1.0 / (1.0 + math.exp(-float(x)))
    
def apply_ema_to_segment(probs, alpha, ema_space="prob", eps=1e-4):
    if len(probs) == 0:
        return []

    probs = [float(p) for p in probs]

    if ema_space == "prob":
        ema = [probs[0]]

        for p in probs[1:]:
            ema.append(alpha * p + (1.0 - alpha) * ema[-1])

        return ema

    elif ema_space == "logit":
        logits = [prob_to_logit_scalar(p, eps=eps) for p in probs]

        ema_logits = [logits[0]]

        for z in logits[1:]:
            ema_logits.append(alpha * z + (1.0 - alpha) * ema_logits[-1])

        ema_probs = [sigmoid_scalar(z) for z in ema_logits]

        return ema_probs

    else:
        raise ValueError(f"Unsupported ema_space: {ema_space}")


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


def infer_temporal_config_from_checkpoint(ckpt, args):
    ckpt_args = ckpt.get("args", {})

    return {
        "target_mode": ckpt_args.get("target_mode", args.target_mode),
        "reduced_dim": int(ckpt_args.get("reduced_dim", args.reduced_dim)),
        "hidden_dim": int(ckpt_args.get("hidden_dim", args.hidden_dim)),
        "num_layers": int(ckpt_args.get("num_layers", args.num_layers)),
        "dropout": float(ckpt_args.get("dropout", args.dropout)),
        "bidirectional": bool(ckpt_args.get("bidirectional", args.bidirectional)),
    }


def load_gonet_t(
    gonet_checkpoint,
    gonet_t_checkpoint,
    device,
    args,
    nz=100,
    use_tanh=False,
):
    temporal_ckpt = torch.load(gonet_t_checkpoint, map_location=device)
    temporal_config = infer_temporal_config_from_checkpoint(temporal_ckpt, args)

    print("Loaded GONet+T config:")
    for k, v in temporal_config.items():
        print(f"  {k}: {v}")

    generator, invg, discriminator = load_backbone_from_gonet_checkpoint(
        checkpoint_path=gonet_checkpoint,
        device=device,
        nz=nz,
        use_tanh=use_tanh,
    )

    feature_reducer = GONetTemporalFeatureReducer(
        reduced_dim=temporal_config["reduced_dim"],
    ).to(device)

    temporal_classifier = GONetTemporalClassifier(
        input_dim=3 * temporal_config["reduced_dim"],
        hidden_dim=temporal_config["hidden_dim"],
        num_layers=temporal_config["num_layers"],
        dropout=temporal_config["dropout"],
        bidirectional=temporal_config["bidirectional"],
    ).to(device)

    model = GONetTemporalFull(
        generator=generator,
        invg=invg,
        discriminator=discriminator,
        feature_reducer=feature_reducer,
        temporal_classifier=temporal_classifier,
    ).to(device)

    model.freeze_gonet_backbone()

    model.feature_reducer.load_state_dict(temporal_ckpt["feature_reducer"])
    model.temporal_classifier.load_state_dict(temporal_ckpt["temporal_classifier"])

    model.eval()

    return model, temporal_config


def load_segment_images(segment_df):
    tensors = []
    original_images = []

    for _, row in segment_df.iterrows():
        path = Path(row["path"])
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise RuntimeError(f"Could not read image: {path}")

        tensor = preprocess_gonet_image(
            image_bgr=img_bgr,
            output_size=128,
            use_fisheye_mask=False,
            rotate_clockwise=False,
        )

        tensors.append(tensor)
        original_images.append(img_bgr)

    images = torch.stack(tensors, dim=0)

    return images, original_images


@torch.no_grad()
def infer_gonet_t_sequence(model, images, device, feature_chunk_size=64):
    model.eval()

    images = images.to(device)
    total_frames = images.shape[0]

    feature_chunks = []

    for start in range(0, total_frames, feature_chunk_size):
        end = min(start + feature_chunk_size, total_frames)

        chunk = images[start:end].unsqueeze(0)
        features = model.extract_temporal_features(chunk)
        feature_chunks.append(features.detach().cpu())

    temporal_features = torch.cat(feature_chunks, dim=1).to(device)
    lengths = torch.tensor([total_frames], dtype=torch.long, device=device)

    logits = model.temporal_classifier(
        temporal_features=temporal_features,
        lengths=lengths,
        return_logits=True,
    )

    probs = torch.sigmoid(logits)
    probs = probs.squeeze(0).squeeze(-1).detach().cpu().tolist()

    return probs


def plot_three_way(segment_df, gonet_t_probs, ema_probs, threshold, output_path, alpha):
    frame_idx = segment_df["frame_idx"].astype(int).tolist()
    vanilla_probs = segment_df["prob_traversable"].astype(float).tolist()

    segment_id = int(segment_df.iloc[0]["segment_id"])
    building_id = int(segment_df.iloc[0]["building_id"])
    side = segment_df.iloc[0]["side"]

    plt.figure(figsize=(14, 5))
    plt.plot(frame_idx, vanilla_probs, marker=".", linewidth=1.4, label="Vanilla GONet")
    plt.plot(frame_idx, gonet_t_probs, marker=".", linewidth=1.4, label="GONet+T")
    plt.plot(frame_idx, ema_probs, marker=".", linewidth=1.4, label=f"EMA alpha={alpha}")
    plt.axhline(threshold, linestyle="--", label=f"threshold={threshold:.2f}")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Frame index")
    plt.ylabel("Traversability probability")
    plt.title(
        f"Vanilla GONet vs GONet+T vs EMA | "
        f"segment={segment_id}, build={building_id}, side={side}, N={len(segment_df)}"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def decision(prob, threshold):
    return "GO" if prob >= threshold else "NO-GO"


def decision_color(decision_text):
    return (0, 180, 0) if decision_text == "GO" else (0, 0, 220)


def draw_three_way_overlay(
    image_bgr,
    vanilla_prob,
    gonet_t_prob,
    ema_prob,
    threshold,
    alpha,
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

    vanilla_dec = decision(vanilla_prob, threshold)
    gonet_t_dec = decision(gonet_t_prob, threshold)
    ema_dec = decision(ema_prob, threshold)

    vanilla_color = decision_color(vanilla_dec)
    gonet_t_color = decision_color(gonet_t_dec)
    ema_color = decision_color(ema_dec)

    # Border uses GONet+T decision
    border_color = gonet_t_color

    cv2.rectangle(image_bgr, (0, 0), (out_w, 165), (0, 0, 0), -1)

    cv2.putText(
        image_bgr,
        f"Vanilla GONet: {vanilla_prob:.3f} | {vanilla_dec}",
        (15, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        vanilla_color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        image_bgr,
        f"GONet+T:       {gonet_t_prob:.3f} | {gonet_t_dec}",
        (15, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        gonet_t_color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        image_bgr,
        f"EMA a={alpha:.2f}:    {ema_prob:.3f} | {ema_dec}",
        (15, 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        ema_color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        image_bgr,
        f"threshold={threshold:.2f} | build={building_id}, side={side}, segment={segment_id}, frame={frame_idx}",
        (15, 138),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

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

    cv2.rectangle(
        image_bgr,
        (0, 0),
        (out_w - 1, out_h - 1),
        border_color,
        8,
    )

    return image_bgr


def make_three_way_video(
    segment_df,
    original_images,
    gonet_t_probs,
    ema_probs,
    threshold,
    alpha,
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

    for image_bgr, (_, row), gonet_t_prob, ema_prob in zip(
        original_images,
        segment_df.iterrows(),
        gonet_t_probs,
        ema_probs,
    ):
        vanilla_prob = float(row["prob_traversable"])

        vis = draw_three_way_overlay(
            image_bgr=image_bgr.copy(),
            vanilla_prob=vanilla_prob,
            gonet_t_prob=float(gonet_t_prob),
            ema_prob=float(ema_prob),
            threshold=threshold,
            alpha=alpha,
            frame_idx=int(row["frame_idx"]),
            segment_id=int(row["segment_id"]),
            building_id=int(row["building_id"]),
            side=row["side"],
            filename=row["filename"],
            scale=scale,
        )

        writer.write(vis)

    writer.release()


def write_three_way_csv(segment_df, gonet_t_probs, ema_probs, threshold, alpha, output_path):
    rows = []

    for (_, row), gonet_t_prob, ema_prob in zip(
        segment_df.iterrows(),
        gonet_t_probs,
        ema_probs,
    ):
        vanilla_prob = float(row["prob_traversable"])
        gonet_t_prob = float(gonet_t_prob)
        ema_prob = float(ema_prob)

        rows.append({
            "segment_id": int(row["segment_id"]),
            "segment_local_index": int(row["segment_local_index"]),
            "building_id": int(row["building_id"]),
            "frame_idx": int(row["frame_idx"]),
            "side": row["side"],
            "filename": row["filename"],
            "path": row["path"],
            "vanilla_prob": vanilla_prob,
            "gonet_t_prob": gonet_t_prob,
            "ema_prob": ema_prob,
            "ema_alpha": float(alpha),
            "threshold": float(threshold),
            "vanilla_decision": decision(vanilla_prob, threshold),
            "gonet_t_decision": decision(gonet_t_prob, threshold),
            "ema_decision": decision(ema_prob, threshold),
            "abs_diff_gonet_t": abs(vanilla_prob - gonet_t_prob),
            "abs_diff_ema": abs(vanilla_prob - ema_prob),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def select_segments(df, segment_ids=None, top_k=5, min_length=1):
    segment_lengths = df.groupby("segment_id").size().sort_values(ascending=False)

    if segment_ids is not None and len(segment_ids) > 0:
        return [int(x) for x in segment_ids]

    segment_lengths = segment_lengths[segment_lengths >= min_length]
    return [int(x) for x in segment_lengths.head(top_k).index]


def print_summary(segment_df, gonet_t_probs, ema_probs, threshold):
    vanilla = segment_df["prob_traversable"].astype(float).values
    gonet_t = pd.Series(gonet_t_probs).values
    ema = pd.Series(ema_probs).values

    vanilla_go = (vanilla >= threshold).sum()
    gonet_t_go = (gonet_t >= threshold).sum()
    ema_go = (ema >= threshold).sum()

    if len(vanilla) > 1:
        vanilla_jitter = abs(pd.Series(vanilla).diff()).iloc[1:].mean()
        gonet_t_jitter = abs(pd.Series(gonet_t).diff()).iloc[1:].mean()
        ema_jitter = abs(pd.Series(ema).diff()).iloc[1:].mean()
    else:
        vanilla_jitter = 0.0
        gonet_t_jitter = 0.0
        ema_jitter = 0.0

    print(f"Mean vanilla prob:       {vanilla.mean():.4f}")
    print(f"Mean GONet+T prob:       {gonet_t.mean():.4f}")
    print(f"Mean EMA prob:           {ema.mean():.4f}")
    print(f"Mean abs diff GONet+T:   {abs(vanilla - gonet_t).mean():.4f}")
    print(f"Mean abs diff EMA:       {abs(vanilla - ema).mean():.4f}")
    print(f"Vanilla GO frames:       {vanilla_go}/{len(vanilla)}")
    print(f"GONet+T GO frames:       {gonet_t_go}/{len(gonet_t)}")
    print(f"EMA GO frames:           {ema_go}/{len(ema)}")
    print(f"Vanilla jitter:          {vanilla_jitter:.4f}")
    print(f"GONet+T jitter:          {gonet_t_jitter:.4f}")
    print(f"EMA jitter:              {ema_jitter:.4f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pseudo-csv", required=True)
    parser.add_argument("--gonet-checkpoint", required=True)
    parser.add_argument("--gonet-t-checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/gonet_vs_gonett_vs_ema")

    parser.add_argument("--segment-id", nargs="*", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-length", type=int, default=40)

    parser.add_argument("--ema-alpha", type=float, default=0.7)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--feature-chunk-size", type=int, default=64)

    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--use-tanh", action="store_true")

    # Fallback if checkpoint has no config
    parser.add_argument("--target-mode", default="prob", choices=["prob", "logit"])
    parser.add_argument("--reduced-dim", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bidirectional", action="store_true")

    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

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

    args = parser.parse_args()

    if not (0.0 < args.ema_alpha <= 1.0):
        raise ValueError("--ema-alpha must be in the range (0, 1].")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Pseudo CSV: {args.pseudo_csv}")
    print(f"EMA alpha: {args.ema_alpha}")

    df = pd.read_csv(args.pseudo_csv)

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

    print("Selected segments:", selected_segments)

    model, temporal_config = load_gonet_t(
        gonet_checkpoint=Path(args.gonet_checkpoint),
        gonet_t_checkpoint=Path(args.gonet_t_checkpoint),
        device=device,
        args=args,
        nz=args.nz,
        use_tanh=args.use_tanh,
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

        vanilla_probs = segment_df["prob_traversable"].astype(float).tolist()
        ema_probs = apply_ema_to_segment(vanilla_probs, alpha=args.ema_alpha, ema_space=args.ema_space, eps=args.logit_eps,)

        images, original_images = load_segment_images(segment_df)

        gonet_t_probs = infer_gonet_t_sequence(
            model=model,
            images=images,
            device=device,
            feature_chunk_size=args.feature_chunk_size,
        )

        base_name = (
            f"segment_{segment_id:05d}_"
            f"build{int(first['building_id'])}_"
            f"{first['side']}_"
            f"{int(first['frame_idx'])}_{int(last['frame_idx'])}_"
            f"ema_{args.ema_space}_{args.ema_alpha:.2f}"
        )

        plot_path = output_dir / f"{base_name}_plot.png"
        video_path = output_dir / f"{base_name}_video.mp4"
        csv_path = output_dir / f"{base_name}_comparison.csv"

        plot_three_way(
            segment_df=segment_df,
            gonet_t_probs=gonet_t_probs,
            ema_probs=ema_probs,
            threshold=args.threshold,
            output_path=plot_path,
            alpha=args.ema_alpha,
        )

        make_three_way_video(
            segment_df=segment_df,
            original_images=original_images,
            gonet_t_probs=gonet_t_probs,
            ema_probs=ema_probs,
            threshold=args.threshold,
            alpha=args.ema_alpha,
            output_path=video_path,
            fps=args.fps,
            scale=args.scale,
        )

        write_three_way_csv(
            segment_df=segment_df,
            gonet_t_probs=gonet_t_probs,
            ema_probs=ema_probs,
            threshold=args.threshold,
            alpha=args.ema_alpha,
            output_path=csv_path,
        )

        print(f"Saved plot:  {plot_path}")
        print(f"Saved video: {video_path}")
        print(f"Saved CSV:   {csv_path}")

        print_summary(
            segment_df=segment_df,
            gonet_t_probs=gonet_t_probs,
            ema_probs=ema_probs,
            threshold=args.threshold,
        )


if __name__ == "__main__":
    main()