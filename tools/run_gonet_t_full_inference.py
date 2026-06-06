#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import cv2
import pandas as pd
import torch
from tqdm import tqdm

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


def load_segment_images(segment_df, output_size=128):
    tensors = []

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

    images = torch.stack(tensors, dim=0)  # [T, 3, 128, 128]
    return images


@torch.no_grad()
def infer_gonet_t_sequence(model, images, device, feature_chunk_size=64):
    """
    Runs GONet+T on a full segment.
    Backbone feature extraction is chunked, LSTM sees the full segment.
    """

    model.eval()

    images = images.to(device)
    total_frames = images.shape[0]

    feature_chunks = []

    for start in range(0, total_frames, feature_chunk_size):
        end = min(start + feature_chunk_size, total_frames)

        chunk = images[start:end].unsqueeze(0)  # [1, Tc, 3, 128, 128]
        features = model.extract_temporal_features(chunk)  # [1, Tc, 30]
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


def compute_summary(df, threshold):
    vanilla = df["vanilla_prob"].astype(float)
    gonet_t = df["gonet_t_prob"].astype(float)

    vanilla_go = vanilla >= threshold
    gonet_t_go = gonet_t >= threshold

    decision_flips = vanilla_go != gonet_t_go

    # Jitter should be computed within each segment, not across segment boundaries.
    vanilla_deltas = []
    gonet_t_deltas = []

    for _, sdf in df.groupby("segment_id"):
        sdf = sdf.sort_values("segment_local_index")

        if len(sdf) < 2:
            continue

        vanilla_deltas.extend(
            sdf["vanilla_prob"].astype(float).diff().abs().iloc[1:].tolist()
        )
        gonet_t_deltas.extend(
            sdf["gonet_t_prob"].astype(float).diff().abs().iloc[1:].tolist()
        )

    vanilla_jitter = sum(vanilla_deltas) / max(1, len(vanilla_deltas))
    gonet_t_jitter = sum(gonet_t_deltas) / max(1, len(gonet_t_deltas))

    jitter_reduction = (
        1.0 - gonet_t_jitter / vanilla_jitter
        if vanilla_jitter > 1e-12
        else 0.0
    )

    summary = {
        "num_frames": int(len(df)),
        "num_segments": int(df["segment_id"].nunique()),

        "mean_vanilla_prob": float(vanilla.mean()),
        "mean_gonet_t_prob": float(gonet_t.mean()),
        "mean_abs_difference": float((vanilla - gonet_t).abs().mean()),

        "vanilla_go_frames": int(vanilla_go.sum()),
        "gonet_t_go_frames": int(gonet_t_go.sum()),

        "decision_flip_count": int(decision_flips.sum()),
        "decision_flip_rate": float(decision_flips.sum() / max(1, len(df))),

        "vanilla_jitter_mean_abs_delta": float(vanilla_jitter),
        "gonet_t_jitter_mean_abs_delta": float(gonet_t_jitter),
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
        help="Pseudo-label CSV for one split/side.",
    )

    parser.add_argument(
        "--gonet-checkpoint",
        required=True,
        help="Vanilla GONet checkpoint.",
    )

    parser.add_argument(
        "--gonet-t-checkpoint",
        required=True,
        help="Trained GONet+T checkpoint.",
    )

    parser.add_argument(
        "--output-csv",
        required=True,
        help="Full output comparison CSV.",
    )

    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Optional summary CSV. Default: output_csv with _summary.csv suffix.",
    )

    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--feature-chunk-size", type=int, default=64)

    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--use-tanh", action="store_true")

    # Fallback values used only if checkpoint has no stored config.
    parser.add_argument("--target-mode", default="prob", choices=["prob", "logit"])
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
    output_csv = Path(args.output_csv)

    if args.summary_csv is None:
        summary_csv = output_csv.with_name(output_csv.stem + "_summary.csv")
    else:
        summary_csv = Path(args.summary_csv)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Pseudo CSV: {pseudo_csv}")
    print(f"Output CSV: {output_csv}")
    print(f"Summary CSV: {summary_csv}")

    df = pd.read_csv(pseudo_csv)

    df["segment_id"] = df["segment_id"].astype(int)
    df["segment_local_index"] = df["segment_local_index"].astype(int)
    df["building_id"] = df["building_id"].astype(int)
    df["frame_idx"] = df["frame_idx"].astype(int)
    df["prob_traversable"] = df["prob_traversable"].astype(float)

    model, temporal_config = load_gonet_t(
        gonet_checkpoint=Path(args.gonet_checkpoint),
        gonet_t_checkpoint=Path(args.gonet_t_checkpoint),
        device=device,
        args=args,
        nz=args.nz,
        use_tanh=args.use_tanh,
    )

    output_rows = []

    grouped = list(df.groupby("segment_id"))
    print(f"Segments to process: {len(grouped)}")
    print(f"Frames to process:   {len(df)}")

    for segment_id, segment_df in tqdm(grouped, desc="Running full GONet+T inference"):
        segment_df = segment_df.sort_values("segment_local_index").reset_index(drop=True)

        images = load_segment_images(segment_df)

        temporal_probs = infer_gonet_t_sequence(
            model=model,
            images=images,
            device=device,
            feature_chunk_size=args.feature_chunk_size,
        )

        for (_, row), gonet_t_prob in zip(segment_df.iterrows(), temporal_probs):
            vanilla_prob = float(row["prob_traversable"])
            gonet_t_prob = float(gonet_t_prob)

            vanilla_decision = "GO" if vanilla_prob >= args.threshold else "NO_GO"
            gonet_t_decision = "GO" if gonet_t_prob >= args.threshold else "NO_GO"

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
                "gonet_t_prob": gonet_t_prob,
                "threshold": float(args.threshold),
                "vanilla_decision": vanilla_decision,
                "gonet_t_decision": gonet_t_decision,
                "abs_difference": abs(vanilla_prob - gonet_t_prob),
            })

    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(output_csv, index=False)

    summary = compute_summary(out_df, threshold=args.threshold)
    save_summary(summary, summary_csv)

    print()
    print(f"Saved full comparison CSV: {output_csv}")
    print(f"Saved summary CSV:         {summary_csv}")

    print()
    print("Full-split summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()