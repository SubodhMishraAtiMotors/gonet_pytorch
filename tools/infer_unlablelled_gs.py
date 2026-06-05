#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from datasets.go_stanford import preprocess_gonet_image
from models.gonet import Generator, InvG, Discriminator, GONetClassifier


def parse_go_stanford_name(path: Path):
    """
    Parses names like:
        img_build9_8060_L.jpg
        img_build12_360_R.jpg

    Returns:
        building number, frame index, side
    """
    pattern = r"img_build(\d+)_(\d+)_([LR])"
    match = re.search(pattern, path.stem)

    if match is None:
        return (999999, 999999, path.stem)

    building = int(match.group(1))
    frame_idx = int(match.group(2))
    side = match.group(3)

    return (building, frame_idx, side)


def collect_sorted_images(folder: Path):
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(folder.glob(ext))

    image_paths = sorted(image_paths, key=parse_go_stanford_name)
    return image_paths


def load_model(checkpoint_path: Path, device: str, nz: int = 100, use_tanh: bool = False):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    generator = Generator(nz=nz, use_tanh=use_tanh).to(device)
    invg = InvG(nz=nz).to(device)
    discriminator = Discriminator().to(device)
    classifier = GONetClassifier().to(device)

    generator.load_state_dict(checkpoint["generator"])
    invg.load_state_dict(checkpoint["invg"])
    discriminator.load_state_dict(checkpoint["discriminator"])
    classifier.load_state_dict(checkpoint["classifier"])

    generator.eval()
    invg.eval()
    discriminator.eval()
    classifier.eval()

    return generator, invg, discriminator, classifier


@torch.no_grad()
def infer_one_image(image_bgr, generator, invg, discriminator, classifier, device):
    image_tensor = preprocess_gonet_image(
        image_bgr=image_bgr,
        output_size=128,
        use_fisheye_mask=False,
        rotate_clockwise=False,
    )

    image_tensor = image_tensor.unsqueeze(0).to(device)

    z_hat = invg(image_tensor)
    img_gen = generator(z_hat)

    dis_real = discriminator(image_tensor)
    dis_gen = discriminator(img_gen)

    prob = classifier(
        img_error=image_tensor - img_gen,
        dis_error=dis_real - dis_gen,
        dis_real=dis_real,
    )

    return float(prob.item())


def draw_overlay(
    image_bgr,
    prob,
    threshold,
    filename=None,
    scale=4,
):
    """
    Draws probability and GO/NO-GO decision on the frame.
    """

    h, w = image_bgr.shape[:2]

    if scale != 1:
        image_bgr = cv2.resize(
            image_bgr,
            (w * scale, h * scale),
            interpolation=cv2.INTER_NEAREST,
        )

    out_h, out_w = image_bgr.shape[:2]

    decision = "GO" if prob >= threshold else "NO-GO"

    if decision == "GO":
        color = (0, 180, 0)
    else:
        color = (0, 0, 220)

    # Top black band
    cv2.rectangle(image_bgr, (0, 0), (out_w, 80), (0, 0, 0), -1)

    text1 = f"GONet prob: {prob:.3f}"
    text2 = f"Decision: {decision}  |  threshold={threshold:.2f}"

    cv2.putText(
        image_bgr,
        text1,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        image_bgr,
        text2,
        (15, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )

    # Border color indicates decision
    thickness = 8
    cv2.rectangle(
        image_bgr,
        (0, 0),
        (out_w - 1, out_h - 1),
        color,
        thickness,
    )

    if filename is not None:
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

    return image_bgr


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)

    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "vali", "validation", "test"],
    )

    parser.add_argument(
        "--side",
        default="L",
        choices=["L", "R"],
    )

    parser.add_argument(
        "--output-video",
        default=None,
    )

    parser.add_argument(
        "--output-csv",
        default=None,
    )

    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--use-tanh", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional limit for quick testing.",
    )

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        device = "cpu"

    split_map = {
        "train": "data_train",
        "val": "data_vali",
        "vali": "data_vali",
        "validation": "data_vali",
        "test": "data_test",
    }

    data_root = Path(args.data_root)
    split_name = split_map[args.split]

    image_folder = (
        data_root
        / "whole_dataset"
        / split_name
        / f"unlabel_{args.side}"
    )

    if not image_folder.exists():
        raise FileNotFoundError(f"Folder not found: {image_folder}")

    image_paths = collect_sorted_images(image_folder)

    if args.max_frames is not None:
        image_paths = image_paths[: args.max_frames]

    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in: {image_folder}")

    print(f"Using device: {device}")
    print(f"Image folder: {image_folder}")
    print(f"Frames: {len(image_paths)}")
    print(f"Threshold: {args.threshold}")

    checkpoint_path = Path(args.checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")

    generator, invg, discriminator, classifier = load_model(
        checkpoint_path=checkpoint_path,
        device=device,
        nz=args.nz,
        use_tanh=args.use_tanh,
    )

    output_dir = Path("outputs/unlabelled_inference_videos")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_video is None:
        output_video = output_dir / f"go_stanford_{args.split}_unlabel_{args.side}_gonet_thr{args.threshold:.2f}.mp4"
    else:
        output_video = Path(args.output_video)
        output_video.parent.mkdir(parents=True, exist_ok=True)

    if args.output_csv is None:
        output_csv = output_video.with_suffix(".csv")
    else:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Prepare video writer
    first_img = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if first_img is None:
        raise RuntimeError(f"Could not read first image: {image_paths[0]}")

    first_vis = draw_overlay(
        image_bgr=first_img.copy(),
        prob=0.0,
        threshold=args.threshold,
        filename=image_paths[0].name,
        scale=args.scale,
    )

    out_h, out_w = first_vis.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_video),
        fourcc,
        args.fps,
        (out_w, out_h),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_video}")

    rows = []

    for idx, path in enumerate(tqdm(image_paths, desc="Running GONet inference")):
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if image_bgr is None:
            print(f"Warning: could not read {path}")
            continue

        prob = infer_one_image(
            image_bgr=image_bgr,
            generator=generator,
            invg=invg,
            discriminator=discriminator,
            classifier=classifier,
            device=device,
        )

        decision = "GO" if prob >= args.threshold else "NO_GO"

        vis = draw_overlay(
            image_bgr=image_bgr.copy(),
            prob=prob,
            threshold=args.threshold,
            filename=path.name,
            scale=args.scale,
        )

        writer.write(vis)

        building, frame_idx, side = parse_go_stanford_name(path)

        rows.append({
            "index": idx,
            "path": str(path),
            "filename": path.name,
            "building": building,
            "frame_idx": frame_idx,
            "side": side,
            "prob_traversable": prob,
            "threshold": args.threshold,
            "decision": decision,
        })

    writer.release()

    with open(output_csv, "w", newline="") as f:
        fieldnames = [
            "index",
            "path",
            "filename",
            "building",
            "frame_idx",
            "side",
            "prob_traversable",
            "threshold",
            "decision",
        ]

        writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    print()
    print(f"Saved video: {output_video}")
    print(f"Saved CSV:   {output_csv}")


if __name__ == "__main__":
    main()