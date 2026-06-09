#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

import cv2
import pandas as pd
import torch
from tqdm import tqdm

from datasets.go_stanford import preprocess_gonet_image
from models.gonet import Generator, InvG, Discriminator, GONetClassifier
from models.gonet_temporal import (
    GONetTemporalFeatureReducer,
    GONetTemporalClassifier,
    GONetTemporalFull,
)


def sync_if_cuda(device):
    if device == "cuda":
        torch.cuda.synchronize()


def collect_image_paths_from_pseudo_csv(pseudo_csv, max_frames=None):
    df = pd.read_csv(pseudo_csv)
    df = df.sort_values(["segment_id", "segment_local_index"]).reset_index(drop=True)

    if max_frames is not None:
        df = df.iloc[:max_frames].copy()

    return [Path(p) for p in df["path"].tolist()]


def load_images_as_tensor(image_paths, device):
    tensors = []

    for path in image_paths:
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

    images = torch.stack(tensors, dim=0).to(device)
    return images


def load_gonet(checkpoint_path, device, nz=100, use_tanh=False):
    ckpt = torch.load(checkpoint_path, map_location=device)

    generator = Generator(nz=nz, use_tanh=use_tanh).to(device)
    invg = InvG(nz=nz).to(device)
    discriminator = Discriminator().to(device)
    classifier = GONetClassifier().to(device)

    generator.load_state_dict(ckpt["generator"])
    invg.load_state_dict(ckpt["invg"])
    discriminator.load_state_dict(ckpt["discriminator"])
    classifier.load_state_dict(ckpt["classifier"])

    generator.eval()
    invg.eval()
    discriminator.eval()
    classifier.eval()

    return generator, invg, discriminator, classifier


@torch.no_grad()
def run_gonet_batch(images, generator, invg, discriminator, classifier):
    z_hat = invg(images)
    img_gen = generator(z_hat)

    dis_real = discriminator(images)
    dis_gen = discriminator(img_gen)

    probs = classifier(
        img_error=images - img_gen,
        dis_error=dis_real - dis_gen,
        dis_real=dis_real,
    )

    return probs


def load_gonet_t(gonet_checkpoint, gonet_t_checkpoint, device, nz=100, use_tanh=False):
    gonet_t_ckpt = torch.load(gonet_t_checkpoint, map_location=device)
    ckpt_args = gonet_t_ckpt.get("args", {})

    reduced_dim = int(ckpt_args.get("reduced_dim", 10))
    hidden_dim = int(ckpt_args.get("hidden_dim", 64))
    num_layers = int(ckpt_args.get("num_layers", 1))
    dropout = float(ckpt_args.get("dropout", 0.0))
    bidirectional = bool(ckpt_args.get("bidirectional", False))

    generator, invg, discriminator, _ = load_gonet(
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

    model.feature_reducer.load_state_dict(gonet_t_ckpt["feature_reducer"])
    model.temporal_classifier.load_state_dict(gonet_t_ckpt["temporal_classifier"])

    model.freeze_gonet_backbone()
    model.eval()

    return model


@torch.no_grad()
def run_gonet_t_sequence(images, model):
    """
    images: [T, 3, 128, 128]
    returns probs: [T]
    """

    seq = images.unsqueeze(0)  # [1, T, 3, 128, 128]

    logits = model(
        seq,
        lengths=torch.tensor([images.shape[0]], device=images.device),
        return_logits=True,
    )

    probs = torch.sigmoid(logits)
    return probs.squeeze(0).squeeze(-1)


def benchmark_vanilla_gonet(
    images,
    generator,
    invg,
    discriminator,
    classifier,
    batch_size,
    warmup_iters,
    repeat_iters,
    device,
):
    n = images.shape[0]

    # Warmup
    for _ in range(warmup_iters):
        batch = images[:batch_size]
        _ = run_gonet_batch(batch, generator, invg, discriminator, classifier)

    sync_if_cuda(device)

    total_frames = 0
    start = time.perf_counter()

    for _ in range(repeat_iters):
        for s in range(0, n, batch_size):
            batch = images[s:s + batch_size]
            _ = run_gonet_batch(batch, generator, invg, discriminator, classifier)
            total_frames += batch.shape[0]

    sync_if_cuda(device)

    elapsed = time.perf_counter() - start

    fps = total_frames / elapsed
    ms_per_frame = 1000.0 / fps

    return elapsed, total_frames, fps, ms_per_frame


def benchmark_gonet_t(
    images,
    model,
    seq_len,
    warmup_iters,
    repeat_iters,
    device,
):
    n = images.shape[0]

    if n < seq_len:
        raise RuntimeError(f"Need at least {seq_len} frames, got {n}")

    # Make non-overlapping chunks.
    chunks = []
    for s in range(0, n - seq_len + 1, seq_len):
        chunks.append(images[s:s + seq_len])

    # Warmup
    for _ in range(warmup_iters):
        _ = run_gonet_t_sequence(chunks[0], model)

    sync_if_cuda(device)

    total_frames = 0
    start = time.perf_counter()

    for _ in range(repeat_iters):
        for chunk in chunks:
            _ = run_gonet_t_sequence(chunk, model)
            total_frames += chunk.shape[0]

    sync_if_cuda(device)

    elapsed = time.perf_counter() - start

    fps = total_frames / elapsed
    ms_per_frame = 1000.0 / fps

    return elapsed, total_frames, fps, ms_per_frame, len(chunks)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pseudo-csv",
        required=True,
        help="Pseudo-label CSV containing image paths.",
    )

    parser.add_argument(
        "--gonet-checkpoint",
        required=True,
        help="Vanilla GONet checkpoint, e.g. checkpoints/gonet_fl/fl_best.pt",
    )

    parser.add_argument(
        "--gonet-t-checkpoint",
        required=True,
        help="GONet+T checkpoint.",
    )

    parser.add_argument("--max-frames", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=64)

    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--repeat-iters", type=int, default=5)

    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--use-tanh", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    torch.backends.cudnn.benchmark = True

    print(f"Using device: {device}")
    print(f"Pseudo CSV: {args.pseudo_csv}")
    print(f"Max frames: {args.max_frames}")

    image_paths = collect_image_paths_from_pseudo_csv(
        pseudo_csv=args.pseudo_csv,
        max_frames=args.max_frames,
    )

    print(f"Loading {len(image_paths)} images into memory...")
    images = load_images_as_tensor(image_paths, device=device)

    print("Images tensor:", tuple(images.shape))

    generator, invg, discriminator, classifier = load_gonet(
        checkpoint_path=args.gonet_checkpoint,
        device=device,
        nz=args.nz,
        use_tanh=args.use_tanh,
    )

    gonet_t_model = load_gonet_t(
        gonet_checkpoint=args.gonet_checkpoint,
        gonet_t_checkpoint=args.gonet_t_checkpoint,
        device=device,
        nz=args.nz,
        use_tanh=args.use_tanh,
    )

    print()
    print("Benchmark 1: Vanilla GONet")
    print(f"Batch size: {args.batch_size}")

    elapsed, total_frames, fps, ms_per_frame = benchmark_vanilla_gonet(
        images=images,
        generator=generator,
        invg=invg,
        discriminator=discriminator,
        classifier=classifier,
        batch_size=args.batch_size,
        warmup_iters=args.warmup_iters,
        repeat_iters=args.repeat_iters,
        device=device,
    )

    print(f"Elapsed:        {elapsed:.4f} s")
    print(f"Frames:         {total_frames}")
    print(f"FPS:            {fps:.2f}")
    print(f"ms / frame:     {ms_per_frame:.3f}")

    print()
    print("Benchmark 2: GONet+T")
    print(f"Sequence length: {args.seq_len}")

    elapsed, total_frames, fps, ms_per_frame, num_chunks = benchmark_gonet_t(
        images=images,
        model=gonet_t_model,
        seq_len=args.seq_len,
        warmup_iters=args.warmup_iters,
        repeat_iters=args.repeat_iters,
        device=device,
    )

    print(f"Chunks:         {num_chunks}")
    print(f"Elapsed:        {elapsed:.4f} s")
    print(f"Frames:         {total_frames}")
    print(f"FPS:            {fps:.2f}")
    print(f"ms / frame:     {ms_per_frame:.3f}")


if __name__ == "__main__":
    main()