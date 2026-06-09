#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

import cv2
import pandas as pd
import torch

from datasets.go_stanford import preprocess_gonet_image
from models.gonet import Generator, InvG, Discriminator, GONetClassifier
from models.gonet_temporal import (
    GONetTemporalFeatureReducer,
    GONetTemporalClassifier,
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


def load_gonet_t_modules(gonet_checkpoint, gonet_t_checkpoint, device, nz=100, use_tanh=False):
    gonet_t_ckpt = torch.load(gonet_t_checkpoint, map_location=device)
    ckpt_args = gonet_t_ckpt.get("args", {})

    reduced_dim = int(ckpt_args.get("reduced_dim", 10))
    hidden_dim = int(ckpt_args.get("hidden_dim", 64))
    num_layers = int(ckpt_args.get("num_layers", 1))
    dropout = float(ckpt_args.get("dropout", 0.0))
    bidirectional = bool(ckpt_args.get("bidirectional", False))

    if bidirectional:
        raise RuntimeError(
            "Stateful online inference is not meaningful for bidirectional LSTM, "
            "because it needs future frames."
        )

    generator, invg, discriminator, classifier = load_gonet(
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

    feature_reducer.load_state_dict(gonet_t_ckpt["feature_reducer"])
    temporal_classifier.load_state_dict(gonet_t_ckpt["temporal_classifier"])

    feature_reducer.eval()
    temporal_classifier.eval()

    for module in [generator, invg, discriminator, classifier, feature_reducer, temporal_classifier]:
        for p in module.parameters():
            p.requires_grad = False

    config = {
        "reduced_dim": reduced_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "bidirectional": bidirectional,
        "target_mode": ckpt_args.get("target_mode", "unknown"),
    }

    return generator, invg, discriminator, classifier, feature_reducer, temporal_classifier, config


@torch.no_grad()
def run_vanilla_gonet_one_frame(image, generator, invg, discriminator, classifier):
    """
    image: [1, 3, 128, 128]
    """

    z_hat = invg(image)
    img_gen = generator(z_hat)

    dis_real = discriminator(image)
    dis_gen = discriminator(img_gen)

    prob = classifier(
        img_error=image - img_gen,
        dis_error=dis_real - dis_gen,
        dis_real=dis_real,
    )

    return prob


@torch.no_grad()
def extract_gonet_t_feature_one_frame(image, generator, invg, discriminator, feature_reducer):
    """
    image: [1, 3, 128, 128]

    returns:
        feature: [1, 1, 30]
    """

    z_hat = invg(image)
    img_gen = generator(z_hat)

    dis_real = discriminator(image)
    dis_gen = discriminator(img_gen)

    img_error = image - img_gen
    dis_error = dis_real - dis_gen

    feature = feature_reducer(
        img_error=img_error,
        dis_error=dis_error,
        dis_real=dis_real,
    )  # [1, 30]

    feature = feature.unsqueeze(1)  # [1, 1, 30]
    return feature


@torch.no_grad()
def run_stateful_lstm_one_step(feature, temporal_classifier, hidden_state):
    """
    feature: [1, 1, 30]
    hidden_state: None or (h, c)

    returns:
        prob: [1, 1, 1]
        new_hidden_state
    """

    output, new_hidden_state = temporal_classifier.lstm(feature, hidden_state)
    logits = temporal_classifier.output(output)
    prob = torch.sigmoid(logits)

    return prob, new_hidden_state


def benchmark_vanilla_online(images, generator, invg, discriminator, classifier, warmup_iters, repeat_iters, device):
    n = images.shape[0]

    # Warmup
    for _ in range(warmup_iters):
        image = images[0:1]
        _ = run_vanilla_gonet_one_frame(image, generator, invg, discriminator, classifier)

    sync_if_cuda(device)

    total_frames = 0
    start = time.perf_counter()

    for _ in range(repeat_iters):
        for i in range(n):
            image = images[i:i + 1]
            _ = run_vanilla_gonet_one_frame(image, generator, invg, discriminator, classifier)
            total_frames += 1

    sync_if_cuda(device)

    elapsed = time.perf_counter() - start
    fps = total_frames / elapsed
    ms_per_frame = 1000.0 / fps

    return elapsed, total_frames, fps, ms_per_frame


def benchmark_stateful_gonet_t_online(
    images,
    generator,
    invg,
    discriminator,
    feature_reducer,
    temporal_classifier,
    warmup_iters,
    repeat_iters,
    reset_every,
    device,
):
    n = images.shape[0]

    # Warmup
    hidden_state = None
    for i in range(warmup_iters):
        image = images[i % n:i % n + 1]
        feature = extract_gonet_t_feature_one_frame(
            image=image,
            generator=generator,
            invg=invg,
            discriminator=discriminator,
            feature_reducer=feature_reducer,
        )
        _, hidden_state = run_stateful_lstm_one_step(
            feature=feature,
            temporal_classifier=temporal_classifier,
            hidden_state=hidden_state,
        )

    sync_if_cuda(device)

    total_frames = 0
    start = time.perf_counter()

    for _ in range(repeat_iters):
        hidden_state = None

        for i in range(n):
            if reset_every > 0 and i % reset_every == 0:
                hidden_state = None

            image = images[i:i + 1]

            feature = extract_gonet_t_feature_one_frame(
                image=image,
                generator=generator,
                invg=invg,
                discriminator=discriminator,
                feature_reducer=feature_reducer,
            )

            _, hidden_state = run_stateful_lstm_one_step(
                feature=feature,
                temporal_classifier=temporal_classifier,
                hidden_state=hidden_state,
            )

            total_frames += 1

    sync_if_cuda(device)

    elapsed = time.perf_counter() - start
    fps = total_frames / elapsed
    ms_per_frame = 1000.0 / fps

    return elapsed, total_frames, fps, ms_per_frame


def benchmark_lstm_only(
    feature_dim,
    temporal_classifier,
    n_frames,
    warmup_iters,
    repeat_iters,
    reset_every,
    device,
):
    features = torch.randn(n_frames, 1, 1, feature_dim, device=device)

    hidden_state = None

    # Warmup
    for i in range(warmup_iters):
        feature = features[i % n_frames]
        _, hidden_state = run_stateful_lstm_one_step(
            feature=feature,
            temporal_classifier=temporal_classifier,
            hidden_state=hidden_state,
        )

    sync_if_cuda(device)

    total_frames = 0
    start = time.perf_counter()

    for _ in range(repeat_iters):
        hidden_state = None

        for i in range(n_frames):
            if reset_every > 0 and i % reset_every == 0:
                hidden_state = None

            feature = features[i]
            _, hidden_state = run_stateful_lstm_one_step(
                feature=feature,
                temporal_classifier=temporal_classifier,
                hidden_state=hidden_state,
            )

            total_frames += 1

    sync_if_cuda(device)

    elapsed = time.perf_counter() - start
    fps = total_frames / elapsed
    ms_per_frame = 1000.0 / fps

    return elapsed, total_frames, fps, ms_per_frame


def print_result(name, elapsed, total_frames, fps, ms_per_frame):
    print()
    print(name)
    print(f"Elapsed:        {elapsed:.4f} s")
    print(f"Frames:         {total_frames}")
    print(f"FPS:            {fps:.2f}")
    print(f"ms / frame:     {ms_per_frame:.3f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pseudo-csv", required=True)
    parser.add_argument("--gonet-checkpoint", required=True)
    parser.add_argument("--gonet-t-checkpoint", required=True)

    parser.add_argument("--max-frames", type=int, default=1024)
    parser.add_argument("--warmup-iters", type=int, default=50)
    parser.add_argument("--repeat-iters", type=int, default=5)

    parser.add_argument(
        "--reset-every",
        type=int,
        default=0,
        help=(
            "Reset LSTM hidden state every N frames. "
            "Use 0 to never reset within a repeat. "
            "For segmented online use, set this to expected segment length."
        ),
    )

    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--use-tanh", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
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

    (
        generator,
        invg,
        discriminator,
        classifier,
        feature_reducer,
        temporal_classifier,
        config,
    ) = load_gonet_t_modules(
        gonet_checkpoint=args.gonet_checkpoint,
        gonet_t_checkpoint=args.gonet_t_checkpoint,
        device=device,
        nz=args.nz,
        use_tanh=args.use_tanh,
    )

    print()
    print("Loaded GONet+T config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    elapsed, total_frames, fps, ms_per_frame = benchmark_vanilla_online(
        images=images,
        generator=generator,
        invg=invg,
        discriminator=discriminator,
        classifier=classifier,
        warmup_iters=args.warmup_iters,
        repeat_iters=args.repeat_iters,
        device=device,
    )

    print_result(
        "Benchmark 1: Vanilla GONet online, one frame at a time",
        elapsed,
        total_frames,
        fps,
        ms_per_frame,
    )

    elapsed, total_frames, fps, ms_per_frame = benchmark_stateful_gonet_t_online(
        images=images,
        generator=generator,
        invg=invg,
        discriminator=discriminator,
        feature_reducer=feature_reducer,
        temporal_classifier=temporal_classifier,
        warmup_iters=args.warmup_iters,
        repeat_iters=args.repeat_iters,
        reset_every=args.reset_every,
        device=device,
    )

    print_result(
        "Benchmark 2: Stateful GONet+T online, one frame at a time",
        elapsed,
        total_frames,
        fps,
        ms_per_frame,
    )

    feature_dim = int(config["reduced_dim"]) * 3

    elapsed, total_frames, fps, ms_per_frame = benchmark_lstm_only(
        feature_dim=feature_dim,
        temporal_classifier=temporal_classifier,
        n_frames=images.shape[0],
        warmup_iters=args.warmup_iters,
        repeat_iters=args.repeat_iters,
        reset_every=args.reset_every,
        device=device,
    )

    print_result(
        "Benchmark 3: LSTM only, one feature timestep at a time",
        elapsed,
        total_frames,
        fps,
        ms_per_frame,
    )

    print()
    print("Interpretation:")
    print("- Benchmark 1 is vanilla GONet online latency.")
    print("- Benchmark 2 is realistic online GONet+T latency with stateful LSTM.")
    print("- Benchmark 3 isolates the LSTM cost only.")
    print("- Difference between Benchmark 2 and Benchmark 1 is the approximate extra cost of feature reduction + LSTM.")


if __name__ == "__main__":
    main()