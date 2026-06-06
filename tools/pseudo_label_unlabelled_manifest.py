#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from datasets.go_stanford import preprocess_gonet_image
from models.gonet import Generator, InvG, Discriminator, GONetClassifier


def load_gonet(checkpoint_path: Path, device: str, nz: int = 100, use_tanh: bool = False):
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
def infer_batch(image_paths, generator, invg, discriminator, classifier, device):
    tensors = []

    valid_paths = []

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image_bgr is None:
            print(f"Warning: could not read image: {image_path}")
            continue

        image_tensor = preprocess_gonet_image(
            image_bgr=image_bgr,
            output_size=128,
            use_fisheye_mask=False,
            rotate_clockwise=False,
        )

        tensors.append(image_tensor)
        valid_paths.append(image_path)

    if len(tensors) == 0:
        return [], []

    images = torch.stack(tensors, dim=0).to(device)

    z_hat = invg(images)
    img_gen = generator(z_hat)

    dis_real = discriminator(images)
    dis_gen = discriminator(img_gen)

    probs = classifier(
        img_error=images - img_gen,
        dis_error=dis_real - dis_gen,
        dis_real=dis_real,
    )

    probs = probs.detach().cpu().view(-1).tolist()

    return valid_paths, probs


def read_manifest(manifest_path: Path):
    rows = []

    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


def write_rows(output_path: Path, rows, fieldnames):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to manifest CSV created by build_unlabelled_sequence_manifest.py",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained vanilla GONet checkpoint, e.g. checkpoints/gonet_fl/fl_best.pt",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output pseudo-label CSV",
    )

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--use-tanh", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    manifest_path = Path(args.manifest)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    print(f"Using device: {device}")
    print(f"Manifest:   {manifest_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output:     {output_path}")

    rows = read_manifest(manifest_path)

    print(f"Rows in manifest: {len(rows)}")

    generator, invg, discriminator, classifier = load_gonet(
        checkpoint_path=checkpoint_path,
        device=device,
        nz=args.nz,
        use_tanh=args.use_tanh,
    )

    output_rows = []

    for start in tqdm(range(0, len(rows), args.batch_size), desc="Pseudo-labelling"):
        batch_rows = rows[start:start + args.batch_size]
        batch_paths = [Path(row["path"]) for row in batch_rows]

        valid_paths, probs = infer_batch(
            image_paths=batch_paths,
            generator=generator,
            invg=invg,
            discriminator=discriminator,
            classifier=classifier,
            device=device,
        )

        path_to_prob = {
            str(path): prob
            for path, prob in zip(valid_paths, probs)
        }

        for row in batch_rows:
            path = row["path"]

            if path not in path_to_prob:
                continue

            out_row = dict(row)
            out_row["prob_traversable"] = path_to_prob[path]
            output_rows.append(out_row)

    fieldnames = list(output_rows[0].keys())

    write_rows(
        output_path=output_path,
        rows=output_rows,
        fieldnames=fieldnames,
    )

    print()
    print(f"Saved pseudo-labels: {output_path}")
    print(f"Input rows:  {len(rows)}")
    print(f"Output rows: {len(output_rows)}")


if __name__ == "__main__":
    main()