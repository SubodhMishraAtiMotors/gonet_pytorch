#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2

from datasets.go_stanford import (
    GOStanfordPositiveDataset,
    GOStanfordLabelledDataset,
    denormalize_gonet_tensor,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        required=True,
        help="Path to go_stanford_dataset",
    )

    parser.add_argument(
        "--dataset-type",
        default="positive",
        choices=["positive", "labelled"],
        help="Which dataset loader to inspect",
    )

    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "vali", "validation", "test"],
    )

    parser.add_argument(
        "--side",
        default="both",
        choices=["left", "right", "L", "R", "both"],
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/preprocessing_debug",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--no-fisheye-mask",
        action="store_true",
        help="Disable circular fisheye masking",
    )

    parser.add_argument(
        "--no-rotate",
        action="store_true",
        help="Disable 90 degree clockwise rotation",
    )

    parser.add_argument("--xc", type=int, default=310)
    parser.add_argument("--yc", type=int, default=321)
    parser.add_argument("--radius", type=int, default=275)
    parser.add_argument("--output-size", type=int, default=128)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_fisheye_mask = not args.no_fisheye_mask
    rotate_clockwise = not args.no_rotate

    if args.dataset_type == "positive":
        dataset = GOStanfordPositiveDataset(
            root=args.data_root,
            split=args.split,
            side=args.side,
            output_size=args.output_size,
            use_fisheye_mask=use_fisheye_mask,
            xc=args.xc,
            yc=args.yc,
            radius=args.radius,
            rotate_clockwise=rotate_clockwise,
        )
    else:
        dataset = GOStanfordLabelledDataset(
            root=args.data_root,
            split=args.split,
            side=args.side,
            output_size=args.output_size,
            use_fisheye_mask=use_fisheye_mask,
            xc=args.xc,
            yc=args.yc,
            radius=args.radius,
            rotate_clockwise=rotate_clockwise,
        )

    print(f"Loaded {len(dataset)} samples")

    n = min(args.num_samples, len(dataset))

    for idx in range(n):
        sample = dataset[idx]

        image_rgb = denormalize_gonet_tensor(sample["image"])
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        input_path = Path(sample["path"])

        if "label" in sample:
            label = int(sample["label"].item())
            label_name = "positive" if label == 1 else "negative"
            out_name = f"{idx:04d}_{label_name}_{input_path.name}"
        else:
            out_name = f"{idx:04d}_{input_path.name}"

        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), image_bgr)

        print(f"[{idx:04d}] {sample['path']} -> {out_path}")

    print()
    print(f"Saved preprocessed images to: {output_dir}")


if __name__ == "__main__":
    main()