#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.go_stanford_temporal import (
    GOStanfordTemporalPseudoDataset,
    temporal_pseudo_collate_fn,
    summarize_temporal_dataset,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pseudo-csv",
        nargs="+",
        required=True,
        help="One or more pseudo-label CSVs.",
    )

    parser.add_argument("--batch-size", type=int, default=4)

    parser.add_argument(
        "--min-length",
        type=int,
        default=1,
        help="Keep segments with at least this many frames.",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional chunk length for very long segments. Does not discard frames.",
    )

    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()

    dataset = GOStanfordTemporalPseudoDataset(
        pseudo_csvs=args.pseudo_csv,
        min_length=args.min_length,
        max_length=args.max_length,
    )

    summary = summarize_temporal_dataset(dataset)

    print("Dataset summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print()
    print("First 10 segments:")
    for idx in range(min(10, len(dataset))):
        seg = dataset.segments[idx]
        sdf = seg["df"]

        first = sdf.iloc[0]
        last = sdf.iloc[-1]

        print(
            f"idx={idx:04d} "
            f"source={Path(seg['source_csv']).name} "
            f"segment={seg['segment_id']} "
            f"chunk={seg['chunk_id']} "
            f"build={int(first['building_id'])} "
            f"side={first['side']} "
            f"frames={int(first['frame_idx'])}->{int(last['frame_idx'])} "
            f"len={len(sdf)}"
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=temporal_pseudo_collate_fn,
    )

    batch = next(iter(loader))

    print()
    print("One padded batch:")
    print("  images:", batch["images"].shape)
    print("  labels:", batch["labels"].shape)
    print("  mask:", batch["mask"].shape)
    print("  lengths:", batch["lengths"])
    print("  frame_indices:", batch["frame_indices"].shape)

    valid_counts = batch["mask"].sum(dim=1)

    print("  valid frame counts from mask:", valid_counts)

    assert torch.all(valid_counts.cpu() == batch["lengths"].cpu())

    print()
    print("Batch looks good.")


if __name__ == "__main__":
    main()