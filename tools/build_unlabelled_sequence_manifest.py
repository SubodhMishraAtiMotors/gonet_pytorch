#!/usr/bin/env python3

import argparse
import csv
import re
from pathlib import Path
from collections import defaultdict


PATTERN = re.compile(r"img_build(\d+)_(\d+)_([LR])")


def parse_filename(path: Path):
    """
    Parses:
        img_build10_1000_L.jpg

    Returns:
        building_id, frame_idx, side
    """
    match = PATTERN.match(path.stem)

    if match is None:
        raise ValueError(f"Filename does not match expected pattern: {path.name}")

    building_id = int(match.group(1))
    frame_idx = int(match.group(2))
    side = match.group(3)

    return building_id, frame_idx, side


def collect_images(folder: Path):
    image_paths = []

    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(folder.glob(ext))

    parsed = []

    for path in image_paths:
        building_id, frame_idx, side = parse_filename(path)
        parsed.append({
            "path": path,
            "filename": path.name,
            "building_id": building_id,
            "frame_idx": frame_idx,
            "side": side,
        })

    parsed = sorted(
        parsed,
        key=lambda x: (x["building_id"], x["side"], x["frame_idx"]),
    )

    return parsed


def assign_segments(parsed_images):
    """
    Splits frames into contiguous temporal segments.

    A new segment starts when:
        - building changes
        - side changes
        - frame_idx is not previous_frame_idx + 1

    No frame is discarded.
    """

    output_rows = []

    current_segment_id = -1
    previous_key = None
    previous_frame_idx = None
    segment_local_index = 0

    for item in parsed_images:
        key = (item["building_id"], item["side"])
        frame_idx = item["frame_idx"]

        start_new_segment = False

        if previous_key is None:
            start_new_segment = True
        elif key != previous_key:
            start_new_segment = True
        elif frame_idx != previous_frame_idx + 1:
            start_new_segment = True

        if start_new_segment:
            current_segment_id += 1
            segment_local_index = 0
        else:
            segment_local_index += 1

        row = {
            "global_index": len(output_rows),
            "segment_id": current_segment_id,
            "segment_local_index": segment_local_index,
            "building_id": item["building_id"],
            "frame_idx": item["frame_idx"],
            "side": item["side"],
            "filename": item["filename"],
            "path": str(item["path"]),
        }

        output_rows.append(row)

        previous_key = key
        previous_frame_idx = frame_idx

    return output_rows


def compute_segment_stats(rows):
    segment_to_rows = defaultdict(list)

    for row in rows:
        segment_to_rows[row["segment_id"]].append(row)

    stats = []

    for segment_id, segment_rows in segment_to_rows.items():
        first = segment_rows[0]
        last = segment_rows[-1]

        stats.append({
            "segment_id": segment_id,
            "building_id": first["building_id"],
            "side": first["side"],
            "start_frame_idx": first["frame_idx"],
            "end_frame_idx": last["frame_idx"],
            "num_frames": len(segment_rows),
        })

    stats = sorted(stats, key=lambda x: x["segment_id"])

    return stats


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        required=True,
        help="Path to go_stanford_dataset",
    )

    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "vali", "validation", "test"],
    )

    parser.add_argument(
        "--side",
        default="L",
        choices=["L", "R"],
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/gonet_t_manifests",
    )

    args = parser.parse_args()

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
        raise FileNotFoundError(f"Folder does not exist: {image_folder}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Image folder: {image_folder}")

    parsed_images = collect_images(image_folder)
    rows = assign_segments(parsed_images)
    stats = compute_segment_stats(rows)

    manifest_path = output_dir / f"{args.split}_unlabel_{args.side}_manifest.csv"
    stats_path = output_dir / f"{args.split}_unlabel_{args.side}_segments.csv"

    write_csv(
        manifest_path,
        rows,
        fieldnames=[
            "global_index",
            "segment_id",
            "segment_local_index",
            "building_id",
            "frame_idx",
            "side",
            "filename",
            "path",
        ],
    )

    write_csv(
        stats_path,
        stats,
        fieldnames=[
            "segment_id",
            "building_id",
            "side",
            "start_frame_idx",
            "end_frame_idx",
            "num_frames",
        ],
    )

    num_frames = len(rows)
    num_segments = len(stats)

    lengths = [s["num_frames"] for s in stats]

    print()
    print(f"Frames:   {num_frames}")
    print(f"Segments: {num_segments}")

    if lengths:
        print(f"Min segment length: {min(lengths)}")
        print(f"Max segment length: {max(lengths)}")
        print(f"Avg segment length: {sum(lengths) / len(lengths):.2f}")

        print()
        print("Top 20 longest segments:")
        for s in sorted(stats, key=lambda x: x["num_frames"], reverse=True)[:20]:
            print(
                f"segment={s['segment_id']:05d} "
                f"build={s['building_id']} "
                f"side={s['side']} "
                f"frames={s['start_frame_idx']}->{s['end_frame_idx']} "
                f"len={s['num_frames']}"
            )

    print()
    print(f"Saved manifest: {manifest_path}")
    print(f"Saved segments: {stats_path}")


if __name__ == "__main__":
    main()