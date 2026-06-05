#!/usr/bin/env python3

from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

from datasets.go_stanford import preprocess_gonet_image


class GOStanfordTemporalPseudoDataset(Dataset):
    """
    Variable-length temporal dataset for GONet+T.

    Reads pseudo-label CSV files with columns:
        global_index
        segment_id
        segment_local_index
        building_id
        frame_idx
        side
        filename
        path
        prob_traversable

    Each dataset item is one contiguous temporal segment.

    Output:
        {
            "images": [T, 3, 128, 128],
            "labels": [T, 1],
            "frame_indices": [T],
            "paths": list[str],
            "segment_id": int,
            "building_id": int,
            "side": str,
            "length": int,
        }
    """

    def __init__(
        self,
        pseudo_csvs: List[str],
        output_size: int = 128,
        min_length: int = 1,
        max_length: Optional[int] = None,
    ):
        self.pseudo_csvs = [Path(p) for p in pseudo_csvs]
        self.output_size = output_size
        self.min_length = min_length
        self.max_length = max_length

        dfs = []

        for csv_path in self.pseudo_csvs:
            if not csv_path.exists():
                raise FileNotFoundError(f"Pseudo-label CSV not found: {csv_path}")

            df = pd.read_csv(csv_path)

            required_cols = [
                "segment_id",
                "segment_local_index",
                "building_id",
                "frame_idx",
                "side",
                "path",
                "prob_traversable",
            ]

            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing column '{col}' in {csv_path}")

            # Preserve source file because segment IDs may collide across L/R CSVs.
            df["source_csv"] = str(csv_path)

            dfs.append(df)

        self.df = pd.concat(dfs, ignore_index=True)

        # Ensure types
        self.df["segment_id"] = self.df["segment_id"].astype(int)
        self.df["segment_local_index"] = self.df["segment_local_index"].astype(int)
        self.df["building_id"] = self.df["building_id"].astype(int)
        self.df["frame_idx"] = self.df["frame_idx"].astype(int)
        self.df["prob_traversable"] = self.df["prob_traversable"].astype(float)

        # Important:
        # segment_id may repeat between L and R CSVs because each manifest was built separately.
        # So the true unique sequence key should include source_csv + segment_id.
        grouped = self.df.groupby(["source_csv", "segment_id"], sort=False)

        self.segments = []

        for (source_csv, segment_id), sdf in grouped:
            sdf = sdf.sort_values("segment_local_index").reset_index(drop=True)

            length = len(sdf)

            if length < self.min_length:
                continue

            if self.max_length is not None and length > self.max_length:
                # Do not discard. Split into chunks of max_length.
                # This keeps every frame and avoids very long memory-heavy sequences.
                start = 0
                chunk_id = 0

                while start < length:
                    end = min(start + self.max_length, length)
                    chunk_df = sdf.iloc[start:end].copy().reset_index(drop=True)

                    self.segments.append(
                        {
                            "source_csv": source_csv,
                            "segment_id": int(segment_id),
                            "chunk_id": chunk_id,
                            "df": chunk_df,
                        }
                    )

                    start = end
                    chunk_id += 1
            else:
                self.segments.append(
                    {
                        "source_csv": source_csv,
                        "segment_id": int(segment_id),
                        "chunk_id": 0,
                        "df": sdf,
                    }
                )

        if len(self.segments) == 0:
            raise RuntimeError("No valid temporal segments found.")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        segment = self.segments[idx]
        sdf = segment["df"]

        images = []
        labels = []
        frame_indices = []
        paths = []

        for _, row in sdf.iterrows():
            path = Path(row["path"])
            image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)

            if image_bgr is None:
                raise RuntimeError(f"Could not read image: {path}")

            image_tensor = preprocess_gonet_image(
                image_bgr=image_bgr,
                output_size=self.output_size,
                use_fisheye_mask=False,
                rotate_clockwise=False,
            )

            label = float(row["prob_traversable"])

            images.append(image_tensor)
            labels.append([label])
            frame_indices.append(int(row["frame_idx"]))
            paths.append(str(path))

        images = torch.stack(images, dim=0)  # [T, 3, H, W]
        labels = torch.tensor(labels, dtype=torch.float32)  # [T, 1]
        frame_indices = torch.tensor(frame_indices, dtype=torch.long)  # [T]

        first = sdf.iloc[0]

        return {
            "images": images,
            "labels": labels,
            "frame_indices": frame_indices,
            "paths": paths,
            "segment_id": int(segment["segment_id"]),
            "chunk_id": int(segment["chunk_id"]),
            "building_id": int(first["building_id"]),
            "side": str(first["side"]),
            "length": int(images.shape[0]),
            "source_csv": str(segment["source_csv"]),
        }


def temporal_pseudo_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Pads variable-length sequences in a batch.

    Input batch:
        list of samples with images [T_i, 3, 128, 128]

    Output:
        images: [B, T_max, 3, 128, 128]
        labels: [B, T_max, 1]
        mask:   [B, T_max], True for valid frames, False for padding
        lengths: [B]
    """

    batch_size = len(batch)
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    max_len = int(lengths.max().item())

    c, h, w = batch[0]["images"].shape[1:]

    images = torch.zeros(batch_size, max_len, c, h, w, dtype=torch.float32)
    labels = torch.zeros(batch_size, max_len, 1, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    frame_indices = torch.full(
        (batch_size, max_len),
        fill_value=-1,
        dtype=torch.long,
    )

    paths = []
    segment_ids = []
    chunk_ids = []
    building_ids = []
    sides = []
    source_csvs = []

    for bidx, item in enumerate(batch):
        length = item["length"]

        images[bidx, :length] = item["images"]
        labels[bidx, :length] = item["labels"]
        mask[bidx, :length] = True
        frame_indices[bidx, :length] = item["frame_indices"]

        paths.append(item["paths"])
        segment_ids.append(item["segment_id"])
        chunk_ids.append(item["chunk_id"])
        building_ids.append(item["building_id"])
        sides.append(item["side"])
        source_csvs.append(item["source_csv"])

    return {
        "images": images,
        "labels": labels,
        "mask": mask,
        "lengths": lengths,
        "frame_indices": frame_indices,
        "paths": paths,
        "segment_ids": segment_ids,
        "chunk_ids": chunk_ids,
        "building_ids": building_ids,
        "sides": sides,
        "source_csvs": source_csvs,
    }


def summarize_temporal_dataset(dataset: GOStanfordTemporalPseudoDataset):
    lengths = [seg["df"].shape[0] for seg in dataset.segments]

    summary = {
        "num_segments": len(lengths),
        "num_frames": int(sum(lengths)),
        "min_length": int(min(lengths)),
        "max_length": int(max(lengths)),
        "mean_length": float(sum(lengths) / len(lengths)),
    }

    return summary