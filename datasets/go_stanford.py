#!/usr/bin/env python3

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]


def collect_images_from_dirs(dirs: List[Path]) -> List[Path]:
    image_paths = []

    for d in dirs:
        if not d.exists():
            raise FileNotFoundError(f"Directory does not exist: {d}")

        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(p)

    return image_paths


def build_circular_mask(
    height: int,
    width: int,
    xc: int,
    yc: int,
    radius: int,
) -> np.ndarray:
    """
    Returns a mask of shape [H, W].
    Valid circular fisheye region = 1.
    Invalid outside region = 0.
    """
    yy, xx = np.ogrid[:height, :width]
    dist_sq = (xx - xc) ** 2 + (yy - yc) ** 2
    mask = dist_sq <= radius ** 2
    return mask.astype(np.uint8)


def preprocess_gonet_image(
    image_bgr: np.ndarray,
    output_size: int = 128,
    use_fisheye_mask: bool = False,
    xc: int = None,
    yc: int = None,
    radius: int = None,
    rotate_clockwise: bool = False,
) -> torch.Tensor:
    """
    Preprocess GO Stanford image for GONet.

    Important:
        The downloaded GO Stanford dataset images are already 128x128.
        The original ROS code used crop/mask parameters for raw camera images,
        not for this released dataset.

    Therefore, default behavior for the downloaded dataset is:
        1. BGR -> RGB
        2. optional resize to 128x128
        3. normalize [0, 255] -> [-1, 1]
        4. HWC -> CHW
    """

    if image_bgr is None:
        raise ValueError("Input image is None.")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    h, w = image_rgb.shape[:2]

    # The released GO Stanford images are already 128x128.
    # Resize only if needed.
    if h != output_size or w != output_size:
        image_rgb = cv2.resize(
            image_rgb,
            (output_size, output_size),
            interpolation=cv2.INTER_AREA,
        )

    image_rgb = image_rgb.astype(np.float32)

    # Original GONet normalization:
    # pixel [0, 255] -> approximately [-1, 1]
    image_rgb = (image_rgb - 128.0) / 128.0

    # HWC -> CHW
    image_chw = np.transpose(image_rgb, (2, 0, 1))

    return torch.from_numpy(image_chw).float()


class GOStanfordPositiveDataset(Dataset):
    """
    Dataset for GAN training and inverse-generator training.

    Loads only positive traversable images from:

        whole_dataset/data_train/positive_L
        whole_dataset/data_train/positive_R

    or corresponding val/test splits.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        side: str = "both",
        output_size: int = 128,
        use_fisheye_mask: bool = False,
        xc: int = None,
        yc: int = None,
        radius: int = None,
        rotate_clockwise: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.side = side
        self.output_size = output_size
        self.use_fisheye_mask = use_fisheye_mask
        self.xc = xc
        self.yc = yc
        self.radius = radius
        self.rotate_clockwise = rotate_clockwise

        split_name_map = {
            "train": "data_train",
            "val": "data_vali",
            "vali": "data_vali",
            "validation": "data_vali",
            "test": "data_test",
        }

        if split not in split_name_map:
            raise ValueError(f"Unknown split: {split}")

        split_name = split_name_map[split]
        base_dir = self.root / "whole_dataset" / split_name

        dirs = []

        if side in ["left", "L", "both"]:
            dirs.append(base_dir / "positive_L")

        if side in ["right", "R", "both"]:
            dirs.append(base_dir / "positive_R")

        self.image_paths = collect_images_from_dirs(dirs)

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No positive images found in: {dirs}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        image_tensor = preprocess_gonet_image(
            image_bgr=image_bgr,
            output_size=self.output_size,
            use_fisheye_mask=self.use_fisheye_mask,
            xc=self.xc,
            yc=self.yc,
            radius=self.radius,
            rotate_clockwise=self.rotate_clockwise,
        )

        return {
            "image": image_tensor,
            "path": str(image_path),
        }


class GOStanfordLabelledDataset(Dataset):
    """
    Dataset for final GONet classification-layer training.

    Loads hand-labelled data:

        positive_L -> label 1
        positive_R -> label 1
        negative_L -> label 0
        negative_R -> label 0
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        side: str = "both",
        output_size: int = 128,
        use_fisheye_mask: bool = False,
        xc: int = None,
        yc: int = None,
        radius: int = None,
        rotate_clockwise: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.side = side
        self.output_size = output_size
        self.use_fisheye_mask = use_fisheye_mask
        self.xc = xc
        self.yc = yc
        self.radius = radius
        self.rotate_clockwise = rotate_clockwise

        split_name_map = {
            "train": "data_train_annotation",
            "val": "data_vali_annotation",
            "vali": "data_vali_annotation",
            "validation": "data_vali_annotation",
            "test": "data_test_annotation",
        }

        if split not in split_name_map:
            raise ValueError(f"Unknown split: {split}")

        split_name = split_name_map[split]
        base_dir = self.root / "hand_labelled_dataset" / split_name

        labelled_dirs: List[Tuple[Path, int]] = []

        if side in ["left", "L", "both"]:
            labelled_dirs.append((base_dir / "positive_L", 1))
            labelled_dirs.append((base_dir / "negative_L", 0))

        if side in ["right", "R", "both"]:
            labelled_dirs.append((base_dir / "positive_R", 1))
            labelled_dirs.append((base_dir / "negative_R", 0))

        samples = []

        for d, label in labelled_dirs:
            paths = collect_images_from_dirs([d])
            for p in paths:
                samples.append((p, label))

        self.samples = samples

        if len(self.samples) == 0:
            raise RuntimeError(f"No labelled images found in: {labelled_dirs}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        image_tensor = preprocess_gonet_image(
            image_bgr=image_bgr,
            output_size=self.output_size,
            use_fisheye_mask=self.use_fisheye_mask,
            xc=self.xc,
            yc=self.yc,
            radius=self.radius,
            rotate_clockwise=self.rotate_clockwise,
        )

        label_tensor = torch.tensor([label], dtype=torch.float32)

        return {
            "image": image_tensor,
            "label": label_tensor,
            "path": str(image_path),
        }


def denormalize_gonet_tensor(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Converts normalized tensor back to uint8 RGB image.

    Input:
        Tensor [3, H, W], approximately in [-1, 1]

    Output:
        RGB image [H, W, 3], uint8
    """
    image = image_tensor.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = image * 128.0 + 128.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image