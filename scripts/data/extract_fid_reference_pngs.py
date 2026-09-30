#!/usr/bin/env python3
"""
One-shot utility: dump N images from an ``ImageNet1KParquetDataset`` or
``CIFAR10Dataset`` as 256x256 (or 32x32) PNGs to a directory, for use as
the real-image reference set for FID computation.
"""

import argparse
import os
import sys
from tqdm.auto import tqdm
import torch
from PIL import Image as PILImage

# The dataset classes live in scripts/evaluate_parameters_edm.py (one level up).
_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from evaluate_parameters_edm import CIFAR10Dataset, ImageNet1KParquetDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["cifar10", "imagenet1k_parquet"])
    p.add_argument("--data_root", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_images", type=int, default=10000)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--parquet_split_prefix", default="validation-")
    p.add_argument("--cifar_split", default="test", choices=["train", "validation", "test"])
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset == "cifar10":
        ds = CIFAR10Dataset(
            root=args.data_root, image_size=args.image_size,
            split=args.cifar_split, max_images=args.num_images, download=False,
        )
    else:
        ds = ImageNet1KParquetDataset(
            root=args.data_root, image_size=args.image_size,
            max_images=args.num_images, split_prefix=args.parquet_split_prefix,
        )
    print(f"dataset: {len(ds)} images at {args.image_size}x{args.image_size}")

    for i in tqdm(range(len(ds)), desc="writing PNGs"):
        img_t, _label = ds[i]
        # img_t is in [-1, 1]; convert to uint8 [0, 255]
        arr = ((img_t.clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8).cpu().numpy().transpose(1, 2, 0)
        PILImage.fromarray(arr).save(os.path.join(args.output_dir, f"ref_{i:06d}.png"))

    n = len(os.listdir(args.output_dir))
    print(f"saved {n} PNGs to {args.output_dir}")


if __name__ == "__main__":
    main()
