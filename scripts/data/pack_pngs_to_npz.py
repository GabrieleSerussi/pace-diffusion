#!/usr/bin/env python3
"""Pack a directory of sampled PNGs into the NPZ layout of the ADM evaluator.

The OpenAI guided-diffusion evaluator (``evaluations/evaluator.py``) reads
sample batches as ``arr_0`` with shape ``[N, H, W, 3]`` and dtype ``uint8``.
``scripts/eval_composite_curve.py`` calls this script on the PNGs written by
``scripts/evaluate_students.py``::

    python scripts/data/pack_pngs_to_npz.py --png_dir <samples> --out samples.npz --size 256

Files are read in sorted name order (``img_000000.png``, ...).  Images whose
side differs from ``--size`` are resized with bicubic resampling and a warning
is printed; the DiT samplers already write images at the evaluation size, so no
resizing happens for the paper's CIFAR-10 (32) and latent (256) DiTs.

This packer re-creates a helper that the original experiments kept next to the
evaluator; it is not part of guided-diffusion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def load_png(path: Path, size: int) -> tuple[np.ndarray, bool]:
    """Return ``(uint8 HWC RGB array, resized)`` for one PNG."""

    with Image.open(path) as image:
        image = image.convert("RGB")
        resized = image.size != (size, size)
        if resized:
            image = image.resize((size, size), Image.BICUBIC)
        return np.asarray(image, dtype=np.uint8), resized


def pack_pngs(png_dir: Path, out: Path, size: int, max_images: int | None = None) -> int:
    paths = sorted(path for path in png_dir.iterdir() if path.suffix.lower() == ".png")
    if max_images is not None:
        paths = paths[:max_images]
    if not paths:
        raise SystemExit(f"no PNG files found in {png_dir}")
    batch = np.empty((len(paths), size, size, 3), dtype=np.uint8)
    num_resized = 0
    for index, path in enumerate(paths):
        batch[index], resized = load_png(path, size)
        num_resized += int(resized)
    if num_resized:
        print(
            f"warning: resized {num_resized} of {len(paths)} images to {size}x{size} (bicubic)",
            file=sys.stderr,
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, arr_0=batch)
    return len(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--png_dir", type=Path, required=True, help="Directory of sampled PNG files.")
    parser.add_argument("--out", type=Path, required=True, help="Output .npz path (arr_0, NHWC uint8).")
    parser.add_argument("--size", type=int, required=True, help="Side length of the packed images.")
    parser.add_argument("--max_images", type=int, default=None, help="Pack only the first N files.")
    args = parser.parse_args()
    if args.size <= 0:
        parser.error("--size must be positive")
    count = pack_pngs(args.png_dir, args.out, args.size, args.max_images)
    print(f"packed {count} images into {args.out}")


if __name__ == "__main__":
    main()
