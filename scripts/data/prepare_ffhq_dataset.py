#!/usr/bin/env python3
"""Prepare canonical FFHQ images at an EDM teacher resolution with LANCZOS."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image

from pace.dataset_specs import FFHQ_PROTOCOL
from pace.image_datasets import (
    SharedImageDataset,
    build_dataset_manifest,
    preflight_dataset,
    write_dataset_manifest,
)


_thread_state = threading.local()


def resize_ffhq_image(image: Image.Image, resolution: int) -> Image.Image:
    if image.mode != "RGB":
        raise ValueError(f"FFHQ source must be RGB, got {image.mode}")
    if image.size != (256, 256):
        raise ValueError(f"FFHQ source must be exactly 256x256, got {image.size}")
    return image.resize((resolution, resolution), resample=Image.Resampling.LANCZOS)


def _read_source_image(source: Path, source_kind: str, relative_path: str) -> Image.Image:
    if source_kind == "directory":
        with Image.open(source / relative_path) as image:
            image.load()
            return image.copy()
    archive_key = str(source.resolve())
    archives = getattr(_thread_state, "archives", None)
    if archives is None:
        archives = {}
        _thread_state.archives = archives
    archive = archives.get(archive_key)
    if archive is None:
        archive = zipfile.ZipFile(source)
        archives[archive_key] = archive
    with archive.open(relative_path) as handle:
        with Image.open(io.BytesIO(handle.read())) as image:
            image.load()
            return image.copy()


def prepare_ffhq_dataset(
    *,
    source: str | Path,
    output_dir: str | Path,
    resolution: int,
    workers: int = 4,
    overwrite: bool = False,
    resume: bool = True,
    source_preflight: bool = False,
    output_preflight: bool = True,
) -> dict[str, Any]:
    source = Path(source).expanduser()
    output_dir = Path(output_dir).expanduser()
    if not 1 <= resolution <= 256:
        raise ValueError(f"FFHQ preparation resolution must be in [1, 256], got {resolution}")
    if workers < 0:
        raise ValueError(f"workers must be non-negative, got {workers}")
    if source.resolve() == output_dir.resolve():
        raise ValueError("Source and output directory must differ")

    source_dataset = SharedImageDataset(
        dataset_id="ffhq",
        root=source,
        image_size=256,
        split="all",
        ffhq_protocol=FFHQ_PROTOCOL,
    )
    if source_preflight:
        preflight_dataset(source_dataset, full_decode=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_images = [path for path in output_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}]
    if existing_images and not overwrite and not resume:
        raise FileExistsError(
            f"Output contains {len(existing_images):,} image files; use the default --resume mode or pass --overwrite: {output_dir}"
        )

    def valid_existing(path: Path) -> bool:
        try:
            with Image.open(path) as image:
                return image.format == "PNG" and image.mode == "RGB" and image.size == (resolution, resolution)
        except (OSError, ValueError):
            return False

    def prepare_one(item: tuple[int, str]) -> tuple[str, str]:
        image_id, relative_path = item
        destination = output_dir / f"{image_id:05d}.png"
        if not overwrite and destination.is_file() and valid_existing(destination):
            return destination.name, "skipped"
        if destination.exists() and not (overwrite or resume):
            raise FileExistsError(f"Existing canonical output is invalid; use --resume or --overwrite to repair: {destination}")
        image = _read_source_image(source, source_dataset.source_kind, relative_path)
        prepared = resize_ffhq_image(image, resolution)
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}-{threading.get_ident()}")
        prepared.save(temporary, format="PNG", compress_level=6)
        os.replace(temporary, destination)
        return destination.name, "written"

    work = list(enumerate(source_dataset.paths))
    if workers in {0, 1}:
        outcomes = [prepare_one(item) for item in work]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            outcomes = list(executor.map(prepare_one, work))

    manifest = build_dataset_manifest(dataset_id="ffhq", root=output_dir, protocol=FFHQ_PROTOCOL)
    manifest["preparation"] = {
        "operation": "square_resize",
        "resampling": "PIL.Image.Resampling.LANCZOS",
        "source": str(source.resolve()),
        "source_kind": source_dataset.source_kind,
        "source_resolution": 256,
        "target_resolution": int(resolution),
        "source_listing_sha256": source_dataset.metadata["source_listing_sha256"],
        "source_records_sha256": source_dataset.metadata["source_records_sha256"],
    }
    manifest_path = output_dir / "dataset_manifest.json"
    write_dataset_manifest(manifest_path, manifest)
    prepared_dataset = SharedImageDataset(
        dataset_id="ffhq",
        root=output_dir,
        image_size=resolution,
        split="all",
        manifest=manifest_path,
    )
    report = preflight_dataset(prepared_dataset, full_decode=True) if output_preflight else prepared_dataset.metadata
    if output_preflight:
        manifest["preflight"] = {
            "valid": True,
            "checked": report["checked"],
            "content_sha256": report["content_sha256"],
            "modes": report["modes"],
            "sizes": report["sizes"],
        }
        write_dataset_manifest(manifest_path, manifest)
    return {
        "output_dir": str(output_dir.resolve()),
        "dataset_manifest": str(manifest_path.resolve()),
        "prepared_count": len(outcomes),
        "written_count": sum(status == "written" for _name, status in outcomes),
        "skipped_valid_count": sum(status == "skipped" for _name, status in outcomes),
        "resolution": int(resolution),
        "resampling": "PIL.Image.Resampling.LANCZOS",
        "preflight": report,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", required=True, help="Canonical flat/recursive FFHQ256 directory or ZIP.")
    parser.add_argument("--output-dir", required=True, help="Destination for canonical numeric PNGs and manifest.")
    parser.add_argument("--resolution", type=int, default=64, help="EDM teacher resolution; only downsampling is allowed.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel decoder/resizer threads; 0 selects sequential work.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate every canonical output PNG, including valid files.")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Refuse a populated destination instead of skipping valid outputs and repairing missing/invalid files.",
    )
    parser.add_argument("--source-preflight", action="store_true", help="Decode all source images before preparation (normally redundant).")
    parser.add_argument("--skip-output-preflight", action="store_true", help="Skip the full prepared-output decode pass.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = prepare_ffhq_dataset(
        source=args.source,
        output_dir=args.output_dir,
        resolution=args.resolution,
        workers=args.workers,
        overwrite=args.overwrite,
        resume=not args.no_resume,
        source_preflight=args.source_preflight,
        output_preflight=not args.skip_output_preflight,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
