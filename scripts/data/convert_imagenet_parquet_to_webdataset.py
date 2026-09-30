#!/usr/bin/env python3
"""Convert ImageNet parquet files into sequential WebDataset tar shards."""

from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
import time
from pathlib import Path
from typing import Any

from PIL import Image


def _infer_first_present(candidates: list[str], available: list[str]) -> str | None:
    available_set = set(available)
    for name in candidates:
        if name in available_set:
            return name
    return None


def _decode_parquet_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(value))).convert("RGB")
    if hasattr(value, "as_py"):
        return _decode_parquet_image(value.as_py())
    if hasattr(value, "shape"):
        return Image.fromarray(value).convert("RGB")
    raise TypeError(f"Unsupported parquet image value type: {type(value).__name__}")


def _encoded_jpeg_bytes(value: Any) -> bytes | None:
    if isinstance(value, dict):
        payload = value.get("bytes")
        if payload is None:
            return None
        return _encoded_jpeg_bytes(payload)
    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
        if payload.startswith(b"\xff\xd8\xff"):
            return payload
    if hasattr(value, "as_py"):
        return _encoded_jpeg_bytes(value.as_py())
    return None


def _image_to_jpeg_bytes(value: Any, *, quality: int, reencode: bool) -> bytes:
    if not reencode:
        encoded = _encoded_jpeg_bytes(value)
        if encoded is not None:
            return encoded
    image = _decode_parquet_image(value)
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG", quality=int(quality))
    return encoded.getvalue()


def _add_bytes_to_tar(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mtime = 0
    tar.addfile(info, io.BytesIO(payload))


class ShardWriter:
    def __init__(self, *, output_root: Path, split: str, samples_per_shard: int) -> None:
        self.output_root = output_root
        self.split = split
        self.samples_per_shard = int(samples_per_shard)
        self.sample_count = 0
        self.shard_index = 0
        self.shards: list[str] = []
        self._tar: tarfile.TarFile | None = None
        self._tmp_path: Path | None = None
        self._final_path: Path | None = None

    def _open_next_shard(self) -> None:
        shard_name = f"{self.split}-{self.shard_index:06d}.tar"
        self._final_path = self.output_root / shard_name
        self._tmp_path = self.output_root / f".{shard_name}.tmp-{os.getpid()}"
        self._tar = tarfile.open(self._tmp_path, "w")
        self.shards.append(shard_name)
        self.shard_index += 1

    def _close_current_shard(self, *, commit: bool) -> None:
        if self._tar is not None:
            self._tar.close()
            self._tar = None
        if self._tmp_path is not None and self._final_path is not None:
            if commit:
                os.replace(self._tmp_path, self._final_path)
            elif self._tmp_path.exists():
                self._tmp_path.unlink()
        self._tmp_path = None
        self._final_path = None

    def add(self, *, image_bytes: bytes, label: int) -> None:
        if self._tar is None:
            self._open_next_shard()
        elif self.sample_count > 0 and self.sample_count % self.samples_per_shard == 0:
            self._close_current_shard(commit=True)
            self._open_next_shard()

        assert self._tar is not None
        key = f"{self.split}-{self.sample_count:09d}"
        _add_bytes_to_tar(self._tar, f"{key}.jpg", image_bytes)
        _add_bytes_to_tar(self._tar, f"{key}.cls", f"{int(label)}\n".encode("utf-8"))
        self.sample_count += 1

    def close(self, *, commit: bool) -> None:
        self._close_current_shard(commit=commit)


def _normalize_split(split: str) -> str:
    split = split.lower()
    if split == "val":
        return "validation"
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    return split


def _find_parquet_files(input_root: Path, split: str) -> list[Path]:
    paths = sorted(input_root.glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found in {input_root} for split '{split}'")
    return paths


def _top_level_schema_names(parquet_file: Any) -> list[str]:
    if hasattr(parquet_file, "schema_arrow"):
        return list(parquet_file.schema_arrow.names)
    return list(parquet_file.schema.names)


def _resolve_columns(parquet_file: Any, image_column: str | None, label_column: str | None) -> tuple[str, str]:
    top_level_names = _top_level_schema_names(parquet_file)
    leaf_names = list(parquet_file.schema.names)
    if image_column is not None:
        if image_column in top_level_names:
            resolved_image = image_column
        elif image_column in leaf_names and "image" in top_level_names:
            resolved_image = "image"
        else:
            raise ValueError(
                f"Requested image column '{image_column}' was not found. "
                f"Top-level columns: {top_level_names}; leaf columns: {leaf_names}"
            )
    else:
        resolved_image = _infer_first_present(["image", "bytes", "jpg", "png", "jpeg", "webp", "path"], top_level_names)
        if resolved_image is None:
            resolved_image = _infer_first_present(["bytes", "jpg", "png", "jpeg", "webp", "path"], leaf_names)
    if label_column is not None:
        if label_column in top_level_names or label_column in leaf_names:
            resolved_label = label_column
        else:
            raise ValueError(
                f"Requested label column '{label_column}' was not found. "
                f"Top-level columns: {top_level_names}; leaf columns: {leaf_names}"
            )
    else:
        resolved_label = _infer_first_present(["label", "labels", "cls", "class", "fine_label"], top_level_names)
        if resolved_label is None:
            resolved_label = _infer_first_present(["label", "labels", "cls", "class", "fine_label"], leaf_names)
    if resolved_image is None:
        raise ValueError(f"Could not infer image column from parquet schema columns {top_level_names}")
    if resolved_label is None:
        raise ValueError(f"Could not infer label column from parquet schema columns {top_level_names}")
    return resolved_image, resolved_label


def convert_imagenet_parquet_to_webdataset(
    *,
    input_root: str | Path,
    output_root: str | Path,
    splits: list[str],
    image_column: str | None,
    label_column: str | None,
    samples_per_shard: int,
    parquet_batch_size: int = 1024,
    jpeg_quality: int = 95,
    reencode_images: bool = False,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "format": "imagenet1k_webdataset_v1",
        "source_root": str(input_root),
        "image_column": image_column,
        "label_column": label_column,
        "samples_per_shard": int(samples_per_shard),
        "jpeg_quality": int(jpeg_quality),
        "reencode_images": bool(reencode_images),
        "created_at_unix": int(time.time()),
        "splits": {},
    }

    for split_arg in splits:
        split = _normalize_split(split_arg)
        parquet_paths = _find_parquet_files(input_root, split)
        first_file = pq.ParquetFile(parquet_paths[0])
        resolved_image_column, resolved_label_column = _resolve_columns(first_file, image_column, label_column)
        if metadata["image_column"] is None:
            metadata["image_column"] = resolved_image_column
        if metadata["label_column"] is None:
            metadata["label_column"] = resolved_label_column

        writer = ShardWriter(output_root=output_root, split=split, samples_per_shard=samples_per_shard)
        try:
            for parquet_path in parquet_paths:
                parquet_file = pq.ParquetFile(parquet_path)
                for batch in parquet_file.iter_batches(
                    batch_size=int(parquet_batch_size),
                    columns=[resolved_image_column, resolved_label_column],
                ):
                    for row in batch.to_pylist():
                        writer.add(
                            image_bytes=_image_to_jpeg_bytes(
                                row[resolved_image_column],
                                quality=jpeg_quality,
                                reencode=reencode_images,
                            ),
                            label=int(row[resolved_label_column]),
                        )
            writer.close(commit=True)
        except Exception:
            writer.close(commit=False)
            raise

        metadata["splits"][split] = {
            "num_samples": int(writer.sample_count),
            "shards": writer.shards,
            "parquet_files": [path.name for path in parquet_paths],
            "resolved_image_column": resolved_image_column,
            "resolved_label_column": resolved_label_column,
        }

    metadata_path = output_root / "metadata.json"
    tmp_metadata_path = output_root / f".metadata.json.tmp-{os.getpid()}"
    try:
        tmp_metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        os.replace(tmp_metadata_path, metadata_path)
    finally:
        if tmp_metadata_path.exists():
            tmp_metadata_path.unlink()
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-root", required=True, help="Root containing ImageNet parquet files.")
    parser.add_argument("--output-root", required=True, help="Directory where WebDataset tar shards will be written.")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"], help="Parquet splits to convert.")
    parser.add_argument("--image-column", default=None, help="Parquet image column. Inferred when omitted.")
    parser.add_argument("--label-column", default=None, help="Parquet label column. Inferred when omitted.")
    parser.add_argument("--samples-per-shard", type=int, default=4096, help="Number of samples per tar shard.")
    parser.add_argument("--parquet-batch-size", type=int, default=1024, help="Rows read from parquet at a time.")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality for re-encoded samples.")
    parser.add_argument(
        "--reencode-images",
        action="store_true",
        help="Decode and re-encode every image instead of copying existing JPEG bytes when possible.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.samples_per_shard <= 0:
        raise ValueError("--samples-per-shard must be positive")
    if args.parquet_batch_size <= 0:
        raise ValueError("--parquet-batch-size must be positive")
    metadata = convert_imagenet_parquet_to_webdataset(
        input_root=args.input_root,
        output_root=args.output_root,
        splits=args.splits,
        image_column=args.image_column,
        label_column=args.label_column,
        samples_per_shard=args.samples_per_shard,
        parquet_batch_size=args.parquet_batch_size,
        jpeg_quality=args.jpeg_quality,
        reencode_images=args.reencode_images,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
