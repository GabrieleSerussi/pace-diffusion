"""Deterministic dataset protocols used by the EDM analysis and distillation scripts.

The protocol layer deliberately deals only in normalized relative image names.  It
is therefore shared by directory and ZIP-backed datasets and can be tested without
loading image pixels.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import heapq
from pathlib import PurePosixPath
from typing import Iterable, Sequence


FFHQ_PROTOCOL = "ffhq256_numeric_v1"
LSUN_BEDROOM_PROTOCOL = "lsun_bedroom256_hash_monitor_v1"
DEFAULT_LSUN_MONITOR_SIZE = 10_000
DEFAULT_LSUN_MONITOR_SEED = 12_345


@dataclass(frozen=True)
class DatasetSplitSpec:
    dataset_id: str
    protocol: str
    split: str
    is_heldout: bool
    description: str


@dataclass(frozen=True)
class DatasetSpec:
    """Top-level, versioned contract for a reproducible image population."""

    dataset_id: str
    protocol: str
    native_resolution: int
    channels: int
    conditional: bool
    train_split: str
    monitor_split: str
    fid_split: str | None
    monitor_is_heldout: bool
    preprocessing: str


DATASET_SPECS: dict[str, DatasetSpec] = {
    "ffhq": DatasetSpec(
        dataset_id="ffhq",
        protocol=FFHQ_PROTOCOL,
        native_resolution=256,
        channels=3,
        conditional=False,
        train_split="train",
        monitor_split="monitor",
        fid_split="fid",
        monitor_is_heldout=False,
        preprocessing="exact RGB square at teacher resolution; offline reductions use LANCZOS",
    ),
    "lsun_bedroom": DatasetSpec(
        dataset_id="lsun_bedroom",
        protocol=LSUN_BEDROOM_PROTOCOL,
        native_resolution=256,
        channels=3,
        conditional=False,
        train_split="train",
        monitor_split="monitor",
        fid_split=None,
        monitor_is_heldout=False,
        preprocessing="LANCZOS short-side resize followed by deterministic center crop",
    ),
}


def dataset_spec(dataset_id: str) -> DatasetSpec:
    try:
        return DATASET_SPECS[dataset_id]
    except KeyError as exc:
        raise ValueError(f"No dataset specification registered for {dataset_id!r}") from exc


def normalize_relative_path(value: str) -> str:
    """Return a safe, portable POSIX relative path."""
    value = str(value).replace("\\", "/")
    raw_parts = value.split("/")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError(f"Image path must be a safe relative path, got {value!r}")
    return path.as_posix()


def ordered_path_digest(paths: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(normalize_relative_path(path).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def canonical_split(dataset_id: str, split: str | None) -> str:
    split = (split or "train").strip().lower()
    aliases = {"val": "monitor", "validation": "monitor", "test": "monitor"}
    if dataset_id in {"ffhq", "lsun_bedroom"}:
        split = aliases.get(split, split)
    allowed = {"all", "train", "monitor", "fid"}
    if split not in allowed:
        raise ValueError(f"Unsupported {dataset_id} split {split!r}; choose from {sorted(allowed)}")
    return split


def split_spec(dataset_id: str, split: str | None) -> DatasetSplitSpec:
    split = canonical_split(dataset_id, split)
    if dataset_id == "ffhq":
        descriptions = {
            "all": "all 70,000 numeric FFHQ images",
            "train": "all 70,000 images; monitoring images are intentionally included",
            "monitor": "numeric IDs 60000--69999; overlapping monitoring subset",
            "fid": "numeric IDs 00000--49999; established 50k reference population",
        }
        return DatasetSplitSpec(dataset_id, FFHQ_PROTOCOL, split, False, descriptions[split])
    if dataset_id == "lsun_bedroom":
        descriptions = {
            "all": "all discovered LSUN Bedroom images",
            "train": "all discovered LSUN Bedroom images",
            "monitor": "deterministic SHA-256-ranked monitoring subset that overlaps training",
            "fid": "requires an explicit reference manifest; no inferred LSUN FID population",
        }
        return DatasetSplitSpec(dataset_id, LSUN_BEDROOM_PROTOCOL, split, False, descriptions[split])
    raise ValueError(f"No dataset protocol registered for {dataset_id!r}")


def _numeric_basename(path: str, *, digits: int, suffix: str) -> int | None:
    name = PurePosixPath(normalize_relative_path(path)).name
    if not name.lower().endswith(suffix):
        return None
    stem = name[: -len(suffix)]
    if len(stem) != digits or not stem.isdigit():
        return None
    return int(stem)


def _validate_contiguous_numeric_listing(
    paths: Sequence[str],
    *,
    count: int,
    digits: int,
    suffix: str,
    display_name: str,
) -> list[str]:
    """Validate a padded numeric population in O(n) time and low extra memory."""
    if len(paths) != count:
        raise ValueError(f"{display_name} protocol validation failed: found {len(paths):,} images, expected {count:,}")
    ordered = paths
    parent: str | None = None
    invalid: list[str] = []
    for image_id, value in enumerate(ordered):
        path = PurePosixPath(normalize_relative_path(value))
        path_parent = path.parent.as_posix()
        if parent is None:
            parent = path_parent
        if path_parent != parent or path.name.lower() != f"{image_id:0{digits}d}{suffix}":
            if len(invalid) < 5:
                invalid.append(value)
    if invalid:
        missing_hint = ""
        if display_name == "FFHQ":
            missing_hint = "; listing has missing numeric IDs, a duplicate, or an out-of-order name"
        raise ValueError(
            f"{display_name} protocol validation failed: expected one directory containing contiguous "
            f"{digits}-digit {suffix} names; first mismatches: {invalid}{missing_hint}"
        )
    return ordered


def validate_ffhq_listing(paths: Sequence[str]) -> dict[str, object]:
    """Validate the canonical FFHQ 70k numeric population, allowing a ZIP prefix."""
    ordered = _validate_contiguous_numeric_listing(
        paths, count=70_000, digits=5, suffix=".png", display_name="FFHQ"
    )
    return {"ordered_paths": ordered, "canonical_digest": ordered_path_digest(f"{i:05d}.png" for i in range(70_000))}


def ffhq_split_paths(paths: Sequence[str], split: str | None) -> Sequence[str]:
    validated = validate_ffhq_listing(paths)
    ordered = validated["ordered_paths"]
    assert isinstance(ordered, Sequence)
    split = canonical_split("ffhq", split)
    if split in {"all", "train"}:
        return ordered
    if split == "monitor":
        return ordered[60_000:70_000]
    return ordered[:50_000]


def validate_lsun_bedroom_listing(paths: Sequence[str]) -> dict[str, object]:
    """Validate the derived flat one-million-image LSUN Bedroom population."""
    ordered = _validate_contiguous_numeric_listing(
        paths, count=1_000_000, digits=7, suffix=".jpg", display_name="LSUN Bedroom"
    )
    return {"ordered_paths": ordered, "canonical_digest": ordered_path_digest(f"{i:07d}.jpg" for i in range(1_000_000))}


def hash_ranked_paths(paths: Sequence[str], *, size: int, seed: int) -> list[str]:
    if size <= 0:
        raise ValueError(f"monitor size must be positive, got {size}")
    if size > len(paths):
        raise ValueError(f"monitor size {size:,} exceeds dataset size {len(paths):,}")
    prefix = str(int(seed)).encode("ascii") + b"\0"
    candidates = (
        (hashlib.sha256(prefix + normalize_relative_path(path).encode("utf-8")).digest(), normalize_relative_path(path))
        for path in paths
    )
    ranked = heapq.nsmallest(size, candidates)
    return [path for _digest, path in ranked]


def deterministic_subset(paths: Sequence[str], *, size: int | None, seed: int) -> Sequence[str]:
    if size is None or size >= len(paths):
        return paths
    return hash_ranked_paths(paths, size=size, seed=seed)
