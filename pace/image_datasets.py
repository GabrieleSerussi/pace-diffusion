"""Shared directory/ZIP image datasets and reproducible dataset manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from array import array
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Sequence, overload
import zipfile

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .dataset_specs import (
    DEFAULT_LSUN_MONITOR_SEED,
    DEFAULT_LSUN_MONITOR_SIZE,
    FFHQ_PROTOCOL,
    LSUN_BEDROOM_PROTOCOL,
    canonical_split,
    dataset_spec,
    deterministic_subset,
    ffhq_split_paths,
    hash_ranked_paths,
    normalize_relative_path,
    ordered_path_digest,
    split_spec,
    validate_ffhq_listing,
    validate_lsun_bedroom_listing,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MANIFEST_FORMAT = "diffdist_image_manifest_v1"


@dataclass(frozen=True, slots=True)
class ImageRecord:
    path: str
    size: int


class NumericPathSequence(Sequence[str]):
    """Compact lazy path sequence for million-image canonical populations."""

    def __init__(self, *, prefix: str, count: int, digits: int, suffix: str) -> None:
        self.prefix = prefix
        self.count = int(count)
        self.digits = int(digits)
        self.suffix = suffix

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int | slice) -> str | list[str]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(self.count))]
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError(index)
        name = f"{index:0{self.digits}d}{self.suffix}"
        return f"{self.prefix}/{name}" if self.prefix else name

    def __iter__(self) -> Iterator[str]:
        for index in range(self.count):
            yield self[index]  # type: ignore[misc]


class NumericImageRecordSequence(Sequence[ImageRecord]):
    def __init__(self, paths: NumericPathSequence, sizes: array) -> None:
        self.paths = paths
        self.sizes = sizes

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int | slice) -> ImageRecord | list[ImageRecord]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        path = self.paths[index]
        assert isinstance(path, str)
        return ImageRecord(path, int(self.sizes[index]))

    def __iter__(self) -> Iterator[ImageRecord]:
        for index in range(len(self)):
            item = self[index]
            assert isinstance(item, ImageRecord)
            yield item


def discover_numeric_image_records(
    root: str | Path,
    *,
    count: int,
    digits: int,
    suffix: str,
    display_name: str,
) -> tuple[str, NumericImageRecordSequence]:
    """Validate a canonical numeric directory without retaining one million strings."""
    root = Path(root).expanduser()
    if not root.is_dir():
        source_kind, records = discover_image_records(root)
        paths = [record.path for record in records]
        # ZIP-backed canonical populations are uncommon and remain materialized.
        if len(paths) != count:
            raise ValueError(f"{display_name} protocol validation failed: found {len(paths):,}, expected {count:,}")
        prefix: str | None = None
        seen = bytearray(count)
        sizes = array("Q", [0]) * count
        invalid: list[str] = []
        for record in records:
            path = PurePosixPath(record.path)
            stem = path.name[: -len(suffix)] if path.name.lower().endswith(suffix) else ""
            record_prefix = "" if path.parent == PurePosixPath(".") else path.parent.as_posix()
            if prefix is None:
                prefix = record_prefix
            valid_stem = len(stem) == digits and stem.isdigit()
            image_id = int(stem) if valid_stem else -1
            canonical_name = f"{image_id:0{digits}d}{suffix}" if valid_stem else None
            if (
                record_prefix != prefix
                or not valid_stem
                or path.name != canonical_name
                or image_id >= count
                or seen[image_id]
            ):
                if len(invalid) < 5:
                    invalid.append(record.path)
                continue
            seen[image_id] = 1
            sizes[image_id] = record.size
        missing = count - int(sum(seen))
        if invalid or missing:
            raise ValueError(
                f"{display_name} protocol validation failed: missing IDs={missing:,}; "
                f"invalid/duplicate examples={invalid}"
            )
        return source_kind, NumericImageRecordSequence(
            NumericPathSequence(prefix=prefix or "", count=count, digits=digits, suffix=suffix), sizes
        )
    seen = bytearray(count)
    sizes = array("Q", [0]) * count
    prefix: str | None = None
    image_count = 0
    invalid: list[str] = []
    pending: list[tuple[str, str]] = [(os.fspath(root), "")]
    while pending:
        directory, relative_dir = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    relative = f"{relative_dir}/{entry.name}" if relative_dir else entry.name
                    if entry.is_dir(follow_symlinks=False):
                        pending.append((entry.path, relative))
                        continue
                    if not entry.is_file(follow_symlinks=False) or not _is_image(entry.name):
                        continue
                    image_count += 1
                    stem = entry.name[: -len(suffix)] if entry.name.lower().endswith(suffix) else ""
                    if len(stem) != digits or not stem.isdigit():
                        if len(invalid) < 5:
                            invalid.append(relative)
                        continue
                    image_id = int(stem)
                    if entry.name != f"{image_id:0{digits}d}{suffix}" or image_id >= count or seen[image_id]:
                        if len(invalid) < 5:
                            invalid.append(relative)
                        continue
                    if prefix is None:
                        prefix = relative_dir
                    if relative_dir != prefix:
                        if len(invalid) < 5:
                            invalid.append(relative)
                        continue
                    seen[image_id] = 1
                    sizes[image_id] = entry.stat().st_size
        except PermissionError as exc:
            raise PermissionError(f"Cannot traverse dataset directory {directory}: {exc}") from exc
    missing = count - int(sum(seen))
    if image_count != count or invalid or missing:
        raise ValueError(
            f"{display_name} protocol validation failed: found {image_count:,} images, expected {count:,}; "
            f"missing IDs={missing:,}; invalid/duplicate examples={invalid}"
        )
    paths = NumericPathSequence(prefix=prefix or "", count=count, digits=digits, suffix=suffix)
    return "directory", NumericImageRecordSequence(paths, sizes)


def _is_image(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def discover_image_records(root: str | Path) -> tuple[str, list[ImageRecord]]:
    """Recursively enumerate images in a directory or ZIP in stable POSIX order."""
    root = Path(root).expanduser()
    if root.is_dir():
        # ``Path.rglob``/``os.walk`` may retain a million-entry filename list for
        # flat LSUN directories.  Stream scandir entries recursively to keep peak
        # memory proportional to the compact slotted records themselves.
        records = []
        pending: list[tuple[str, str]] = [(os.fspath(root), "")]
        while pending:
            directory, relative_dir = pending.pop()
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        relative = f"{relative_dir}/{entry.name}" if relative_dir else entry.name
                        if entry.is_dir(follow_symlinks=False):
                            pending.append((entry.path, relative))
                        elif entry.is_file(follow_symlinks=False) and _is_image(entry.name):
                            records.append(ImageRecord(relative.replace("\\", "/"), entry.stat().st_size))
            except PermissionError as exc:
                raise PermissionError(f"Cannot traverse dataset directory {directory}: {exc}") from exc
        source_kind = "directory"
    elif root.is_file() and root.suffix.lower() == ".zip":
        with zipfile.ZipFile(root) as archive:
            records = []
            for info in archive.infolist():
                if info.is_dir() or not _is_image(info.filename):
                    continue
                normalized = normalize_relative_path(info.filename)
                records.append(ImageRecord(normalized, int(info.file_size)))
        source_kind = "zip"
    elif root.exists():
        raise ValueError(f"Dataset source must be a directory or .zip archive: {root}")
    else:
        raise FileNotFoundError(f"Dataset source does not exist: {root}")
    records.sort(key=lambda record: record.path)
    paths = [record.path for record in records]
    if any(left == right for left, right in zip(paths, paths[1:])):
        raise ValueError(f"Dataset source contains duplicate normalized image paths: {root}")
    if not records:
        raise ValueError(f"No supported images found recursively in {root}")
    return source_kind, records


def records_fingerprint(records: Sequence[ImageRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(normalize_relative_path(record.path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(record.size)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _manifest_entries(value: Any) -> list[str]:
    if isinstance(value, dict):
        value = value.get("entries")
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("Manifest entries must be a JSON list of relative image paths")
    normalized = [normalize_relative_path(item) for item in value]
    if len(normalized) != len(set(normalized)):
        raise ValueError("Manifest contains duplicate image paths")
    return normalized


def read_dataset_manifest(path: str | Path, *, split: str | None = None) -> tuple[list[str], dict[str, Any]]:
    path = Path(path).expanduser()
    text = path.read_text()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        entries = [normalize_relative_path(line.strip()) for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
        if not entries:
            raise ValueError(f"Manifest is empty: {path}")
        return entries, {"format": "text", "entries_sha256": ordered_path_digest(entries)}
    if isinstance(payload, list):
        entries = _manifest_entries(payload)
    elif isinstance(payload, dict):
        if payload.get("format") not in {None, MANIFEST_FORMAT}:
            raise ValueError(f"Unsupported manifest format {payload.get('format')!r} in {path}")
        splits = payload.get("splits", {})
        if split is not None and splits:
            if split not in splits:
                raise ValueError(f"Manifest does not define requested split {split!r}: {path}")
            selection = splits[split]
            if isinstance(selection, dict) and selection.get("kind") == "all":
                entries = _manifest_entries(payload)
            else:
                entries = _manifest_entries(selection)
        else:
            manifest_split = payload.get("split")
            if split is not None and manifest_split is not None and manifest_split != split:
                raise ValueError(f"Manifest split is {manifest_split!r}, requested {split!r}: {path}")
            entries = _manifest_entries(payload)
        expected = payload.get("source_listing_sha256")
        if "entries_sha256" in payload and split not in splits and payload["entries_sha256"] != ordered_path_digest(entries):
            raise ValueError(f"Manifest entries digest does not match contents: {path}")
        payload = dict(payload)
        payload["selected_entries_sha256"] = ordered_path_digest(entries)
        payload["expected_source_listing_sha256"] = expected
        return entries, payload
    else:
        raise ValueError(f"Manifest root must be a JSON object/list or newline-delimited paths: {path}")
    return entries, {"format": "json-list", "entries_sha256": ordered_path_digest(entries)}


def build_dataset_manifest(
    *,
    dataset_id: str,
    root: str | Path,
    protocol: str | None = None,
    lsun_monitor_size: int = DEFAULT_LSUN_MONITOR_SIZE,
    lsun_monitor_seed: int = DEFAULT_LSUN_MONITOR_SEED,
    strict_protocol: bool = True,
) -> dict[str, Any]:
    if dataset_id == "ffhq" and strict_protocol:
        source_kind, records = discover_numeric_image_records(
            root, count=70_000, digits=5, suffix=".png", display_name="FFHQ"
        )
    elif dataset_id == "lsun_bedroom" and strict_protocol:
        source_kind, records = discover_numeric_image_records(
            root, count=1_000_000, digits=7, suffix=".jpg", display_name="LSUN Bedroom"
        )
    else:
        source_kind, records = discover_image_records(root)
    paths = records.paths if isinstance(records, NumericImageRecordSequence) else [record.path for record in records]
    splits: dict[str, Any]
    if dataset_id == "ffhq":
        protocol = protocol or FFHQ_PROTOCOL
        if strict_protocol:
            ordered = validate_ffhq_listing(paths)["ordered_paths"]
            assert isinstance(ordered, Sequence)
        else:
            ordered = paths
        splits = {
            "all": {"kind": "all", "count": len(ordered), "is_heldout": False},
            "train": {"kind": "all", "count": len(ordered), "is_heldout": False},
            "monitor": {"entries": ordered[60_000:70_000], "is_heldout": False},
            "fid": {"entries": ordered[:50_000], "is_heldout": False},
        }
        paths = ordered
    elif dataset_id == "lsun_bedroom":
        protocol = protocol or LSUN_BEDROOM_PROTOCOL
        if strict_protocol:
            ordered = validate_lsun_bedroom_listing(paths)["ordered_paths"]
            assert isinstance(ordered, Sequence)
        else:
            ordered = paths
        monitor = hash_ranked_paths(ordered, size=min(lsun_monitor_size, len(ordered)), seed=lsun_monitor_seed)
        splits = {
            "all": {"kind": "all", "count": len(ordered), "is_heldout": False},
            "train": {"kind": "all", "count": len(ordered), "is_heldout": False},
            "monitor": {
                "entries": monitor,
                "entries_sha256": ordered_path_digest(monitor),
                "is_heldout": False,
                "selection": "sha256(seed + NUL + normalized_relative_path)",
                "seed": int(lsun_monitor_seed),
            },
        }
        paths = ordered
    else:
        protocol = protocol or "recursive_image_manifest_v1"
        splits = {"all": {"kind": "all", "count": len(paths)}, "train": {"kind": "all", "count": len(paths)}}
    return {
        "format": MANIFEST_FORMAT,
        "dataset_id": dataset_id,
        "dataset_spec": asdict(dataset_spec(dataset_id)) if dataset_id in {"ffhq", "lsun_bedroom"} else None,
        "protocol": protocol,
        "source": str(Path(root).expanduser().resolve()),
        "source_kind": source_kind,
        "source_listing_sha256": ordered_path_digest(paths),
        "source_records_sha256": records_fingerprint(records),
        "entries_sha256": ordered_path_digest(paths),
        "count": len(paths),
        # NumericPathSequence keeps discovery/preflight memory bounded, but a
        # persisted v1 manifest intentionally contains explicit relative names
        # so it is self-contained and portable across readers.
        "entries": list(paths),
        "splits": splits,
    }


def write_dataset_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


class SharedImageDataset(Dataset):
    """Unconditional RGB dataset with deterministic split and subset selection."""

    def __init__(
        self,
        *,
        dataset_id: str,
        root: str | Path,
        image_size: int,
        split: str = "train",
        manifest: str | Path | None = None,
        max_images: int | None = None,
        subset_seed: int = 0,
        preflight: bool = False,
        ffhq_protocol: str = FFHQ_PROTOCOL,
        lsun_monitor_size: int = DEFAULT_LSUN_MONITOR_SIZE,
        lsun_monitor_seed: int = DEFAULT_LSUN_MONITOR_SEED,
        strict_protocol: bool = True,
    ) -> None:
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        self.dataset_id = dataset_id
        self.root = Path(root).expanduser()
        self.image_size = int(image_size)
        self.split = canonical_split(dataset_id, split) if dataset_id in {"ffhq", "lsun_bedroom"} else split
        if dataset_id == "ffhq" and strict_protocol:
            self.source_kind, all_records = discover_numeric_image_records(
                self.root, count=70_000, digits=5, suffix=".png", display_name="FFHQ"
            )
        elif dataset_id == "lsun_bedroom" and strict_protocol:
            self.source_kind, all_records = discover_numeric_image_records(
                self.root, count=1_000_000, digits=7, suffix=".jpg", display_name="LSUN Bedroom"
            )
        else:
            self.source_kind, all_records = discover_image_records(self.root)
        all_paths = all_records.paths if isinstance(all_records, NumericImageRecordSequence) else [record.path for record in all_records]
        manifest_metadata: dict[str, Any] = {}
        if manifest is not None:
            selected, manifest_metadata = read_dataset_manifest(manifest, split=self.split)
            expected_listing = manifest_metadata.get("expected_source_listing_sha256")
            actual_listing = ordered_path_digest(all_paths)
            if expected_listing is not None and expected_listing != actual_listing:
                raise ValueError(
                    f"Dataset source listing fingerprint differs from manifest: expected {expected_listing}, got {actual_listing}"
                )
            source_path_set = set(all_paths)
            missing = [path for path in selected if path not in source_path_set]
            del source_path_set
            if missing:
                raise FileNotFoundError(f"Manifest references {len(missing)} missing images; examples: {missing[:5]}")
            # Avoid materializing a second million-record list/dictionary for
            # canonical all/train manifests.  The ordered digest and membership
            # checks above prove that this is the complete source population.
            if len(selected) == len(all_paths) and ordered_path_digest(selected) == actual_listing:
                selected = all_paths
        elif dataset_id == "ffhq":
            if ffhq_protocol != FFHQ_PROTOCOL:
                raise ValueError(f"Unsupported FFHQ protocol: {ffhq_protocol}")
            if strict_protocol:
                selected = ffhq_split_paths(all_paths, self.split)
            elif self.split in {"all", "train"}:
                selected = list(all_paths)
            else:
                raise ValueError(
                    f"FFHQ split {self.split!r} requires the complete validated 70k numeric population; "
                    "strict_protocol=False is only valid for all/train"
                )
        elif dataset_id == "lsun_bedroom":
            if lsun_monitor_seed < 0:
                raise ValueError("LSUN monitor seed must be non-negative")
            if strict_protocol:
                validated = validate_lsun_bedroom_listing(all_paths)
                ordered = validated["ordered_paths"]
                assert isinstance(ordered, Sequence)
            else:
                ordered = all_paths
            if self.split == "monitor":
                selected = hash_ranked_paths(ordered, size=min(lsun_monitor_size, len(ordered)), seed=lsun_monitor_seed)
            elif self.split == "fid":
                raise ValueError("LSUN Bedroom FID split requires an explicit --dataset-manifest; it is not inferred")
            else:
                selected = ordered
        else:
            selected = list(all_paths)
        # Preserve the historical image_folder ``max_images`` sorted-prefix
        # behavior.  Named protocols use hash-ranked subsets so debug caps are
        # deterministic without being biased toward numeric filename prefixes.
        if dataset_id == "image_folder" and max_images is not None:
            selected = selected[:max_images]
        else:
            selected = deterministic_subset(selected, size=max_images, seed=subset_seed)
        if selected is all_paths:
            self.records = all_records
            self.paths = all_paths
        else:
            selected_set = set(selected)
            selected_records = {record.path: record for record in all_records if record.path in selected_set}
            self.records = [selected_records[path] for path in selected]
            self.paths = list(selected)
        self.manifest_metadata = manifest_metadata
        spec = split_spec(dataset_id, self.split) if dataset_id in {"ffhq", "lsun_bedroom"} else None
        self.metadata = {
            "dataset_id": dataset_id,
            "split": self.split,
            "protocol": None if spec is None else spec.protocol,
            "is_heldout": None if spec is None else spec.is_heldout,
            "count": len(self.paths),
            "source_count": len(all_paths),
            "entries_sha256": ordered_path_digest(self.paths),
            "source_listing_sha256": ordered_path_digest(all_paths),
            "source_records_sha256": records_fingerprint(all_records),
            "source_kind": self.source_kind,
            "root": str(self.root.resolve()),
            "image_size": self.image_size,
            "dataset_spec": asdict(dataset_spec(dataset_id)) if dataset_id in {"ffhq", "lsun_bedroom"} else None,
        }
        self._archive: zipfile.ZipFile | None = None
        if dataset_id == "ffhq":
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ])
        elif dataset_id == "lsun_bedroom":
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ])
        if preflight:
            preflight_dataset(self, full_decode=True)

    def __len__(self) -> int:
        return len(self.records)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_archive"] = None
        return state

    def _open(self, record: ImageRecord) -> Image.Image:
        if self.source_kind == "directory":
            with Image.open(self.root / record.path) as image:
                if self.dataset_id in {"ffhq", "lsun_bedroom"} and image.mode != "RGB":
                    raise ValueError(f"{record.path} has mode {image.mode}; expected RGB")
                return image.convert("RGB")
        if self._archive is None:
            self._archive = zipfile.ZipFile(self.root)
        with self._archive.open(record.path) as handle:
            with Image.open(io.BytesIO(handle.read())) as image:
                if self.dataset_id in {"ffhq", "lsun_bedroom"} and image.mode != "RGB":
                    raise ValueError(f"{record.path} has mode {image.mode}; expected RGB")
                return image.convert("RGB")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = self._open(self.records[index])
        if self.dataset_id == "ffhq" and image.size != (self.image_size, self.image_size):
            raise ValueError(
                f"FFHQ image {self.records[index].path} has size {image.size}; expected exact "
                f"{self.image_size}x{self.image_size}. Prepare this teacher resolution with "
                "scripts/data/prepare_ffhq_dataset.py (LANCZOS)."
            )
        if self.dataset_id == "lsun_bedroom" and min(image.size) != self.image_size:
            raise ValueError(
                f"LSUN Bedroom image {self.records[index].path} has size {image.size}; expected short side "
                f"exactly {self.image_size} before deterministic center crop"
            )
        return self.transform(image), -1


def _read_original_bytes(
    dataset: SharedImageDataset,
    record: ImageRecord,
    *,
    archive: zipfile.ZipFile | None = None,
) -> bytes:
    if dataset.source_kind == "directory":
        with open(dataset.root / record.path, "rb") as handle:
            return handle.read()
    if archive is not None:
        return archive.read(record.path)
    with zipfile.ZipFile(dataset.root) as opened_archive:
        return opened_archive.read(record.path)


def _decode_original(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        return image.copy()


def validate_dataset_readability(dataset: SharedImageDataset) -> dict[str, Any]:
    """Fail before model loading when directory entries are not process-readable."""
    if dataset.source_kind != "directory":
        return {"checked": len(dataset.records), "unreadable": 0, "error_examples": []}
    unreadable: list[str] = []
    unreadable_count = 0
    for record in dataset.records:
        source_path = dataset.root / record.path
        if not os.access(source_path, os.R_OK):
            unreadable_count += 1
            if len(unreadable) < 20:
                mode_bits = source_path.stat().st_mode & 0o777
                unreadable.append(f"{record.path}: not readable by the current process (mode {mode_bits:o})")
    if unreadable_count:
        raise ValueError(
            f"Dataset validation failed: {unreadable_count:,}/{len(dataset.records):,} image files are unreadable; "
            "fix file/group permissions before model loading. Examples: " + " | ".join(unreadable[:5])
        )
    return {"checked": len(dataset.records), "unreadable": 0, "error_examples": []}


def preflight_dataset(dataset: SharedImageDataset, *, full_decode: bool = True) -> dict[str, Any]:
    """Check readability, RGB mode, and source geometry for every selected record."""
    errors: list[str] = []
    modes: dict[str, int] = {}
    sizes: dict[str, int] = {}
    checked = 0
    content_digest = hashlib.sha256() if full_decode else None
    validate_dataset_readability(dataset)
    archive = zipfile.ZipFile(dataset.root) if dataset.source_kind == "zip" else None
    try:
        for record in dataset.records:
            try:
                if full_decode:
                    data = _read_original_bytes(dataset, record, archive=archive)
                    image = _decode_original(data)
                    assert content_digest is not None
                    content_digest.update(record.path.encode("utf-8"))
                    content_digest.update(b"\0")
                    content_digest.update(hashlib.sha256(data).digest())
                    content_digest.update(b"\n")
                else:
                    image = dataset._open(record)
                modes[image.mode] = modes.get(image.mode, 0) + 1
                size_key = f"{image.width}x{image.height}"
                sizes[size_key] = sizes.get(size_key, 0) + 1
                if image.mode != "RGB":
                    raise ValueError(f"mode={image.mode}, expected RGB")
                if dataset.dataset_id == "ffhq" and image.size != (dataset.image_size, dataset.image_size):
                    raise ValueError(f"size={image.size}, expected exact {dataset.image_size}x{dataset.image_size}")
                if dataset.dataset_id == "lsun_bedroom" and min(image.size) != dataset.image_size:
                    raise ValueError(f"size={image.size}, expected short side exactly {dataset.image_size}")
                checked += 1
            except Exception as exc:
                if len(errors) < 20:
                    errors.append(f"{record.path}: {exc}")
    finally:
        if archive is not None:
            archive.close()
    failed = len(dataset.records) - checked
    report = {
        **dataset.metadata,
        "checked": checked,
        "failed": failed,
        "full_decode": bool(full_decode),
        "modes": modes,
        "sizes": sizes,
        "error_examples": errors,
        "content_sha256": content_digest.hexdigest() if content_digest is not None and failed == 0 else None,
        "valid": failed == 0,
    }
    if failed:
        raise ValueError(
            f"Dataset preflight failed for {failed:,}/{len(dataset.records):,} images; examples: " + " | ".join(errors[:5])
        )
    return report


# Backward-compatible name used by evaluate_parameters_edm tests/importers.  Its
# behavior is extended to recursive directories and ZIP archives.
class ImageFolderFlat(SharedImageDataset):
    def __init__(self, root: str, image_size: int, max_images: int | None = None):
        super().__init__(dataset_id="image_folder", root=root, image_size=image_size, split="all", max_images=max_images)
