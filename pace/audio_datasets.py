"""Deterministic raw-waveform datasets used by audio diffusion experiments.

The SC09 contract in this module intentionally follows the preprocessing used
by the pretrained unconditional DiffWave checkpoint, while adding the official
Speech Commands v0.02 splits that the upstream loader omitted.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
import wave

import torch
import torchaudio
from torch.utils.data import Dataset

from .dataset_specs import deterministic_subset, normalize_relative_path, ordered_path_digest


SC09_PROTOCOL = "sc09_speech_commands_v002_official_lists_raw16k_v1"
SC09_DATASET_COMMIT = "fe62f33d2af5db6f01e504ec1f360da7df9692e8"
SC09_ARCHIVE_URL = (
    "https://huggingface.co/datasets/krandiash/sc09/resolve/"
    f"{SC09_DATASET_COMMIT}/sc09.zip"
)
SC09_ARCHIVE_SIZE = 893_855_183
SC09_ARCHIVE_SHA256 = "ca0cff7168708fe3e1d2d6fd8ac7b0c26e4c574143ea708951e19ed6a58f7792"
SC09_ARCHIVE_FILENAME = "sc09-fe62f33.zip"
SC09_VALIDATION_LIST_SHA256 = "5747407275538b4056e823982f0db1fc993776ab532048196a19be701bdc87d2"
SC09_TESTING_LIST_SHA256 = "2d17c6b3faf63be43eda93cfeb0c747cfd79b7b236282039dbac65a2cb5f1df5"
SC09_SOURCE_RECORDS_SHA256 = "9e846a24d5fdbd576b5e3a3859ef1a05cca20eac4e1d9ca7f9a1b99849930d54"

SC09_SAMPLE_RATE = 16_000
SC09_SAMPLE_LENGTH = 16_000
SC09_CHANNELS = 1
SC09_LABELS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
SC09_LABEL_TO_INDEX = {label: index for index, label in enumerate(SC09_LABELS)}
SC09_SPLIT_COUNTS = {"all": 38_908, "train": 31_158, "validation": 3_643, "test": 4_107}
SC09_SPLIT_ENTRIES_SHA256 = {
    "all": "8e733389f286f84a8a85fdb19223d099069b6acd7a36d7ebd21a3aa07c501dcb",
    "train": "74cc8d696630e3fd0377b77ee9323e829b17624ca0c9eb802d947306ce577b1c",
    "validation": "d018c4eb4f4510721f65b3a61a04b139a6c32706e995d1fda485b3b113226f6a",
    "test": "2afe068f16ad2a50a54d561acadb1d2036c0e0920a0f215573f133a98a645dca",
}
SC09_CLASS_SPLIT_COUNTS: dict[str, dict[str, int]] = {
    "zero": {"train": 3_250, "validation": 384, "test": 418, "all": 4_052},
    "one": {"train": 3_140, "validation": 351, "test": 399, "all": 3_890},
    "two": {"train": 3_111, "validation": 345, "test": 424, "all": 3_880},
    "three": {"train": 2_966, "validation": 356, "test": 405, "all": 3_727},
    "four": {"train": 2_955, "validation": 373, "test": 400, "all": 3_728},
    "five": {"train": 3_240, "validation": 367, "test": 445, "all": 4_052},
    "six": {"train": 3_088, "validation": 378, "test": 394, "all": 3_860},
    "seven": {"train": 3_205, "validation": 387, "test": 406, "all": 3_998},
    "eight": {"train": 3_033, "validation": 346, "test": 408, "all": 3_787},
    "nine": {"train": 3_170, "validation": 356, "test": 408, "all": 3_934},
}
SC09_MANIFEST_FORMAT = "diffdist_sc09_manifest_v1"


@dataclass(frozen=True, slots=True)
class AudioRecord:
    path: str
    size: int
    label: str


def canonical_sc09_split(split: str | None) -> str:
    value = (split or "validation").strip().lower()
    value = {"val": "validation", "valid": "validation", "monitor": "validation"}.get(value, value)
    if value not in SC09_SPLIT_COUNTS:
        raise ValueError(f"Unsupported SC09 split {split!r}; choose from {sorted(SC09_SPLIT_COUNTS)}")
    return value


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_sc09_filename(name: str) -> bool:
    if not name.endswith(".wav") or "_nohash_" not in name:
        return False
    speaker, utterance = name[:-4].rsplit("_nohash_", 1)
    return bool(speaker) and utterance.isdigit()


def discover_sc09_records(root: str | Path, *, strict: bool = True) -> list[AudioRecord]:
    """Discover digit WAVs in stable POSIX order without following nested layouts."""
    root = Path(root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"SC09 root is not a directory: {root}")
    records: list[AudioRecord] = []
    missing_labels: list[str] = []
    malformed: list[str] = []
    for label in SC09_LABELS:
        directory = root / label
        if not directory.is_dir():
            missing_labels.append(label)
            continue
        if directory.is_symlink() and strict:
            raise ValueError(f"SC09 label directory must not be a symlink in strict mode: {directory}")
        for path in directory.iterdir():
            if not path.is_file() or path.suffix != ".wav":
                continue
            relative = f"{label}/{path.name}"
            if not _is_sc09_filename(path.name):
                malformed.append(relative)
                continue
            if path.is_symlink() and strict:
                raise ValueError(f"SC09 WAV must not be a symlink in strict mode: {path}")
            records.append(AudioRecord(relative, path.stat().st_size, label))
        nested = [path for path in directory.rglob("*.wav") if path.parent != directory]
        if nested:
            examples = [path.relative_to(root).as_posix() for path in nested[:5]]
            raise ValueError(f"SC09 WAVs must be directly inside digit directories; nested examples: {examples}")
    if strict and missing_labels:
        raise ValueError(f"SC09 root is missing digit directories: {missing_labels}")
    if malformed:
        raise ValueError(f"Malformed SC09 `_nohash_` WAV names: {malformed[:5]}")
    records.sort(key=lambda record: record.path)
    paths = [record.path for record in records]
    if len(paths) != len(set(paths)):
        raise ValueError("SC09 source contains duplicate normalized WAV paths")
    if not records:
        raise ValueError(f"No SC09 digit WAVs found under {root}")
    return records


def _read_split_list(path: Path) -> tuple[set[str], str]:
    try:
        data = path.read_bytes()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required SC09 split list is missing: {path}") from exc
    entries: set[str] = set()
    duplicates: list[str] = []
    for raw_line in data.decode("utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        normalized = normalize_relative_path(line)
        parts = PurePosixPath(normalized).parts
        if parts[0] not in SC09_LABEL_TO_INDEX:
            continue
        if len(parts) != 2:
            raise ValueError(f"SC09 split entry must have `label/file.wav` form, got {normalized!r}")
        if normalized in entries:
            duplicates.append(normalized)
        entries.add(normalized)
    if duplicates:
        raise ValueError(f"SC09 split list contains duplicate digit entries: {duplicates[:5]}")
    return entries, _sha256_bytes(data)


def partition_sc09_paths(
    all_paths: Sequence[str],
    *,
    validation_paths: Iterable[str],
    testing_paths: Iterable[str],
    expected_counts: Mapping[str, int] | None = None,
) -> dict[str, list[str]]:
    """Partition paths using official lists and validate the split as a true disjoint union."""
    ordered = sorted(normalize_relative_path(path) for path in all_paths)
    if len(ordered) != len(set(ordered)):
        raise ValueError("SC09 population contains duplicate paths")
    population = set(ordered)
    validation = {normalize_relative_path(path) for path in validation_paths}
    testing = {normalize_relative_path(path) for path in testing_paths}
    overlap = validation & testing
    if overlap:
        raise ValueError(f"SC09 validation and test lists overlap: {sorted(overlap)[:5]}")
    missing_validation = validation - population
    missing_testing = testing - population
    if missing_validation or missing_testing:
        raise ValueError(
            "SC09 split lists reference missing digit WAVs: "
            f"validation={sorted(missing_validation)[:5]}, test={sorted(missing_testing)[:5]}"
        )
    train = population - validation - testing
    result = {
        "all": ordered,
        "train": sorted(train),
        "validation": sorted(validation),
        "test": sorted(testing),
    }
    if expected_counts is not None:
        mismatches: dict[str, tuple[int | None, int]] = {}
        for split, expected in expected_counts.items():
            observed = len(result[split]) if split in result else None
            if observed != int(expected):
                mismatches[split] = (observed, int(expected))
        if mismatches:
            raise ValueError(f"SC09 split counts do not match the declared protocol: {mismatches}")
    return result


def audio_records_fingerprint(records: Sequence[AudioRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(record.size)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def audio_content_fingerprint(root: Path, records: Sequence[AudioRecord]) -> str:
    """Hash selected WAV identities and bytes for cache-safe reuse."""

    digest = hashlib.sha256()
    for record in records:
        digest.update(record.path.encode("utf-8"))
        digest.update(b"\0")
        with (root / record.path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\n")
    return digest.hexdigest()


def _class_counts(paths: Iterable[str]) -> dict[str, int]:
    counts = {label: 0 for label in SC09_LABELS}
    for path in paths:
        counts[PurePosixPath(path).parts[0]] += 1
    return counts


def _wave_metadata(path: Path) -> dict[str, int | str]:
    try:
        with wave.open(os.fspath(path), "rb") as handle:
            metadata: dict[str, int | str] = {
                "channels": handle.getnchannels(),
                "sample_width": handle.getsampwidth(),
                "sample_rate": handle.getframerate(),
                "frames": handle.getnframes(),
                "compression": handle.getcomptype(),
            }
    except (EOFError, wave.Error) as exc:
        raise ValueError(f"Invalid WAV file {path}: {exc}") from exc
    if metadata["channels"] != SC09_CHANNELS:
        raise ValueError(f"SC09 WAV must be mono, got {metadata['channels']} channels: {path}")
    if metadata["sample_width"] != 2 or metadata["compression"] != "NONE":
        raise ValueError(f"SC09 WAV must be uncompressed PCM16: {path}")
    if metadata["sample_rate"] != SC09_SAMPLE_RATE:
        raise ValueError(f"SC09 WAV must be 16000 Hz, got {metadata['sample_rate']}: {path}")
    if int(metadata["frames"]) <= 0:
        raise ValueError(f"SC09 WAV contains no samples: {path}")
    return metadata


def load_sc09_waveform(path: str | Path) -> torch.Tensor:
    """Load normalized PCM with torchaudio, then apply the teacher length rule."""
    path = Path(path)
    metadata = _wave_metadata(path)
    waveform, sample_rate = torchaudio.load(
        os.fspath(path),
        normalize=True,
        channels_first=True,
    )
    if sample_rate != SC09_SAMPLE_RATE:
        raise ValueError(f"SC09 WAV must be 16000 Hz, got {sample_rate}: {path}")
    if waveform.dtype != torch.float32 or waveform.ndim != 2 or waveform.shape[0] != 1:
        raise ValueError(
            "torchaudio.load must return normalized mono float32 audio, got "
            f"shape={tuple(waveform.shape)}, dtype={waveform.dtype}: {path}"
        )
    if waveform.shape[1] != int(metadata["frames"]):
        raise ValueError(
            f"Decoded SC09 length differs from its WAV header: {waveform.shape[1]} "
            f"!= {metadata['frames']}: {path}"
        )
    if not torch.isfinite(waveform).all():
        raise ValueError(f"SC09 WAV decoded to non-finite samples: {path}")
    fixed = torch.zeros((SC09_CHANNELS, SC09_SAMPLE_LENGTH), dtype=torch.float32)
    copy_length = min(waveform.shape[1], SC09_SAMPLE_LENGTH)
    fixed[:, :copy_length] = waveform[:, :copy_length]
    return fixed


class SC09Dataset(Dataset[tuple[torch.Tensor, int]]):
    """Speech Commands v0.02 digits with official deterministic splits."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "validation",
        max_samples: int | None = None,
        subset_seed: int = 0,
        strict_protocol: bool = True,
        preflight: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split = canonical_sc09_split(split)
        if max_samples is not None and max_samples <= 0:
            raise ValueError(f"max_samples must be positive or None, got {max_samples}")
        if subset_seed < 0:
            raise ValueError(f"subset_seed must be non-negative, got {subset_seed}")
        self.strict_protocol = bool(strict_protocol)
        all_records = discover_sc09_records(self.root, strict=self.strict_protocol)
        record_by_path = {record.path: record for record in all_records}
        validation_paths, validation_hash = _read_split_list(self.root / "validation_list.txt")
        testing_paths, testing_hash = _read_split_list(self.root / "testing_list.txt")
        if self.strict_protocol:
            if validation_hash != SC09_VALIDATION_LIST_SHA256:
                raise ValueError(
                    "SC09 validation_list.txt hash does not match Speech Commands v0.02: "
                    f"{validation_hash} != {SC09_VALIDATION_LIST_SHA256}"
                )
            if testing_hash != SC09_TESTING_LIST_SHA256:
                raise ValueError(
                    "SC09 testing_list.txt hash does not match Speech Commands v0.02: "
                    f"{testing_hash} != {SC09_TESTING_LIST_SHA256}"
                )
        split_paths = partition_sc09_paths(
            [record.path for record in all_records],
            validation_paths=validation_paths,
            testing_paths=testing_paths,
            expected_counts=SC09_SPLIT_COUNTS if self.strict_protocol else None,
        )
        if self.strict_protocol:
            split_class_counts = {
                name: _class_counts(paths) for name, paths in split_paths.items()
            }
            observed_classes = {
                label: {name: counts[label] for name, counts in split_class_counts.items()}
                for label in SC09_LABELS
            }
            if observed_classes != SC09_CLASS_SPLIT_COUNTS:
                raise ValueError("SC09 per-digit split counts do not match Speech Commands v0.02")
            observed_split_hashes = {
                name: ordered_path_digest(paths) for name, paths in split_paths.items()
            }
            if observed_split_hashes != SC09_SPLIT_ENTRIES_SHA256:
                raise ValueError("SC09 split entries do not exactly match the pinned v0.02 digit archive")
            observed_records_hash = audio_records_fingerprint(all_records)
            if observed_records_hash != SC09_SOURCE_RECORDS_SHA256:
                raise ValueError("SC09 WAV paths/sizes do not exactly match the pinned v0.02 digit archive")
        population_paths = split_paths[self.split]
        selected_paths = list(deterministic_subset(population_paths, size=max_samples, seed=subset_seed))
        self.paths = selected_paths
        self.records = [record_by_path[path] for path in selected_paths]
        self._split_paths = split_paths
        self.metadata: dict[str, Any] = {
            "dataset_id": "sc09",
            "protocol": SC09_PROTOCOL,
            "split": self.split,
            "is_heldout": self.split in {"validation", "test"},
            "root": os.fspath(self.root),
            "population_count": len(population_paths),
            "selected_count": len(selected_paths),
            "source_count": len(all_records),
            "split_counts": {name: len(paths) for name, paths in split_paths.items()},
            "class_counts": _class_counts(population_paths),
            "selected_class_counts": _class_counts(selected_paths),
            "source_listing_sha256": ordered_path_digest(record.path for record in all_records),
            "source_records_sha256": audio_records_fingerprint(all_records),
            "split_entries_sha256": ordered_path_digest(population_paths),
            "selected_entries_sha256": ordered_path_digest(selected_paths),
            "selected_content_sha256": audio_content_fingerprint(self.root, self.records),
            "validation_list_sha256": validation_hash,
            "testing_list_sha256": testing_hash,
            "subset_seed": int(subset_seed),
            "max_samples": max_samples,
            "sample_rate": SC09_SAMPLE_RATE,
            "sample_length": SC09_SAMPLE_LENGTH,
            "channels": SC09_CHANNELS,
            "input_representation": "raw_waveform",
            "normalization": "signed_pcm16_div_32768",
            "length_policy": "leading_crop_then_right_zero_pad",
            "archive_url": SC09_ARCHIVE_URL,
            "archive_size": SC09_ARCHIVE_SIZE,
            "archive_sha256": SC09_ARCHIVE_SHA256,
            "dataset_commit": SC09_DATASET_COMMIT,
        }
        if preflight:
            self.metadata["preflight"] = preflight_sc09_dataset(self)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        waveform = load_sc09_waveform(self.root / record.path)
        return waveform, SC09_LABEL_TO_INDEX[record.label]


def preflight_sc09_dataset(dataset: SC09Dataset, *, full_decode: bool = False) -> dict[str, Any]:
    """Validate selected WAV headers and optionally hash normalized waveforms."""
    frame_lengths: list[int] = []
    content_digest = hashlib.sha256() if full_decode else None
    for index, record in enumerate(dataset.records):
        metadata = _wave_metadata(dataset.root / record.path)
        frames = int(metadata["frames"])
        frame_lengths.append(frames)
        if content_digest is not None:
            waveform, label = dataset[index]
            content_digest.update(record.path.encode("utf-8"))
            content_digest.update(b"\0")
            content_digest.update(waveform.numpy().tobytes(order="C"))
            content_digest.update(bytes([label]))
    return {
        "valid": True,
        "checked": len(dataset.records),
        "sample_rate": SC09_SAMPLE_RATE,
        "sample_length": SC09_SAMPLE_LENGTH,
        "channels": SC09_CHANNELS,
        "pcm_bits": 16,
        "min_source_frames": min(frame_lengths) if frame_lengths else None,
        "max_source_frames": max(frame_lengths) if frame_lengths else None,
        "right_padded": sum(frames < SC09_SAMPLE_LENGTH for frames in frame_lengths),
        "exact_length": sum(frames == SC09_SAMPLE_LENGTH for frames in frame_lengths),
        "leading_cropped": sum(frames > SC09_SAMPLE_LENGTH for frames in frame_lengths),
        "content_sha256": content_digest.hexdigest() if content_digest is not None else None,
    }


def build_sc09_manifest(dataset: SC09Dataset, *, preflight: Mapping[str, Any] | None = None) -> dict[str, Any]:
    manifest = {
        "format": SC09_MANIFEST_FORMAT,
        "dataset": dict(dataset.metadata),
        "labels": list(SC09_LABELS),
        "splits": {
            name: {
                "count": len(paths),
                "entries_sha256": ordered_path_digest(paths),
                "class_counts": _class_counts(paths),
                "is_heldout": name in {"validation", "test"},
            }
            for name, paths in dataset._split_paths.items()
        },
    }
    if preflight is not None:
        manifest["preflight"] = dict(preflight)
    return manifest


def write_sc09_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


__all__ = [
    "AudioRecord",
    "SC09_ARCHIVE_SHA256",
    "SC09_ARCHIVE_SIZE",
    "SC09_ARCHIVE_URL",
    "SC09_ARCHIVE_FILENAME",
    "SC09_CLASS_SPLIT_COUNTS",
    "SC09_DATASET_COMMIT",
    "SC09_LABELS",
    "SC09_PROTOCOL",
    "SC09_SAMPLE_LENGTH",
    "SC09_SAMPLE_RATE",
    "SC09_SOURCE_RECORDS_SHA256",
    "SC09_SPLIT_COUNTS",
    "SC09_SPLIT_ENTRIES_SHA256",
    "SC09Dataset",
    "audio_content_fingerprint",
    "audio_records_fingerprint",
    "build_sc09_manifest",
    "canonical_sc09_split",
    "discover_sc09_records",
    "load_sc09_waveform",
    "partition_sc09_paths",
    "preflight_sc09_dataset",
    "write_sc09_manifest",
]
