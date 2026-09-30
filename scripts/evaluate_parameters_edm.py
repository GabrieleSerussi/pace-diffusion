#!/usr/bin/env python3
"""
Measure noise-level-binned parameter usage in an EDM network using group ablations.

This is the U-Net profiler of PACE (Section 3.2 of the paper).  It targets
official EDM checkpoints from "Elucidating the Design Space of Diffusion-Based
Generative Models" (Karras et al., 2022) and the other supported teachers in
``pace.teacher_models``.

The defaults are the paper settings: one group per convolution output channel
(``--grouping per_filter``), activation permutation importance with a seeded,
fixed-point-free permutation shared by all groups at each exact noise level
(``--ablation_mode pfi``), and 20 noise bins (``--num_bins 20``).  The legacy
``random_same_norm`` and ``zero`` interventions remain available;
``random_same_norm`` produced the released CIFAR-10 and ImageNet-64 profiles.

Important note
--------------
This measures functional importance, not literal hardware parameter access.
The same frozen network parameters are executed for each noise level; the metric here
asks which groups matter most for denoising quality at different corruption levels.

Checkpoint support
------------------
Official EDM checkpoints are typically pickled networks whose custom classes live in
NVLabs EDM modules. That code must therefore be importable in the active Python
environment when loading a `.pkl` checkpoint.
"""

import argparse
import bisect
import gc
import glob
import hashlib
import io
import json
import math
import os
import pickle
import random
import shlex
import struct
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
import torch
import torch.distributed as dist
from PIL import Image
import pyarrow.parquet as pq
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from pace.dataset_specs import DEFAULT_LSUN_MONITOR_SEED, DEFAULT_LSUN_MONITOR_SIZE, FFHQ_PROTOCOL
from pace.evaluation_protocols import benchmark_protocol_records
from pace.image_datasets import ImageFolderFlat as SharedImageFolder, SharedImageDataset, validate_dataset_readability
from pace import parameter_analysis as shared_analysis

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:  # Optional: only the interactive HTML heatmaps of large matrices use plotly.
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - exercised only without plotly.
    go = None


# ---------------------------
# Reproducibility
# ---------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_invocation_command(argv: List[str]) -> str:
    return " ".join(shlex.quote(arg) for arg in argv)


def build_profile_cost_estimate(
    *,
    grouping: str,
    full_group_count: int,
    selected_group_count: int,
    bounded_by_max_groups: bool,
    dataset_images: int,
    sigma_levels_per_image: int,
    world_size: int,
    requested_batch_size: Optional[int] = None,
    actual_batch_sizes: Optional[List[int]] = None,
    homogeneous_batches_per_evaluation: Optional[int] = None,
) -> dict:
    """Return a conservative, serializable work estimate before ablations run."""

    values = (full_group_count, selected_group_count, dataset_images, sigma_levels_per_image)
    if any(value < 0 for value in values) or world_size <= 0:
        raise ValueError("profile cost dimensions must be non-negative and world_size must be positive")
    corrupted_examples = dataset_images * sigma_levels_per_image
    estimate = {
        "estimate_format": "diffdist_edm_profile_cost_estimate_v1",
        "grouping": grouping,
        "full_group_count": int(full_group_count),
        "selected_group_count": int(selected_group_count),
        "bounded_by_max_groups": bool(bounded_by_max_groups),
        "dataset_images": int(dataset_images),
        "sigma_levels_per_image": int(sigma_levels_per_image),
        "corrupted_examples_per_evaluation": int(corrupted_examples),
        "baseline_evaluations": 1,
        "ablation_evaluations": int(selected_group_count),
        "estimated_forward_examples": int((selected_group_count + 1) * corrupted_examples),
        "world_size": int(world_size),
        "estimated_max_group_evaluations_per_rank": int(math.ceil(selected_group_count / world_size)),
        "note": "Estimate excludes retries, checkpoint inspection, plotting, and data-loader overhead.",
    }
    if actual_batch_sizes is not None:
        if not actual_batch_sizes or any(size < 2 for size in actual_batch_sizes):
            raise ValueError("PFI actual batch sizes must be non-empty and at least two")
        expected_homogeneous_batches = len(actual_batch_sizes) * int(sigma_levels_per_image)
        if (
            homogeneous_batches_per_evaluation is not None
            and homogeneous_batches_per_evaluation != expected_homogeneous_batches
        ):
            raise ValueError(
                "PFI homogeneous batch count disagrees with the per-sigma batch plan: "
                f"{homogeneous_batches_per_evaluation} != {expected_homogeneous_batches}"
            )
        histogram: Dict[str, int] = {}
        for size in actual_batch_sizes:
            key = str(int(size))
            histogram[key] = histogram.get(key, 0) + 1
        estimate.update(
            {
                "requested_batch_size": int(requested_batch_size) if requested_batch_size is not None else None,
                "actual_batch_size_histogram": {
                    key: count * int(sigma_levels_per_image) for key, count in histogram.items()
                },
                "actual_batch_size_histogram_per_sigma": histogram,
                "actual_batch_size_min": int(min(actual_batch_sizes)),
                "actual_batch_size_max": int(max(actual_batch_sizes)),
                "homogeneous_batches_per_evaluation": int(homogeneous_batches_per_evaluation)
                if homogeneous_batches_per_evaluation is not None
                else expected_homogeneous_batches,
            }
        )
    return estimate


def resolve_shared_dataset_image_size(args: argparse.Namespace) -> int:
    """Resolve resolution from CLI/preset metadata without loading the teacher."""
    if args.image_size is not None:
        return int(args.image_size)
    from pace.teacher_models import resolve_teacher_spec

    teacher_spec = resolve_teacher_spec(
        args.network_pkl,
        network_format=args.network_format,
        preset=args.network_preset,
    )
    value = teacher_spec.model_config.get("image_size") or teacher_spec.model_config.get("img_resolution")
    if value is None:
        raise ValueError(
            "--image_size is required for FFHQ/LSUN when teacher resolution is not available from preset metadata"
        )
    return int(value)


def make_shared_analysis_dataset(args: argparse.Namespace, *, image_size: int) -> SharedImageDataset:
    dataset = SharedImageDataset(
        dataset_id=args.dataset,
        root=args.data_root,
        image_size=image_size,
        split=args.dataset_split,
        manifest=args.dataset_manifest,
        max_images=args.max_images,
        subset_seed=args.seed,
        preflight=args.dataset_preflight,
        ffhq_protocol=args.ffhq_protocol,
        lsun_monitor_size=args.lsun_monitor_size,
        lsun_monitor_seed=args.lsun_monitor_seed,
    )
    validate_dataset_readability(dataset)
    return dataset


# ---------------------------
# Dataset
# ---------------------------

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# Legacy flat-folder dataset: sorted os.listdir order, max_images prefix truncation,
# opt-in random horizontal flip (hflip=False is the historical pipeline). Kept under
# its historical name for the importers that rely on that exact contract
# (train_phase_students.py, precompute_latent_cache.py, evaluate_parameters_dit.py
# and their tests). This script's own `--dataset image_folder` path uses the shared
# recursive / ZIP-aware SharedImageFolder instead (see main()).
class ImageFolderFlat(Dataset):
    def __init__(self, root: str, image_size: int, max_images: Optional[int] = None,
                 hflip: bool = False):
        self.root = root
        self.paths = []
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if os.path.isfile(path) and name.lower().endswith(IMG_EXTS):
                self.paths.append(path)

        if max_images is not None:
            self.paths = self.paths[:max_images]

        if not self.paths:
            raise ValueError(f"No images found in {root}")

        # hflip=False (default): transform pipeline unchanged from before this flag
        # existed -> byte-identical batches. hflip=True (opt-in, --hflip in the
        # trainer; same convention as CIFAR10Dataset above) inserts a random
        # horizontal flip (p=0.5) -- used for small unconditional image_folder
        # datasets (e.g. FFHQ256) as data regularization.
        transform_steps = [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
        ]
        if hflip:
            transform_steps.append(transforms.RandomHorizontalFlip(p=0.5))
        transform_steps += [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
        self.transform = transforms.Compose(transform_steps)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image), -1


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        root: str,
        image_size: int,
        split: str = "validation",
        max_images: Optional[int] = None,
        download: bool = False,
        hflip: bool = False,
    ):
        split = split.lower()
        if split == "validation":
            train = False
        elif split == "test":
            train = False
        elif split == "train":
            train = True
        else:
            raise ValueError(f"Unsupported CIFAR-10 split: {split}")

        self.split = split
        self.dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
        )
        self.dataset_length = len(self.dataset)
        self.max_images = self.dataset_length if max_images is None else min(max_images, self.dataset_length)
        self.sample_seed = 0
        # hflip=False (default): transform pipeline unchanged from before this flag
        # existed -> byte-identical batches. hflip=True (opt-in, --hflip in the
        # trainer) inserts a random horizontal flip (p=0.5) into the CIFAR-10
        # TRAIN pipeline only -- callers building held-out/eval splits never pass
        # hflip=True (see build_val_loader / eval scripts).
        transform_steps = [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
        ]
        if hflip:
            transform_steps.append(transforms.RandomHorizontalFlip(p=0.5))
        transform_steps += [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
        self.transform = transforms.Compose(transform_steps)

    def set_sampling_seed(self, seed: int) -> None:
        self.sample_seed = seed

    def __len__(self) -> int:
        return self.max_images

    def __getitem__(self, idx: int):
        # Map logical indices to a seed-controlled permutation so each evaluation can
        # see a different subset without rebuilding the dataset wrapper.
        if self.max_images < self.dataset_length:
            rng = random.Random(self.sample_seed)
            offset = rng.randrange(self.dataset_length)
            stride = rng.randrange(1, self.dataset_length)
            while math.gcd(stride, self.dataset_length) != 1:
                stride += 1
                if stride >= self.dataset_length:
                    stride = 1
            dataset_idx = (offset + idx * stride) % self.dataset_length
        else:
            dataset_idx = idx
        image, label = self.dataset[dataset_idx]
        return self.transform(image), int(label)


def _infer_first_present(candidates: List[str], available: List[str]) -> Optional[str]:
    available_set = set(available)
    for name in candidates:
        if name in available_set:
            return name
    return None


def _decode_parquet_image(value):
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


def _normalize_imagenet_parquet_split(split: Optional[str]) -> Optional[str]:
    split = (split or "all").lower()
    if split in ("", "all"):
        return None
    if split == "val":
        return "validation"
    if split not in ("train", "validation", "test"):
        raise ValueError(
            "parquet split must be one of: all, train, validation, val, test"
        )
    return split


class ImageNet1KParquetDataset(Dataset):
    def __init__(
        self,
        root: str,
        image_size: int,
        max_images: Optional[int] = None,
        image_column: Optional[str] = None,
        label_column: Optional[str] = None,
        split: Optional[str] = "all",
        split_prefix: Optional[str] = None,
    ):
        self.root = root
        self.split = _normalize_imagenet_parquet_split(split)
        # An explicit split_prefix (e.g. 'train-') wins; otherwise derive it from split.
        if split_prefix is None and self.split is not None:
            split_prefix = f"{self.split}-"
        self.split_prefix = split_prefix
        self.paths = sorted(
            os.path.join(root, name)
            for name in os.listdir(root)
            if (
                os.path.isfile(os.path.join(root, name))
                and name.endswith(".parquet")
                and (split_prefix is None or name.startswith(split_prefix))
            )
        )
        if not self.paths:
            scope = f"in {root}" if split_prefix is None else f"matching '{split_prefix}*.parquet' in {root}"
            split_msg = "" if self.split is None else f" for split '{self.split}'"
            raise ValueError(f"No parquet files found {scope}{split_msg}")

        self._parquet_files = [pq.ParquetFile(path) for path in self.paths]
        # Use schema_arrow (high-level), not schema (low-level) -- HF datasets often store
        # image as a struct<bytes, path> whose low-level leaf names hide the top-level column.
        schema_names = self._parquet_files[0].schema_arrow.names
        self.image_column = image_column or _infer_first_present(
            ["image", "bytes", "jpg", "png", "jpeg", "webp", "path"], schema_names,
        )
        self.label_column = label_column or _infer_first_present(["label", "labels", "cls", "class", "fine_label"], schema_names)
        if self.image_column is None:
            raise ValueError(
                f"Could not infer image column from parquet schema columns {schema_names}. "
                "Pass --parquet_image_column explicitly."
            )
        if self.label_column is None:
            raise ValueError(
                f"Could not infer label column from parquet schema columns {schema_names}. "
                "Pass --parquet_label_column explicitly."
            )

        self.row_offsets = [0]
        for parquet_file in self._parquet_files:
            self.row_offsets.append(self.row_offsets[-1] + parquet_file.metadata.num_rows)
        self.dataset_length = self.row_offsets[-1]
        self.max_images = self.dataset_length if max_images is None else min(max_images, self.dataset_length)
        self.sample_seed = 0
        self._cached_file_idx: Optional[int] = None
        self._cached_table = None
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def set_sampling_seed(self, seed: int) -> None:
        self.sample_seed = seed

    def __len__(self) -> int:
        return self.max_images

    def _resolve_dataset_idx(self, idx: int) -> int:
        if self.max_images < self.dataset_length:
            rng = random.Random(self.sample_seed)
            offset = rng.randrange(self.dataset_length)
            stride = rng.randrange(1, self.dataset_length)
            while math.gcd(stride, self.dataset_length) != 1:
                stride += 1
                if stride >= self.dataset_length:
                    stride = 1
            return (offset + idx * stride) % self.dataset_length
        return idx

    def _load_row(self, dataset_idx: int):
        file_idx = bisect.bisect_right(self.row_offsets, dataset_idx) - 1
        local_idx = dataset_idx - self.row_offsets[file_idx]
        if self._cached_file_idx != file_idx or self._cached_table is None:
            self._cached_table = self._parquet_files[file_idx].read(columns=[self.image_column, self.label_column])
            self._cached_file_idx = file_idx
        row = self._cached_table.slice(local_idx, 1).to_pylist()[0]
        return row

    def __getitem__(self, idx: int):
        dataset_idx = self._resolve_dataset_idx(idx)
        row = self._load_row(dataset_idx)
        image = _decode_parquet_image(row[self.image_column])
        label = int(row[self.label_column])
        return self.transform(image), label


class ImageNetDataset(Dataset):
    """
    Load ImageNet data for evaluation. Supports two modes:

    1. **Local ImageFolder** (default): point ``--data_root`` at a directory that
       contains ``train/`` and/or ``val/`` sub-directories in the standard
       torchvision ImageFolder layout (``<split>/<class_name>/*.JPEG``).
    2. **HuggingFace datasets**: if the local path does not exist and
       ``--download`` is set, fall back to
       ``datasets.load_dataset("ILSVRC/imagenet-1k", ...)``.
    """

    def __init__(
        self,
        root: str,
        image_size: int,
        split: str = "val",
        max_images: Optional[int] = None,
        download: bool = False,
    ):
        self.split = split.lower()
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        split_dir = os.path.join(root, self.split)
        if os.path.isdir(split_dir):
            self.dataset = datasets.ImageFolder(root=split_dir)
            self.source = "local"
        elif download:
            try:
                from datasets import load_dataset as hf_load_dataset

                hf_split = "validation" if self.split == "val" else self.split
                self.dataset = hf_load_dataset(
                    "ILSVRC/imagenet-1k",
                    split=hf_split,
                    trust_remote_code=True,
                )
                self.source = "huggingface"
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load ImageNet from HuggingFace: {exc}. "
                    f"Provide a local directory at {split_dir} or authenticate "
                    "with `huggingface-cli login` and accept the ImageNet terms."
                ) from exc
        else:
            raise FileNotFoundError(
                f"ImageNet split directory not found: {split_dir}. "
                "Provide a valid --data_root with train/ and/or val/ subdirectories, "
                "or add --download to attempt loading from HuggingFace."
            )

        self.dataset_length = len(self.dataset)
        self.max_images = self.dataset_length if max_images is None else min(max_images, self.dataset_length)
        self.sample_seed = 0

    def set_sampling_seed(self, seed: int) -> None:
        self.sample_seed = seed

    def __len__(self) -> int:
        return self.max_images

    def _resolve_dataset_idx(self, idx: int) -> int:
        if self.max_images < self.dataset_length:
            rng = random.Random(self.sample_seed)
            offset = rng.randrange(self.dataset_length)
            stride = rng.randrange(1, self.dataset_length)
            while math.gcd(stride, self.dataset_length) != 1:
                stride += 1
                if stride >= self.dataset_length:
                    stride = 1
            return (offset + idx * stride) % self.dataset_length
        return idx

    def __getitem__(self, idx: int):
        dataset_idx = self._resolve_dataset_idx(idx)
        if self.source == "huggingface":
            item = self.dataset[dataset_idx]
            image = item["image"]
            label = int(item["label"])
        else:
            image, label = self.dataset[dataset_idx]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        return self.transform(image), label


class SigmaCorruptionDataset(Dataset):
    """
    Wraps an image dataset and pre-generates fixed (image_idx, sigma_idx, noise_seed) tuples.
    Each image is paired with every sigma value in `sigma_values`, and the same per-image noise
    seed is reused across that image's noise-level samples. This ensures baseline and ablation
    runs see identical corruption settings.

    ``order`` controls the sample ordering (which, with ``shuffle=False`` and a
    fixed ``batch_size``, determines each forward batch's composition):

    - ``"level_major"`` (default): samples grouped by sigma/timestep-level, so a
      batch contains many DIFFERENT images at the SAME level. This is required
      for permutation importance to be measured with ``t`` held fixed (the
      per-batch permutation then only swaps activations between images at the
      same level).
    - ``"image_major"``: legacy order (each image's levels are contiguous). With
      ``batch_size < num_levels`` a batch is a narrow t-window of ONE image, so
      the permutation swaps activations ACROSS t-levels and injects a spurious
      period-``batch_size`` ripple into n_eff. Kept only to reproduce old runs.
    """

    def __init__(
        self,
        image_dataset: Dataset,
        sigma_values: torch.Tensor,
        samples_per_image: int,
        seed: int = 0,
        sigma_stride: int = 1,
        order: str = "level_major",
    ):
        if sigma_stride <= 0:
            raise ValueError(f"sigma_stride must be positive, got {sigma_stride}")
        if order not in ("level_major", "image_major"):
            raise ValueError(f"order must be 'level_major' or 'image_major', got {order!r}")

        self.image_dataset = image_dataset
        self.sigma_values = sigma_values.detach().cpu().clone()
        self.order = order
        self.sigma_indices = list(range(0, len(self.sigma_values), sigma_stride))
        self.num_images = len(image_dataset)
        self.samples: List[Tuple[int, int, int]] = []
        rng = random.Random(seed)

        # Draw a fixed per-image noise seed first (independent of ordering) so the
        # exact same (image, level, noise) triples are produced in either order.
        noise_seeds = [rng.randrange(0, 2**31 - 1) for _ in range(self.num_images)]
        if order == "image_major":
            for image_idx in range(self.num_images):
                for sigma_idx in self.sigma_indices:
                    self.samples.append((image_idx, sigma_idx, noise_seeds[image_idx]))
        else:  # level_major
            for sigma_idx in self.sigma_indices:
                for image_idx in range(self.num_images):
                    self.samples.append((image_idx, sigma_idx, noise_seeds[image_idx]))

    def __len__(self) -> int:
        return len(self.samples)

    def sample_index(self, image_idx: int, sigma_position: int) -> int:
        """Flat index of ``(image_idx, sigma_position)`` in ``self.samples`` under the
        dataset's ``order`` (the PFI batch planner addresses samples through this)."""
        if not 0 <= image_idx < self.num_images:
            raise IndexError(image_idx)
        if not 0 <= sigma_position < len(self.sigma_indices):
            raise IndexError(sigma_position)
        if self.order == "image_major":
            return image_idx * len(self.sigma_indices) + sigma_position
        return sigma_position * self.num_images + image_idx

    def __getitem__(self, idx: Union[int, "PFISampleIndex"]):
        pfi_ref = idx if isinstance(idx, PFISampleIndex) else None
        if pfi_ref is not None:
            idx = pfi_ref.sample_index
        image_idx, sigma_idx, noise_seed = self.samples[idx]
        if pfi_ref is not None:
            if image_idx != pfi_ref.image_index or sigma_idx != pfi_ref.sigma_index:
                raise RuntimeError("PFI batch plan does not match the corruption dataset layout")
        item = self.image_dataset[image_idx]
        if isinstance(item, tuple):
            image = item[0]
            class_idx = int(item[1]) if len(item) > 1 else -1
        else:
            image = item
            class_idx = -1
        if pfi_ref is None:
            return image, class_idx, sigma_idx, noise_seed
        return (
            image,
            class_idx,
            sigma_idx,
            noise_seed,
            image_idx,
            pfi_ref.donor_position,
            pfi_ref.batch_ordinal,
        )


def collate_corruption(batch):
    images, class_indices, sigma_indices, noise_seeds = zip(*batch)
    return (
        torch.stack(images, dim=0),
        torch.tensor(class_indices, dtype=torch.long),
        torch.tensor(sigma_indices, dtype=torch.long),
        torch.tensor(noise_seeds, dtype=torch.long),
    )


PFI_PLAN_FORMAT = "diffdist_batch_local_exact_sigma_pfi_plan_v1"
PFI_CHECKPOINT_FORMAT = "diffdist_edm_pfi_checkpoint_v1"
PFI_BASELINE_FORMAT = "diffdist_edm_pfi_baseline_v1"
ABLATION_PROTOCOL_FORMAT = "diffdist_edm_ablation_protocol_v1"
PROFILE_FINGERPRINT_FORMAT = "diffdist_edm_profile_fingerprint_v1"
FILTER_SAMPLING_PROTOCOL_FORMAT = "diffdist_edm_filter_sampling_protocol_v1"
FILTER_SAMPLING_PROTOCOL_IDS = {
    "exhaustive": "per_filter_exhaustive_v1",
    "stratified_module": "per_module_hash_stratified_filter_sampling_v1",
}
FILTER_SAMPLING_MEMBERSHIP_DOMAIN = "diffdist_edm_per_module_filter_sample_v1"


def _stable_sha256_fields(*values: object) -> bytes:
    """Hash typed text fields without delimiter ambiguity or runtime RNG state."""
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
    return digest.digest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_filter_sampling_options(
    *,
    grouping: str,
    filter_sampling: str,
    filters_per_module: Optional[int],
    max_groups: Optional[int],
) -> None:
    """Validate the two group-selection mechanisms before expensive setup."""

    if filter_sampling not in FILTER_SAMPLING_PROTOCOL_IDS:
        raise ValueError(f"Unsupported filter_sampling mode: {filter_sampling!r}")
    if filter_sampling == "stratified_module":
        if grouping != "per_filter":
            raise ValueError("--filter_sampling stratified_module requires --grouping per_filter")
        if filters_per_module is None or filters_per_module <= 0:
            raise ValueError(
                "--filters_per_module is required and must be positive when "
                "--filter_sampling stratified_module is selected"
            )
        if max_groups is not None:
            raise ValueError(
                "--filter_sampling stratified_module cannot be combined with --max_groups; "
                "use one deterministic group selector"
            )
    elif filters_per_module is not None:
        raise ValueError("--filters_per_module is only valid with --filter_sampling stratified_module")


def _filter_group_identity(group_name: str) -> Tuple[str, int]:
    """Split a canonical ``<module>.filter_<index>`` group name."""

    try:
        module_name, raw_filter_index = str(group_name).rsplit(".filter_", 1)
        filter_index = int(raw_filter_index)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid per-filter group name: {group_name!r}") from exc
    if not module_name or filter_index < 0 or raw_filter_index != str(filter_index):
        raise ValueError(f"Invalid per-filter group name: {group_name!r}")
    return module_name, filter_index


def select_filter_groups(
    groups: Dict[str, Any],
    *,
    mode: str,
    filters_per_module: Optional[int],
    seed: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Select per-filter groups and return a self-contained sampling protocol.

    Stratified membership is SHA-256 ranked independently within every raw
    convolution module.  The selected output order is canonical, so neither
    input mapping order nor distributed world size can change selection,
    fingerprinting, or rank assignment.
    """

    if not groups:
        raise ValueError("Per-filter sampling requires at least one group")
    if mode not in FILTER_SAMPLING_PROTOCOL_IDS:
        raise ValueError(f"Unsupported filter_sampling mode: {mode!r}")
    if mode == "stratified_module" and (filters_per_module is None or filters_per_module <= 0):
        raise ValueError("filters_per_module must be positive for stratified_module sampling")
    if mode == "exhaustive" and filters_per_module is not None:
        raise ValueError("filters_per_module must be omitted for exhaustive sampling")

    modules: Dict[str, List[Tuple[int, str]]] = {}
    for group_name in groups:
        module_name, filter_index = _filter_group_identity(group_name)
        modules.setdefault(module_name, []).append((filter_index, group_name))

    selected_names: List[str] = []
    module_counts: Dict[str, Dict[str, Union[int, float]]] = {}
    selected_probabilities: Dict[str, float] = {}
    selected_expansion_weights: Dict[str, float] = {}
    protocol_id = FILTER_SAMPLING_PROTOCOL_IDS[mode]

    if mode == "exhaustive":
        # Preserve the collector's legacy ordering exactly in exhaustive mode.
        selected_names = list(groups)
    else:
        assert filters_per_module is not None
        for module_name in sorted(modules):
            population = modules[module_name]
            ranked = sorted(
                population,
                key=lambda item: (
                    _stable_sha256_fields(
                        FILTER_SAMPLING_MEMBERSHIP_DOMAIN,
                        int(seed),
                        module_name,
                        item[0],
                    ),
                    item[1],
                ),
            )
            chosen = ranked[: min(filters_per_module, len(ranked))]
            # Hash ranking determines membership.  A separate canonical order
            # makes the returned plan invariant to discovery/mapping order.
            selected_names.extend(name for _index, name in sorted(chosen, key=lambda item: (item[0], item[1])))

    selected_set = set(selected_names)
    for module_name in sorted(modules):
        population_count = len(modules[module_name])
        selected_count = sum(name in selected_set for _index, name in modules[module_name])
        if selected_count <= 0:
            raise RuntimeError(f"Filter sampling selected no groups for module {module_name!r}")
        inclusion_probability = selected_count / population_count
        expansion_weight = population_count / selected_count
        module_counts[module_name] = {
            "population_filter_count": int(population_count),
            "selected_filter_count": int(selected_count),
            "inclusion_probability": float(inclusion_probability),
            "expansion_weight": float(expansion_weight),
        }
        if mode == "stratified_module":
            for _filter_index, group_name in modules[module_name]:
                if group_name in selected_set:
                    selected_probabilities[group_name] = float(inclusion_probability)
                    selected_expansion_weights[group_name] = float(expansion_weight)

    selected_groups = {name: groups[name] for name in selected_names}
    canonical_population_names = sorted(groups)
    canonical_selected_names = sorted(selected_names)
    protocol: Dict[str, Any] = {
        "format": FILTER_SAMPLING_PROTOCOL_FORMAT,
        "protocol_id": protocol_id,
        "mode": mode,
        "seed": int(seed),
        "filters_per_module": int(filters_per_module) if filters_per_module is not None else None,
        "population_group_count": int(len(groups)),
        "selected_group_count": int(len(selected_groups)),
        "population_module_count": int(len(modules)),
        "selected_module_count": int(sum(record["selected_filter_count"] > 0 for record in module_counts.values())),
        "module_counts": module_counts,
        "population_sha256": canonical_json_sha256(canonical_population_names),
        "selection_sha256": canonical_json_sha256(canonical_selected_names),
        "population_digest_encoding": "sha256(canonical_json(sorted_full_group_names))",
        "selection_digest_encoding": "sha256(canonical_json(sorted_selected_group_names))",
        "membership_hash_algorithm": "sha256",
        "membership_hash_field_encoding": "uint64_be_length_prefixed_utf8",
        "membership_hash_domain": FILTER_SAMPLING_MEMBERSHIP_DOMAIN,
        "membership_hash_fields": [
            "membership_hash_domain",
            "seed",
            "raw_module_name",
            "filter_index",
        ],
        "membership_hash_tie_breaker": "full_group_name_ascending",
        "full_group_name_encoding": "<raw_module_name>.filter_<base10_filter_index>",
        "selection_scope": "independent_per_raw_module",
        "selected_group_order": "module_name_ascending_then_filter_index_ascending"
        if mode == "stratified_module"
        else "legacy_collector_order",
    }
    if mode == "stratified_module":
        protocol.update(
            {
                "selected_group_names": selected_names,
                "selected_group_inclusion_probabilities": {
                    name: selected_probabilities[name] for name in selected_names
                },
                "selected_group_expansion_weights": {
                    name: selected_expansion_weights[name] for name in selected_names
                },
            }
        )
    return selected_groups, protocol


def balanced_batch_sizes(population_size: int, requested_batch_size: int) -> List[int]:
    """Partition a population without drops or singleton PFI batches."""
    if population_size < 2:
        raise ValueError(f"PFI requires at least two distinct examples, got {population_size}")
    if requested_batch_size < 2:
        raise ValueError(f"PFI batch size must be at least 2, got {requested_batch_size}")
    num_batches = math.ceil(population_size / requested_batch_size)
    if population_size // num_batches < 2:
        raise ValueError(
            f"Cannot partition N={population_size} into no-drop PFI batches of size at most "
            f"{requested_batch_size} without a singleton"
        )
    base, extra = divmod(population_size, num_batches)
    sizes = [base + 1] * extra + [base] * (num_batches - extra)
    if sum(sizes) != population_size or min(sizes) < 2 or max(sizes) > requested_batch_size:
        raise RuntimeError(f"Could not form non-singleton PFI batches for N={population_size}")
    return sizes


@dataclass(frozen=True, slots=True)
class PFISampleIndex:
    sample_index: int
    image_index: int
    sigma_index: int
    donor_position: int
    batch_ordinal: int


class ExactSigmaPFIBatchSampler(Sampler[List[PFISampleIndex]]):
    """Deterministic, balanced batches and derangements at one exact sigma.

    Batch membership is independently SHA-256 ranked at every selected sigma.
    Within every batch, a second SHA-256 ranking defines a single-cycle
    fixed-point-free permutation.  The fully materialized compact plan is reused
    by every DataLoader traversal, group, worker count, and distributed rank.
    """

    def __init__(
        self,
        dataset: SigmaCorruptionDataset,
        *,
        batch_size: int,
        pfi_seed: int,
        population_fingerprint: str,
    ) -> None:
        self.dataset = dataset
        self.requested_batch_size = int(batch_size)
        self.pfi_seed = int(pfi_seed)
        self.population_fingerprint = str(population_fingerprint)
        self.batch_sizes = balanced_batch_sizes(dataset.num_images, self.requested_batch_size)
        self.batch_offsets = [0]
        for size in self.batch_sizes:
            self.batch_offsets.append(self.batch_offsets[-1] + size)

        num_levels = len(dataset.sigma_indices)
        num_images = dataset.num_images
        self._member_orders = torch.empty((num_levels, num_images), dtype=torch.int32)
        self._donor_positions = torch.empty((num_levels, num_images), dtype=torch.int32)
        plan_digest = hashlib.sha256()
        header = {
            "format": PFI_PLAN_FORMAT,
            "pfi_seed": self.pfi_seed,
            "population_fingerprint": self.population_fingerprint,
            "population_size": num_images,
            "sigma_indices": list(dataset.sigma_indices),
            "requested_batch_size": self.requested_batch_size,
            "balanced_batch_sizes": list(self.batch_sizes),
        }
        plan_digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8"))

        for sigma_position, sigma_index in enumerate(dataset.sigma_indices):
            members = sorted(
                range(num_images),
                key=lambda image_index: (
                    _stable_sha256_fields(
                        PFI_PLAN_FORMAT,
                        "membership",
                        self.pfi_seed,
                        self.population_fingerprint,
                        sigma_index,
                        image_index,
                    ),
                    image_index,
                ),
            )
            self._member_orders[sigma_position] = torch.tensor(members, dtype=torch.int32)
            plan_digest.update(struct.pack(">Q", int(sigma_index)))
            for batch_in_level, (start, end) in enumerate(zip(self.batch_offsets, self.batch_offsets[1:])):
                batch_members = members[start:end]
                ranked_positions = sorted(
                    range(len(batch_members)),
                    key=lambda position: (
                        _stable_sha256_fields(
                            PFI_PLAN_FORMAT,
                            "derangement",
                            self.pfi_seed,
                            self.population_fingerprint,
                            sigma_index,
                            batch_in_level,
                            batch_members[position],
                        ),
                        batch_members[position],
                    ),
                )
                donor_positions = [0] * len(batch_members)
                for rank, recipient_position in enumerate(ranked_positions):
                    donor_positions[recipient_position] = ranked_positions[(rank + 1) % len(ranked_positions)]
                if sorted(donor_positions) != list(range(len(batch_members))):
                    raise RuntimeError("PFI donor mapping is not a permutation")
                if any(position == donor for position, donor in enumerate(donor_positions)):
                    raise RuntimeError("PFI donor mapping contains a fixed point")
                self._donor_positions[sigma_position, start:end] = torch.tensor(
                    donor_positions, dtype=torch.int32
                )
                plan_digest.update(struct.pack(">QI", batch_in_level, len(batch_members)))
                for image_index, donor_position in zip(batch_members, donor_positions):
                    plan_digest.update(struct.pack(">QI", image_index, donor_position))

        self.plan_sha256 = plan_digest.hexdigest()
        batch_size_histogram: Dict[str, int] = {}
        for size in self.batch_sizes:
            key = str(int(size))
            batch_size_histogram[key] = batch_size_histogram.get(key, 0) + 1
        self.artifact = {
            **header,
            "batches_per_sigma": len(self.batch_sizes),
            "total_batches": num_levels * len(self.batch_sizes),
            "homogeneous_batches_per_evaluation": num_levels * len(self.batch_sizes),
            "actual_batch_size_histogram": {
                key: count * num_levels for key, count in batch_size_histogram.items()
            },
            "actual_batch_size_histogram_per_sigma": batch_size_histogram,
            "actual_batch_size_min": min(self.batch_sizes),
            "actual_batch_size_max": max(self.batch_sizes),
            "compact_table_dtype": "int32",
            "compact_table_shape": [num_levels, num_images],
            "member_orders_encoding": (
                "row follows sigma_indices; column is global position within that sigma's balanced plan; "
                "value is logical image index"
            ),
            "donor_positions_encoding": (
                "row follows sigma_indices; column follows member_orders; value is the donor's zero-based "
                "local position within the same balanced batch"
            ),
            "hash_algorithm": "sha256",
            "hash_field_encoding": "uint64_be_length_prefixed_utf8",
            "membership_hash_fields": [
                "format",
                "membership_domain",
                "pfi_seed",
                "population_fingerprint",
                "exact_sigma_index",
                "image_index",
            ],
            "derangement_hash_fields": [
                "format",
                "derangement_domain",
                "pfi_seed",
                "population_fingerprint",
                "exact_sigma_index",
                "batch_ordinal_within_sigma",
                "image_index",
            ],
            "hash_tie_breaker": "image_index_ascending",
            "membership": "sha256-ranked independently per exact sigma",
            "derangement": "sha256-ranked single cycle within each batch",
            "remainder_policy": "balanced_no_drop_no_singletons",
            "class_stratified": False,
            "fixed_points": 0,
            "plan_sha256": self.plan_sha256,
        }

    def __len__(self) -> int:
        return len(self.dataset.sigma_indices) * len(self.batch_sizes)

    def __iter__(self) -> Iterator[List[PFISampleIndex]]:
        batches_per_sigma = len(self.batch_sizes)
        for sigma_position, sigma_index in enumerate(self.dataset.sigma_indices):
            members = self._member_orders[sigma_position]
            donors = self._donor_positions[sigma_position]
            for batch_in_level, (start, end) in enumerate(zip(self.batch_offsets, self.batch_offsets[1:])):
                batch_ordinal = sigma_position * batches_per_sigma + batch_in_level
                yield [
                    PFISampleIndex(
                        sample_index=self.dataset.sample_index(int(members[position]), sigma_position),
                        image_index=int(members[position]),
                        sigma_index=int(sigma_index),
                        donor_position=int(donors[position]),
                        batch_ordinal=batch_ordinal,
                    )
                    for position in range(start, end)
                ]


def collate_corruption_pfi(batch):
    if len(batch) < 2:
        raise ValueError("PFI batches must contain at least two examples")
    images, class_indices, sigma_indices, noise_seeds, image_indices, donor_positions, batch_ordinals = zip(*batch)
    if len(set(sigma_indices)) != 1:
        raise ValueError(f"PFI batch crosses exact sigma levels: {sorted(set(sigma_indices))}")
    if len(set(batch_ordinals)) != 1:
        raise ValueError("PFI batch contains records from multiple plan batches")
    if len(set(image_indices)) != len(image_indices):
        raise ValueError("PFI batch must contain distinct calibration examples")
    expected = list(range(len(batch)))
    if sorted(donor_positions) != expected:
        raise ValueError("PFI donor positions must form a bijection over the batch")
    if any(position == donor for position, donor in enumerate(donor_positions)):
        raise ValueError("PFI donor permutation must be fixed-point-free")
    return (
        torch.stack(images, dim=0),
        torch.tensor(class_indices, dtype=torch.long),
        torch.tensor(sigma_indices, dtype=torch.long),
        torch.tensor(noise_seeds, dtype=torch.long),
        torch.tensor(donor_positions, dtype=torch.long),
    )


def image_population_fingerprint(image_dataset: Dataset) -> str:
    metadata = getattr(image_dataset, "metadata", {})
    record = {
        "dataset_class": image_dataset.__class__.__name__,
        "count": len(image_dataset),
        "entries_sha256": metadata.get("entries_sha256") if isinstance(metadata, dict) else None,
        "source_listing_sha256": metadata.get("source_listing_sha256") if isinstance(metadata, dict) else None,
        "source_records_sha256": metadata.get("source_records_sha256") if isinstance(metadata, dict) else None,
        "split": metadata.get("split") if isinstance(metadata, dict) else getattr(image_dataset, "split", None),
    }
    return canonical_json_sha256(record)


def build_ablation_protocol(
    ablation_mode: "AblationMode",
    *,
    pfi_plan: Optional[ExactSigmaPFIBatchSampler] = None,
    pfi_seed: Optional[int] = None,
) -> Dict[str, Any]:
    replacement = {
        "zero": "zeros",
        "random_same_norm": "gaussian_random_noise_with_per_example_l2_norm",
        "pfi": "batch_local_exact_sigma_whole_group_activation_exchange",
    }[ablation_mode]
    record: Dict[str, Any] = {
        "format": ABLATION_PROTOCOL_FORMAT,
        "protocol_id": "batch_local_exact_sigma_pfi_v1" if ablation_mode == "pfi" else f"legacy_{ablation_mode}_v1",
        "mode": ablation_mode,
        "replacement": replacement,
        "score": "mean_ablated_loss_minus_mean_paired_baseline_loss",
        "signed_scores_retained": ablation_mode == "pfi",
        "allocation_uses_positive_part": True,
    }
    if ablation_mode == "pfi":
        if pfi_plan is None or pfi_seed is None:
            raise ValueError("PFI ablation protocol requires its deterministic batch plan and seed")
        record["pfi"] = {
            "estimand": "batch_local_exact_sigma_activation_permutation_importance",
            "seed": int(pfi_seed),
            "exact_sigma_conditioned": True,
            "whole_group_tensor": True,
            "fixed_point_free": True,
            "same_plan_for_all_groups_and_ranks": True,
            "class_stratified": False,
            "permutation_repetitions": 1,
            "compact_plan_artifact": "pfi_plan.pt",
            "plan": dict(pfi_plan.artifact),
        }
    return record


# ---------------------------
# Helpers
# ---------------------------

def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def choose_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def level_to_bin(indices: torch.Tensor, num_levels: int, num_bins: int) -> torch.Tensor:
    bins = torch.floor(indices.float() * num_bins / num_levels).long()
    return torch.clamp(bins, min=0, max=num_bins - 1)


def mse_per_example(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, pred.ndim))
    return ((pred - target) ** 2).mean(dim=dims)


def sanitize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return shared_analysis.sanitize_tensor(tensor)


def compute_signed_and_positive_deltas(
    ablated_means: Dict[str, torch.Tensor],
    baseline_mean: torch.Tensor,
    group_names: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Retain the signed estimand while deriving legacy allocation inputs."""
    return shared_analysis.compute_signed_and_positive_deltas(
        ablated_means,
        baseline_mean,
        group_names,
    )


def make_log_sigma_schedule(sigma_min: float, sigma_max: float, num_levels: int, device: torch.device) -> torch.Tensor:
    if num_levels < 2:
        return torch.tensor([sigma_max], dtype=torch.float32, device=device)
    start = math.log(float(sigma_max))
    end = math.log(float(sigma_min))
    return torch.exp(torch.linspace(start, end, num_levels, device=device, dtype=torch.float32))


def infer_model_family(net: torch.nn.Module) -> Literal["edm", "vp", "ve"]:
    class_name = net.__class__.__name__.lower()
    if "edmprecond" in class_name:
        return "edm"
    if "vpprecond" in class_name:
        return "vp"
    if "veprecond" in class_name:
        return "ve"
    return "edm"


def make_sigma_schedule_for_family(
    net: torch.nn.Module,
    model_family: Literal["edm", "vp", "ve"],
    sigma_min: float,
    sigma_max: float,
    num_levels: int,
    device: torch.device,
) -> torch.Tensor:
    if model_family == "vp" and hasattr(net, "sigma") and hasattr(net, "epsilon_t"):
        epsilon_t = float(getattr(net, "epsilon_t"))
        if num_levels < 2:
            t_values = torch.tensor([1.0], dtype=torch.float32, device=device)
        else:
            t_values = torch.linspace(1.0, epsilon_t, num_levels, device=device, dtype=torch.float32)
        sigma_values = net.sigma(t_values)
        return torch.as_tensor(sigma_values, dtype=torch.float32, device=device)
    return make_log_sigma_schedule(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        num_levels=num_levels,
        device=device,
    )


def loss_weights_for_family(
    sigmas: torch.Tensor,
    model_family: Literal["edm", "vp", "ve"],
    sigma_data: float,
) -> torch.Tensor:
    if model_family in ("vp", "ve"):
        return 1.0 / (sigmas ** 2)
    return (sigmas ** 2 + sigma_data ** 2) / ((sigmas * sigma_data) ** 2)


def make_class_labels(
    batch_size: int,
    label_dim: int,
    class_idx: Optional[int],
    device: torch.device,
    dataset_class_indices: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    if label_dim <= 0:
        return None

    if class_idx is not None:
        if class_idx < 0 or class_idx >= label_dim:
            raise ValueError(f"class_idx must be in [0, {label_dim - 1}], got {class_idx}")
        labels = torch.zeros(batch_size, label_dim, device=device, dtype=torch.float32)
        labels[:, class_idx] = 1.0
        return labels

    if dataset_class_indices is None:
        raise ValueError(
            "This EDM checkpoint is class-conditional. Provide --class_idx or use a labeled dataset like CIFAR-10."
        )

    dataset_class_indices = dataset_class_indices.to(device=device, dtype=torch.long)
    if (dataset_class_indices < 0).any():
        raise ValueError(
            "This EDM checkpoint is class-conditional, but the current dataset does not provide labels."
        )
    if (dataset_class_indices >= label_dim).any():
        bad = dataset_class_indices[dataset_class_indices >= label_dim][:5].tolist()
        raise ValueError(
            f"Dataset labels exceed label_dim={label_dim}. Example offending labels: {bad}"
        )

    labels = torch.zeros(batch_size, label_dim, device=device, dtype=torch.float32)
    labels.scatter_(1, dataset_class_indices.unsqueeze(1), 1.0)
    return labels


def make_sigma_bin_labels(sigma_values: torch.Tensor, num_bins: int) -> List[str]:
    labels: List[str] = []
    num_levels = len(sigma_values)
    sigma_values_cpu = sigma_values.detach().cpu()
    for bin_idx in range(num_bins):
        start_idx = (bin_idx * num_levels) // num_bins
        end_idx = ((bin_idx + 1) * num_levels) // num_bins - 1
        end_idx = max(start_idx, end_idx)
        start_sigma = float(sigma_values_cpu[start_idx].item())
        end_sigma = float(sigma_values_cpu[end_idx].item())
        labels.append(f"{start_sigma:.4g}-{end_sigma:.4g}" if start_idx != end_idx else f"{start_sigma:.4g}")
    return labels


# ---------------------------
# Group selection
# ---------------------------

def collect_block_groups_edm(net: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    groups: Dict[str, torch.nn.Module] = {}
    for name, module in net.named_modules():
        if not name:
            continue
        class_name = module.__class__.__name__.lower()
        if "block" in class_name and count_parameters(module) > 0:
            groups[name] = module
    return groups


def collect_attention_groups_edm(net: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    groups: Dict[str, torch.nn.Module] = {}
    openai_format = getattr(net, "checkpoint_format", None) == "openai_consistency_edm_state_dict_v1"
    for name, module in net.named_modules():
        if not name:
            continue
        class_name = module.__class__.__name__.lower()
        if openai_format:
            if class_name == "attentionblock":
                groups[name] = module
        elif "attention" in class_name or class_name.startswith("attn"):
            groups[name] = module
    return groups


def _get_module_num_heads(module: torch.nn.Module) -> Optional[int]:
    for attr in ("num_heads", "heads", "n_heads"):
        value = getattr(module, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def collect_attention_head_groups_edm(
    net: torch.nn.Module,
) -> Dict[str, Union[Tuple[torch.nn.Module, int], Tuple[torch.nn.Module, int, int]]]:
    groups: Dict[str, Union[Tuple[torch.nn.Module, int], Tuple[torch.nn.Module, int, int]]] = {}
    if getattr(net, "checkpoint_format", None) == "openai_consistency_edm_state_dict_v1":
        named_modules = dict(net.named_modules())
        for name, module in named_modules.items():
            if module.__class__.__name__.lower() != "qkvflashattention":
                continue
            num_heads = _get_module_num_heads(module)
            if num_heads is None:
                continue
            parent = named_modules.get(name.rsplit(".attention", 1)[0])
            module._diffdist_attention_parameter_count = count_parameters(parent) if parent is not None else 0
            for head_idx in range(num_heads):
                groups[f"{name}.head_{head_idx}"] = (module, head_idx, 1)
        return groups

    for name, module in collect_attention_groups_edm(net).items():
        num_heads = _get_module_num_heads(module)
        if num_heads is None:
            continue
        for head_idx in range(num_heads):
            groups[f"{name}.head_{head_idx}"] = (module, head_idx)
    return groups


def _is_conv_like_module(module: torch.nn.Module) -> bool:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return False
    if weight.ndim < 3:
        return False
    class_name = module.__class__.__name__.lower()
    if "conv" in class_name:
        return True
    return hasattr(module, "up") or hasattr(module, "down") or hasattr(module, "resample_filter")


def collect_per_filter_groups_edm(net: torch.nn.Module) -> Dict[str, Tuple[torch.nn.Module, int, str]]:
    groups: Dict[str, Tuple[torch.nn.Module, int, str]] = {}
    for name, module in net.named_modules():
        if not name:
            continue
        if not _is_conv_like_module(module):
            continue
        weight = getattr(module, "weight", None)
        out_channels = int(weight.shape[0]) if isinstance(weight, torch.Tensor) else 0
        if out_channels <= 0:
            continue
        for filter_idx in range(out_channels):
            groups[f"{name}.filter_{filter_idx}"] = (module, filter_idx, "filter")
    return groups


def count_filter_parameters(module: torch.nn.Module, filter_idx: int) -> int:
    weight = getattr(module, "weight", None)
    if weight is None:
        return 0
    count = int(weight[filter_idx].numel())
    bias = getattr(module, "bias", None)
    if bias is not None:
        count += 1
    return count


# ---------------------------
# Ablation hook
# ---------------------------

AblationMode = Literal["zero", "random_same_norm", "permutation", "pfi"]
AblationTarget = Union[
    torch.nn.Module,
    Tuple[torch.nn.Module, int],
    Tuple[torch.nn.Module, int, Union[str, int]],
]


def random_same_norm_like(tensor: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    if tensor.numel() == 0:
        return torch.zeros_like(tensor)

    original_dtype = tensor.dtype
    tensor_f32 = tensor.detach().to(dtype=torch.float32)
    noise = torch.randn(
        tensor.shape,
        device=tensor.device,
        dtype=torch.float32,
        generator=generator,
    )

    if tensor.ndim == 0:
        flat_tensor = tensor_f32.reshape(1, -1)
        flat_noise = noise.reshape(1, -1)
    else:
        flat_tensor = tensor_f32.reshape(tensor.shape[0], -1)
        flat_noise = noise.reshape(tensor.shape[0], -1)

    target_norm = torch.linalg.vector_norm(flat_tensor, ord=2, dim=1, keepdim=True)
    noise_norm = torch.linalg.vector_norm(flat_noise, ord=2, dim=1, keepdim=True)
    valid = (
        torch.isfinite(target_norm)
        & torch.isfinite(noise_norm)
        & (target_norm > 0)
        & (noise_norm > 0)
    )

    scale = torch.zeros_like(target_norm)
    scale[valid] = target_norm[valid] / noise_norm[valid]
    replaced = (flat_noise * scale).reshape(tensor.shape)
    return replaced.to(dtype=original_dtype)


class RandomSameNormMixin:
    def __init__(self, random_seed: int):
        self.random_seed = int(random_seed)
        self.generators: Dict[str, torch.Generator] = {}

    def _generator_for_device(self, device: torch.device) -> torch.Generator:
        key = str(device)
        generator = self.generators.get(key)
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.random_seed)
            self.generators[key] = generator
        return generator

    def _random_same_norm(self, tensor: torch.Tensor) -> torch.Tensor:
        return random_same_norm_like(tensor, generator=self._generator_for_device(tensor.device))


def permute_along_batch_like(tensor: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Permutation-feature-importance corruption.

    Shuffle ``tensor`` across the batch dimension (dim 0), so each example
    receives another example's activation for this group. This breaks the
    per-example correspondence between the network input and the group's
    activation while leaving the group's *marginal* activation distribution
    exactly unchanged (it is a permutation, not noise). The increase in loss is
    Breiman's permutation importance applied at the activation level.
    """
    if not torch.is_tensor(tensor) or tensor.numel() == 0 or tensor.shape[0] <= 1:
        return tensor.clone() if torch.is_tensor(tensor) else tensor
    perm = torch.randperm(tensor.shape[0], device=tensor.device, generator=generator)
    return tensor[perm].clone()


def permute_along_batch_within_groups(
    tensor: torch.Tensor, group_ids: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """Permutation-FI corruption restricted to batch elements sharing a group id.

    Identical to :func:`permute_along_batch_like` except each example only ever
    receives another example *from the same group*. When the group id is the
    diffusion/flow timestep-level of each sample, this keeps ``t`` FIXED while
    breaking the per-example input<->activation correspondence — the correct
    Breiman permutation importance for a t-resolved measurement.

    Motivation: with the plain batch permutation, whenever a forward batch spans
    several t-levels (image-major sample order + ``batch_size < num_levels``), a
    head's activation is swapped between DIFFERENT t-levels of the same image.
    The measured "importance" then leaks the head's t-sensitivity, peaking at the
    edges of each ``batch_size``-wide t-window and producing a spurious period-
    ``batch_size`` ripple in the per-level / per-bin ``n_eff`` curve. Restricting
    the permutation to same-t groups removes that ripple and makes the result
    invariant to ``batch_size`` and sample ordering.
    """
    if not torch.is_tensor(tensor) or tensor.numel() == 0 or tensor.shape[0] <= 1:
        return tensor.clone() if torch.is_tensor(tensor) else tensor
    b = tensor.shape[0]
    if group_ids is None or int(group_ids.shape[0]) != b:
        # Fall back to a plain batch permutation when no compatible grouping is
        # available (preserves behaviour for callers that do not set groups).
        return permute_along_batch_like(tensor, generator=generator)
    gid = group_ids.to(device=tensor.device).reshape(-1)
    perm = torch.arange(b, device=tensor.device)
    for g in torch.unique(gid, sorted=True):
        idx = torch.nonzero(gid == g, as_tuple=False).flatten()
        if idx.numel() > 1:
            shuffled = idx[torch.randperm(idx.numel(), generator=generator, device=tensor.device)]
            perm[idx] = shuffled
    return tensor[perm].clone()


class PermutationMixin:
    """Holds a per-device seeded generator so permutations are reproducible.

    The generator persists across the whole evaluation loop, so each forward
    batch draws a fresh (but seed-determined) permutation — matching the
    advancing-generator semantics of ``RandomSameNormMixin``.

    If :meth:`set_permutation_groups` has supplied a per-example group id for the
    current batch (typically the timestep-level), permutations are restricted to
    within-group so ``t`` is held fixed (see
    :func:`permute_along_batch_within_groups`); otherwise the plain full-batch
    permutation is used.
    """

    def __init__(self, random_seed: int):
        self.random_seed = int(random_seed)
        self.generators: Dict[str, torch.Generator] = {}
        self._group_ids: Optional[torch.Tensor] = None

    def _generator_for_device(self, device: torch.device) -> torch.Generator:
        key = str(device)
        generator = self.generators.get(key)
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.random_seed)
            self.generators[key] = generator
        return generator

    def set_permutation_groups(self, group_ids: Optional[torch.Tensor]) -> None:
        """Set the current batch's per-example grouping for within-group
        permutation (e.g. the timestep-level index tensor). Pass ``None`` to
        restore the plain full-batch permutation for subsequent forwards."""
        self._group_ids = None if group_ids is None else group_ids.detach().reshape(-1)

    def _permute_along_batch(self, tensor: torch.Tensor) -> torch.Tensor:
        generator = self._generator_for_device(tensor.device)
        if self._group_ids is not None:
            return permute_along_batch_within_groups(tensor, self._group_ids, generator=generator)
        return permute_along_batch_like(tensor, generator=generator)


class PermutationOutputHook(PermutationMixin):
    def __init__(self, module: torch.nn.Module, random_seed: int):
        super().__init__(random_seed=random_seed)
        self.module = module
        self.handle = None

    def _hook(self, module, inputs, output):
        if torch.is_tensor(output):
            return self._permute_along_batch(output)
        if isinstance(output, tuple):
            return tuple(self._permute_along_batch(item) if torch.is_tensor(item) else item for item in output)
        return output

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class HeadPermutationHook(PermutationMixin):
    """Permutation FI for one attention head whose output channels are laid out
    as equal contiguous head chunks. The head's slice is permuted across the
    batch (post-output, EDM-style fallback)."""

    def __init__(self, module: torch.nn.Module, head_idx: int, random_seed: int):
        super().__init__(random_seed=random_seed)
        self.module = module
        self.head_idx = head_idx
        self.handle = None

    def _ablate_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim < 3:
            raise ValueError(
                f"Cannot ablate attention head for output with shape {tuple(tensor.shape)}"
            )
        num_heads = _get_module_num_heads(self.module)
        if num_heads is None:
            raise ValueError(
                f"Module {self.module.__class__.__name__} does not expose a head count"
            )
        channel_dim = 1 if tensor.ndim == 4 else tensor.ndim - 1
        width = tensor.shape[channel_dim]
        if width % num_heads != 0:
            raise ValueError(
                f"Output width {width} is not divisible by num_heads={num_heads} for "
                f"{self.module.__class__.__name__}"
            )
        head_width = width // num_heads
        start = self.head_idx * head_width
        end = start + head_width
        output = tensor.clone()
        slicer = [slice(None)] * output.ndim
        slicer[channel_dim] = slice(start, end)
        selected = output[tuple(slicer)]
        output[tuple(slicer)] = self._permute_along_batch(selected)
        return output

    def _hook(self, module, inputs, output):
        if torch.is_tensor(output):
            return self._ablate_tensor(output)
        if isinstance(output, tuple):
            return tuple(self._ablate_tensor(item) if torch.is_tensor(item) else item for item in output)
        return output

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class FilterPermutationHook(PermutationMixin):
    def __init__(self, module: torch.nn.Module, filter_idx: int, random_seed: int):
        super().__init__(random_seed=random_seed)
        self.module = module
        self.filter_idx = filter_idx
        self.handle = None

    def _ablate_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim < 2:
            raise ValueError(
                f"Cannot ablate convolution filter for output with shape {tuple(tensor.shape)}"
            )
        channel_dim = 1
        num_channels = tensor.shape[channel_dim]
        if self.filter_idx < 0 or self.filter_idx >= num_channels:
            raise ValueError(
                f"Filter index {self.filter_idx} is out of range for output with {num_channels} channels"
            )
        output = tensor.clone()
        slicer = [slice(None)] * output.ndim
        slicer[channel_dim] = self.filter_idx
        selected = output[tuple(slicer)]
        output[tuple(slicer)] = self._permute_along_batch(selected)
        return output

    def _hook(self, module, inputs, output):
        if torch.is_tensor(output):
            return self._ablate_tensor(output)
        if isinstance(output, tuple):
            return tuple(self._ablate_tensor(item) if torch.is_tensor(item) else item for item in output)
        return output

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class PFIPermutationMixin:
    """Apply the externally planned donor mapping along activation batch dim 0."""

    def __init__(self) -> None:
        self._pfi_permutation: Optional[torch.Tensor] = None

    def set_pfi_permutation(self, permutation: torch.Tensor) -> None:
        permutation = permutation.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        size = int(permutation.numel())
        if size < 2 or sorted(permutation.tolist()) != list(range(size)):
            raise ValueError("PFI permutation must be a bijection over at least two batch rows")
        if torch.equal(permutation, torch.arange(size, dtype=torch.long)) or bool(
            torch.any(permutation == torch.arange(size, dtype=torch.long))
        ):
            raise ValueError("PFI permutation must be fixed-point-free")
        self._pfi_permutation = permutation

    def _pfi_replace(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._pfi_permutation is None:
            raise RuntimeError("PFI hook was invoked before the batch permutation was set")
        if tensor.ndim == 0 or tensor.shape[0] != self._pfi_permutation.numel():
            raise ValueError(
                f"PFI expects activation batch dimension 0 to have size {self._pfi_permutation.numel()}, "
                f"got shape {tuple(tensor.shape)}"
            )
        permutation = self._pfi_permutation.to(device=tensor.device)
        return tensor.index_select(0, permutation).clone()


class ZeroOutputHook:
    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.handle = None

    def _hook(self, module, inputs, output):
        if torch.is_tensor(output):
            return torch.zeros_like(output)
        if isinstance(output, tuple):
            out = []
            for item in output:
                if torch.is_tensor(item):
                    out.append(torch.zeros_like(item))
                else:
                    out.append(item)
            return tuple(out)
        return output

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class RandomSameNormOutputHook(RandomSameNormMixin):
    def __init__(self, module: torch.nn.Module, random_seed: int):
        super().__init__(random_seed=random_seed)
        self.module = module
        self.handle = None

    def _hook(self, module, inputs, output):
        if torch.is_tensor(output):
            return self._random_same_norm(output)
        if isinstance(output, tuple):
            out = []
            for item in output:
                if torch.is_tensor(item):
                    out.append(self._random_same_norm(item))
                else:
                    out.append(item)
            return tuple(out)
        return output

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class PFIOutputHook(PFIPermutationMixin):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.handle = None

    def _hook(self, module, inputs, output):
        if torch.is_tensor(output):
            return self._pfi_replace(output)
        if isinstance(output, tuple):
            # The same donor row is used for every tensor leaf of a structured
            # group output; drawing independently here would not exchange a
            # complete group activation.
            return tuple(self._pfi_replace(item) if torch.is_tensor(item) else item for item in output)
        raise TypeError(f"PFI cannot exchange unsupported output type {type(output).__name__}")

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# Keep the historical script-level imports stable while making the generic
# output interventions shared by image and audio profilers.
random_same_norm_like = shared_analysis.random_same_norm_like
RandomSameNormMixin = shared_analysis.RandomSameNormMixin
PFIPermutationMixin = shared_analysis.PFIPermutationMixin
ZeroOutputHook = shared_analysis.ZeroOutputHook
RandomSameNormOutputHook = shared_analysis.RandomSameNormOutputHook
PFIOutputHook = shared_analysis.PFIOutputHook


class HeadZeroHook:
    """
    Generic per-head output ablation for attention modules whose output channels are laid out
    as equal contiguous head chunks. This is intentionally conservative and will fail fast if
    the output shape is incompatible.
    """

    def __init__(self, module: torch.nn.Module, head_idx: int, channel_dim: int | None = None):
        self.module = module
        self.head_idx = head_idx
        self.channel_dim = channel_dim
        self.handle = None

    def _ablate_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim < 3:
            raise ValueError(
                f"Cannot ablate attention head for output with shape {tuple(tensor.shape)}"
            )
        num_heads = _get_module_num_heads(self.module)
        if num_heads is None:
            raise ValueError(
                f"Module {self.module.__class__.__name__} does not expose a head count"
            )
        channel_dim = self.channel_dim if self.channel_dim is not None else (1 if tensor.ndim == 4 else tensor.ndim - 1)
        if channel_dim < 0:
            channel_dim += tensor.ndim
        if channel_dim <= 0 or channel_dim >= tensor.ndim:
            raise ValueError(f"Invalid attention output channel_dim={self.channel_dim} for shape {tuple(tensor.shape)}")
        width = tensor.shape[channel_dim]
        if width % num_heads != 0:
            raise ValueError(
                f"Output width {width} is not divisible by num_heads={num_heads} for "
                f"{self.module.__class__.__name__}"
            )

        head_width = width // num_heads
        start = self.head_idx * head_width
        end = start + head_width
        output = tensor.clone()

        slicer = [slice(None)] * output.ndim
        slicer[channel_dim] = slice(start, end)
        output[tuple(slicer)] = 0
        return output

    def _hook(self, module, inputs, output):
        if torch.is_tensor(output):
            return self._ablate_tensor(output)
        if isinstance(output, tuple):
            out = []
            for item in output:
                if torch.is_tensor(item):
                    out.append(self._ablate_tensor(item))
                else:
                    out.append(item)
            return tuple(out)
        return output

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class HeadRandomSameNormHook(RandomSameNormMixin):
    """
    Generic per-head output replacement for attention modules whose output channels are laid
    out as equal contiguous head chunks. The selected head slice is replaced with random
    noise whose per-example L2 norm matches the original slice.
    """

    def __init__(self, module: torch.nn.Module, head_idx: int, random_seed: int, channel_dim: int | None = None):
        super().__init__(random_seed=random_seed)
        self.module = module
        self.head_idx = head_idx
        self.channel_dim = channel_dim
        self.handle = None

    def _ablate_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim < 3:
            raise ValueError(
                f"Cannot ablate attention head for output with shape {tuple(tensor.shape)}"
            )
        num_heads = _get_module_num_heads(self.module)
        if num_heads is None:
            raise ValueError(
                f"Module {self.module.__class__.__name__} does not expose a head count"
            )
        channel_dim = self.channel_dim if self.channel_dim is not None else (1 if tensor.ndim == 4 else tensor.ndim - 1)
        if channel_dim < 0:
            channel_dim += tensor.ndim
        if channel_dim <= 0 or channel_dim >= tensor.ndim:
            raise ValueError(f"Invalid attention output channel_dim={self.channel_dim} for shape {tuple(tensor.shape)}")
        width = tensor.shape[channel_dim]
        if width % num_heads != 0:
            raise ValueError(
                f"Output width {width} is not divisible by num_heads={num_heads} for "
                f"{self.module.__class__.__name__}"
            )

        head_width = width // num_heads
        start = self.head_idx * head_width
        end = start + head_width
        output = tensor.clone()

        slicer = [slice(None)] * output.ndim
        slicer[channel_dim] = slice(start, end)
        selected = output[tuple(slicer)]
        output[tuple(slicer)] = self._random_same_norm(selected)
        return output

    def _hook(self, module, inputs, output):
        if torch.is_tensor(output):
            return self._ablate_tensor(output)
        if isinstance(output, tuple):
            out = []
            for item in output:
                if torch.is_tensor(item):
                    out.append(self._ablate_tensor(item))
                else:
                    out.append(item)
            return tuple(out)
        return output

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class HeadPFIHook(PFIPermutationMixin):
    """Exchange one complete attention-head slice between planned donor rows."""

    def __init__(self, module: torch.nn.Module, head_idx: int, channel_dim: int | None = None):
        super().__init__()
        self.module = module
        self.head_idx = int(head_idx)
        self.channel_dim = channel_dim
        self.handle = None

    def _ablate_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim < 3:
            raise ValueError(f"Cannot apply PFI to attention output with shape {tuple(tensor.shape)}")
        num_heads = _get_module_num_heads(self.module)
        if num_heads is None:
            raise ValueError(f"Module {self.module.__class__.__name__} does not expose a head count")
        channel_dim = self.channel_dim if self.channel_dim is not None else (1 if tensor.ndim == 4 else tensor.ndim - 1)
        if channel_dim < 0:
            channel_dim += tensor.ndim
        if channel_dim <= 0 or channel_dim >= tensor.ndim:
            raise ValueError(f"Invalid attention output channel_dim={self.channel_dim} for shape {tuple(tensor.shape)}")
        width = tensor.shape[channel_dim]
        if width % num_heads != 0 or not 0 <= self.head_idx < num_heads:
            raise ValueError(f"Invalid head {self.head_idx} for output width={width}, num_heads={num_heads}")
        head_width = width // num_heads
        slicer = [slice(None)] * tensor.ndim
        slicer[channel_dim] = slice(self.head_idx * head_width, (self.head_idx + 1) * head_width)
        replaced = tensor.clone()
        replaced[tuple(slicer)] = self._pfi_replace(tensor[tuple(slicer)])
        return replaced

    def _hook(self, module, inputs, output):
        if torch.is_tensor(output):
            return self._ablate_tensor(output)
        if isinstance(output, tuple):
            return tuple(self._ablate_tensor(item) if torch.is_tensor(item) else item for item in output)
        raise TypeError(f"PFI cannot exchange unsupported output type {type(output).__name__}")

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class FilterZeroHook:
    def __init__(self, module: torch.nn.Module, filter_idx: int):
        self.module = module
        self.filter_idx = filter_idx
        self.handle = None

    def _ablate_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim < 2:
            raise ValueError(
                f"Cannot ablate convolution filter for output with shape {tuple(tensor.shape)}"
            )
        channel_dim = 1
        num_channels = tensor.shape[channel_dim]
        if self.filter_idx < 0 or self.filter_idx >= num_channels:
            raise ValueError(
                f"Filter index {self.filter_idx} is out of range for output with {num_channels} channels"
            )
        output = tensor.clone()
        slicer = [slice(None)] * output.ndim
        slicer[channel_dim] = self.filter_idx
        output[tuple(slicer)] = 0
        return output

    def _hook(self, module, inputs, output):
        if torch.is_tensor(output):
            return self._ablate_tensor(output)
        if isinstance(output, tuple):
            out = []
            for item in output:
                if torch.is_tensor(item):
                    out.append(self._ablate_tensor(item))
                else:
                    out.append(item)
            return tuple(out)
        return output

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class FilterRandomSameNormHook(RandomSameNormMixin):
    def __init__(self, module: torch.nn.Module, filter_idx: int, random_seed: int):
        super().__init__(random_seed=random_seed)
        self.module = module
        self.filter_idx = filter_idx
        self.handle = None

    def _ablate_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim < 2:
            raise ValueError(
                f"Cannot ablate convolution filter for output with shape {tuple(tensor.shape)}"
            )
        channel_dim = 1
        num_channels = tensor.shape[channel_dim]
        if self.filter_idx < 0 or self.filter_idx >= num_channels:
            raise ValueError(
                f"Filter index {self.filter_idx} is out of range for output with {num_channels} channels"
            )
        output = tensor.clone()
        slicer = [slice(None)] * output.ndim
        slicer[channel_dim] = self.filter_idx
        selected = output[tuple(slicer)]
        output[tuple(slicer)] = self._random_same_norm(selected)
        return output

    def _hook(self, module, inputs, output):
        if torch.is_tensor(output):
            return self._ablate_tensor(output)
        if isinstance(output, tuple):
            out = []
            for item in output:
                if torch.is_tensor(item):
                    out.append(self._ablate_tensor(item))
                else:
                    out.append(item)
            return tuple(out)
        return output

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class FilterPFIHook(PFIPermutationMixin):
    """Exchange one complete convolution-filter activation between donor rows."""

    def __init__(self, module: torch.nn.Module, filter_idx: int):
        super().__init__()
        self.module = module
        self.filter_idx = int(filter_idx)
        self.handle = None

    def _ablate_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim < 2:
            raise ValueError(f"Cannot apply PFI to convolution output with shape {tuple(tensor.shape)}")
        if not 0 <= self.filter_idx < tensor.shape[1]:
            raise ValueError(f"Filter index {self.filter_idx} is out of range for output with {tensor.shape[1]} channels")
        replaced = tensor.clone()
        selected = tensor[:, self.filter_idx]
        replaced[:, self.filter_idx] = self._pfi_replace(selected)
        return replaced

    def _hook(self, module, inputs, output):
        if torch.is_tensor(output):
            return self._ablate_tensor(output)
        if isinstance(output, tuple):
            return tuple(self._ablate_tensor(item) if torch.is_tensor(item) else item for item in output)
        raise TypeError(f"PFI cannot exchange unsupported output type {type(output).__name__}")

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# ---------------------------
# Stats / distributed
# ---------------------------

@dataclass
class BinStats:
    num_bins: int

    def __post_init__(self):
        self.sum = torch.zeros(self.num_bins, dtype=torch.float64)
        self.sum_sq = torch.zeros(self.num_bins, dtype=torch.float64)
        self.count = torch.zeros(self.num_bins, dtype=torch.long)

    def update(self, bin_ids: torch.Tensor, losses: torch.Tensor):
        b_cpu = bin_ids.detach().cpu()
        l_cpu = losses.detach().cpu().to(torch.float64)
        for b, l in zip(b_cpu.tolist(), l_cpu.tolist()):
            self.sum[b] += l
            self.sum_sq[b] += l * l
            self.count[b] += 1

    def mean(self) -> torch.Tensor:
        out = torch.zeros_like(self.sum)
        mask = self.count > 0
        out[mask] = self.sum[mask] / self.count[mask]
        return out

    def stderr(self) -> torch.Tensor:
        out = torch.zeros_like(self.sum)
        mask = self.count > 1
        mean = self.mean()
        var = torch.zeros_like(self.sum)
        var[mask] = self.sum_sq[mask] / self.count[mask] - mean[mask] ** 2
        var = torch.clamp(var, min=0.0)
        out[mask] = torch.sqrt(var[mask] / self.count[mask].to(torch.float64))
        return out


# Preserve evaluate_parameters_edm.BinStats as a public compatibility name.
BinStats = shared_analysis.BinStats


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed(
    device_arg: str,
    timeout_seconds: int = 86_400,
) -> Tuple[str, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return device_arg, 0, 1

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    backend = "nccl" if device_arg.startswith("cuda") else "gloo"
    if timeout_seconds <= 0:
        raise ValueError("distributed timeout must be positive")
    dist.init_process_group(
        backend=backend,
        timeout=timedelta(seconds=int(timeout_seconds)),
    )

    if device_arg.startswith("cuda"):
        torch.cuda.set_device(local_rank)
        device_arg = f"cuda:{local_rank}"

    return device_arg, rank, world_size


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def release_cuda_memory(*objects) -> None:
    for obj in objects:
        if obj is not None:
            del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def shard_group_items(
    group_items: List[Tuple[str, Union[torch.nn.Module, Tuple[torch.nn.Module, int]]]],
    rank: int,
    world_size: int,
) -> List[Tuple[str, Union[torch.nn.Module, Tuple[torch.nn.Module, int]]]]:
    if world_size <= 1:
        return group_items
    return group_items[rank::world_size]


def gather_ablation_results(local_results: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not is_distributed():
        return local_results

    gathered: List[Optional[Dict[str, torch.Tensor]]] = [None] * get_world_size()
    payload = {name: tensor.cpu() for name, tensor in local_results.items()}
    dist.all_gather_object(gathered, payload)

    merged: Dict[str, torch.Tensor] = {}
    for part in gathered:
        if part is not None:
            merged.update(part)
    return merged


def checkpoint_name_for_rank(ablation_mode: AblationMode, rank: int) -> str:
    if ablation_mode == "zero":
        return f"checkpoint_rank{rank}.pt"
    return f"checkpoint_{ablation_mode}_rank{rank}.pt"


def checkpoint_pattern_for_mode(output_dir: str, ablation_mode: AblationMode) -> str:
    if ablation_mode == "zero":
        return os.path.join(output_dir, "checkpoint_rank*.pt")
    return os.path.join(output_dir, f"checkpoint_{ablation_mode}_rank*.pt")


def pfi_plan_envelope(plan: ExactSigmaPFIBatchSampler) -> Dict[str, Any]:
    """Return the complete compact plan, including every member and donor row."""

    return {
        "format": PFI_PLAN_FORMAT,
        "metadata": dict(plan.artifact),
        "member_orders": plan._member_orders.detach().cpu().clone(),
        "donor_positions": plan._donor_positions.detach().cpu().clone(),
    }


def persist_or_validate_pfi_plan(
    path: str | os.PathLike[str],
    plan: ExactSigmaPFIBatchSampler,
    *,
    allow_create: bool,
) -> None:
    """Atomically publish a plan, or strictly verify an already published plan."""

    path = Path(path)
    expected = pfi_plan_envelope(plan)
    if not path.exists():
        if not allow_create:
            raise FileNotFoundError(f"Expected PFI plan artifact does not exist: {path}")
        atomic_torch_save(expected, path)

    saved = torch.load(path, weights_only=True, map_location="cpu")
    if not isinstance(saved, dict) or saved.get("format") != PFI_PLAN_FORMAT:
        raise ValueError(f"PFI plan {path} is not a {PFI_PLAN_FORMAT} artifact")
    if saved.get("metadata") != expected["metadata"]:
        raise ValueError(f"PFI plan {path} metadata differs from the requested plan")
    for key in ("member_orders", "donor_positions"):
        value = saved.get(key)
        expected_value = expected[key]
        if not torch.is_tensor(value) or value.dtype != torch.int32:
            raise ValueError(f"PFI plan {path} has invalid {key}")
        if tuple(value.shape) != tuple(expected_value.shape) or not torch.equal(value, expected_value):
            raise ValueError(f"PFI plan {path} {key} differs from the requested plan")


def pfi_checkpoint_envelope(
    groups: Dict[str, torch.Tensor],
    *,
    rank: int,
    num_bins: int,
    profile_fingerprint: Dict[str, Any],
    profile_fingerprint_sha256: str,
) -> Dict[str, Any]:
    if canonical_json_sha256(profile_fingerprint) != profile_fingerprint_sha256:
        raise ValueError("PFI profile fingerprint digest does not match its record")
    return {
        "format": PFI_CHECKPOINT_FORMAT,
        "rank": int(rank),
        "num_bins": int(num_bins),
        "profile_fingerprint": profile_fingerprint,
        "profile_fingerprint_sha256": profile_fingerprint_sha256,
        "completed_groups": list(groups),
        "groups": {name: value.detach().cpu() for name, value in groups.items()},
    }


def pfi_baseline_envelope(
    *,
    mean: torch.Tensor,
    stderr: torch.Tensor,
    count: torch.Tensor,
    num_bins: int,
    profile_fingerprint: Dict[str, Any],
    profile_fingerprint_sha256: str,
) -> Dict[str, Any]:
    if canonical_json_sha256(profile_fingerprint) != profile_fingerprint_sha256:
        raise ValueError("PFI profile fingerprint digest does not match its record")
    return {
        "format": PFI_BASELINE_FORMAT,
        "num_bins": int(num_bins),
        "profile_fingerprint": profile_fingerprint,
        "profile_fingerprint_sha256": profile_fingerprint_sha256,
        "mean": mean.detach().cpu(),
        "stderr": stderr.detach().cpu(),
        "count": count.detach().cpu(),
    }


def load_pfi_baseline(
    path: str | os.PathLike[str],
    *,
    num_bins: int,
    profile_fingerprint: Dict[str, Any],
    profile_fingerprint_sha256: str,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    path = Path(path)
    if not path.exists():
        return None
    saved = torch.load(path, weights_only=True, map_location="cpu")
    if not isinstance(saved, dict) or saved.get("format") != PFI_BASELINE_FORMAT:
        raise ValueError(f"PFI baseline {path} is not a {PFI_BASELINE_FORMAT} envelope")
    if canonical_json_sha256(profile_fingerprint) != profile_fingerprint_sha256:
        raise ValueError("Expected PFI profile fingerprint record and digest disagree")
    if saved.get("profile_fingerprint_sha256") != profile_fingerprint_sha256 or saved.get("profile_fingerprint") != profile_fingerprint:
        raise ValueError(f"PFI baseline {path} belongs to a different profile fingerprint")
    if saved.get("num_bins") != num_bins:
        raise ValueError(f"PFI baseline {path} has num_bins={saved.get('num_bins')}, expected {num_bins}")
    tensors = (saved.get("mean"), saved.get("stderr"), saved.get("count"))
    if any(not torch.is_tensor(value) or tuple(value.shape) != (num_bins,) for value in tensors):
        raise ValueError(f"PFI baseline {path} has invalid statistic shapes")
    mean, stderr, count = tensors
    assert torch.is_tensor(mean) and torch.is_tensor(stderr) and torch.is_tensor(count)
    if not torch.isfinite(mean).all() or not torch.isfinite(stderr).all() or bool(torch.any(count < 0)):
        raise ValueError(f"PFI baseline {path} contains invalid statistic values")
    return tensors  # type: ignore[return-value]


def load_ablation_checkpoints(
    output_dir: str,
    ablation_mode: AblationMode,
    expected_group_names: List[str],
    num_bins: int,
    profile_fingerprint: Optional[Dict[str, Any]] = None,
    profile_fingerprint_sha256: Optional[str] = None,
) -> Tuple[Dict[str, torch.Tensor], List[str], int, List[Tuple[str, str, Tuple[int, ...]]]]:
    expected = set(expected_group_names)
    checkpoint_paths = sorted(glob.glob(checkpoint_pattern_for_mode(output_dir, ablation_mode)))
    merged: Dict[str, torch.Tensor] = {}
    ignored_unknown = 0
    ignored_bad_shape: List[Tuple[str, str, Tuple[int, ...]]] = []

    for path in checkpoint_paths:
        saved = torch.load(path, weights_only=True, map_location="cpu")
        if not isinstance(saved, dict):
            raise ValueError(f"Checkpoint {path} should contain a dict, got {type(saved).__name__}")
        if ablation_mode == "pfi":
            if profile_fingerprint is None or profile_fingerprint_sha256 is None:
                raise ValueError("PFI checkpoint loading requires the complete profile fingerprint")
            if saved.get("format") != PFI_CHECKPOINT_FORMAT or not isinstance(saved.get("groups"), dict):
                raise ValueError(f"PFI checkpoint {path} is not a {PFI_CHECKPOINT_FORMAT} envelope")
            actual_digest = canonical_json_sha256(profile_fingerprint)
            if actual_digest != profile_fingerprint_sha256:
                raise ValueError("Expected PFI profile fingerprint record and digest disagree")
            if saved.get("profile_fingerprint_sha256") != profile_fingerprint_sha256:
                raise ValueError(f"PFI checkpoint {path} belongs to a different profile fingerprint")
            if saved.get("profile_fingerprint") != profile_fingerprint:
                raise ValueError(f"PFI checkpoint {path} profile fingerprint record differs")
            if not isinstance(saved.get("rank"), int) or saved["rank"] < 0:
                raise ValueError(f"PFI checkpoint {path} has invalid rank metadata")
            expected_rank_suffix = f"rank{saved['rank']}.pt"
            if not Path(path).name.endswith(expected_rank_suffix):
                raise ValueError(f"PFI checkpoint {path} rank metadata does not match its filename")
            if saved.get("num_bins") != num_bins:
                raise ValueError(f"PFI checkpoint {path} has num_bins={saved.get('num_bins')}, expected {num_bins}")
            saved_groups = saved["groups"]
            if saved.get("completed_groups") != list(saved_groups):
                raise ValueError(f"PFI checkpoint {path} completed_groups does not match its group payload")
            unknown_completed = [name for name in saved["completed_groups"] if name not in expected]
            if unknown_completed:
                raise ValueError(f"PFI checkpoint {path} contains unexpected completed groups: {unknown_completed[:5]}")
        else:
            saved_groups = saved
        for name, value in saved_groups.items():
            if name not in expected:
                ignored_unknown += 1
                continue
            if not torch.is_tensor(value) or tuple(value.shape) != (num_bins,):
                shape = tuple(value.shape) if torch.is_tensor(value) else ()
                if ablation_mode == "pfi":
                    raise ValueError(
                        f"PFI checkpoint {path} group {name} has shape {shape}, expected ({num_bins},)"
                    )
                ignored_bad_shape.append((path, str(name), shape))
                continue
            if ablation_mode == "pfi" and not torch.isfinite(value).all():
                raise ValueError(f"PFI checkpoint {path} group {name} contains non-finite values")
            merged[str(name)] = value.cpu()

    return merged, checkpoint_paths, ignored_unknown, ignored_bad_shape


# ---------------------------
# Checkpoint loading
# ---------------------------

def _open_maybe_url(path_or_url: str):
    parsed = urllib.parse.urlparse(path_or_url)
    if parsed.scheme in ("http", "https"):
        return urllib.request.urlopen(path_or_url)
    return open(path_or_url, "rb")


def load_edm_network(
    network_pkl: str,
    device: torch.device,
    dtype: torch.dtype,
    *,
    network_format: str | None = None,
    network_preset: str | None = None,
    model_cache_dir: str | None = None,
    trust_local_pickle: bool = False,
) -> torch.nn.Module:
    from pace.teacher_models import load_teacher_network

    try:
        return load_teacher_network(
            network_pkl,
            device=device,
            dtype=dtype,
            network_format=network_format,
            preset=network_preset,
            cache_dir=model_cache_dir,
            trust_local_pickle=trust_local_pickle,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Failed to load the EDM checkpoint because required NVLabs EDM modules are not "
            f"importable in this Python environment (missing module: {exc.name}). "
            "Make sure the official NVLabs edm repository is on PYTHONPATH so modules like "
            "dnnlib and torch_utils can be imported before running this script."
        ) from exc

# ---------------------------
# Evaluator
# ---------------------------

class EDMUsageEvaluator:
    def __init__(
        self,
        network_pkl: str,
        device: str,
        dtype: torch.dtype,
        sigma_min: Optional[float],
        sigma_max: Optional[float],
        sigma_data: Optional[float],
        num_sigma_levels: int,
        class_idx: Optional[int],
        model_family: str,
        network_format: str | None = None,
        network_preset: str | None = None,
        model_cache_dir: str | None = None,
        trust_local_pickle: bool = False,
    ):
        from pace.teacher_models import resolve_teacher_spec, teacher_model_metadata

        self.device = torch.device(device)
        self.teacher_spec = resolve_teacher_spec(
            network_pkl,
            network_format=network_format,
            preset=network_preset,
        )
        self.net = load_edm_network(
            network_pkl=network_pkl,
            device=self.device,
            dtype=dtype,
            network_format=self.teacher_spec.format,
            network_preset=self.teacher_spec.preset,
            model_cache_dir=model_cache_dir,
            trust_local_pickle=trust_local_pickle,
        )
        self.teacher_spec = resolve_teacher_spec(getattr(self.net, "teacher_spec", self.teacher_spec.to_dict()))
        self.model_metadata = teacher_model_metadata(self.net, self.teacher_spec)
        # OpenAI's official mixed-FP16 path deliberately keeps the time
        # embedding and output layers in FP32.  Inputs to the preconditioning
        # wrapper must remain FP32; the wrapper casts only the UNet torso.
        self.net_dtype = (
            torch.float32
            if self.teacher_spec.format == "openai_consistency_edm_state_dict_v1"
            else next(self.net.parameters()).dtype
        )
        inferred_family = infer_model_family(self.net)
        if model_family == "auto":
            self.model_family = inferred_family
        else:
            self.model_family = model_family
        self.class_idx = class_idx
        self.label_dim = int(getattr(self.net, "label_dim", 0))
        self.img_resolution = int(getattr(self.net, "img_resolution"))
        self.img_channels = int(getattr(self.net, "img_channels"))
        self.sigma_data = float(sigma_data if sigma_data is not None else getattr(self.net, "sigma_data", 0.5))

        network_sigma_min = getattr(self.net, "sigma_min", None)
        network_sigma_max = getattr(self.net, "sigma_max", None)

        default_sigma_min = 0.002
        default_sigma_max = 80.0

        resolved_sigma_min = sigma_min if sigma_min is not None else network_sigma_min
        resolved_sigma_max = sigma_max if sigma_max is not None else network_sigma_max

        self.sigma_min = float(resolved_sigma_min) if resolved_sigma_min is not None else default_sigma_min
        self.sigma_max = float(resolved_sigma_max) if resolved_sigma_max is not None else default_sigma_max

        if not math.isfinite(self.sigma_min) or self.sigma_min <= 0:
            self.sigma_min = default_sigma_min
        if not math.isfinite(self.sigma_max) or self.sigma_max <= 0:
            self.sigma_max = default_sigma_max
        if self.sigma_min > self.sigma_max:
            raise ValueError(
                f"Invalid sigma range: sigma_min={self.sigma_min}, sigma_max={self.sigma_max}"
            )

        self.sigma_values = make_sigma_schedule_for_family(
            net=self.net,
            model_family=self.model_family,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            num_levels=num_sigma_levels,
            device=self.device,
        )

    @torch.no_grad()
    def forward_losses_from_fixed_corruption(
        self,
        images: torch.Tensor,
        class_indices: torch.Tensor,
        sigma_indices: torch.Tensor,
        noise_seeds: torch.Tensor,
    ) -> torch.Tensor:
        images = images.to(device=self.device, dtype=self.net_dtype)
        class_indices = class_indices.to(self.device)
        sigma_indices = sigma_indices.to(self.device)
        sigmas = self.sigma_values[sigma_indices]

        noise = torch.empty_like(images)
        for i in range(images.shape[0]):
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(noise_seeds[i].item()))
            noise[i] = torch.randn(images[i].shape, generator=gen, device=self.device, dtype=images.dtype)

        noisy_images = images + noise * sigmas.view(-1, 1, 1, 1)
        labels = make_class_labels(
            batch_size=images.shape[0],
            label_dim=self.label_dim,
            class_idx=self.class_idx,
            device=self.device,
            dataset_class_indices=class_indices,
        )

        denoised = self.net(noisy_images, sigmas, labels)
        weights = loss_weights_for_family(
            sigmas=sigmas,
            model_family=self.model_family,
            sigma_data=self.sigma_data,
        )
        losses = mse_per_example(denoised.float(), images.float()) * weights.float()
        return losses

    def evaluate(
        self,
        dataloader: DataLoader,
        num_bins: int,
        ablate_target: Optional[AblationTarget] = None,
        ablation_mode: AblationMode = "zero",
        ablation_random_seed: Optional[int] = None,
        progress_desc: Optional[str] = None,
    ) -> BinStats:
        stats = BinStats(num_bins=num_bins)
        if isinstance(ablate_target, tuple):
            if len(ablate_target) == 3 and ablate_target[2] == "filter":
                if ablation_mode == "zero":
                    context = FilterZeroHook(ablate_target[0], ablate_target[1])
                elif ablation_mode == "random_same_norm":
                    if ablation_random_seed is None:
                        raise ValueError("ablation_random_seed is required for random_same_norm ablation")
                    context = FilterRandomSameNormHook(ablate_target[0], ablate_target[1], ablation_random_seed)
                elif ablation_mode == "pfi":
                    context = FilterPFIHook(ablate_target[0], ablate_target[1])
                else:
                    raise ValueError(f"Unsupported ablation_mode: {ablation_mode}")
            else:
                channel_dim = int(ablate_target[2]) if len(ablate_target) == 3 else None
                if ablation_mode == "zero":
                    context = HeadZeroHook(ablate_target[0], ablate_target[1], channel_dim=channel_dim)
                elif ablation_mode == "random_same_norm":
                    if ablation_random_seed is None:
                        raise ValueError("ablation_random_seed is required for random_same_norm ablation")
                    context = HeadRandomSameNormHook(
                        ablate_target[0],
                        ablate_target[1],
                        random_seed=ablation_random_seed,
                        channel_dim=channel_dim,
                    )
                elif ablation_mode == "pfi":
                    context = HeadPFIHook(
                        ablate_target[0], ablate_target[1], channel_dim=channel_dim
                    )
                else:
                    raise ValueError(f"Unsupported ablation_mode: {ablation_mode}")
        elif ablate_target is not None:
            if ablation_mode == "zero":
                context = ZeroOutputHook(ablate_target)
            elif ablation_mode == "random_same_norm":
                if ablation_random_seed is None:
                    raise ValueError("ablation_random_seed is required for random_same_norm ablation")
                context = RandomSameNormOutputHook(ablate_target, ablation_random_seed)
            elif ablation_mode == "pfi":
                context = PFIOutputHook(ablate_target)
            else:
                raise ValueError(f"Unsupported ablation_mode: {ablation_mode}")
        else:
            context = torch.no_grad()

        progress = tqdm(
            dataloader,
            total=len(dataloader),
            desc=progress_desc or "Evaluating",
            dynamic_ncols=True,
            leave=False,
            position=get_rank(),
        )

        if ablate_target is not None:
            context.__enter__()

        try:
            for batch in progress:
                if len(batch) == 4:
                    images, class_indices, sigma_indices, noise_seeds = batch
                    pfi_permutation = None
                elif len(batch) == 5:
                    images, class_indices, sigma_indices, noise_seeds, pfi_permutation = batch
                else:
                    raise ValueError(f"Unexpected corruption batch with {len(batch)} fields")
                if ablation_mode == "pfi" and ablate_target is not None:
                    if pfi_permutation is None:
                        raise ValueError("PFI evaluation requires an ExactSigmaPFIBatchSampler")
                    if not hasattr(context, "set_pfi_permutation"):
                        raise RuntimeError("PFI ablation context does not accept planned permutations")
                    context.set_pfi_permutation(pfi_permutation)
                losses = self.forward_losses_from_fixed_corruption(
                    images=images,
                    class_indices=class_indices,
                    sigma_indices=sigma_indices,
                    noise_seeds=noise_seeds,
                )
                if not torch.isfinite(losses).all():
                    bad_idx = (~torch.isfinite(losses)).nonzero(as_tuple=False).flatten().tolist()
                    raise ValueError(
                        f"Non-finite losses encountered for batch indices {bad_idx}. "
                        "Try --dtype fp32 or a smaller --image_size if this persists."
                    )
                bin_ids = level_to_bin(
                    indices=sigma_indices,
                    num_levels=len(self.sigma_values),
                    num_bins=num_bins,
                )
                stats.update(bin_ids, losses)
        finally:
            progress.close()
            if ablate_target is not None:
                context.__exit__(None, None, None)

        return stats


# ---------------------------
# Plotting / saving
# ---------------------------

def save_json(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def atomic_torch_save(obj, path, keep_prev=False) -> None:
    """``torch.save`` via a temp file + ``os.replace``.

    Preemption hardening item 5: a bare
    ``torch.save(obj, path)`` truncated by a preemption/OOM-kill/node-failure
    mid-write leaves a corrupt file AT the real path -- the exact shape this
    codebase's own resumable checkpoints (permutation-importance ablation
    checkpoints, curve/step_<k>.pt snapshots, the legacy resume.pt/student.pt
    paths) are read back on the very next attempt, with no try/except around
    that ``torch.load`` in most call sites, so it becomes a hard crash that
    blocks resume until a human deletes the corrupt file by hand. Writing to
    a hidden per-process temp file next to ``path`` and ``os.replace``-ing it into place means a crash
    mid-write leaves the OLD file (if any) intact and the new one simply
    never appears -- never a half-written file at the real path.

    ``keep_prev`` (default ``False``, byte-identical to every call before
    this parameter existed): if ``True`` and ``path`` already holds a
    previous save, that file is atomically renamed to ``path + ".prev"``
    immediately before the new payload's own atomic replace -- a one-
    generation-back fallback for a FILESYSTEM-level corruption of the bytes
    already on disk (bit rot, a bad block), which the tmp+replace write
    itself cannot protect against (it only protects against a torn write)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(obj, tmp)
        if keep_prev and path.exists():
            os.replace(path, path.with_name(path.name + ".prev"))   # rotate BEFORE the new atomic replace
        os.replace(tmp, path)
    finally:
        if tmp.exists():   # only after a failed save/replace
            tmp.unlink()


def plot_baseline(mean: torch.Tensor, stderr: torch.Tensor, out_path: str) -> None:
    x = list(range(len(mean)))
    m = mean.cpu().numpy()
    s = stderr.cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(x, m, label="Baseline weighted denoising MSE")
    plt.fill_between(x, m - 1.96 * s, m + 1.96 * s, alpha=0.2)
    plt.xlabel("Noise-level bin")
    plt.ylabel("Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_peff(p_eff: torch.Tensor, out_path: str) -> None:
    x = list(range(len(p_eff)))
    y = p_eff.cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(x, y, label="Effective parameter usage")
    plt.xlabel("Noise-level bin")
    plt.ylabel("P_eff(bin)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_neff(n_eff: torch.Tensor, out_path: str) -> None:
    x = list(range(len(n_eff)))
    y = n_eff.cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(x, y, label="Effective number of active groups")
    plt.xlabel("Noise-level bin")
    plt.ylabel("N_eff(bin)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _sparse_tick_positions(num_items: int, max_ticks: int = 24) -> List[int]:
    if num_items <= max_ticks:
        return list(range(num_items))
    step = math.ceil(num_items / max_ticks)
    return list(range(0, num_items, step))


def row_normalize_matrix(matrix: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return shared_analysis.row_normalize_matrix(matrix, eps=eps)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


# Valid values for --importance_clip_mode (evaluate_parameters_edm.py and siblings
# evaluate_parameters_dit.py / evaluate_parameters_dit_micro.py). See
# compute_usage_metrics() below for the statistical rationale.
IMPORTANCE_CLIP_MODES = ("per_entry", "post_agg", "none")


def build_delta_stack(
    ablated_means: Dict[str, torch.Tensor],
    baseline_mean: torch.Tensor,
    names: List[str],
    clip_mode: str = "per_entry",
) -> torch.Tensor:
    """Stack per-group (ablated_mean - baseline_mean) deltas into a (group, bin) matrix.

    ``clip_mode="per_entry"`` clamps each entry at 0 immediately (byte-identical to
    the original, unconditional ``clamp(..., min=0.0)``); ``"post_agg"``/``"none"``
    keep the matrix signed, deferring any non-negativity floor to
    ``compute_usage_metrics``'s ``clip_mode`` handling (or to whatever consumes the
    saved ``delta_stack`` downstream).
    """
    if clip_mode not in IMPORTANCE_CLIP_MODES:
        raise ValueError(f"Unsupported clip_mode: {clip_mode!r} (expected one of {IMPORTANCE_CLIP_MODES})")
    raw = [sanitize_tensor(ablated_means[name] - baseline_mean) for name in names]
    if clip_mode == "per_entry":
        raw = [torch.clamp(d, min=0.0) for d in raw]
    return torch.stack(raw, dim=0)


def compute_usage_metrics(
    delta_stack: torch.Tensor,
    baseline_mean: torch.Tensor,
    group_param_counts: Dict[str, int],
    group_names: List[str],
    compute_group_correlation: bool = True,
    clip_mode: str = "per_entry",
    group_sampling_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Optional[torch.Tensor]]:
    """Turn a raw per-(group, bin) delta matrix into the saved/reported usage metrics.

    ``clip_mode`` controls how the non-negativity floor (permutation ablation
    "improving" the loss is not a real finding) is applied:

    - ``"per_entry"`` (default, byte-identical to the original behavior): the
      caller is expected to have already clamped ``delta_stack`` at 0 entry-by-entry
      (i.e. ``clamp(ablated_mean - baseline_mean, min=0)`` per (group, bin) cell)
      *before* calling this function, so every metric derived here (including the
      cross-group aggregation feeding ``weights``/``n_eff``/``p_eff``) is computed on
      already-non-negative data. Passing an already-non-negative ``delta_stack`` with
      the default ``clip_mode`` reproduces the pre-existing formula exactly.
    - ``"post_agg"``: ``delta_stack`` is expected to be the SIGNED (unclamped) per-
      entry delta. The saved ``delta_stack``/``relative_delta_stack`` (and the
      correlation matrices derived from them) stay signed, so noise that happens to
      be negative in one (group, bin) cell can offset noise that happens to be
      positive elsewhere once something downstream aggregates across bins or heads,
      instead of being floored to 0 and creating a one-sided positive bias
      (``E[max(0, X)] > 0`` for zero-mean noise ``X``). Only the ``weights``/
      ``n_eff``/``p_eff`` aggregation (which represents an "effective count" of
      active groups and is only meaningful for non-negative mass) clamps its input
      internally, so those specific metrics stay well-behaved/bounded.
    - ``"none"``: like ``"post_agg"``, but nothing is clamped anywhere, including the
      ``weights``/``n_eff``/``p_eff`` aggregation input. Intended for analysis of the
      fully signed statistic; downstream consumers that need non-negativity (e.g. an
      allocator's ``sqrt(score / mean)`` weighting) must clamp at the point they
      consume it.
    """
    if clip_mode not in IMPORTANCE_CLIP_MODES:
        raise ValueError(f"Unsupported clip_mode: {clip_mode!r} (expected one of {IMPORTANCE_CLIP_MODES})")

    delta_stack = sanitize_tensor(delta_stack.to(torch.float64))
    if clip_mode == "per_entry":
        # Defensive: real callers (build_delta_stack) already clamp per_entry's
        # delta_stack before it reaches here, so this is a no-op (byte-identical
        # output preserved). Clamping here too means the "per_entry never returns
        # negative delta_stack/relative_delta_stack" contract doesn't silently
        # depend on caller discipline.
        delta_stack = torch.clamp(delta_stack, min=0.0)
    baseline_mean = sanitize_tensor(baseline_mean.to(torch.float64))
    relative_delta_stack = sanitize_tensor(delta_stack / (baseline_mean.unsqueeze(0) + 1e-12))
    row_normalized_relative_delta_stack = row_normalize_matrix(relative_delta_stack)

    # weights/n_eff/p_eff represent an "effective count"/allocation of active groups
    # and are only well-defined for non-negative importance mass. clip_mode="none" is
    # the sole exception (fully signed, for analysis only); clip_mode="per_entry"
    # receives an already-non-negative delta_stack from the caller, so this clamp is
    # a no-op there (preserving byte-identical output).
    weights_input = delta_stack if clip_mode == "none" else torch.clamp(delta_stack, min=0.0)
    if group_sampling_weights is None:
        expansion = torch.ones((len(group_names), 1), dtype=torch.float64)
    else:
        missing = [name for name in group_names if name not in group_sampling_weights]
        extra = sorted(set(group_sampling_weights) - set(group_names))
        if missing or extra:
            raise ValueError(
                "group_sampling_weights must align exactly with evaluated groups; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        expansion_values = [float(group_sampling_weights[name]) for name in group_names]
        if any(not math.isfinite(value) or value <= 0 for value in expansion_values):
            raise ValueError("group_sampling_weights values must be finite and positive")
        expansion = torch.tensor(expansion_values, dtype=torch.float64).unsqueeze(1)

    # Horvitz-Thompson expansion estimates the full filter-population mass.
    # ``weights`` contains aggregate mass represented by each sampled filter.
    expanded_delta_stack = sanitize_tensor(weights_input * expansion)
    positive_delta_sums = expanded_delta_stack.sum(dim=0, keepdim=True) + 1e-12
    weights = sanitize_tensor(expanded_delta_stack / positive_delta_sums)
    # Each sampled row represents ``expansion`` population filters.  Splitting
    # its aggregate mass evenly among those represented filters gives the
    # population-scale inverse-Simpson effective group count.
    n_eff = sanitize_tensor(1.0 / ((weights.pow(2) / expansion).sum(dim=0) + 1e-12))

    p = torch.tensor([group_param_counts[name] for name in group_names], dtype=torch.float64).unsqueeze(1)
    p_eff = sanitize_tensor((weights * p).sum(dim=0))

    if compute_group_correlation:
        if relative_delta_stack.shape[0] == 1:
            group_correlation = torch.ones((1, 1), dtype=torch.float64)
        elif relative_delta_stack.shape[1] < 2:
            group_correlation = torch.zeros(
                (relative_delta_stack.shape[0], relative_delta_stack.shape[0]),
                dtype=torch.float64,
            )
        else:
            group_correlation = sanitize_tensor(torch.corrcoef(relative_delta_stack))
    else:
        group_correlation = None

    if relative_delta_stack.shape[1] == 1:
        noise_level_correlation = torch.ones((1, 1), dtype=torch.float64)
    elif relative_delta_stack.shape[0] < 2:
        noise_level_correlation = torch.zeros(
            (relative_delta_stack.shape[1], relative_delta_stack.shape[1]),
            dtype=torch.float64,
        )
    elif torch.all(expansion == 1):
        # Preserve the exact exhaustive/legacy numerical path.
        noise_level_correlation = sanitize_tensor(torch.corrcoef(relative_delta_stack.T))
    else:
        # Weighted Pearson correlation over sampled filters.  Square-root
        # expansion after weighted centering is equivalent to repeating each
        # sampled observation 1/pi times, without materializing those rows.
        observation_weights = expansion.squeeze(1)
        weighted_mean = (
            relative_delta_stack * observation_weights.unsqueeze(1)
        ).sum(dim=0) / observation_weights.sum()
        centered = relative_delta_stack - weighted_mean.unsqueeze(0)
        weighted_centered = centered * observation_weights.sqrt().unsqueeze(1)
        covariance = weighted_centered.T @ weighted_centered
        variances = torch.diag(covariance).clamp_min(0.0)
        denominator = torch.sqrt(variances.unsqueeze(1) * variances.unsqueeze(0))
        noise_level_correlation = sanitize_tensor(covariance / (denominator + 1e-12))

    return {
        "delta_stack": delta_stack,
        "relative_delta_stack": relative_delta_stack,
        "row_normalized_relative_delta_stack": row_normalized_relative_delta_stack,
        "sampling_adjusted_delta_stack": expanded_delta_stack,
        "group_sampling_expansion_weights": expansion.squeeze(1),
        "weights": weights,
        "n_eff": n_eff,
        "p_eff": p_eff,
        "C_groups": group_correlation,
        "C_noise_levels": noise_level_correlation,
    }


def should_compute_group_correlation(mode: str, num_groups: int, max_groups: int) -> bool:
    return shared_analysis.should_compute_group_correlation(mode, num_groups, max_groups)


def tensor_or_none_to_list(value: Optional[torch.Tensor]):
    return None if value is None else value.tolist()


def compute_raw_baseline_metrics(
    baseline_mean: torch.Tensor,
    baseline_stderr: torch.Tensor,
    sigma_values_strided: torch.Tensor,
    model_family: str,
    sigma_data: float,
    num_bins: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    loss_w = loss_weights_for_family(sigma_values_strided, model_family, sigma_data)
    avg_weight_per_bin = torch.zeros(num_bins, dtype=torch.float64)
    weight_count = torch.zeros(num_bins, dtype=torch.long)
    for i in range(len(sigma_values_strided)):
        b = min(int(i * num_bins / len(sigma_values_strided)), num_bins - 1)
        avg_weight_per_bin[b] += float(loss_w[i])
        weight_count[b] += 1
    avg_weight_per_bin = avg_weight_per_bin / weight_count.clamp(min=1).to(torch.float64)
    raw_baseline_mean = sanitize_tensor(baseline_mean.to(torch.float64) / (avg_weight_per_bin + 1e-12))
    raw_baseline_stderr = sanitize_tensor(baseline_stderr.to(torch.float64) / (avg_weight_per_bin + 1e-12))
    return raw_baseline_mean, raw_baseline_stderr


def infer_bin_axis_metadata(results: Dict, num_bins: int) -> Dict[str, Union[str, List[str]]]:
    return shared_analysis.infer_bin_axis_metadata(results, num_bins)


def save_results_and_plots(
    output_dir: str,
    results: Dict,
    baseline_mean: torch.Tensor,
    baseline_stderr: torch.Tensor,
    raw_baseline_mean: torch.Tensor,
    raw_baseline_stderr: torch.Tensor,
    delta_stack: torch.Tensor,
    relative_delta_stack: torch.Tensor,
    row_normalized_relative_delta_stack: torch.Tensor,
    C_groups: Optional[torch.Tensor],
    C_levels: torch.Tensor,
    p_eff: torch.Tensor,
    n_eff: torch.Tensor,
    names: List[str],
    bin_labels: List[str],
    correlation_plot_name: str,
    correlation_title: str,
    correlation_axis_title: str,
    correlation_row_hover_label: str,
    correlation_col_hover_label: str,
) -> None:
    shared_analysis.save_results_and_plots(
        output_dir=output_dir,
        results=results,
        baseline_mean=baseline_mean,
        baseline_stderr=baseline_stderr,
        raw_baseline_mean=raw_baseline_mean,
        raw_baseline_stderr=raw_baseline_stderr,
        delta_stack=delta_stack,
        relative_delta_stack=relative_delta_stack,
        row_normalized_relative_delta_stack=row_normalized_relative_delta_stack,
        C_groups=C_groups,
        C_levels=C_levels,
        p_eff=p_eff,
        n_eff=n_eff,
        names=names,
        axis_metadata={
            "bin_labels": bin_labels,
            "axis_title": correlation_axis_title,
            "correlation_plot_name": correlation_plot_name,
            "correlation_title": correlation_title,
            "row_hover_label": correlation_row_hover_label,
            "col_hover_label": correlation_col_hover_label,
        },
    )


def log_wandb_outputs(
    output_dir: str,
    num_groups: int,
    total_parameters: int,
    baseline_mean: torch.Tensor,
    p_eff: torch.Tensor,
    n_eff: torch.Tensor,
    correlation_plot_name: str,
) -> None:
    wandb_metrics = {
        "num_groups": num_groups,
        "total_parameters": total_parameters,
    }
    for i in range(len(baseline_mean)):
        wandb_metrics[f"baseline/mean_bin_{i}"] = float(baseline_mean[i])
    for i in range(len(p_eff)):
        wandb_metrics[f"p_eff/bin_{i}"] = float(p_eff[i])
        wandb_metrics[f"n_eff/bin_{i}"] = float(n_eff[i])
    for plot_name in [
        "baseline_error.png",
        "baseline_error_unweighted.png",
        "effective_parameter_usage.png",
        "effective_group_count.png",
        "delta_heatmap.png",
        "relative_delta_heatmap.png",
        "row_normalized_relative_delta_heatmap.png",
        "relative_delta_heatmap_clipped.png",
        "group_correlation_heatmap.png",
        correlation_plot_name,
    ]:
        plot_path = os.path.join(output_dir, plot_name)
        if os.path.exists(plot_path):
            key = os.path.splitext(plot_name)[0]
            wandb_metrics[key] = wandb.Image(plot_path)
    wandb.log(wandb_metrics)
    for html_name in [
        "delta_heatmap.html",
        "relative_delta_heatmap.html",
        "row_normalized_relative_delta_heatmap.html",
        "relative_delta_heatmap_clipped.html",
        "group_correlation_heatmap.html",
        os.path.splitext(correlation_plot_name)[0] + ".html",
    ]:
        html_path = os.path.join(output_dir, html_name)
        if os.path.exists(html_path):
            wandb.save(html_path, base_path=output_dir)


def recompute_metrics_from_results(
    results: Dict,
    compute_group_correlation: bool = True,
    clip_mode: Optional[str] = None,
) -> Tuple[Dict, Dict[str, Optional[torch.Tensor]]]:
    """Recompute derived metrics (n_eff/p_eff/relative_delta_stack/...) from a saved
    results.json's raw ``delta_stack``, without rerunning ablations.

    ``clip_mode`` defaults to whatever mode the results were originally measured
    with (``results["importance_clip_mode"]``, falling back to the same
    ``config["importance_clip_mode"]`` recorded by argparse, and finally to
    ``"per_entry"`` for results.json files saved before this option existed --
    those were always produced with the per-entry clamp). Pass an explicit
    ``clip_mode`` to recompute a saved (signed) ``delta_stack`` under a different
    mode without a new GPU measurement.
    """
    group_names = list(results["group_names"])
    group_param_counts = {str(k): int(v) for k, v in results["group_param_counts"].items()}
    baseline_mean = torch.tensor(results["baseline_mean"], dtype=torch.float64)
    baseline_stderr = torch.tensor(results.get("baseline_stderr", [0.0] * len(baseline_mean)), dtype=torch.float64)
    delta_stack = torch.tensor(results["delta_stack"], dtype=torch.float64)
    direct_sampling_weights = results.get("group_sampling_weights")
    filter_sampling = results.get("filter_sampling")
    nested_sampling_weights = (
        filter_sampling.get("selected_group_expansion_weights")
        if isinstance(filter_sampling, dict)
        else None
    )

    def _coerce_sampling_weights(value: Any, field_name: str) -> Optional[Dict[str, float]]:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError(f"{field_name} must be a group-name-to-expansion-weight mapping")
        return {str(name): float(weight) for name, weight in value.items()}

    direct_sampling_weights = _coerce_sampling_weights(
        direct_sampling_weights,
        "group_sampling_weights",
    )
    nested_sampling_weights = _coerce_sampling_weights(
        nested_sampling_weights,
        "filter_sampling.selected_group_expansion_weights",
    )
    if (
        direct_sampling_weights is not None
        and nested_sampling_weights is not None
        and direct_sampling_weights != nested_sampling_weights
    ):
        raise ValueError(
            "group_sampling_weights does not match "
            "filter_sampling.selected_group_expansion_weights"
        )
    group_sampling_weights = (
        direct_sampling_weights
        if direct_sampling_weights is not None
        else nested_sampling_weights
    )

    if clip_mode is None:
        clip_mode = results.get("importance_clip_mode") or results.get("config", {}).get(
            "importance_clip_mode", "per_entry"
        )

    usage_metrics = compute_usage_metrics(
        delta_stack=delta_stack,
        baseline_mean=baseline_mean,
        group_param_counts=group_param_counts,
        group_names=group_names,
        group_sampling_weights=group_sampling_weights,
        compute_group_correlation=compute_group_correlation,
        clip_mode=clip_mode,
    )

    config = results.get("config", {})
    model_info = results.get("model_info", {})
    num_bins = int(config.get("num_bins", len(baseline_mean)))
    sigma_values_strided = None
    if "sigma_values" in results:
        sigma_values = torch.tensor(results["sigma_values"], dtype=torch.float64)
        sigma_stride = int(config.get("sigma_stride", 1))
        sigma_values_strided = sigma_values[::sigma_stride]
        model_family = str(model_info.get("model_family") or config.get("model_family") or "edm")
        sigma_data = float(config.get("sigma_data") if config.get("sigma_data") is not None else 0.5)
        raw_baseline_mean, raw_baseline_stderr = compute_raw_baseline_metrics(
            baseline_mean=baseline_mean,
            baseline_stderr=baseline_stderr,
            sigma_values_strided=sigma_values_strided,
            model_family=model_family,
            sigma_data=sigma_data,
            num_bins=num_bins,
        )
    else:
        raw_baseline_mean = torch.tensor(results.get("raw_baseline_mean", results["baseline_mean"]), dtype=torch.float64)
        raw_baseline_stderr = torch.tensor(
            results.get("raw_baseline_stderr", results.get("baseline_stderr", [0.0] * len(baseline_mean))),
            dtype=torch.float64,
        )

    axis_metadata = infer_bin_axis_metadata(results, num_bins=len(baseline_mean))

    updated_results = dict(results)
    updated_results["group_param_counts"] = group_param_counts
    updated_results["raw_baseline_mean"] = raw_baseline_mean.tolist()
    updated_results["raw_baseline_stderr"] = raw_baseline_stderr.tolist()
    updated_results["relative_delta_stack"] = usage_metrics["relative_delta_stack"].tolist()
    updated_results["row_normalized_relative_delta_stack"] = usage_metrics["row_normalized_relative_delta_stack"].tolist()
    updated_results["sampling_adjusted_delta_stack"] = usage_metrics["sampling_adjusted_delta_stack"].tolist()
    updated_results["C_groups"] = tensor_or_none_to_list(usage_metrics["C_groups"])
    updated_results[str(axis_metadata["correlation_key"])] = usage_metrics["C_noise_levels"].tolist()
    updated_results["weights"] = usage_metrics["weights"].tolist()
    updated_results["n_eff"] = usage_metrics["n_eff"].tolist()
    updated_results["p_eff"] = usage_metrics["p_eff"].tolist()
    updated_results["importance_clip_mode"] = clip_mode
    if group_sampling_weights is not None:
        updated_results["group_sampling_weights"] = group_sampling_weights
    return updated_results, {
        **usage_metrics,
        "baseline_mean": baseline_mean,
        "baseline_stderr": baseline_stderr,
        "raw_baseline_mean": raw_baseline_mean,
        "raw_baseline_stderr": raw_baseline_stderr,
        "sigma_values_strided": sigma_values_strided,
        "axis_metadata": axis_metadata,
    }


def plot_labeled_heatmap(
    matrix: torch.Tensor,
    row_labels: List[str],
    col_labels: List[str],
    out_path: str,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    colorbar_title: str,
    row_hover_label: str,
    col_hover_label: str,
    hover_value_label: str,
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
) -> None:
    arr = matrix.cpu().numpy()

    use_plotly = max(len(row_labels), len(col_labels)) > 100
    if use_plotly:
        html_path = os.path.splitext(out_path)[0] + ".html"
        if go is None:
            print(f"plotly is not installed; skipping the interactive heatmap {html_path}")
            return
        fig = go.Figure(
            data=go.Heatmap(
                z=arr,
                x=col_labels,
                y=row_labels,
                zmin=zmin,
                zmax=zmax,
                colorbar={"title": colorbar_title},
                hovertemplate=(
                    f"{row_hover_label}=%{{y}}<br>"
                    f"{col_hover_label}=%{{x}}<br>"
                    f"{hover_value_label}=%{{z}}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            height=min(max(600, len(row_labels) * 14), 2400),
        )
        fig.update_xaxes(type="category")
        fig.update_yaxes(type="category")

        x_tick_positions = _sparse_tick_positions(len(col_labels))
        y_tick_positions = _sparse_tick_positions(len(row_labels))
        fig.update_xaxes(
            tickmode="array",
            tickvals=[col_labels[i] for i in x_tick_positions],
            ticktext=[col_labels[i] for i in x_tick_positions],
        )
        fig.update_yaxes(
            tickmode="array",
            tickvals=[row_labels[i] for i in y_tick_positions],
            ticktext=[row_labels[i] for i in y_tick_positions],
        )
        fig.write_html(html_path)
        return

    plt.figure(figsize=(12, max(4, min(20, 0.25 * len(row_labels)))))
    plt.imshow(arr, aspect="auto", interpolation="nearest", vmin=zmin, vmax=zmax)
    plt.xticks(range(len(col_labels)), col_labels, rotation=90, fontsize=8)
    plt.yticks(range(len(row_labels)), row_labels, fontsize=8)
    plt.xlabel(xaxis_title)
    plt.ylabel(yaxis_title)
    plt.title(title)
    plt.colorbar(label=colorbar_title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_delta_heatmap(
    names: List[str],
    delta_stack: torch.Tensor,
    out_path: str,
    title: str = "Ablation delta heatmap",
    colorbar_title: str = "Delta",
    hover_label: str = "delta",
) -> None:
    plot_labeled_heatmap(
        matrix=delta_stack,
        row_labels=names,
        col_labels=[str(i) for i in range(delta_stack.shape[1])],
        out_path=out_path,
        title=title,
        xaxis_title="Noise-level bin",
        yaxis_title="Group",
        colorbar_title=colorbar_title,
        row_hover_label="group",
        col_hover_label="bin",
        hover_value_label=hover_label,
    )


# ---------------------------
# Main
# ---------------------------

def main():
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="image_folder",
        choices=["image_folder", "cifar10", "imagenet1k_parquet", "imagenet", "ffhq", "lsun_bedroom"],
    )
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--dataset-manifest", "--dataset_manifest", dest="dataset_manifest", default=None)
    parser.add_argument(
        "--dataset-split",
        "--dataset_split",
        dest="dataset_split",
        choices=["all", "train", "monitor", "validation", "val", "fid"],
        default="monitor",
        help="Named FFHQ/LSUN split. Monitoring subsets intentionally overlap training.",
    )
    parser.add_argument("--dataset-preflight", "--dataset_preflight", dest="dataset_preflight", action="store_true")
    parser.add_argument("--ffhq-protocol", "--ffhq_protocol", dest="ffhq_protocol", default=FFHQ_PROTOCOL)
    parser.add_argument("--lsun-monitor-size", "--lsun_monitor_size", dest="lsun_monitor_size", type=int, default=DEFAULT_LSUN_MONITOR_SIZE)
    parser.add_argument("--lsun-monitor-seed", "--lsun_monitor_seed", dest="lsun_monitor_seed", type=int, default=DEFAULT_LSUN_MONITOR_SEED)
    parser.add_argument("--parquet_image_column", type=str, default="image")
    parser.add_argument("--parquet_label_column", type=str, default="label")
    parser.add_argument(
        "--parquet_split",
        type=str,
        default="all",
        choices=["all", "train", "validation", "val", "test"],
        help=(
            "For --dataset imagenet1k_parquet, filter flat parquet shards by "
            "filename prefix. Use 'val' as an alias for 'validation'."
        ),
    )
    parser.add_argument("--cifar_split", type=str, default="validation", choices=["validation", "test", "train"])
    parser.add_argument("--imagenet_split", type=str, default="val", choices=["val", "train"])
    parser.add_argument("--download", action="store_true", help="Download the dataset if needed. Useful for CIFAR-10.")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--network-source",
        "--network-pkl",
        "--network_pkl",
        dest="network_pkl",
        type=str,
        default=None,
        help="Teacher checkpoint path/URL; --network-pkl is retained as a legacy alias.",
    )
    parser.add_argument("--network-format", "--network_format", dest="network_format", default=None)
    parser.add_argument("--network-preset", "--network_preset", dest="network_preset", default=None)
    parser.add_argument("--model-cache-dir", "--model_cache_dir", dest="model_cache_dir", default=None)
    parser.add_argument(
        "--trust-local-pickle",
        action="store_true",
        help="Allow unpickling a local/third-party NVLabs checkpoint after verifying its origin.",
    )
    parser.add_argument("--recompute_metrics_from", type=str, default=None, help="Recompute derived metrics and plots from an existing results.json without rerunning ablations.")
    parser.add_argument(
        "--grouping",
        type=str,
        default="per_filter",
        choices=["blocks", "attention", "attention_heads", "per_filter"],
        help="Ablation groups. per_filter (default, the paper setting) uses one group per convolution output channel.",
    )
    parser.add_argument(
        "--ablation_mode",
        type=str,
        default="pfi",
        choices=["zero", "random_same_norm", "pfi"],
        help=(
            "How to replace group outputs. pfi (default, the paper protocol) exchanges complete "
            "activations using a deterministic, fixed-point-free batch-local permutation at each "
            "exact sigma. random_same_norm is the legacy protocol of the released CIFAR-10 and "
            "ImageNet-64 profiles; zero is a legacy baseline."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--pfi_seed",
        "--pfi-seed",
        dest="pfi_seed",
        type=int,
        default=None,
        help="Global seed for exact-sigma PFI; defaults to --seed when omitted.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument(
        "--filter_sampling",
        "--filter-sampling",
        dest="filter_sampling",
        choices=["exhaustive", "stratified_module"],
        default="exhaustive",
        help=(
            "Per-filter population selector. exhaustive preserves the legacy all-filter profile; "
            "stratified_module deterministically samples up to --filters_per_module filters "
            "independently from every raw convolution module."
        ),
    )
    parser.add_argument(
        "--filters_per_module",
        "--filters-per-module",
        dest="filters_per_module",
        type=int,
        default=None,
        help="Required positive per-module cap for --filter_sampling stratified_module.",
    )
    parser.add_argument(
        "--filter_sampling_seed",
        "--filter-sampling-seed",
        dest="filter_sampling_seed",
        type=int,
        default=None,
        help="Stable per-module filter-membership seed; defaults to --seed.",
    )
    parser.add_argument("--max_groups", type=int, default=None, help="If set, evaluate only N collected groups.")
    parser.add_argument(
        "--confirm-full-profile",
        action="store_true",
        help="Explicitly acknowledge a full FFHQ/Bedroom per-filter profile without --max_groups.",
    )
    parser.add_argument("--max_groups_mode", type=str, default="random", choices=["first", "random"], help="How to choose groups when --max_groups is set.")
    parser.add_argument("--samples_per_image", type=int, default=2, help="Deprecated: ignored because every image is now paired with all sampled sigma levels.")
    parser.add_argument("--num_bins", type=int, default=20, help="Number of noise bins (the paper uses 20).")
    parser.add_argument("--num_sigma_levels", type=int, default=256)
    parser.add_argument("--sigma_stride", type=int, default=1)
    parser.add_argument(
        "--corruption_order", type=str, default="level_major",
        choices=["level_major", "image_major"],
        help="Sample ordering for the corruption dataset. 'level_major' (default) groups same-level "
             "examples into each batch so permutation importance holds t fixed. 'image_major' is the "
             "legacy order that (with batch_size<num_timestep_levels) causes the period-batch_size "
             "n_eff artifact; use only to reproduce old runs.",
    )
    parser.add_argument("--image_size", type=int, default=None, help="Defaults to the checkpoint resolution.")
    parser.add_argument("--sigma_min", type=float, default=None)
    parser.add_argument("--sigma_max", type=float, default=None)
    parser.add_argument("--sigma_data", type=float, default=None)
    parser.add_argument("--model_family", type=str, default="auto", choices=["auto", "edm", "vp", "ve"])
    parser.add_argument("--class_idx", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--distributed-timeout-seconds",
        type=int,
        default=86_400,
        help=(
            "Process-group timeout for uneven profile shards. The 24-hour default lets "
            "ranks with fewer groups wait while slower ranks finish and is not part of "
            "the analysis fingerprint."
        ),
    )
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--group_correlation",
        type=str,
        default="auto",
        choices=["auto", "always", "never"],
        help=(
            "Whether to compute and save the group-by-group correlation matrix. "
            "auto skips it when the group count is larger than --max_group_correlation_groups."
        ),
    )
    parser.add_argument(
        "--max_group_correlation_groups",
        type=int,
        default=2048,
        help="Maximum group count for --group_correlation auto.",
    )
    parser.add_argument("--wandb_project", type=str, default=None, help="If set, log metrics and plots to this W&B project.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Optional W&B run name.")
    parser.add_argument(
        "--importance_clip_mode",
        type=str,
        default=None,
        choices=list(IMPORTANCE_CLIP_MODES),
        help=(
            "How the negative-delta floor is applied to per-head permutation-importance "
            "deltas. 'per_entry' (default) clamps each (group, bin) delta at 0 before any "
            "aggregation -- byte-identical to this script's original behavior. 'post_agg' "
            "keeps the per-entry deltas signed (saved delta_stack/relative_delta_stack stay "
            "signed) and only clamps the weights/n_eff/p_eff aggregation input. 'none' saves "
            "fully signed aggregates with no clamp anywhere; downstream consumers that need "
            "non-negativity must handle it at consumption. With --recompute_metrics_from and "
            "no explicit value, the mode recorded in the loaded results.json is reused."
        ),
    )
    args = parser.parse_args()
    importance_clip_mode_explicit = args.importance_clip_mode
    if args.importance_clip_mode is None:
        args.importance_clip_mode = "per_entry"
    if args.pfi_seed is None:
        args.pfi_seed = args.seed
    if args.filter_sampling_seed is None:
        args.filter_sampling_seed = args.seed

    if args.recompute_metrics_from is None:
        try:
            validate_filter_sampling_options(
                grouping=args.grouping,
                filter_sampling=args.filter_sampling,
                filters_per_module=args.filters_per_module,
                max_groups=args.max_groups,
            )
        except ValueError as exc:
            parser.error(str(exc))
        if args.output_dir is None:
            parser.error("--output_dir is required unless --recompute_metrics_from is set")
        if args.network_pkl is None:
            if args.network_preset is None:
                parser.error("--network-source or --network-preset is required unless --recompute_metrics_from is set")
            from pace.teacher_models import teacher_preset_config

            args.network_pkl = teacher_preset_config(args.network_preset)["source"]
        if urllib.parse.urlparse(args.network_pkl).scheme in {"http", "https"} and args.model_cache_dir is None:
            parser.error("Remote teacher checkpoints require --model-cache-dir")
        if (
            args.dataset in {"ffhq", "lsun_bedroom"}
            and args.grouping == "per_filter"
            and args.filter_sampling == "exhaustive"
            and args.max_groups is None
            and not args.confirm_full_profile
        ):
            estimated_groups = 32_387 if args.dataset == "ffhq" else 119_555
            parser.error(
                f"A full {args.dataset} per-filter profile is estimated to contain "
                f"{estimated_groups:,} groups and requires one complete ablation evaluation "
                "per group. Pass --max_groups for a bounded smoke run, or "
                "--confirm-full-profile to acknowledge the full cost."
            )
    elif args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.recompute_metrics_from)) or "."

    # Validate named image populations (and, when requested, every pixel) before
    # initializing distributed GPU state or loading a multi-GB teacher.
    preloaded_shared_dataset = None
    preloaded_shared_image_size = None
    if args.recompute_metrics_from is None and args.dataset in {"ffhq", "lsun_bedroom"}:
        try:
            preloaded_shared_image_size = resolve_shared_dataset_image_size(args)
            preloaded_shared_dataset = make_shared_analysis_dataset(
                args,
                image_size=preloaded_shared_image_size,
            )
        except (FileNotFoundError, PermissionError, ValueError) as exc:
            parser.error(f"Dataset validation failed before teacher loading: {exc}")

    device, rank, world_size = init_distributed(
        args.device,
        timeout_seconds=args.distributed_timeout_seconds,
    )
    args.device = device

    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
    if is_distributed():
        dist.barrier()

    set_seed(args.seed + rank)
    dtype = choose_dtype(args.dtype)

    use_wandb = args.wandb_project is not None and is_main_process()
    if use_wandb:
        if not HAS_WANDB:
            raise ImportError("wandb is required when --wandb_project is set. Install with: pip install wandb")
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    evaluator = None
    dataloader = None
    image_dataset = None
    corruption_dataset = None
    try:
        if args.recompute_metrics_from is not None:
            if not is_main_process():
                return
            print(f"Recomputing derived metrics from {args.recompute_metrics_from}...")
            results = load_json(args.recompute_metrics_from)
            num_groups = len(results.get("group_names", []))
            compute_group_correlation = should_compute_group_correlation(
                args.group_correlation,
                num_groups=num_groups,
                max_groups=args.max_group_correlation_groups,
            )
            if not compute_group_correlation:
                print(
                    f"Skipping group correlation matrix for {num_groups} groups "
                    f"(--group_correlation={args.group_correlation})."
                )
            results, tensors = recompute_metrics_from_results(
                results,
                compute_group_correlation=compute_group_correlation,
                clip_mode=importance_clip_mode_explicit,
            )
            results["group_correlation"] = {
                "mode": args.group_correlation,
                "computed": compute_group_correlation,
                "max_groups": args.max_group_correlation_groups,
                "num_groups": num_groups,
            }
            axis_metadata = tensors["axis_metadata"]
            if "sigma_bin_labels" not in results and tensors["sigma_values_strided"] is not None:
                sigma_bin_labels = make_sigma_bin_labels(
                    sigma_values=tensors["sigma_values_strided"],
                    num_bins=len(tensors["baseline_mean"]),
                )
                results["sigma_bin_labels"] = sigma_bin_labels
                axis_metadata = infer_bin_axis_metadata(results, num_bins=len(tensors["baseline_mean"]))
            save_results_and_plots(
                output_dir=args.output_dir,
                results=results,
                baseline_mean=tensors["baseline_mean"],
                baseline_stderr=tensors["baseline_stderr"],
                raw_baseline_mean=tensors["raw_baseline_mean"],
                raw_baseline_stderr=tensors["raw_baseline_stderr"],
                delta_stack=tensors["delta_stack"],
                relative_delta_stack=tensors["relative_delta_stack"],
                row_normalized_relative_delta_stack=tensors["row_normalized_relative_delta_stack"],
                C_groups=tensors["C_groups"],
                C_levels=tensors["C_noise_levels"],
                p_eff=tensors["p_eff"],
                n_eff=tensors["n_eff"],
                names=list(results["group_names"]),
                bin_labels=list(axis_metadata["bin_labels"]),
                correlation_plot_name=str(axis_metadata["correlation_plot_name"]),
                correlation_title=str(axis_metadata["correlation_title"]),
                correlation_axis_title=str(axis_metadata["axis_title"]),
                correlation_row_hover_label=str(axis_metadata["row_hover_label"]),
                correlation_col_hover_label=str(axis_metadata["col_hover_label"]),
            )
            if use_wandb:
                log_wandb_outputs(
                    output_dir=args.output_dir,
                    num_groups=len(results["group_names"]),
                    total_parameters=sum(int(v) for v in results["group_param_counts"].values()),
                    baseline_mean=tensors["baseline_mean"],
                    p_eff=tensors["p_eff"],
                    n_eff=tensors["n_eff"],
                    correlation_plot_name=str(axis_metadata["correlation_plot_name"]),
                )
            print("Done.")
            print(f"Results saved to: {args.output_dir}")
            return

        print(f"[rank {rank}] Loading EDM network on {args.device}...")
        evaluator = EDMUsageEvaluator(
            network_pkl=args.network_pkl,
            device=args.device,
            dtype=dtype,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            sigma_data=args.sigma_data,
            num_sigma_levels=args.num_sigma_levels,
            class_idx=args.class_idx,
            model_family=args.model_family,
            network_format=args.network_format,
            network_preset=args.network_preset,
            model_cache_dir=args.model_cache_dir,
            trust_local_pickle=args.trust_local_pickle,
        )

        if is_main_process():
            print(f"Using model_family={evaluator.model_family}")

        image_size = preloaded_shared_image_size or args.image_size or evaluator.img_resolution
        if image_size != evaluator.img_resolution:
            print(
                f"[rank {rank}] Warning: using image_size={image_size} for a checkpoint with "
                f"img_resolution={evaluator.img_resolution}"
            )

        if evaluator.img_channels != 3:
            raise ValueError(
                f"Only RGB EDM checkpoints are supported by this script, got img_channels={evaluator.img_channels}"
            )

        print("Preparing dataset...")
        if args.dataset == "cifar10":
            image_dataset = CIFAR10Dataset(
                root=args.data_root,
                image_size=image_size,
                split=args.cifar_split,
                max_images=args.max_images,
                download=args.download,
            )
            if is_main_process() and args.cifar_split == "validation":
                print("Using CIFAR-10 test split as the validation set proxy.")
        elif args.dataset == "imagenet1k_parquet":
            image_dataset = ImageNet1KParquetDataset(
                root=args.data_root,
                image_size=image_size,
                max_images=args.max_images,
                image_column=args.parquet_image_column,
                label_column=args.parquet_label_column,
                split=args.parquet_split,
            )
            if is_main_process():
                split_name = image_dataset.split or "all"
                print(
                    f"Using ImageNet-1K parquet split {split_name} "
                    f"({len(image_dataset)} images, {len(image_dataset.paths)} shards)."
                )
        elif args.dataset == "imagenet":
            image_dataset = ImageNetDataset(
                root=args.data_root,
                image_size=image_size,
                split=args.imagenet_split,
                max_images=args.max_images,
                download=args.download,
            )
            if is_main_process():
                print(f"Using ImageNet {args.imagenet_split} split ({len(image_dataset)} images).")
        elif args.dataset in {"ffhq", "lsun_bedroom"}:
            assert preloaded_shared_dataset is not None
            image_dataset = preloaded_shared_dataset
            if is_main_process():
                print(
                    f"Using {args.dataset} {image_dataset.split} split "
                    f"({len(image_dataset):,} images, held_out={image_dataset.metadata['is_heldout']})."
                )
        else:
            if not args.image_root:
                raise ValueError("--image_root is required when --dataset image_folder")
            image_dataset = SharedImageFolder(
                root=args.image_root,
                image_size=image_size,
                max_images=args.max_images,
            )

        corruption_dataset = SigmaCorruptionDataset(
            image_dataset=image_dataset,
            sigma_values=evaluator.sigma_values,
            samples_per_image=args.samples_per_image,
            seed=args.seed,
            sigma_stride=args.sigma_stride,
            order=args.corruption_order,
        )
        if is_main_process() and args.samples_per_image != 2:
            print("Note: --samples_per_image is deprecated and ignored; every image is evaluated at every sampled sigma level.")

        pfi_plan = None
        if args.ablation_mode == "pfi":
            pfi_plan = ExactSigmaPFIBatchSampler(
                corruption_dataset,
                batch_size=args.batch_size,
                pfi_seed=args.pfi_seed,
                population_fingerprint=image_population_fingerprint(image_dataset),
            )
            dataloader = DataLoader(
                corruption_dataset,
                batch_sampler=pfi_plan,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=collate_corruption_pfi,
            )
            if is_main_process():
                print("PFI batch plan:")
                print(json.dumps(pfi_plan.artifact, indent=2, sort_keys=True))
            pfi_plan_path = os.path.join(args.output_dir, "pfi_plan.pt")
            if is_main_process():
                persist_or_validate_pfi_plan(pfi_plan_path, pfi_plan, allow_create=True)
            if is_distributed():
                dist.barrier()
            persist_or_validate_pfi_plan(pfi_plan_path, pfi_plan, allow_create=False)
        else:
            dataloader = DataLoader(
                corruption_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=collate_corruption,
            )

        print(f"[rank {rank}] Collecting groups...")
        if args.grouping == "blocks":
            groups = collect_block_groups_edm(evaluator.net)
        elif args.grouping == "attention":
            groups = collect_attention_groups_edm(evaluator.net)
        elif args.grouping == "attention_heads":
            groups = collect_attention_head_groups_edm(evaluator.net)
        else:
            groups = collect_per_filter_groups_edm(evaluator.net)

        if not groups:
            raise ValueError(
                f"No groups found for grouping={args.grouping}. "
                "If this checkpoint uses different module names, the collector may need to be specialized."
            )

        full_group_count = len(groups)
        filter_sampling_protocol = None
        if args.grouping == "per_filter":
            groups, filter_sampling_protocol = select_filter_groups(
                groups,
                mode=args.filter_sampling,
                filters_per_module=args.filters_per_module,
                seed=args.filter_sampling_seed,
            )
            if (
                args.filter_sampling == "stratified_module"
                and len(groups) == full_group_count
                and args.dataset in {"ffhq", "lsun_bedroom"}
                and not args.confirm_full_profile
            ):
                raise ValueError(
                    "--filter_sampling stratified_module selected the complete per-filter population; "
                    "pass --confirm-full-profile or choose a smaller --filters_per_module"
                )
            if is_main_process():
                print(
                    f"Filter sampling mode={args.filter_sampling} selected {len(groups):,}/"
                    f"{full_group_count:,} filters across "
                    f"{filter_sampling_protocol['selected_module_count']:,} modules "
                    f"(selection_sha256={filter_sampling_protocol['selection_sha256']})."
                )
        if args.max_groups is not None:
            if args.max_groups <= 0:
                raise ValueError(f"max_groups must be positive, got {args.max_groups}")
            group_items_full = list(groups.items())
            selected_count = min(args.max_groups, len(group_items_full))
            if args.max_groups_mode == "random":
                selection_rng = random.Random(args.seed)
                selected_group_items = selection_rng.sample(group_items_full, k=selected_count)
                groups = dict(selected_group_items)
                if is_main_process():
                    print(
                        f"Randomly sampled {len(groups)} groups from {len(group_items_full)} total "
                        f"due to --max_groups={args.max_groups}."
                    )
            else:
                groups = dict(group_items_full[:selected_count])
                if is_main_process():
                    print(f"Limiting evaluation to the first {len(groups)} groups due to --max_groups={args.max_groups}.")

            # ``max_groups`` is the legacy bounded/smoke selector, not an
            # exhaustive population profile and not the stratified estimator.
            # Omitting the exhaustive record prevents downstream consumers
            # from mistaking the truncated rows for a complete population.
            filter_sampling_protocol = None

        group_sampling_weights = None
        if filter_sampling_protocol is not None and args.filter_sampling == "stratified_module":
            expansion_weights = filter_sampling_protocol["selected_group_expansion_weights"]
            group_sampling_weights = {name: float(expansion_weights[name]) for name in groups}

        if args.grouping == "attention_heads":
            group_param_counts = {
                name: max(
                    1,
                    int(getattr(module, "_diffdist_attention_parameter_count", count_parameters(module)))
                    // max(1, _get_module_num_heads(module) or 1),
                )
                for name, target in groups.items()
                for module in [target[0]]
            }
        elif args.grouping == "per_filter":
            group_param_counts = {
                name: count_filter_parameters(module, filter_idx)
                for name, (module, filter_idx, _) in groups.items()
            }
        else:
            group_param_counts = {name: count_parameters(module) for name, module in groups.items()}

        if is_main_process():
            print(f"Found {len(groups)} groups across {world_size} process(es):")
            group_preview_limit = 50
            for name, pcount in list(group_param_counts.items())[:group_preview_limit]:
                print(f"  {name}: {pcount:,}")
            if len(group_param_counts) > group_preview_limit:
                print(f"  ... {len(group_param_counts) - group_preview_limit:,} additional groups omitted")

        profile_cost_estimate = build_profile_cost_estimate(
            grouping=args.grouping,
            full_group_count=full_group_count,
            selected_group_count=len(groups),
            bounded_by_max_groups=args.max_groups is not None,
            dataset_images=len(image_dataset),
            sigma_levels_per_image=len(range(0, len(evaluator.sigma_values), args.sigma_stride)),
            world_size=world_size,
            requested_batch_size=args.batch_size if pfi_plan is not None else None,
            actual_batch_sizes=list(pfi_plan.batch_sizes) if pfi_plan is not None else None,
            homogeneous_batches_per_evaluation=len(pfi_plan) if pfi_plan is not None else None,
        )
        if filter_sampling_protocol is not None:
            profile_cost_estimate["filter_sampling"] = {
                "mode": filter_sampling_protocol["mode"],
                "bounded_by_filter_sampling": filter_sampling_protocol["selected_group_count"]
                < filter_sampling_protocol["population_group_count"],
                "population_group_count": filter_sampling_protocol["population_group_count"],
                "selected_group_count_before_max_groups": filter_sampling_protocol["selected_group_count"],
                "evaluated_group_count": len(groups),
                "filters_per_module": filter_sampling_protocol["filters_per_module"],
                "selection_sha256": filter_sampling_protocol["selection_sha256"],
                "population_sha256": filter_sampling_protocol["population_sha256"],
            }
        if is_main_process():
            print("Profile cost estimate:")
            print(json.dumps(profile_cost_estimate, indent=2, sort_keys=True))

        ablation_protocol = build_ablation_protocol(
            args.ablation_mode,
            pfi_plan=pfi_plan,
            pfi_seed=args.pfi_seed if args.ablation_mode == "pfi" else None,
        )
        dataset_metadata = getattr(image_dataset, "metadata", {})
        grouping_fingerprint = {
            "kind": args.grouping,
            "group_names": list(groups),
            "group_param_counts": {name: int(value) for name, value in group_param_counts.items()},
        }
        # Deliberately retain the byte-for-byte legacy exhaustive fingerprint
        # shape so active exhaustive PFI shards remain resumable.  Stratified
        # profiles bind the complete versioned selector record strictly.
        if filter_sampling_protocol is not None and args.filter_sampling == "stratified_module":
            grouping_fingerprint["filter_sampling"] = filter_sampling_protocol

        profile_fingerprint = {
            "format": PROFILE_FINGERPRINT_FORMAT,
            "teacher": evaluator.teacher_spec.to_dict(),
            "model": {
                "checkpoint_sha256": evaluator.model_metadata.get("checkpoint_sha256"),
                "checkpoint_size_bytes": evaluator.model_metadata.get("checkpoint_size_bytes"),
                "checkpoint_format": evaluator.model_metadata.get("checkpoint_format"),
                "net_class_name": evaluator.model_metadata.get("net_class_name"),
                "model_family": evaluator.model_family,
                "requested_dtype": args.dtype,
                "resolved_net_dtype": str(evaluator.net_dtype),
                "image_size": int(image_size),
            },
            "runtime": {
                "dtype": args.dtype,
                "resolved_net_dtype": str(evaluator.net_dtype),
                "image_size": int(image_size),
                "requested_batch_size": int(args.batch_size),
            },
            "dataset": {
                "dataset": args.dataset,
                "population_fingerprint": image_population_fingerprint(image_dataset),
                "count": len(image_dataset),
                "split": dataset_metadata.get("split") if isinstance(dataset_metadata, dict) else getattr(image_dataset, "split", None),
                "entries_sha256": dataset_metadata.get("entries_sha256") if isinstance(dataset_metadata, dict) else None,
                "source_listing_sha256": dataset_metadata.get("source_listing_sha256") if isinstance(dataset_metadata, dict) else None,
                "source_records_sha256": dataset_metadata.get("source_records_sha256") if isinstance(dataset_metadata, dict) else None,
                "source_kind": dataset_metadata.get("source_kind") if isinstance(dataset_metadata, dict) else None,
                "image_size": dataset_metadata.get("image_size") if isinstance(dataset_metadata, dict) else int(image_size),
                "protocol": dataset_metadata.get("protocol") if isinstance(dataset_metadata, dict) else None,
                "dataset_spec": dataset_metadata.get("dataset_spec") if isinstance(dataset_metadata, dict) else None,
                "manifest_metadata": getattr(image_dataset, "manifest_metadata", {}),
            },
            "corruption": {
                "seed": int(args.seed),
                "sigma_indices": list(corruption_dataset.sigma_indices),
                "sigma_values": [
                    float(evaluator.sigma_values[index].detach().cpu().item())
                    for index in corruption_dataset.sigma_indices
                ],
                "num_bins": int(args.num_bins),
                "requested_batch_size": int(args.batch_size),
            },
            "grouping": grouping_fingerprint,
            "ablation_protocol": ablation_protocol,
        }
        profile_fingerprint_sha256 = canonical_json_sha256(profile_fingerprint)

        baseline_mean = torch.zeros(args.num_bins, dtype=torch.float64)
        baseline_stderr = torch.zeros(args.num_bins, dtype=torch.float64)
        baseline_count = torch.zeros(args.num_bins, dtype=torch.long)
        if hasattr(image_dataset, "set_sampling_seed"):
            image_dataset.set_sampling_seed(args.seed)
        if is_main_process():
            pfi_baseline_path = os.path.join(args.output_dir, "baseline_pfi.pt")
            cached_baseline = (
                load_pfi_baseline(
                    pfi_baseline_path,
                    num_bins=args.num_bins,
                    profile_fingerprint=profile_fingerprint,
                    profile_fingerprint_sha256=profile_fingerprint_sha256,
                )
                if args.ablation_mode == "pfi"
                else None
            )
            if cached_baseline is not None:
                print(f"Resuming PFI baseline from {pfi_baseline_path}")
                baseline_mean, baseline_stderr, baseline_count = cached_baseline
            else:
                print("Running baseline evaluation...")
                stats = evaluator.evaluate(
                    dataloader=dataloader,
                    num_bins=args.num_bins,
                    ablate_target=None,
                    progress_desc="baseline",
                )
                baseline_mean = stats.mean()
                baseline_stderr = stats.stderr()
                baseline_count = stats.count.clone()
                if args.ablation_mode == "pfi":
                    atomic_torch_save(
                        pfi_baseline_envelope(
                            mean=baseline_mean,
                            stderr=baseline_stderr,
                            count=baseline_count,
                            num_bins=args.num_bins,
                            profile_fingerprint=profile_fingerprint,
                            profile_fingerprint_sha256=profile_fingerprint_sha256,
                        ),
                        pfi_baseline_path,
                    )

        group_items = list(groups.items())
        indexed_group_items = list(enumerate(group_items))
        local_group_items = indexed_group_items[rank::world_size]

        print(f"[rank {rank}] Running {len(local_group_items)} / {len(group_items)} group ablations...")
        checkpoint_name = checkpoint_name_for_rank(args.ablation_mode, rank)
        checkpoint_path = os.path.join(args.output_dir, checkpoint_name)
        group_names_for_resume = [name for name, _ in group_items]
        completed_ablated_means, loaded_checkpoint_paths, ignored_unknown, ignored_bad_shape = load_ablation_checkpoints(
            output_dir=args.output_dir,
            ablation_mode=args.ablation_mode,
            expected_group_names=group_names_for_resume,
            num_bins=args.num_bins,
            profile_fingerprint=profile_fingerprint if args.ablation_mode == "pfi" else None,
            profile_fingerprint_sha256=profile_fingerprint_sha256 if args.ablation_mode == "pfi" else None,
        )
        local_ablated_means = {
            name: completed_ablated_means[name]
            for _, (name, _) in local_group_items
            if name in completed_ablated_means
        }
        if loaded_checkpoint_paths:
            print(
                f"[rank {rank}] Resumed {len(completed_ablated_means)} compatible groups "
                f"from {len(loaded_checkpoint_paths)} checkpoint shard(s)."
            )
            if ignored_unknown:
                print(f"[rank {rank}] Ignored {ignored_unknown} checkpoint entries for groups not in this run.")
            if ignored_bad_shape:
                preview = ", ".join(
                    f"{os.path.basename(path)}:{name}{shape}"
                    for path, name, shape in ignored_bad_shape[:3]
                )
                print(
                    f"[rank {rank}] Ignored {len(ignored_bad_shape)} checkpoint entries with incompatible shapes "
                    f"(expected ({args.num_bins},)); examples: {preview}"
                )

        for local_idx, (global_idx, (name, module)) in enumerate(local_group_items, start=1):
            if name in local_ablated_means:
                continue
            print(f"[rank {rank}] [{local_idx}/{len(local_group_items)}] Ablating {name}")
            if hasattr(image_dataset, "set_sampling_seed"):
                image_dataset.set_sampling_seed(args.seed)
            group_random_seed = args.seed + 2_000_000 + global_idx
            stats = evaluator.evaluate(
                dataloader=dataloader,
                num_bins=args.num_bins,
                ablate_target=module,
                ablation_mode=args.ablation_mode,
                ablation_random_seed=group_random_seed,
                progress_desc=f"rank {rank} ablation {local_idx}/{len(local_group_items)} {name}",
            )
            local_ablated_means[name] = stats.mean()
            if use_wandb:
                wandb.log({"ablation_progress": local_idx / len(local_group_items)})
            if args.ablation_mode == "pfi" or local_idx % 10 == 0:
                checkpoint_payload = (
                    pfi_checkpoint_envelope(
                        local_ablated_means,
                        rank=rank,
                        num_bins=args.num_bins,
                        profile_fingerprint=profile_fingerprint,
                        profile_fingerprint_sha256=profile_fingerprint_sha256,
                    )
                    if args.ablation_mode == "pfi"
                    else local_ablated_means
                )
                atomic_torch_save(checkpoint_payload, checkpoint_path)
            release_cuda_memory(stats)

        final_checkpoint_payload = (
            pfi_checkpoint_envelope(
                local_ablated_means,
                rank=rank,
                num_bins=args.num_bins,
                profile_fingerprint=profile_fingerprint,
                profile_fingerprint_sha256=profile_fingerprint_sha256,
            )
            if args.ablation_mode == "pfi"
            else local_ablated_means
        )
        atomic_torch_save(final_checkpoint_payload, checkpoint_path)

        if is_distributed():
            dist.barrier()

        ablated_means = gather_ablation_results(local_ablated_means)
        if len(ablated_means) != len(group_items):
            missing = sorted(set(name for name, _ in group_items) - set(ablated_means))
            raise RuntimeError(f"Missing ablation results for {len(missing)} groups: {missing[:5]}")

        if is_main_process():
            print("Computing deltas and usage metrics...")
            names = list(groups.keys())
            structural_mapper = getattr(evaluator.net, "structural_key_for_module", None)
            group_structural_keys = {}
            from pace.edm_distillation import module_to_structural_key

            for name in names:
                module_name = name.rsplit(".filter_", 1)[0]
                group_structural_keys[name] = str(
                    structural_mapper(module_name)
                    if callable(structural_mapper)
                    else module_to_structural_key(module_name)
                )
            signed_delta_stack, _ = compute_signed_and_positive_deltas(ablated_means, baseline_mean, names)
            # Allocation input: clip_mode decides whether the (group, bin) deltas are
            # floored per entry (legacy default, identical to the positive part above)
            # or kept signed.
            delta_stack = build_delta_stack(
                ablated_means=ablated_means,
                baseline_mean=baseline_mean,
                names=names,
                clip_mode=args.importance_clip_mode,
            )
            # Capacity allocation continues to use the positive part for full
            # backward compatibility.  PFI results additionally retain the
            # signed estimand so improvements under exchange are not erased.
            compute_group_correlation = should_compute_group_correlation(
                args.group_correlation,
                num_groups=len(names),
                max_groups=args.max_group_correlation_groups,
            )
            if not compute_group_correlation:
                print(
                    f"Skipping group correlation matrix for {len(names)} groups "
                    f"(--group_correlation={args.group_correlation})."
                )
            usage_metrics = compute_usage_metrics(
                delta_stack=delta_stack,
                baseline_mean=baseline_mean,
                group_param_counts=group_param_counts,
                group_names=names,
                group_sampling_weights=group_sampling_weights,
                compute_group_correlation=compute_group_correlation,
                clip_mode=args.importance_clip_mode,
            )

            sigma_values_strided = evaluator.sigma_values[::args.sigma_stride]
            raw_baseline_mean, raw_baseline_stderr = compute_raw_baseline_metrics(
                baseline_mean=baseline_mean,
                baseline_stderr=baseline_stderr,
                sigma_values_strided=sigma_values_strided,
                model_family=evaluator.model_family,
                sigma_data=evaluator.sigma_data,
                num_bins=args.num_bins,
            )
            sigma_bin_labels = make_sigma_bin_labels(
                sigma_values=sigma_values_strided,
                num_bins=args.num_bins,
            )
            axis_metadata = infer_bin_axis_metadata({"sigma_bin_labels": sigma_bin_labels}, num_bins=args.num_bins)

            print("Saving outputs...")
            invocation_argv = list(sys.argv)
            benchmark_protocols = list(benchmark_protocol_records(args.dataset))
            results = {
                "config": vars(args),
                "importance_clip_mode": args.importance_clip_mode,
                "invocation": {
                    "argv": invocation_argv,
                    "command": format_invocation_command(invocation_argv),
                },
                "dataset_info": {
                    "dataset": args.dataset,
                    "cifar_split": args.cifar_split if args.dataset == "cifar10" else None,
                    "cifar_classes": CIFAR10_CLASSES if args.dataset == "cifar10" else None,
                    "imagenet_split": args.imagenet_split if args.dataset == "imagenet" else None,
                    "parquet_split": getattr(image_dataset, "split", None) if args.dataset == "imagenet1k_parquet" else None,
                    "parquet_image_column": getattr(image_dataset, "image_column", None),
                    "parquet_label_column": getattr(image_dataset, "label_column", None),
                    "protocol": getattr(image_dataset, "metadata", {}).get("protocol"),
                    "split": getattr(image_dataset, "metadata", {}).get("split"),
                    "is_heldout": getattr(image_dataset, "metadata", {}).get("is_heldout"),
                    "count": getattr(image_dataset, "metadata", {}).get("count"),
                    "entries_sha256": getattr(image_dataset, "metadata", {}).get("entries_sha256"),
                    "source_listing_sha256": getattr(image_dataset, "metadata", {}).get("source_listing_sha256"),
                    "source_records_sha256": getattr(image_dataset, "metadata", {}).get("source_records_sha256"),
                    "source_kind": getattr(image_dataset, "metadata", {}).get("source_kind"),
                    "image_size": getattr(image_dataset, "metadata", {}).get("image_size"),
                    "dataset_spec": getattr(image_dataset, "metadata", {}).get("dataset_spec"),
                    "manifest_metadata": getattr(image_dataset, "manifest_metadata", {}),
                },
                "teacher": evaluator.teacher_spec.to_dict(),
                "benchmark_protocol_ids": [record["identity"] for record in benchmark_protocols],
                "benchmark_protocols": benchmark_protocols,
                "model_info": {
                    **evaluator.model_metadata,
                    "model_family": evaluator.model_family,
                },
                "ablation_info": {
                    "mode": args.ablation_mode,
                    "norm": "per_example_l2" if args.ablation_mode == "random_same_norm" else None,
                    "replacement": {
                        "zero": "zeros",
                        "random_same_norm": "gaussian_random_noise",
                        "pfi": "batch_local_exact_sigma_activation_permutation",
                    }[args.ablation_mode],
                },
                "ablation_protocol": ablation_protocol,
                "filter_sampling": filter_sampling_protocol,
                "profile_fingerprint": profile_fingerprint,
                "profile_fingerprint_sha256": profile_fingerprint_sha256,
                "group_names": names,
                "group_param_counts": {k: int(v) for k, v in group_param_counts.items()},
                "group_sampling_weights": group_sampling_weights,
                "group_structural_keys": group_structural_keys,
                "profile_cost_estimate": profile_cost_estimate,
                "baseline_mean": baseline_mean.tolist(),
                "baseline_stderr": baseline_stderr.tolist(),
                "raw_baseline_mean": raw_baseline_mean.tolist(),
                "raw_baseline_stderr": raw_baseline_stderr.tolist(),
                "baseline_count": baseline_count.tolist(),
                "sigma_values": evaluator.sigma_values.detach().cpu().tolist(),
                "delta_stack": delta_stack.tolist(),
                "relative_delta_stack": usage_metrics["relative_delta_stack"].tolist(),
                "row_normalized_relative_delta_stack": usage_metrics["row_normalized_relative_delta_stack"].tolist(),
                "sampling_adjusted_delta_stack": usage_metrics["sampling_adjusted_delta_stack"].tolist(),
                "C_groups": tensor_or_none_to_list(usage_metrics["C_groups"]),
                "C_noise_levels": usage_metrics["C_noise_levels"].tolist(),
                "sigma_bin_labels": sigma_bin_labels,
                "weights": usage_metrics["weights"].tolist(),
                "n_eff": usage_metrics["n_eff"].tolist(),
                "p_eff": usage_metrics["p_eff"].tolist(),
                "distributed": {
                    "world_size": world_size,
                },
                "group_correlation": {
                    "mode": args.group_correlation,
                    "computed": compute_group_correlation,
                    "max_groups": args.max_group_correlation_groups,
                    "num_groups": len(names),
                    "population_scope": (
                        "selected_filters_only"
                        if group_sampling_weights is not None
                        else "evaluated_groups"
                    ),
                    "sampling_expansion_applied": False,
                },
            }
            if args.ablation_mode == "pfi":
                results["signed_delta_stack"] = signed_delta_stack.tolist()
            save_results_and_plots(
                output_dir=args.output_dir,
                results=results,
                baseline_mean=baseline_mean,
                baseline_stderr=baseline_stderr,
                raw_baseline_mean=raw_baseline_mean,
                raw_baseline_stderr=raw_baseline_stderr,
                delta_stack=delta_stack,
                relative_delta_stack=usage_metrics["relative_delta_stack"],
                row_normalized_relative_delta_stack=usage_metrics["row_normalized_relative_delta_stack"],
                C_groups=usage_metrics["C_groups"],
                C_levels=usage_metrics["C_noise_levels"],
                p_eff=usage_metrics["p_eff"],
                n_eff=usage_metrics["n_eff"],
                names=names,
                bin_labels=sigma_bin_labels,
                correlation_plot_name=str(axis_metadata["correlation_plot_name"]),
                correlation_title=str(axis_metadata["correlation_title"]),
                correlation_axis_title=str(axis_metadata["axis_title"]),
                correlation_row_hover_label=str(axis_metadata["row_hover_label"]),
                correlation_col_hover_label=str(axis_metadata["col_hover_label"]),
            )

            if use_wandb:
                log_wandb_outputs(
                    output_dir=args.output_dir,
                    num_groups=len(names),
                    total_parameters=sum(group_param_counts.values()),
                    baseline_mean=baseline_mean,
                    p_eff=usage_metrics["p_eff"],
                    n_eff=usage_metrics["n_eff"],
                    correlation_plot_name=str(axis_metadata["correlation_plot_name"]),
                )

            print("Done.")
            print(f"Results saved to: {args.output_dir}")
    finally:
        release_cuda_memory(dataloader, corruption_dataset, image_dataset, evaluator)
        if is_main_process():
            elapsed = time.perf_counter() - start_time
            print(f"Total runtime: {elapsed:.2f}s ({elapsed / 60:.2f} min)")
            if use_wandb:
                wandb.log({"runtime_seconds": elapsed})
                wandb.finish()
        cleanup_distributed()


if __name__ == "__main__":
    main()
