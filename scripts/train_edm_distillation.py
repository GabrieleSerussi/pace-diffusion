#!/usr/bin/env python3
"""Distill an EDM U-Net student system from a prepared PACE architecture plan.

The plan is one ``architecture_plan.json`` written by
``scripts/prepare_edm_distillation.py`` (or a released
``artifacts/plans/<dataset>/<variant>/architecture_plan.json``).  The loss is
the hybrid objective of Eq. 2: ``--kd-weight`` times the teacher-matching MSE
plus ``--data-weight`` times the clean-data MSE, both with the EDM
noise-level weighting.  Building the students needs the NVlabs/edm checkout
(``$EDM_REPO`` or ``../edm``).
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import io
import json
import math
import os
import random
import signal
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.parse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from pace.dataset_specs import DEFAULT_LSUN_MONITOR_SEED, DEFAULT_LSUN_MONITOR_SIZE, FFHQ_PROTOCOL
from pace.image_datasets import SharedImageDataset, validate_dataset_readability
from pace.jsonio import load_json
from pace.profile_provenance import provenance_from_artifact

from pace.edm_distillation import (
    construct_student_from_plan,
    create_ema,
    hybrid_distillation_loss,
    load_edm_network,
    loss_weights_for_family,
    make_class_labels,
    mse_per_example,
    update_ema,
    teacher_spec_from_plan,
)
from sample_edm_distilled import StackedRandomGenerator, edm_sampler, make_labels as make_sample_labels

LOWER_IS_BETTER = "↓"
HIGHER_IS_BETTER = "↑"
TRAINING_STATE_FILENAME = "training-state-latest.pt"
LATEST_CHECKPOINT_MANIFEST = "latest-checkpoint.json"


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    is_main: bool
    device: torch.device


class GracefulInterrupt:
    def __init__(self) -> None:
        self.requested = False
        self._previous_handlers: dict[signal.Signals, Any] = {}

    def __enter__(self) -> GracefulInterrupt:
        self.install()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.restore()

    def install(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle)

    def restore(self) -> None:
        for signum, handler in self._previous_handlers.items():
            signal.signal(signum, handler)
        self._previous_handlers.clear()

    def _handle(self, signum: int, frame: Any) -> None:
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True


def init_distributed(device_arg: str, *, timeout_minutes: float = 180.0) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        device = torch.device(device_arg)
        return DistributedContext(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            is_main=True,
            device=device,
        )

    if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
        raise RuntimeError("WORLD_SIZE > 1 requires launching with torchrun so RANK and LOCAL_RANK are set")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=float(timeout_minutes)))

    if device_arg.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(device_arg)
    return DistributedContext(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        is_main=(rank == 0),
        device=device,
    )


def cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.enabled and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def require_divisible_global_batch(name: str, value: int, world_size: int) -> int:
    if value % world_size != 0:
        raise ValueError(f"--{name.replace('_', '-')}={value} must be divisible by distributed world size {world_size}")
    local_value = value // world_size
    if local_value <= 0:
        raise ValueError(f"local {name} must be positive, got {local_value}")
    return local_value


def make_dist_sync_run_id(ctx: DistributedContext, device: torch.device) -> str:
    if not ctx.enabled:
        return "single-process"
    run_id = f"{int(time.time() * 1_000_000)}-{os.getpid()}" if ctx.is_main else None
    payload: list[str | None] = [run_id]
    kwargs = {"device": device} if device.type == "cuda" else {}
    dist.broadcast_object_list(payload, src=0, **kwargs)
    assert payload[0] is not None
    return str(payload[0])


def eval_error_path(output_dir: Path, *, step: int, run_id: str) -> Path:
    return Path(output_dir) / ".dist_sync" / run_id / f"step{int(step):06d}.error.json"


def read_eval_error_message(error_path: Path) -> str:
    try:
        payload = json.loads(error_path.read_text())
        return str(payload.get("error", str(payload)))
    except Exception:
        return error_path.read_text(errors="replace")


def sync_eval_completion(
    *,
    stop_after_eval: bool,
    ctx: DistributedContext,
    output_dir: Path,
    step: int,
    run_id: str,
    poll_seconds: float = 1.0,
) -> bool:
    if not ctx.enabled:
        return bool(stop_after_eval)

    sync_dir = output_dir / ".dist_sync" / run_id
    sync_path = sync_dir / f"step{int(step):06d}.json"
    error_path = eval_error_path(output_dir, step=step, run_id=run_id)
    if ctx.is_main:
        sync_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = sync_path.with_name(f"{sync_path.name}.tmp-{os.getpid()}")
        tmp_path.write_text(json.dumps({"step": int(step), "stop_after_eval": bool(stop_after_eval)}) + "\n")
        os.replace(tmp_path, sync_path)
        return bool(stop_after_eval)

    while True:
        if error_path.exists():
            raise RuntimeError(f"Rank 0 evaluation failed at step {int(step)}: {read_eval_error_message(error_path)}")
        if sync_path.exists():
            payload = json.loads(sync_path.read_text())
            if int(payload.get("step", -1)) == int(step):
                return bool(payload.get("stop_after_eval", False))
        time.sleep(float(poll_seconds))


def mark_eval_error(output_dir: Path, *, step: int, run_id: str, error: BaseException) -> None:
    sync_dir = output_dir / ".dist_sync" / run_id
    sync_dir.mkdir(parents=True, exist_ok=True)
    error_path = eval_error_path(output_dir, step=step, run_id=run_id)
    tmp_path = error_path.with_name(f"{error_path.name}.tmp-{os.getpid()}")
    payload = {
        "step": int(step),
        "error_type": error.__class__.__name__,
        "error": str(error),
    }
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_path, error_path)


def average_scalar_dict(values: dict[str, float], ctx: DistributedContext, device: torch.device) -> dict[str, float]:
    if not ctx.enabled:
        return dict(values)
    keys = list(values.keys())
    scalars = torch.tensor([float(values[key]) for key in keys], device=device, dtype=torch.float64)
    dist.all_reduce(scalars, op=dist.ReduceOp.SUM)
    scalars /= ctx.world_size
    return {key: float(value) for key, value in zip(keys, scalars.detach().cpu().tolist())}


def sync_stop_requested(requested: bool, ctx: DistributedContext, device: torch.device) -> bool:
    if not ctx.enabled:
        return bool(requested)
    flag = torch.tensor(int(requested), device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def load_plan(path: str | Path) -> dict:
    """Load a plain or gzip-compressed architecture plan."""
    return load_json(path)


def make_cifar_loader(
    *,
    data_root: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    download: bool,
    max_images: int | None,
    train: bool = True,
    shuffle: bool = True,
    drop_last: bool = True,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    dataset = datasets.CIFAR10(root=data_root, train=train, download=download, transform=transform)
    if max_images is not None:
        dataset = Subset(dataset, range(min(max_images, len(dataset))))
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        if distributed
        else None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _decode_parquet_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            import io

            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, (bytes, bytearray, memoryview)):
        import io

        return Image.open(io.BytesIO(bytes(value))).convert("RGB")
    if hasattr(value, "as_py"):
        return _decode_parquet_image(value.as_py())
    if hasattr(value, "shape"):
        return Image.fromarray(value).convert("RGB")
    raise TypeError(f"Unsupported parquet image value type: {type(value).__name__}")


def _infer_first_present(candidates: list[str], available: list[str]) -> str | None:
    available_set = set(available)
    for name in candidates:
        if name in available_set:
            return name
    return None


def normalize_imagenet_parquet_split(split: str | None) -> str | None:
    split = (split or "all").lower()
    if split in {"", "all"}:
        return None
    if split == "val":
        return "validation"
    if split not in {"train", "validation", "test"}:
        raise ValueError("parquet split must be one of: all, train, validation, val, test")
    return split


def _parquet_top_level_schema_names(parquet_file: Any) -> list[str]:
    if hasattr(parquet_file, "schema_arrow"):
        return list(parquet_file.schema_arrow.names)
    return list(parquet_file.schema.names)


def resolve_parquet_columns(parquet_file: Any, image_column: str | None, label_column: str | None) -> tuple[str, str]:
    top_level_names = _parquet_top_level_schema_names(parquet_file)
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


def normalize_imagenet_webdataset_split(split: str | None) -> str:
    split = (split or "train").lower()
    if split == "val":
        return "validation"
    if split not in {"train", "validation", "test"}:
        raise ValueError("WebDataset split must be one of: train, validation, val, test")
    return split


class ImageNet1KParquetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        root: str,
        image_size: int,
        split: str | None,
        max_images: int | None,
        image_column: str | None,
        label_column: str | None,
    ) -> None:
        import pyarrow.parquet as pq

        self.root = root
        self.split = normalize_imagenet_parquet_split(split)
        split_prefix = f"{self.split}-" if self.split is not None else None
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
            split_msg = "" if self.split is None else f" for split '{self.split}'"
            raise ValueError(f"No parquet files found in {root}{split_msg}")

        self._parquet_files = [pq.ParquetFile(path) for path in self.paths]
        self.image_column, self.label_column = resolve_parquet_columns(
            self._parquet_files[0],
            image_column=image_column,
            label_column=label_column,
        )

        self.row_offsets = [0]
        for parquet_file in self._parquet_files:
            self.row_offsets.append(self.row_offsets[-1] + parquet_file.metadata.num_rows)
        self.dataset_length = self.row_offsets[-1]
        self.max_images = self.dataset_length if max_images is None else min(max_images, self.dataset_length)
        self._cached_file_idx: int | None = None
        self._cached_table: Any | None = None
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return self.max_images

    def _load_row(self, dataset_idx: int) -> Any:
        file_idx = bisect.bisect_right(self.row_offsets, dataset_idx) - 1
        local_idx = dataset_idx - self.row_offsets[file_idx]
        if self._cached_file_idx != file_idx or self._cached_table is None:
            self._cached_table = self._parquet_files[file_idx].read(columns=[self.image_column, self.label_column])
            self._cached_file_idx = file_idx
        return self._cached_table.slice(local_idx, 1).to_pylist()[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self._load_row(idx)
        image = _decode_parquet_image(row[self.image_column])
        return self.transform(image), int(row[self.label_column])


class ImageNet1KWebDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        *,
        root: str,
        image_size: int,
        split: str,
        max_images: int | None,
        shuffle_shards: bool,
        shuffle_samples: bool,
        shuffle_buffer: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.root = Path(root)
        self.split = normalize_imagenet_webdataset_split(split)
        self.max_images = max_images
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_samples = bool(shuffle_samples)
        self.shuffle_buffer = max(0, int(shuffle_buffer))
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self.paths = sorted(self.root.glob(f"{self.split}-*.tar"))
        if not self.paths:
            raise ValueError(f"No WebDataset tar shards found in {self.root} for split '{self.split}'")

        self.metadata = self._load_metadata()
        split_metadata = (self.metadata.get("splits") or {}).get(self.split, {})
        self.dataset_length = int(split_metadata.get("num_samples", 0) or 0)
        samples_per_shard = int(self.metadata.get("samples_per_shard", 0) or 0)
        if self.max_images is not None and samples_per_shard > 0 and self.dataset_length > 0:
            num_needed = max(1, math.ceil(min(int(self.max_images), max(0, self.dataset_length)) / samples_per_shard))
            self.paths = self.paths[:num_needed]

    def _load_metadata(self) -> dict[str, Any]:
        metadata_path = self.root / "metadata.json"
        if not metadata_path.exists():
            return {}
        return json.loads(metadata_path.read_text())

    def __len__(self) -> int:
        if self.dataset_length <= 0:
            raise TypeError("ImageNet1KWebDataset length is unknown without metadata.json")
        if self.max_images is None:
            return self.dataset_length
        return min(int(self.max_images), self.dataset_length)

    def _paths_for_worker(self, paths: list[Path]) -> list[Path]:
        if self.world_size > 1:
            paths = paths[self.rank :: self.world_size]
        worker = torch.utils.data.get_worker_info()
        if worker is None:
            return paths
        return paths[worker.id :: worker.num_workers]

    def _rngs_for_worker(self) -> tuple[random.Random, random.Random]:
        worker = torch.utils.data.get_worker_info()
        if worker is None:
            base_seed = int(torch.initial_seed())
            worker_id = 0
        else:
            base_seed = int(worker.seed) - int(worker.id)
            worker_id = int(worker.id)
        return random.Random(base_seed), random.Random(base_seed + 1009 * (worker_id + 1))

    @staticmethod
    def _sample_index_from_key(key: str) -> int | None:
        try:
            return int(key.rsplit("-", 1)[-1])
        except ValueError:
            return None

    def _raw_samples_from_shard(self, path: Path):
        pending: dict[str, dict[str, bytes]] = {}
        with tarfile.open(path, "r:*") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                stem, suffix = os.path.splitext(os.path.basename(member.name))
                if suffix not in {".jpg", ".jpeg", ".cls"}:
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                bucket = pending.setdefault(stem, {})
                bucket[suffix.lstrip(".")] = extracted.read()
                image_bytes = bucket.get("jpg") or bucket.get("jpeg")
                label_bytes = bucket.get("cls")
                if image_bytes is None or label_bytes is None:
                    continue
                if self.max_images is not None:
                    sample_index = self._sample_index_from_key(stem)
                    if sample_index is not None and sample_index >= int(self.max_images):
                        pending.pop(stem, None)
                        continue
                pending.pop(stem, None)
                yield image_bytes, int(label_bytes.decode("utf-8").strip())

    def _decode_sample(self, sample: tuple[bytes, int]) -> tuple[torch.Tensor, int]:
        image_bytes, label = sample
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.transform(image), int(label)

    def _iter_raw_samples(self, paths: list[Path]):
        for path in paths:
            yield from self._raw_samples_from_shard(path)

    def _iter_shuffled_samples(self, paths: list[Path], sample_rng: random.Random):
        buffer: list[tuple[bytes, int]] = []
        for sample in self._iter_raw_samples(paths):
            buffer.append(sample)
            if len(buffer) < self.shuffle_buffer:
                continue
            index = sample_rng.randrange(len(buffer))
            yield buffer.pop(index)
        while buffer:
            index = sample_rng.randrange(len(buffer))
            yield buffer.pop(index)

    def __iter__(self):
        shard_rng, sample_rng = self._rngs_for_worker()
        paths = list(self.paths)
        if self.shuffle_shards:
            shard_rng.shuffle(paths)
        paths = self._paths_for_worker(paths)

        if self.shuffle_samples and self.shuffle_buffer > 1:
            raw_iter = self._iter_shuffled_samples(paths, sample_rng)
        else:
            raw_iter = self._iter_raw_samples(paths)
        for sample in raw_iter:
            yield self._decode_sample(sample)


class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        root: str,
        image_size: int,
        split: str,
        max_images: int | None,
        download: bool,
    ) -> None:
        self.split = split.lower()
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        split_dir = os.path.join(root, self.split)
        if os.path.isdir(split_dir):
            self.dataset = datasets.ImageFolder(root=split_dir)
            self.source = "local"
        elif download:
            from datasets import load_dataset as hf_load_dataset

            hf_split = "validation" if self.split == "val" else self.split
            self.dataset = hf_load_dataset("ILSVRC/imagenet-1k", split=hf_split, trust_remote_code=True)
            self.source = "huggingface"
        else:
            raise FileNotFoundError(
                f"ImageNet split directory not found: {split_dir}. "
                "Provide a valid --data-root with train/ and/or val/ subdirectories."
            )
        self.dataset_length = len(self.dataset)
        self.max_images = self.dataset_length if max_images is None else min(max_images, self.dataset_length)

    def __len__(self) -> int:
        return self.max_images

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        if self.source == "huggingface":
            item = self.dataset[idx]
            image = item["image"]
            label = int(item["label"])
        else:
            image, label = self.dataset[idx]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        return self.transform(image.convert("RGB")), int(label)


def make_image_loader(
    *,
    dataset: str,
    data_root: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    download: bool,
    max_images: int | None,
    train: bool,
    shuffle: bool,
    drop_last: bool,
    imagenet_split: str,
    val_imagenet_split: str,
    parquet_split: str,
    val_parquet_split: str,
    parquet_image_column: str | None,
    parquet_label_column: str | None,
    dataset_manifest: str | None = None,
    dataset_split: str | None = None,
    dataset_preflight: bool = False,
    ffhq_protocol: str = FFHQ_PROTOCOL,
    lsun_monitor_size: int = DEFAULT_LSUN_MONITOR_SIZE,
    lsun_monitor_seed: int = DEFAULT_LSUN_MONITOR_SEED,
    subset_seed: int = 0,
    webdataset_shuffle_buffer: int = 2048,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    if dataset == "cifar10":
        return make_cifar_loader(
            data_root=data_root,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            download=download,
            max_images=max_images,
            train=train,
            shuffle=shuffle,
            drop_last=drop_last,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )
    if dataset in {"ffhq", "lsun_bedroom"}:
        resolved_split = dataset_split or ("train" if train else "monitor")
        image_dataset = SharedImageDataset(
            dataset_id=dataset,
            root=data_root,
            image_size=image_size,
            split=resolved_split,
            manifest=dataset_manifest,
            max_images=max_images,
            subset_seed=subset_seed,
            preflight=dataset_preflight,
            ffhq_protocol=ffhq_protocol,
            lsun_monitor_size=lsun_monitor_size,
            lsun_monitor_seed=lsun_monitor_seed,
        )
    elif dataset == "imagenet1k_parquet":
        image_dataset = ImageNet1KParquetDataset(
            root=data_root,
            image_size=image_size,
            split=parquet_split if train else val_parquet_split,
            max_images=max_images,
            image_column=parquet_image_column,
            label_column=parquet_label_column,
        )
    elif dataset == "imagenet1k_webdataset":
        image_dataset = ImageNet1KWebDataset(
            root=data_root,
            image_size=image_size,
            split=parquet_split if train else val_parquet_split,
            max_images=max_images,
            shuffle_shards=train and shuffle,
            shuffle_samples=train and shuffle,
            shuffle_buffer=webdataset_shuffle_buffer,
            rank=rank if distributed else 0,
            world_size=world_size if distributed else 1,
        )
        loader_num_workers = num_workers if train else 0
        return DataLoader(
            image_dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=loader_num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    elif dataset == "imagenet":
        image_dataset = ImageNetDataset(
            root=data_root,
            image_size=image_size,
            split=imagenet_split if train else val_imagenet_split,
            max_images=max_images,
            download=download,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    sampler = (
        DistributedSampler(
            image_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        if distributed
        else None
    )
    return DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def repeat_loader(loader: DataLoader, sampler: DistributedSampler | None = None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        yielded = False
        for batch in loader:
            yielded = True
            yield batch
        if not yielded:
            raise ValueError("training loader produced no batches")
        epoch += 1


def state_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(state_to_cpu(item) for item in value)
    return value


def capture_rng_state(module: torch.nn.Module) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    try:
        device = next(module.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device=device)
    return state


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def save_snapshot(
    path: Path,
    *,
    step: int,
    student: torch.nn.Module,
    ema: torch.nn.Module,
    plan: dict,
    stats: list[dict],
    optimizer: torch.optim.Optimizer | None = None,
    update_manifest: bool = False,
) -> None:
    def state_dict_to_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu().clone()
            for key, value in module.state_dict().items()
        }

    profile_provenance = provenance_from_artifact(plan)
    payload = {
        "snapshot_format": "diffdist_edm_state_dict_v2",
        "step": int(step),
        "student_state_dict": state_dict_to_cpu(student),
        "ema_state_dict": state_dict_to_cpu(ema),
        "architecture_plan": plan,
        "benchmark_protocol_ids": plan.get("benchmark_protocol_ids", []),
        "benchmark_protocols": plan.get("benchmark_protocols", []),
        **profile_provenance,
        "stats": stats,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = state_to_cpu(optimizer.state_dict())
        payload["rng_state"] = capture_rng_state(student)
    atomic_torch_save(payload, path)
    if update_manifest:
        atomic_json_save(
            {
                "checkpoint_manifest_format": "diffdist_edm_latest_checkpoint_v1",
                "path": path.name,
                "step": int(step),
            },
            path.parent / LATEST_CHECKPOINT_MANIFEST,
        )


def save_training_state(path: Path, *, step: int, optimizer: torch.optim.Optimizer, student: torch.nn.Module) -> None:
    atomic_torch_save(
        {
            "training_state_format": "diffdist_edm_training_state_v1",
            "step": int(step),
            "optimizer_state_dict": state_to_cpu(optimizer.state_dict()),
            "rng_state": capture_rng_state(student),
        },
        path,
    )


def prune_periodic_snapshots(output_dir: Path, *, keep: int, training_active: bool) -> list[Path]:
    """Bound rolling periodic checkpoints without touching best/final snapshots.

    A zero retention count still keeps the newest periodic checkpoint while the
    run is active so ``--resume auto`` remains useful after a machine failure.
    Once a final snapshot has been written, zero removes every periodic copy.
    """

    if keep < 0:
        raise ValueError(f"periodic snapshot retention must be non-negative, got {keep}")
    snapshots = sorted(output_dir.glob("student-snapshot-step*.pt"))
    retained = max(keep, 1) if training_active else keep
    doomed = snapshots[:-retained] if retained else snapshots
    removed: list[Path] = []
    for path in doomed:
        if path.is_file():
            path.unlink()
            removed.append(path)
    return removed


def load_torch_payload(path: Path, *, kind: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{kind} does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{kind} must contain a dictionary payload: {path}")
    return payload


def is_model_snapshot(payload: dict[str, Any]) -> bool:
    return (
        isinstance(payload.get("step"), int)
        and isinstance(payload.get("student_state_dict"), dict)
        and isinstance(payload.get("ema_state_dict"), dict)
    )


def resolve_resume_snapshot(
    output_dir: Path,
    resume: str,
    *,
    expected_step: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    if resume != "auto":
        path = Path(resume).expanduser()
        payload = load_torch_payload(path, kind="resume snapshot")
        if not is_model_snapshot(payload):
            raise ValueError(f"Resume snapshot is missing valid model state or step: {path}")
        payload = dict(payload)
        payload["_resume_selection_reason"] = "explicit_path"
        return path, payload

    manifest_path = output_dir / LATEST_CHECKPOINT_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text())
        manifest_candidate = output_dir / str(manifest["path"])
        if manifest_candidate.parent == output_dir and manifest_candidate.is_file():
            payload = load_torch_payload(manifest_candidate, kind="manifest checkpoint")
            if is_model_snapshot(payload) and int(payload["step"]) == int(manifest["step"]):
                known_paths = [
                    output_dir / "student-final.pt",
                    output_dir / "student-best-fid.pt",
                    output_dir / "student-best-val.pt",
                    *output_dir.glob("student-snapshot-step*.pt"),
                ]
                manifest_mtime = manifest_path.stat().st_mtime_ns
                if not any(path.is_file() and path.stat().st_mtime_ns > manifest_mtime for path in known_paths):
                    payload = dict(payload)
                    payload["_resume_selection_reason"] = "latest_manifest"
                    return manifest_candidate, payload
    except Exception:
        pass

    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    paths = {
        output_dir / "student-final.pt",
        output_dir / "student-best-fid.pt",
        output_dir / "student-best-val.pt",
        *output_dir.glob("student-snapshot-step*.pt"),
    }
    for path in paths:
        if not path.is_file():
            continue
        try:
            payload = load_torch_payload(path, kind="resume candidate")
        except Exception:
            continue
        if is_model_snapshot(payload):
            candidates.append((int(payload["step"]), path, payload))

    if not candidates:
        raise FileNotFoundError(f"Could not find a valid model snapshot in {output_dir}")
    _, path, payload = max(candidates, key=lambda item: (item[0], item[1].name))
    payload = dict(payload)
    payload["_resume_selection_reason"] = "directory_scan"
    return path, payload


def restore_model_snapshot(
    payload: dict[str, Any],
    *,
    student: torch.nn.Module,
    ema: torch.nn.Module,
) -> int:
    if "student_state_dict" not in payload or "ema_state_dict" not in payload:
        raise ValueError("Resume snapshot must contain student_state_dict and ema_state_dict")
    if "step" not in payload:
        raise ValueError("Resume snapshot is missing its training step")
    student.load_state_dict(payload["student_state_dict"], strict=True)
    ema.load_state_dict(payload["ema_state_dict"], strict=True)
    return int(payload["step"])


def restore_training_state(
    path: Path,
    *,
    expected_step: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    restore_rng: bool,
    bundled_payload: dict[str, Any] | None = None,
) -> bool:
    payload = bundled_payload
    if payload is None or int(payload.get("step", -1)) != int(expected_step) or not isinstance(
        payload.get("optimizer_state_dict"), dict
    ):
        if not path.is_file():
            return False
        payload = load_torch_payload(path, kind="training state")
    if int(payload.get("step", -1)) != int(expected_step):
        return False
    optimizer_state = payload.get("optimizer_state_dict")
    if not isinstance(optimizer_state, dict):
        return False
    optimizer.load_state_dict(optimizer_state)
    for parameter_state in optimizer.state.values():
        for key, value in parameter_state.items():
            if isinstance(value, torch.Tensor):
                parameter_state[key] = value.to(device=device)

    rng_state = payload.get("rng_state")
    if restore_rng and isinstance(rng_state, dict):
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"])
        if device.type == "cuda" and "cuda" in rng_state:
            torch.cuda.set_rng_state(rng_state["cuda"], device=device)
    return True


def load_jsonl_rows(path: Path, *, required: bool = False) -> list[dict[str, Any]]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Required resume log does not exist: {path}")
        return []
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}; repair the log before resuming") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path} at line {line_number}")
            rows.append(row)
    return rows


def repair_resume_log(path: Path, *, resume_step: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not path.is_file():
        return [], {"original_rows": 0, "kept_rows": 0, "discarded_rows": 0, "discarded_malformed": 0}
    lines = path.read_text().splitlines()
    nonempty_indices = [index for index, line in enumerate(lines) if line.strip()]
    rows: list[dict[str, Any]] = []
    malformed_tail = 0
    for position, index in enumerate(nonempty_indices):
        try:
            row = json.loads(lines[index])
        except json.JSONDecodeError as exc:
            if position != len(nonempty_indices) - 1:
                raise ValueError(f"Invalid JSON in the middle of {path} at line {index + 1}") from exc
            malformed_tail = 1
            break
        if not isinstance(row, dict):
            raise ValueError(f"Expected a JSON object in {path} at line {index + 1}")
        rows.append(row)

    for row in rows:
        if "step" not in row:
            raise ValueError(f"Resume log row is missing 'step': {path}")
    kept = [row for row in rows if int(row["step"]) <= int(resume_step)]
    validate_resume_log(kept, path, resume_step=resume_step, require_last=False)
    discarded = len(rows) - len(kept)
    if discarded or malformed_tail:
        tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        try:
            tmp_path.write_text("".join(json.dumps(row) + "\n" for row in kept))
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    return kept, {
        "original_rows": len(rows),
        "kept_rows": len(kept),
        "discarded_rows": discarded,
        "discarded_malformed": malformed_tail,
    }


def validate_resume_log(rows: list[dict[str, Any]], path: Path, *, resume_step: int, require_last: bool) -> None:
    previous_step = -1
    for row in rows:
        if "step" not in row:
            raise ValueError(f"Resume log row is missing 'step': {path}")
        step = int(row["step"])
        if step <= previous_step:
            raise ValueError(f"Resume log steps must be strictly increasing in {path}: {previous_step} then {step}")
        if step > resume_step:
            raise ValueError(f"Resume log {path} contains step {step} beyond model step {resume_step}")
        previous_step = step
    if require_last and (not rows or previous_step != resume_step):
        actual = None if not rows else previous_step
        raise ValueError(f"Last training-stat step {actual} does not match model step {resume_step}")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def is_scalar_value(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def tensorboard_metric_direction(metric_name: str) -> str | None:
    name = metric_name.lower()
    lower_tokens = ("loss", "fid", "grad_norm", "seconds", "bad_checks")
    higher_tokens = ("examples_per_second", "count", "num_samples", "batch_size", "num_steps", "improved")
    if any(token in name for token in lower_tokens):
        return LOWER_IS_BETTER
    if any(token in name for token in higher_tokens):
        return HIGHER_IS_BETTER
    return None


def tensorboard_metric_path(namespace: str, metric_name: str) -> str | None:
    if metric_name == "step" or metric_name.endswith("_seed"):
        return None
    name = metric_name
    if namespace == "validation" and name.startswith("val_"):
        name = name[len("val_") :]
    if namespace == "fid" and name.startswith("fid_"):
        name = name[len("fid_") :]
    if namespace == "stopping" and name.startswith("early_stop_"):
        name = name[len("early_stop_") :]
    parts = name.split("_")
    if len(parts) >= 4 and parts[0] == "block" and parts[2].isdigit():
        name = f"block_{parts[1]}/{parts[2]}"
    if len(parts) >= 3 and parts[0] == "block" and parts[1].isdigit():
        name = f"block_{parts[1]}/{'_'.join(parts[2:])}"
    direction = tensorboard_metric_direction(name)
    if direction is None:
        return None
    return f"{namespace}/{name} {direction}"


class TensorBoardLogger:
    def __init__(self, writer: Any | None):
        self.writer = writer

    @property
    def enabled(self) -> bool:
        return self.writer is not None

    def add_text(self, tag: str, text: str, step: int = 0) -> None:
        if self.writer is not None:
            self.writer.add_text(tag, text, global_step=int(step))

    def log_row(self, namespace: str, row: dict[str, Any], *, step: int | None = None) -> None:
        if self.writer is None:
            return
        global_step = int(row.get("step", 0) if step is None else step)
        for key, value in row.items():
            if not is_scalar_value(value):
                continue
            tag = tensorboard_metric_path(namespace, key)
            if tag is None:
                continue
            self.writer.add_scalar(tag, float(value), global_step)

    def log_metric(self, tag: str, value: float, *, step: int, direction: str | None) -> None:
        if self.writer is None or not math.isfinite(float(value)):
            return
        tagged = f"{tag} {direction}" if direction else tag
        self.writer.add_scalar(tagged, float(value), int(step))

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def create_tensorboard_logger(
    *,
    output_dir: Path,
    tensorboard_dir: str | None,
    disabled: bool,
    purge_step: int | None = None,
) -> TensorBoardLogger:
    if disabled:
        return TensorBoardLogger(None)
    log_dir = Path(tensorboard_dir) if tensorboard_dir is not None else output_dir / "tensorboard"
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        missing = getattr(exc, "name", None)
        if missing == "pkg_resources":
            print("TensorBoard logging needs pkg_resources from setuptools; install the train extra (pip install -e '.[train]') to get setuptools<81, or pass --no-tensorboard.")
        else:
            print("TensorBoard package is not installed; skipping TensorBoard logging. Install the train extra (pip install -e '.[train]') or pass --no-tensorboard.")
        return TensorBoardLogger(None)
    log_dir.mkdir(parents=True, exist_ok=True)
    return TensorBoardLogger(SummaryWriter(log_dir=str(log_dir), purge_step=purge_step))


def log_tensorboard_metadata(
    logger: TensorBoardLogger,
    *,
    args: argparse.Namespace,
    plan: dict[str, Any],
    output_dir: Path,
    step: int = 0,
) -> None:
    if not logger.enabled:
        return
    args_payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    plan_payload = {
        "variant": plan.get("variant"),
        "network_pkl": plan.get("network_pkl"),
        "num_students": len(plan.get("students", [])),
        "timestep_blocks": plan.get("timestep_blocks"),
        "num_sigma_bins": plan.get("num_sigma_bins"),
        "model_family": plan.get("model_family", "vp"),
        "output_dir": str(output_dir),
    }
    logger.add_text("config/args", json.dumps(args_payload, indent=2, sort_keys=True), step=step)
    logger.add_text("config/architecture_plan", json.dumps(plan_payload, indent=2, sort_keys=True), step=step)


def make_seeded_generator(device: torch.device, seed: int) -> torch.Generator:
    if device.type == "cuda":
        return torch.Generator(device=device).manual_seed(int(seed))
    return torch.Generator().manual_seed(int(seed))


def denoising_loss_components(
    *,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor | None,
    sigmas: torch.Tensor,
    noise: torch.Tensor,
    model_family: str,
    sigma_data: float,
    kd_weight: float,
    data_weight: float,
) -> dict[str, torch.Tensor]:
    noisy = images + noise * sigmas.reshape(-1, 1, 1, 1)
    teacher_denoised = teacher(noisy, sigmas, labels)
    student_denoised = student(noisy, sigmas, labels)
    weights = loss_weights_for_family(sigmas=sigmas, model_family=model_family, sigma_data=sigma_data).float()
    kd = mse_per_example(student_denoised.float(), teacher_denoised.float()) * weights
    data = mse_per_example(student_denoised.float(), images.float()) * weights
    return {
        "loss": float(kd_weight) * kd + float(data_weight) * data,
        "kd_loss": kd,
        "data_loss": data,
    }


def infer_block_ids(student: torch.nn.Module, sigmas: torch.Tensor, *, num_blocks: int) -> torch.Tensor:
    if hasattr(student, "block_ids_for_sigma"):
        block_ids = student.block_ids_for_sigma(sigmas)  # type: ignore[attr-defined]
    else:
        block_ids = torch.zeros(sigmas.shape[0], device=sigmas.device, dtype=torch.long)
    return block_ids.to(device=sigmas.device, dtype=torch.long).clamp(0, max(0, int(num_blocks) - 1))


def restore_training_mode(module: torch.nn.Module, was_training: bool) -> None:
    if was_training:
        module.train()
    else:
        module.eval()


def evaluate_validation_loss(
    *,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    loader: DataLoader,
    sigma_values: torch.Tensor,
    label_dim: int,
    device: torch.device,
    microbatch: int,
    seed: int,
    step: int,
    plan: dict,
    model_family: str,
    sigma_data: float,
    kd_weight: float,
    data_weight: float,
) -> dict[str, Any]:
    if microbatch <= 0:
        raise ValueError(f"validation microbatch must be positive, got {microbatch}")

    student_was_training = student.training
    teacher_was_training = teacher.training
    student.eval()
    teacher.eval()

    generator = make_seeded_generator(device, seed)
    totals = {"loss": 0.0, "kd_loss": 0.0, "data_loss": 0.0}
    count = 0
    num_blocks = max(1, len(plan.get("timestep_blocks") or [[0, int(plan.get("num_sigma_bins", 1))]]))
    block_totals = {
        "loss": [0.0 for _ in range(num_blocks)],
        "kd_loss": [0.0 for _ in range(num_blocks)],
        "data_loss": [0.0 for _ in range(num_blocks)],
    }
    block_counts = [0 for _ in range(num_blocks)]

    try:
        with torch.inference_mode():
            for images_cpu, labels_cpu in loader:
                images = images_cpu.to(device=device, dtype=torch.float32)
                labels = make_class_labels(labels_cpu.to(device), label_dim=label_dim, device=device)
                sigma_indices = torch.randint(
                    0,
                    sigma_values.numel(),
                    (images.shape[0],),
                    device=device,
                    generator=generator,
                )
                sigmas = sigma_values[sigma_indices]
                noise = torch.randn(images.shape, device=device, dtype=torch.float32, generator=generator)

                for start in range(0, images.shape[0], microbatch):
                    end = min(start + microbatch, images.shape[0])
                    components = denoising_loss_components(
                        student=student,
                        teacher=teacher,
                        images=images[start:end],
                        labels=None if labels is None else labels[start:end],
                        sigmas=sigmas[start:end],
                        noise=noise[start:end],
                        model_family=model_family,
                        sigma_data=sigma_data,
                        kd_weight=kd_weight,
                        data_weight=data_weight,
                    )
                    batch_count = int(end - start)
                    count += batch_count
                    for key, values in components.items():
                        totals[key] += float(values.sum().detach().cpu().item())

                    block_ids = infer_block_ids(student, sigmas[start:end], num_blocks=num_blocks)
                    for block_index in range(num_blocks):
                        mask = block_ids == block_index
                        block_count = int(mask.sum().detach().cpu().item())
                        if block_count == 0:
                            continue
                        block_counts[block_index] += block_count
                        for key, values in components.items():
                            block_totals[key][block_index] += float(values[mask].sum().detach().cpu().item())
    finally:
        restore_training_mode(student, student_was_training)
        restore_training_mode(teacher, teacher_was_training)

    if count == 0:
        raise ValueError("validation loader produced no examples")

    row: dict[str, Any] = {
        "step": int(step),
        "val_loss": totals["loss"] / count,
        "val_kd_loss": totals["kd_loss"] / count,
        "val_data_loss": totals["data_loss"] / count,
        "val_count": int(count),
    }
    for block_index in range(num_blocks):
        block_count = block_counts[block_index]
        row[f"val_block_{block_index}_count"] = int(block_count)
        for key, prefix in (("loss", "loss"), ("kd_loss", "kd_loss"), ("data_loss", "data_loss")):
            row[f"val_block_{block_index}_{prefix}"] = (
                block_totals[key][block_index] / block_count if block_count else None
            )
    return row


@dataclass
class EarlyStopResult:
    improved: bool
    should_stop: bool
    bad_checks: int


@dataclass
class ValidationEarlyStopper:
    patience: int
    min_steps: int
    min_delta: float
    disabled: bool = False
    best_loss: float = float("inf")
    best_step: int | None = None
    bad_checks: int = 0

    def update(self, *, step: int, val_loss: float) -> EarlyStopResult:
        improved = float(val_loss) < self.best_loss - float(self.min_delta)
        if improved:
            self.best_loss = float(val_loss)
            self.best_step = int(step)
            self.bad_checks = 0
        else:
            self.bad_checks += 1

        should_stop = (
            not self.disabled
            and int(self.patience) > 0
            and int(step) >= int(self.min_steps)
            and self.bad_checks >= int(self.patience)
        )
        return EarlyStopResult(improved=improved, should_stop=should_stop, bad_checks=int(self.bad_checks))


@dataclass
class BestMetricTracker:
    metric_name: str
    snapshot_name: str
    best_value: float = float("inf")
    best_step: int | None = None
    best_row: dict[str, Any] | None = None
    snapshot_path: str | None = None

    def update(self, row: dict[str, Any], *, output_dir: Path) -> bool:
        value = row.get(self.metric_name)
        if value is None:
            return False
        if float(value) >= self.best_value:
            return False
        self.best_value = float(value)
        self.best_step = int(row["step"])
        self.best_row = dict(row)
        self.snapshot_path = str(output_dir / self.snapshot_name)
        return True

    def to_dict(self) -> dict[str, Any] | None:
        if self.best_step is None:
            return None
        return {
            "metric_name": self.metric_name,
            "best_value": self.best_value,
            "best_step": self.best_step,
            "snapshot_path": self.snapshot_path,
            "row": self.best_row,
        }


@dataclass(frozen=True)
class FidReferenceConfig:
    dataset: str
    data_root: str
    image_size: int
    num_workers: int
    download: bool
    imagenet_split: str
    val_imagenet_split: str
    parquet_split: str
    val_parquet_split: str
    parquet_image_column: str | None
    parquet_label_column: str | None
    dataset_manifest: str | None = None
    dataset_split: str | None = None
    dataset_preflight: bool = False
    ffhq_protocol: str = FFHQ_PROTOCOL
    lsun_monitor_size: int = DEFAULT_LSUN_MONITOR_SIZE
    lsun_monitor_seed: int = DEFAULT_LSUN_MONITOR_SEED


def should_run_interval(step: int, interval: int) -> bool:
    return int(interval) > 0 and int(step) > 0 and int(step) % int(interval) == 0


def fid_sample_dir(output_dir: Path, step: int, num_samples: int, seed: int) -> Path:
    return Path(output_dir) / "fid_samples" / f"step{int(step):06d}_n{int(num_samples)}_seed{int(seed)}"


def fid_sample_manifest_path(output_dir: Path, step: int, num_samples: int, seed: int, rank: int, world_size: int) -> Path:
    sample_dir = fid_sample_dir(output_dir, step, num_samples, seed)
    manifest_name = "sample_manifest.json" if int(world_size) == 1 else f"sample_manifest_rank{int(rank)}.json"
    return sample_dir / manifest_name


def wait_for_fid_sample_manifests(
    output_dir: Path,
    *,
    step: int,
    num_samples: int,
    seed: int,
    world_size: int,
    run_id: str | None,
    poll_seconds: float = 1.0,
    timeout_seconds: float | None = 7200.0,
) -> None:
    if int(world_size) <= 1:
        return
    all_seeds = list(range(int(seed), int(seed) + int(num_samples)))
    pending = set(range(int(world_size)))
    start_time = time.monotonic()
    error_path = eval_error_path(output_dir, step=step, run_id=run_id) if run_id is not None else None
    while pending:
        if error_path is not None and error_path.exists():
            raise RuntimeError(f"FID sample generation failed at step {int(step)}: {read_eval_error_message(error_path)}")
        for rank in list(pending):
            manifest_path = fid_sample_manifest_path(output_dir, step, num_samples, seed, rank, world_size)
            if not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text())
            except json.JSONDecodeError:
                continue
            if int(manifest.get("step", -1)) != int(step):
                continue
            if int(manifest.get("seed", -1)) != int(seed):
                continue
            if int(manifest.get("num_samples", -1)) != int(num_samples):
                continue
            if int(manifest.get("rank", -1)) != int(rank):
                continue
            if int(manifest.get("world_size", -1)) != int(world_size):
                continue
            if run_id is not None and manifest.get("run_id") != run_id:
                continue
            if len(manifest.get("images", [])) != len(all_seeds[rank::int(world_size)]):
                continue
            pending.remove(rank)
        if pending:
            if timeout_seconds is not None and time.monotonic() - start_time > float(timeout_seconds):
                missing = [
                    str(fid_sample_manifest_path(output_dir, step, num_samples, seed, rank, world_size))
                    for rank in sorted(pending)
                ]
                raise TimeoutError(f"Timed out waiting for FID sample manifests: {missing}")
            time.sleep(float(poll_seconds))


def generate_fid_samples(
    *,
    net: torch.nn.Module,
    output_dir: Path,
    step: int,
    seed: int,
    num_samples: int,
    batch_size: int,
    num_steps: int,
    device: torch.device,
    rank: int = 0,
    world_size: int = 1,
    run_id: str | None = None,
) -> Path:
    if num_samples <= 0:
        raise ValueError(f"FID num samples must be positive, got {num_samples}")
    if batch_size <= 0:
        raise ValueError(f"FID batch size must be positive, got {batch_size}")
    if world_size <= 0:
        raise ValueError(f"FID world size must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"FID rank must be in [0, {world_size}), got {rank}")

    sample_dir = fid_sample_dir(output_dir, step, num_samples, seed)
    sample_dir.mkdir(parents=True, exist_ok=True)
    all_seeds = list(range(int(seed), int(seed) + int(num_samples)))
    seeds = all_seeds[rank::world_size]
    saved: list[str] = []
    was_training = net.training
    net.eval()
    try:
        with torch.inference_mode():
            for start in tqdm(
                range(0, len(seeds), batch_size),
                desc=f"fid samples {step}",
                dynamic_ncols=True,
                leave=False,
                disable=(world_size > 1 and rank != 0),
            ):
                batch_seeds = seeds[start : start + batch_size]
                rnd = StackedRandomGenerator(device, batch_seeds)
                latents = rnd.randn(
                    [len(batch_seeds), int(net.img_channels), int(net.img_resolution), int(net.img_resolution)],
                    device=device,
                    dtype=torch.float32,
                )
                labels = make_sample_labels(batch_seeds, int(getattr(net, "label_dim", 0)), device)
                images = edm_sampler(net, latents, class_labels=labels, randn_like=rnd.randn_like, num_steps=num_steps)
                images = (images * 0.5 + 0.5).clamp(0, 1)
                for sample_seed, image in zip(batch_seeds, images):
                    path = sample_dir / f"seed{sample_seed:06d}.png"
                    save_image(image, path)
                    saved.append(str(path))
    finally:
        restore_training_mode(net, was_training)

    manifest = {
        "step": int(step),
        "seed": int(seed),
        "num_samples": int(num_samples),
        "batch_size": int(batch_size),
        "num_steps": int(num_steps),
        "rank": int(rank),
        "world_size": int(world_size),
        "images": saved,
    }
    if run_id is not None:
        manifest["run_id"] = str(run_id)
    manifest_path = fid_sample_manifest_path(output_dir, step, num_samples, seed, rank, world_size)
    tmp_path = manifest_path.with_name(f"{manifest_path.name}.tmp-{os.getpid()}")
    try:
        tmp_path.write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(tmp_path, manifest_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return sample_dir


def compute_clean_fid(sample_dir: Path, *, ref_split: str) -> float:
    return compute_clean_fid_for_dataset(
        sample_dir,
        dataset_name="cifar10",
        dataset_res=32,
        dataset_split=ref_split,
        mode="clean",
    )


def compute_clean_fid_for_dataset(
    sample_dir: Path,
    *,
    dataset_name: str,
    dataset_res: int,
    dataset_split: str,
    mode: str,
    ref_stats_cache_dir: Path | None = None,
    ref_config: FidReferenceConfig | None = None,
    feature_batch_size: int = 32,
    num_workers: int = 12,
    device: torch.device | None = None,
) -> float:
    try:
        from cleanfid import fid
    except ImportError as exc:
        raise ImportError("CleanFID evaluation requires `clean-fid`; install the eval extra (pip install -e '.[eval]') or pass --fid-every 0.") from exc

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_path = None
    if ref_stats_cache_dir is not None and ref_config is not None:
        cache_path = fid_reference_stats_path(
            cache_dir=ref_stats_cache_dir,
            ref_config=ref_config,
            dataset_name=dataset_name,
            dataset_res=dataset_res,
            dataset_split=dataset_split,
            mode=mode,
        )
        if cache_path.exists():
            return compute_fid_with_reference_stats(
                fid_module=fid,
                sample_dir=sample_dir,
                stats_path=cache_path,
                mode=mode,
                feature_batch_size=feature_batch_size,
                num_workers=num_workers,
                device=device,
            )

    try:
        return float(
            fid.compute_fid(
                str(sample_dir),
                dataset_name=dataset_name,
                dataset_res=dataset_res,
                dataset_split=dataset_split,
                mode=mode,
                batch_size=feature_batch_size,
                num_workers=num_workers,
                device=device,
            )
        )
    except (urllib.error.HTTPError, urllib.error.URLError):
        if cache_path is None or ref_config is None:
            raise

    assert cache_path is not None
    assert ref_config is not None
    with locked_fid_reference_stats(cache_path):
        if not cache_path.exists():
            compute_and_cache_reference_stats(
                fid_module=fid,
                stats_path=cache_path,
                ref_config=ref_config,
                dataset_name=dataset_name,
                dataset_res=dataset_res,
                dataset_split=dataset_split,
                mode=mode,
                feature_batch_size=feature_batch_size,
                device=device,
            )
    return float(
        compute_fid_with_reference_stats(
            fid_module=fid,
            sample_dir=sample_dir,
            stats_path=cache_path,
            mode=mode,
            feature_batch_size=feature_batch_size,
            num_workers=num_workers,
            device=device,
        )
    )


def _cache_component(value: Any) -> str:
    text = str(value).lower()
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text).strip("_") or "none"


def _fid_reference_metadata(
    *,
    ref_config: FidReferenceConfig,
    dataset_name: str,
    dataset_res: int,
    dataset_split: str,
    mode: str,
) -> dict[str, Any]:
    return {
        "dataset": ref_config.dataset,
        "data_root": str(Path(ref_config.data_root).expanduser()),
        "image_size": int(ref_config.image_size),
        "download": bool(ref_config.download),
        "imagenet_split": ref_config.imagenet_split,
        "val_imagenet_split": ref_config.val_imagenet_split,
        "parquet_split": ref_config.parquet_split,
        "val_parquet_split": ref_config.val_parquet_split,
        "parquet_image_column": ref_config.parquet_image_column,
        "parquet_label_column": ref_config.parquet_label_column,
        "dataset_manifest": ref_config.dataset_manifest,
        "dataset_split": ref_config.dataset_split,
        "dataset_preflight": ref_config.dataset_preflight,
        "ffhq_protocol": ref_config.ffhq_protocol,
        "lsun_monitor_size": ref_config.lsun_monitor_size,
        "lsun_monitor_seed": ref_config.lsun_monitor_seed,
        "fid_ref_dataset_name": dataset_name,
        "fid_ref_dataset_res": int(dataset_res),
        "fid_ref_split": dataset_split,
        "fid_mode": mode,
    }


def fid_reference_stats_path(
    *,
    cache_dir: Path,
    ref_config: FidReferenceConfig,
    dataset_name: str,
    dataset_res: int,
    dataset_split: str,
    mode: str,
) -> Path:
    metadata = _fid_reference_metadata(
        ref_config=ref_config,
        dataset_name=dataset_name,
        dataset_res=dataset_res,
        dataset_split=dataset_split,
        mode=mode,
    )
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    stem = "_".join(
        [
            _cache_component(dataset_name),
            _cache_component(mode),
            _cache_component(dataset_split),
            _cache_component(dataset_res),
            digest,
        ]
    )
    return Path(cache_dir) / f"{stem}.npz"


@contextmanager
def locked_fid_reference_stats(stats_path: Path):
    import fcntl

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = stats_path.with_suffix(stats_path.suffix + ".lock")
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _load_reference_stats(stats_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(stats_path) as stats:
        return np.asarray(stats["mu"]), np.asarray(stats["sigma"])


def _cleanfid_batch_from_normalized_images(
    images: torch.Tensor,
    *,
    fid_module: Any,
    mode: str,
) -> torch.Tensor:
    resizer = fid_module.build_resizer(mode)
    to_tensor = transforms.ToTensor()
    images_255 = ((images.detach().cpu().to(torch.float32) * 0.5 + 0.5).clamp(0, 1) * 255.0)
    resized: list[torch.Tensor] = []
    for image in images_255:
        image_np = image.permute(1, 2, 0).numpy().astype(np.float32, copy=False)
        image_resized = resizer(image_np)
        if np.asarray(image_resized).dtype == np.uint8:
            image_tensor = to_tensor(np.asarray(image_resized)) * 255.0
        else:
            image_tensor = to_tensor(np.asarray(image_resized, dtype=np.float32))
        resized.append(image_tensor)
    return torch.stack(resized, dim=0)


def compute_reference_stats_from_dataset(
    *,
    fid_module: Any,
    ref_config: FidReferenceConfig,
    mode: str,
    feature_batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, int]:
    loader = make_image_loader(
        dataset=ref_config.dataset,
        data_root=ref_config.data_root,
        image_size=ref_config.image_size,
        batch_size=feature_batch_size,
        num_workers=ref_config.num_workers,
        download=ref_config.download,
        max_images=None,
        train=False,
        shuffle=False,
        drop_last=False,
        imagenet_split=ref_config.imagenet_split,
        val_imagenet_split=ref_config.val_imagenet_split,
        parquet_split=ref_config.parquet_split,
        val_parquet_split=ref_config.val_parquet_split,
        parquet_image_column=ref_config.parquet_image_column,
        parquet_label_column=ref_config.parquet_label_column,
        dataset_manifest=ref_config.dataset_manifest,
        dataset_split="fid" if ref_config.dataset in {"ffhq", "lsun_bedroom"} else ref_config.dataset_split,
        dataset_preflight=ref_config.dataset_preflight,
        ffhq_protocol=ref_config.ffhq_protocol,
        lsun_monitor_size=ref_config.lsun_monitor_size,
        lsun_monitor_seed=ref_config.lsun_monitor_seed,
    )
    feature_model = fid_module.build_feature_extractor(mode, device)
    features: list[np.ndarray] = []
    for images, _labels in tqdm(loader, desc="fid reference stats", dynamic_ncols=True, leave=False):
        cleanfid_images = _cleanfid_batch_from_normalized_images(images, fid_module=fid_module, mode=mode)
        features.append(fid_module.get_batch_features(cleanfid_images, feature_model, device))
    if not features:
        raise ValueError("FID reference dataset produced no examples")
    np_features = np.concatenate(features, axis=0)
    mu = np.mean(np_features, axis=0)
    sigma = np.cov(np_features, rowvar=False)
    return mu, sigma, int(np_features.shape[0])


def compute_and_cache_reference_stats(
    *,
    fid_module: Any,
    stats_path: Path,
    ref_config: FidReferenceConfig,
    dataset_name: str,
    dataset_res: int,
    dataset_split: str,
    mode: str,
    feature_batch_size: int,
    device: torch.device,
) -> None:
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    mu, sigma, count = compute_reference_stats_from_dataset(
        fid_module=fid_module,
        ref_config=ref_config,
        mode=mode,
        feature_batch_size=feature_batch_size,
        device=device,
    )
    metadata = _fid_reference_metadata(
        ref_config=ref_config,
        dataset_name=dataset_name,
        dataset_res=dataset_res,
        dataset_split=dataset_split,
        mode=mode,
    )
    metadata["num_reference_images"] = int(count)
    tmp_path = stats_path.with_name(f"{stats_path.name}.tmp-{os.getpid()}")
    try:
        with tmp_path.open("wb") as tmp_file:
            np.savez_compressed(tmp_file, mu=mu, sigma=sigma, metadata=json.dumps(metadata, sort_keys=True))
        os.replace(tmp_path, stats_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def compute_fid_with_reference_stats(
    *,
    fid_module: Any,
    sample_dir: Path,
    stats_path: Path,
    mode: str,
    feature_batch_size: int,
    num_workers: int,
    device: torch.device,
) -> float:
    ref_mu, ref_sigma = _load_reference_stats(stats_path)
    feature_model = fid_module.build_feature_extractor(mode, device)
    features = fid_module.get_folder_features(
        str(sample_dir),
        feature_model,
        num_workers=num_workers,
        batch_size=feature_batch_size,
        device=device,
        mode=mode,
        description=f"FID {sample_dir.name} : ",
        verbose=True,
    )
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return float(fid_module.frechet_distance(mu, sigma, ref_mu, ref_sigma))


def run_fid_evaluation(
    *,
    net: torch.nn.Module,
    output_dir: Path,
    step: int,
    seed: int,
    num_samples: int,
    batch_size: int,
    num_steps: int,
    ref_split: str,
    ref_dataset_name: str,
    ref_dataset_res: int,
    fid_mode: str,
    device: torch.device,
    ref_stats_cache_dir: Path | None = None,
    ref_config: FidReferenceConfig | None = None,
    feature_batch_size: int = 32,
    num_workers: int = 12,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    run_id: str | None = None,
    discard_samples: bool = False,
) -> dict[str, Any]:
    sample_dir = generate_fid_samples(
        net=net,
        output_dir=output_dir,
        step=step,
        seed=seed,
        num_samples=num_samples,
        batch_size=batch_size,
        num_steps=num_steps,
        device=device,
        rank=rank,
        world_size=world_size,
        run_id=run_id,
    )
    if distributed and rank == 0:
        wait_for_fid_sample_manifests(
            output_dir,
            step=step,
            num_samples=num_samples,
            seed=seed,
            world_size=world_size,
            run_id=run_id,
        )
    if distributed and rank != 0:
        return {
            "step": int(step),
            "fid": None,
            "fid_num_samples": int(num_samples),
            "fid_batch_size": int(batch_size),
            "fid_num_steps": int(num_steps),
            "fid_ref_dataset_name": str(ref_dataset_name),
            "fid_ref_dataset_res": int(ref_dataset_res),
            "fid_ref_split": str(ref_split),
            "fid_mode": str(fid_mode),
            "fid_seed": int(seed),
            "sample_dir": str(sample_dir),
            "rank": int(rank),
            "world_size": int(world_size),
        }
    fid_value = compute_clean_fid_for_dataset(
        sample_dir,
        dataset_name=ref_dataset_name,
        dataset_res=ref_dataset_res,
        dataset_split=ref_split,
        mode=fid_mode,
        ref_stats_cache_dir=ref_stats_cache_dir,
        ref_config=ref_config,
        feature_batch_size=feature_batch_size,
        num_workers=num_workers,
        device=device,
    )
    row = {
        "step": int(step),
        "fid": float(fid_value),
        "fid_num_samples": int(num_samples),
        "fid_batch_size": int(batch_size),
        "fid_num_steps": int(num_steps),
        "fid_ref_dataset_name": str(ref_dataset_name),
        "fid_ref_dataset_res": int(ref_dataset_res),
        "fid_ref_split": str(ref_split),
        "fid_mode": str(fid_mode),
        "fid_seed": int(seed),
        "sample_dir": str(sample_dir),
        "rank": int(rank),
        "world_size": int(world_size),
    }
    if discard_samples:
        shutil.rmtree(sample_dir)
    return row


def build_checkpoint_selection(
    *,
    stop_reason: str,
    final_step: int,
    final_snapshot: Path,
    best_val: BestMetricTracker,
    best_fid: BestMetricTracker,
    selection_policy: str = "auto",
) -> dict[str, Any]:
    best_fid_payload = best_fid.to_dict()
    best_val_payload = best_val.to_dict()
    if selection_policy not in {"auto", "best_val"}:
        raise ValueError(f"Unsupported checkpoint selection policy: {selection_policy}")
    if selection_policy == "best_val" and best_val_payload is not None:
        selected_kind = "best_val"
        selected_snapshot = best_val.snapshot_path
        selected_step = best_val.best_step
    elif selection_policy == "auto" and best_fid_payload is not None:
        selected_kind = "best_fid"
        selected_snapshot = best_fid.snapshot_path
        selected_step = best_fid.best_step
    elif best_val_payload is not None:
        selected_kind = "best_val"
        selected_snapshot = best_val.snapshot_path
        selected_step = best_val.best_step
    else:
        selected_kind = "final"
        selected_snapshot = str(final_snapshot)
        selected_step = int(final_step)

    return {
        "selection_policy": selection_policy,
        "validation_split": {"is_heldout": False, "relationship": "overlapping_monitoring_subset"},
        "stop_reason": stop_reason,
        "final_step": int(final_step),
        "final_snapshot": str(final_snapshot),
        "best_validation": best_val_payload,
        "best_fid": best_fid_payload,
        "selected": {
            "kind": selected_kind,
            "step": selected_step,
            "snapshot_path": selected_snapshot,
        },
    }


def validate_args(args: argparse.Namespace) -> None:
    positive_names = [
        "steps",
        "batch_size",
        "microbatch",
        "num_workers",
        "val_batch_size",
        "val_microbatch",
        "early_stop_patience",
        "early_stop_min_steps",
        "snapshot_every",
    ]
    for name in positive_names:
        value = getattr(args, name)
        if name in {"num_workers", "snapshot_every"}:
            if value < 0:
                raise ValueError(f"--{name.replace('_', '-')} must be non-negative, got {value}")
        elif value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive, got {value}")
    if args.val_every < 0:
        raise ValueError(f"--val-every must be non-negative, got {args.val_every}")
    if args.fid_every < 0:
        raise ValueError(f"--fid-every must be non-negative, got {args.fid_every}")
    if args.ddp_timeout_minutes <= 0:
        raise ValueError(f"--ddp-timeout-minutes must be positive, got {args.ddp_timeout_minutes}")
    if args.fid_every > 0:
        for name in ("fid_num_samples", "fid_batch_size", "fid_num_steps", "fid_feature_batch_size"):
            value = getattr(args, name)
            if value <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive when FID is enabled, got {value}")
    if args.val_max_images is not None and args.val_max_images <= 0:
        raise ValueError(f"--val-max-images must be positive when set, got {args.val_max_images}")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError(f"--max-images must be positive when set, got {args.max_images}")
    if args.webdataset_shuffle_buffer < 0:
        raise ValueError(f"--webdataset-shuffle-buffer must be non-negative, got {args.webdataset_shuffle_buffer}")
    if args.early_stop_min_delta < 0:
        raise ValueError(f"--early-stop-min-delta must be non-negative, got {args.early_stop_min_delta}")
    if args.keep_last_snapshots is not None and args.keep_last_snapshots < 0:
        raise ValueError(
            f"--keep-last-snapshots must be non-negative when set, got {args.keep_last_snapshots}"
        )
    if args.lsun_monitor_size <= 0:
        raise ValueError(f"--lsun-monitor-size must be positive, got {args.lsun_monitor_size}")
    if args.lsun_monitor_seed < 0:
        raise ValueError(f"--lsun-monitor-seed must be non-negative, got {args.lsun_monitor_seed}")


def build_arg_parser() -> argparse.ArgumentParser:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--architecture-plan",
        required=True,
        help="Path to architecture_plan.json produced by prepare_edm_distillation.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for stats and snapshots. Uses the architecture plan directory when omitted.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        metavar="SNAPSHOT",
        help=(
            "Resume and append to the existing output directory. With no SNAPSHOT, load the newest valid "
            "checkpoint and repair logs to its step. --steps remains the total step target."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=["cifar10", "imagenet", "imagenet1k_parquet", "imagenet1k_webdataset", "ffhq", "lsun_bedroom"],
        default="cifar10",
        help="Image dataset used for distillation.",
    )
    parser.add_argument("--data-root", default="./data", help="Root directory for the selected dataset.")
    parser.add_argument("--dataset-manifest", default=None, help="Versioned JSON/text manifest for FFHQ/LSUN.")
    parser.add_argument(
        "--dataset-split",
        choices=["all", "train", "monitor", "validation", "val", "fid"],
        default=None,
        help="Override the named FFHQ/LSUN split; defaults to train for training and monitor for validation.",
    )
    parser.add_argument("--dataset-preflight", action="store_true", help="Fully validate/decode the selected dataset before use.")
    parser.add_argument("--ffhq-protocol", default=FFHQ_PROTOCOL)
    parser.add_argument("--lsun-monitor-size", type=int, default=DEFAULT_LSUN_MONITOR_SIZE)
    parser.add_argument("--lsun-monitor-seed", type=int, default=DEFAULT_LSUN_MONITOR_SEED)
    parser.add_argument("--download", action="store_true", help="Download supported datasets if missing.")
    parser.add_argument("--imagenet-split", default="train", help="ImageNet directory/HuggingFace split for training.")
    parser.add_argument("--val-imagenet-split", default="val", help="ImageNet directory/HuggingFace split for validation.")
    parser.add_argument("--parquet-split", default="train", help="ImageNet parquet/WebDataset split for training.")
    parser.add_argument("--val-parquet-split", default="validation", help="ImageNet parquet/WebDataset split for validation.")
    parser.add_argument("--parquet-image-column", default=None, help="Image column for ImageNet parquet data.")
    parser.add_argument("--parquet-label-column", default=None, help="Label column for ImageNet parquet data.")
    parser.add_argument(
        "--webdataset-shuffle-buffer",
        type=int,
        default=2048,
        help="Bounded sample shuffle buffer for ImageNet WebDataset training.",
    )
    parser.add_argument("--device", default=default_device, help="Torch device used for teacher/student training.")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32", help="Teacher inference dtype.")
    parser.add_argument(
        "--model-cache-dir",
        default=None,
        help="Required cache directory when the architecture plan references a remote teacher URL.",
    )
    parser.add_argument(
        "--trust-local-pickle",
        action="store_true",
        help="Allow a local/third-party NVLabs teacher pickle after verifying its origin.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Total optimizer-step target; resumed runs continue from their saved global step.",
    )
    parser.add_argument("--batch-size", type=int, default=10000, help="Total training batch size per optimizer step.")
    parser.add_argument(
        "--microbatch",
        type=int,
        default=1000,
        help=(
            "Sub-batch size used for gradient accumulation inside each batch. "
            "Lower this to reduce memory without changing --batch-size."
        ),
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Adam learning rate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for training, sigma sampling, and data order.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader worker processes.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap on training images for quick debug runs.")
    parser.add_argument("--kd-weight", type=float, default=1.0, help="Weight for matching the teacher denoised output.")
    parser.add_argument("--data-weight", type=float, default=0.25, help="Weight for the clean-image denoising reconstruction term.")
    parser.add_argument("--ema-beta", type=float, default=0.999, help="Per-step EMA decay for the saved evaluation weights.")
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=100,
        help="Save an intermediate snapshot every N steps. Set to 0 to save only the final snapshot.",
    )
    parser.add_argument(
        "--keep-last-snapshots",
        type=int,
        default=None,
        help=(
            "Bound periodic student-snapshot-step*.pt retention. The legacy default keeps all; "
            "0 keeps one resumable snapshot while training and removes periodic copies after final save."
        ),
    )
    parser.add_argument(
        "--checkpoint-selection",
        choices=("auto", "best_val"),
        default="auto",
        help="Choose the published checkpoint by legacy auto policy or deterministic monitoring loss.",
    )
    parser.add_argument("--val-every", type=int, default=5000, help="Evaluate EMA validation loss every N steps. Set to 0 to skip validation and early stopping.")
    parser.add_argument("--val-max-images", type=int, default=2048, help="Maximum validation images used for each validation pass.")
    parser.add_argument("--val-batch-size", type=int, default=256, help="Validation batch size before microbatch splitting.")
    parser.add_argument("--val-microbatch", type=int, default=64, help="Validation sub-batch size used to control memory.")
    parser.add_argument("--val-seed", type=int, default=12345, help="Seed for deterministic validation sigma and noise sampling.")
    parser.add_argument("--early-stop-patience", type=int, default=4, help="Stop after this many validation checks without sufficient improvement.")
    parser.add_argument("--early-stop-min-steps", type=int, default=10000, help="Do not early-stop before this optimizer step.")
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4, help="Required validation-loss decrease to count as an improvement.")
    parser.add_argument("--disable-early-stop", action="store_true", help="Keep validation logging/checkpointing but never stop early.")
    parser.add_argument("--fid-every", type=int, default=10000, help="Run CleanFID monitoring every N steps. Set to 0 to disable.")
    parser.add_argument("--fid-num-samples", type=int, default=5000, help="Number of generated samples for each CleanFID monitoring pass.")
    parser.add_argument("--fid-batch-size", type=int, default=64, help="Sampling batch size for CleanFID image generation.")
    parser.add_argument("--fid-num-steps", type=int, default=18, help="EDM sampler steps for CleanFID monitoring samples.")
    parser.add_argument(
        "--fid-ref-stats-cache-dir",
        default=None,
        help="Local directory for cached FID reference statistics. Defaults to <output-dir>/fid_reference_stats.",
    )
    parser.add_argument("--fid-feature-batch-size", type=int, default=32, help="Batch size for CleanFID feature extraction.")
    parser.add_argument("--fid-ref-split", default="train", help="Reference split passed to CleanFID.")
    parser.add_argument("--fid-ref-dataset-name", default="cifar10", help="Reference dataset name passed to CleanFID.")
    parser.add_argument("--fid-ref-dataset-res", type=int, default=32, help="Reference dataset resolution passed to CleanFID.")
    parser.add_argument("--fid-mode", default="clean", help="CleanFID mode.")
    parser.add_argument("--fid-seed", type=int, default=0, help="First sample seed used for deterministic CleanFID monitoring samples.")
    parser.add_argument(
        "--discard-fid-samples",
        action="store_true",
        help="Delete generated FID sample images after a successful CleanFID evaluation.",
    )
    parser.add_argument("--tensorboard-dir", default=None, help="TensorBoard log directory. Defaults to <output-dir>/tensorboard.")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard event logging.")
    parser.add_argument(
        "--ddp-timeout-minutes",
        type=float,
        default=180.0,
        help="Distributed process-group timeout. Increase when rank-0-only validation or FID can take a long time.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)

    plan_path = Path(args.architecture_plan)
    plan = load_plan(plan_path)
    teacher_spec = teacher_spec_from_plan(plan)
    if urllib.parse.urlparse(teacher_spec.source).scheme in {"http", "https"} and args.model_cache_dir is None:
        parser.error("A remote teacher in the architecture plan requires --model-cache-dir")
    image_size = int(plan["students"][0]["model_kwargs"]["img_resolution"])
    preloaded_train_dataset: SharedImageDataset | None = None
    if args.dataset in {"ffhq", "lsun_bedroom"}:
        try:
            preloaded_train_dataset = SharedImageDataset(
                dataset_id=args.dataset,
                root=args.data_root,
                image_size=image_size,
                split=args.dataset_split or "train",
                manifest=args.dataset_manifest,
                max_images=args.max_images,
                subset_seed=args.seed,
                preflight=args.dataset_preflight,
                ffhq_protocol=args.ffhq_protocol,
                lsun_monitor_size=args.lsun_monitor_size,
                lsun_monitor_seed=args.lsun_monitor_seed,
            )
            validate_dataset_readability(preloaded_train_dataset)
        except (FileNotFoundError, PermissionError, ValueError) as exc:
            parser.error(f"Dataset validation failed before GPU/teacher initialization: {exc}")

    dist_ctx = init_distributed(args.device, timeout_minutes=args.ddp_timeout_minutes)
    device = dist_ctx.device
    local_batch_size = require_divisible_global_batch("batch_size", args.batch_size, dist_ctx.world_size)
    torch.manual_seed(args.seed + dist_ctx.rank)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    output_dir = Path(args.output_dir) if args.output_dir is not None else plan_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "distillation_stats.jsonl"
    validation_stats_path = output_dir / "validation_stats.jsonl"
    fid_stats_path = output_dir / "fid_stats.jsonl"

    resume_path: Path | None = None
    resume_payload: dict[str, Any] | None = None
    resume_step = 0
    stats: list[dict[str, Any]] = []
    validation_stats: list[dict[str, Any]] = []
    fid_stats: list[dict[str, Any]] = []
    repaired_logs: dict[str, dict[str, int]] = {}
    if args.resume is not None:
        resume_path, resume_payload = resolve_resume_snapshot(
            output_dir,
            str(args.resume),
        )
        resume_step = int(resume_payload.get("step", -1))
        stats, repaired_logs[stats_path.name] = repair_resume_log(stats_path, resume_step=resume_step)
        validation_stats, repaired_logs[validation_stats_path.name] = repair_resume_log(
            validation_stats_path, resume_step=resume_step
        )
        fid_stats, repaired_logs[fid_stats_path.name] = repair_resume_log(fid_stats_path, resume_step=resume_step)
        if args.steps <= resume_step:
            raise ValueError(
                f"--steps is the total target and must be greater than resume step {resume_step}; got {args.steps}"
            )
    elif stats_path.is_file() and stats_path.stat().st_size > 0:
        raise ValueError(f"Training stats already exist in {output_dir}; pass --resume or choose a new --output-dir")

    dist_sync_run_id = make_dist_sync_run_id(dist_ctx, device)
    tensorboard = create_tensorboard_logger(
        output_dir=output_dir,
        tensorboard_dir=args.tensorboard_dir,
        disabled=args.no_tensorboard or not dist_ctx.is_main,
        purge_step=resume_step + 1 if resume_payload is not None else None,
    )

    student = construct_student_from_plan(plan).to(device=device)
    teacher = load_edm_network(
        teacher_spec,
        device=device,
        dtype=dtype,
        cache_dir=args.model_cache_dir,
        trust_local_pickle=args.trust_local_pickle,
    )
    student.train()
    ema = create_ema(student).to(device=device)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    optimizer_restored = False
    if resume_payload is not None:
        loaded_step = restore_model_snapshot(resume_payload, student=student, ema=ema)
        if loaded_step != resume_step:
            raise ValueError(f"Resume snapshot step changed while loading: expected {resume_step}, got {loaded_step}")
        optimizer_restored = restore_training_state(
            output_dir / TRAINING_STATE_FILENAME,
            expected_step=resume_step,
            optimizer=optimizer,
            device=device,
            restore_rng=not dist_ctx.enabled,
            bundled_payload=resume_payload,
        )

    train_student: torch.nn.Module = student
    if dist_ctx.enabled:
        train_student = DistributedDataParallel(
            student,
            device_ids=[dist_ctx.local_rank] if device.type == "cuda" else None,
            output_device=dist_ctx.local_rank if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
    if dist_ctx.is_main:
        log_tensorboard_metadata(
            tensorboard,
            args=args,
            plan=plan,
            output_dir=output_dir,
            step=resume_step,
        )
        if resume_path is not None:
            resume_metadata = {
                "checkpoint": str(resume_path),
                "resume_step": resume_step,
                "target_step": int(args.steps),
                "selection_reason": resume_payload.get("_resume_selection_reason", "unknown"),
                "repaired_logs": repaired_logs,
                "optimizer_restored": optimizer_restored,
                "tensorboard_dir": str(Path(args.tensorboard_dir) if args.tensorboard_dir else output_dir / "tensorboard"),
            }
            tensorboard.add_text("resume/state", json.dumps(resume_metadata, indent=2, sort_keys=True), step=resume_step)
            tensorboard.log_metric("resume/start_step", resume_step, step=resume_step, direction=HIGHER_IS_BETTER)
            print(json.dumps({"resume": resume_metadata}, indent=2))
            if not optimizer_restored:
                print(
                    f"Warning: {TRAINING_STATE_FILENAME} is missing or does not match step {resume_step}; "
                    "continuing with a fresh Adam optimizer."
                )

    sigma_values = torch.as_tensor(plan["sigma_values"], device=device, dtype=torch.float32)
    if sigma_values.numel() == 0:
        raise ValueError("architecture plan must contain sigma_values")

    label_dim = int(plan["students"][0]["model_kwargs"]["label_dim"])
    fid_ref_stats_cache_dir = (
        Path(args.fid_ref_stats_cache_dir)
        if args.fid_ref_stats_cache_dir is not None
        else output_dir / "fid_reference_stats"
    )
    fid_ref_config = FidReferenceConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        image_size=image_size,
        num_workers=args.num_workers,
        download=args.download,
        imagenet_split=args.imagenet_split,
        val_imagenet_split=args.val_imagenet_split,
        parquet_split=args.parquet_split,
        val_parquet_split=args.val_parquet_split,
        parquet_image_column=args.parquet_image_column,
        parquet_label_column=args.parquet_label_column,
        dataset_manifest=args.dataset_manifest,
        dataset_split="fid" if args.dataset in {"ffhq", "lsun_bedroom"} else args.dataset_split,
        dataset_preflight=args.dataset_preflight,
        ffhq_protocol=args.ffhq_protocol,
        lsun_monitor_size=args.lsun_monitor_size,
        lsun_monitor_seed=args.lsun_monitor_seed,
    )
    if preloaded_train_dataset is not None:
        train_sampler_override = (
            DistributedSampler(
                preloaded_train_dataset,
                num_replicas=dist_ctx.world_size,
                rank=dist_ctx.rank,
                shuffle=True,
                drop_last=True,
            )
            if dist_ctx.enabled
            else None
        )
        loader = DataLoader(
            preloaded_train_dataset,
            batch_size=local_batch_size,
            shuffle=train_sampler_override is None,
            sampler=train_sampler_override,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        loader = make_image_loader(
            dataset=args.dataset,
            data_root=args.data_root,
            image_size=image_size,
            batch_size=local_batch_size,
            num_workers=args.num_workers,
            download=args.download,
            max_images=args.max_images,
            train=True,
            shuffle=True,
            drop_last=True,
            imagenet_split=args.imagenet_split,
            val_imagenet_split=args.val_imagenet_split,
            parquet_split=args.parquet_split,
            val_parquet_split=args.val_parquet_split,
            parquet_image_column=args.parquet_image_column,
            parquet_label_column=args.parquet_label_column,
            webdataset_shuffle_buffer=args.webdataset_shuffle_buffer,
            distributed=dist_ctx.enabled,
            rank=dist_ctx.rank,
            world_size=dist_ctx.world_size,
        )
    val_loader = None
    if args.val_every > 0 and dist_ctx.is_main:
        val_loader = make_image_loader(
            dataset=args.dataset,
            data_root=args.data_root,
            image_size=image_size,
            batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            download=args.download,
            max_images=args.val_max_images,
            train=False,
            shuffle=False,
            drop_last=False,
            imagenet_split=args.imagenet_split,
            val_imagenet_split=args.val_imagenet_split,
            parquet_split=args.parquet_split,
            val_parquet_split=args.val_parquet_split,
            parquet_image_column=args.parquet_image_column,
            parquet_label_column=args.parquet_label_column,
            dataset_manifest=args.dataset_manifest,
            dataset_split="monitor" if args.dataset in {"ffhq", "lsun_bedroom"} else args.dataset_split,
            dataset_preflight=False,
            ffhq_protocol=args.ffhq_protocol,
            lsun_monitor_size=args.lsun_monitor_size,
            lsun_monitor_seed=args.lsun_monitor_seed,
            subset_seed=args.val_seed,
            webdataset_shuffle_buffer=args.webdataset_shuffle_buffer,
        )
    if dist_ctx.is_main and isinstance(loader.dataset, SharedImageDataset):
        dataset_metadata = {
            "train": loader.dataset.metadata,
            "validation": val_loader.dataset.metadata if val_loader is not None else None,
            "validation_is_heldout": False,
        }
        atomic_json_save(dataset_metadata, output_dir / "dataset_metadata.json")
    train_sampler = loader.sampler if isinstance(loader.sampler, DistributedSampler) else None
    batches = repeat_loader(loader, sampler=train_sampler)
    best_val = BestMetricTracker(metric_name="val_loss", snapshot_name="student-best-val.pt")
    best_fid = BestMetricTracker(metric_name="fid", snapshot_name="student-best-fid.pt")
    early_stopper = ValidationEarlyStopper(
        patience=args.early_stop_patience,
        min_steps=args.early_stop_min_steps,
        min_delta=args.early_stop_min_delta,
        disabled=args.disable_early_stop or args.val_every == 0,
    )
    if resume_payload is not None and dist_ctx.is_main:
        for row in validation_stats:
            best_val.update(row, output_dir=output_dir)
            early_stopper.update(step=int(row["step"]), val_loss=float(row["val_loss"]))
        for row in fid_stats:
            best_fid.update(row, output_dir=output_dir)

    stop_reason = "max_steps"
    last_step = resume_step
    last_validation_step: int | None = max(
        (int(row["step"]) for row in validation_stats),
        default=None,
    )

    progress = tqdm(
        range(resume_step + 1, args.steps + 1),
        initial=resume_step,
        total=args.steps,
        desc=f"distill {plan['variant']}",
        dynamic_ncols=True,
        disable=not dist_ctx.is_main,
    )
    interrupt = GracefulInterrupt()
    interrupt.install()
    try:
        for step in progress:
            step_start_time = time.perf_counter()
            images_cpu, labels_cpu = next(batches)
            images = images_cpu.to(device=device, dtype=torch.float32)
            labels = make_class_labels(labels_cpu.to(device), label_dim=label_dim, device=device)
            sigma_indices = torch.randint(0, sigma_values.numel(), (images.shape[0],), device=device)
            sigmas = sigma_values[sigma_indices]
            noise = torch.randn_like(images)

            optimizer.zero_grad(set_to_none=True)
            step_metrics: dict[str, float] = {"loss": 0.0, "kd_loss": 0.0, "data_loss": 0.0}
            for start in range(0, images.shape[0], args.microbatch):
                end = min(start + args.microbatch, images.shape[0])
                scale = (end - start) / images.shape[0]
                loss, metrics = hybrid_distillation_loss(
                    student=train_student,
                    teacher=teacher,
                    images=images[start:end],
                    labels=None if labels is None else labels[start:end],
                    sigmas=sigmas[start:end],
                    noise=noise[start:end],
                    model_family=plan.get("model_family", "vp"),
                    sigma_data=float(getattr(teacher, "sigma_data", 0.5)),
                    kd_weight=args.kd_weight,
                    data_weight=args.data_weight,
                )
                (loss * scale).backward()
                for key in step_metrics:
                    step_metrics[key] += metrics[key] * scale

            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            update_ema(ema, student, beta=args.ema_beta)
            step_seconds = time.perf_counter() - step_start_time
            reduced_metrics = average_scalar_dict(step_metrics, dist_ctx, device)
            reduced_perf = average_scalar_dict(
                {
                    "grad_norm": float(grad_norm.detach().cpu().item()),
                    "step_seconds": float(step_seconds),
                },
                dist_ctx,
                device,
            )
            examples_per_second = args.batch_size / reduced_perf["step_seconds"] if reduced_perf["step_seconds"] > 0 else 0.0

            row = {
                "step": step,
                **reduced_metrics,
                "grad_norm": reduced_perf["grad_norm"],
                "step_seconds": reduced_perf["step_seconds"],
                "examples_per_second": float(examples_per_second),
            }
            if dist_ctx.is_main:
                stats.append(row)
                append_jsonl(stats_path, row)
                tensorboard.log_row("train", row)
                tensorboard.log_metric("optimizer/learning_rate", optimizer.param_groups[0]["lr"], step=step, direction=None)
                tensorboard.log_metric("optimizer/ema_beta", args.ema_beta, step=step, direction=None)
                progress.set_postfix({key: f"{value:.4g}" for key, value in reduced_metrics.items()})
            last_step = step

            if sync_stop_requested(interrupt.requested, dist_ctx, device):
                stop_reason = "interrupted"
                if dist_ctx.is_main:
                    print(json.dumps({"stop_reason": stop_reason, "last_step": last_step}, indent=2))
                break

            snapshot_due = args.snapshot_every > 0 and step % args.snapshot_every == 0
            if dist_ctx.is_main and snapshot_due:
                try:
                    save_snapshot(
                        output_dir / f"student-snapshot-step{step:06d}.pt",
                        step=step,
                        student=student,
                        ema=ema,
                        plan=plan,
                        stats=stats,
                        optimizer=optimizer,
                        update_manifest=True,
                    )
                    save_training_state(
                        output_dir / TRAINING_STATE_FILENAME,
                        step=step,
                        optimizer=optimizer,
                        student=student,
                    )
                    if args.keep_last_snapshots is not None:
                        prune_periodic_snapshots(
                            output_dir,
                            keep=args.keep_last_snapshots,
                            training_active=True,
                        )
                except Exception as exc:
                    if dist_ctx.enabled:
                        mark_eval_error(output_dir, step=step, run_id=dist_sync_run_id, error=exc)
                    raise

            stop_after_eval = False
            val_due = should_run_interval(step, args.val_every)
            fid_due = should_run_interval(step, args.fid_every)
            if val_due and val_loader is not None:
                try:
                    val_row = evaluate_validation_loss(
                        student=ema,
                        teacher=teacher,
                        loader=val_loader,
                        sigma_values=sigma_values,
                        label_dim=label_dim,
                        device=device,
                        microbatch=args.val_microbatch,
                        seed=args.val_seed,
                        step=step,
                        plan=plan,
                        model_family=plan.get("model_family", "vp"),
                        sigma_data=float(getattr(teacher, "sigma_data", 0.5)),
                        kd_weight=args.kd_weight,
                        data_weight=args.data_weight,
                    )
                    last_validation_step = step
                    validation_stats.append(val_row)
                    append_jsonl(validation_stats_path, val_row)
                    tensorboard.log_row("validation", val_row)
                    val_improved = best_val.update(val_row, output_dir=output_dir)
                    if val_improved:
                        save_snapshot(
                            output_dir / best_val.snapshot_name,
                            step=step,
                            student=student,
                            ema=ema,
                            plan=plan,
                            stats=stats,
                            optimizer=optimizer,
                            update_manifest=True,
                        )
                    early_result = early_stopper.update(step=step, val_loss=float(val_row["val_loss"]))
                    tensorboard.log_row(
                        "stopping",
                        {
                            "step": step,
                            "early_stop_bad_checks": early_result.bad_checks,
                            "early_stop_improved": int(early_result.improved),
                            "best_val_loss": early_stopper.best_loss,
                        },
                    )
                    tensorboard.log_metric(
                        "checkpoint/best_val_updated",
                        float(int(val_improved)),
                        step=step,
                        direction=HIGHER_IS_BETTER,
                    )
                    progress.set_postfix(
                        {
                            "loss": f"{step_metrics['loss']:.4g}",
                            "val": f"{val_row['val_loss']:.4g}",
                            "bad": early_result.bad_checks,
                        }
                    )
                    if early_result.should_stop:
                        stop_reason = "early_stop_val_loss"
                        stop_after_eval = True
                except Exception as exc:
                    if dist_ctx.enabled and dist_ctx.is_main:
                        mark_eval_error(output_dir, step=step, run_id=dist_sync_run_id, error=exc)
                    raise

            if fid_due:
                try:
                    fid_row = run_fid_evaluation(
                        net=ema,
                        output_dir=output_dir,
                        step=step,
                        seed=args.fid_seed,
                        num_samples=args.fid_num_samples,
                        batch_size=args.fid_batch_size,
                        num_steps=args.fid_num_steps,
                        ref_split=args.fid_ref_split,
                        ref_dataset_name=args.fid_ref_dataset_name,
                        ref_dataset_res=args.fid_ref_dataset_res,
                        fid_mode=args.fid_mode,
                        device=device,
                        ref_stats_cache_dir=fid_ref_stats_cache_dir,
                        ref_config=fid_ref_config,
                        feature_batch_size=args.fid_feature_batch_size,
                        num_workers=args.num_workers,
                        distributed=dist_ctx.enabled,
                        rank=dist_ctx.rank,
                        world_size=dist_ctx.world_size,
                        run_id=dist_sync_run_id if dist_ctx.enabled else None,
                        discard_samples=args.discard_fid_samples,
                    )
                    if dist_ctx.is_main:
                        fid_stats.append(fid_row)
                        append_jsonl(fid_stats_path, fid_row)
                        tensorboard.log_row("fid", fid_row)
                        fid_improved = best_fid.update(fid_row, output_dir=output_dir)
                        if fid_improved:
                            save_snapshot(
                                output_dir / best_fid.snapshot_name,
                                step=step,
                                student=student,
                                ema=ema,
                                plan=plan,
                                stats=stats,
                                optimizer=optimizer,
                                update_manifest=True,
                            )
                        tensorboard.log_metric(
                            "checkpoint/best_fid_updated",
                            float(int(fid_improved)),
                            step=step,
                            direction=HIGHER_IS_BETTER,
                        )
                except Exception as exc:
                    if dist_ctx.enabled:
                        mark_eval_error(output_dir, step=step, run_id=dist_sync_run_id, error=exc)
                    raise

            if snapshot_due or val_due or fid_due:
                stop_after_eval = sync_eval_completion(
                    stop_after_eval=stop_after_eval,
                    ctx=dist_ctx,
                    output_dir=output_dir,
                    step=step,
                    run_id=dist_sync_run_id,
                )
                if sync_stop_requested(interrupt.requested, dist_ctx, device):
                    stop_reason = "interrupted"
                    stop_after_eval = True
            if stop_after_eval:
                break
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        if dist_ctx.is_main:
            print(json.dumps({"stop_reason": stop_reason, "last_step": last_step}, indent=2))
    finally:
        interrupt.restore()

    if dist_ctx.is_main and stop_reason != "interrupted" and val_loader is not None and last_step > 0 and last_validation_step != last_step:
        val_row = evaluate_validation_loss(
            student=ema,
            teacher=teacher,
            loader=val_loader,
            sigma_values=sigma_values,
            label_dim=label_dim,
            device=device,
            microbatch=args.val_microbatch,
            seed=args.val_seed,
            step=last_step,
            plan=plan,
            model_family=plan.get("model_family", "vp"),
            sigma_data=float(getattr(teacher, "sigma_data", 0.5)),
            kd_weight=args.kd_weight,
            data_weight=args.data_weight,
        )
        validation_stats.append(val_row)
        append_jsonl(validation_stats_path, val_row)
        tensorboard.log_row("validation", val_row)
        val_improved = best_val.update(val_row, output_dir=output_dir)
        if val_improved:
            save_snapshot(
                output_dir / best_val.snapshot_name,
                step=last_step,
                student=student,
                ema=ema,
                plan=plan,
                stats=stats,
                optimizer=optimizer,
                update_manifest=True,
            )
        tensorboard.log_metric(
            "checkpoint/best_val_updated",
            float(int(val_improved)),
            step=last_step,
            direction=HIGHER_IS_BETTER,
        )

    if dist_ctx.is_main:
        step_snapshot = output_dir / f"student-snapshot-step{last_step:06d}.pt"
        save_snapshot(
            step_snapshot,
            step=last_step,
            student=student,
            ema=ema,
            plan=plan,
            stats=stats,
            optimizer=optimizer,
            update_manifest=True,
        )
        final_snapshot = output_dir / "student-final.pt"
        save_snapshot(
            final_snapshot,
            step=last_step,
            student=student,
            ema=ema,
            plan=plan,
            stats=stats,
            optimizer=optimizer,
            update_manifest=True,
        )
        save_training_state(
            output_dir / TRAINING_STATE_FILENAME,
            step=last_step,
            optimizer=optimizer,
            student=student,
        )
        if args.keep_last_snapshots is not None:
            prune_periodic_snapshots(
                output_dir,
                keep=args.keep_last_snapshots,
                training_active=False,
            )
        selection = build_checkpoint_selection(
            stop_reason=stop_reason,
            final_step=last_step,
            final_snapshot=final_snapshot,
            best_val=best_val,
            best_fid=best_fid,
            selection_policy=args.checkpoint_selection,
        )
        (output_dir / "checkpoint_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
        tensorboard.add_text("checkpoint/selection", json.dumps(selection, indent=2, sort_keys=True), step=last_step)
        tensorboard.log_metric("checkpoint/final_step", last_step, step=last_step, direction=HIGHER_IS_BETTER)
        tensorboard.flush()
        tensorboard.close()
        print(json.dumps({"output_dir": str(output_dir), "final_snapshot": str(final_snapshot), "selection": selection["selected"]}, indent=2))
    else:
        tensorboard.close()
    cleanup_distributed(dist_ctx)


if __name__ == "__main__":
    main()
