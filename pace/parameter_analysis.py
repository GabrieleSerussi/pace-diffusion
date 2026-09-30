"""Modality-neutral helpers for diffusion parameter-usage analysis.

The image and audio profilers share the same paired-ablation estimand, PFI
mechanics, effective-capacity metrics, and artifact plots.  Model adapters own
only corruption/model execution and the enumeration of analysis groups.
"""

from __future__ import annotations

import fnmatch
import glob
import hashlib
import json
import math
import os
import random
import struct
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

import matplotlib
import torch
import torch.distributed as dist
from torch.utils.data import Sampler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:  # Optional: only the interactive HTML heatmaps of large matrices use plotly.
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - exercised only without plotly.
    go = None


AblationMode = Literal["zero", "random_same_norm", "pfi"]


@dataclass(frozen=True, slots=True)
class DistributedAnalysisContext:
    """Process identity and device selected for an analysis worker.

    Analysis ranks hold independent frozen model replicas.  The context is
    deliberately smaller than DDP: it only provides coordination primitives
    for shared plans, resumable result shards, and rank-zero publication.
    """

    device: torch.device
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: Optional[str] = None
    owns_process_group: bool = False

    def __post_init__(self) -> None:
        if self.rank < 0 or self.local_rank < 0 or self.world_size <= 0:
            raise ValueError("Distributed rank metadata must be non-negative with world_size > 0")
        if self.rank >= self.world_size:
            raise ValueError(f"rank {self.rank} must be smaller than world_size {self.world_size}")

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.is_distributed:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("Distributed analysis context has no initialized process group")
            dist.barrier()


def singleton_analysis_context(device: str | torch.device = "cpu") -> DistributedAnalysisContext:
    """Return a non-collective context for tests and single-process runs."""

    return DistributedAnalysisContext(device=torch.device(device))


def init_distributed_analysis(
    device: str | torch.device,
    *,
    timeout_seconds: int = 86_400,
) -> DistributedAnalysisContext:
    """Initialize the torchrun process group and bind CUDA to ``LOCAL_RANK``.

    A process group is only created when ``WORLD_SIZE`` is greater than one.
    If a caller already initialized one, it is reused and ownership remains
    with that caller.
    """

    if timeout_seconds <= 0:
        raise ValueError("distributed timeout must be positive")
    requested_device = torch.device(device)
    already_initialized = dist.is_available() and dist.is_initialized()
    environment_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if environment_world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {environment_world_size}")

    owns_process_group = False
    if already_initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        backend = str(dist.get_backend())
    elif environment_world_size > 1:
        if not dist.is_available():
            raise RuntimeError("torch.distributed is unavailable for a multi-rank analysis")
        backend = "nccl" if requested_device.type == "cuda" else "gloo"
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(seconds=int(timeout_seconds)),
        )
        owns_process_group = True
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        return singleton_analysis_context(requested_device)

    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if requested_device.type == "cuda":
        torch.cuda.set_device(local_rank)
        requested_device = torch.device("cuda", local_rank)
    return DistributedAnalysisContext(
        device=requested_device,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        backend=backend,
        owns_process_group=owns_process_group,
    )


def cleanup_distributed_analysis(
    context: Optional[DistributedAnalysisContext] = None,
    *,
    force: bool = False,
) -> None:
    """Destroy a process group created by :func:`init_distributed_analysis`."""

    if not dist.is_available() or not dist.is_initialized():
        return
    if force or context is None or context.owns_process_group:
        dist.destroy_process_group()


def _all_gather_analysis_objects(
    context: DistributedAnalysisContext,
    value: Any,
) -> list[Any]:
    if not context.is_distributed:
        return [value]
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Distributed analysis context has no initialized process group")
    gathered: list[Any] = [None] * context.world_size
    dist.all_gather_object(gathered, value)
    return gathered


def run_rank_zero_analysis_operation(
    context: DistributedAnalysisContext,
    name: str,
    operation: Callable[[], Any],
) -> Any:
    """Run one shared preparation step on rank zero and propagate failures.

    A bare barrier after rank-zero-only filesystem work strands peers when the
    writer raises.  This helper instead broadcasts a small success/error
    envelope, ensuring every rank either receives the result or raises the
    same actionable error before subsequent collectives.
    """

    if not name:
        raise ValueError("Rank-zero operation name must be non-empty")
    if not context.is_distributed:
        return operation()
    envelope: list[Any] = [None]
    original_error: Optional[Exception] = None
    if context.is_main_process:
        try:
            envelope[0] = {"ok": True, "value": operation()}
        except Exception as exc:  # noqa: BLE001 - the error is propagated to every rank.
            original_error = exc
            envelope[0] = {
                "ok": False,
                "error_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                "error_message": str(exc),
            }
    dist.broadcast_object_list(envelope, src=0)
    record = envelope[0]
    if not isinstance(record, Mapping) or not isinstance(record.get("ok"), bool):
        raise RuntimeError(f"Rank-zero operation {name!r} broadcast an invalid status envelope")
    if not record["ok"]:
        error = RuntimeError(
            f"Rank-zero operation {name!r} failed: "
            f"{record.get('error_type', 'Exception')}: {record.get('error_message', '')}"
        )
        if original_error is not None:
            raise error from original_error
        raise error
    return record.get("value")


def assert_distributed_hash_invariant(
    context: DistributedAnalysisContext,
    name: str,
    sha256: str,
) -> str:
    """Require every rank to report the same canonical SHA-256 digest."""

    if not name:
        raise ValueError("Distributed invariant name must be non-empty")
    digest = str(sha256).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"Distributed invariant {name!r} is not a SHA-256 digest: {sha256!r}")
    gathered = _all_gather_analysis_objects(
        context,
        {"rank": context.rank, "sha256": digest},
    )
    details = sorted(
        (int(item.get("rank", -1)), str(item.get("sha256")))
        for item in gathered
        if isinstance(item, Mapping)
    )
    observed = {item_digest for _, item_digest in details}
    observed_ranks = {rank for rank, _ in details}
    if (
        len(details) != context.world_size
        or observed_ranks != set(range(context.world_size))
        or observed != {digest}
    ):
        raise ValueError(f"Distributed invariant {name!r} differs across ranks: {details}")
    return digest


def assert_distributed_canonical_invariant(
    context: DistributedAnalysisContext,
    name: str,
    payload: Any,
) -> str:
    """Hash a JSON-compatible record and require rank-wise identity."""

    digest = canonical_json_sha256(payload)
    return assert_distributed_hash_invariant(context, name, digest)


@dataclass(frozen=True)
class AnalysisAxis:
    """Ordered native and normalized coordinates for one diffusion profile."""

    kind: str
    ordering: str
    native_coordinate: str
    normalized_coordinate: str
    native_values: tuple[int | float, ...]
    normalized_values: tuple[float, ...]
    bin_labels: tuple[str, ...]
    bin_members: tuple[tuple[int | float, ...], ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.native_values:
            raise ValueError("AnalysisAxis requires at least one native value")
        if len(self.native_values) != len(self.normalized_values):
            raise ValueError("AnalysisAxis native and normalized coordinates must align")
        if not self.bin_labels or len(self.bin_labels) != len(self.bin_members):
            raise ValueError("AnalysisAxis bin labels and members must align")
        flattened = tuple(value for members in self.bin_members for value in members)
        if flattened != self.native_values:
            raise ValueError("AnalysisAxis bins must partition native_values in order")

    @property
    def num_levels(self) -> int:
        return len(self.native_values)

    @property
    def num_bins(self) -> int:
        return len(self.bin_labels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "ordering": self.ordering,
            "native_coordinate": self.native_coordinate,
            "normalized_coordinate": self.normalized_coordinate,
            "native_values": list(self.native_values),
            "normalized_values": list(self.normalized_values),
            "bin_labels": list(self.bin_labels),
            "bin_members": [list(values) for values in self.bin_members],
            **dict(self.metadata),
        }


@dataclass(frozen=True)
class IndexedOutputTarget:
    """One output channel of a module, addressed without mutating its output.

    Dimension zero is reserved for the activation batch. Negative channel
    dimensions are accepted and resolved once the hooked output shape is
    known.
    """

    module: torch.nn.Module
    channel_index: int
    channel_dim: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.module, torch.nn.Module):
            raise TypeError("IndexedOutputTarget.module must be a torch.nn.Module")
        if isinstance(self.channel_index, bool) or not isinstance(self.channel_index, int):
            raise TypeError("IndexedOutputTarget.channel_index must be an integer")
        if self.channel_index < 0:
            raise ValueError("IndexedOutputTarget.channel_index must be non-negative")
        if isinstance(self.channel_dim, bool) or not isinstance(self.channel_dim, int):
            raise TypeError("IndexedOutputTarget.channel_dim must be an integer")
        if self.channel_dim == 0:
            raise ValueError("IndexedOutputTarget.channel_dim cannot be the batch dimension 0")


AnalysisTarget = torch.nn.Module | IndexedOutputTarget


@runtime_checkable
class DiffusionAnalysisAdapter(Protocol):
    """Minimal model-facing contract consumed by the shared analysis loop."""

    model: torch.nn.Module
    axis: AnalysisAxis

    def named_analysis_groups(self) -> tuple[tuple[str, AnalysisTarget], ...]: ...

    def predict_noise(self, x_t: torch.Tensor, levels: torch.Tensor) -> torch.Tensor: ...

    def forward_losses_from_fixed_corruption(
        self,
        *loss_inputs: Any,
    ) -> torch.Tensor: ...

    def unpack_analysis_batch(self, batch: Any) -> "BinnedEvaluationBatch": ...


@dataclass(frozen=True)
class BinnedEvaluationBatch:
    """Modality-neutral inputs needed by the shared binned evaluator."""

    loss_inputs: tuple[Any, ...]
    level_positions: torch.Tensor
    pfi_permutation: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if not isinstance(self.loss_inputs, tuple):
            raise TypeError("BinnedEvaluationBatch.loss_inputs must be a tuple")
        if not torch.is_tensor(self.level_positions):
            raise TypeError("BinnedEvaluationBatch.level_positions must be a tensor")
        if self.pfi_permutation is not None and not torch.is_tensor(self.pfi_permutation):
            raise TypeError("BinnedEvaluationBatch.pfi_permutation must be a tensor or None")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def choose_dtype(name: str) -> torch.dtype:
    choices = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    try:
        return choices[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def mse_per_example(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dimensions = tuple(range(1, prediction.ndim))
    return ((prediction - target) ** 2).mean(dim=dimensions)


def level_to_bin(indices: torch.Tensor, num_levels: int, num_bins: int) -> torch.Tensor:
    if num_levels <= 0:
        raise ValueError(f"num_levels must be positive, got {num_levels}")
    if not 0 < num_bins <= num_levels:
        raise ValueError(f"num_bins must be in [1, {num_levels}], got {num_bins}")
    bins = torch.floor(indices.float() * num_bins / num_levels).long()
    return torch.clamp(bins, min=0, max=num_bins - 1)


def sanitize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def compute_signed_and_positive_deltas(
    ablated_means: Mapping[str, torch.Tensor],
    baseline_mean: torch.Tensor,
    group_names: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    signed = torch.stack(
        [sanitize_tensor(ablated_means[name] - baseline_mean) for name in group_names],
        dim=0,
    )
    return signed, torch.clamp(signed, min=0.0)


def row_normalize_matrix(matrix: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    row_sums = matrix.sum(dim=1, keepdim=True) + eps
    return sanitize_tensor(matrix / row_sums)


def validate_positive_delta_mass(delta_stack: torch.Tensor) -> torch.Tensor:
    """Require at least one strictly positive sensitivity contribution per bin."""

    if delta_stack.ndim != 2:
        raise ValueError(
            "delta_stack must be two-dimensional [num_groups, num_bins], "
            f"got shape {tuple(delta_stack.shape)}"
        )
    mass = sanitize_tensor(delta_stack.to(torch.float64)).sum(dim=0)
    empty_bins = torch.nonzero(mass <= 0, as_tuple=False).flatten().tolist()
    if empty_bins:
        raise ValueError(
            "Positive ablation-delta mass is zero for timestep bins "
            f"{empty_bins}; n_eff would be undefined for this profile"
        )
    return mass


def compute_usage_metrics(
    delta_stack: torch.Tensor,
    baseline_mean: torch.Tensor,
    group_param_counts: Mapping[str, int],
    group_names: Sequence[str],
    compute_group_correlation: bool = True,
    require_positive_delta_mass: bool = False,
) -> Dict[str, Optional[torch.Tensor]]:
    """Compute the repository's canonical sensitivity/capacity metrics."""

    delta_stack = sanitize_tensor(delta_stack.to(torch.float64))
    baseline_mean = sanitize_tensor(baseline_mean.to(torch.float64))
    if delta_stack.ndim != 2:
        raise ValueError(f"delta_stack must be 2D, got shape {tuple(delta_stack.shape)}")
    if baseline_mean.ndim != 1 or baseline_mean.shape[0] != delta_stack.shape[1]:
        raise ValueError(
            "baseline_mean must be 1D and match the delta timestep dimension, "
            f"got {tuple(baseline_mean.shape)} and {tuple(delta_stack.shape)}"
        )
    if len(group_names) != delta_stack.shape[0]:
        raise ValueError(
            f"group_names has {len(group_names)} entries for {delta_stack.shape[0]} delta rows"
        )
    missing_counts = [name for name in group_names if name not in group_param_counts]
    if missing_counts:
        raise KeyError(f"Missing parameter counts for groups: {missing_counts[:5]}")

    positive_delta_mass = delta_stack.sum(dim=0)
    if require_positive_delta_mass:
        positive_delta_mass = validate_positive_delta_mass(delta_stack)

    relative_delta_stack = sanitize_tensor(delta_stack / (baseline_mean.unsqueeze(0) + 1e-12))
    row_normalized_relative_delta_stack = row_normalize_matrix(relative_delta_stack)
    weights = sanitize_tensor(delta_stack / (positive_delta_mass.unsqueeze(0) + 1e-12))
    n_eff = sanitize_tensor(1.0 / (weights.pow(2).sum(dim=0) + 1e-12))

    parameters = torch.tensor(
        [group_param_counts[name] for name in group_names],
        dtype=torch.float64,
    ).unsqueeze(1)
    p_eff = sanitize_tensor((weights * parameters).sum(dim=0))

    if compute_group_correlation:
        if relative_delta_stack.shape[0] == 1:
            group_correlation: Optional[torch.Tensor] = torch.ones((1, 1), dtype=torch.float64)
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
        level_correlation = torch.ones((1, 1), dtype=torch.float64)
    elif relative_delta_stack.shape[0] < 2:
        level_correlation = torch.zeros(
            (relative_delta_stack.shape[1], relative_delta_stack.shape[1]),
            dtype=torch.float64,
        )
    else:
        level_correlation = sanitize_tensor(torch.corrcoef(relative_delta_stack.T))

    return {
        "delta_stack": delta_stack,
        "relative_delta_stack": relative_delta_stack,
        "row_normalized_relative_delta_stack": row_normalized_relative_delta_stack,
        "positive_delta_mass": positive_delta_mass,
        "weights": weights,
        "n_eff": n_eff,
        "p_eff": p_eff,
        "C_groups": group_correlation,
        # Retain the historical internal name; serializers select C_timesteps
        # or C_noise_levels from axis metadata.
        "C_noise_levels": level_correlation,
    }


def should_compute_group_correlation(mode: str, num_groups: int, max_groups: int) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    if mode != "auto":
        raise ValueError(f"Unsupported group correlation mode: {mode}")
    return num_groups <= max_groups


def tensor_or_none_to_list(value: Optional[torch.Tensor]) -> Optional[list[Any]]:
    return None if value is None else value.tolist()


class BinStats:
    def __init__(self, num_bins: int):
        if num_bins <= 0:
            raise ValueError(f"num_bins must be positive, got {num_bins}")
        self.sum = torch.zeros(num_bins, dtype=torch.float64)
        self.sum_sq = torch.zeros(num_bins, dtype=torch.float64)
        self.count = torch.zeros(num_bins, dtype=torch.long)

    def update(self, bin_ids: torch.Tensor, values: torch.Tensor) -> None:
        bin_ids = bin_ids.detach().cpu().long().reshape(-1)
        values = values.detach().cpu().double().reshape(-1)
        if bin_ids.shape != values.shape:
            raise ValueError("bin_ids and values must contain the same number of elements")
        self.sum.scatter_add_(0, bin_ids, values)
        self.sum_sq.scatter_add_(0, bin_ids, values.square())
        self.count.scatter_add_(0, bin_ids, torch.ones_like(bin_ids, dtype=torch.long))

    def mean(self) -> torch.Tensor:
        return self.sum / self.count.clamp(min=1).to(torch.float64)

    def stderr(self) -> torch.Tensor:
        count = self.count.clamp(min=1).to(torch.float64)
        mean = self.mean()
        variance = torch.clamp(self.sum_sq / count - mean.square(), min=0.0)
        return torch.sqrt(variance / count)


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
    leading = tensor.shape[0] if tensor.ndim else 1
    flat_tensor = tensor_f32.reshape(leading, -1)
    flat_noise = noise.reshape(leading, -1)
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
    return (flat_noise * scale).reshape(tensor.shape).to(dtype=original_dtype)


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
        return random_same_norm_like(tensor, self._generator_for_device(tensor.device))


class PFIPermutationMixin:
    def __init__(self) -> None:
        self._pfi_permutation: Optional[torch.Tensor] = None

    def set_pfi_permutation(self, permutation: torch.Tensor) -> None:
        permutation = permutation.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        size = int(permutation.numel())
        identity = torch.arange(size, dtype=torch.long)
        if size < 2 or sorted(permutation.tolist()) != list(range(size)):
            raise ValueError("PFI permutation must be a bijection over at least two batch rows")
        if bool(torch.any(permutation == identity)):
            raise ValueError("PFI permutation must be fixed-point-free")
        self._pfi_permutation = permutation

    def _pfi_replace(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._pfi_permutation is None:
            raise RuntimeError("PFI hook was invoked before the batch permutation was set")
        if tensor.ndim == 0 or tensor.shape[0] != self._pfi_permutation.numel():
            raise ValueError(
                "PFI expects activation batch dimension 0 to have size "
                f"{self._pfi_permutation.numel()}, got shape {tuple(tensor.shape)}"
            )
        return tensor.index_select(0, self._pfi_permutation.to(tensor.device)).clone()


def _map_tensor_leaves(value: Any, transform) -> Any:
    if torch.is_tensor(value):
        return transform(value)
    if isinstance(value, tuple):
        return tuple(_map_tensor_leaves(item, transform) for item in value)
    if isinstance(value, list):
        return [_map_tensor_leaves(item, transform) for item in value]
    if isinstance(value, dict):
        return {key: _map_tensor_leaves(item, transform) for key, item in value.items()}
    return value


class ZeroOutputHook:
    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.handle = None

    def _hook(self, module, inputs, output):
        del module, inputs
        return _map_tensor_leaves(output, torch.zeros_like)

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del exc_type, exc_val, exc_tb
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class RandomSameNormOutputHook(RandomSameNormMixin):
    def __init__(self, module: torch.nn.Module, random_seed: int):
        super().__init__(random_seed=random_seed)
        self.module = module
        self.handle = None

    def _hook(self, module, inputs, output):
        del module, inputs
        return _map_tensor_leaves(output, self._random_same_norm)

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del exc_type, exc_val, exc_tb
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class PFIOutputHook(PFIPermutationMixin):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.handle = None

    def _hook(self, module, inputs, output):
        del module, inputs
        return _map_tensor_leaves(output, self._pfi_replace)

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del exc_type, exc_val, exc_tb
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def _indexed_channel_slice(
    target: IndexedOutputTarget,
    output: Any,
) -> tuple[torch.Tensor, tuple[slice | int, ...]]:
    if not torch.is_tensor(output):
        raise TypeError(
            "Indexed output hooks require a single tensor output, got "
            f"{type(output).__name__}"
        )
    if output.ndim < 2:
        raise ValueError(
            "Indexed output hooks require an activation with a batch and channel "
            f"dimension, got shape {tuple(output.shape)}"
        )
    channel_dim = target.channel_dim
    if channel_dim < 0:
        channel_dim += output.ndim
    if channel_dim <= 0 or channel_dim >= output.ndim:
        raise ValueError(
            f"Invalid channel_dim={target.channel_dim} for output shape {tuple(output.shape)}"
        )
    num_channels = output.shape[channel_dim]
    if target.channel_index >= num_channels:
        raise ValueError(
            f"Channel index {target.channel_index} is out of range for output with "
            f"{num_channels} channels along dimension {channel_dim}"
        )
    selector: list[slice | int] = [slice(None)] * output.ndim
    selector[channel_dim] = target.channel_index
    return output, tuple(selector)


class IndexedZeroOutputHook:
    """Zero exactly one output channel while leaving every other channel intact."""

    def __init__(self, target: IndexedOutputTarget):
        self.target = target
        self.module = target.module
        self.handle = None

    def _hook(self, module, inputs, output):
        del module, inputs
        output, selector = _indexed_channel_slice(self.target, output)
        replaced = output.clone()
        replaced[selector] = 0
        return replaced

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del exc_type, exc_val, exc_tb
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class IndexedRandomSameNormOutputHook(RandomSameNormMixin):
    """Replace one output channel with same-norm Gaussian noise."""

    def __init__(self, target: IndexedOutputTarget, random_seed: int):
        super().__init__(random_seed=random_seed)
        self.target = target
        self.module = target.module
        self.handle = None

    def _hook(self, module, inputs, output):
        del module, inputs
        output, selector = _indexed_channel_slice(self.target, output)
        replaced = output.clone()
        replaced[selector] = self._random_same_norm(output[selector])
        return replaced

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del exc_type, exc_val, exc_tb
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class IndexedPFIOutputHook(PFIPermutationMixin):
    """Exchange one output channel between externally planned donor rows."""

    def __init__(self, target: IndexedOutputTarget):
        super().__init__()
        self.target = target
        self.module = target.module
        self.handle = None

    def _hook(self, module, inputs, output):
        del module, inputs
        output, selector = _indexed_channel_slice(self.target, output)
        replaced = output.clone()
        replaced[selector] = self._pfi_replace(output[selector])
        return replaced

    def __enter__(self):
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del exc_type, exc_val, exc_tb
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def make_output_hook(
    target: AnalysisTarget,
    mode: AblationMode,
    *,
    random_seed: Optional[int] = None,
):
    if isinstance(target, IndexedOutputTarget):
        if mode == "zero":
            return IndexedZeroOutputHook(target)
        if mode == "random_same_norm":
            if random_seed is None:
                raise ValueError("random_seed is required for random_same_norm ablation")
            return IndexedRandomSameNormOutputHook(target, random_seed=random_seed)
        if mode == "pfi":
            return IndexedPFIOutputHook(target)
        raise ValueError(f"Unsupported ablation mode: {mode}")

    module = target
    if mode == "zero":
        return ZeroOutputHook(module)
    if mode == "random_same_norm":
        if random_seed is None:
            raise ValueError("random_seed is required for random_same_norm ablation")
        return RandomSameNormOutputHook(module, random_seed=random_seed)
    if mode == "pfi":
        return PFIOutputHook(module)
    raise ValueError(f"Unsupported ablation mode: {mode}")


def evaluate_binned_losses(
    adapter: DiffusionAnalysisAdapter,
    dataloader: Iterable[Any],
    *,
    num_bins: int,
    ablate_target: Optional[AnalysisTarget] = None,
    ablation_mode: AblationMode = "zero",
    ablation_random_seed: Optional[int] = None,
    progress_desc: Optional[str] = None,
    progress_factory: Optional[
        Callable[[Iterable[Any], Optional[str]], Iterable[Any]]
    ] = None,
) -> BinStats:
    """Evaluate fixed-corruption losses and aggregate them on an analysis axis.

    The runner owns hook lifetime, PFI permutation installation, finite-value
    checks, and bin accumulation.  An adapter only has to decode its modality's
    loader batch and compute paired per-example losses.
    """

    statistics = BinStats(num_bins)
    context = None
    if ablate_target is not None:
        context = make_output_hook(
            ablate_target,
            ablation_mode,
            random_seed=ablation_random_seed,
        )
        context.__enter__()
    try:
        batches: Iterable[Any] = dataloader
        if progress_factory is not None:
            batches = progress_factory(dataloader, progress_desc)
        for raw_batch in batches:
            batch = adapter.unpack_analysis_batch(raw_batch)
            if batch.pfi_permutation is not None and context is not None:
                if not hasattr(context, "set_pfi_permutation"):
                    raise RuntimeError("A non-PFI hook received a PFI permutation")
                context.set_pfi_permutation(batch.pfi_permutation)
            losses = adapter.forward_losses_from_fixed_corruption(*batch.loss_inputs)
            if not torch.is_tensor(losses):
                raise TypeError("Analysis adapters must return a tensor of per-example losses")
            if not torch.isfinite(losses).all():
                raise ValueError("Diffusion analysis produced non-finite losses")
            statistics.update(
                level_to_bin(
                    batch.level_positions,
                    num_levels=adapter.axis.num_levels,
                    num_bins=num_bins,
                ),
                losses,
            )
    finally:
        if context is not None:
            context.__exit__(None, None, None)
    return statistics


def _stable_sha256_fields(*values: object) -> bytes:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
    return digest.digest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def balanced_batch_sizes(population_size: int, requested_batch_size: int) -> List[int]:
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
        raise RuntimeError("Could not form non-singleton PFI batches")
    return sizes


@dataclass(frozen=True, slots=True)
class PFILevelSampleIndex:
    sample_index: int
    example_index: int
    level_index: int
    donor_position: int
    batch_ordinal: int

    @property
    def image_index(self) -> int:
        return self.example_index

    @property
    def sigma_index(self) -> int:
        return self.level_index


class ExactLevelDataset(Protocol):
    num_examples: int
    level_indices: Sequence[int]

    def sample_index(self, example_index: int, level_position: int) -> int: ...


class ExactLevelPFIBatchSampler(Sampler[List[PFILevelSampleIndex]]):
    """Deterministic exact-level PFI batches for any diffusion modality."""

    plan_format = "diffdist_batch_local_exact_level_pfi_plan_v1"

    def __init__(
        self,
        dataset: ExactLevelDataset,
        *,
        batch_size: int,
        pfi_seed: int,
        population_fingerprint: str,
        level_kind: str,
    ) -> None:
        self.dataset = dataset
        self.requested_batch_size = int(batch_size)
        self.pfi_seed = int(pfi_seed)
        self.population_fingerprint = str(population_fingerprint)
        self.level_kind = str(level_kind)
        self.batch_sizes = balanced_batch_sizes(dataset.num_examples, self.requested_batch_size)
        self.batch_offsets = [0]
        for size in self.batch_sizes:
            self.batch_offsets.append(self.batch_offsets[-1] + size)

        num_levels = len(dataset.level_indices)
        self._member_orders = torch.empty((num_levels, dataset.num_examples), dtype=torch.int32)
        self._donor_positions = torch.empty((num_levels, dataset.num_examples), dtype=torch.int32)
        header = {
            "format": self.plan_format,
            "level_kind": self.level_kind,
            "pfi_seed": self.pfi_seed,
            "population_fingerprint": self.population_fingerprint,
            "population_size": dataset.num_examples,
            "level_indices": [int(value) for value in dataset.level_indices],
            "requested_batch_size": self.requested_batch_size,
            "balanced_batch_sizes": list(self.batch_sizes),
        }
        digest = hashlib.sha256(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8"))

        for level_position, level_index in enumerate(dataset.level_indices):
            members = sorted(
                range(dataset.num_examples),
                key=lambda example_index: (
                    _stable_sha256_fields(
                        self.plan_format,
                        "membership",
                        self.pfi_seed,
                        self.population_fingerprint,
                        level_index,
                        example_index,
                    ),
                    example_index,
                ),
            )
            self._member_orders[level_position] = torch.tensor(members, dtype=torch.int32)
            digest.update(struct.pack(">Q", int(level_index)))
            for batch_ordinal, (start, end) in enumerate(zip(self.batch_offsets, self.batch_offsets[1:])):
                batch_members = members[start:end]
                ranked_positions = sorted(
                    range(len(batch_members)),
                    key=lambda position: (
                        _stable_sha256_fields(
                            self.plan_format,
                            "derangement",
                            self.pfi_seed,
                            self.population_fingerprint,
                            level_index,
                            batch_ordinal,
                            batch_members[position],
                        ),
                        batch_members[position],
                    ),
                )
                donor_positions = [0] * len(batch_members)
                for rank, recipient_position in enumerate(ranked_positions):
                    donor_positions[recipient_position] = ranked_positions[(rank + 1) % len(ranked_positions)]
                if any(position == donor for position, donor in enumerate(donor_positions)):
                    raise RuntimeError("PFI planner produced a fixed point")
                self._donor_positions[level_position, start:end] = torch.tensor(
                    donor_positions,
                    dtype=torch.int32,
                )
                digest.update(torch.tensor(batch_members, dtype=torch.int32).numpy().tobytes())
                digest.update(torch.tensor(donor_positions, dtype=torch.int32).numpy().tobytes())

        self.plan_sha256 = digest.hexdigest()
        self.artifact = {**header, "plan_sha256": self.plan_sha256}

    def __iter__(self) -> Iterable[List[PFILevelSampleIndex]]:
        for level_position, level_index in enumerate(self.dataset.level_indices):
            members = self._member_orders[level_position].tolist()
            donors = self._donor_positions[level_position].tolist()
            for batch_ordinal, (start, end) in enumerate(zip(self.batch_offsets, self.batch_offsets[1:])):
                yield [
                    PFILevelSampleIndex(
                        sample_index=self.dataset.sample_index(int(members[position]), level_position),
                        example_index=int(members[position]),
                        level_index=int(level_index),
                        donor_position=int(donors[position]),
                        batch_ordinal=batch_ordinal,
                    )
                    for position in range(start, end)
                ]

    def __len__(self) -> int:
        return len(self.dataset.level_indices) * len(self.batch_sizes)


def dataset_population_fingerprint(dataset: Any) -> str:
    metadata = getattr(dataset, "metadata", {})
    record = {
        "dataset_class": dataset.__class__.__name__,
        "count": len(dataset),
        "selected_entries_sha256": metadata.get("selected_entries_sha256") if isinstance(metadata, Mapping) else None,
        "selected_content_sha256": metadata.get("selected_content_sha256") if isinstance(metadata, Mapping) else None,
        "entries_sha256": metadata.get("entries_sha256") if isinstance(metadata, Mapping) else None,
        "source_listing_sha256": metadata.get("source_listing_sha256") if isinstance(metadata, Mapping) else None,
        "source_records_sha256": metadata.get("source_records_sha256") if isinstance(metadata, Mapping) else None,
        "split": metadata.get("split") if isinstance(metadata, Mapping) else getattr(dataset, "split", None),
    }
    return canonical_json_sha256(record)


def build_ablation_protocol(
    mode: AblationMode,
    *,
    level_kind: str,
    pfi_plan: Optional[ExactLevelPFIBatchSampler] = None,
    pfi_seed: Optional[int] = None,
) -> Dict[str, Any]:
    replacement = {
        "zero": "zeros",
        "random_same_norm": "gaussian_random_noise_with_per_example_l2_norm",
        "pfi": f"batch_local_exact_{level_kind}_whole_group_activation_exchange",
    }[mode]
    record: Dict[str, Any] = {
        "format": "diffdist_ablation_protocol_v1",
        "protocol_id": f"batch_local_exact_{level_kind}_pfi_v1" if mode == "pfi" else f"legacy_{mode}_v1",
        "mode": mode,
        "replacement": replacement,
        "score": "mean_ablated_loss_minus_mean_paired_baseline_loss",
        "signed_scores_retained": mode == "pfi",
        "allocation_uses_positive_part": True,
    }
    if mode == "pfi":
        if pfi_plan is None or pfi_seed is None:
            raise ValueError("PFI protocol requires a deterministic plan and seed")
        record["pfi"] = {
            "estimand": f"batch_local_exact_{level_kind}_activation_permutation_importance",
            "seed": int(pfi_seed),
            f"exact_{level_kind}_conditioned": True,
            "whole_group_tensor": True,
            "fixed_point_free": True,
            "same_plan_for_all_groups": True,
            "permutation_repetitions": 1,
            "plan": dict(pfi_plan.artifact),
        }
    return record


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> None:
    """Atomically replace a torch artifact without leaving a partial cache."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def persist_exact_level_pfi_plan(
    path: str | Path,
    plan: Optional[ExactLevelPFIBatchSampler],
) -> Optional[dict[str, Any]]:
    """Persist or validate the compact plan used by every group evaluation."""

    if plan is None:
        return None
    destination = Path(path)
    payload = {
        "metadata": plan.artifact,
        "member_orders": plan._member_orders,
        "donor_positions": plan._donor_positions,
    }
    if destination.is_file():
        existing = torch.load(destination, map_location="cpu", weights_only=True)
        existing_members = existing.get("member_orders") if isinstance(existing, Mapping) else None
        existing_donors = existing.get("donor_positions") if isinstance(existing, Mapping) else None
        if (
            not isinstance(existing, Mapping)
            or existing.get("metadata") != plan.artifact
            or not torch.is_tensor(existing_members)
            or not torch.equal(existing_members, plan._member_orders)
            or not torch.is_tensor(existing_donors)
            or not torch.equal(existing_donors, plan._donor_positions)
        ):
            raise ValueError(
                f"Existing PFI plan differs from the requested profile: {destination}"
            )
    else:
        atomic_torch_save(payload, destination)
    return {
        "path": str(destination),
        "plan_sha256": plan.plan_sha256,
        **plan.artifact,
    }


def _validate_cache_fingerprint(
    payload: Mapping[str, Any],
    *,
    expected_format: str,
    profile_fingerprint: Mapping[str, Any],
    profile_fingerprint_sha256: str,
    path: Path,
) -> None:
    if payload.get("format") != expected_format:
        raise ValueError(f"Unexpected cache format in {path}: {payload.get('format')!r}")
    if payload.get("profile_fingerprint_sha256") != profile_fingerprint_sha256:
        raise ValueError(f"Existing cache belongs to a different profile fingerprint: {path}")
    if payload.get("profile_fingerprint") != profile_fingerprint:
        raise ValueError(
            f"Existing cache fingerprint payload differs from requested profile: {path}"
        )


def load_binned_statistics_cache(
    path: str | Path,
    *,
    cache_format: str,
    num_bins: int,
    profile_fingerprint: Mapping[str, Any],
    profile_fingerprint_sha256: str,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Load fingerprint-bound mean, stderr, and count vectors."""

    source = Path(path)
    if not source.is_file():
        return None
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Binned-statistics cache must contain a mapping: {source}")
    _validate_cache_fingerprint(
        payload,
        expected_format=cache_format,
        profile_fingerprint=profile_fingerprint,
        profile_fingerprint_sha256=profile_fingerprint_sha256,
        path=source,
    )
    if int(payload.get("num_bins", -1)) != num_bins:
        raise ValueError(f"Binned-statistics cache num_bins mismatch: {source}")
    tensors = tuple(payload.get(key) for key in ("mean", "stderr", "count"))
    expected_dtypes = (torch.float64, torch.float64, torch.long)
    for key, tensor, dtype in zip(
        ("mean", "stderr", "count"), tensors, expected_dtypes, strict=True
    ):
        if not torch.is_tensor(tensor) or tensor.shape != (num_bins,) or tensor.dtype != dtype:
            raise ValueError(f"Invalid cached binned-statistics tensor {key!r} in {source}")
    mean, stderr, count = tensors
    assert torch.is_tensor(mean) and torch.is_tensor(stderr) and torch.is_tensor(count)
    if not torch.isfinite(mean).all() or not torch.isfinite(stderr).all() or bool(torch.any(count < 0)):
        raise ValueError(f"Invalid cached binned-statistics values in {source}")
    return tensors  # type: ignore[return-value]


def save_binned_statistics_cache(
    path: str | Path,
    *,
    cache_format: str,
    mean: torch.Tensor,
    stderr: torch.Tensor,
    count: torch.Tensor,
    profile_fingerprint: Mapping[str, Any],
    profile_fingerprint_sha256: str,
) -> None:
    """Save fingerprint-bound mean, stderr, and count vectors."""

    if canonical_json_sha256(profile_fingerprint) != profile_fingerprint_sha256:
        raise ValueError("Profile fingerprint digest does not match its canonical record")
    if mean.ndim != 1 or stderr.shape != mean.shape or count.shape != mean.shape:
        raise ValueError("Binned-statistics tensors must be aligned one-dimensional vectors")
    if not torch.isfinite(mean).all() or not torch.isfinite(stderr).all() or bool(torch.any(count < 0)):
        raise ValueError("Binned-statistics tensors must contain finite values and non-negative counts")
    atomic_torch_save(
        {
            "format": cache_format,
            "num_bins": int(mean.numel()),
            "profile_fingerprint": dict(profile_fingerprint),
            "profile_fingerprint_sha256": profile_fingerprint_sha256,
            "mean": mean.detach().cpu().to(torch.float64),
            "stderr": stderr.detach().cpu().to(torch.float64),
            "count": count.detach().cpu().to(torch.long),
        },
        path,
    )


def load_group_tensor_cache(
    path: str | Path,
    *,
    cache_format: str,
    tensor_key: str,
    completed_key: str,
    group_names: Sequence[str],
    num_bins: int,
    profile_fingerprint: Mapping[str, Any],
    profile_fingerprint_sha256: str,
) -> dict[str, torch.Tensor]:
    """Load ordered per-group vectors bound to one complete profile."""

    source = Path(path)
    if not source.is_file():
        return {}
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Group tensor cache must contain a mapping: {source}")
    _validate_cache_fingerprint(
        payload,
        expected_format=cache_format,
        profile_fingerprint=profile_fingerprint,
        profile_fingerprint_sha256=profile_fingerprint_sha256,
        path=source,
    )
    if int(payload.get("num_bins", -1)) != num_bins:
        raise ValueError(f"Group tensor cache num_bins mismatch: {source}")
    raw_values = payload.get(tensor_key)
    if not isinstance(raw_values, Mapping):
        raise ValueError(f"Group tensor cache is missing {tensor_key}: {source}")
    unknown = sorted(set(raw_values) - set(group_names))
    if unknown:
        raise ValueError(f"Group tensor cache contains unknown groups {unknown[:5]}: {source}")
    loaded: dict[str, torch.Tensor] = {}
    for name, value in raw_values.items():
        if (
            not torch.is_tensor(value)
            or value.shape != (num_bins,)
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Invalid cached group tensor for {name!r}: {source}")
        loaded[str(name)] = value.detach().cpu().to(torch.float64)
    if list(payload.get(completed_key, [])) != list(loaded):
        raise ValueError(
            f"Group tensor cache {completed_key} does not match its payload: {source}"
        )
    return loaded


def group_catalog_sha256(group_names: Sequence[str]) -> str:
    """Hash the complete, ordered group catalog used for sharding and resume."""

    names = [str(name) for name in group_names]
    if len(names) != len(set(names)):
        raise ValueError("Analysis group names must be unique")
    return canonical_json_sha256(
        {
            "format": "diffdist_analysis_group_catalog_v1",
            "group_names": names,
        }
    )


def ranked_group_cache_metadata(
    context: DistributedAnalysisContext,
    *,
    group_catalog_digest: str,
) -> dict[str, Any]:
    """Return mandatory provenance fields for one resumable rank shard."""

    digest = str(group_catalog_digest).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("group_catalog_digest must be a SHA-256 digest")
    return {
        "rank": int(context.rank),
        "writer_world_size": int(context.world_size),
        "group_catalog_sha256": digest,
    }


@dataclass(frozen=True)
class GroupTensorShardMerge:
    """Strict merge result for all resumable group-cache shards."""

    values: Mapping[str, torch.Tensor]
    shard_values: Mapping[Path, Mapping[str, torch.Tensor]]
    shard_ranks: Mapping[Path, int]
    paths: tuple[Path, ...]
    duplicate_groups: tuple[str, ...]
    missing_groups: tuple[str, ...]


def discover_group_tensor_cache_shards(pattern: str | Path) -> tuple[Path, ...]:
    """Return existing shard paths matching an absolute or relative glob."""

    return tuple(Path(path).resolve() for path in sorted(glob.glob(str(pattern))))


def load_group_tensor_cache_shards(
    paths: Iterable[str | Path],
    *,
    cache_format: str,
    tensor_key: str,
    completed_key: str,
    group_names: Sequence[str],
    num_bins: int,
    profile_fingerprint: Mapping[str, Any],
    profile_fingerprint_sha256: str,
    group_catalog_digest: Optional[str] = None,
    require_rank_metadata: bool = True,
) -> GroupTensorShardMerge:
    """Strictly load and merge every compatible rank cache.

    Identical duplicate values are accepted to make restarts independent of a
    previous worker count.  Conflicting duplicates are fatal: silently taking
    the last shard would make the final profile depend on filename ordering.
    """

    canonical_names = [str(name) for name in group_names]
    canonical_set = set(canonical_names)
    if len(canonical_names) != len(canonical_set):
        raise ValueError("Analysis group names must be unique")
    expected_catalog_digest = group_catalog_digest or group_catalog_sha256(canonical_names)
    if expected_catalog_digest != group_catalog_sha256(canonical_names):
        raise ValueError("Group catalog digest does not match the ordered group names")
    if canonical_json_sha256(profile_fingerprint) != profile_fingerprint_sha256:
        raise ValueError("Profile fingerprint digest does not match its canonical record")

    normalized_paths = tuple(Path(path).resolve() for path in paths)
    if len(normalized_paths) != len(set(normalized_paths)):
        raise ValueError("Group tensor cache shard paths must be unique")

    merged: dict[str, torch.Tensor] = {}
    shard_values: dict[Path, Mapping[str, torch.Tensor]] = {}
    shard_ranks: dict[Path, int] = {}
    observed_ranks: dict[int, Path] = {}
    duplicate_groups: set[str] = set()
    for source in normalized_paths:
        if not source.is_file():
            raise FileNotFoundError(f"Group tensor cache shard does not exist: {source}")
        payload = torch.load(source, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Group tensor cache shard must contain a mapping: {source}")
        _validate_cache_fingerprint(
            payload,
            expected_format=cache_format,
            profile_fingerprint=profile_fingerprint,
            profile_fingerprint_sha256=profile_fingerprint_sha256,
            path=source,
        )
        if int(payload.get("num_bins", -1)) != num_bins:
            raise ValueError(f"Group tensor cache shard num_bins mismatch: {source}")

        if require_rank_metadata:
            rank = payload.get("rank")
            writer_world_size = payload.get("writer_world_size")
            if (
                not isinstance(rank, int)
                or isinstance(rank, bool)
                or rank < 0
                or not isinstance(writer_world_size, int)
                or isinstance(writer_world_size, bool)
                or writer_world_size <= rank
            ):
                raise ValueError(f"Group tensor cache shard has invalid rank metadata: {source}")
            if payload.get("group_catalog_sha256") != expected_catalog_digest:
                raise ValueError(f"Group tensor cache shard has a different group catalog: {source}")
            if rank in observed_ranks:
                raise ValueError(
                    f"Multiple group tensor cache shards claim rank {rank}: "
                    f"{observed_ranks[rank]} and {source}"
                )
            observed_ranks[rank] = source
            shard_ranks[source] = rank

        raw_values = payload.get(tensor_key)
        if not isinstance(raw_values, Mapping):
            raise ValueError(f"Group tensor cache shard is missing {tensor_key}: {source}")
        completed = payload.get(completed_key)
        if not isinstance(completed, list) or completed != list(raw_values):
            raise ValueError(
                f"Group tensor cache shard {completed_key} does not match its payload: {source}"
            )
        unknown = [str(name) for name in raw_values if str(name) not in canonical_set]
        if unknown:
            raise ValueError(
                f"Group tensor cache shard contains unexpected groups {unknown[:5]}: {source}"
            )

        loaded_shard: dict[str, torch.Tensor] = {}
        for raw_name, value in raw_values.items():
            name = str(raw_name)
            if (
                not torch.is_tensor(value)
                or value.shape != (num_bins,)
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"Invalid cached group tensor for {name!r}: {source}")
            normalized = value.detach().cpu().to(torch.float64)
            if name in merged:
                if not torch.equal(merged[name], normalized):
                    raise ValueError(
                        f"Conflicting duplicate cached group {name!r} in shard {source}"
                    )
                duplicate_groups.add(name)
            else:
                merged[name] = normalized
            loaded_shard[name] = normalized
        shard_values[source] = loaded_shard

    ordered_values = {name: merged[name] for name in canonical_names if name in merged}
    missing = tuple(name for name in canonical_names if name not in merged)
    return GroupTensorShardMerge(
        values=ordered_values,
        shard_values=shard_values,
        shard_ranks=shard_ranks,
        paths=normalized_paths,
        duplicate_groups=tuple(name for name in canonical_names if name in duplicate_groups),
        missing_groups=missing,
    )


def shard_missing_group_items(
    group_items: Sequence[tuple[str, Any]],
    completed_group_names: Iterable[str],
    *,
    rank: int,
    world_size: int,
) -> tuple[tuple[int, tuple[str, Any]], ...]:
    """Round-robin the globally missing canonical catalog across ranks."""

    if rank < 0 or world_size <= 0 or rank >= world_size:
        raise ValueError(f"Invalid shard coordinates rank={rank}, world_size={world_size}")
    names = [str(name) for name, _ in group_items]
    if len(names) != len(set(names)):
        raise ValueError("Analysis group names must be unique")
    known = set(names)
    completed = {str(name) for name in completed_group_names}
    unknown = sorted(completed - known)
    if unknown:
        raise ValueError(f"Completed group set contains unknown names: {unknown[:5]}")
    missing = tuple(
        (global_index, item)
        for global_index, item in enumerate(group_items)
        if item[0] not in completed
    )
    return missing[rank::world_size]


def canonicalize_complete_group_results(
    values: Mapping[str, torch.Tensor],
    group_names: Sequence[str],
    *,
    num_bins: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """Validate completeness and restore the canonical catalog order."""

    names = [str(name) for name in group_names]
    if len(names) != len(set(names)):
        raise ValueError("Analysis group names must be unique")
    unknown = sorted(set(values) - set(names))
    if unknown:
        raise ValueError(f"Ablation results contain unknown groups: {unknown[:5]}")
    missing = [name for name in names if name not in values]
    if missing:
        raise RuntimeError(f"Missing ablation results for {len(missing)} groups: {missing[:5]}")
    ordered: dict[str, torch.Tensor] = {}
    for name in names:
        value = values[name]
        if not torch.is_tensor(value) or not torch.isfinite(value).all():
            raise ValueError(f"Ablation result for {name!r} is not a finite tensor")
        if num_bins is not None and value.shape != (num_bins,):
            raise ValueError(
                f"Ablation result for {name!r} has shape {tuple(value.shape)}, "
                f"expected {(num_bins,)}"
            )
        ordered[name] = value.detach().cpu().to(torch.float64)
    return ordered


def save_group_tensor_cache(
    path: str | Path,
    *,
    cache_format: str,
    tensor_key: str,
    completed_key: str,
    values: Mapping[str, torch.Tensor],
    num_bins: int,
    profile_fingerprint: Mapping[str, Any],
    profile_fingerprint_sha256: str,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """Save ordered per-group vectors using a caller-owned cache schema."""

    if canonical_json_sha256(profile_fingerprint) != profile_fingerprint_sha256:
        raise ValueError("Profile fingerprint digest does not match its canonical record")
    normalized: dict[str, torch.Tensor] = {}
    for name, value in values.items():
        if not torch.is_tensor(value) or value.shape != (num_bins,):
            shape = tuple(value.shape) if torch.is_tensor(value) else None
            raise ValueError(
                f"Group tensor {name!r} has shape {shape}, expected {(num_bins,)}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"Group tensor {name!r} contains non-finite values")
        normalized[name] = value.detach().cpu().to(torch.float64)
    payload: dict[str, Any] = {"format": cache_format}
    if extra_metadata is not None:
        payload.update(extra_metadata)
    payload.update(
        {
            "format": cache_format,
            "num_bins": num_bins,
            completed_key: list(normalized),
            tensor_key: normalized,
            "profile_fingerprint": dict(profile_fingerprint),
            "profile_fingerprint_sha256": profile_fingerprint_sha256,
        }
    )
    atomic_torch_save(payload, path)


@dataclass(frozen=True)
class FingerprintedBinnedCache:
    """Cache locations and caller-owned formats for one binned profile."""

    baseline_path: Path
    baseline_format: str
    groups_path: Path
    groups_format: str
    profile_fingerprint: Mapping[str, Any]
    profile_fingerprint_sha256: str
    group_tensor_key: str = "ablated_means"
    completed_groups_key: str = "completed_groups"
    group_extra_metadata: Optional[Mapping[str, Any]] = None
    group_shard_pattern: Optional[str | Path] = None
    require_rank_metadata: bool = False


@dataclass(frozen=True)
class BinnedAblationProfile:
    baseline_mean: torch.Tensor
    baseline_stderr: torch.Tensor
    baseline_count: torch.Tensor
    ablated_means: Mapping[str, torch.Tensor]


def run_binned_ablation_profile(
    adapter: DiffusionAnalysisAdapter,
    dataloader: Iterable[Any],
    *,
    num_bins: int,
    ablation_mode: AblationMode,
    ablation_random_seed: int,
    progress_prefix: str = "Diffusion",
    progress_factory: Optional[
        Callable[[Iterable[Any], Optional[str]], Iterable[Any]]
    ] = None,
    cache: Optional[FingerprintedBinnedCache] = None,
    distributed_context: Optional[DistributedAnalysisContext] = None,
    checkpoint_interval_groups: int = 1,
) -> BinnedAblationProfile:
    """Run a baseline and every adapter group with resumable generic caches.

    In distributed mode each rank owns an independent adapter/model replica.
    The complete group catalog and fixed evaluation population stay identical;
    only globally missing analysis groups are round-robin sharded.  Rank cache
    files are merged from disk before assignment and after a final barrier, so
    restart behavior is independent of the current worker count.
    """

    if checkpoint_interval_groups <= 0:
        raise ValueError("checkpoint_interval_groups must be positive")
    context = distributed_context or singleton_analysis_context(
        next(adapter.model.parameters(), torch.empty(0)).device
    )
    group_items = tuple(adapter.named_analysis_groups())
    group_names = [name for name, _ in group_items]
    catalog_digest = group_catalog_sha256(group_names)
    assert_distributed_hash_invariant(context, "group_catalog", catalog_digest)
    if cache is not None:
        computed_profile_digest = assert_distributed_canonical_invariant(
            context,
            "profile_fingerprint_record",
            cache.profile_fingerprint,
        )
        declared_profile_digest = assert_distributed_hash_invariant(
            context,
            "profile_fingerprint_digest",
            cache.profile_fingerprint_sha256,
        )
        if computed_profile_digest != declared_profile_digest:
            raise ValueError("Profile fingerprint digest does not match its canonical record")

    def prepare_baseline() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cached_baseline = None
        if cache is not None:
            cached_baseline = load_binned_statistics_cache(
                cache.baseline_path,
                cache_format=cache.baseline_format,
                num_bins=num_bins,
                profile_fingerprint=cache.profile_fingerprint,
                profile_fingerprint_sha256=cache.profile_fingerprint_sha256,
            )
        if cached_baseline is None:
            baseline = evaluate_binned_losses(
                adapter,
                dataloader,
                num_bins=num_bins,
                ablation_mode=ablation_mode,
                progress_desc=f"{progress_prefix} baseline",
                progress_factory=progress_factory,
            )
            prepared = (baseline.mean(), baseline.stderr(), baseline.count.clone())
            if cache is not None:
                save_binned_statistics_cache(
                    cache.baseline_path,
                    cache_format=cache.baseline_format,
                    mean=prepared[0],
                    stderr=prepared[1],
                    count=prepared[2],
                    profile_fingerprint=cache.profile_fingerprint,
                    profile_fingerprint_sha256=cache.profile_fingerprint_sha256,
                )
            return prepared
        return cached_baseline

    if context.is_distributed:
        baseline_values = run_rank_zero_analysis_operation(
            context,
            f"{progress_prefix} baseline preparation",
            prepare_baseline,
        )
        if cache is not None:
            baseline_values = load_binned_statistics_cache(
                cache.baseline_path,
                cache_format=cache.baseline_format,
                num_bins=num_bins,
                profile_fingerprint=cache.profile_fingerprint,
                profile_fingerprint_sha256=cache.profile_fingerprint_sha256,
            )
            if baseline_values is None:
                raise RuntimeError(f"Rank-zero baseline cache was not published: {cache.baseline_path}")
    else:
        baseline_values = prepare_baseline()
    baseline_mean, baseline_stderr, baseline_count = baseline_values

    use_rank_shards = context.is_distributed or (
        cache is not None and cache.group_shard_pattern is not None
    )
    if use_rank_shards and cache is None:
        initial_values: Mapping[str, torch.Tensor] = {}
        local_values: dict[str, torch.Tensor] = {}
    elif use_rank_shards:
        assert cache is not None
        if cache.group_shard_pattern is None:
            raise ValueError("Distributed binned analysis requires cache.group_shard_pattern")
        current_path = cache.groups_path.resolve()
        absolute_pattern = os.path.abspath(str(cache.group_shard_pattern))
        if not fnmatch.fnmatch(str(current_path), absolute_pattern):
            raise ValueError(
                f"Rank cache path {current_path} does not match shard pattern {absolute_pattern}"
            )
        gathered_paths = _all_gather_analysis_objects(context, str(current_path))
        if len(set(str(path) for path in gathered_paths)) != context.world_size:
            raise ValueError(f"Distributed ranks must use distinct cache paths: {gathered_paths}")
        initial_merge = load_group_tensor_cache_shards(
            discover_group_tensor_cache_shards(absolute_pattern),
            cache_format=cache.groups_format,
            tensor_key=cache.group_tensor_key,
            completed_key=cache.completed_groups_key,
            group_names=group_names,
            num_bins=num_bins,
            profile_fingerprint=cache.profile_fingerprint,
            profile_fingerprint_sha256=cache.profile_fingerprint_sha256,
            group_catalog_digest=catalog_digest,
            require_rank_metadata=cache.require_rank_metadata or context.is_distributed,
        )
        claimed_path = next(
            (path for path, rank in initial_merge.shard_ranks.items() if rank == context.rank),
            None,
        )
        if claimed_path is not None and claimed_path != current_path:
            raise ValueError(
                f"Existing shard {claimed_path} already claims current rank {context.rank}; "
                f"refusing to overwrite {current_path}"
            )
        initial_values = initial_merge.values
        local_values = dict(initial_merge.shard_values.get(current_path, {}))
    else:
        initial_values = {}
        if cache is not None:
            initial_values = load_group_tensor_cache(
                cache.groups_path,
                cache_format=cache.groups_format,
                tensor_key=cache.group_tensor_key,
                completed_key=cache.completed_groups_key,
                group_names=group_names,
                num_bins=num_bins,
                profile_fingerprint=cache.profile_fingerprint,
                profile_fingerprint_sha256=cache.profile_fingerprint_sha256,
            )
        local_values = dict(initial_values)

    local_group_items = shard_missing_group_items(
        group_items,
        initial_values,
        rank=context.rank,
        world_size=context.world_size,
    )
    newly_completed = 0
    for global_index, (name, target) in local_group_items:
        statistics = evaluate_binned_losses(
            adapter,
            dataloader,
            num_bins=num_bins,
            ablate_target=target,
            ablation_mode=ablation_mode,
            ablation_random_seed=ablation_random_seed + global_index + 1,
            progress_desc=(
                f"{progress_prefix} rank {context.rank} ablation "
                f"{global_index + 1}/{len(group_items)} {name}"
            ),
            progress_factory=progress_factory,
        )
        local_values[name] = statistics.mean()
        newly_completed += 1
        if cache is not None and newly_completed % checkpoint_interval_groups == 0:
            extra_metadata = dict(cache.group_extra_metadata or {})
            if use_rank_shards:
                extra_metadata.update(
                    ranked_group_cache_metadata(
                        context,
                        group_catalog_digest=catalog_digest,
                    )
                )
            save_group_tensor_cache(
                cache.groups_path,
                cache_format=cache.groups_format,
                tensor_key=cache.group_tensor_key,
                completed_key=cache.completed_groups_key,
                values=local_values,
                num_bins=num_bins,
                profile_fingerprint=cache.profile_fingerprint,
                profile_fingerprint_sha256=cache.profile_fingerprint_sha256,
                extra_metadata=extra_metadata,
            )

    if cache is not None:
        extra_metadata = dict(cache.group_extra_metadata or {})
        if use_rank_shards:
            extra_metadata.update(
                ranked_group_cache_metadata(context, group_catalog_digest=catalog_digest)
            )
        save_group_tensor_cache(
            cache.groups_path,
            cache_format=cache.groups_format,
            tensor_key=cache.group_tensor_key,
            completed_key=cache.completed_groups_key,
            values=local_values,
            num_bins=num_bins,
            profile_fingerprint=cache.profile_fingerprint,
            profile_fingerprint_sha256=cache.profile_fingerprint_sha256,
            extra_metadata=extra_metadata,
        )

    if context.is_distributed:
        context.barrier()
    if use_rank_shards and cache is not None:
        final_merge = load_group_tensor_cache_shards(
            discover_group_tensor_cache_shards(os.path.abspath(str(cache.group_shard_pattern))),
            cache_format=cache.groups_format,
            tensor_key=cache.group_tensor_key,
            completed_key=cache.completed_groups_key,
            group_names=group_names,
            num_bins=num_bins,
            profile_fingerprint=cache.profile_fingerprint,
            profile_fingerprint_sha256=cache.profile_fingerprint_sha256,
            group_catalog_digest=catalog_digest,
            require_rank_metadata=cache.require_rank_metadata or context.is_distributed,
        )
        ablated_means = canonicalize_complete_group_results(
            final_merge.values,
            group_names,
            num_bins=num_bins,
        )
    elif context.is_distributed:
        gathered_values = _all_gather_analysis_objects(
            context,
            {name: value.detach().cpu() for name, value in local_values.items()},
        )
        merged_values: dict[str, torch.Tensor] = {}
        for part in gathered_values:
            if not isinstance(part, Mapping):
                raise ValueError("Distributed ablation result payload must be a mapping")
            for name, value in part.items():
                if name in merged_values and not torch.equal(merged_values[name], value):
                    raise ValueError(f"Conflicting distributed ablation result for {name!r}")
                merged_values[str(name)] = value
        ablated_means = canonicalize_complete_group_results(
            merged_values,
            group_names,
            num_bins=num_bins,
        )
    else:
        ablated_means = canonicalize_complete_group_results(
            local_values,
            group_names,
            num_bins=num_bins,
        )
    return BinnedAblationProfile(
        baseline_mean=baseline_mean,
        baseline_stderr=baseline_stderr,
        baseline_count=baseline_count,
        ablated_means=ablated_means,
    )


def infer_bin_axis_metadata(results: Mapping[str, Any], num_bins: int) -> Dict[str, Any]:
    if "sigma_bin_labels" in results:
        return {
            "kind": "noise_level",
            "bin_labels": list(results["sigma_bin_labels"]),
            "correlation_key": "C_noise_levels",
            "correlation_plot_name": "noise_level_correlation_heatmap.png",
            "correlation_title": "Noise-level correlation heatmap",
            "axis_title": "Noise-level bin",
            "row_hover_label": "noise_bin_y",
            "col_hover_label": "noise_bin_x",
        }
    if "timestep_bin_labels" in results:
        return {
            "kind": "timestep",
            "bin_labels": list(results["timestep_bin_labels"]),
            "correlation_key": "C_timesteps",
            "correlation_plot_name": "timestep_correlation_heatmap.png",
            "correlation_title": "Timestep correlation heatmap",
            "axis_title": "Timestep bin",
            "row_hover_label": "timestep_bin_y",
            "col_hover_label": "timestep_bin_x",
        }
    return {
        "kind": "bin",
        "bin_labels": [str(index) for index in range(num_bins)],
        "correlation_key": "C_bins",
        "correlation_plot_name": "bin_correlation_heatmap.png",
        "correlation_title": "Bin correlation heatmap",
        "axis_title": "Bin",
        "row_hover_label": "bin_y",
        "col_hover_label": "bin_x",
    }


def save_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n")


def _plot_series(
    values: torch.Tensor,
    out_path: str | Path,
    *,
    label: str,
    ylabel: str,
    axis_title: str,
    bin_labels: Sequence[str],
    stderr: Optional[torch.Tensor] = None,
) -> None:
    x = list(range(len(values)))
    y = values.detach().cpu().numpy()
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, label=label)
    if stderr is not None:
        error = stderr.detach().cpu().numpy()
        plt.fill_between(x, y - 1.96 * error, y + 1.96 * error, alpha=0.2)
    positions = _sparse_tick_positions(len(bin_labels))
    plt.xticks(positions, [bin_labels[index] for index in positions], rotation=90, fontsize=8)
    plt.xlabel(axis_title)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_binned_series(
    values: torch.Tensor,
    out_path: str | Path,
    *,
    label: str,
    ylabel: str,
    axis_title: str,
    bin_labels: Sequence[str],
    stderr: Optional[torch.Tensor] = None,
) -> None:
    """Render a modality-neutral scalar profile over ordered analysis bins."""

    _plot_series(
        values,
        out_path,
        label=label,
        ylabel=ylabel,
        axis_title=axis_title,
        bin_labels=bin_labels,
        stderr=stderr,
    )


def _sparse_tick_positions(num_items: int, max_ticks: int = 24) -> List[int]:
    if num_items <= max_ticks:
        return list(range(num_items))
    return list(range(0, num_items, math.ceil(num_items / max_ticks)))


def plot_labeled_heatmap(
    matrix: torch.Tensor,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    out_path: str | Path,
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
    array = matrix.detach().cpu().numpy()
    if max(len(row_labels), len(col_labels)) > 100:
        html_path = os.path.splitext(str(out_path))[0] + ".html"
        if go is None:
            print(f"plotly is not installed; skipping the interactive heatmap {html_path}")
            return
        figure = go.Figure(
            data=go.Heatmap(
                z=array,
                x=list(col_labels),
                y=list(row_labels),
                zmin=zmin,
                zmax=zmax,
                colorbar={"title": colorbar_title},
                hovertemplate=(
                    f"{row_hover_label}=%{{y}}<br>{col_hover_label}=%{{x}}<br>"
                    f"{hover_value_label}=%{{z}}<extra></extra>"
                ),
            )
        )
        figure.update_layout(
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            height=min(max(600, len(row_labels) * 14), 2400),
        )
        x_positions = _sparse_tick_positions(len(col_labels))
        y_positions = _sparse_tick_positions(len(row_labels))
        figure.update_xaxes(
            tickmode="array",
            tickvals=[col_labels[index] for index in x_positions],
            ticktext=[col_labels[index] for index in x_positions],
        )
        figure.update_yaxes(
            tickmode="array",
            tickvals=[row_labels[index] for index in y_positions],
            ticktext=[row_labels[index] for index in y_positions],
        )
        figure.write_html(html_path)
        return

    plt.figure(figsize=(12, max(4, min(20, 0.25 * len(row_labels)))))
    plt.imshow(array, aspect="auto", interpolation="nearest", vmin=zmin, vmax=zmax)
    plt.xticks(range(len(col_labels)), col_labels, rotation=90, fontsize=8)
    plt.yticks(range(len(row_labels)), row_labels, fontsize=8)
    plt.xlabel(xaxis_title)
    plt.ylabel(yaxis_title)
    plt.title(title)
    plt.colorbar(label=colorbar_title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_results_and_plots(
    *,
    output_dir: str | Path,
    results: Mapping[str, Any],
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
    names: Sequence[str],
    axis_metadata: Mapping[str, Any],
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "results.json", results)
    labels = [str(value) for value in axis_metadata["bin_labels"]]
    axis_title = str(axis_metadata["axis_title"])
    _plot_series(
        baseline_mean,
        output / "baseline_error.png",
        label="Baseline denoising MSE",
        ylabel="Error",
        axis_title=axis_title,
        bin_labels=labels,
        stderr=baseline_stderr,
    )
    _plot_series(
        raw_baseline_mean,
        output / "baseline_error_unweighted.png",
        label="Baseline unweighted denoising MSE",
        ylabel="Error",
        axis_title=axis_title,
        bin_labels=labels,
        stderr=raw_baseline_stderr,
    )
    _plot_series(
        p_eff,
        output / "effective_parameter_usage.png",
        label="Effective parameter usage",
        ylabel="P_eff(bin)",
        axis_title=axis_title,
        bin_labels=labels,
    )
    _plot_series(
        n_eff,
        output / "effective_group_count.png",
        label="Effective number of active groups",
        ylabel="N_eff(bin)",
        axis_title=axis_title,
        bin_labels=labels,
    )

    heatmaps = [
        (delta_stack, "delta_heatmap.png", "Ablation delta heatmap", "Delta", "delta", None, None),
        (
            relative_delta_stack,
            "relative_delta_heatmap.png",
            "Relative ablation delta heatmap",
            "Relative delta",
            "relative_delta",
            None,
            None,
        ),
        (
            row_normalized_relative_delta_stack,
            "row_normalized_relative_delta_heatmap.png",
            "Row-normalized relative ablation delta heatmap",
            "Row-normalized relative delta",
            "row_normalized_relative_delta",
            0.0,
            1.0,
        ),
    ]
    for matrix, filename, title, colorbar, hover, zmin, zmax in heatmaps:
        plot_labeled_heatmap(
            matrix=matrix,
            row_labels=names,
            col_labels=labels,
            out_path=output / filename,
            title=title,
            xaxis_title=axis_title,
            yaxis_title="Group",
            colorbar_title=colorbar,
            row_hover_label="group",
            col_hover_label="bin",
            hover_value_label=hover,
            zmin=zmin,
            zmax=zmax,
        )

    positive = relative_delta_stack[relative_delta_stack > 0]
    if positive.numel() > 0:
        plot_labeled_heatmap(
            matrix=relative_delta_stack,
            row_labels=names,
            col_labels=labels,
            out_path=output / "relative_delta_heatmap_clipped.png",
            title="Relative ablation delta heatmap (clipped)",
            xaxis_title=axis_title,
            yaxis_title="Group",
            colorbar_title="Relative delta",
            row_hover_label="group",
            col_hover_label="bin",
            hover_value_label="relative_delta",
            zmin=0.0,
            zmax=float(torch.quantile(positive, 0.99).item()),
        )
    if C_groups is not None:
        plot_labeled_heatmap(
            matrix=C_groups,
            row_labels=names,
            col_labels=names,
            out_path=output / "group_correlation_heatmap.png",
            title="Group correlation heatmap",
            xaxis_title="Group",
            yaxis_title="Group",
            colorbar_title="Correlation",
            row_hover_label="group_y",
            col_hover_label="group_x",
            hover_value_label="correlation",
            zmin=-1.0,
            zmax=1.0,
        )
    plot_labeled_heatmap(
        matrix=C_levels,
        row_labels=labels,
        col_labels=labels,
        out_path=output / str(axis_metadata["correlation_plot_name"]),
        title=str(axis_metadata["correlation_title"]),
        xaxis_title=axis_title,
        yaxis_title=axis_title,
        colorbar_title="Correlation",
        row_hover_label=str(axis_metadata["row_hover_label"]),
        col_hover_label=str(axis_metadata["col_hover_label"]),
        hover_value_label="correlation",
        zmin=-1.0,
        zmax=1.0,
    )
