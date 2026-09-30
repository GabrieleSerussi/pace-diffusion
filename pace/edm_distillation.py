"""EDM student construction and smoke distillation helpers.

The helpers in this module use the analysis artifacts produced by
``scripts/evaluate_parameters_edm.py`` as the source of truth for student
capacity targets.  They intentionally keep the PACE-specific distillation
logic inside this repository while reusing the local NVLabs EDM implementation
for primitive layers and checkpoint compatibility.  The NVLabs EDM checkout is
located by :func:`pace.external_repos.ensure_edm_importable`.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import os
import pickle
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.nn.functional import silu

from .teacher_models import (
    TeacherSpec,
    load_teacher_network,
    resolve_teacher_spec,
    teacher_spec_from_plan,
    teacher_spec_from_results,
)
from .profile_provenance import (
    provenance_from_results,
    validate_matching_source_profile,
)
from .filter_sampling import aligned_group_expansion_weights
from .evaluation_protocols import file_digest
from .external_repos import ensure_edm_importable as _ensure_edm_checkout
from .jsonio import find_json_file, load_json as _load_json_file


SUPPORTED_DISTILLATION_VARIANTS: tuple[str, ...] = (
    "global",
    "uniform_blockwise",
    "blockwise_capacity",
    "shuffled_capacity",
    "layerwise_capacity",
    "reversed_layerwise_capacity",
    "combined_blockwise",
    "combined_layerwise",
)

# The four allocation variants compared in the paper (Section 3.4, Table 2).
PAPER_DISTILLATION_VARIANTS: tuple[str, ...] = (
    "global",
    "uniform_blockwise",
    "combined_blockwise",
    "combined_layerwise",
)

# Kept for the older capacity-control variants; not the paper default.
LEGACY_DISTILLATION_VARIANTS: tuple[str, ...] = (
    "global",
    "uniform_blockwise",
    "blockwise_capacity",
    "shuffled_capacity",
    "layerwise_capacity",
    "reversed_layerwise_capacity",
)

DEFAULT_DISTILLATION_VARIANTS: tuple[str, ...] = PAPER_DISTILLATION_VARIANTS

DEFAULT_CHANNEL_MULT: tuple[int, ...] = (2, 2, 2)
DEFAULT_CHANNEL_CANDIDATES: tuple[int, ...] = tuple(range(8, 129, 8))
LARGE_EDM_CHANNEL_CANDIDATES: tuple[int, ...] = tuple(range(8, 385, 8))
EXTRA_UNIFORM_ARCH_CANDIDATES: tuple[tuple[int, tuple[int, int, int]], ...] = (
    (192, (1, 3, 2)),
    (128, (3, 2, 1)),
    (128, (1, 1, 3)),
    (128, (1, 3, 1)),
    (96, (1, 3, 3)),
    (128, (1, 2, 1)),
    (128, (1, 1, 2)),
    (96, (1, 3, 1)),
    (96, (1, 1, 3)),
    (24, (3, 3, 1)),
    (24, (3, 1, 3)),
    (32, (3, 1, 1)),
    (24, (2, 3, 2)),
    (24, (2, 2, 3)),
    (40, (1, 1, 2)),
    (40, (1, 2, 1)),
    (32, (1, 3, 1)),
    (32, (1, 1, 3)),
    (56, (1, 1, 1)),
)

_EDM_SYMBOLS: Optional[dict[str, Any]] = None
_UNIFORM_STUDENT_CACHE: dict[tuple[Any, ...], StudentArchitectureReport] = {}
_UNIFORM_ARCHITECTURE_CANDIDATE_CACHE: dict[
    tuple[Any, ...], tuple[tuple[int, tuple[int, ...], int, int], ...]
] = {}


def ensure_edm_importable(edm_root: str | Path | None = None) -> None:
    """Put the NVLabs EDM checkout on ``sys.path`` if needed.

    The checkout is taken from ``edm_root``, ``$EDM_REPO``, a sibling ``edm``
    folder or ``PYTHONPATH``; a clear error explains the setup otherwise.
    """

    _ensure_edm_checkout(edm_root)


def _edm_symbols() -> dict[str, Any]:
    global _EDM_SYMBOLS
    if _EDM_SYMBOLS is None:
        ensure_edm_importable()
        from training.networks import (  # type: ignore[import-not-found]
            Conv2d,
            FourierEmbedding,
            GroupNorm,
            Linear,
            PositionalEmbedding,
            UNetBlock,
        )

        try:
            from torch_utils import persistence  # type: ignore[import-not-found]
        except Exception:  # pragma: no cover - only used in unusual import envs.
            persistence = None

        _EDM_SYMBOLS = {
            "Conv2d": Conv2d,
            "FourierEmbedding": FourierEmbedding,
            "GroupNorm": GroupNorm,
            "Linear": Linear,
            "PositionalEmbedding": PositionalEmbedding,
            "UNetBlock": UNetBlock,
            "persistence": persistence,
        }
    return _EDM_SYMBOLS


def round_channels(value: float, *, multiple: int = 8, minimum: int = 8, maximum: int | None = None) -> int:
    """Round a hidden width to architecture-friendly channel counts."""

    if not math.isfinite(float(value)):
        raise ValueError(f"channel value must be finite, got {value!r}")
    rounded = int(round(float(value) / multiple) * multiple)
    rounded = max(minimum, rounded)
    if maximum is not None:
        rounded = min(maximum, rounded)
    return int(rounded)


def edm_group_norm_groups(num_channels: int) -> int:
    """Return the group count used by NVLabs EDM ``GroupNorm``."""

    return min(32, int(num_channels) // 4)


def safe_group_norm_groups(num_channels: int, *, max_groups: int = 32, min_channels_per_group: int = 4) -> int:
    """Choose a valid group count for arbitrary narrow/concatenated widths."""

    upper = min(max_groups, max(1, int(num_channels) // min_channels_per_group))
    for groups in range(upper, 0, -1):
        if int(num_channels) % groups == 0:
            return groups
    return 1


def is_edm_group_norm_safe(num_channels: int) -> bool:
    """Whether EDM's fixed ``GroupNorm`` implementation accepts this width."""

    groups = edm_group_norm_groups(num_channels)
    return groups > 0 and int(num_channels) % groups == 0


def make_group_norm_modules_safe(module: torch.nn.Module) -> None:
    """Patch EDM GroupNorm modules to use valid group counts for their channels."""

    for child in module.modules():
        if not hasattr(child, "num_groups"):
            continue
        weight = getattr(child, "weight", None)
        if not isinstance(weight, torch.nn.Parameter):
            continue
        num_channels = int(weight.numel())
        groups = int(getattr(child, "num_groups"))
        if groups <= 0 or num_channels % groups != 0:
            child.num_groups = safe_group_norm_groups(num_channels)


def round_group_norm_safe_channels(
    value: float,
    *,
    multiple: int = 8,
    minimum: int = 8,
    maximum: int | None = None,
) -> int:
    """Round a hidden width while preserving EDM ``GroupNorm`` divisibility.

    EDM's ``GroupNorm`` uses ``min(32, channels // 4)`` groups.  Widths below
    128 are safe when divisible by 4, but widths above that must also be
    divisible by 32.  PACE uses this helper for all hidden student widths.
    """

    rounded = round_channels(value, multiple=multiple, minimum=minimum, maximum=maximum)
    if is_edm_group_norm_safe(rounded):
        return rounded

    limit = maximum if maximum is not None else max(rounded * 2, minimum)
    candidates = [
        width
        for width in range(minimum, int(limit) + 1, multiple)
        if is_edm_group_norm_safe(width)
    ]
    if not candidates:
        raise ValueError(
            f"No EDM GroupNorm-safe channel width found for value={value} "
            f"within [{minimum}, {limit}]"
        )
    return min(candidates, key=lambda width: (abs(width - float(value)), -width))


def is_uniform_architecture_safe(model_channels: int, channel_mult: Sequence[int] = DEFAULT_CHANNEL_MULT) -> bool:
    """Check the hidden widths produced by a SongUNet channel schedule."""

    widths = {int(model_channels)}
    widths.update(int(model_channels) * int(mult) for mult in channel_mult)
    return all(is_edm_group_norm_safe(width) for width in widths)


def strip_filter_suffix(group_name: str) -> str:
    if ".filter_" not in group_name:
        return group_name
    return group_name.rsplit(".filter_", 1)[0]


def module_to_structural_key(module_name: str) -> str:
    """Map a per-filter module name to the structural width it controls."""

    name = strip_filter_suffix(module_name)
    if name.startswith("model."):
        name = name[len("model.") :]
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] in {"enc", "dec"}:
        parent = f"{parts[0]}.{parts[1]}"
        if parts[2] in {"conv0", "conv1", "skip", "qkv", "proj"}:
            return parent
    return name


def structural_width_from_module(module_name: str, out_channels: int) -> int:
    if module_name.endswith(".qkv"):
        return max(1, int(out_channels) // 3)
    return int(out_channels)


def _is_conv_like_module(module: torch.nn.Module) -> bool:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim < 3:
        return False
    class_name = module.__class__.__name__.lower()
    return "conv" in class_name or hasattr(module, "up") or hasattr(module, "down")


def count_filter_parameters(module: torch.nn.Module, filter_idx: int) -> int:
    weight = getattr(module, "weight", None)
    if weight is None:
        return 0
    count = int(weight[filter_idx].numel())
    bias = getattr(module, "bias", None)
    if bias is not None:
        count += 1
    return count


def count_grouped_parameters(model: torch.nn.Module) -> int:
    """Count parameters with the same per-filter proxy used by the analysis run."""

    total = 0
    for module in model.modules():
        if not _is_conv_like_module(module):
            continue
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            continue
        for filter_idx in range(int(weight.shape[0])):
            total += count_filter_parameters(module, filter_idx)
    return int(total)


def count_full_parameters(model: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def construct_student_for_counting(model_kwargs: Mapping[str, Any]) -> torch.nn.Module:
    """Construct a storage-free meta student for architecture budget searches."""

    with torch.device("meta"):
        return construct_student_from_kwargs(model_kwargs)


def make_class_labels(
    labels: torch.Tensor,
    *,
    label_dim: int,
    device: torch.device,
) -> torch.Tensor | None:
    if label_dim <= 0:
        return None
    labels = labels.to(device=device, dtype=torch.long)
    out = torch.zeros(labels.shape[0], label_dim, device=device, dtype=torch.float32)
    out.scatter_(1, labels.reshape(-1, 1), 1.0)
    return out


def mse_per_example(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, pred.ndim))
    return ((pred - target) ** 2).mean(dim=dims)


def loss_weights_for_family(sigmas: torch.Tensor, model_family: str, sigma_data: float) -> torch.Tensor:
    if model_family in {"vp", "ve"}:
        return 1.0 / (sigmas**2)
    return (sigmas**2 + sigma_data**2) / ((sigmas * sigma_data) ** 2)


class NarrowSongUNet(torch.nn.Module):
    """Variable-width SongUNet compatible with the EDM preconditioning wrapper."""

    def __init__(
        self,
        img_resolution: int,
        in_channels: int,
        out_channels: int,
        label_dim: int = 0,
        augment_dim: int = 0,
        model_channels: int = 128,
        channel_mult: Sequence[int] = DEFAULT_CHANNEL_MULT,
        channel_mult_emb: int = 4,
        num_blocks: int = 4,
        attn_resolutions: Sequence[int] = (16,),
        dropout: float = 0.10,
        label_dropout: float = 0,
        embedding_type: str = "positional",
        channel_mult_noise: int = 1,
        encoder_type: str = "standard",
        decoder_type: str = "standard",
        resample_filter: Sequence[int] = (1, 1),
        width_profile: Mapping[str, int] | None = None,
    ):
        super().__init__()
        if embedding_type not in {"fourier", "positional"}:
            raise ValueError(f"Unsupported embedding_type={embedding_type!r}")
        if encoder_type != "standard" or decoder_type != "standard":
            raise ValueError("NarrowSongUNet currently supports standard DDPM++ encoder/decoder only")

        symbols = _edm_symbols()
        Conv2d = symbols["Conv2d"]
        FourierEmbedding = symbols["FourierEmbedding"]
        Linear = symbols["Linear"]
        PositionalEmbedding = symbols["PositionalEmbedding"]
        UNetBlock = symbols["UNetBlock"]

        self.label_dropout = label_dropout
        self.width_profile = {str(k).removeprefix("model."): int(v) for k, v in (width_profile or {}).items()}
        self._unet_block_cls = UNetBlock

        emb_channels = int(model_channels) * int(channel_mult_emb)
        noise_channels = int(model_channels) * int(channel_mult_noise)
        init = dict(init_mode="xavier_uniform")
        init_zero = dict(init_mode="xavier_uniform", init_weight=1e-5)
        init_attn = dict(init_mode="xavier_uniform", init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels,
            num_heads=1,
            dropout=dropout,
            skip_scale=np.sqrt(0.5),
            eps=1e-6,
            resample_filter=list(resample_filter),
            resample_proj=True,
            adaptive_scale=False,
            init=init,
            init_zero=init_zero,
            init_attn=init_attn,
        )

        self.map_noise = (
            PositionalEmbedding(num_channels=noise_channels, endpoint=True)
            if embedding_type == "positional"
            else FourierEmbedding(num_channels=noise_channels)
        )
        self.map_label = Linear(in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)

        self.enc = torch.nn.ModuleDict()
        cout = int(in_channels)
        for level, mult in enumerate(channel_mult):
            res = int(img_resolution) >> level
            if level == 0:
                cin = cout
                cout = self._width(f"enc.{res}x{res}_conv", int(model_channels))
                self.enc[f"{res}x{res}_conv"] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                down_key = f"enc.{res}x{res}_down"
                down_out = self._width(down_key, cout)
                self.enc[f"{res}x{res}_down"] = UNetBlock(in_channels=cout, out_channels=down_out, down=True, **block_kwargs)
                cout = down_out
            for idx in range(num_blocks):
                cin = cout
                block_key = f"enc.{res}x{res}_block{idx}"
                cout = self._width(block_key, int(model_channels) * int(mult))
                attn = res in attn_resolutions
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin,
                    out_channels=cout,
                    attention=attn,
                    **block_kwargs,
                )
        skips = [block.out_channels for _, block in self.enc.items()]

        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = int(img_resolution) >> level
            if level == len(channel_mult) - 1:
                in0_out = self._width(f"dec.{res}x{res}_in0", cout)
                self.dec[f"{res}x{res}_in0"] = UNetBlock(in_channels=cout, out_channels=in0_out, attention=True, **block_kwargs)
                cout = in0_out
                in1_out = self._width(f"dec.{res}x{res}_in1", cout)
                self.dec[f"{res}x{res}_in1"] = UNetBlock(in_channels=cout, out_channels=in1_out, **block_kwargs)
                cout = in1_out
            else:
                up_out = self._width(f"dec.{res}x{res}_up", cout)
                self.dec[f"{res}x{res}_up"] = UNetBlock(in_channels=cout, out_channels=up_out, up=True, **block_kwargs)
                cout = up_out
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                block_key = f"dec.{res}x{res}_block{idx}"
                cout = self._width(block_key, int(model_channels) * int(mult))
                attn = idx == num_blocks and res in attn_resolutions
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin,
                    out_channels=cout,
                    attention=attn,
                    **block_kwargs,
                )
            if level == 0:
                GroupNorm = symbols["GroupNorm"]
                self.dec[f"{res}x{res}_aux_norm"] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f"{res}x{res}_aux_conv"] = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)
        make_group_norm_modules_safe(self)

    def _width(self, key: str, default: int) -> int:
        return int(self.width_profile.get(key, default))

    def forward(
        self,
        x: torch.Tensor,
        noise_labels: torch.Tensor,
        class_labels: torch.Tensor | None,
        augment_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        if self.map_label is not None:
            if class_labels is None:
                class_labels = torch.zeros([x.shape[0], self.map_label.in_features], device=x.device, dtype=torch.float32)
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        skips: list[torch.Tensor] = []
        for block in self.enc.values():
            x = block(x, emb) if isinstance(block, self._unet_block_cls) else block(x)
            skips.append(x)

        aux = None
        tmp = None
        for name, block in self.dec.items():
            if "aux_norm" in name:
                tmp = block(x)
            elif "aux_conv" in name:
                if tmp is None:
                    raise RuntimeError("aux_norm must run before aux_conv")
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        if aux is None:
            raise RuntimeError("NarrowSongUNet decoder did not produce an output")
        return aux


class NarrowEDMPrecond(torch.nn.Module):
    """EDM preconditioning wrapper around ``NarrowSongUNet``."""

    def __init__(
        self,
        img_resolution: int,
        img_channels: int,
        label_dim: int = 0,
        use_fp16: bool = False,
        sigma_min: float = 0,
        sigma_max: float = float("inf"),
        sigma_data: float = 0.5,
        model_type: str = "NarrowSongUNet",
        **model_kwargs: Any,
    ):
        super().__init__()
        if model_type not in {"NarrowSongUNet", "SongUNet"}:
            raise ValueError(f"NarrowEDMPrecond supports NarrowSongUNet/SongUNet, got {model_type!r}")
        self.img_resolution = int(img_resolution)
        self.img_channels = int(img_channels)
        self.label_dim = int(label_dim)
        self.use_fp16 = bool(use_fp16)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.sigma_data = float(sigma_data)
        self.model_family = "edm"
        self.model = NarrowSongUNet(
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels,
            label_dim=label_dim,
            **model_kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        class_labels: torch.Tensor | None = None,
        force_fp32: bool = False,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        if self.label_dim == 0:
            class_labels = None
        elif class_labels is None:
            class_labels = torch.zeros([x.shape[0], self.label_dim], device=x.device)
        else:
            class_labels = class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == "cuda") else torch.float32

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels=class_labels, **model_kwargs)
        return c_skip * x + c_out * F_x.to(torch.float32)

    def round_sigma(self, sigma: torch.Tensor | float) -> torch.Tensor:
        return torch.as_tensor(sigma)


class NarrowVPPrecond(NarrowEDMPrecond):
    """VP preconditioning wrapper for narrowed DDPM++/SongUNet students."""

    def __init__(
        self,
        img_resolution: int,
        img_channels: int,
        label_dim: int = 0,
        use_fp16: bool = False,
        beta_d: float = 19.9,
        beta_min: float = 0.1,
        M: int = 1000,
        epsilon_t: float = 1e-5,
        model_type: str = "NarrowSongUNet",
        **model_kwargs: Any,
    ):
        super().__init__(
            img_resolution=img_resolution,
            img_channels=img_channels,
            label_dim=label_dim,
            use_fp16=use_fp16,
            model_type=model_type,
            **model_kwargs,
        )
        self.beta_d = float(beta_d)
        self.beta_min = float(beta_min)
        self.M = int(M)
        self.epsilon_t = float(epsilon_t)
        self.sigma_min = math.sqrt(
            math.exp(0.5 * self.beta_d * self.epsilon_t**2 + self.beta_min * self.epsilon_t) - 1
        )
        self.sigma_max = math.sqrt(math.exp(0.5 * self.beta_d + self.beta_min) - 1)
        self.sigma_data = 1.0
        self.model_family = "vp"

    def sigma(self, t: torch.Tensor | float) -> torch.Tensor:
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * t.square() + self.beta_min * t).exp() - 1).sqrt()

    def sigma_inv(self, sigma: torch.Tensor | float) -> torch.Tensor:
        sigma = torch.as_tensor(sigma)
        return (
            (self.beta_min**2 + 2 * self.beta_d * (1 + sigma.square()).log()).sqrt()
            - self.beta_min
        ) / self.beta_d

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        class_labels: torch.Tensor | None = None,
        force_fp32: bool = False,
        **model_kwargs: Any,
    ) -> torch.Tensor:
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        if self.label_dim == 0:
            class_labels = None
        elif class_labels is None:
            class_labels = torch.zeros([x.shape[0], self.label_dim], device=x.device)
        else:
            class_labels = class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == "cuda") else torch.float32
        c_in = 1 / (sigma.square() + 1).sqrt()
        c_noise = (self.M - 1) * self.sigma_inv(sigma)
        prediction = self.model(
            (c_in * x).to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )
        return x - sigma * prediction.to(torch.float32)


class BlockwiseEDMStudent(torch.nn.Module):
    """Dispatch inputs to one student per timestep/sigma block."""

    def __init__(
        self,
        students: Sequence[torch.nn.Module],
        timestep_blocks: Sequence[Sequence[int]],
        *,
        sigma_values: Sequence[float] | torch.Tensor,
        num_sigma_bins: int,
    ):
        super().__init__()
        if not students:
            raise ValueError("students must not be empty")
        if len(students) != len(timestep_blocks):
            raise ValueError("students and timestep_blocks must have the same length")
        self.students = torch.nn.ModuleList(students)
        self.timestep_blocks = [(int(start), int(end)) for start, end in timestep_blocks]
        self.num_sigma_bins = int(num_sigma_bins)
        self.register_buffer("sigma_values", torch.as_tensor(sigma_values, dtype=torch.float32), persistent=False)
        first = students[0]
        self.img_resolution = int(getattr(first, "img_resolution"))
        self.img_channels = int(getattr(first, "img_channels"))
        self.label_dim = int(getattr(first, "label_dim", 0))
        self.sigma_min = float(getattr(first, "sigma_min", 0))
        self.sigma_max = float(getattr(first, "sigma_max", float("inf")))
        self.sigma_data = float(getattr(first, "sigma_data", 0.5))

    def sigma_to_bin_ids(self, sigma: torch.Tensor) -> torch.Tensor:
        if self.sigma_values.numel() == 0:
            return torch.zeros_like(sigma, dtype=torch.long)
        schedule = self.sigma_values.to(device=sigma.device, dtype=torch.float32)
        sigma_flat = sigma.to(torch.float32).flatten().clamp_min(1e-12)
        distances = (sigma_flat.log().unsqueeze(1) - schedule.clamp_min(1e-12).log().unsqueeze(0)).abs()
        sigma_indices = distances.argmin(dim=1)
        bins = torch.floor(sigma_indices.float() * self.num_sigma_bins / schedule.numel()).long()
        return bins.clamp(min=0, max=self.num_sigma_bins - 1)

    def block_ids_for_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        bins = self.sigma_to_bin_ids(sigma)
        block_ids = torch.zeros_like(bins)
        for index, (start, end) in enumerate(self.timestep_blocks):
            block_ids[(bins >= start) & (bins < end)] = index
        return block_ids

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, class_labels: torch.Tensor | None = None, **model_kwargs: Any) -> torch.Tensor:
        if len(self.students) == 1:
            return self.students[0](x, sigma, class_labels, **model_kwargs)

        sigma_flat = sigma.flatten()
        block_ids = self.block_ids_for_sigma(sigma_flat)
        output = torch.empty_like(x, dtype=torch.float32)
        for block_index, student in enumerate(self.students):
            mask = block_ids == block_index
            if not bool(mask.any()):
                continue
            sub_kwargs = {}
            for key, value in model_kwargs.items():
                if torch.is_tensor(value) and value.shape[:1] == x.shape[:1]:
                    sub_kwargs[key] = value[mask]
                else:
                    sub_kwargs[key] = value
            sub_labels = class_labels[mask] if class_labels is not None else None
            output[mask] = student(x[mask], sigma_flat[mask], sub_labels, **sub_kwargs)
        return output

    def round_sigma(self, sigma: torch.Tensor | float) -> torch.Tensor:
        return torch.as_tensor(sigma)


@dataclass
class StudentArchitectureReport:
    student_index: int
    timestep_block: list[int]
    target_grouped_budget: float
    realized_grouped_budget: int
    relative_mismatch: float
    full_parameter_count: int
    model_kwargs: dict[str, Any]
    width_profile: dict[str, int] = field(default_factory=dict)
    layerwise_target_sum: float | None = None
    rounding: dict[str, Any] = field(default_factory=dict)


@dataclass
class VariantArchitecturePlan:
    variant: str
    source_allocation_path: str
    source_allocation_sha256: str
    allocation_rule: dict[str, Any]
    grouping_identity: dict[str, Any]
    timestep_blocks: list[list[int]]
    students: list[StudentArchitectureReport]
    shuffle_seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "source_allocation_path": self.source_allocation_path,
            "source_allocation_sha256": self.source_allocation_sha256,
            "allocation_rule": self.allocation_rule,
            "grouping_identity": self.grouping_identity,
            "timestep_blocks": self.timestep_blocks,
            "shuffle_seed": self.shuffle_seed,
            "students": [asdict(student) for student in self.students],
        }


def default_model_kwargs(
    *,
    model_channels: int,
    label_dim: int,
    img_resolution: int,
    img_channels: int = 3,
    dropout: float = 0.13,
    width_profile: Mapping[str, int] | None = None,
    channel_mult: Sequence[int] = DEFAULT_CHANNEL_MULT,
    num_blocks: int = 4,
    attn_resolutions: Sequence[int] = (16,),
    model_family: str = "edm",
    preconditioning: Mapping[str, Any] | None = None,
    augment_dim: int = 0,
) -> dict[str, Any]:
    family = str(model_family).lower()
    if family not in {"edm", "vp"}:
        raise ValueError(f"Narrow students currently support edm or vp preconditioning, got {model_family!r}")
    kwargs = {
        "img_resolution": int(img_resolution),
        "img_channels": int(img_channels),
        "label_dim": int(label_dim),
        "augment_dim": int(augment_dim),
        "model_type": "NarrowSongUNet",
        "preconditioning": family,
        "embedding_type": "positional",
        "encoder_type": "standard",
        "decoder_type": "standard",
        "channel_mult_noise": 1,
        "resample_filter": [1, 1],
        "model_channels": int(model_channels),
        "channel_mult": [int(value) for value in channel_mult],
        "num_blocks": int(num_blocks),
        "attn_resolutions": [int(value) for value in attn_resolutions],
        "dropout": float(dropout),
        "use_fp16": False,
        "width_profile": dict(width_profile or {}),
    }
    kwargs.update(dict(preconditioning or {}))
    return kwargs


def default_proxy_model_channels(*, label_dim: int, img_resolution: int) -> int:
    return 288 if int(label_dim) >= 1000 or int(img_resolution) >= 64 else 128


def construct_student_from_kwargs(model_kwargs: Mapping[str, Any]) -> NarrowEDMPrecond:
    kwargs = dict(model_kwargs)
    preconditioning = str(kwargs.pop("preconditioning", "edm")).lower()
    wrapper = NarrowVPPrecond if preconditioning == "vp" else NarrowEDMPrecond
    if preconditioning not in {"edm", "vp"}:
        raise ValueError(f"Unsupported narrow student preconditioning {preconditioning!r}")
    return wrapper(**kwargs)


def infer_student_topology(results: Mapping[str, Any]) -> dict[str, Any]:
    """Return the teacher-matched serialized topology for narrowed students."""

    model_info = results.get("model_info", {})
    model_config = dict(model_info.get("model_config") or {})
    teacher = model_info.get("teacher") or results.get("teacher") or results.get("config", {}).get("teacher")
    if isinstance(teacher, Mapping):
        model_config = {**dict(teacher.get("model_config") or {}), **model_config}
    topology = dict(model_config.get("student_topology") or {})
    if not topology:
        topology = {
            "channel_mult": model_config.get("channel_mult", DEFAULT_CHANNEL_MULT),
            "num_blocks": model_config.get("num_blocks", model_config.get("num_res_blocks", 4)),
            "attn_resolutions": model_config.get(
                "attn_resolutions", model_config.get("attention_resolutions", [16])
            ),
            "dropout": model_config.get("dropout", 0.13),
        }
    resolved = {
        "channel_mult": [int(value) for value in topology.get("channel_mult", DEFAULT_CHANNEL_MULT)],
        "num_blocks": int(topology.get("num_blocks", 4)),
        "attn_resolutions": [int(value) for value in topology.get("attn_resolutions", [16])],
        "dropout": float(topology.get("dropout", 0.13)),
        "augment_dim": int(topology.get("augment_dim", model_config.get("augment_dim", 0))),
        "model_family": infer_model_family(results),
        "preconditioning": {
            key: model_config[key]
            for key in ("sigma_min", "sigma_max", "sigma_data", "beta_d", "beta_min", "M", "epsilon_t")
            if key in model_config
        },
    }
    teacher_model_channels = topology.get("model_channels", model_config.get("model_channels"))
    if teacher_model_channels is not None:
        resolved["model_channels"] = int(teacher_model_channels)
    return resolved


def build_uniform_candidate_table(
    *,
    label_dim: int,
    img_resolution: int,
    img_channels: int = 3,
    candidates: Iterable[int] | None = None,
    topology: Mapping[str, Any] | None = None,
) -> list[tuple[int, int, int]]:
    topology = dict(topology or {})
    channel_mult = tuple(int(value) for value in topology.get("channel_mult", DEFAULT_CHANNEL_MULT))
    if candidates is None:
        candidates = LARGE_EDM_CHANNEL_CANDIDATES if int(label_dim) >= 1000 or int(img_resolution) >= 64 else DEFAULT_CHANNEL_CANDIDATES
    teacher_model_channels = topology.get("model_channels")
    table: list[tuple[int, int, int]] = []
    for model_channels in candidates:
        if teacher_model_channels is not None and int(model_channels) > int(teacher_model_channels):
            continue
        if not is_uniform_architecture_safe(int(model_channels), channel_mult):
            continue
        kwargs = default_model_kwargs(
            model_channels=int(model_channels),
            label_dim=label_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            channel_mult=channel_mult,
            num_blocks=int(topology.get("num_blocks", 4)),
            attn_resolutions=topology.get("attn_resolutions", [16]),
            dropout=float(topology.get("dropout", 0.13)),
            model_family=str(topology.get("model_family", "edm")),
            preconditioning=topology.get("preconditioning", {}),
            augment_dim=int(topology.get("augment_dim", 0)),
        )
        model = construct_student_for_counting(kwargs)
        table.append((int(model_channels), count_grouped_parameters(model), count_full_parameters(model)))
        del model
    return table


def compile_uniform_student(
    *,
    student_index: int,
    timestep_block: Sequence[int],
    target_grouped_budget: float,
    label_dim: int,
    img_resolution: int,
    img_channels: int = 3,
    candidate_table: Sequence[tuple[int, int, int]] | None = None,
    budget_tolerance: float = 0.05,
    topology: Mapping[str, Any] | None = None,
) -> StudentArchitectureReport:
    topology = dict(topology or {})
    cache_key = (
        round(float(target_grouped_budget), 3),
        int(label_dim),
        int(img_resolution),
        int(img_channels),
        round(float(budget_tolerance), 6),
        tuple(candidate_table or ()),
        json.dumps(topology, sort_keys=True),
    )
    cached = _UNIFORM_STUDENT_CACHE.get(cache_key)
    if cached is not None:
        report = copy.deepcopy(cached)
        report.student_index = int(student_index)
        report.timestep_block = [int(timestep_block[0]), int(timestep_block[1])]
        return report

    table = list(candidate_table or build_uniform_candidate_table(label_dim=label_dim, img_resolution=img_resolution, img_channels=img_channels, topology=topology))
    best = min(table, key=lambda item: abs(item[1] - target_grouped_budget))
    model_channels, realized_grouped, full_count = best
    relative_mismatch = abs(float(realized_grouped) - float(target_grouped_budget)) / float(target_grouped_budget)
    if relative_mismatch > budget_tolerance:
        extra_candidate = _compile_extra_arch_candidate_student(
            student_index=student_index,
            timestep_block=timestep_block,
            target_grouped_budget=target_grouped_budget,
            label_dim=label_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            budget_tolerance=budget_tolerance,
            topology=topology,
        )
        if extra_candidate is not None:
            _UNIFORM_STUDENT_CACHE[cache_key] = copy.deepcopy(extra_candidate)
            return extra_candidate
        structural = _compile_structural_uniform_student(
            student_index=student_index,
            timestep_block=timestep_block,
            target_grouped_budget=target_grouped_budget,
            label_dim=label_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            budget_tolerance=budget_tolerance,
            topology=topology,
        )
        _UNIFORM_STUDENT_CACHE[cache_key] = copy.deepcopy(structural)
        return structural
    kwargs = default_model_kwargs(
        model_channels=model_channels,
        label_dim=label_dim,
        img_resolution=img_resolution,
        img_channels=img_channels,
        channel_mult=topology.get("channel_mult", DEFAULT_CHANNEL_MULT),
        num_blocks=int(topology.get("num_blocks", 4)),
        attn_resolutions=topology.get("attn_resolutions", [16]),
        dropout=float(topology.get("dropout", 0.13)),
        model_family=str(topology.get("model_family", "edm")),
        preconditioning=topology.get("preconditioning", {}),
        augment_dim=int(topology.get("augment_dim", 0)),
    )
    report = StudentArchitectureReport(
        student_index=int(student_index),
        timestep_block=[int(timestep_block[0]), int(timestep_block[1])],
        target_grouped_budget=float(target_grouped_budget),
        realized_grouped_budget=int(realized_grouped),
        relative_mismatch=float(relative_mismatch),
        full_parameter_count=int(full_count),
        model_kwargs=kwargs,
        width_profile={},
        rounding={
            "mode": "uniform_model_channels",
            "channel_divisibility": 8,
            "selected_model_channels": int(model_channels),
        },
    )
    _UNIFORM_STUDENT_CACHE[cache_key] = copy.deepcopy(report)
    return report


def _uniform_architecture_specs(topology: Mapping[str, Any]) -> list[tuple[int, tuple[int, ...]]]:
    """Enumerate deterministic width schedules without exceeding the teacher.

    Older result files do not serialize the teacher's base channel count, so
    they retain the small, curated three-level compatibility table.  Newer
    profiles can search schedules of the same depth while keeping every
    resolution at or below its teacher width.
    """

    teacher_model_channels = topology.get("model_channels")
    teacher_channel_mult = tuple(int(value) for value in topology.get("channel_mult", DEFAULT_CHANNEL_MULT))
    if teacher_model_channels is None:
        if teacher_channel_mult != DEFAULT_CHANNEL_MULT:
            return []
        return [
            (int(channels), tuple(mult))
            for channels, mult in EXTRA_UNIFORM_ARCH_CANDIDATES
            if is_uniform_architecture_safe(int(channels), mult)
        ]

    teacher_model_channels = int(teacher_model_channels)
    if teacher_model_channels < 8 or not teacher_channel_mult:
        return []
    max_multiplier = max(4, max(teacher_channel_mult))
    multiplier_values = range(1, max_multiplier + 1)
    if teacher_channel_mult[0] == 1:
        schedules = (
            (1, *tail)
            for tail in itertools.combinations_with_replacement(
                multiplier_values, len(teacher_channel_mult) - 1
            )
        )
    else:
        schedules = itertools.combinations_with_replacement(
            multiplier_values, len(teacher_channel_mult)
        )
    schedule_list = [tuple(int(value) for value in schedule) for schedule in schedules]
    teacher_widths = tuple(
        teacher_model_channels * multiplier for multiplier in teacher_channel_mult
    )

    specs: list[tuple[int, tuple[int, ...]]] = []
    for model_channels in range(8, teacher_model_channels + 1, 8):
        for channel_mult in schedule_list:
            if any(
                model_channels * multiplier > maximum
                for multiplier, maximum in zip(channel_mult, teacher_widths)
            ):
                continue
            if not is_uniform_architecture_safe(model_channels, channel_mult):
                continue
            specs.append((model_channels, channel_mult))
    return specs


def _build_uniform_architecture_candidate_table(
    *,
    label_dim: int,
    img_resolution: int,
    img_channels: int,
    topology: Mapping[str, Any],
) -> tuple[tuple[int, tuple[int, ...], int, int], ...]:
    cache_key = (
        int(label_dim),
        int(img_resolution),
        int(img_channels),
        json.dumps(dict(topology), sort_keys=True),
    )
    cached = _UNIFORM_ARCHITECTURE_CANDIDATE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    table: list[tuple[int, tuple[int, ...], int, int]] = []
    for model_channels, channel_mult in _uniform_architecture_specs(topology):
        kwargs = default_model_kwargs(
            model_channels=model_channels,
            label_dim=label_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            channel_mult=channel_mult,
            num_blocks=int(topology.get("num_blocks", 4)),
            attn_resolutions=topology.get("attn_resolutions", [16]),
            dropout=float(topology.get("dropout", 0.13)),
            model_family=str(topology.get("model_family", "edm")),
            preconditioning=topology.get("preconditioning", {}),
            augment_dim=int(topology.get("augment_dim", 0)),
        )
        model = construct_student_for_counting(kwargs)
        table.append(
            (
                int(model_channels),
                tuple(channel_mult),
                count_grouped_parameters(model),
                count_full_parameters(model),
            )
        )
        del model
    result = tuple(table)
    _UNIFORM_ARCHITECTURE_CANDIDATE_CACHE[cache_key] = result
    return result


def _compile_extra_arch_candidate_student(
    *,
    student_index: int,
    timestep_block: Sequence[int],
    target_grouped_budget: float,
    label_dim: int,
    img_resolution: int,
    img_channels: int,
    budget_tolerance: float,
    topology: Mapping[str, Any] | None = None,
) -> StudentArchitectureReport | None:
    topology = dict(topology or {})
    table = _build_uniform_architecture_candidate_table(
        label_dim=label_dim,
        img_resolution=img_resolution,
        img_channels=img_channels,
        topology=topology,
    )
    best: tuple[float, int, int, int, tuple[int, ...]] | None = None
    for model_channels, channel_mult, realized, full_count in table:
        mismatch = abs(float(realized) - float(target_grouped_budget)) / float(target_grouped_budget)
        candidate = (mismatch, realized, full_count, model_channels, channel_mult)
        if best is None or (candidate[0], candidate[3], candidate[4]) < (best[0], best[3], best[4]):
            best = candidate

    if best is None or best[0] > budget_tolerance:
        return None
    relative_mismatch, realized_grouped, full_count, model_channels, channel_mult = best
    kwargs = default_model_kwargs(
        model_channels=model_channels,
        label_dim=label_dim,
        img_resolution=img_resolution,
        img_channels=img_channels,
        channel_mult=channel_mult,
        num_blocks=int(topology.get("num_blocks", 4)),
        attn_resolutions=topology.get("attn_resolutions", [16]),
        dropout=float(topology.get("dropout", 0.13)),
        model_family=str(topology.get("model_family", "edm")),
        preconditioning=topology.get("preconditioning", {}),
        augment_dim=int(topology.get("augment_dim", 0)),
    )
    return StudentArchitectureReport(
        student_index=int(student_index),
        timestep_block=[int(timestep_block[0]), int(timestep_block[1])],
        target_grouped_budget=float(target_grouped_budget),
        realized_grouped_budget=int(realized_grouped),
        relative_mismatch=float(relative_mismatch),
        full_parameter_count=int(full_count),
        model_kwargs=kwargs,
        width_profile={},
        rounding={
            "mode": "uniform_architecture_candidate",
            "channel_divisibility": 8,
            "selected_model_channels": int(model_channels),
            "selected_channel_mult": list(channel_mult),
            "teacher_model_channels": topology.get("model_channels"),
            "teacher_channel_mult": list(topology.get("channel_mult", DEFAULT_CHANNEL_MULT)),
            "max_hidden_channels": "teacher_width",
        },
    )


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a plain or gzip-compressed (``.json.gz``) JSON artifact."""

    return _load_json_file(path)


def infer_label_dim(results: Mapping[str, Any]) -> int:
    model_info = results.get("model_info", {})
    if "label_dim" in model_info:
        return int(model_info["label_dim"])
    model_config = model_info.get("model_config", {})
    if "label_dim" in model_config:
        return int(model_config["label_dim"])
    teacher = model_info.get("teacher") or results.get("teacher") or results.get("config", {}).get("teacher")
    if isinstance(teacher, Mapping) and "label_dim" in teacher.get("model_config", {}):
        return int(teacher["model_config"]["label_dim"])
    classes = results.get("dataset_info", {}).get("cifar_classes")
    if isinstance(classes, list) and classes:
        return len(classes)
    dataset = str(results.get("dataset_info", {}).get("dataset") or results.get("config", {}).get("dataset") or "").lower()
    if dataset in {"imagenet", "imagenet1k_parquet"}:
        return 1000
    network_pkl = str(results.get("config", {}).get("network_pkl") or "").lower()
    if "imagenet" in network_pkl:
        return 1000
    if dataset in {"ffhq", "lsun_bedroom", "lsun-bedroom", "bedroom"}:
        return 0
    return 10


def infer_img_resolution(results: Mapping[str, Any]) -> int:
    config = results.get("config", {})
    model_info = results.get("model_info", {})
    if model_info.get("img_resolution"):
        return int(model_info["img_resolution"])
    if model_info.get("model_config", {}).get("image_size"):
        return int(model_info["model_config"]["image_size"])
    teacher = model_info.get("teacher") or results.get("teacher") or config.get("teacher")
    if isinstance(teacher, Mapping) and teacher.get("model_config", {}).get("image_size"):
        return int(teacher["model_config"]["image_size"])
    image_size = config.get("image_size")
    if image_size:
        return int(image_size)
    for name in results.get("group_names", []):
        if isinstance(name, str) and "x" in name:
            for token in name.split("."):
                if "x" not in token:
                    continue
                left, _, right = token.partition("x")
                if left.isdigit() and right.split("_", 1)[0].isdigit():
                    return int(left)
    network_pkl = str(config.get("network_pkl") or "").lower()
    if "64x64" in network_pkl:
        return 64
    if "32x32" in network_pkl:
        return 32
    return 32


def infer_model_family(results: Mapping[str, Any]) -> str:
    model_info_family = results.get("model_info", {}).get("model_family")
    if model_info_family:
        return str(model_info_family)
    config_family = results.get("config", {}).get("model_family")
    if config_family and str(config_family) != "auto":
        return str(config_family)
    teacher = results.get("model_info", {}).get("teacher") or results.get("teacher")
    if isinstance(teacher, Mapping) and teacher.get("model_config", {}).get("model_family"):
        return str(teacher["model_config"]["model_family"])
    return "edm"


def group_original_structural_budgets(results: Mapping[str, Any]) -> dict[str, float]:
    names = list(results["group_names"])
    counts_obj = results["group_param_counts"]
    if isinstance(counts_obj, Mapping):
        counts = [float(counts_obj[name]) for name in names]
    else:
        counts = [float(v) for v in counts_obj]
    expansion = aligned_group_expansion_weights(results, names)

    budgets: dict[str, float] = {}
    structural_keys = results.get("group_structural_keys", {})
    for name, count, expansion_weight in zip(names, counts, expansion):
        key = str(structural_keys.get(name) or module_to_structural_key(strip_filter_suffix(name)))
        if key.endswith("_aux_conv"):
            continue
        budgets[key] = budgets.get(key, 0.0) + float(expansion_weight) * float(count)
    return budgets


def group_layerwise_structural_budgets(results: Mapping[str, Any], layer_budgets: Sequence[float]) -> dict[str, float]:
    names = list(results["group_names"])
    if len(names) != len(layer_budgets):
        raise ValueError(f"layer budget length {len(layer_budgets)} does not match group_names length {len(names)}")
    budgets: dict[str, float] = {}
    structural_keys = results.get("group_structural_keys", {})
    for name, budget in zip(names, layer_budgets):
        key = str(structural_keys.get(name) or module_to_structural_key(strip_filter_suffix(name)))
        if key.endswith("_aux_conv"):
            continue
        budgets[key] = budgets.get(key, 0.0) + float(budget)
    return budgets


def collect_structural_widths(model: torch.nn.Module) -> dict[str, int]:
    widths: dict[str, int] = {}
    inner = getattr(model, "model", model)
    for module_name, module in inner.named_modules():
        if not module_name or not _is_conv_like_module(module):
            continue
        if module_name.endswith("_aux_conv"):
            continue
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            continue
        key = module_to_structural_key(f"model.{module_name}")
        width = structural_width_from_module(module_name, int(weight.shape[0]))
        widths[key] = max(widths.get(key, 0), width)
    return widths


def _layerwise_profile_for_scale(
    desired_widths: Mapping[str, float],
    original_widths: Mapping[str, int],
    scale: float,
    *,
    multiple: int = 8,
    minimum: int = 8,
) -> dict[str, int]:
    profile: dict[str, int] = {}
    for key, desired in desired_widths.items():
        maximum = int(original_widths[key])
        profile[key] = round_group_norm_safe_channels(desired * scale, multiple=multiple, minimum=minimum, maximum=maximum)
    return profile


def _uniform_profile_for_scale(original_widths: Mapping[str, int], scale: float) -> dict[str, int]:
    return {
        key: round_group_norm_safe_channels(width * scale, maximum=int(width))
        for key, width in original_widths.items()
    }


def _profile_signature(profile: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((str(key), int(width)) for key, width in profile.items()))


def _refine_structural_profile_bracket(
    lower_profile: Mapping[str, int],
    upper_profile: Mapping[str, int],
    *,
    target_grouped_budget: float,
    evaluate: Callable[[Mapping[str, int]], tuple[int, int]],
    multiple: int = 8,
) -> tuple[float, int, int, dict[str, int], int] | None:
    """Walk a deterministic, GroupNorm-safe path across a rounding plateau.

    Scalar width searches can make dozens of equal-width blocks jump at the
    same threshold.  Moving those structural widths one at a time exposes the
    valid intermediate architectures without changing topology or exceeding
    either bracket's per-layer maximum.
    """

    if set(lower_profile) != set(upper_profile):
        raise ValueError("structural profile brackets must contain identical keys")
    if any(int(lower_profile[key]) > int(upper_profile[key]) for key in lower_profile):
        raise ValueError("structural profile lower bracket must not exceed upper bracket")

    current = {str(key): int(width) for key, width in lower_profile.items()}
    best: tuple[float, int, int, dict[str, int], int] | None = None
    evaluated_steps = 0
    for key in sorted(current):
        lower_width = int(current[key])
        upper_width = int(upper_profile[key])
        widths = [
            width
            for width in range(lower_width + multiple, upper_width + 1, multiple)
            if is_edm_group_norm_safe(width)
        ]
        if upper_width > lower_width and (not widths or widths[-1] != upper_width):
            widths.append(upper_width)
        for width in widths:
            current[key] = int(width)
            realized, full_count = evaluate(current)
            evaluated_steps += 1
            mismatch = abs(float(realized) - float(target_grouped_budget)) / float(target_grouped_budget)
            candidate = (mismatch, int(realized), int(full_count), dict(current), evaluated_steps)
            if best is None or (candidate[0], candidate[1], _profile_signature(candidate[3])) < (
                best[0],
                best[1],
                _profile_signature(best[3]),
            ):
                best = candidate
            if realized >= target_grouped_budget:
                return best
    return best


def _compile_structural_uniform_student(
    *,
    student_index: int,
    timestep_block: Sequence[int],
    target_grouped_budget: float,
    label_dim: int,
    img_resolution: int,
    img_channels: int,
    budget_tolerance: float,
    search_steps: int = 22,
    topology: Mapping[str, Any] | None = None,
) -> StudentArchitectureReport:
    topology = dict(topology or {})
    topology_kwargs = {
        "channel_mult": topology.get("channel_mult", DEFAULT_CHANNEL_MULT),
        "num_blocks": int(topology.get("num_blocks", 4)),
        "attn_resolutions": topology.get("attn_resolutions", [16]),
        "dropout": float(topology.get("dropout", 0.13)),
        "model_family": str(topology.get("model_family", "edm")),
        "preconditioning": topology.get("preconditioning", {}),
        "augment_dim": int(topology.get("augment_dim", 0)),
    }
    maximum_model_channels = int(
        topology.get("model_channels", default_proxy_model_channels(label_dim=label_dim, img_resolution=img_resolution))
    )
    student_model_channels = int(topology.get("model_channels", 128))
    base_kwargs = default_model_kwargs(
        model_channels=maximum_model_channels,
        label_dim=label_dim,
        img_resolution=img_resolution,
        img_channels=img_channels,
        **topology_kwargs,
    )
    base_model = construct_student_for_counting(base_kwargs)
    original_widths = collect_structural_widths(base_model)
    del base_model

    evaluated_profiles: dict[tuple[tuple[str, int], ...], tuple[int, int]] = {}

    def evaluate(profile: Mapping[str, int]) -> tuple[int, int]:
        signature = _profile_signature(profile)
        cached = evaluated_profiles.get(signature)
        if cached is not None:
            return cached
        kwargs = default_model_kwargs(
            model_channels=student_model_channels,
            label_dim=label_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            width_profile=profile,
            **topology_kwargs,
        )
        model = construct_student_for_counting(kwargs)
        counts = (count_grouped_parameters(model), count_full_parameters(model))
        del model
        evaluated_profiles[signature] = counts
        return counts

    best: tuple[float, int, int, dict[str, int], float] | None = None
    lower_bracket: tuple[int, int, dict[str, int], float] | None = None
    upper_bracket: tuple[int, int, dict[str, int], float] | None = None
    low = 0.05
    high = 1.0
    effective_search_steps = max(int(search_steps), 24) if int(label_dim) >= 1000 or int(img_resolution) >= 64 else int(search_steps)
    for _ in range(max(1, effective_search_steps)):
        mid = (low + high) / 2
        profile = _uniform_profile_for_scale(original_widths, mid)
        realized, full_count = evaluate(profile)
        mismatch = abs(float(realized) - float(target_grouped_budget)) / float(target_grouped_budget)
        if best is None or mismatch < best[0]:
            best = (mismatch, realized, full_count, dict(profile), mid)
        if realized <= target_grouped_budget and (
            lower_bracket is None or realized > lower_bracket[0]
        ):
            lower_bracket = (realized, full_count, dict(profile), mid)
        if realized >= target_grouped_budget and (
            upper_bracket is None or realized < upper_bracket[0]
        ):
            upper_bracket = (realized, full_count, dict(profile), mid)
        if realized < target_grouped_budget:
            low = mid
        else:
            high = mid

    if best is None:
        raise RuntimeError("structural uniform width search did not evaluate any candidates")
    relative_mismatch, realized_grouped, full_count, profile, selected_scale = best
    refinement_steps = 0
    if relative_mismatch > budget_tolerance and lower_bracket is not None and upper_bracket is not None:
        refined = _refine_structural_profile_bracket(
            lower_bracket[2],
            upper_bracket[2],
            target_grouped_budget=target_grouped_budget,
            evaluate=evaluate,
        )
        if refined is not None and refined[0] < relative_mismatch:
            relative_mismatch, realized_grouped, full_count, profile, refinement_steps = refined
            selected_scale = lower_bracket[3]
    if relative_mismatch > budget_tolerance:
        raise ValueError(
            f"Could not match target grouped budget {target_grouped_budget:.1f} within "
            f"{budget_tolerance:.1%}; structural uniform realized={realized_grouped}"
        )
    kwargs = default_model_kwargs(
        model_channels=student_model_channels,
        label_dim=label_dim,
        img_resolution=img_resolution,
        img_channels=img_channels,
        width_profile=profile,
        **topology_kwargs,
    )
    return StudentArchitectureReport(
        student_index=int(student_index),
        timestep_block=[int(timestep_block[0]), int(timestep_block[1])],
        target_grouped_budget=float(target_grouped_budget),
        realized_grouped_budget=int(realized_grouped),
        relative_mismatch=float(relative_mismatch),
        full_parameter_count=int(full_count),
        model_kwargs=kwargs,
        width_profile=dict(profile),
        rounding={
            "mode": "uniform_structural_widths",
            "channel_divisibility": 8,
            "min_hidden_channels": 8,
            "max_hidden_channels": "teacher_width",
            "selected_scale": float(selected_scale),
            "profile_refinement": "deterministic_bracket_path" if refinement_steps else None,
            "profile_refinement_steps": int(refinement_steps),
        },
    )


def compile_layerwise_student(
    *,
    student_index: int,
    timestep_block: Sequence[int],
    target_grouped_budget: float,
    layerwise_budgets: Sequence[float],
    results: Mapping[str, Any],
    label_dim: int,
    img_resolution: int,
    img_channels: int = 3,
    budget_tolerance: float = 0.05,
    search_steps: int = 12,
    topology: Mapping[str, Any] | None = None,
) -> StudentArchitectureReport:
    topology = dict(topology or {})
    topology_kwargs = {
        "channel_mult": topology.get("channel_mult", DEFAULT_CHANNEL_MULT),
        "num_blocks": int(topology.get("num_blocks", 4)),
        "attn_resolutions": topology.get("attn_resolutions", [16]),
        "dropout": float(topology.get("dropout", 0.13)),
        "model_family": str(topology.get("model_family", "edm")),
        "preconditioning": topology.get("preconditioning", {}),
        "augment_dim": int(topology.get("augment_dim", 0)),
    }
    proxy_model_channels = int(
        topology.get("model_channels", default_proxy_model_channels(label_dim=label_dim, img_resolution=img_resolution))
    )
    base_kwargs = default_model_kwargs(
        model_channels=proxy_model_channels,
        label_dim=label_dim,
        img_resolution=img_resolution,
        img_channels=img_channels,
        **topology_kwargs,
    )
    base_model = construct_student_for_counting(base_kwargs)
    original_widths = collect_structural_widths(base_model)
    del base_model

    original_budgets = group_original_structural_budgets(results)
    target_structural_budgets = group_layerwise_structural_budgets(results, layerwise_budgets)
    total_original_budget = max(float(sum(original_budgets.values())), 1e-12)
    total_target_budget = max(float(sum(layerwise_budgets)), 0.0)
    fallback_width_scale = math.sqrt(total_target_budget / total_original_budget)
    desired_widths: dict[str, float] = {}
    for key, original_width in original_widths.items():
        original_budget = float(original_budgets.get(key, 0.0))
        target_budget = max(float(target_structural_budgets.get(key, 0.0)), 0.0)
        if original_budget <= 0.0 and target_budget <= 0.0:
            desired_widths[key] = float(original_width) * fallback_width_scale
        else:
            desired_widths[key] = float(original_width) * math.sqrt(target_budget / max(original_budget, 1e-12))

    evaluated_profiles: dict[tuple[tuple[str, int], ...], tuple[int, int]] = {}

    def evaluate(profile: Mapping[str, int]) -> tuple[int, int]:
        signature = _profile_signature(profile)
        cached = evaluated_profiles.get(signature)
        if cached is not None:
            return cached
        kwargs = default_model_kwargs(
            model_channels=proxy_model_channels,
            label_dim=label_dim,
            img_resolution=img_resolution,
            img_channels=img_channels,
            width_profile=profile,
            **topology_kwargs,
        )
        model = construct_student_for_counting(kwargs)
        counts = (count_grouped_parameters(model), count_full_parameters(model))
        del model
        evaluated_profiles[signature] = counts
        return counts

    best: tuple[float, int, int, dict[str, int], float] | None = None
    lower_bracket: tuple[int, int, dict[str, int], float] | None = None
    upper_bracket: tuple[int, int, dict[str, int], float] | None = None
    low = 0.05
    high = 4096.0 if int(label_dim) >= 1000 or int(img_resolution) >= 64 else 1.75
    effective_search_steps = max(int(search_steps), 24) if int(label_dim) >= 1000 or int(img_resolution) >= 64 else int(search_steps)
    for _ in range(max(1, effective_search_steps)):
        mid = (low + high) / 2
        profile = _layerwise_profile_for_scale(desired_widths, original_widths, mid)
        realized, full_count = evaluate(profile)
        mismatch = abs(float(realized) - float(target_grouped_budget)) / float(target_grouped_budget)
        if best is None or mismatch < best[0]:
            best = (mismatch, realized, full_count, dict(profile), mid)
        if realized <= target_grouped_budget and (
            lower_bracket is None or realized > lower_bracket[0]
        ):
            lower_bracket = (realized, full_count, dict(profile), mid)
        if realized >= target_grouped_budget and (
            upper_bracket is None or realized < upper_bracket[0]
        ):
            upper_bracket = (realized, full_count, dict(profile), mid)
        if realized < target_grouped_budget:
            low = mid
        else:
            high = mid

    if best is None:
        raise RuntimeError("layerwise width search did not evaluate any candidates")
    relative_mismatch, realized_grouped, full_count, profile, selected_scale = best
    refinement_steps = 0
    if relative_mismatch > budget_tolerance and lower_bracket is not None and upper_bracket is not None:
        refined = _refine_structural_profile_bracket(
            lower_bracket[2],
            upper_bracket[2],
            target_grouped_budget=target_grouped_budget,
            evaluate=evaluate,
        )
        if refined is not None and refined[0] < relative_mismatch:
            relative_mismatch, realized_grouped, full_count, profile, refinement_steps = refined
            selected_scale = lower_bracket[3]
    if relative_mismatch > budget_tolerance:
        raise ValueError(
            f"Could not match layerwise target grouped budget {target_grouped_budget:.1f} within "
            f"{budget_tolerance:.1%}; realized={realized_grouped}"
        )
    kwargs = default_model_kwargs(
        model_channels=proxy_model_channels,
        label_dim=label_dim,
        img_resolution=img_resolution,
        img_channels=img_channels,
        width_profile=profile,
        **topology_kwargs,
    )
    return StudentArchitectureReport(
        student_index=int(student_index),
        timestep_block=[int(timestep_block[0]), int(timestep_block[1])],
        target_grouped_budget=float(target_grouped_budget),
        realized_grouped_budget=int(realized_grouped),
        relative_mismatch=float(relative_mismatch),
        full_parameter_count=int(full_count),
        model_kwargs=kwargs,
        width_profile=dict(profile),
        layerwise_target_sum=float(np.asarray(layerwise_budgets, dtype=np.float64).sum()),
        rounding={
            "mode": "layerwise_structural_widths",
            "channel_divisibility": 8,
            "min_hidden_channels": 8,
            "max_hidden_channels": "teacher_width",
            "selected_scale": float(selected_scale),
            "profile_refinement": "deterministic_bracket_path" if refinement_steps else None,
            "profile_refinement_steps": int(refinement_steps),
        },
    )


def _allocation_path(eval_output_dir: Path, variant: str, allocation_results_dir: str | Path | None = None) -> Path:
    directory = (
        Path(allocation_results_dir)
        if allocation_results_dir is not None
        else eval_output_dir / "allocation_results"
    )
    # Released allocations may be gzip-compressed; the plain file wins when both exist.
    found = find_json_file(directory, variant)
    return found if found is not None else directory / f"{variant}.json"


def _compact_allocation_rule(
    variant: str,
    allocation_payload: Mapping[str, Any],
) -> dict[str, Any]:
    serialized_rule = allocation_payload.get("allocation_rule")
    rule = serialized_rule if isinstance(serialized_rule, Mapping) else {}
    score_sources = allocation_payload.get("score_sources")
    score_sources = score_sources if isinstance(score_sources, Mapping) else {}
    block_source = score_sources.get("block_capacity_scores")
    block_source = block_source if isinstance(block_source, Mapping) else {}
    layer_source = score_sources.get("layer_capacity_scores")
    layer_source = layer_source if isinstance(layer_source, Mapping) else {}

    allocation_metric = rule.get("allocation_metric")
    if allocation_metric is None:
        allocation_metric = block_source.get("metric") or block_source.get("kind")
    score_reduction = rule.get("score_reduction")
    if score_reduction is None:
        score_reduction = block_source.get("reduction") or layer_source.get("reduction")
    layer_score_source = rule.get("layer_score_source")
    if layer_score_source is None:
        layer_score_source = layer_source.get("score_source") or layer_source.get("kind")

    return {
        "student_variant": str(allocation_payload.get("student_variant", variant)),
        "allocation_metric": allocation_metric,
        "score_reduction": score_reduction,
        "layer_score_source": layer_score_source,
        "allocation_alpha": rule.get("allocation_alpha"),
        "shuffle_seed": (
            allocation_payload.get("shuffle_seed") if variant == "shuffled_capacity" else None
        ),
    }


def _resolve_grouping_path(
    allocation_payload: Mapping[str, Any],
    *,
    allocation_path: Path,
    eval_output_dir: Path,
) -> Path | None:
    raw_path = allocation_payload.get("timestep_grouping_path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    recorded = Path(raw_path).expanduser()
    if recorded.is_absolute():
        candidates = [recorded]
    else:
        candidates = [Path.cwd() / recorded, eval_output_dir / recorded, allocation_path.parent / recorded]
        config_path = allocation_payload.get("config_path")
        if isinstance(config_path, str) and config_path:
            candidates.append(Path(config_path).expanduser().parent / recorded)
        # Released artifacts record repository-relative grouping paths.
        candidates.append(Path(__file__).resolve().parents[1] / recorded)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _grouping_identity(
    allocation_payload: Mapping[str, Any],
    *,
    allocation_path: Path,
    eval_output_dir: Path,
) -> dict[str, Any]:
    raw_path = allocation_payload.get("timestep_grouping_path")
    timestep_blocks = [
        [int(start), int(end)]
        for start, end in allocation_payload.get("timestep_blocks", [])
    ]
    identity: dict[str, Any] = {
        "path": raw_path,
        "sha256": None,
        "timestep_blocks": timestep_blocks,
    }
    resolved_path = _resolve_grouping_path(
        allocation_payload,
        allocation_path=allocation_path,
        eval_output_dir=eval_output_dir,
    )
    if resolved_path is None:
        return identity
    grouping_payload = load_json(resolved_path)
    identity["resolved_path"] = str(resolved_path)
    identity["sha256"] = file_digest(resolved_path)
    for key in ("num_blocks", "boundaries", "builtin_cost", "pairwise_normalization"):
        if key in grouping_payload:
            identity[key] = grouping_payload[key]
    return identity


def _variant_block_budgets(
    variant: str,
    allocation_payload: Mapping[str, Any],
    *,
    shuffle_seed: int,
) -> list[float]:
    if variant == "shuffled_capacity":
        artifact_seed = allocation_payload.get("shuffle_seed")
        if isinstance(artifact_seed, bool) or not isinstance(artifact_seed, int):
            raise ValueError(
                "shuffled_capacity allocation artifact must record an integer shuffle_seed"
            )
        if artifact_seed != shuffle_seed:
            raise ValueError(
                "shuffled_capacity allocation seed mismatch: "
                f"artifact shuffle_seed={artifact_seed}, planner shuffle_seed={shuffle_seed}"
            )
    return [float(x) for x in allocation_payload["target_budget_plan"]["block_budgets"]]


def compile_variant_architecture_plan(
    *,
    eval_output_dir: str | Path,
    variant: str,
    results: Mapping[str, Any],
    candidate_table: Sequence[tuple[int, int, int]] | None = None,
    allocation_results_dir: str | Path | None = None,
    budget_tolerance: float = 0.05,
    shuffle_seed: int = 3,
    layerwise_search_steps: int = 12,
) -> VariantArchitecturePlan:
    if variant not in SUPPORTED_DISTILLATION_VARIANTS:
        raise ValueError(f"variant must be one of {SUPPORTED_DISTILLATION_VARIANTS}, got {variant!r}")
    eval_dir = Path(eval_output_dir)
    allocation_path = _allocation_path(eval_dir, variant, allocation_results_dir)
    allocation = load_json(allocation_path)

    label_dim = infer_label_dim(results)
    img_resolution = infer_img_resolution(results)
    topology = infer_student_topology(results)
    block_budgets = _variant_block_budgets(
        variant,
        allocation,
        shuffle_seed=shuffle_seed,
    )
    if variant == "global":
        num_bins = len(results.get("sigma_bin_labels", [])) or len(results.get("n_eff", []))
        timestep_blocks = [[0, int(num_bins)]]
    else:
        timestep_blocks = [[int(a), int(b)] for a, b in allocation["timestep_blocks"]]

    students: list[StudentArchitectureReport] = []
    for index, budget in enumerate(block_budgets):
        block = timestep_blocks[0] if variant == "global" else timestep_blocks[index]
        if variant in {"layerwise_capacity", "reversed_layerwise_capacity", "combined_layerwise"}:
            layer_budgets = allocation["target_budget_plan"]["layer_budgets"][index]
            students.append(
                compile_layerwise_student(
                    student_index=index,
                    timestep_block=block,
                    target_grouped_budget=budget,
                    layerwise_budgets=layer_budgets,
                    results=results,
                    label_dim=label_dim,
                    img_resolution=img_resolution,
                    budget_tolerance=budget_tolerance,
                    search_steps=layerwise_search_steps,
                    topology=topology,
                )
            )
        else:
            students.append(
                compile_uniform_student(
                    student_index=index,
                    timestep_block=block,
                    target_grouped_budget=budget,
                    label_dim=label_dim,
                    img_resolution=img_resolution,
                    candidate_table=candidate_table,
                    budget_tolerance=budget_tolerance,
                    topology=topology,
                )
            )

    return VariantArchitecturePlan(
        variant=variant,
        source_allocation_path=str(allocation_path),
        source_allocation_sha256=file_digest(allocation_path),
        allocation_rule=_compact_allocation_rule(variant, allocation),
        grouping_identity=_grouping_identity(
            allocation,
            allocation_path=allocation_path,
            eval_output_dir=eval_dir,
        ),
        timestep_blocks=timestep_blocks,
        students=students,
        shuffle_seed=shuffle_seed if variant == "shuffled_capacity" else None,
    )


def prepare_distillation_architectures(
    *,
    eval_output_dir: str | Path = "out_eval_edm_cifar10",
    output_dir: str | Path = "out_distill_edm_cifar10/smoke",
    allocation_results_dir: str | Path | None = None,
    variants: Sequence[str] = DEFAULT_DISTILLATION_VARIANTS,
    budget_tolerance: float = 0.05,
    shuffle_seed: int = 3,
    layerwise_search_steps: int = 12,
    results_json_path: str | Path | None = None,
) -> dict[str, Any]:
    """Compile one architecture plan per variant from a profile and its allocations.

    ``results_json_path`` selects the profile explicitly (for example a released
    ``artifacts/profiles/*.json.gz``); by default ``<eval_output_dir>/results.json``
    (or ``results.json.gz``) is used.
    """
    eval_dir = Path(eval_output_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if results_json_path is not None:
        results_path = Path(results_json_path)
    else:
        results_path = find_json_file(eval_dir, "results") or eval_dir / "results.json"
    results = load_json(results_path)
    profile_provenance = provenance_from_results(results, results_path)
    topology = infer_student_topology(results)
    candidate_table = build_uniform_candidate_table(
        label_dim=infer_label_dim(results),
        img_resolution=infer_img_resolution(results),
        topology=topology,
    )

    plans: dict[str, Any] = {}
    for variant in variants:
        plan = compile_variant_architecture_plan(
            eval_output_dir=eval_dir,
            variant=variant,
            results=results,
            candidate_table=candidate_table,
            allocation_results_dir=allocation_results_dir,
            budget_tolerance=budget_tolerance,
            shuffle_seed=shuffle_seed,
            layerwise_search_steps=layerwise_search_steps,
        )
        allocation_source = Path(plan.source_allocation_path)
        validate_matching_source_profile(
            profile_provenance["source_profile"],
            load_json(allocation_source),
            context=f"allocation {allocation_source}",
            expected_ablation_protocol=profile_provenance["ablation_protocol"],
            expected_filter_sampling=profile_provenance.get("filter_sampling"),
        )
        variant_dir = out_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        payload = plan.to_dict()
        try:
            teacher_spec = teacher_spec_from_results(results)
            teacher_payload: dict[str, Any] | None = teacher_spec.to_dict()
        except ValueError:
            teacher_payload = None
        payload.update(
            {
                "results_json_path": str(results_path),
                "network_pkl": results.get("config", {}).get("network_pkl"),
                "teacher": teacher_payload,
                "student_topology": topology,
                "benchmark_protocol_ids": results.get("benchmark_protocol_ids", []),
                "benchmark_protocols": results.get("benchmark_protocols", []),
                "sigma_values": results.get("sigma_values", []),
                "num_sigma_bins": len(results.get("sigma_bin_labels", [])) or len(results.get("n_eff", [])),
                "model_family": infer_model_family(results),
                "dataset_info": results.get("dataset_info", {}),
                **profile_provenance,
            }
        )
        (variant_dir / "architecture_plan.json").write_text(json.dumps(payload, indent=2) + "\n")
        plans[variant] = payload

    summary = {
        "eval_output_dir": str(eval_dir),
        "allocation_results_dir": None if allocation_results_dir is None else str(allocation_results_dir),
        "output_dir": str(out_dir),
        "results_json_path": str(results_path),
        "variants": list(plans.keys()),
        "variant_plan_paths": {
            variant: str(out_dir / variant / "architecture_plan.json")
            for variant in plans
        },
        **profile_provenance,
        "plans": plans,
    }
    (out_dir / "architecture_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def construct_student_from_plan(plan: Mapping[str, Any]) -> torch.nn.Module:
    students = [construct_student_from_kwargs(student["model_kwargs"]) for student in plan["students"]]
    if len(students) == 1:
        return students[0]
    return BlockwiseEDMStudent(
        students=students,
        timestep_blocks=plan["timestep_blocks"],
        sigma_values=plan.get("sigma_values", []),
        num_sigma_bins=int(plan["num_sigma_bins"]),
    )


def _open_maybe_url(path_or_url: str):
    parsed = urllib.parse.urlparse(path_or_url)
    if parsed.scheme in {"http", "https"}:
        return urllib.request.urlopen(path_or_url)
    return open(path_or_url, "rb")


def load_edm_network(
    network_pkl: str | os.PathLike[str] | Mapping[str, Any] | TeacherSpec,
    *,
    device: torch.device,
    dtype: torch.dtype,
    cache_dir: str | Path | None = None,
    network_format: str | None = None,
    preset: str | None = None,
    trust_local_pickle: bool = False,
) -> torch.nn.Module:
    """Backward-compatible alias for the shared structured teacher loader."""

    return load_teacher_network(
        network_pkl,
        device=device,
        dtype=dtype,
        cache_dir=cache_dir,
        network_format=network_format,
        preset=preset,
        trust_local_pickle=trust_local_pickle,
    )


def hybrid_distillation_loss(
    *,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor | None,
    sigmas: torch.Tensor,
    noise: torch.Tensor,
    model_family: str = "vp",
    sigma_data: float = 0.5,
    kd_weight: float = 1.0,
    data_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    noisy = images + noise * sigmas.reshape(-1, 1, 1, 1)
    with torch.no_grad():
        teacher_denoised = teacher(noisy, sigmas, labels)
    student_denoised = student(noisy, sigmas, labels)
    weights = loss_weights_for_family(sigmas=sigmas, model_family=model_family, sigma_data=sigma_data)
    kd = mse_per_example(student_denoised.float(), teacher_denoised.float()) * weights.float()
    data = mse_per_example(student_denoised.float(), images.float()) * weights.float()
    kd_mean = kd.mean()
    data_mean = data.mean()
    loss = float(kd_weight) * kd_mean + float(data_weight) * data_mean
    metrics = {
        "loss": float(loss.detach().cpu().item()),
        "kd_loss": float(kd_mean.detach().cpu().item()),
        "data_loss": float(data_mean.detach().cpu().item()),
    }
    return loss, metrics


def update_ema(ema: torch.nn.Module, model: torch.nn.Module, beta: float) -> None:
    with torch.no_grad():
        for ema_param, model_param in zip(ema.parameters(), model.parameters()):
            ema_param.copy_(model_param.detach().lerp(ema_param, beta))
        for ema_buffer, model_buffer in zip(ema.buffers(), model.buffers()):
            ema_buffer.copy_(model_buffer)


def create_ema(model: torch.nn.Module) -> torch.nn.Module:
    return copy.deepcopy(model).eval().requires_grad_(False)
