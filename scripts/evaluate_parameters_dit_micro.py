#!/usr/bin/env python3
"""
Per-head parameter importance analysis for the CIFAR-10-native DiT-Micro
checkpoint (`normalcomputing/dit-cifar10-32x32-class`).

This mirrors the ablation pipeline used by ``evaluate_parameters_dit.py`` for
DiT-XL/2 (default ``--ablation_mode permutation``, the protocol of the archived
DiT-Micro profile of Appendix C.2), but adapts to three differences in the
CIFAR-trained DiT-Micro:

1. The model is much smaller (8 blocks, hidden=192, ~5.5M params).
2. Inputs are native 32x32 RGB pixels -- no VAE, no upsampling.
3. The attention module is ``torch.nn.MultiheadAttention`` (out_proj/in_proj
   layout), not the timm-style ``Attention(qkv, proj)`` used by DiT-XL/2.

The DiT-Micro architecture is reconstructed from the state-dict keys directly
since the upstream repo only ships .pt files (no source code, no model card).

OPT-IN NarrowDiT support (``--arch_cfg``, default None -- everything above is
unchanged when unset): checkpoints trained by this repository (the CIFAR-10
DiT-S/2-style teacher and any width-allocation student) are ``pace.dit_arch_alloc.NarrowDiT``
instances, not the legacy reconstruction above -- a different (timm-style
qkv/proj) attention module, arbitrary (hidden_size, depth), and an optional
``augment_dim``/``dropout``. When ``--arch_cfg`` points at that checkpoint's
sibling ``arch_cfg.json`` (as written by ``train_phase_students.py`` /
``dit_arch_to_plans.py``), the model is instead built via ``build_narrow_dit``
and the checkpoint is strict-loaded into it; grouping/hooks for
``--grouping attention_heads`` reuse ``evaluate_parameters_dit.py``'s
timm-style implementations (NarrowAttention's ``qkv``/``proj``/``num_heads``/
``head_dim`` layout already satisfies their duck-typed interface -- no new hook
classes needed). The EDM sigma-corruption math, dataset, and permutation
machinery below are otherwise identical for both checkpoint kinds.
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from evaluate_parameters_edm import (
    AblationMode,
    BinStats,
    CIFAR10Dataset,
    FilterPermutationHook,
    FilterRandomSameNormHook,
    FilterZeroHook,
    HAS_WANDB,
    PermutationMixin,
    PermutationOutputHook,
    RandomSameNormMixin,
    RandomSameNormOutputHook,
    SigmaCorruptionDataset,
    ZeroOutputHook,
    IMPORTANCE_CLIP_MODES,
    atomic_torch_save,
    build_delta_stack,
    checkpoint_name_for_rank,
    choose_dtype,
    cleanup_distributed,
    collate_corruption,
    compute_usage_metrics,
    count_parameters,
    format_invocation_command,
    gather_ablation_results,
    get_rank,
    get_world_size,
    init_distributed,
    is_distributed,
    is_main_process,
    level_to_bin,
    load_ablation_checkpoints,
    mse_per_example,
    release_cuda_memory,
    sanitize_tensor,
    save_results_and_plots,
    set_seed,
    should_compute_group_correlation,
    tensor_or_none_to_list,
)
from evaluate_parameters_dit import (
    TransformerHeadPermutationHook,
    TransformerHeadRandomSameNormHook,
    TransformerHeadZeroHook,
    _is_transformer_head_target,
    collect_attention_head_groups_dit,
    make_ddpm_alpha_schedule,
    make_timestep_bin_labels,
    make_timestep_schedule,
)

# Repo root (one level above scripts/) for the NarrowDiT import below -- mirrors
# the convention in scripts/dit_arch_to_plans.py.
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from pace.dit_arch_alloc import NarrowDiT, build_narrow_dit  # noqa: E402
from pace.external_repos import configure_dit_repo  # noqa: E402

if HAS_WANDB:
    import wandb


# ---------------------------------------------------------------------------
# DiT-Micro architecture (reconstructed from state-dict)
# ---------------------------------------------------------------------------

def _sinusoidal_timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Standard sinusoidal timestep embedding (matches DiT/DDPM convention)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half,
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def _build_2d_sincos_pos_embed(grid_size: int, embed_dim: int) -> torch.Tensor:
    """2D sin-cos positional embedding for (grid_size, grid_size) tokens.

    Returns a tensor of shape (1, grid_size**2, embed_dim).
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    gh, gw = torch.meshgrid(grid_h, grid_w, indexing="ij")
    grid = torch.stack([gw, gh], dim=0).reshape(2, -1)  # (2, N)
    assert embed_dim % 4 == 0
    half = embed_dim // 2

    def _1d(pos):
        omega = torch.arange(half // 2, dtype=torch.float32) / (half // 2)
        omega = 1.0 / (10000 ** omega)
        out = torch.einsum("n,d->nd", pos, omega)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=-1)

    emb_w = _1d(grid[0])
    emb_h = _1d(grid[1])
    emb = torch.cat([emb_w, emb_h], dim=-1)  # (N, embed_dim)
    return emb.unsqueeze(0)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = _sinusoidal_timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int):
        super().__init__()
        # +1 row for the null/unconditional class used by classifier-free guidance.
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.embedding_table(labels)


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, C, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x


class Mlp(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, mlp_hidden)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(mlp_hidden, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class HookableMultiheadSelfAttention(nn.Module):
    """State-dict-compatible replacement for ``nn.MultiheadAttention(batch_first=True)``
    that does an explicit ``self.out_proj(...)`` module call so that forward pre-hooks
    on ``out_proj`` actually fire.

    ``nn.MultiheadAttention`` routes its forward through fused kernels
    (``torch._native_multi_head_attention``) or ``F.multi_head_attention_forward``,
    both of which call ``F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)``
    instead of invoking ``self.out_proj`` as a module — so hooks registered on
    ``out_proj`` silently never fire. This implementation produces the same numerical
    output (modulo kernel differences) with the same parameter names so the upstream
    checkpoints load with no surgery.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # Names mirror nn.MultiheadAttention's state-dict layout exactly.
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        qkv = torch.nn.functional.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        B, N, D = x.shape
        H, hd = self.num_heads, self.head_dim
        q = q.reshape(B, N, H, hd).transpose(1, 2)  # (B, H, N, hd)
        k = k.reshape(B, N, H, hd).transpose(1, 2)
        v = v.reshape(B, N, H, hd).transpose(1, 2)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # (B, H, N, hd)
        # Use H*hd (not D) so the module also works when
        # num_heads*head_dim < embed_dim.
        attn = attn.transpose(1, 2).contiguous().reshape(B, N, H * hd)
        return self.out_proj(attn)


class DiTMicroBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = HookableMultiheadSelfAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        params = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = params.chunk(6, dim=1)
        h = _modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn(h)
        x = x + gate_msa.unsqueeze(1) * attn_out
        x = x + gate_mlp.unsqueeze(1) * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = _modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiTMicro(nn.Module):
    """DiT-Micro reconstructed from `normalcomputing/dit-cifar10-32x32-class`.

    Inferred config (from state-dict shapes):
      - input_size = 32, in_channels = 3 (raw RGB)
      - patch_size = 2 -> 16x16 = 256 tokens
      - hidden_size = 192, depth = 8 blocks
      - mlp_ratio = 4 (hidden 192 -> 768 -> 192)
      - num_classes = 10 (+1 null) ; embedding table is 11 x 192
      - final linear: 192 -> 12 = patch_size**2 * in_channels -> predicts noise only

    The number of attention heads is not stored in the state dict; we assume
    ``num_heads = 3`` (head_dim = 64), the DiT-S-style convention for
    hidden=192. If a checkpoint was trained with a different head count,
    inference outputs will be silently wrong; we sanity-check baseline loss
    in main() and warn if it is unreasonably large.
    """

    def __init__(
        self,
        input_size: int = 32,
        in_channels: int = 3,
        patch_size: int = 2,
        hidden_size: int = 192,
        depth: int = 8,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        num_classes: int = 10,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.grid_size = input_size // patch_size

        self.x_embedder = PatchEmbed(in_channels, hidden_size, patch_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)
        self.register_buffer(
            "pos_embed",
            _build_2d_sincos_pos_embed(self.grid_size, hidden_size),
            persistent=False,
        )
        self.blocks = nn.ModuleList([
            DiTMicroBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, patch*patch*C) -> (B, C, H, W)
        B = x.shape[0]
        p = self.patch_size
        h = w = self.grid_size
        x = x.reshape(B, h, w, p, p, self.in_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.reshape(B, self.in_channels, h * p, w * p)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.x_embedder(x) + self.pos_embed
        c = self.t_embedder(t) + self.y_embedder(y)
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        return self.unpatchify(x)


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def load_dit_micro_network(
    checkpoint_path: str,
    device: str = "cuda",
    num_heads: int = 3,
) -> DiTMicro:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        sd = state["model"]
    else:
        sd = state

    # Strip the ``base_model.`` prefix the upstream checkpoints use.
    sd = {
        (k[len("base_model."):] if k.startswith("base_model.") else k): v
        for k, v in sd.items()
    }

    model = DiTMicro(num_heads=num_heads).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # The pos_embed buffer is generated rather than loaded; everything else must match.
    extra_unexpected = [k for k in unexpected if not k.endswith("pos_embed")]
    extra_missing = [k for k in missing if not k.endswith("pos_embed")]
    if extra_missing or extra_unexpected:
        raise RuntimeError(
            f"State-dict mismatch loading {checkpoint_path}. "
            f"Missing (excl. pos_embed): {extra_missing[:5]}{'...' if len(extra_missing) > 5 else ''}. "
            f"Unexpected (excl. pos_embed): {extra_unexpected[:5]}{'...' if len(extra_unexpected) > 5 else ''}."
        )
    model.eval().requires_grad_(False)
    return model


def load_narrow_dit_network(
    checkpoint_path: str,
    arch_cfg_path: str,
    device: str = "cuda",
) -> NarrowDiT:
    """Build a ``NarrowDiT`` from an explicit ``arch_cfg.json`` (as written by
    ``train_phase_students.py`` / ``dit_arch_to_plans.py``) and strict-load a
    NarrowDiT-format checkpoint into it.

    Opt-in alternative to ``load_dit_micro_network``'s legacy DiT-Micro
    (hidden_size=192, depth=8, ``nn.MultiheadAttention``) reconstruction, used
    when the checkpoint under study is one of this repo's own NarrowDiT
    teachers/students (e.g. teacher_v3/v4) rather than the third-party
    ``normalcomputing/dit-cifar10-32x32-class`` checkpoint. ``arch_cfg.json``
    fully specifies the architecture (including ``augment_dim``/``dropout``
    when present) UNDER A ``"cfg"`` KEY -- the same on-disk schema
    ``scripts/evaluate_students.py``'s ``load_narrow_dit_from_dir`` reads
    (``arch_cfg["cfg"]``) -- so the checkpoint's state dict matches a freshly
    built model of that config exactly. Mirrors that function's EMA-preference
    ``--full_ckpt`` payload handling too: ``{"model": sd, "ema": sd_or_None}``
    prefers ``ema`` when present, else falls back to ``model``; a bare
    ``state_dict`` (legacy/non-full runs) is used unchanged.
    """
    arch_cfg = json.load(open(arch_cfg_path))
    model = build_narrow_dit(arch_cfg["cfg"]).to(device)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        ema = state.get("ema")
        sd = ema if ema is not None else state["model"]
    else:
        sd = state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # pos_embed is regenerated deterministically (sincos) rather than loaded;
    # everything else -- including aug_embedder when augment_dim > 0 -- must match.
    extra_missing = [k for k in missing if not k.endswith("pos_embed")]
    extra_unexpected = [k for k in unexpected if not k.endswith("pos_embed")]
    if extra_missing or extra_unexpected:
        raise RuntimeError(
            f"State-dict mismatch loading {checkpoint_path} against arch_cfg "
            f"{arch_cfg_path}. Missing (excl. pos_embed): "
            f"{extra_missing[:5]}{'...' if len(extra_missing) > 5 else ''}. "
            f"Unexpected (excl. pos_embed): "
            f"{extra_unexpected[:5]}{'...' if len(extra_unexpected) > 5 else ''}."
        )
    model.eval().requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Group collectors
# ---------------------------------------------------------------------------

def _collect_submodule_groups(
    net: DiTMicro, attr: Optional[str] = None,
) -> Dict[str, nn.Module]:
    groups: Dict[str, nn.Module] = {}
    for i, block in enumerate(net.blocks):
        if attr is None:
            groups[f"blocks.{i}"] = block
        else:
            groups[f"blocks.{i}.{attr}"] = getattr(block, attr)
    return groups


def collect_attention_head_groups_micro(net: DiTMicro) -> Dict[str, Tuple[nn.Module, int]]:
    groups: Dict[str, Tuple[nn.Module, int]] = {}
    for i, block in enumerate(net.blocks):
        for h in range(block.attn.num_heads):
            groups[f"blocks.{i}.attn.head_{h}"] = (block.attn, h)
    return groups


# ---------------------------------------------------------------------------
# Per-head ablation hooks for nn.MultiheadAttention
# ---------------------------------------------------------------------------

class MhaHeadZeroHook:
    """Zero one head's contiguous slice in the input to ``attn.out_proj``.

    Works with ``HookableMultiheadSelfAttention`` (above), NOT bare
    ``nn.MultiheadAttention`` -- the latter's fused-kernel forward never
    invokes ``out_proj`` as a discrete module so the hook silently never fires.
    """

    def __init__(self, attn_module: nn.Module, head_idx: int):
        self.proj = attn_module.out_proj
        self.head_idx = head_idx
        self.num_heads = attn_module.num_heads
        self.head_dim = attn_module.head_dim
        self.handle = None

    def _hook(self, module, args):
        x = args[0].clone()
        start = self.head_idx * self.head_dim
        x[..., start : start + self.head_dim] = 0
        return (x,) + args[1:]

    def __enter__(self):
        self.handle = self.proj.register_forward_pre_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class MhaHeadRandomSameNormHook(RandomSameNormMixin):
    """random_same_norm replacement for one head, BEFORE ``attn.out_proj``."""

    def __init__(self, attn_module: nn.Module, head_idx: int, random_seed: int):
        super().__init__(random_seed=random_seed)
        self.proj = attn_module.out_proj
        self.head_idx = head_idx
        self.num_heads = attn_module.num_heads
        self.head_dim = attn_module.head_dim
        self.handle = None

    def _hook(self, module, args):
        x = args[0].clone()
        start = self.head_idx * self.head_dim
        end = start + self.head_dim
        selected = x[..., start:end]
        x[..., start:end] = self._random_same_norm(selected)
        return (x,) + args[1:]

    def __enter__(self):
        self.handle = self.proj.register_forward_pre_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class MhaHeadPermutationHook(PermutationMixin):
    """Permutation FI for one head, BEFORE ``attn.out_proj``: permute the head's
    contiguous slice across the batch dimension."""

    def __init__(self, attn_module: nn.Module, head_idx: int, random_seed: int):
        super().__init__(random_seed=random_seed)
        self.proj = attn_module.out_proj
        self.head_idx = head_idx
        self.num_heads = attn_module.num_heads
        self.head_dim = attn_module.head_dim
        self.handle = None

    def _hook(self, module, args):
        x = args[0].clone()
        start = self.head_idx * self.head_dim
        end = start + self.head_dim
        selected = x[..., start:end]
        x[..., start:end] = self._permute_along_batch(selected)
        return (x,) + args[1:]

    def __enter__(self):
        self.handle = self.proj.register_forward_pre_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def _is_mha_head_target(ablate_target) -> bool:
    return (
        isinstance(ablate_target, tuple)
        and len(ablate_target) == 2
        and isinstance(ablate_target[0], (nn.MultiheadAttention, HookableMultiheadSelfAttention))
    )


# ---------------------------------------------------------------------------
# Dataset wrapper: raw 32x32 RGB latents-as-pixels
# ---------------------------------------------------------------------------

class _CifarImageCache(Dataset):
    """Wrap CIFAR10Dataset so it presents a (image, label) interface compatible
    with SigmaCorruptionDataset. Caches every image up-front to host RAM so the
    inner loop is pure GPU forward passes."""

    def __init__(self, image_dataset: Dataset):
        imgs, labels = [], []
        loader = DataLoader(image_dataset, batch_size=64, shuffle=False, num_workers=2)
        for x, y in tqdm(loader, desc="Caching CIFAR images", leave=False):
            imgs.append(x)
            labels.extend(int(c) for c in y)
        self.images = torch.cat(imgs, dim=0)
        self.labels = labels

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class DiTMicroEvaluator:
    def __init__(
        self,
        checkpoint_path: str,
        device: str,
        dtype: torch.dtype,
        num_timesteps: int,
        num_timestep_levels: int,
        num_heads: int = 3,
        class_idx_override: Optional[int] = None,
        prediction_target: str = "x0",
        diffusion: str = "edm",
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        rho: float = 7.0,
        arch_cfg_path: Optional[str] = None,
    ):
        self.device = torch.device(device)
        if arch_cfg_path is not None:
            # Opt-in NarrowDiT path (this repo's own teacher/student checkpoints).
            # num_heads is ignored here -- NarrowDiT's per-block head counts come
            # from arch_cfg_path itself.
            self.net = load_narrow_dit_network(
                checkpoint_path=checkpoint_path,
                arch_cfg_path=arch_cfg_path,
                device=device,
            )
        else:
            self.net = load_dit_micro_network(
                checkpoint_path=checkpoint_path,
                device=device,
                num_heads=num_heads,
            )
        self.net_dtype = next(self.net.parameters()).dtype
        self.compute_dtype = dtype
        self.in_channels = int(self.net.in_channels)
        self.class_idx_override = class_idx_override
        if prediction_target not in ("eps", "x0"):
            raise ValueError(f"prediction_target must be 'eps' or 'x0', got {prediction_target!r}")
        self.prediction_target = prediction_target
        if diffusion not in ("ddpm", "edm"):
            raise ValueError(f"diffusion must be 'ddpm' or 'edm', got {diffusion!r}")
        self.diffusion = diffusion
        self.sigma_data = float(sigma_data)

        if diffusion == "edm":
            # EDM (Karras 2022) variance-exploding schedule. This is the parameterization
            # the normalcomputing DiT-Micro checkpoint was actually trained with: the network
            # is a preconditioned denoiser F_theta, D = c_skip*x + c_out*F(c_in*x, c_noise, y),
            # c_noise = 0.25*ln(sigma). `timestep_values` holds the sigma levels (descending:
            # index 0 = highest noise), so the generic level/bin machinery works unchanged.
            i = torch.arange(num_timestep_levels, dtype=torch.float64, device=self.device)
            denom = max(num_timestep_levels - 1, 1)
            sig = (sigma_max ** (1.0 / rho)
                   + i / denom * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))) ** rho
            self.timestep_values = sig.to(torch.float32)  # sigma per level, high->low noise
        else:
            self.alpha_bar = make_ddpm_alpha_schedule(num_timesteps=num_timesteps).to(self.device)
            self.timestep_values = make_timestep_schedule(num_timestep_levels, self.device)

    @torch.no_grad()
    def forward_losses_from_fixed_corruption(
        self,
        latents: torch.Tensor,
        class_indices: torch.Tensor,
        timestep_indices: torch.Tensor,
        noise_seeds: torch.Tensor,
    ) -> torch.Tensor:
        latents = latents.to(device=self.device, dtype=torch.float32)
        class_indices = class_indices.to(self.device)
        if self.class_idx_override is not None:
            class_indices = torch.full_like(class_indices, fill_value=self.class_idx_override)
        timestep_indices = timestep_indices.to(self.device)

        levels = self.timestep_values[timestep_indices]  # timesteps (ddpm) or sigmas (edm)

        noise = torch.empty_like(latents)
        for i in range(latents.shape[0]):
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(noise_seeds[i].item()))
            noise[i] = torch.randn(
                latents[i].shape, generator=gen, device=self.device, dtype=latents.dtype,
            )

        autocast_enabled = self.compute_dtype != torch.float32 and self.device.type == "cuda"

        if self.diffusion == "edm":
            # EDM forward: x_t = x0 + sigma*n; preconditioned denoiser D; loss = MSE(D, x0).
            sigma = levels.float().view(-1, 1, 1, 1)
            sd = self.sigma_data
            x_t = latents + sigma * noise
            c_in = 1.0 / torch.sqrt(sigma ** 2 + sd ** 2)
            c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
            c_out = sigma * sd / torch.sqrt(sigma ** 2 + sd ** 2)
            c_noise = 0.25 * torch.log(sigma).view(-1)
            with torch.autocast(device_type=self.device.type, dtype=self.compute_dtype, enabled=autocast_enabled):
                F = self.net(c_in * x_t, c_noise, class_indices)
            denoised = c_skip * x_t + c_out * F.float()
            losses = mse_per_example(denoised.float(), latents.float())
            return losses

        # DDPM forward (legacy path; NOT correct for this checkpoint, kept for comparison).
        ab = self.alpha_bar[levels].float()
        sqrt_ab = torch.sqrt(ab).view(-1, 1, 1, 1)
        sqrt_one_minus_ab = torch.sqrt(1.0 - ab).view(-1, 1, 1, 1)
        x_t = sqrt_ab * latents + sqrt_one_minus_ab * noise
        with torch.autocast(device_type=self.device.type, dtype=self.compute_dtype, enabled=autocast_enabled):
            pred = self.net(x_t, levels, class_indices)
        target = latents if self.prediction_target == "x0" else noise
        losses = mse_per_example(pred.float(), target.float())
        return losses

    def _make_ablation_context(
        self,
        ablate_target,
        ablation_mode: AblationMode,
        ablation_random_seed: Optional[int],
    ):
        if ablate_target is None:
            return torch.no_grad()
        if isinstance(ablate_target, tuple):
            if len(ablate_target) == 3 and ablate_target[2] == "filter":
                if ablation_mode == "zero":
                    return FilterZeroHook(ablate_target[0], ablate_target[1])
                if ablation_mode == "random_same_norm":
                    if ablation_random_seed is None:
                        raise ValueError("ablation_random_seed required for random_same_norm")
                    return FilterRandomSameNormHook(
                        ablate_target[0], ablate_target[1], ablation_random_seed,
                    )
                if ablation_mode == "permutation":
                    if ablation_random_seed is None:
                        raise ValueError("ablation_random_seed required for permutation")
                    return FilterPermutationHook(
                        ablate_target[0], ablate_target[1], ablation_random_seed,
                    )
                raise ValueError(f"Unsupported ablation_mode: {ablation_mode}")
            if _is_mha_head_target(ablate_target):
                if ablation_mode == "zero":
                    return MhaHeadZeroHook(ablate_target[0], ablate_target[1])
                if ablation_mode == "random_same_norm":
                    if ablation_random_seed is None:
                        raise ValueError("ablation_random_seed required for random_same_norm")
                    return MhaHeadRandomSameNormHook(
                        ablate_target[0], ablate_target[1], random_seed=ablation_random_seed,
                    )
                if ablation_mode == "permutation":
                    if ablation_random_seed is None:
                        raise ValueError("ablation_random_seed required for permutation")
                    return MhaHeadPermutationHook(
                        ablate_target[0], ablate_target[1], random_seed=ablation_random_seed,
                    )
                raise ValueError(f"Unsupported ablation_mode: {ablation_mode}")
            # NarrowDiT (timm-style qkv/proj attention, e.g. teacher_v3/v4 --
            # loaded via --arch_cfg): reuse evaluate_parameters_dit.py's hooks,
            # which already duck-type on (proj, head_dim, num_heads) -- the same
            # attributes NarrowAttention exposes.
            if _is_transformer_head_target(ablate_target):
                if ablation_mode == "zero":
                    return TransformerHeadZeroHook(ablate_target[0], ablate_target[1])
                if ablation_mode == "random_same_norm":
                    if ablation_random_seed is None:
                        raise ValueError("ablation_random_seed required for random_same_norm")
                    return TransformerHeadRandomSameNormHook(
                        ablate_target[0], ablate_target[1], random_seed=ablation_random_seed,
                    )
                if ablation_mode == "permutation":
                    if ablation_random_seed is None:
                        raise ValueError("ablation_random_seed required for permutation")
                    return TransformerHeadPermutationHook(
                        ablate_target[0], ablate_target[1], random_seed=ablation_random_seed,
                    )
                raise ValueError(f"Unsupported ablation_mode: {ablation_mode}")
            raise ValueError(f"Unsupported ablate_target tuple: {ablate_target}")
        # whole-module
        if ablation_mode == "zero":
            return ZeroOutputHook(ablate_target)
        if ablation_mode == "random_same_norm":
            if ablation_random_seed is None:
                raise ValueError("ablation_random_seed required for random_same_norm")
            return RandomSameNormOutputHook(ablate_target, ablation_random_seed)
        if ablation_mode == "permutation":
            if ablation_random_seed is None:
                raise ValueError("ablation_random_seed required for permutation")
            return PermutationOutputHook(ablate_target, ablation_random_seed)
        raise ValueError(f"Unsupported ablation_mode: {ablation_mode}")

    def evaluate(
        self,
        dataloader: DataLoader,
        num_bins: int,
        ablate_target=None,
        ablation_mode: AblationMode = "zero",
        ablation_random_seed: Optional[int] = None,
        progress_desc: Optional[str] = None,
        permutation_group_by_level: bool = True,
    ) -> BinStats:
        stats = BinStats(num_bins=num_bins)
        context = self._make_ablation_context(ablate_target, ablation_mode, ablation_random_seed)
        # Permutation importance must hold t fixed: restrict the batch permutation
        # to same-timestep-level examples.
        # Mirrors evaluate_parameters_dit.py's wiring exactly.
        group_by_level = (
            permutation_group_by_level
            and ablation_mode == "permutation"
            and hasattr(context, "set_permutation_groups")
        )

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
            for images, class_indices, timestep_indices, noise_seeds in progress:
                if group_by_level:
                    context.set_permutation_groups(timestep_indices)
                losses = self.forward_losses_from_fixed_corruption(
                    latents=images,
                    class_indices=class_indices,
                    timestep_indices=timestep_indices,
                    noise_seeds=noise_seeds,
                )
                if not torch.isfinite(losses).all():
                    bad_idx = (~torch.isfinite(losses)).nonzero(as_tuple=False).flatten().tolist()
                    raise ValueError(
                        f"Non-finite losses at batch indices {bad_idx}. "
                        "Try --dtype fp32 or check num_heads."
                    )
                bin_ids = level_to_bin(
                    indices=timestep_indices,
                    num_levels=len(self.timestep_values),
                    num_bins=num_bins,
                )
                stats.update(bin_ids, losses)
        finally:
            progress.close()
            if ablate_target is not None:
                context.__exit__(None, None, None)
        return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--cifar_split", type=str, default="train",
                        choices=["validation", "test", "train"])
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Local path to a DiT-Micro .pt checkpoint (HF: normalcomputing/dit-cifar10-32x32-class), "
                             "or (with --arch_cfg) a NarrowDiT checkpoint trained by this repo (teachers, students).")
    parser.add_argument("--dit_repo", type=str, default=None,
                        help="facebookresearch/DiT checkout (default: $DIT_REPO).")
    parser.add_argument("--arch_cfg", type=str, default=None,
                        help="Path to a NarrowDiT arch_cfg.json (sibling of --checkpoint, as written by "
                             "train_phase_students.py / dit_arch_to_plans.py). OPT-IN: when set, --checkpoint is "
                             "loaded as a NarrowDiT built from this config (arbitrary hidden_size/depth, timm-style "
                             "qkv/proj attention, optional augment_dim/dropout) instead of the legacy hardcoded "
                             "DiT-Micro-D192/depth8 nn.MultiheadAttention reconstruction. Default None preserves the "
                             "previous behavior exactly. --num_heads is ignored when this is set.")
    parser.add_argument("--num_heads", type=int, default=3,
                        help="Number of attention heads in the DiT-Micro architecture. "
                             "Inferred default (3) follows DiT-S convention for hidden_size=192. "
                             "Ignored when --arch_cfg is set.")
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument(
        "--grouping", type=str, default="attention_heads",
        choices=["blocks", "attention", "mlp", "adaln", "attention_heads"],
    )
    parser.add_argument(
        "--ablation_mode", type=str, default="permutation",
        choices=["zero", "random_same_norm", "permutation"],
        help="'permutation' (default, the protocol of the archived DiT-Micro profile) shuffles the "
             "group's activation across the batch (Breiman permutation feature importance); "
             "'random_same_norm' injects matched-norm Gaussian noise; 'zero' zeroes it.",
    )
    parser.add_argument(
        "--permutation_group_by_level",
        type=lambda s: str(s).lower() not in ("0", "false", "no"),
        default=True,
        help="For --ablation_mode permutation, restrict the batch permutation to examples at the "
             "SAME timestep-level (holds t fixed; correct Breiman importance). Default True. Set "
             "False only to reproduce the legacy behaviour that swaps activations across t-levels "
             "and can inject a spurious period-batch_size ripple into n_eff when batch_size < "
             "num_timestep_levels. Mirrors "
             "evaluate_parameters_dit.py's flag of the same name.",
    )
    parser.add_argument(
        "--corruption_order", type=str, default="level_major",
        choices=["level_major", "image_major"],
        help="Sample ordering for the corruption dataset. 'level_major' (default) groups same-level "
             "examples into each batch so permutation importance holds t fixed. 'image_major' is the "
             "legacy order that (with batch_size<num_timestep_levels) causes the period-batch_size "
             "n_eff artifact; use only to reproduce old runs. Mirrors evaluate_parameters_dit.py's "
             "flag of the same name.",
    )
    parser.add_argument(
        "--class_idx_override", type=int, default=None,
        help="Replace every dataset-provided class label with this integer "
             "(use num_classes, e.g. 10, for the null/unconditional class).",
    )
    parser.add_argument(
        "--prediction_target", type=str, default="x0", choices=["eps", "x0"],
        help="(ddpm mode only) what the network output represents.",
    )
    parser.add_argument(
        "--diffusion", type=str, default="edm", choices=["ddpm", "edm"],
        help="Forward-process parameterization. The normalcomputing DiT-Micro checkpoint "
             "was trained as an EDM (Karras 2022) preconditioned denoiser, so default is 'edm'.",
    )
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--sigma_data", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--max_groups", type=int, default=None)
    parser.add_argument("--samples_per_image", type=int, default=2,
                        help="Deprecated: ignored.")
    parser.add_argument("--num_bins", type=int, default=20)
    parser.add_argument("--num_timestep_levels", type=int, default=64)
    parser.add_argument("--timestep_stride", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group_correlation", type=str, default="auto",
                        choices=["auto", "always", "never"])
    parser.add_argument("--max_group_correlation_groups", type=int, default=2048)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument(
        "--importance_clip_mode",
        type=str,
        default="per_entry",
        choices=list(IMPORTANCE_CLIP_MODES),
        help=(
            "How the negative-delta floor is applied to per-head permutation-importance "
            "deltas. 'per_entry' (default) clamps each (group, bin) delta at 0 before any "
            "aggregation -- byte-identical to this script's original behavior. 'post_agg' "
            "keeps the per-entry deltas signed (saved delta_stack/relative_delta_stack stay "
            "signed) and only clamps the weights/n_eff/p_eff aggregation input. 'none' saves "
            "fully signed aggregates with no clamp anywhere; downstream consumers that need "
            "non-negativity must handle it at consumption."
        ),
    )
    args = parser.parse_args()
    configure_dit_repo(args.dit_repo)

    device, rank, world_size = init_distributed(args.device)
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
            raise ImportError("wandb is required when --wandb_project is set.")
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    evaluator = None
    dataloader = None
    image_dataset = None
    corruption_dataset = None
    try:
        print(f"[rank {rank}] Preparing CIFAR-10 dataset (raw 32x32 RGB)...")
        image_dataset = CIFAR10Dataset(
            root=args.data_root, image_size=32,
            split=args.cifar_split, max_images=args.max_images,
            download=args.download,
        )
        image_cache = _CifarImageCache(image_dataset)

        print(f"[rank {rank}] Loading DiT-Micro on {args.device} (num_heads={args.num_heads})...")
        evaluator = DiTMicroEvaluator(
            checkpoint_path=args.checkpoint,
            device=args.device,
            dtype=dtype,
            num_timesteps=args.num_timesteps,
            num_timestep_levels=args.num_timestep_levels,
            num_heads=args.num_heads,
            class_idx_override=args.class_idx_override,
            prediction_target=args.prediction_target,
            diffusion=args.diffusion,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            sigma_data=args.sigma_data,
            rho=args.rho,
            arch_cfg_path=args.arch_cfg,
        )
        is_narrow = isinstance(evaluator.net, NarrowDiT)
        num_blocks = len(evaluator.net.blocks)

        if is_main_process():
            print(
                f"{'NarrowDiT' if is_narrow else 'DiT-Micro'}: {num_blocks} blocks * "
                f"{evaluator.net.num_heads} heads = {num_blocks * evaluator.net.num_heads} head groups; "
                f"ablation_mode: {args.ablation_mode}; grouping: {args.grouping}"
            )

        corruption_dataset = SigmaCorruptionDataset(
            image_dataset=image_cache,
            sigma_values=evaluator.timestep_values.float(),
            samples_per_image=args.samples_per_image,
            seed=args.seed,
            sigma_stride=args.timestep_stride,
            order=args.corruption_order,
        )

        dataloader = DataLoader(
            corruption_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, collate_fn=collate_corruption,
        )

        print(f"[rank {rank}] Collecting groups...")
        _GROUPING_ATTRS = {
            "blocks": None,
            "attention": "attn",
            "mlp": "mlp",
            "adaln": "adaLN_modulation",
        }
        if args.grouping in _GROUPING_ATTRS:
            groups = _collect_submodule_groups(evaluator.net, _GROUPING_ATTRS[args.grouping])
        elif args.grouping == "attention_heads":
            groups = (
                collect_attention_head_groups_dit(evaluator.net) if is_narrow
                else collect_attention_head_groups_micro(evaluator.net)
            )
        else:
            raise ValueError(f"Unknown grouping: {args.grouping}")

        if not groups:
            raise ValueError(f"No groups found for grouping={args.grouping}.")

        if args.max_groups is not None:
            group_items_full = list(groups.items())
            groups = dict(group_items_full[: args.max_groups])

        if args.grouping == "attention_heads":
            group_param_counts = {
                name: max(1, count_parameters(mod) // max(1, mod.num_heads or 1))
                for name, (mod, _) in groups.items()
            }
        else:
            group_param_counts = {name: count_parameters(mod) for name, mod in groups.items()}

        if is_main_process():
            print(f"Found {len(groups)} groups across {world_size} process(es).")

        baseline_mean = torch.zeros(args.num_bins, dtype=torch.float64)
        baseline_stderr = torch.zeros(args.num_bins, dtype=torch.float64)
        baseline_count = torch.zeros(args.num_bins, dtype=torch.long)
        if is_main_process():
            print("Running baseline evaluation...")
            stats = evaluator.evaluate(
                dataloader=dataloader, num_bins=args.num_bins,
                ablate_target=None, progress_desc="baseline",
            )
            baseline_mean = stats.mean()
            baseline_stderr = stats.stderr()
            baseline_count = stats.count.clone()
            print(f"baseline_mean range: [{baseline_mean.min():.4e}, {baseline_mean.max():.4e}]")
            if baseline_mean.max() > 5.0:
                print("WARNING: baseline_mean unusually large -- try a different --num_heads.")

        group_items = list(groups.items())
        indexed_group_items = list(enumerate(group_items))
        local_group_items = indexed_group_items[rank::world_size]

        if is_main_process():
            baseline_ckpt = os.path.join(args.output_dir, "baseline.pt")
            atomic_torch_save({
                "mean": baseline_mean, "stderr": baseline_stderr, "count": baseline_count,
                "max_images": len(image_cache), "num_bins": args.num_bins,
                "num_timestep_levels": args.num_timestep_levels,
                "ablation_mode": args.ablation_mode,
                "num_heads": evaluator.net.num_heads,
            }, baseline_ckpt)

        print(f"[rank {rank}] Running {len(local_group_items)} / {len(group_items)} group ablations...")
        checkpoint_name = checkpoint_name_for_rank(args.ablation_mode, rank)
        checkpoint_path = os.path.join(args.output_dir, checkpoint_name)
        group_names_for_resume = [name for name, _ in group_items]
        completed, loaded_paths, ignored_unknown, ignored_bad_shape = load_ablation_checkpoints(
            output_dir=args.output_dir,
            ablation_mode=args.ablation_mode,
            expected_group_names=group_names_for_resume,
            num_bins=args.num_bins,
        )
        local_ablated_means = {
            name: completed[name]
            for _, (name, _) in local_group_items
            if name in completed
        }

        for local_idx, (global_idx, (name, module)) in enumerate(local_group_items, start=1):
            if name in local_ablated_means:
                continue
            print(f"[rank {rank}] [{local_idx}/{len(local_group_items)}] Ablating {name}")
            group_random_seed = args.seed + 2_000_000 + global_idx
            stats = evaluator.evaluate(
                dataloader=dataloader, num_bins=args.num_bins,
                ablate_target=module,
                ablation_mode=args.ablation_mode,
                ablation_random_seed=group_random_seed,
                progress_desc=f"rank {rank} ablation {local_idx}/{len(local_group_items)} {name}",
                permutation_group_by_level=args.permutation_group_by_level,
            )
            local_ablated_means[name] = stats.mean()
            if use_wandb:
                wandb.log({"ablation_progress": local_idx / len(local_group_items)})
            if local_idx % 10 == 0:
                atomic_torch_save(local_ablated_means, checkpoint_path)
            release_cuda_memory(stats)

        atomic_torch_save(local_ablated_means, checkpoint_path)
        if is_distributed():
            dist.barrier()
        ablated_means = gather_ablation_results(local_ablated_means)
        if len(ablated_means) != len(group_items):
            missing = sorted(set(name for name, _ in group_items) - set(ablated_means))
            raise RuntimeError(f"Missing ablation results for {len(missing)} groups: {missing[:5]}")

        if is_main_process():
            print("Computing deltas and usage metrics...")
            names = list(groups.keys())
            delta_stack = build_delta_stack(
                ablated_means=ablated_means,
                baseline_mean=baseline_mean,
                names=names,
                clip_mode=args.importance_clip_mode,
            )
            compute_group_correlation = should_compute_group_correlation(
                args.group_correlation,
                num_groups=len(names),
                max_groups=args.max_group_correlation_groups,
            )
            usage_metrics = compute_usage_metrics(
                delta_stack=delta_stack,
                baseline_mean=baseline_mean,
                group_param_counts=group_param_counts,
                group_names=names,
                compute_group_correlation=compute_group_correlation,
                clip_mode=args.importance_clip_mode,
            )

            raw_baseline_mean = baseline_mean.clone()
            raw_baseline_stderr = baseline_stderr.clone()
            timestep_values_strided = evaluator.timestep_values[::args.timestep_stride]
            timestep_bin_labels = make_timestep_bin_labels(timestep_values_strided, args.num_bins)

            invocation_argv = list(sys.argv)
            results = {
                "config": vars(args),
                "importance_clip_mode": args.importance_clip_mode,
                "invocation": {
                    "argv": invocation_argv,
                    "command": format_invocation_command(invocation_argv),
                },
                "dataset_info": {
                    "dataset": "cifar10",
                    "cifar_split": args.cifar_split,
                },
                "model_info": (
                    {
                        "model": "narrow_dit",
                        "arch_cfg": args.arch_cfg,
                        "depth": num_blocks,
                        "hidden_size": json.load(open(args.arch_cfg))["cfg"]["hidden_size"],
                        "num_heads": evaluator.net.num_heads,
                        "patch_size": 2,
                        "in_channels": evaluator.net.in_channels,
                        "num_classes": 10,
                        "prediction_target": args.prediction_target,
                    } if is_narrow else {
                        "model": "dit-micro-32x32-3-cifar10-class",
                        "depth": 8,
                        "hidden_size": 192,
                        "num_heads": evaluator.net.num_heads,
                        "patch_size": 2,
                        "in_channels": 3,
                        "num_classes": 10,
                        "prediction_target": args.prediction_target,
                    }
                ),
                "ablation_info": {
                    "mode": args.ablation_mode,
                    "norm": "per_example_l2" if args.ablation_mode == "random_same_norm" else None,
                    "replacement": {
                        "random_same_norm": "gaussian_random_noise",
                        "permutation": "batch_permuted_activation",
                        "zero": "zeros",
                    }[args.ablation_mode],
                    "group_definition": "attention_head" if args.grouping == "attention_heads" else args.grouping,
                    "ablation_site": "pre_out_proj" if args.grouping == "attention_heads" else "module_output",
                },
                "group_names": names,
                "group_param_counts": {k: int(v) for k, v in group_param_counts.items()},
                "baseline_mean": baseline_mean.tolist(),
                "baseline_stderr": baseline_stderr.tolist(),
                "raw_baseline_mean": raw_baseline_mean.tolist(),
                "raw_baseline_stderr": raw_baseline_stderr.tolist(),
                "baseline_count": baseline_count.tolist(),
                "timestep_values": evaluator.timestep_values.detach().cpu().tolist(),
                "delta_stack": delta_stack.tolist(),
                "relative_delta_stack": usage_metrics["relative_delta_stack"].tolist(),
                "row_normalized_relative_delta_stack": usage_metrics["row_normalized_relative_delta_stack"].tolist(),
                "C_groups": tensor_or_none_to_list(usage_metrics["C_groups"]),
                "C_timesteps": usage_metrics["C_noise_levels"].tolist(),
                "timestep_bin_labels": timestep_bin_labels,
                "weights": usage_metrics["weights"].tolist(),
                "n_eff": usage_metrics["n_eff"].tolist(),
                "p_eff": usage_metrics["p_eff"].tolist(),
                "distributed": {"world_size": world_size},
                "group_correlation": {
                    "mode": args.group_correlation,
                    "computed": compute_group_correlation,
                    "max_groups": args.max_group_correlation_groups,
                    "num_groups": len(names),
                },
            }
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
                bin_labels=timestep_bin_labels,
                correlation_plot_name="timestep_correlation_heatmap.png",
                correlation_title="Timestep correlation heatmap",
                correlation_axis_title="Timestep bin",
                correlation_row_hover_label="timestep_bin_y",
                correlation_col_hover_label="timestep_bin_x",
            )
            print("Done.")
            print(f"Results saved to: {args.output_dir}")
    finally:
        if is_distributed():
            try:
                dist.barrier()
            except Exception:
                pass
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
