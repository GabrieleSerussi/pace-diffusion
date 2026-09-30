#!/usr/bin/env python3
"""
Measure timestep-binned parameter importance in a DiT network using group ablations.

Mirrors the reporting pipeline in ``evaluate_parameters_edm.py``, but targets
DiT (Scalable Diffusion Models with Transformers) checkpoints operating in VAE
latent space with a DDPM noise schedule.

The default ``--ablation_mode permutation`` is the permutation importance of
the paper's DiT profiles: when a group (here = attention head) is ablated, its
output is exchanged with the output of another example of the batch at the
same timestep level (a ``torch.randperm`` permutation per group; fixed points
are possible, unlike the U-Net ``pfi`` protocol).  The ``random_same_norm``
mode (Gaussian noise with the per-example L2 norm of the original output) and
``zero`` remain available.  The default of 20 noise bins is the paper setting.

The facebookresearch/DiT checkout is taken from ``--dit_repo`` or ``$DIT_REPO``.

For attention-head grouping the replacement happens BEFORE the output
projection of each transformer block, where the per-head channel layout is
still contiguous; for block / mlp / adaLN grouping it happens at the module
output.
"""

import argparse
import gc
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pace.external_repos import configure_dit_repo, import_dit_module
from evaluate_parameters_edm import (
    AblationMode,
    BinStats,
    CIFAR10_CLASSES,
    CIFAR10Dataset,
    FilterPermutationHook,
    FilterRandomSameNormHook,
    FilterZeroHook,
    HAS_WANDB,
    HeadPermutationHook,
    HeadRandomSameNormHook,
    HeadZeroHook,
    ImageFolderFlat,
    ImageNet1KParquetDataset,
    ImageNetDataset,
    PermutationMixin,
    PermutationOutputHook,
    RandomSameNormMixin,
    RandomSameNormOutputHook,
    SigmaCorruptionDataset,
    ZeroOutputHook,
    IMPORTANCE_CLIP_MODES,
    _get_module_num_heads,
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

if HAS_WANDB:
    import wandb


# ---------------------------------------------------------------------------
# DDPM schedule helpers
# ---------------------------------------------------------------------------

def make_ddpm_alpha_schedule(
    num_timesteps: int = 1000,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """Return alpha_bar for each timestep (shape ``[num_timesteps]``)."""
    betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float64)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return alpha_bar.float()


def make_timestep_schedule(num_levels: int, device: torch.device) -> torch.Tensor:
    """Return uniformly spaced integer timesteps from 999 down to 0."""
    if num_levels < 2:
        return torch.tensor([999], dtype=torch.long, device=device)
    return torch.linspace(999, 0, num_levels, device=device).long()


def make_timestep_bin_labels(
    timestep_values: torch.Tensor, num_bins: int, fractional: bool = False,
) -> List[str]:
    """Bin labels. ``fractional=True`` formats the values as 2-decimal floats
    instead of ints (for fractional time grids)."""
    labels: List[str] = []
    num_levels = len(timestep_values)
    t_cpu = timestep_values.detach().cpu()
    fmt = (lambda v: f"{v:.2f}") if fractional else (lambda v: str(int(v)))
    for bin_idx in range(num_bins):
        start_idx = (bin_idx * num_levels) // num_bins
        end_idx = ((bin_idx + 1) * num_levels) // num_bins - 1
        end_idx = max(start_idx, end_idx)
        s = fmt(t_cpu[start_idx].item())
        e = fmt(t_cpu[end_idx].item())
        labels.append(f"{s}-{e}" if start_idx != end_idx else s)
    return labels


# ---------------------------------------------------------------------------
# DiT model loading
# ---------------------------------------------------------------------------

def _load_dit_checkpoint_prefer_ema(checkpoint_path: str):
    """Load a local DiT checkpoint for evaluation, preferring EMA weights.

    Historically this delegated to ``DiT/download.py``'s ``find_model()``,
    which (a) calls ``torch.load`` with no ``weights_only=False`` -- broken
    under torch>=2.6's flipped default, since our own ``--full_ckpt``
    payloads carry non-tensor Python/numpy RNG state the restrictive
    unpickler rejects (``_pickle.UnpicklingError: ... numpy.core.multiarray
    ._reconstruct``, first observed with torch 2.9.1) and (b) once
    ``weights_only=False`` is supplied, its ``if "ema" in checkpoint`` extraction
    doesn't guard against an explicitly-``None`` "ema" (written whenever a run
    trains without EMA, see ``train_phase_students.save_full_ckpt``). We only
    ever pass a local file path here (never one of the two bare
    ``pretrained_models`` filenames ``find_model`` special-cases for on-the-fly
    web download), so this reimplements exactly its local-checkpoint branch
    with both fixed: prefer ``payload["ema"]`` when present and non-None, else
    ``payload["model"]``, else the raw payload unchanged (a bare state_dict --
    e.g. the officially-released DiT-XL/2 weights, or any legacy non-full-ckpt
    run). Mirrors ``evaluate_students._student_state_from_payload`` exactly
    (duplicated rather than cross-imported to avoid a new inter-script
    dependency for one tiny helper).
    """
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage,
                             weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        ema = checkpoint.get("ema")
        return ema if ema is not None else checkpoint["model"]
    return checkpoint


def load_dit_network(
    checkpoint_path: str,
    dit_model: str = "DiT-XL/2",
    input_size: int = 32,
    num_classes: int = 1000,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> torch.nn.Module:
    DiT_models = import_dit_module("models").DiT_models

    model = DiT_models[dit_model](input_size=input_size, num_classes=num_classes).to(device)
    state_dict = _load_dit_checkpoint_prefer_ema(checkpoint_path)
    model.load_state_dict(state_dict)
    model.eval().requires_grad_(False)
    # Keep model weights in fp32. DiT's TimestepEmbedder calls .float() on its sinusoidal
    # embedding internally, which collides with bf16/fp16 weights. Lower-precision compute
    # is handled via torch.autocast in the evaluator, not via model-level casting.
    return model


# ---------------------------------------------------------------------------
# Latent cache dataset
# ---------------------------------------------------------------------------

class LatentCacheDataset(Dataset):
    """Pre-encode every image through the VAE at init, then serve latents."""

    def __init__(self, image_dataset: Dataset, vae, device: str, batch_size: int = 32, num_workers: int = 4):
        latent_list: List[torch.Tensor] = []
        label_list: List[int] = []

        vae = vae.to(device).eval()
        with torch.no_grad():
            loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            for imgs, class_indices in tqdm(loader, desc="Encoding images to latents", leave=False):
                imgs = imgs.to(device)
                posterior = vae.encode(imgs).latent_dist
                z = posterior.sample() * 0.18215
                latent_list.append(z.cpu())
                label_list.extend(int(c) for c in class_indices)

        self.latents = torch.cat(latent_list, dim=0)
        self.labels = label_list

    def __len__(self) -> int:
        return self.latents.shape[0]

    def __getitem__(self, idx: int):
        return self.latents[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Group collectors for DiT
# ---------------------------------------------------------------------------

def _collect_submodule_groups_dit(
    net: torch.nn.Module, attr: Optional[str] = None,
) -> Dict[str, torch.nn.Module]:
    groups: Dict[str, torch.nn.Module] = {}
    for i, block in enumerate(net.blocks):
        if attr is None:
            groups[f"blocks.{i}"] = block
        else:
            groups[f"blocks.{i}.{attr}"] = getattr(block, attr)
    return groups


def collect_attention_head_groups_dit(
    net: torch.nn.Module,
) -> Dict[str, Tuple[torch.nn.Module, int]]:
    groups: Dict[str, Tuple[torch.nn.Module, int]] = {}
    for i, block in enumerate(net.blocks):
        num_heads = _get_module_num_heads(block.attn)
        if num_heads is None:
            continue
        for h in range(num_heads):
            groups[f"blocks.{i}.attn.head_{h}"] = (block.attn, h)
    return groups


# ---------------------------------------------------------------------------
# Pre-projection head ablation hooks for DiT
# ---------------------------------------------------------------------------

class TransformerHeadZeroHook:
    """Zero a specific attention head's contribution BEFORE the output projection.

    DiT's attention module merges heads via ``proj``, so we register a forward
    pre-hook on ``proj`` and zero the contiguous head chunk in the input where
    heads are still separated (channel dim = ``num_heads * head_dim``).
    """

    def __init__(self, attn_module: torch.nn.Module, head_idx: int):
        self.proj = attn_module.proj
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


class TransformerHeadRandomSameNormHook(RandomSameNormMixin):
    """Random-same-norm replacement for a single attention head, BEFORE ``proj``.

    Mirrors ``HeadRandomSameNormHook`` but works on the head-separated input to
    ``proj`` rather than the post-projection output (where heads are mixed).
    Replaces the head's contiguous slice with Gaussian noise whose per-example
    L2 norm matches the original slice's per-example L2 norm.
    """

    def __init__(self, attn_module: torch.nn.Module, head_idx: int, random_seed: int):
        super().__init__(random_seed=random_seed)
        self.proj = attn_module.proj
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


class TransformerHeadPermutationHook(PermutationMixin):
    """Permutation-FI replacement for a single attention head, BEFORE ``proj``.

    Mirrors ``TransformerHeadRandomSameNormHook`` but instead of injecting
    matched-norm Gaussian noise, it permutes the head's contiguous slice across
    the batch dimension — Breiman permutation importance applied to the
    head-separated input of the output projection.
    """

    def __init__(self, attn_module: torch.nn.Module, head_idx: int, random_seed: int):
        super().__init__(random_seed=random_seed)
        self.proj = attn_module.proj
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


def _is_transformer_head_target(ablate_target) -> bool:
    if not isinstance(ablate_target, tuple) or len(ablate_target) < 2:
        return False
    module = ablate_target[0]
    return hasattr(module, "proj") and hasattr(module, "head_dim") and hasattr(module, "num_heads")


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class DiTUsageEvaluator:
    def __init__(
        self,
        checkpoint_path: str,
        dit_model: str,
        device: str,
        dtype: torch.dtype,
        num_timesteps: int,
        num_timestep_levels: int,
        input_size: int = 32,
        num_classes: int = 1000,
        class_idx_override: Optional[int] = None,
        objective: str = "ddpm",
        arch_cfg: Optional[str] = None,
    ):
        self.device = torch.device(device)
        if objective != "ddpm":
            raise ValueError(f"Unsupported objective {objective!r}; only 'ddpm' is available")
        self.objective = objective
        if arch_cfg is not None:
            # Opt-in NarrowDiT-format checkpoint (a from-scratch --arch_plan
            # student -- e.g. the DiT-B/2 FFHQ and LSUN Bedroom teachers --
            # rather than a stock
            # facebookresearch/DiT checkpoint). Mirrors
            # evaluate_parameters_dit_micro.py's own --arch_cfg opt-in
            # exactly; NarrowAttention exposes the same
            # ``proj``/``num_heads``/``head_dim`` attributes the head hooks
            # and collect_attention_head_groups_dit below already key off of,
            # so no other change is needed to measure a NarrowDiT this way.
            # Lazy import (not module-level) to avoid a circular import:
            # evaluate_parameters_dit_micro.py itself imports FROM this module.
            from evaluate_parameters_dit_micro import load_narrow_dit_network
            self.net = load_narrow_dit_network(
                checkpoint_path=checkpoint_path, arch_cfg_path=arch_cfg, device=device,
            )
        else:
            self.net = load_dit_network(
                checkpoint_path=checkpoint_path,
                dit_model=dit_model,
                input_size=input_size,
                num_classes=num_classes,
                device=device,
                dtype=dtype,
            )
        self.net_dtype = next(self.net.parameters()).dtype  # always fp32 for DiT
        self.compute_dtype = dtype  # autocast dtype for the forward pass
        self.in_channels = int(self.net.in_channels)
        # Replace dataset-provided class labels with this integer if set. ``num_classes``
        # (i.e. 1000 for ImageNet) is DiT's null/unconditional embedding used during CFG.
        self.class_idx_override = class_idx_override

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

        timesteps = self.timestep_values[timestep_indices]

        noise = torch.empty_like(latents)
        for i in range(latents.shape[0]):
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(noise_seeds[i].item()))
            noise[i] = torch.randn(
                latents[i].shape, generator=gen, device=self.device, dtype=latents.dtype,
            )

        autocast_enabled = self.compute_dtype != torch.float32 and self.device.type == "cuda"

        ab = self.alpha_bar[timesteps].float()
        sqrt_ab = torch.sqrt(ab).view(-1, 1, 1, 1)
        sqrt_one_minus_ab = torch.sqrt(1.0 - ab).view(-1, 1, 1, 1)
        x_t = sqrt_ab * latents + sqrt_one_minus_ab * noise

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.compute_dtype,
            enabled=autocast_enabled,
        ):
            output = self.net(x_t, timesteps, class_indices)
        eps_pred = output[:, : self.in_channels]

        losses = mse_per_example(eps_pred.float(), noise.float())
        return losses

    def _make_ablation_context(
        self,
        ablate_target: Optional[Union[torch.nn.Module, Tuple]],
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
                        raise ValueError("ablation_random_seed is required for random_same_norm ablation")
                    return FilterRandomSameNormHook(ablate_target[0], ablate_target[1], ablation_random_seed)
                if ablation_mode == "permutation":
                    if ablation_random_seed is None:
                        raise ValueError("ablation_random_seed is required for permutation ablation")
                    return FilterPermutationHook(ablate_target[0], ablate_target[1], ablation_random_seed)
                raise ValueError(f"Unsupported ablation_mode: {ablation_mode}")
            if _is_transformer_head_target(ablate_target):
                if ablation_mode == "zero":
                    return TransformerHeadZeroHook(ablate_target[0], ablate_target[1])
                if ablation_mode == "random_same_norm":
                    if ablation_random_seed is None:
                        raise ValueError("ablation_random_seed is required for random_same_norm ablation")
                    return TransformerHeadRandomSameNormHook(
                        ablate_target[0], ablate_target[1], random_seed=ablation_random_seed,
                    )
                if ablation_mode == "permutation":
                    if ablation_random_seed is None:
                        raise ValueError("ablation_random_seed is required for permutation ablation")
                    return TransformerHeadPermutationHook(
                        ablate_target[0], ablate_target[1], random_seed=ablation_random_seed,
                    )
                raise ValueError(f"Unsupported ablation_mode: {ablation_mode}")
            # Fallback: generic post-output head ablation (EDM-style).
            if ablation_mode == "zero":
                return HeadZeroHook(*ablate_target)
            if ablation_mode == "random_same_norm":
                if ablation_random_seed is None:
                    raise ValueError("ablation_random_seed is required for random_same_norm ablation")
                return HeadRandomSameNormHook(*ablate_target, random_seed=ablation_random_seed)
            if ablation_mode == "permutation":
                if ablation_random_seed is None:
                    raise ValueError("ablation_random_seed is required for permutation ablation")
                return HeadPermutationHook(*ablate_target, random_seed=ablation_random_seed)
            raise ValueError(f"Unsupported ablation_mode: {ablation_mode}")
        # Whole-module ablation.
        if ablation_mode == "zero":
            return ZeroOutputHook(ablate_target)
        if ablation_mode == "random_same_norm":
            if ablation_random_seed is None:
                raise ValueError("ablation_random_seed is required for random_same_norm ablation")
            return RandomSameNormOutputHook(ablate_target, ablation_random_seed)
        if ablation_mode == "permutation":
            if ablation_random_seed is None:
                raise ValueError("ablation_random_seed is required for permutation ablation")
            return PermutationOutputHook(ablate_target, ablation_random_seed)
        raise ValueError(f"Unsupported ablation_mode: {ablation_mode}")

    def evaluate(
        self,
        dataloader: DataLoader,
        num_bins: int,
        ablate_target: Optional[Union[torch.nn.Module, Tuple]] = None,
        ablation_mode: AblationMode = "zero",
        ablation_random_seed: Optional[int] = None,
        progress_desc: Optional[str] = None,
        permutation_group_by_level: bool = True,
    ) -> BinStats:
        stats = BinStats(num_bins=num_bins)
        context = self._make_ablation_context(ablate_target, ablation_mode, ablation_random_seed)
        # Permutation importance must hold t fixed: restrict the batch permutation
        # to same-timestep-level examples. Without this, an image-major batch that
        # spans several t-levels (batch_size < num_timestep_levels) swaps head
        # activations across levels and injects a spurious period-batch_size ripple
        # into n_eff.
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
            for latents, class_indices, timestep_indices, noise_seeds in progress:
                if group_by_level:
                    context.set_permutation_groups(timestep_indices)
                losses = self.forward_losses_from_fixed_corruption(
                    latents=latents,
                    class_indices=class_indices,
                    timestep_indices=timestep_indices,
                    noise_seeds=noise_seeds,
                )
                if not torch.isfinite(losses).all():
                    bad_idx = (~torch.isfinite(losses)).nonzero(as_tuple=False).flatten().tolist()
                    raise ValueError(
                        f"Non-finite losses at batch indices {bad_idx}. "
                        "Try --dtype fp32 or check data integrity."
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

def _build_image_dataset(args, image_size: int):
    if args.dataset == "cifar10":
        return CIFAR10Dataset(
            root=args.data_root, image_size=image_size,
            split=args.cifar_split, max_images=args.max_images,
            download=args.download,
        )
    if args.dataset == "imagenet1k_parquet":
        return ImageNet1KParquetDataset(
            root=args.data_root, image_size=image_size,
            max_images=args.max_images,
            image_column=args.parquet_image_column,
            label_column=args.parquet_label_column,
            split_prefix=args.parquet_split_prefix,
        )
    if args.dataset == "imagenet":
        return ImageNetDataset(
            root=args.data_root, image_size=image_size,
            split=args.imagenet_split, max_images=args.max_images,
            download=args.download,
        )
    if not args.image_root:
        raise ValueError("--image_root is required when --dataset image_folder")
    return ImageFolderFlat(
        root=args.image_root, image_size=image_size,
        max_images=args.max_images,
    )


def main():
    start_time = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="imagenet1k_parquet",
                        choices=["image_folder", "cifar10", "imagenet1k_parquet", "imagenet"])
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--parquet_image_column", type=str, default=None)
    parser.add_argument("--parquet_label_column", type=str, default=None)
    parser.add_argument("--parquet_split_prefix", type=str, default=None,
                        help="Filter parquet files in --data_root by this filename prefix (e.g. 'train-').")
    parser.add_argument("--cifar_split", type=str, default="validation",
                        choices=["validation", "test", "train"])
    parser.add_argument("--imagenet_split", type=str, default="train", choices=["val", "train"])
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="DiT-XL-2-256x256.pt",
                        help="DiT checkpoint path or name for download.find_model.")
    parser.add_argument("--dit_model", type=str, default="DiT-XL/2",
                        help="Model architecture key in DiT_models. Ignored when --arch_cfg is set.")
    parser.add_argument("--arch_cfg", type=str, default=None,
                        help="OPT-IN: path to a NarrowDiT arch_cfg.json (as written by "
                             "train_phase_students.py's phase save / dit_arch_to_plans.py), "
                             "e.g. a from-scratch --arch_plan student's "
                             "phase_0/arch_cfg.json. When set, --checkpoint is loaded as a "
                             "NarrowDiT built from this config instead of the stock "
                             "DiT_models[--dit_model] reconstruction -- --dit_model/"
                             "--num_classes are then ignored (the arch_cfg's own "
                             "hidden_size/depth/num_classes/etc. fully determine the "
                             "architecture). Default None "
                             "preserves the previous behavior exactly.")
    parser.add_argument("--dit_repo", type=str, default=None,
                        help="facebookresearch/DiT checkout (default: $DIT_REPO).")
    parser.add_argument("--num_timesteps", type=int, default=1000,
                        help="DDPM schedule length.")
    parser.add_argument("--grouping", type=str, default="attention_heads",
                        choices=["blocks", "attention", "mlp", "adaln", "attention_heads"])
    parser.add_argument(
        "--ablation_mode",
        type=str,
        default="permutation",
        choices=["zero", "random_same_norm", "permutation"],
        help="How to corrupt ablated group outputs. 'permutation' (default, the protocol of the "
             "paper's DiT profiles) shuffles the group's activation across the batch at the same "
             "timestep level (Breiman permutation feature importance); 'random_same_norm' injects "
             "Gaussian noise with matching per-example L2 norm; 'zero' zeroes it.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--permutation_group_by_level",
        type=lambda s: str(s).lower() not in ("0", "false", "no"),
        default=True,
        help="For --ablation_mode permutation, restrict the batch permutation to examples at the "
             "SAME timestep-level (holds t fixed; correct Breiman importance). Default True. Set "
             "False only to reproduce the legacy behaviour that swaps activations across t-levels "
             "and injects a spurious period-batch_size ripple into n_eff.",
    )
    parser.add_argument(
        "--corruption_order", type=str, default="level_major",
        choices=["level_major", "image_major"],
        help="Sample ordering for the corruption dataset. 'level_major' (default) groups same-level "
             "examples into each batch so permutation importance holds t fixed. 'image_major' is the "
             "legacy order that (with batch_size<num_timestep_levels) causes the period-batch_size "
             "n_eff artifact; use only to reproduce old runs.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--vae_batch_size", type=int, default=32)
    parser.add_argument("--vae_num_workers", type=int, default=4)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--max_groups", type=int, default=None)
    parser.add_argument("--samples_per_image", type=int, default=2,
                        help="Deprecated: ignored.")
    parser.add_argument("--num_bins", type=int, default=20, help="Number of noise bins (the paper uses 20).")
    parser.add_argument("--num_timestep_levels", type=int, default=256,
                        help="Number of uniformly spaced timestep levels to evaluate.")
    parser.add_argument("--timestep_stride", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="fp32",
                        choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--class_idx_override",
        type=int,
        default=None,
        help="If set, replace every dataset-provided class label with this integer. "
             "Pass the DiT null/unconditional class (num_classes, e.g. 1000 for ImageNet) "
             "to run DiT in fully unconditional mode regardless of dataset labels.",
    )
    parser.add_argument(
        "--group_correlation",
        type=str,
        default="auto",
        choices=["auto", "always", "never"],
    )
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
        image_size = args.image_size

        print(f"[rank {rank}] Preparing image dataset...")
        image_dataset = _build_image_dataset(args, image_size=image_size)
        if is_main_process():
            print(f"Image dataset: {len(image_dataset)} images.")

        print(f"[rank {rank}] Loading VAE for latent encoding...")
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        latent_dataset = LatentCacheDataset(
            image_dataset, vae, device=args.device,
            batch_size=args.vae_batch_size, num_workers=args.vae_num_workers,
        )
        del vae
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[rank {rank}] Loading DiT network on {args.device}...")
        evaluator = DiTUsageEvaluator(
            checkpoint_path=args.checkpoint,
            dit_model=args.dit_model,
            device=args.device,
            dtype=dtype,
            num_timesteps=args.num_timesteps,
            num_timestep_levels=args.num_timestep_levels,
            input_size=image_size // 8,
            class_idx_override=args.class_idx_override,
            arch_cfg=args.arch_cfg,
        )

        if is_main_process():
            print(
                f"DiT model: {args.dit_model}, timesteps: {args.num_timesteps}, "
                f"ablation_mode: {args.ablation_mode}, grouping: {args.grouping}"
            )

        corruption_dataset = SigmaCorruptionDataset(
            image_dataset=latent_dataset,
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
            groups = _collect_submodule_groups_dit(evaluator.net, _GROUPING_ATTRS[args.grouping])
        elif args.grouping == "attention_heads":
            groups = collect_attention_head_groups_dit(evaluator.net)
        else:
            raise ValueError(f"Unknown grouping: {args.grouping}")

        if not groups:
            raise ValueError(f"No groups found for grouping={args.grouping}.")

        if args.max_groups is not None:
            if args.max_groups <= 0:
                raise ValueError(f"max_groups must be positive, got {args.max_groups}")
            group_items_full = list(groups.items())
            selected_count = min(args.max_groups, len(group_items_full))
            groups = dict(group_items_full[:selected_count])
            if is_main_process():
                print(f"Limiting to first {len(groups)} groups due to --max_groups.")

        if args.grouping == "attention_heads":
            group_param_counts = {
                name: max(1, count_parameters(mod) // max(1, _get_module_num_heads(mod) or 1))
                for name, (mod, _) in groups.items()
            }
        else:
            group_param_counts = {name: count_parameters(mod) for name, mod in groups.items()}

        if is_main_process():
            print(f"Found {len(groups)} groups across {world_size} process(es):")
            for name, pcount in list(group_param_counts.items())[:8]:
                print(f"  {name}: {pcount:,}")
            if len(group_param_counts) > 8:
                print(f"  ... ({len(group_param_counts) - 8} more)")

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

        group_items = list(groups.items())
        indexed_group_items = list(enumerate(group_items))
        local_group_items = indexed_group_items[rank::world_size]

        if is_main_process():
            baseline_ckpt = os.path.join(args.output_dir, "baseline.pt")
            atomic_torch_save({
                "mean": baseline_mean, "stderr": baseline_stderr, "count": baseline_count,
                "max_images": len(latent_dataset), "num_bins": args.num_bins,
                "num_timestep_levels": args.num_timestep_levels,
                "ablation_mode": args.ablation_mode,
            }, baseline_ckpt)

        print(f"[rank {rank}] Running {len(local_group_items)} / {len(group_items)} group ablations...")
        checkpoint_name = checkpoint_name_for_rank(args.ablation_mode, rank)
        checkpoint_path = os.path.join(args.output_dir, checkpoint_name)
        group_names_for_resume = [name for name, _ in group_items]
        completed_ablated_means, loaded_checkpoint_paths, ignored_unknown, ignored_bad_shape = load_ablation_checkpoints(
            output_dir=args.output_dir,
            ablation_mode=args.ablation_mode,
            expected_group_names=group_names_for_resume,
            num_bins=args.num_bins,
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
                print(
                    f"[rank {rank}] Ignored {len(ignored_bad_shape)} checkpoint entries with incompatible shapes "
                    f"(expected ({args.num_bins},))."
                )

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
                permutation_group_by_level=args.permutation_group_by_level,
                progress_desc=f"rank {rank} ablation {local_idx}/{len(local_group_items)} {name}",
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
                compute_group_correlation=compute_group_correlation,
                clip_mode=args.importance_clip_mode,
            )

            raw_baseline_mean = baseline_mean.clone()
            raw_baseline_stderr = baseline_stderr.clone()

            timestep_values_strided = evaluator.timestep_values[::args.timestep_stride]
            timestep_bin_labels = make_timestep_bin_labels(
                timestep_values_strided, args.num_bins,
            )

            invocation_argv = list(sys.argv)
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
                    "parquet_image_column": getattr(image_dataset, "image_column", None),
                    "parquet_label_column": getattr(image_dataset, "label_column", None),
                    "parquet_split_prefix": getattr(image_dataset, "split_prefix", None),
                },
                "model_info": {
                    "dit_model": args.dit_model,
                    "num_timesteps": args.num_timesteps,
                    "net_class_name": evaluator.net.__class__.__name__,
                },
                "ablation_info": {
                    "mode": args.ablation_mode,
                    "norm": "per_example_l2" if args.ablation_mode == "random_same_norm" else None,
                    "replacement": {
                        "random_same_norm": "gaussian_random_noise",
                        "permutation": "batch_permuted_activation",
                        "zero": "zeros",
                    }[args.ablation_mode],
                    "group_definition": "attention_head" if args.grouping == "attention_heads" else args.grouping,
                    "ablation_site": "pre_proj" if args.grouping == "attention_heads" else "module_output",
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
