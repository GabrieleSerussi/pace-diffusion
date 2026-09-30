#!/usr/bin/env python3
"""
Phase-specialist student fine-tuning.

Initializes a student model from a teacher's weights, then fine-tunes it
on the timestep range of a single phase (read from ``timestep_grouping.json``).
The student keeps the teacher's architecture exactly; only its weights drift
to specialize on its phase's noise levels.

Supports two teacher families:

  * ``--model_type dit_xl``:  DiT-XL/2 (ImageNet, eps-prediction, VAE latents),
    and, with ``--teacher_arch_cfg``, the from-scratch DiT-B/2 latent teachers
    (FFHQ, LSUN Bedroom).
  * ``--model_type dit_micro``:  CIFAR-trained DiT-Micro (raw RGB,
    x_0-prediction, HookableMultiheadSelfAttention by default). Pass
    ``--teacher_arch_cfg`` to instead load a NarrowDiT-format teacher (for
    example the CIFAR-10 DiT-S/2-style teacher, or any width-allocation student
    used as a teacher) -- arbitrary hidden_size/depth, timm-style qkv/proj
    attention, optional augment_dim/dropout.

With ``--arch_plan`` (from ``scripts/dit_arch_to_plans.py``) every phase
student is a fresh NarrowDiT built from its phase's width configuration and
distilled from the teacher; this is the path of the paper's DiT students.  The
same script also trains the from-scratch teachers (``--kd_weight 0``).

The facebookresearch/DiT checkout is taken from ``--dit_repo`` or ``$DIT_REPO``.

Distributed training via ``torchrun`` (DistributedDataParallel). The script
iterates over the phases in the supplied grouping JSON and trains one student
per phase. Each phase writes to ``output_dir/phase_{i}/``:

  - ``student.pt``         : final fine-tuned state_dict
  - ``training_meta.json`` : phase boundaries, loss history, hyperparams.
"""

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from evaluate_parameters_edm import (
    CIFAR10Dataset,
    HAS_WANDB,
    ImageFolderFlat,
    ImageNet1KParquetDataset,
    ImageNetDataset,
    atomic_torch_save,
    choose_dtype,
    cleanup_distributed,
    get_rank,
    get_world_size,
    init_distributed,
    is_distributed,
    is_main_process,
    save_json,
    set_seed,
)
from evaluate_parameters_dit import (
    load_dit_network,
    make_ddpm_alpha_schedule,
)
from evaluate_parameters_dit_micro import (
    load_dit_micro_network,
    load_narrow_dit_network,
)

# Width-allocation (from-scratch NarrowDiT) distillation support. The upstream
# facebookresearch/DiT checkout (--dit_repo or $DIT_REPO) is imported lazily,
# on first use, by pace.dit_arch_alloc and the teacher loaders.
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pace.external_repos import configure_dit_repo

if HAS_WANDB:
    import wandb


# ---------------------------------------------------------------------------
# WSD (warmup-stable-decay) learning-rate schedule
# ---------------------------------------------------------------------------

def wsd_lr_lambda(step, *, total_steps, warmup_steps, cooldown_frac):
    """WSD multiplier on peak LR: linear warmup 0->1 over warmup_steps; constant 1.0
    during the stable phase; 1 - sqrt(frac) decay to 0 over the last cooldown_frac."""
    if step >= total_steps:
        return 0.0
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    decay_start = int(round(total_steps * (1.0 - cooldown_frac)))
    if step < decay_start:
        return 1.0
    frac = (step - decay_start) / max(1, total_steps - decay_start)
    return 1.0 - math.sqrt(frac)


# ---------------------------------------------------------------------------
# Phase boundary parsing
# ---------------------------------------------------------------------------

def load_phases(grouping_json_path: str) -> Tuple[List[Dict[str, int]], int]:
    """Read phase boundaries from a ``timestep_grouping.json`` and return
    ``(phases, num_bins)``. Each phase has ``index``, ``start``, ``end``."""
    with open(grouping_json_path) as f:
        g = json.load(f)
    boundaries = list(g["boundaries"])
    num_bins = int(g["num_timesteps"])
    if boundaries[0] != 0 or boundaries[-1] != num_bins:
        raise ValueError(
            f"Boundaries {boundaries} do not span [0, {num_bins}] from {grouping_json_path}"
        )
    phases = []
    for i in range(len(boundaries) - 1):
        phases.append({"index": i, "start": int(boundaries[i]), "end": int(boundaries[i + 1])})
    return phases, num_bins


def load_phases_from_arch_plan(
    arch_plan_path: str, variant: str
) -> Tuple[List[Dict[str, int]], int, List[Dict[str, Any]]]:
    """Read phase boundaries + per-phase NarrowDiT cfg from an arch-plan JSON.

    Mirrors ``load_phases`` (same ``[{index,start,end}]`` + ``num_bins`` contract),
    but the boundaries come from the plan's per-phase ``bins`` instead of a
    ``timestep_grouping.json``. This is the SAME phase-grouping mechanism used by
    the teacher-initialized path: the plan's phases align 1:1 with the grouping boundaries, so
    for the ``global`` variant this yields a single phase spanning ``[0, num_bins]``
    (a 1-phase grouping), and for the multi-phase variants it reproduces the
    grouping's boundary partition exactly. Also returns the ordered per-phase cfg
    dicts (one NarrowDiT kwargs dict per phase)."""
    with open(arch_plan_path) as f:
        plan = json.load(f)
    if variant not in plan:
        raise ValueError(
            f"variant {variant!r} not in {arch_plan_path}; available: {list(plan)}"
        )
    plan_phases = plan[variant]["phases"]
    if not plan_phases:
        raise ValueError(f"variant {variant!r} in {arch_plan_path} has no phases")
    # bins are [start, end) pairs; they must tile [0, num_bins] contiguously.
    plan_phases = sorted(plan_phases, key=lambda p: p["bins"][0])
    num_bins = int(plan_phases[-1]["bins"][1])
    phases: List[Dict[str, int]] = []
    cfgs: List[Dict[str, Any]] = []
    expect = 0
    for i, pp in enumerate(plan_phases):
        s, e = int(pp["bins"][0]), int(pp["bins"][1])
        if s != expect:
            raise ValueError(
                f"arch plan {variant!r} phase {i} bins start at {s}, expected {expect} "
                f"(phases must contiguously tile [0, {num_bins}])"
            )
        expect = e
        phases.append({"index": i, "start": s, "end": e})
        cfgs.append(pp["cfg"])
    if expect != num_bins:
        raise ValueError(
            f"arch plan {variant!r} phases end at {expect}, expected {num_bins}"
        )
    return phases, num_bins, cfgs


def phase_resume_action(phase_dir: str, full_ckpt: bool = False) -> str:
    """Decide how to (re)start a phase from disk state, for preemption-safe resume:
    'skip' if the phase already finished (``student.pt`` present), 'resume' if a
    mid-phase checkpoint is present, else 'fresh'.

    A finished phase is always ``student.pt``. With ``full_ckpt`` the mid-phase
    checkpoint is the rolling ``last.pt`` (full: model+ema+opt+sched+step+epoch+rng);
    without it, the legacy weights+opt+sched+epoch ``resume.pt``."""
    if os.path.exists(os.path.join(phase_dir, "student.pt")):
        return "skip"
    mid = "last.pt" if full_ckpt else "resume.pt"
    if os.path.exists(os.path.join(phase_dir, mid)):
        return "resume"
    return "fresh"


def sample_phase_timesteps(
    batch_size: int, phase_start: int, phase_end: int,
    num_bins: int, num_timesteps: int, device: torch.device,
) -> torch.Tensor:
    """(DDPM) Uniform sample of ``num_timesteps``-indexed timesteps falling inside the
    bin range [phase_start, phase_end). bin 0 = highest noise (t near num_timesteps-1)."""
    t_max = num_timesteps - 1
    t_low = max(0, int(round(t_max * (1.0 - phase_end / num_bins))))
    t_high = max(t_low + 1, min(t_max, int(round(t_max * (1.0 - phase_start / num_bins)))))
    return torch.randint(t_low, t_high + 1, (batch_size,), device=device)


def _sigma_at_fraction(f: float, sigma_min: float, sigma_max: float, rho: float) -> float:
    """Karras sigma at schedule fraction f in [0,1]; f=0 -> sigma_max (high noise)."""
    return (sigma_max ** (1.0 / rho) + f * (sigma_min ** (1.0 / rho) - sigma_max ** (1.0 / rho))) ** rho


def sample_phase_sigmas(
    batch_size: int, phase_start: int, phase_end: int, num_bins: int,
    sigma_min: float, sigma_max: float, rho: float, device: torch.device,
) -> torch.Tensor:
    """(EDM) Log-uniform sample of sigmas within the phase's bin range.
    bin 0 = highest noise (sigma_max), bin num_bins-1 = lowest (sigma_min)."""
    sig_hi = _sigma_at_fraction(phase_start / num_bins, sigma_min, sigma_max, rho)
    sig_lo = _sigma_at_fraction(phase_end / num_bins, sigma_min, sigma_max, rho)
    lo, hi = min(sig_lo, sig_hi), max(sig_lo, sig_hi)
    u = torch.rand(batch_size, device=device)
    log_lo, log_hi = math.log(lo), math.log(hi)
    return torch.exp(log_lo + u * (log_hi - log_lo))


def sample_phase_sigmas_lognormal(
    batch_size: int, phase_start: int, phase_end: int, num_bins: int,
    sigma_min: float, sigma_max: float, rho: float,
    p_mean: float, p_std: float, device: torch.device,
    max_resample: int = 100,
) -> torch.Tensor:
    """(EDM) log-normal sample of sigmas: ``sigma = exp(p_mean + p_std * randn)``,
    clipped to ``[sigma_min, sigma_max]`` -- EDM's own training-time noise
    distribution (Karras et al., defaults P_mean=-1.2, P_std=1.2).

    Respects the phase's bin restriction the SAME WAY ``sample_phase_sigmas``
    does: samples outside the phase's ``[lo, hi]`` sigma sub-range (computed via
    the same ``_sigma_at_fraction`` schedule) are rejected and redrawn, up to
    ``max_resample`` rounds; any stragglers left after that are clamped into
    range so the call always returns a full batch. For a ``global`` phase (the
    full ``[0, num_bins]`` range), ``[lo, hi] == [sigma_min, sigma_max]`` so the
    initial clip already satisfies the restriction and no rejection occurs.

    Draws from torch's global RNG (no explicit generator), matching the other
    per-batch draws in this file, so full-ckpt RNG resume stays exact.
    """
    sig_hi = _sigma_at_fraction(phase_start / num_bins, sigma_min, sigma_max, rho)
    sig_lo = _sigma_at_fraction(phase_end / num_bins, sigma_min, sigma_max, rho)
    lo, hi = min(sig_lo, sig_hi), max(sig_lo, sig_hi)

    def _draw(n: int) -> torch.Tensor:
        s = torch.exp(p_mean + p_std * torch.randn(n, device=device))
        return s.clamp(min=sigma_min, max=sigma_max)

    sigma = _draw(batch_size)
    pending = (sigma < lo) | (sigma > hi)
    tries = 0
    while bool(pending.any()) and tries < max_resample:
        n = int(pending.sum())
        sigma[pending] = _draw(n)
        pending = (sigma < lo) | (sigma > hi)
        tries += 1
    if bool(pending.any()):
        sigma[pending] = sigma[pending].clamp(min=lo, max=hi)
    return sigma


def draw_phase_sigmas(
    args, batch_size: int, phase_start: int, phase_end: int, num_bins: int,
    device: torch.device,
) -> torch.Tensor:
    """Dispatch to the configured ``--sigma_sampling`` strategy for EDM training.
    Default ``loguniform`` reproduces ``sample_phase_sigmas`` exactly (no new RNG
    draws, byte-identical to before). ``lognormal`` uses EDM's own training
    distribution (``--p_mean``/``--p_std``), truncated to the phase's bin range."""
    if getattr(args, "sigma_sampling", "loguniform") == "lognormal":
        return sample_phase_sigmas_lognormal(
            batch_size=batch_size, phase_start=phase_start, phase_end=phase_end,
            num_bins=num_bins, sigma_min=args.sigma_min, sigma_max=args.sigma_max,
            rho=args.rho, p_mean=args.p_mean, p_std=args.p_std, device=device,
        )
    return sample_phase_sigmas(
        batch_size=batch_size, phase_start=phase_start, phase_end=phase_end,
        num_bins=num_bins, sigma_min=args.sigma_min, sigma_max=args.sigma_max,
        rho=args.rho, device=device,
    )


# ---------------------------------------------------------------------------
# Per-rank latent cache (DiT-XL/2 only)
# ---------------------------------------------------------------------------

class PerRankLatentCache(torch.utils.data.Dataset):
    """Pre-encode this rank's disjoint slice of the image dataset to VAE latents
    in CPU memory, then serve (latent, label) pairs directly.

    Eliminates the parquet I/O hotspot during training: shuffled batches across
    290 train parquet files cause repeated per-worker cache misses; pre-caching
    once turns that into a single sequential read.
    """

    def __init__(
        self,
        image_dataset: torch.utils.data.Dataset,
        vae: nn.Module,
        rank: int,
        world_size: int,
        device: torch.device,
        encode_batch_size: int = 128,
        num_workers: int = 4,
    ):
        # Round-robin shard so the rank's slice covers the dataset uniformly.
        indices = list(range(rank, len(image_dataset), world_size))
        if is_main_process():
            print(
                f"  pre-encoding rank's slice: {len(indices):,} images "
                f"(of {len(image_dataset):,} total) via VAE on {device}"
            )

        subset = torch.utils.data.Subset(image_dataset, indices)
        loader = torch.utils.data.DataLoader(
            subset, batch_size=encode_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, drop_last=False,
        )

        latents_buf: List[torch.Tensor] = []
        labels_buf: List[int] = []
        vae = vae.to(device).eval()
        progress = tqdm(
            loader, desc=f"VAE-encoding rank {rank}", disable=not is_main_process(),
            dynamic_ncols=True, leave=False,
        )
        with torch.no_grad():
            for imgs, labs in progress:
                imgs = imgs.to(device=device, dtype=torch.float32, non_blocking=True)
                posterior = vae.encode(imgs).latent_dist
                z = (posterior.sample() * 0.18215).to(dtype=torch.float16)  # half to fit RAM
                latents_buf.append(z.detach().cpu())
                labels_buf.extend(int(c) for c in labs)
                progress.set_postfix(cached=len(labels_buf))
        self.latents = torch.cat(latents_buf, dim=0)
        self.labels = torch.tensor(labels_buf, dtype=torch.long)
        if is_main_process():
            shape = tuple(self.latents.shape)
            mb = self.latents.element_size() * self.latents.numel() / (1024 ** 2)
            print(f"  cache built: latents {shape} fp16 ({mb:.0f} MB), labels {self.labels.shape[0]}")

    def __len__(self) -> int:
        return self.latents.shape[0]

    def __getitem__(self, idx: int):
        return self.latents[idx], int(self.labels[idx])


# ---------------------------------------------------------------------------
# Persistent (cross-restart) latent cache -- opt-in via --latent_cache_dir.
#
# PerRankLatentCache above re-encodes the whole dataset through the VAE at
# EVERY process start (~10-15 min for FFHQ's 70k images, ~50 min for
# Bedroom's 1M) -- under preemption+requeue (for example on a preemptible
# cluster partition) that tax repeats on every restart even though the
# datasets are static. --latent_cache_dir persists the encode to
# disk ONCE and reuses it forever after.
#
# Design:
#  * On-disk layout is in CANONICAL (dataset-index) order, split into
#    `num_shards` CONTIGUOUS blocks -- NOT PerRankLatentCache's round-robin
#    split, which is tied to one particular world_size. num_shards is simply
#    whatever world_size happened to build the cache; a later load with a
#    DIFFERENT world_size still works (PersistentLatentCache reconstructs the
#    canonical array from however many shard files exist, then does ITS OWN
#    round-robin over that).
#  * Fingerprinted by dataset path + image count + VAE id + resolution.
#    Mismatch (or a missing/garbled meta.json, or a missing DONE marker --
#    e.g. a write a preemption interrupted mid-way) means the cache is not
#    trusted at all; the simplest-correct response is a full rebuild, never
#    an incremental/partial resume.
#  * Stochasticity: PerRankLatentCache samples z = mu + std*eps ONCE per
#    (rank, process) at construction and reuses it for every epoch of that
#    run (there is no per-epoch resampling in __getitem__ -- see its
#    __getitem__ above, a plain index into a precomputed tensor). This cache
#    stores mu/logvar (deterministic given the frozen VAE weights and
#    pixels), NOT an already-sampled latent, and PersistentLatentCache draws
#    its own fresh eps once at construction -- reproducing the EXACT same
#    cadence (fresh stochastic sample every process start) while skipping
#    only the expensive deterministic VAE forward pass.
# ---------------------------------------------------------------------------

_LATENT_CACHE_VERSION = 1
_LATENT_CACHE_DONE_NAME = "DONE"
_LATENT_CACHE_META_NAME = "meta.json"
_VAE_PRETRAINED_ID = "stabilityai/sd-vae-ft-mse"
_VAE_SCALE_FACTOR = 0.18215


def _dataset_identity_path(args) -> str:
    """Canonical filesystem path identifying WHICH images a dataset spec points
    at, for fingerprinting the persistent latent cache. image_folder (FFHQ256/
    Bedroom256) is the only dataset kind currently paired with a latent-space
    model_type in this workspace, but this stays dataset-kind-agnostic."""
    if args.dataset == "image_folder":
        return os.path.abspath(args.image_root)
    root = getattr(args, "data_root", None)
    return os.path.abspath(root) if root else ""


def latent_cache_fingerprint(args, image_dataset, image_size: int) -> Dict[str, Any]:
    """Identity of a persistent latent cache: dataset path + image count + VAE
    id + resolution. Any change here invalidates every existing cache dir."""
    return {
        "version": _LATENT_CACHE_VERSION,
        "dataset_kind": args.dataset,
        "dataset_path": _dataset_identity_path(args),
        "image_count": len(image_dataset),
        "vae_id": _VAE_PRETRAINED_ID,
        "image_size": int(image_size),
    }


def _fingerprint_hash(fingerprint: Dict[str, Any]) -> str:
    """Short, stable hash of a latent-cache fingerprint dict, used to key its
    on-disk subdirectory (see ``resolve_latent_cache_dir``)."""
    blob = json.dumps(fingerprint, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def resolve_latent_cache_dir(cache_dir: str, fingerprint: Dict[str, Any]) -> str:
    """Hardening item 7: pick the ACTUAL on-disk directory to build/load a
    persistent latent cache in, keyed by ``fingerprint`` so a mismatched
    config never clobbers an existing cache of a different shape.

    Before this existed, every ``--latent_cache_dir`` invocation used the
    same flat ``cache_dir`` regardless of fingerprint. A run with a DIFFERENT
    fingerprint pointed at the same ``cache_dir`` (e.g. a smoke test adding
    ``--max_images``, which changes ``image_count`` and therefore the
    fingerprint) found the on-disk cache "incomplete" (fingerprint mismatch)
    and silently REBUILT IT FROM SCRATCH AT THE SAME PATH -- overwriting a
    full-dataset cache with a smoke-sized one. This happened in production
    (a smoke run clobbered the 1M-image Bedroom cache).

    Fix: key the real shard directory by a hash of the fingerprint,
    ``cache_dir/<fingerprint_hash>/``, so any mismatched config writes BESIDE
    the existing cache, never over it.

    Backward-compatible: if the flat ``cache_dir`` path itself ALREADY holds
    a complete cache matching THIS fingerprint (the pre-existing, pre-this-
    fix on-disk layout used by every cache built before this function
    existed), that flat directory is recognized and reused in place --
    zero migration needed for caches already on disk. Only a fingerprint
    that does NOT match whatever is (or isn't) at the flat path gets routed
    to its own new, disjoint subdirectory -- which is exactly the case that
    used to clobber.
    """
    if latent_cache_is_complete(cache_dir, fingerprint):
        return cache_dir
    return os.path.join(cache_dir, _fingerprint_hash(fingerprint))


def _latent_cache_meta_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, _LATENT_CACHE_META_NAME)


def _latent_cache_done_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, _LATENT_CACHE_DONE_NAME)


def _latent_cache_shard_paths(cache_dir: str, shard_idx: int) -> Tuple[str, str, str]:
    base = os.path.join(cache_dir, f"shard_{shard_idx:05d}")
    return base + "_mu.npy", base + "_logvar.npy", base + "_labels.npy"


def latent_cache_is_complete(cache_dir: str, fingerprint: Dict[str, Any]) -> bool:
    """True iff cache_dir holds a cache matching ``fingerprint`` EXACTLY, with
    its DONE marker present (written last, only after every shard file and
    meta.json land -- see build_persistent_latent_cache). Anything else --
    missing directory, missing/corrupt meta.json, a fingerprint mismatch, or a
    write a preemption interrupted before DONE was written -- returns False,
    which the caller treats as \"(re)build from scratch\", never a partial
    resume."""
    meta_path = _latent_cache_meta_path(cache_dir)
    done_path = _latent_cache_done_path(cache_dir)
    if not (os.path.isfile(meta_path) and os.path.isfile(done_path)):
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return meta.get("fingerprint") == fingerprint


def trusted_latent_cache_fingerprint(cache_dir: str) -> Dict[str, Any]:
    """Opt-in ``--latent_cache_trust``: return the fingerprint recorded in a COMPLETE
    persistent cache under ``cache_dir`` (the flat dir itself, or exactly one fingerprint-keyed
    subdirectory) without touching the image dataset. Lets a machine that holds only the cache
    (no images) train latent-space students. Raises if no complete cache is found or if the
    choice is ambiguous -- never guesses."""
    subdirs = os.listdir(cache_dir) if os.path.isdir(cache_dir) else []
    cands = [cache_dir] + sorted(os.path.join(cache_dir, d) for d in subdirs
                                 if os.path.isdir(os.path.join(cache_dir, d)))
    found = []
    for d in cands:
        mp, dp = _latent_cache_meta_path(d), _latent_cache_done_path(d)
        if os.path.isfile(mp) and os.path.isfile(dp):
            with open(mp) as f:
                meta = json.load(f)
            if isinstance(meta.get("fingerprint"), dict):
                found.append((d, meta["fingerprint"]))
    if len(found) != 1:
        raise FileNotFoundError(
            f"--latent_cache_trust: expected exactly one complete cache under {cache_dir}, "
            f"found {len(found)}: {[d for d, _ in found]}")
    return found[0][1]


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _atomic_write_text(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def _canonical_block(n: int, shard_idx: int, num_shards: int) -> Tuple[int, int]:
    """[start, end) of the CONTIGUOUS canonical-order block owned by shard
    ``shard_idx`` of ``num_shards``, covering ``n`` items total via ceil-div
    chunking (the tail shard may be smaller or empty)."""
    block = -(-n // max(num_shards, 1))  # ceil div
    start = min(shard_idx * block, n)
    end = min(start + block, n)
    return start, end


def build_persistent_latent_cache(
    image_dataset: torch.utils.data.Dataset,
    vae: nn.Module,
    cache_dir: str,
    fingerprint: Dict[str, Any],
    rank: int,
    world_size: int,
    device: torch.device,
    encode_batch_size: int = 128,
    num_workers: int = 4,
) -> None:
    """(Re)build the on-disk persistent latent cache at ``cache_dir`` from
    scratch. Every rank VAE-encodes its own CONTIGUOUS block of the dataset in
    CANONICAL (dataset-index) order (see _canonical_block) and writes its own
    shard files; only once every rank's shard is on disk does rank 0 write
    meta.json + the DONE marker, atomically and LAST, so any reader that
    observes DONE is guaranteed a fully-written, consistent cache. A stale
    partial write (no DONE) is simply overwritten wholesale next time this is
    called -- no incremental/partial resume, the simplest correct behavior.

    Caches ``mu``/``logvar`` -- the VAE posterior's deterministic sufficient
    statistics -- NOT a sampled latent, so PersistentLatentCache can draw a
    fresh stochastic sample every time it loads (see its docstring)."""
    if is_main_process():
        os.makedirs(cache_dir, exist_ok=True)
    if is_distributed():
        dist.barrier()

    n = len(image_dataset)
    start, end = _canonical_block(n, rank, world_size)
    indices = list(range(start, end))
    if is_main_process():
        print(f"  [latent_cache] building persistent cache at {cache_dir} "
              f"({n:,} images total, {world_size} shard(s))")

    mu_path, logvar_path, labels_path = _latent_cache_shard_paths(cache_dir, rank)
    if indices:
        subset = torch.utils.data.Subset(image_dataset, indices)
        loader = torch.utils.data.DataLoader(
            subset, batch_size=encode_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, drop_last=False,
        )
        mu_buf: List[torch.Tensor] = []
        logvar_buf: List[torch.Tensor] = []
        labels_buf: List[int] = []
        vae = vae.to(device).eval()
        progress = tqdm(
            loader, desc=f"VAE-caching shard {rank} [{start},{end})",
            disable=not is_main_process(), dynamic_ncols=True, leave=False,
        )
        with torch.no_grad():
            for imgs, labs in progress:
                imgs = imgs.to(device=device, dtype=torch.float32, non_blocking=True)
                posterior = vae.encode(imgs).latent_dist
                mu_buf.append(posterior.mean.detach().to(dtype=torch.float16).cpu())
                logvar_buf.append(posterior.logvar.detach().to(dtype=torch.float16).cpu())
                labels_buf.extend(int(c) for c in labs)
                progress.set_postfix(cached=len(labels_buf))
        mu = torch.cat(mu_buf, dim=0).numpy()
        logvar = torch.cat(logvar_buf, dim=0).numpy()
        labels = np.array(labels_buf, dtype=np.int64)
    else:
        # More shards than images: a tail shard's block can be empty. Still
        # write a (correctly-shaped-for-concat) empty shard so shard indices
        # stay contiguous 0..world_size-1 for the loader.
        mu = np.zeros((0,), dtype=np.float16)
        logvar = np.zeros((0,), dtype=np.float16)
        labels = np.zeros((0,), dtype=np.int64)

    for path, arr in ((mu_path, mu), (logvar_path, logvar), (labels_path, labels)):
        tmp = path + ".tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, path)

    if is_distributed():
        dist.barrier()

    if is_main_process():
        shard_sizes = [
            _canonical_block(n, i, world_size)[1] - _canonical_block(n, i, world_size)[0]
            for i in range(world_size)
        ]
        assert sum(shard_sizes) == n
        meta = {"fingerprint": fingerprint, "num_shards": world_size, "shard_sizes": shard_sizes}
        _atomic_write_json(_latent_cache_meta_path(cache_dir), meta)
        # DONE marker LAST -- only once meta.json + every shard file are on disk.
        _atomic_write_text(_latent_cache_done_path(cache_dir), "ok\n")
        print(f"  [latent_cache] persistent cache complete: {cache_dir} "
              f"({n:,} images, {world_size} shard(s))")
    if is_distributed():
        dist.barrier()


class PersistentLatentCache(torch.utils.data.Dataset):
    """Load a COMPLETE persistent latent cache from ``cache_dir`` (see
    build_persistent_latent_cache) and serve this rank's round-robin slice as
    (latent, label) pairs -- a drop-in replacement for PerRankLatentCache that
    skips the VAE forward pass entirely.

    Stochasticity is preserved exactly as PerRankLatentCache's own cadence:
    the cache holds mu/logvar (deterministic), and __init__ draws ONE fresh
    ``z = mu + std*eps`` sample (torch's global RNG, no explicit generator --
    matching every other undocumented-generator draw in this file) and reuses
    it for the whole run, exactly like PerRankLatentCache's self.latents.
    Every process start (i.e. every preemption/requeue) still gets an
    independent fresh stochastic sample; only the expensive, deterministic
    VAE network forward is skipped.

    ``hflip``: REFUSED. This cache stores post-encode mu/logvar only, and the
    old latent-space flip fallback (flip the sampled latent along width) was
    measured to be a bad approximation: the sd-vae is not reflection-
    equivariant (77% mean relative latent error on FFHQ; decodes 21.8 dB vs
    29.6 dB; training on the 50% latent-flipped samples degraded the FFHQ
    teacher from FID 10 to 27). Runs needing
    hflip must use on-the-fly VAE encoding (unset --latent_cache_dir).
    """

    def __init__(self, cache_dir: str, rank: int, world_size: int, hflip: bool = False):
        with open(_latent_cache_meta_path(cache_dir)) as f:
            meta = json.load(f)
        shard_sizes: List[int] = meta["shard_sizes"]
        num_shards: int = meta["num_shards"]
        assert len(shard_sizes) == num_shards

        mu_pieces: List[np.ndarray] = []
        logvar_pieces: List[np.ndarray] = []
        label_pieces: List[np.ndarray] = []
        offset = 0
        for shard_idx, size in enumerate(shard_sizes):
            if size > 0:
                local_start = (rank - offset) % world_size
                if local_start < size:
                    mu_path, logvar_path, labels_path = _latent_cache_shard_paths(cache_dir, shard_idx)
                    mu_shard = np.load(mu_path, mmap_mode="r")
                    logvar_shard = np.load(logvar_path, mmap_mode="r")
                    labels_shard = np.load(labels_path, mmap_mode="r")
                    mu_pieces.append(np.array(mu_shard[local_start::world_size]))
                    logvar_pieces.append(np.array(logvar_shard[local_start::world_size]))
                    label_pieces.append(np.array(labels_shard[local_start::world_size]))
            offset += size

        if mu_pieces:
            mu = torch.from_numpy(np.concatenate(mu_pieces, axis=0)).float()
            logvar = torch.from_numpy(np.concatenate(logvar_pieces, axis=0)).float()
            labels = torch.from_numpy(np.concatenate(label_pieces, axis=0)).long()
        else:
            mu = torch.zeros((0, 0, 0, 0))
            logvar = torch.zeros((0, 0, 0, 0))
            labels = torch.zeros((0,), dtype=torch.long)

        std = torch.exp(0.5 * logvar)
        z = (mu + std * torch.randn_like(mu)) * _VAE_SCALE_FACTOR
        if hflip:
            # flip(encode(x)) vs encode(flip(x)) has 77% mean relative latent error on FFHQ
            # (decodes 21.8 dB vs 29.6 dB, visibly garbled) -- the sd-vae is NOT reflection-
            # equivariant, so training on latent-flipped samples degrades FID badly
            # (10 -> 27 on the FFHQ teacher).
            raise RuntimeError(
                "--hflip is incompatible with --latent_cache_dir: latent-space flipping is a "
                "measured-bad approximation of pixel-space hflip (the sd-vae is not reflection-equivariant). "
                "Unset --latent_cache_dir to use on-the-fly VAE encoding with true pixel flips.")
        self.latents = z.to(dtype=torch.float16)
        self.labels = labels
        if is_main_process():
            shape = tuple(self.latents.shape)
            mb = self.latents.element_size() * self.latents.numel() / (1024 ** 2)
            print(f"  [latent_cache] loaded from {cache_dir}: latents {shape} fp16 ({mb:.0f} MB), "
                  f"labels {self.labels.shape[0]} (persistent cache HIT -- VAE encode skipped)")

    def __len__(self) -> int:
        return self.latents.shape[0]

    def __getitem__(self, idx: int):
        return self.latents[idx], int(self.labels[idx])


# ---------------------------------------------------------------------------
# Per-model loaders + data + forward
# ---------------------------------------------------------------------------

def _build_image_dataset(args, image_size: int):
    if args.dataset == "cifar10":
        return CIFAR10Dataset(
            root=args.data_root, image_size=image_size,
            split=args.cifar_split, max_images=args.max_images,
            download=args.download, hflip=bool(getattr(args, "hflip", False)),
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
    return ImageFolderFlat(root=args.image_root, image_size=image_size, max_images=args.max_images,
                           hflip=bool(getattr(args, "hflip", False)))


def load_teacher_for_model_type(args, device, dtype):
    if args.model_type == "dit_xl":
        if getattr(args, "teacher_arch_cfg", None):
            # Opt-in NarrowDiT-format teacher under the dit_xl (DDPM/VAE-latent)
            # family -- mirrors the dit_micro branch below exactly (same flag,
            # same loader). Needed for a from-scratch --arch_plan "global"
            # teacher built via this family's own machinery (e.g. the
            # from-scratch DiT-B/2 FFHQ and LSUN Bedroom teachers) used as the
            # KD teacher for a NARROWER --arch_plan
            # student: unlike a stock DiT_models["DiT-XL/2"] checkpoint, its
            # state_dict shape depends on arch_cfg.json (hidden_size/depth/
            # num_classes), which load_dit_network's hardcoded DiT_models[...]
            # reconstruction cannot represent. Default None preserves the
            # previous behavior exactly -- a stock DiT-XL/2 --teacher_checkpoint
            # is unaffected.
            return load_narrow_dit_network(
                checkpoint_path=args.teacher_checkpoint,
                arch_cfg_path=args.teacher_arch_cfg,
                device=device,
            )
        return load_dit_network(
            checkpoint_path=args.teacher_checkpoint,
            dit_model=args.dit_model,
            input_size=args.image_size // 8,
            num_classes=1000,
            device=device,
            dtype=dtype,
        )
    if args.model_type == "dit_micro":
        if getattr(args, "teacher_arch_cfg", None):
            # Opt-in NarrowDiT-format teacher (checkpoints trained by this
            # script, or any width-allocation student used as a teacher), e.g.
            # the CIFAR-10 DiT-S/2-style teacher (D=384/depth12,
            # augment_dim=9). num_heads is ignored here -- NarrowDiT's
            # per-block head counts come from arch_cfg_path itself.
            return load_narrow_dit_network(
                checkpoint_path=args.teacher_checkpoint,
                arch_cfg_path=args.teacher_arch_cfg,
                device=device,
            )
        return load_dit_micro_network(
            checkpoint_path=args.teacher_checkpoint,
            device=device,
            num_heads=args.num_heads,
        )
    raise ValueError(f"Unknown --model_type {args.model_type!r}")


def should_skip_teacher(args) -> bool:
    """True iff teacher loading/forward should be skipped entirely: --arch_plan
    (the student is a FRESH from-scratch NarrowDiT, never teacher-initialized)
    with --kd_weight <= 0 (the KD forward is already gated on kd_weight>0
    everywhere, so the teacher would otherwise be loaded and never used --
    data-loss-only, --gt_weight-only training). The teacher-initialized path
    always needs the teacher as the student's weight-init source regardless of
    kd_weight, so this is False whenever --arch_plan is unset."""
    return bool(getattr(args, "arch_plan", None)) and float(args.kd_weight) <= 0.0


def get_in_channels(args, teacher: Optional[nn.Module]) -> int:
    """``teacher`` may be ``None`` when ``--kd_weight 0`` skips teacher loading
    (--arch_plan mode only, see main()). in_channels is a fixed constant per
    model_type regardless of the specific teacher checkpoint, so this returns
    the identical value either way."""
    if teacher is not None:
        if args.model_type == "dit_xl":
            return int(teacher.in_channels)  # 4 (latent channels)
        if args.model_type == "dit_micro":
            return int(teacher.in_channels)  # 3 (raw RGB)
        raise ValueError(args.model_type)
    if args.model_type == "dit_xl":
        return 4
    if args.model_type == "dit_micro":
        return 3
    raise ValueError(args.model_type)


@torch.no_grad()
def vae_encode_batch(vae: Optional[nn.Module], images: torch.Tensor) -> torch.Tensor:
    """For DiT-XL/2: VAE-encode a batch of raw images to latents.
    For DiT-Micro: pass-through (no VAE)."""
    if vae is None:
        return images
    posterior = vae.encode(images).latent_dist
    z = posterior.sample() * 0.18215
    return z


def model_forward_for_loss(
    model: nn.Module,
    args,
    clean: torch.Tensor,           # (B, C, H, W) -- latents for dit_xl, raw RGB for dit_micro
    timesteps: torch.Tensor,       # (B,) int64 in [0, num_timesteps)
    labels: torch.Tensor,          # (B,) int64
    alpha_bar: torch.Tensor,       # (num_timesteps,) fp32
    in_channels: int,
    compute_dtype: torch.dtype,
    noise: Optional[torch.Tensor] = None,
    augment_labels: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run one forward pass returning ``(prediction, target)`` where both are fp32.

    DiT-XL/2 predicts epsilon -> target is the sampled noise.
    DiT-Micro predicts x_0    -> target is the clean input.

    ``noise`` may be supplied to reuse the exact same corruption across two
    models (teacher and student) so a distillation MSE between their predictions
    is computed on an identical ``x_t``. If ``None``, fresh noise is drawn.

    ``augment_labels`` is the non-leaky augmentation-parameter vector produced by
    ``pace.edm_augment.AugmentPipe`` (see ``--augment_prob``). It is forwarded
    to the network as a keyword argument ONLY when not None, so nets without an
    augment-conditioning input (the stock DiT teachers) are called exactly as before.

    ``--unconditional`` (opt-in, default OFF) forces every label to the single
    learned class row (index 0) before ANY branch below runs -- a constant y=0
    convention for unconditional
    ``--model_type dit_xl --diffusion ddpm`` training on an ``image_folder``
    dataset (whose labels are the sentinel -1, see ``ImageFolderFlat``, which
    must never reach an embedding table). Default False is a no-op: labels pass
    through untouched, byte-identical to before this flag existed.

    ``--force_label`` (opt-in, default None) generalizes ``--unconditional`` to
    an ARBITRARY fixed row index instead of the hardcoded 0 -- e.g. index
    ``num_classes`` (the pretrained null/CFG row of a real class-conditional
    checkpoint such as DiT-XL/2, which already has a trained "unconditional"
    meaning from the original label-dropout schedule, unlike an arbitrary real
    class index). Takes precedence over ``--unconditional`` when both/either is
    set (checked first below); ``--unconditional`` remains untouched (still
    exactly index 0) for existing from-scratch NarrowDiT callers. Default None
    is a no-op, byte-identical to before this flag existed. Mirrors
    ``evaluate_parameters_dit.py``'s own ``--force_label`` convention.
    """
    device = clean.device
    force_label = getattr(args, "force_label", None)
    if force_label is not None:
        labels = torch.full_like(labels, int(force_label))
    elif getattr(args, "unconditional", False):
        labels = torch.zeros_like(labels)
    if noise is None:
        noise = torch.randn_like(clean)
    autocast_enabled = compute_dtype != torch.float32 and device.type == "cuda"
    # Absent (the default) -> the model is called with the identical positional
    # signature it has always been called with.
    aug_kw = {} if augment_labels is None else {"augment_labels": augment_labels}

    if getattr(args, "diffusion", "ddpm") == "edm":
        # EDM preconditioned-denoiser fine-tune (DiT-Micro). `timesteps` carries float sigmas.
        sigma = timesteps.float().view(-1, 1, 1, 1)
        sd = args.sigma_data
        x_t = clean + sigma * noise
        c_in = 1.0 / torch.sqrt(sigma ** 2 + sd ** 2)
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out = sigma * sd / torch.sqrt(sigma ** 2 + sd ** 2)
        c_noise = 0.25 * torch.log(sigma).view(-1)
        with torch.autocast(device_type=device.type, dtype=compute_dtype, enabled=autocast_enabled):
            F_out = model(c_in * x_t, c_noise, labels, **aug_kw)
        if getattr(args, "edm_loss_space", "f") == "f":
            # EDM's own objective: uniformly-weighted MSE in the network's F-space.
            # F_target = (x0 - c_skip*x_t)/c_out has unit variance across sigma, so the
            # loss weights all noise levels equally -- unlike unweighted denoised-space MSE,
            # which down-weights low sigma by c_out^2 (~100x) and barely trains those phases.
            f_target = (clean - c_skip * x_t) / c_out
            return F_out.float(), f_target.float()
        denoised = c_skip * x_t + c_out * F_out.float()
        return denoised.float(), clean.float()

    ab = alpha_bar[timesteps].float().view(-1, 1, 1, 1)
    sqrt_ab = torch.sqrt(ab)
    sqrt_one_minus_ab = torch.sqrt(1.0 - ab)
    x_t = sqrt_ab * clean + sqrt_one_minus_ab * noise

    with torch.autocast(device_type=device.type, dtype=compute_dtype, enabled=autocast_enabled):
        raw = model(x_t, timesteps, labels)

    if args.model_type == "dit_xl":
        pred = raw[:, :in_channels]
        target = noise
    else:  # dit_micro
        pred = raw
        target = clean
    return pred.float(), target.float()


def drop_labels_for_cfg(labels: torch.Tensor, p: float, null_class: int) -> torch.Tensor:
    """CFG-consistent KD label dropout: remap an i.i.d. Bernoulli(p) subset of the
    batch's labels to ``null_class`` (= num_classes, the LabelEmbedder null row --
    both teacher and student tables have num_classes+1 rows, so direct indexing
    hits the null row deterministically while both embedders keep dropout_prob=0).

    The ONE returned tensor feeds BOTH the teacher (KD target) and student
    forwards, so their conditioning stays identical: on dropped samples the
    student distills the teacher's UNCONDITIONAL function, training the null row
    that classifier-free guidance needs at eval. The mask is drawn from torch's
    global generator (like the noise draws), so full-ckpt RNG resume stays exact.
    ``p <= 0`` returns ``labels`` unchanged WITHOUT consuming RNG, keeping
    --cfg_label_drop 0 (the default) byte-identical to before."""
    if p <= 0.0:
        return labels
    drop = torch.rand(labels.shape[0], device=labels.device) < p
    return torch.where(drop, labels.new_full((), null_class), labels)


# ---------------------------------------------------------------------------
# EMA of the student (mirrors pace.edm_distillation create_ema/update_ema)
# ---------------------------------------------------------------------------

def create_ema(model: nn.Module) -> nn.Module:
    """Deep-copy ``model`` into a frozen, eval-mode EMA shadow."""
    ema = copy.deepcopy(model)
    ema.requires_grad_(False)
    ema.eval()
    return ema


def ema_update(ema: nn.Module, model: nn.Module, beta: float) -> None:
    """EMA step: ``ema <- beta*ema + (1-beta)*model``; buffers copied verbatim.

    ``tensor.lerp(other, w) = tensor + w*(other-tensor)``, so
    ``model_param.lerp(ema_param, beta) = beta*ema + (1-beta)*model``.
    """
    with torch.no_grad():
        for ep, mp in zip(ema.parameters(), model.parameters()):
            ep.copy_(mp.detach().lerp(ep, beta))
        for eb, mb in zip(ema.buffers(), model.buffers()):
            eb.copy_(mb)


# ---------------------------------------------------------------------------
# Full checkpointing (opt-in via --full_ckpt): optimizer + scheduler + step +
# epoch + RNG + ema, enabling EXACT resume. Also a retention/pruning policy for
# the coarse periodic curve checkpoints. All new; the weights-only paths remain
# the default so existing studies are byte-unchanged.
# ---------------------------------------------------------------------------

def save_full_ckpt(path, *, model, ema, optimizer, scheduler, step, epoch, cfg, kind,
                    keep_prev=False):
    """Atomically write a full checkpoint enabling exact resume.

    ``keep_prev`` (default ``False``, byte-identical to every call before this
    parameter existed -- hardening item 5): if ``True`` and ``path`` already
    holds a previous checkpoint, that file is renamed to ``path + ".prev"``
    (an atomic rename) immediately BEFORE the new payload's own atomic
    tmp-file replace. ``os.replace`` already makes a single write crash-safe
    (a preemption mid-write can never truncate the file AT ``path``), but it
    cannot protect against a FILESYSTEM-level corruption of the bytes already
    on disk (bit rot, a bad block) discovered later -- ``path + ".prev"``
    gives one generation of fallback for exactly that case. Opt-in: doubles
    the steady-state disk footprint of whichever checkpoint(s) pass
    ``keep_prev=True``, so callers choose it per checkpoint kind (typically
    just the resume-critical rolling ``last.pt``) via ``--keep_prev_ckpt``."""
    payload = {
        "model": model.state_dict(),
        "ema": (ema.state_dict() if ema is not None else None),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step, "epoch": epoch, "cfg": cfg, "kind": kind,
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
            "numpy": __import__("numpy").random.get_state(),
            "python": __import__("random").getstate(),
        },
    }
    atomic_torch_save(payload, path, keep_prev=keep_prev)


def load_full_ckpt(path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


# Every key save_full_ckpt() writes. Used by validate_full_ckpt_payload to
# recognize a genuine full checkpoint vs. some other kind of .pt file placed
# in the resume slot by mistake.
_FULL_CKPT_REQUIRED_KEYS = ("model", "optimizer", "scheduler", "step", "epoch", "rng")


def validate_full_ckpt_payload(payload, path):
    """Preemption hardening item 4 (bugs #15a/b): a
    ``last.pt`` that is NOT actually a full checkpoint -- e.g. a curve
    checkpoint (``{"model": ..., "ema": ...}`` only, no optimizer/scheduler/
    step/epoch/rng) copied into the resume slot as a seed, or a bare
    ``model.state_dict()`` -- makes every downstream ``resume_ckpt["optimizer"]``
    / ``["epoch"]`` / ``["rng"]`` access raise a bare, unhelpful ``KeyError``
    deep inside training (or, worse for a bare state_dict, a confusing
    ``TypeError`` since indexing a state_dict with a string key differs from
    indexing a ``dict`` of tensors only by luck of the key names colliding).

    Validate the payload's SHAPE right after ``load_full_ckpt``, before any
    of its fields are used, and fail with a one-line, actionable error naming
    the offending file and its actual top-level keys -- this converts a
    confusing crash into a clear one (opt-in-safe: it can only ever turn an
    already-fatal situation into a better-diagnosed fatal situation; a valid
    full checkpoint is completely unaffected).
    """
    if not isinstance(payload, dict):
        raise ValueError(
            f"--full_ckpt resume payload at {path!r} is not a dict "
            f"(got {type(payload).__name__}) -- this is not a full checkpoint "
            f"written by save_full_ckpt(). Refusing to resume from it."
        )
    missing = [k for k in _FULL_CKPT_REQUIRED_KEYS if k not in payload]
    if missing:
        raise ValueError(
            f"--full_ckpt resume payload at {path!r} is missing required "
            f"key(s) {missing} -- it does not look like a full checkpoint "
            f"written by save_full_ckpt() (e.g. a curve-snapshot checkpoint "
            f"or a bare state_dict may have been placed here by mistake, "
            f"such as by a seeding step that copied the wrong source file). "
            f"Actual top-level keys found: {sorted(payload.keys())}. "
            f"Refusing to resume from it."
        )
    return payload


def _as_cpu_byte(t):
    """Coerce a saved RNG-state tensor back to a CPU uint8 ByteTensor.
    Full ckpts are loaded with map_location=device, so the RNG ByteTensors can come
    back on CUDA; torch.set_rng_state / set_rng_state_all require a CPU ByteTensor."""
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu()
        if t.dtype != torch.uint8:
            t = t.to(torch.uint8)
    return t


def _restore_cuda_rng_states(cuda_states, *, device_count, current_device,
                              set_rng_state_all, set_rng_state, warn=warnings.warn):
    """Restore saved per-visible-device CUDA RNG states (a plain list of
    tensors, as produced by ``torch.cuda.get_rng_state_all()``).

    A rank's own ``torch.randn``/``torch.rand`` etc. calls always draw from
    ``torch.cuda.current_device()`` -- never another rank's device -- so at a
    MATCHED device count, replaying the full saved list via
    ``set_rng_state_all`` (the historical behavior) is byte-identical to the
    pre-fix code and is kept exactly as-is (same call, same argument).

    At a MISMATCHED device count (e.g. a ``--full_ckpt`` written by an 8-GPU
    job resumed at a smaller/larger world size via ``--grad_accum``), naively
    replaying every saved index crashes with ``IndexError: tuple index out of
    range`` the moment the saved list is longer than the number of devices now
    visible (``torch.cuda.default_generators`` only has ``device_count``
    entries). Since only the CALLING RANK'S OWN current device is ever
    actually consumed, restore just that one entry (if the save side
    captured it) via the single-device ``set_rng_state``; if the calling
    rank's device index was never saved (e.g. a shrink-then-grow round trip
    landing on a higher device index than the save side had), warn and skip
    rather than raise -- training resumes normally, just without an exactly
    reproduced noise/timestep stream for that rank from this point on.

    Dependency-injected (``device_count``/``current_device``/``set_rng_state*``
    /``warn``) so this is exercised CPU-only in tests (no real CUDA needed),
    mirroring how ``tests/test_grad_accum.py`` monkeypatches trainer-level
    functions to stay CPU-only.
    """
    n_visible = device_count()
    if len(cuda_states) == n_visible:
        set_rng_state_all([_as_cpu_byte(c) for c in cuda_states])
        return
    my_device = current_device()
    if my_device < len(cuda_states):
        set_rng_state(_as_cpu_byte(cuda_states[my_device]), my_device)
    else:
        warn(
            f"restore_rng: checkpoint has {len(cuda_states)} saved CUDA RNG "
            f"state(s) but {n_visible} device(s) are visible now and this "
            f"rank's current device index is {my_device} -- skipping CUDA "
            f"RNG restore for this rank (world_size changed since the "
            f"checkpoint was written; training resumes normally, just "
            f"without an exactly reproduced noise/timestep stream)."
        )


def restore_rng(rng):
    torch.set_rng_state(_as_cpu_byte(rng["torch"]))
    cuda_states = rng.get("cuda")
    if cuda_states is not None and torch.cuda.is_available():
        _restore_cuda_rng_states(
            cuda_states,
            device_count=torch.cuda.device_count,
            current_device=torch.cuda.current_device,
            set_rng_state_all=torch.cuda.set_rng_state_all,
            set_rng_state=torch.cuda.set_rng_state,
        )
    __import__("numpy").random.set_state(rng["numpy"])
    __import__("random").setstate(rng["python"])


def prune_curve_ckpts(curve_dir, keep=10):
    """Keep only the ``keep`` newest ``step_<k>.pt`` files in ``curve_dir``."""
    import glob, re
    files = glob.glob(os.path.join(curve_dir, "step_*.pt"))
    def k(f):
        return int(re.search(r"step_(\d+)\.pt", f).group(1))
    for f in (sorted(files, key=k)[:-keep] if len(files) > keep else []):
        os.remove(f)


# ---------------------------------------------------------------------------
# Periodic held-out validation + best-checkpoint tracking (opt-in --val_every).
# When --val_every == 0 (default) none of this runs: no val loader is built and
# validate() is never called, so behaviour is byte-identical to before.
# ---------------------------------------------------------------------------

class BestTracker:
    """Tracks the lowest value seen; ``update`` returns True on a new best that
    beats the previous best by more than ``min_delta`` (no early stopping here,
    just best-selection for saving best.pt / plateau analysis)."""

    def __init__(self, min_delta=1e-4):
        self.min_delta = float(min_delta)
        self.best_value = float("inf")
        self.best_step = -1

    def update(self, value, step):
        if value < self.best_value - self.min_delta:
            self.best_value = float(value)
            self.best_step = int(step)
            return True
        return False


def build_val_loader(args, image_size, batch_size, max_batches):
    """Build a small held-out validation loader (only used when --val_every>0).

    Uses a DIFFERENT split from training: CIFAR-10 -> the ``test`` split;
    ImageNet -> the ``val`` split. For the remaining sources there is no obvious
    held-out split, so we reuse the configured dataset/split (best-effort). The
    dataset is capped to ``max_batches * batch_size`` images to keep it cheap."""
    cap = max_batches * batch_size
    if args.dataset == "cifar10":
        ds = CIFAR10Dataset(
            root=args.data_root, image_size=image_size,
            split="test", max_images=cap, download=args.download,
        )
    elif args.dataset == "imagenet":
        ds = ImageNetDataset(
            root=args.data_root, image_size=image_size,
            split="val", max_images=cap, download=args.download,
        )
    else:
        # No canonical held-out split; reuse the configured dataset, capped small.
        saved = args.max_images
        try:
            args.max_images = cap if saved is None else min(saved, cap)
            ds = _build_image_dataset(args, image_size)
        finally:
            args.max_images = saved
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    args,
    device: torch.device,
    phase: Dict[str, int],
    num_bins: int,
    alpha_bar: torch.Tensor,
    in_channels: int,
    compute_dtype: torch.dtype,
    vae: Optional[nn.Module] = None,
    teacher: Optional[nn.Module] = None,
    max_batches: int = 16,
) -> float:
    """Mean held-out loss for ``model`` (the EMA shadow when available, else the
    raw student). Reuses the EXACT per-phase sigma/timestep sampling helpers of
    ``train_phase`` (``sample_phase_sigmas`` for EDM, ``sample_phase_timesteps``
    for DDPM) and the SAME ``model_forward_for_loss`` path with the SAME
    ``gt_weight``/``kd_weight`` combination, so the reported value is directly
    comparable to the training loss. Restores the model to train() afterwards
    (leaving an eval-mode EMA in eval)."""
    was_training = model.training
    model.eval()
    total = 0.0
    n = 0
    for i, (images, labels) in enumerate(val_loader):
        if i >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        clean = vae_encode_batch(vae, images.to(torch.float32))
        if getattr(args, "diffusion", "ddpm") == "edm":
            timesteps = draw_phase_sigmas(
                args, batch_size=clean.shape[0],
                phase_start=phase["start"], phase_end=phase["end"], num_bins=num_bins,
                device=device,
            )
        else:
            timesteps = sample_phase_timesteps(
                batch_size=clean.shape[0],
                phase_start=phase["start"], phase_end=phase["end"],
                num_bins=num_bins, num_timesteps=args.num_timesteps,
                device=device,
            )
        noise = torch.randn_like(clean)
        pred, target = model_forward_for_loss(
            model=model, args=args, clean=clean, timesteps=timesteps, labels=labels,
            alpha_bar=alpha_bar, in_channels=in_channels,
            compute_dtype=compute_dtype, noise=noise,
        )
        loss_gt = F.mse_loss(pred, target)
        if args.kd_weight > 0.0 and teacher is not None:
            teacher_pred, _ = model_forward_for_loss(
                model=teacher, args=args, clean=clean, timesteps=timesteps, labels=labels,
                alpha_bar=alpha_bar, in_channels=in_channels,
                compute_dtype=compute_dtype, noise=noise,
            )
            loss_kd = F.mse_loss(pred, teacher_pred)
        else:
            loss_kd = torch.zeros((), device=device)
        loss = args.gt_weight * loss_gt + args.kd_weight * loss_kd
        total += float(loss.detach())
        n += 1
    if was_training:
        model.train()
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# Training one phase
# ---------------------------------------------------------------------------

def train_phase(
    args,
    teacher: nn.Module,
    student_ddp: nn.Module,
    raw_student: nn.Module,        # underlying student (unwrapped from DDP)
    vae: Optional[nn.Module],
    dataloader: DataLoader,
    sampler: Optional[DistributedSampler],
    phase: Dict[str, int],
    num_bins: int,
    alpha_bar: torch.Tensor,
    in_channels: int,
    compute_dtype: torch.dtype,
    phase_dir: str,
    use_wandb: bool,
    resume_ckpt: Optional[dict] = None,
    augment_pipe: Optional[Any] = None,
) -> Dict[str, Any]:
    device = next(raw_student.parameters()).device
    pidx = phase["index"]
    if is_main_process():
        print(f"Phase {pidx}: bins [{phase['start']}, {phase['end']}); {len(dataloader)} batches/rank/epoch")

    n_params = sum(p.numel() for p in raw_student.parameters())
    if is_main_process():
        print(f"  student params: {n_params:,}")

    opt_cls = torch.optim.Adam if getattr(args, "optimizer", "adamw") == "adam" else torch.optim.AdamW
    optimizer = opt_cls(
        student_ddp.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    # --- Gradient accumulation (--grad_accum N, default 1 = historical path) ---
    # INVARIANT: one `step` == one OPTIMIZER step, never a microbatch. With
    # --grad_accum N the inner loop runs forward/backward on N consecutive
    # microbatches (each loss scaled by 1/N, so the summed gradient equals the
    # mean-MSE gradient of the concatenated batch of size N*batch_size), then
    # performs exactly ONE optimizer.step() + ONE scheduler.step() + ONE
    # ema_update() and increments `step` by ONE. Everything keyed on `step` --
    # ckpt_every, val_every, log_every, the WSD warmup/cooldown boundaries,
    # curve checkpoint naming, and the step stored in full checkpoints --
    # therefore sees IDENTICAL step semantics to a single-shot run at
    # batch_size*N.
    # At the default grad_accum == 1 the accumulation branch is skipped
    # entirely: the loss is NEVER divided (not even by 1) and zero_grad/
    # backward/clip/step run in the historical order, so the sequence of
    # floating-point operations is byte-identical to the pre-flag code.
    grad_accum = int(getattr(args, "grad_accum", 1) or 1)
    # Optimizer steps per epoch. Microbatches beyond the last FULL accumulation
    # group of an epoch are dropped (no forward) -- mirroring the dataloaders'
    # own drop_last=True convention one level up, so an optimizer step is only
    # ever taken on a full N*batch_size effective batch. At grad_accum == 1
    # this is len(dataloader) exactly, as before.
    steps_per_epoch = len(dataloader) // grad_accum
    if grad_accum > 1 and steps_per_epoch == 0:
        raise ValueError(
            f"--grad_accum {grad_accum} > microbatches per epoch ({len(dataloader)}): "
            f"no full accumulation group fits, so no optimizer step would ever run"
        )
    total_steps = args.num_epochs * steps_per_epoch
    if getattr(args, "lr_schedule", "cosine") == "wsd":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda s: wsd_lr_lambda(
                s, total_steps=total_steps, warmup_steps=args.warmup_steps,
                cooldown_frac=args.cooldown_frac),
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=args.lr * args.lr_min_ratio,
        )

    full_ckpt = bool(getattr(args, "full_ckpt", False))
    # Sanitized args snapshot embedded in full checkpoints for provenance/exact rebuild.
    cfg_payload = {k: v for k, v in vars(args).items() if not k.startswith("_")}

    start_epoch = 0
    if resume_ckpt is not None:
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        scheduler.load_state_dict(resume_ckpt["scheduler"])
        start_epoch = int(resume_ckpt["epoch"]) + 1
        if is_main_process():
            print(f"  resuming phase {pidx} at epoch {start_epoch + 1}/{args.num_epochs}")

    history: Dict[str, List[float]] = {"loss": [], "loss_gt": [], "loss_kd": [], "lr": []}
    step = start_epoch * steps_per_epoch  # optimizer steps (== batches at grad_accum 1)
    # Full-ckpt resume: use the exact saved step (== start_epoch*len at an epoch
    # boundary, so this is a no-op for the weights-only path but authoritative here).
    if full_ckpt and resume_ckpt is not None and "step" in resume_ckpt:
        step = int(resume_ckpt["step"])
    if teacher is not None:
        teacher.eval()
    student_ddp.train()

    # EMA shadow of the UNDERLYING student (not the DDP wrapper); opt-in via
    # --ema_beta. Stays None when off so behaviour is byte-identical to before.
    # Consumed by the checkpoint/eval tasks (Tasks 4/6).
    ema_beta = float(getattr(args, "ema_beta", 0.0) or 0.0)
    ema = create_ema(raw_student) if ema_beta > 0 else None

    # CFG-consistent KD label dropout (--cfg_label_drop): p>0 remaps a shared
    # Bernoulli(p) subset of each batch's labels to the null-class index for both
    # the teacher and student forwards (see drop_labels_for_cfg). Default 0.0 =
    # OFF: no remap, no RNG draw -> existing runs byte-identical.
    cfg_label_drop = float(getattr(args, "cfg_label_drop", 0.0) or 0.0)
    if cfg_label_drop > 0.0:
        # Teacher may be None (--kd_weight 0 --arch_plan skip); the student's own
        # y_embedder has the SAME num_classes (from the plan's cfg), so the null
        # row index is identical either way.
        null_class = int((teacher if teacher is not None else raw_student).y_embedder.num_classes)
    else:
        null_class = -1

    # Full-ckpt resume also restores the EMA shadow and the RNG streams so
    # data sampling / noise draws continue exactly where they left off. (The
    # model weights themselves are loaded by the caller before train_phase.)
    if full_ckpt and resume_ckpt is not None:
        if ema is not None and resume_ckpt.get("ema") is not None:
            ema.load_state_dict(resume_ckpt["ema"])
        if resume_ckpt.get("rng") is not None:
            restore_rng(resume_ckpt["rng"])

    # Periodic held-out validation (opt-in via --val_every). Built ONCE per phase
    # and only on the main process (that is where val_loss is logged / best.pt is
    # written). When --val_every == 0 no loader is built and validate() is never
    # called below, so the phase is byte-identical to before. NO early stopping:
    # BestTracker only selects the lowest-val step for logging + best.pt.
    val_every = int(getattr(args, "val_every", 0) or 0)
    val_loader = None
    best_tracker = None
    if val_every > 0 and is_main_process():
        image_size = args.image_size if args.model_type == "dit_xl" else 32
        val_loader = build_val_loader(
            args, image_size=image_size,
            batch_size=int(getattr(args, "val_batch_size", 128)),
            max_batches=int(getattr(args, "val_max_batches", 16)),
        )
        best_tracker = BestTracker(getattr(args, "val_min_delta", 1e-4) or 1e-4)
        print(f"  [val] phase {pidx}: held-out loader built "
              f"({len(val_loader.dataset)} imgs); validating every {val_every} steps")

    if grad_accum > 1 and is_main_process():
        print(f"  [grad_accum] {grad_accum} microbatches/optimizer step: "
              f"{steps_per_epoch} optimizer steps/epoch "
              f"({len(dataloader) - steps_per_epoch * grad_accum} tail microbatches dropped), "
              f"effective batch {args.batch_size * grad_accum}/rank")

    for epoch in range(start_epoch, args.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_loss_gt = 0.0
        epoch_loss_kd = 0.0
        n_batches = 0
        micro_in_group = 0  # position within the current accumulation group
        progress = tqdm(
            dataloader,
            desc=f"phase {pidx} epoch {epoch + 1}/{args.num_epochs} (rank {get_rank()})",
            disable=not is_main_process(),
            dynamic_ncols=True,
            leave=False,
        )
        for micro_idx, batch in enumerate(progress):
            # Epoch boundary under accumulation: drop the tail microbatches that
            # cannot fill a complete group (see steps_per_epoch above). Skipped
            # BEFORE any forward/RNG so a partial group never leaves stale
            # gradients behind for the next epoch's first group. Never triggers
            # at grad_accum == 1 (steps_per_epoch * 1 == len(dataloader)).
            if grad_accum > 1 and micro_idx >= steps_per_epoch * grad_accum:
                break
            images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            augment_labels = None
            with torch.no_grad():
                # DiT-XL/2 path: cache already holds VAE latents (fp16) -> just cast.
                # DiT-Micro path: raw RGB -> still raw RGB (vae is None).
                images_f = images.to(torch.float32)
                # Karras' non-leaky augmentation pipeline (--augment_prob). Applied to
                # the [-1,1] images exactly where EDM applies it (EDMLoss.__call__:
                # `y, augment_labels = augment_pipe(images)`), i.e. BEFORE the noise
                # corruption; the drawn parameters are fed back to the net as
                # conditioning below, which is what makes it provably non-leaky.
                # augment_pipe is None unless --augment_prob > 0, so no RNG is drawn
                # and this is byte-identical to before for every existing run.
                if augment_pipe is not None:
                    images_f, augment_labels = augment_pipe(images_f)
                clean = vae_encode_batch(vae, images_f)

            if getattr(args, "diffusion", "ddpm") == "edm":
                timesteps = draw_phase_sigmas(
                    args, batch_size=clean.shape[0],
                    phase_start=phase["start"], phase_end=phase["end"], num_bins=num_bins,
                    device=device,
                )
            else:
                timesteps = sample_phase_timesteps(
                    batch_size=clean.shape[0],
                    phase_start=phase["start"], phase_end=phase["end"],
                    num_bins=num_bins, num_timesteps=args.num_timesteps,
                    device=device,
                )
            # Shared corruption so the KD target is computed on the SAME x_t.
            noise = torch.randn_like(clean)
            # One shared label-drop per batch: the SINGLE remapped labels tensor
            # feeds both forwards below (and hence both the KD and data losses).
            labels = drop_labels_for_cfg(labels, cfg_label_drop, null_class)
            pred, target = model_forward_for_loss(
                model=student_ddp, args=args,
                clean=clean, timesteps=timesteps, labels=labels,
                alpha_bar=alpha_bar, in_channels=in_channels,
                compute_dtype=compute_dtype, noise=noise,
                augment_labels=augment_labels,
            )
            loss_gt = F.mse_loss(pred, target)

            if args.kd_weight > 0.0 and teacher is not None:
                with torch.no_grad():
                    teacher_pred, _ = model_forward_for_loss(
                        model=teacher, args=args,
                        clean=clean, timesteps=timesteps, labels=labels,
                        alpha_bar=alpha_bar, in_channels=in_channels,
                        compute_dtype=compute_dtype, noise=noise,
                    )
                loss_kd = F.mse_loss(pred, teacher_pred)
            else:
                loss_kd = torch.zeros((), device=device)

            loss = args.gt_weight * loss_gt + args.kd_weight * loss_kd

            if grad_accum == 1:
                # Historical single-batch path, bit-for-bit: no loss scaling is
                # applied (not even a divide by 1), preserving the exact FP-op
                # sequence of every existing run.
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
            else:
                if micro_in_group == 0:
                    optimizer.zero_grad(set_to_none=True)
                is_final_micro = micro_in_group == grad_accum - 1
                # 1/N scaling: F.mse_loss reduces by mean, so the sum of the N
                # scaled microbatch gradients equals the gradient of the
                # mean-MSE over the concatenated N*batch_size batch. The loss
                # is fp32 here for BOTH families (model_forward_for_loss casts
                # pred/target to .float() before the MSE; the bf16
                # autocast only wraps the model forward), and this repo uses no
                # GradScaler (bf16 needs none), so plain scaled .backward()
                # accumulation is exact -- no scaler bookkeeping to interact with.
                scaled_loss = loss / grad_accum
                if isinstance(student_ddp, DDP) and not is_final_micro:
                    # DDP + accumulation: skip the gradient all-reduce on
                    # non-final microbatches; the final microbatch's backward
                    # all-reduces the full accumulated gradient once.
                    with student_ddp.no_sync():
                        scaled_loss.backward()
                else:
                    scaled_loss.backward()
            micro_in_group += 1

            # Per-MICROBATCH bookkeeping: epoch-mean losses and the progress bar
            # count every microbatch (identical denominators at grad_accum == 1;
            # loss_val is the UNSCALED microbatch loss so logs stay comparable).
            loss_val = float(loss.detach())
            loss_gt_val = float(loss_gt.detach())
            loss_kd_val = float(loss_kd.detach())
            epoch_loss += loss_val
            epoch_loss_gt += loss_gt_val
            epoch_loss_kd += loss_kd_val
            n_batches += 1

            if micro_in_group < grad_accum:
                # Mid-group microbatch: gradients accumulated; NO optimizer /
                # scheduler / EMA / checkpoint / `step` activity (the INVARIANT
                # above). Unreachable at grad_accum == 1.
                if is_main_process():
                    progress.set_postfix(
                        loss=f"{loss_val:.4f}", gt=f"{loss_gt_val:.4f}",
                        kd=f"{loss_kd_val:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    )
                continue
            micro_in_group = 0

            torch.nn.utils.clip_grad_norm_(student_ddp.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            scheduler.step()

            if ema is not None:  # no-op when --ema_beta == 0 (ema is None)
                ema_update(ema, raw_student, ema_beta)

            if args.ckpt_every and is_main_process() and (step % args.ckpt_every == 0):
                if full_ckpt:
                    # Coarse FID-curve checkpoint: a small {model, ema} dict under
                    # curve/ so the evaluation loader can prefer the ema. Retain only the
                    # newest --curve_keep of them.
                    curve_dir = os.path.join(phase_dir, "curve")
                    os.makedirs(curve_dir, exist_ok=True)
                    atomic_torch_save(
                        {"model": raw_student.state_dict(),
                         "ema": (ema.state_dict() if ema is not None else None)},
                        os.path.join(curve_dir, f"step_{step}.pt"),
                    )
                    prune_curve_ckpts(curve_dir, getattr(args, "curve_keep", 10))
                else:
                    atomic_torch_save(raw_student.state_dict(), os.path.join(phase_dir, f"step_{step}.pt"))

            step += 1

            # WSD only: snapshot the model state once, at the boundary where the
            # LR decay (cooldown) begins. ``step`` here is the count of completed
            # optimizer steps, matching the scheduler's internal counter, so the
            # first trigger lands exactly at decay_start. Guarded by os.path.exists
            # so it fires once even across a preemption/resume.
            if (full_ckpt and is_main_process()
                    and getattr(args, "lr_schedule", "cosine") == "wsd"):
                decay_start = int(round(total_steps * (1.0 - getattr(args, "cooldown_frac", 0.2))))
                pc_path = os.path.join(phase_dir, "pre_cooldown.pt")
                if step >= decay_start and not os.path.exists(pc_path):
                    save_full_ckpt(
                        pc_path, model=raw_student, ema=ema, optimizer=optimizer,
                        scheduler=scheduler, step=step, epoch=epoch,
                        cfg=cfg_payload, kind="pre_cooldown",
                        keep_prev=getattr(args, "keep_prev_ckpt", False),
                    )

            if is_main_process():
                progress.set_postfix(
                    loss=f"{loss_val:.4f}", gt=f"{loss_gt_val:.4f}",
                    kd=f"{loss_kd_val:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )
                if use_wandb and step % args.log_every == 0:
                    wandb.log({
                        f"phase_{pidx}/loss": loss_val,
                        f"phase_{pidx}/loss_gt": loss_gt_val,
                        f"phase_{pidx}/loss_kd": loss_kd_val,
                        f"phase_{pidx}/lr": scheduler.get_last_lr()[0],
                        f"phase_{pidx}/step": step,
                    })

            # Periodic held-out validation on the EMA weights (raw student when no
            # EMA). val_loader is only built on main when --val_every>0, so this is
            # a no-op on other ranks / when validation is off (short-circuits before
            # the modulo, avoiding any div-by-zero). Track+log+save best; no early stop.
            if val_loader is not None and val_every > 0 and step % val_every == 0:
                val = validate(
                    ema if ema is not None else raw_student, val_loader, args, device,
                    phase, num_bins, alpha_bar, in_channels, compute_dtype,
                    vae=vae, teacher=teacher,
                    max_batches=int(getattr(args, "val_max_batches", 16)),
                )
                if use_wandb:
                    wandb.log({f"phase_{pidx}/val_loss": val}, step=step)
                if best_tracker.update(val, step) and full_ckpt:
                    save_full_ckpt(
                        os.path.join(phase_dir, "best.pt"),
                        model=raw_student, ema=ema, optimizer=optimizer,
                        scheduler=scheduler, step=step, epoch=epoch,
                        cfg=cfg_payload, kind="best",
                        keep_prev=getattr(args, "keep_prev_ckpt", False),
                    )
                if is_main_process():
                    print(f"  [val] phase {pidx} step {step}: val_loss={val:.6f}"
                          f" (best {best_tracker.best_value:.6f} @ {best_tracker.best_step})")

        denom = max(n_batches, 1)
        avg = epoch_loss / denom
        avg_gt = epoch_loss_gt / denom
        avg_kd = epoch_loss_kd / denom
        if is_distributed():
            t = torch.tensor([avg, avg_gt, avg_kd], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.AVG)
            avg, avg_gt, avg_kd = float(t[0]), float(t[1]), float(t[2])
        history["loss"].append(avg)
        history["loss_gt"].append(avg_gt)
        history["loss_kd"].append(avg_kd)
        history["lr"].append(scheduler.get_last_lr()[0])
        if is_main_process():
            print(f"  phase {pidx} epoch {epoch + 1} mean loss: {avg:.6f} (gt {avg_gt:.6f} | kd {avg_kd:.6f})")
            if use_wandb:
                wandb.log({f"phase_{pidx}/epoch_loss": avg, f"phase_{pidx}/epoch": epoch + 1})
        # Preemption-safe resume point: model (+ema+opt+sched+step+rng under
        # --full_ckpt) for the just-completed epoch. Full mode writes a rolling
        # last.pt enabling EXACT resume; legacy mode keeps the weights-only resume.pt.
        if is_main_process():
            if full_ckpt:
                save_full_ckpt(
                    os.path.join(phase_dir, "last.pt"),
                    model=raw_student, ema=ema, optimizer=optimizer,
                    scheduler=scheduler, step=step, epoch=epoch,
                    cfg=cfg_payload, kind="last",
                    keep_prev=getattr(args, "keep_prev_ckpt", False),
                )
            else:
                atomic_torch_save({"model": raw_student.state_dict(),
                                    "optimizer": optimizer.state_dict(),
                                    "scheduler": scheduler.state_dict(),
                                    "epoch": epoch},
                                   os.path.join(phase_dir, "resume.pt"))
        if is_distributed():
            dist.barrier()

    if is_main_process():
        ckpt_path = os.path.join(phase_dir, "student.pt")
        if full_ckpt:
            # Final = full checkpoint (kind="final"); downstream "student.pt exists
            # -> skip" and eval keep working. The rolling last.pt / pre_cooldown.pt /
            # best.pt / curve/ are retained by policy; nothing else is written.
            save_full_ckpt(
                ckpt_path, model=raw_student, ema=ema, optimizer=optimizer,
                scheduler=scheduler, step=step, epoch=args.num_epochs - 1,
                cfg=cfg_payload, kind="final",
                keep_prev=getattr(args, "keep_prev_ckpt", False),
            )
        else:
            atomic_torch_save(raw_student.state_dict(), ckpt_path)
            # phase finished -> drop the resume checkpoint
            rp = os.path.join(phase_dir, "resume.pt")
            if os.path.exists(rp):
                os.remove(rp)
        print(f"  saved {ckpt_path}")

    meta = {
        "phase_index": pidx,
        "phase_start": phase["start"],
        "phase_end": phase["end"],
        "num_bins": num_bins,
        "num_params": n_params,
        "loss_history": history["loss"],
        "loss_gt_history": history["loss_gt"],
        "loss_kd_history": history["loss_kd"],
        "lr_history": history["lr"],
        "gt_weight": args.gt_weight,
        "kd_weight": args.kd_weight,
        "edm_loss_space": getattr(args, "edm_loss_space", "f"),
        "args": {k: v for k, v in vars(args).items() if not k.startswith("_")},
    }
    return meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", required=True, choices=["dit_xl", "dit_micro"])
    p.add_argument("--dit_repo", default=None,
                   help="facebookresearch/DiT checkout (default: $DIT_REPO).")
    p.add_argument("--teacher_checkpoint", required=True)
    p.add_argument("--grouping_json", default=None,
                   help="timestep_grouping.json defining the phase boundaries. Required for "
                        "the teacher-initialized path; ignored when --arch_plan is set (the plan "
                        "supplies the phase bins).")
    p.add_argument("--output_dir", required=True)

    # dataset
    p.add_argument("--dataset", required=True,
                   choices=["cifar10", "imagenet1k_parquet", "imagenet", "image_folder"])
    p.add_argument("--data_root", default="./data")
    p.add_argument("--image_root", default=None)
    p.add_argument("--cifar_split", default="train", choices=["train", "validation", "test"])
    p.add_argument("--imagenet_split", default="train", choices=["train", "val"])
    p.add_argument("--parquet_split_prefix", default=None)
    p.add_argument("--parquet_image_column", default=None)
    p.add_argument("--parquet_label_column", default=None)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--download", action="store_true")
    p.add_argument("--augment_prob", type=float, default=0.0,
                   help="Karras/EDM NON-LEAKY augmentation probability (EDM's --augment; "
                        "paper value for CIFAR-10 = 0.12). 0.0 (default) = OFF: no pipe is "
                        "built, no RNG is drawn, and the net gets no augment-conditioning "
                        "input, so existing runs stay byte-identical. When > 0 the EDM "
                        "CIFAR-10 pipe (x-flip @p=1, plus y-flip / isotropic scale / "
                        "fractional rotation / anisotropic scale / fractional translation, "
                        "each @p) augments each batch and its 9-dim per-sample parameter "
                        "vector is fed to the network as conditioning (that is what makes "
                        "it non-leaky). Requires --model_type dit_micro --diffusion edm "
                        "--arch_plan, --kd_weight 0, and is MUTUALLY EXCLUSIVE with --hflip "
                        "(the pipe's x-flip already fires with probability 1 and subsumes it).")
    p.add_argument("--net_dropout", type=float, default=0.0,
                   help="Dropout probability inside the NarrowDiT blocks (EDM's --dropout; "
                        "paper value for CIFAR-10 = 0.13). Applied on both residual "
                        "branches' outputs: the attention output projection and inside the "
                        "MLP. 0.0 (default) = OFF and byte-identical to before. Requires "
                        "--arch_plan (NarrowDiT students).")
    p.add_argument("--hflip", action="store_true", default=False,
                   help="Random horizontal flip (p=0.5) in the CIFAR-10 or image_folder "
                        "TRAIN pipeline only (never the held-out --val_every split). "
                        "Default OFF = byte-identical transform pipeline to before this "
                        "flag existed.")
    p.add_argument("--unconditional", action="store_true", default=False,
                   help="Force every label to the single learned class row (index 0) "
                        "before every --diffusion branch (a constant y=0 convention for "
                        "--model_type dit_xl --diffusion ddpm "
                        "unconditional pretraining on an image_folder dataset -- see "
                        "model_forward_for_loss). Default OFF = byte-identical (labels "
                        "pass through untouched). Incompatible with --cfg_label_drop > 0 "
                        "(no null row / nothing meaningful to drop to).")
    p.add_argument("--force_label", type=int, default=None,
                   help="Force every label to this fixed integer row index before every "
                        "--diffusion branch (generalizes --unconditional's hardcoded 0; "
                        "takes precedence when set -- see model_forward_for_loss). Pass "
                        "the checkpoint's null/CFG row (= its num_classes, e.g. 1000 for "
                        "the stock DiT-XL/2 1001-row table) to fine-tune/probe through the "
                        "ALREADY-TRAINED unconditional row instead of an arbitrary real "
                        "class index. Default None = byte-identical (no-op). Incompatible "
                        "with --cfg_label_drop > 0, same rationale as --unconditional.")

    # DiT-XL/2 specifics
    p.add_argument("--dit_model", default="DiT-XL/2")
    p.add_argument("--image_size", type=int, default=256,
                   help="Source image resolution; latent side = image_size // 8 for dit_xl.")

    # DiT-Micro specifics
    p.add_argument("--num_heads", type=int, default=3,
                   help="DiT-Micro head count (state-dict doesn't store it). Should match the analysis run.")
    p.add_argument("--teacher_arch_cfg", default=None,
                   help="--model_type dit_micro or dit_xl. Path to a NarrowDiT arch_cfg.json "
                        "(sibling of --teacher_checkpoint, as written by train_phase_students.py's "
                        "own phase save / dit_arch_to_plans.py). OPT-IN: when set, --teacher_checkpoint "
                        "is loaded as a NarrowDiT built from this config (arbitrary hidden_size/"
                        "depth, optional augment_dim/dropout -- e.g. the CIFAR-10 DiT-S/2-style teacher "
                        "for dit_micro, or a from-scratch --arch_plan 'global' teacher such as the "
                        "DiT-B/2 FFHQ and LSUN Bedroom teachers for dit_xl) instead of the model_type's legacy stock-"
                        "architecture reconstruction (load_dit_micro_network for dit_micro; "
                        "load_dit_network's hardcoded DiT_models[--dit_model] for dit_xl). Default "
                        "None preserves the previous behavior exactly -- a --teacher_checkpoint "
                        "for the third-party normalcomputing/dit-cifar10-32x32-class "
                        "model or a stock DiT-XL/2 is "
                        "unaffected. The KD forward (model_forward_for_loss) already calls the teacher "
                        "with no augment_labels (every call site does), which a NarrowDiT built with "
                        "augment_dim>0 handles correctly -- the conditioning term is simply omitted, "
                        "matching how the model is used at inference/sampling time.")

    # diffusion schedule
    p.add_argument("--num_timesteps", type=int, default=1000)
    p.add_argument("--diffusion", type=str, default="ddpm", choices=["ddpm", "edm"],
                   help="Forward process. 'ddpm' (default) for the latent DiTs (1000-step "
                        "linear-beta eps-prediction). Use 'edm' for the pixel-space CIFAR-10 "
                        "DiTs (preconditioned denoiser, x_t = x0 + sigma*n).")
    p.add_argument("--edm_loss_space", type=str, default="f", choices=["f", "denoised"],
                   help="EDM loss parameterization. 'f' = EDM's own uniformly-weighted F-space MSE "
                        "(correct; trains all noise levels equally). 'denoised' = plain MSE on the "
                        "denoised output (legacy; down-weights low sigma by c_out^2).")
    p.add_argument("--sigma_min", type=float, default=0.002)
    p.add_argument("--sigma_max", type=float, default=80.0)
    p.add_argument("--sigma_data", type=float, default=0.5)
    p.add_argument("--rho", type=float, default=7.0)
    p.add_argument("--sigma_sampling", type=str, default="loguniform",
                   choices=["loguniform", "lognormal"],
                   help="EDM training-sigma distribution (per-phase bin restriction applies "
                        "either way). 'loguniform' (default) = current sample_phase_sigmas "
                        "behavior (byte-identical). 'lognormal' = EDM's own training "
                        "distribution, sigma = exp(p_mean + p_std*randn) clipped to "
                        "[sigma_min, sigma_max], truncated/rejected to the phase's sigma "
                        "sub-range (see sample_phase_sigmas_lognormal).")
    p.add_argument("--p_mean", type=float, default=-1.2,
                   help="lognormal sigma sampling: mean of log(sigma) (EDM default -1.2).")
    p.add_argument("--p_std", type=float, default=1.2,
                   help="lognormal sigma sampling: std of log(sigma) (EDM default 1.2).")

    # training
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--grad_accum", type=int, default=1,
                   help="Gradient accumulation: N microbatches of --batch_size per "
                        "optimizer step (effective batch = batch_size*N per rank), for "
                        "preserving a recipe's global batch on fewer/smaller GPUs (e.g. "
                        "a global 256 = 8x32 becomes --batch_size 64 --grad_accum 4 "
                        "on ONE GPU). Each microbatch loss is scaled by 1/N so the "
                        "accumulated gradient equals the full-batch mean-MSE gradient; "
                        "ONE optimizer+scheduler+EMA step and ONE `step` increment per "
                        "group, so ckpt_every/val_every/log_every/WSD boundaries keep "
                        "optimizer-step semantics. Tail microbatches that cannot fill a "
                        "group are dropped per epoch (drop_last convention). Default 1 = "
                        "OFF: the loss is never divided and the historical FP-op "
                        "sequence is byte-identical.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--vae_batch_size", type=int, default=64,
                   help="VAE encode batch size (dit_xl only); should match training batch_size.")
    p.add_argument("--latent_cache_trust", action="store_true",
                   help="Opt-in: with --latent_cache_dir pointing at a COMPLETE persistent cache, take "
                        "the fingerprint from its meta.json and do NOT build/open the image dataset at "
                        "all (a machine holding only the cache can train). Default off = byte-identical "
                        "to before. Requires --val_every 0.")
    p.add_argument("--latent_cache_dir", default=None,
                   help="Opt-in PERSISTENT (cross-restart) VAE latent cache directory, latent-space "
                        "model types only (dit_xl). Default None = OFF: byte-identical to "
                        "before this flag existed -- PerRankLatentCache re-encodes the whole dataset "
                        "through the VAE at every process start, exactly as always. When set: if "
                        "this directory holds a COMPLETE cache matching the current run's fingerprint "
                        "(dataset path + image count + VAE id + resolution -- see "
                        "latent_cache_fingerprint/latent_cache_is_complete), it is loaded instead of "
                        "re-encoding (PersistentLatentCache); otherwise the dataset is encoded ONCE "
                        "and written here (build_persistent_latent_cache) before training proceeds. "
                        "Eliminates the ~10-15 min (FFHQ) / ~50 min (Bedroom) re-encode tax that "
                        "otherwise repeats on every preemption+requeue. Sampling stochasticity is "
                        "preserved: mu/logvar are cached, not an already-sampled latent.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--arch_plan", default=None,
                   help="Path to an edm_plans.json from dit_arch_alloc.build_plan. When set, "
                        "each phase-student is a FRESH from-scratch NarrowDiT built from that "
                        "phase's cfg (NOT teacher-initialized) and distilled from the teacher via "
                        "the usual GT+KD loss. The plan's per-phase bins supply the timestep "
                        "grouping (global variant = 1 phase over the full range), so --grouping_json "
                        "is ignored in this mode. Use with --variant to pick the plan key.")
    p.add_argument("--variant", default=None,
                   help="With --arch_plan: which plan variant key to use "
                        "(global|uniform_blockwise|blockwise_capacity|layerwise_capacity).")
    p.add_argument("--ckpt_every", type=int, default=0,
                   help="If >0, save phase student every N optimizer steps for the FID "
                        "curve. Legacy: phase_<p>/step_<k>.pt (bare state_dict). With "
                        "--full_ckpt: phase_<p>/curve/step_<k>.pt ({model,ema}), pruned "
                        "to the newest --curve_keep.")
    p.add_argument("--full_ckpt", action="store_true", default=False,
                   help="Opt-in full checkpointing for EXACT resume. Writes a rolling "
                        "last.pt (model+ema+optimizer+scheduler+step+epoch+rng), a one-time "
                        "pre_cooldown.pt at WSD cooldown start, a final student.pt "
                        "(kind=final), and coarse curve/step_<k>.pt; retains only those "
                        "(+ best.pt). Default OFF => byte-identical weights-only behavior.")
    p.add_argument("--curve_keep", type=int, default=10,
                   help="With --full_ckpt: keep only the newest N curve/step_<k>.pt.")
    p.add_argument("--keep_prev_ckpt", action="store_true", default=False,
                   help="Preemption hardening: with "
                        "--full_ckpt, before overwriting last.pt/pre_cooldown.pt/best.pt/"
                        "student.pt, rename the existing file to '<name>.prev' first, so a "
                        "filesystem-level-corrupt checkpoint (not a torn write -- the atomic "
                        "tmp+os.replace already prevents that -- but e.g. bit rot found "
                        "later) has a one-generation-back fallback on disk. Default OFF "
                        "(byte-identical disk footprint to every run before this flag "
                        "existed).")
    p.add_argument("--ema_beta", type=float, default=0.0,
                   help="EMA decay for a shadow copy of the student, updated every "
                        "optimizer step as beta*ema + (1-beta)*model (mirrors "
                        "pace.edm_distillation.update_ema). Default 0.0 = OFF (no EMA created/updated). "
                        "The EMA is consumed by the checkpoint/eval tasks (Tasks 4/6).")
    p.add_argument("--gt_weight", type=float, default=1.0,
                   help="Weight on the ground-truth diffusion MSE loss.")
    p.add_argument("--kd_weight", type=float, default=1.0,
                   help="Weight on the distillation loss (MSE between student and frozen-teacher "
                        "predictions on the same x_t). Set 0 to recover pure GT fine-tuning.")
    p.add_argument("--cfg_label_drop", type=float, default=0.0,
                   help="CFG-consistent KD label dropout: per batch, remap a Bernoulli(p) subset "
                        "of labels to the null-class index (= num_classes) for BOTH the teacher "
                        "KD forward and the student forward, so on dropped samples the student "
                        "distills the teacher's unconditional function -- training the null row "
                        "that classifier-free guidance needs at eval. Default 0.0 = OFF (labels "
                        "untouched, no RNG consumed; existing runs byte-identical).")
    p.add_argument("--lr_min_ratio", type=float, default=0.01)
    p.add_argument("--lr_schedule", choices=["cosine", "wsd"], default="cosine",
                   help="LR schedule. 'cosine' (default) = current CosineAnnealingLR behavior. "
                        "'wsd' = warmup-stable-decay: linear warmup over --warmup_steps, constant "
                        "peak LR, then 1-sqrt(frac) decay to 0 over the last --cooldown_frac.")
    p.add_argument("--warmup_steps", type=int, default=1000,
                   help="WSD only: linear warmup steps from 0 to peak LR.")
    p.add_argument("--cooldown_frac", type=float, default=0.20,
                   help="WSD only: fraction of total steps spent in the final decay-to-0 phase.")
    p.add_argument("--optimizer", choices=["adamw", "adam"], default="adamw",
                   help="Optimizer. 'adamw' (default) = current torch.optim.AdamW behavior. "
                        "'adam' = torch.optim.Adam (weight_decay applied as L2, not decoupled).")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # periodic held-out validation (opt-in). --val_every 0 (default) = OFF: no
    # val loader is built and validate() is never called -> byte-identical run.
    p.add_argument("--val_every", type=int, default=0,
                   help="If >0, every N optimizer steps run a held-out validation pass on "
                        "the EMA weights (raw student if no EMA), log phase_<p>/val_loss to "
                        "wandb, and (with --full_ckpt) save best.pt at the lowest val_loss. "
                        "Default 0 = OFF (no val loader built, no validate calls).")
    p.add_argument("--val_max_batches", type=int, default=16,
                   help="Number of held-out batches averaged per validation pass.")
    p.add_argument("--val_batch_size", type=int, default=128,
                   help="Batch size for the held-out validation loader.")
    p.add_argument("--val_min_delta", type=float, default=1e-4,
                   help="Minimum val_loss improvement over the running best to count as a "
                        "new best (BestTracker; drives best.pt saving).")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--phases", default=None,
                   help="Comma-separated phase indices to train (e.g. '0,1'). Defaults to all.")

    # logging
    p.add_argument("--device", default="cuda")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_run_id", default=None,
                   help="Stable wandb run id; with resume='allow' a requeued job logs to "
                        "the SAME run/file instead of starting a new one.")
    return p.parse_args()


def main():
    t0 = time.perf_counter()
    args = parse_args()

    configure_dit_repo(args.dit_repo)
    if args.arch_plan and not args.variant:
        raise SystemExit("--arch_plan requires --variant")
    if args.arch_plan and args.model_type not in ("dit_micro", "dit_xl"):
        raise SystemExit(
            "--arch_plan (NarrowDiT width allocation) is only defined for "
            "--model_type dit_micro or dit_xl"
        )
    if not args.arch_plan and not args.grouping_json:
        raise SystemExit("--grouping_json is required unless --arch_plan is set")
    if int(getattr(args, "grad_accum", 1) or 1) < 1:
        raise SystemExit("--grad_accum must be >= 1")
    if args.unconditional and float(args.cfg_label_drop or 0.0) > 0.0:
        raise SystemExit("--unconditional is incompatible with --cfg_label_drop (single "
                          "dummy class has no null row / nothing meaningful to drop to)")
    if getattr(args, "force_label", None) is not None:
        if args.unconditional:
            raise SystemExit("--force_label and --unconditional are mutually exclusive "
                              "(both fix every label to a single row; pass one)")
        if float(args.cfg_label_drop or 0.0) > 0.0:
            raise SystemExit("--force_label is incompatible with --cfg_label_drop (single "
                              "fixed row has no null row / nothing meaningful to drop to)")
        if args.force_label < 0:
            raise SystemExit("--force_label must be >= 0")
    # --- Karras/EDM regularizers (both opt-in, defaults off) -------------------
    # Only wired for the from-scratch NarrowDiT path: --net_dropout needs NarrowDiT
    # blocks, and --augment_prob additionally needs the raw-pixel EDM path (the pipe
    # augments images in [-1,1] before the EDM corruption) plus the NarrowDiT
    # augment-conditioning input.
    if float(args.net_dropout or 0.0) < 0.0 or float(args.net_dropout or 0.0) >= 1.0:
        raise SystemExit("--net_dropout must be in [0, 1)")
    if float(args.net_dropout or 0.0) > 0.0 and not args.arch_plan:
        raise SystemExit("--net_dropout requires --arch_plan (NarrowDiT students)")
    if float(args.augment_prob or 0.0) < 0.0 or float(args.augment_prob or 0.0) > 1.0:
        raise SystemExit("--augment_prob must be in [0, 1]")
    if float(args.augment_prob or 0.0) > 0.0:
        if not args.arch_plan:
            raise SystemExit("--augment_prob requires --arch_plan (NarrowDiT students)")
        if args.model_type != "dit_micro" or args.diffusion != "edm":
            raise SystemExit(
                "--augment_prob is only wired for --model_type dit_micro --diffusion edm "
                "(the raw-pixel EDM path the augmentation pipe and its conditioning target)"
            )
        if args.hflip:
            # EDM's CIFAR-10 pipe passes xflip=1e8 so its x-flip fires with probability
            # 1 -- a dataset-level --hflip on top would flip a second time WITHOUT that
            # being recorded in augment_labels, breaking the non-leaky guarantee the
            # conditioning vector exists to provide. Enforced, not warned.
            raise SystemExit(
                "--augment_prob > 0 is mutually exclusive with --hflip: the EDM "
                "augmentation pipe already applies a random x-flip with probability 1 "
                "and reports it in augment_labels. Drop --hflip."
            )
        if float(args.kd_weight or 0.0) > 0.0:
            # The KD teacher is a stock DiT with no augment-conditioning input, so it
            # cannot be told which augmentation was applied -- its prediction on an
            # augmented batch would be an ill-defined distillation target.
            raise SystemExit(
                "--augment_prob > 0 requires --kd_weight 0: the stock-DiT teacher has no "
                "augment-conditioning input, so a KD target computed on augmented images "
                "would be ill-defined."
            )

    device, rank, world_size = init_distributed(args.device)
    args.device = device
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
    if is_distributed():
        dist.barrier()
    set_seed(args.seed + rank)
    compute_dtype = choose_dtype(args.dtype)

    use_wandb = args.wandb_project is not None and is_main_process()
    if use_wandb:
        if not HAS_WANDB:
            raise ImportError("--wandb_project set but wandb is not installed.")
        init_kwargs = dict(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
        if args.wandb_run_id:
            # Stable id + resume -> a requeued/preempted job continues the SAME wandb run.
            init_kwargs.update(id=args.wandb_run_id, resume="allow")
        wandb.init(**init_kwargs)
        # Run.get_url() is missing in older wandb releases; fall back to the .url attribute.
        run_url = getattr(wandb.run, "get_url", lambda: getattr(wandb.run, "url", "n/a"))()
        print(f"[wandb] run: {run_url}", flush=True)

    if is_main_process():
        print(f"world_size={world_size}, model_type={args.model_type}, dtype={args.dtype}")

    # Phase boundaries. In --arch_plan mode the plan's per-phase bins supply the
    # grouping (and the fresh NarrowDiT cfg for each phase); otherwise read the
    # standard timestep_grouping.json.
    arch_cfgs: Optional[List[Dict[str, Any]]] = None
    if args.arch_plan:
        phases, num_bins, arch_cfgs = load_phases_from_arch_plan(args.arch_plan, args.variant)
    else:
        phases, num_bins = load_phases(args.grouping_json)
    total_num_phases = len(phases)  # full N, before any --phases subset
    if args.phases is not None:
        wanted = {int(x) for x in args.phases.split(",")}
        phases = [p for p in phases if p["index"] in wanted]
        if not phases:
            raise ValueError(f"No phases matched --phases={args.phases}")
    if is_main_process():
        print(f"phases: {[(p['index'], p['start'], p['end']) for p in phases]}, num_bins={num_bins}")

    # Teacher. Skipped entirely (no load, no forward) when --kd_weight 0 under
    # --arch_plan: the student is a FRESH from-scratch NarrowDiT there (never
    # teacher-initialized) and the KD forward is already gated on kd_weight>0
    # everywhere below, so with kd_weight==0 the teacher would otherwise be
    # loaded and never used -- data-loss-only (--gt_weight only) training. The
    # teacher-initialized path always needs the teacher as the student's
    # weight-init source regardless of kd_weight, so it is unaffected.
    skip_teacher = should_skip_teacher(args)
    if skip_teacher:
        if is_main_process():
            print(f"--kd_weight={args.kd_weight} with --arch_plan: skipping teacher "
                  f"load/forward (data-loss-only training)")
        teacher = None
    else:
        if is_main_process():
            print(f"loading teacher: {args.teacher_checkpoint}")
        teacher = load_teacher_for_model_type(args, device, compute_dtype)
        teacher.eval().requires_grad_(False)
    in_channels = get_in_channels(args, teacher)
    alpha_bar = make_ddpm_alpha_schedule(num_timesteps=args.num_timesteps).to(device)

    # Dataset (latent-space models -- dit_xl -- train on image_size images
    # pre-encoded to VAE latents; dit_micro on raw 32x32 RGB).
    _latent_space = args.model_type == "dit_xl"
    if is_main_process():
        print(f"building dataset: {args.dataset} (image_size={args.image_size if _latent_space else 32})")
    image_size = args.image_size if _latent_space else 32
    _trust = (bool(getattr(args, "latent_cache_trust", False)) and _latent_space
              and bool(getattr(args, "latent_cache_dir", None)))
    if _trust and getattr(args, "val_every", 0):
        raise ValueError("--latent_cache_trust requires --val_every 0 (the validation loader needs the images)")
    image_dataset = None if _trust else _build_image_dataset(args, image_size)

    # For the latent DiTs we pre-encode this rank's disjoint slice of images
    # to VAE latents in RAM. Subsequent training reads from RAM only -- no shuffled-read stalls.
    vae = None
    if _latent_space:
        cache_dir = getattr(args, "latent_cache_dir", None)
        hflip = bool(getattr(args, "hflip", False))
        if cache_dir:
            # Opt-in persistent cache (--latent_cache_dir): reuse a completed
            # on-disk cache across job restarts instead of re-encoding the
            # whole dataset through the VAE every process start (the
            # requeue tax under preemption -- see build_persistent_latent_cache
            # / PersistentLatentCache docstrings). Absent/incomplete/
            # fingerprint-mismatched cache -> encode once and write it, then
            # load it via the exact same path a cache HIT would take.
            fingerprint = (trusted_latent_cache_fingerprint(cache_dir) if _trust
                           else latent_cache_fingerprint(args, image_dataset, image_size))
            # Hardening item 7: resolve to a fingerprint-keyed subdirectory
            # unless the flat `cache_dir` already holds a complete cache for
            # THIS exact fingerprint (backward-compatible with every cache
            # built before this existed) -- see resolve_latent_cache_dir's
            # docstring for the clobber incident this prevents.
            shard_dir = resolve_latent_cache_dir(cache_dir, fingerprint)
            if not latent_cache_is_complete(shard_dir, fingerprint):
                if is_main_process():
                    print(f"[latent_cache] no valid persistent cache at {shard_dir} "
                          f"(base --latent_cache_dir={cache_dir}) -> encoding once and writing it")
                print("loading VAE for one-shot pre-encoding...")
                from diffusers import AutoencoderKL
                vae = AutoencoderKL.from_pretrained(_VAE_PRETRAINED_ID).to(device)
                vae.eval().requires_grad_(False)
                build_persistent_latent_cache(
                    image_dataset=image_dataset, vae=vae, cache_dir=shard_dir,
                    fingerprint=fingerprint, rank=rank, world_size=world_size,
                    device=device, encode_batch_size=args.vae_batch_size,
                    num_workers=args.num_workers,
                )
                del vae
                vae = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            elif is_main_process():
                print(f"[latent_cache] found complete persistent cache at {shard_dir} "
                      f"-> loading (VAE encode skipped)")
            latent_cache = PersistentLatentCache(
                cache_dir=shard_dir, rank=rank, world_size=world_size, hflip=hflip,
            )
        else:
            # Default OFF: byte-identical to before --latent_cache_dir existed.
            if is_main_process():
                print("loading VAE for one-shot pre-encoding...")
            from diffusers import AutoencoderKL
            vae = AutoencoderKL.from_pretrained(_VAE_PRETRAINED_ID).to(device)
            vae.eval().requires_grad_(False)
            if is_main_process():
                print("pre-encoding rank-disjoint latent cache (one-shot)...")
            latent_cache = PerRankLatentCache(
                image_dataset=image_dataset, vae=vae,
                rank=rank, world_size=world_size, device=device,
                encode_batch_size=args.vae_batch_size, num_workers=args.num_workers,
            )
            del vae
            vae = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        train_source = latent_cache
        sampler = None  # cache is already rank-sharded; just shuffle in-rank each epoch
        dataloader = DataLoader(
            train_source,
            batch_size=args.batch_size, shuffle=True,
            num_workers=0,  # cache is in RAM; workers add no value and waste CPU
            pin_memory=True, drop_last=True,
        )
    else:
        # DiT-Micro: tiny dataset (CIFAR-10 50k), no VAE step -- standard DistributedSampler.
        sampler = DistributedSampler(image_dataset, shuffle=True, seed=args.seed) if is_distributed() else None
        dataloader = DataLoader(
            image_dataset,
            batch_size=args.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=args.num_workers > 0,
        )

    # Karras/EDM non-leaky augmentation pipeline (--augment_prob). None unless the
    # flag is > 0, in which case the batch loop applies it and threads its per-sample
    # parameter vector into the network as conditioning.
    from pace.edm_augment import CIFAR10_AUGMENT_DIM, build_augment_pipe
    augment_pipe = build_augment_pipe(getattr(args, "augment_prob", 0.0))
    if augment_pipe is not None and is_main_process():
        print(f"[augment] EDM non-leaky pipe ON: p={args.augment_prob} "
              f"(x-flip @p=1 + y-flip/scale/rotate_frac/aniso/translate_frac @p), "
              f"augment_dim={augment_pipe.label_dim}")
    net_dropout = float(getattr(args, "net_dropout", 0.0) or 0.0)

    if is_main_process():
        save_json(os.path.join(args.output_dir, "run_config.json"), {
            "args": vars(args),
            "phases": phases,
            "num_bins": num_bins,
        })

    all_meta = []
    for phase in phases:
        pidx = phase["index"]
        phase_dir = os.path.join(args.output_dir, f"phase_{pidx}")
        if is_main_process():
            os.makedirs(phase_dir, exist_ok=True)
        if is_distributed():
            dist.barrier()

        action = phase_resume_action(phase_dir, full_ckpt=args.full_ckpt)
        if action == "skip":
            if is_main_process():
                print(f"\n=== phase {pidx}: already complete (student.pt) -> skipping ===")
            mp = os.path.join(phase_dir, "training_meta.json")
            if os.path.exists(mp):
                all_meta.append(json.load(open(mp)))
            continue

        if is_main_process():
            print(f"\n=== Training phase {pidx} (bins [{phase['start']}, {phase['end']})) ===")

        if args.arch_plan:
            # From-scratch width-allocation student: build a FRESH NarrowDiT from the
            # plan's per-phase cfg (random init, NOT teacher-initialized -- that is the
            # whole point). It is still distilled from the teacher via the SAME GT+KD loss
            # below (train_phase -> model_forward_for_loss), for whichever forward process
            # --diffusion selects. This branch is deliberately model_type-agnostic: the
            # ONLY difference from the teacher-initialized path is fresh-NarrowDiT
            # construction here vs deepcopy(teacher), so the student is called
            # identically downstream.
            #
            #   * dit_micro (--diffusion edm): the student's forward(x, c_noise, y) returns
            #     the F-space output F_theta, exactly like the teacher net; the EDM
            #     preconditioning (c_in/c_skip/c_out) is applied identically to both inside
            #     model_forward_for_loss -- no separate precond wrapper needed (mirrors how
            #     load_dit_micro_network returns a bare F-space net). NarrowDiT learn_sigma
            #     is False here, so it outputs in_channels and is used directly.
            #   * dit_xl (--diffusion ddpm): the student's forward(x, t, y) returns eps in
            #     the first in_channels of a 2*in_channels output (learn_sigma=True, matching
            #     the DiT-XL/2 teacher). model_forward_for_loss's ddpm branch takes
            #     pred = raw[:, :in_channels] -- the SAME slice applied to a teacher-
            #     initialized XL student (see the "if args.model_type == 'dit_xl': pred =
            #     raw[:, :in_channels]" line in model_forward_for_loss). VAE latents +
            #     teacher loading are handled upstream identically (load_teacher_for_model_type
            #     -> load_dit_network; PerRankLatentCache VAE pre-encode). An UNCONDITIONAL
            #     variant of this (--kd_weight 0, no teacher, --dataset image_folder,
            #     --unconditional -- the from-scratch DiT-B/2 teachers on
            #     LSUN Bedroom and FFHQ 256 latents) forces every label to 0 in
            #     model_forward_for_loss so the dataset's sentinel -1 label never reaches
            #     the (single-row) LabelEmbedder.
            from pace.dit_arch_alloc import build_narrow_dit, count_dit_params
            cfg = arch_cfgs[pidx]
            # Opt-in Karras/EDM regularizers are recorded IN the cfg so that (a) the
            # built model matches the flags and (b) arch_cfg.json below stays the single
            # self-contained rebuild recipe for eval (build_narrow_dit(arch_cfg["cfg"])).
            # Untouched when the flags are off -> cfg is the plan's cfg exactly as before.
            if augment_pipe is not None or net_dropout > 0.0:
                cfg = dict(cfg)
                if augment_pipe is not None:
                    cfg["augment_dim"] = int(augment_pipe.label_dim)
                    assert cfg["augment_dim"] == CIFAR10_AUGMENT_DIM
                if net_dropout > 0.0:
                    cfg["dropout"] = net_dropout
            raw_student = build_narrow_dit(cfg).to(device)
            realized = count_dit_params(raw_student)
            if is_main_process():
                save_json(os.path.join(phase_dir, "arch_cfg.json"), {
                    "variant": args.variant,
                    "arch_plan": os.path.abspath(args.arch_plan),
                    "phase": pidx,
                    "bins": [phase["start"], phase["end"]],
                    "realized_params": int(realized),
                    "cfg": cfg,
                })
                print(f"  [arch:{args.variant}] phase {pidx}: fresh NarrowDiT from scratch, "
                      f"{realized:,} params (plan realized_params)")
            resume_ckpt = None
            if action == "resume":
                if args.full_ckpt:
                    _last_pt = os.path.join(phase_dir, "last.pt")
                    resume_ckpt = validate_full_ckpt_payload(
                        load_full_ckpt(_last_pt, map_location=device), _last_pt)
                    _mid = "last.pt"
                else:
                    resume_ckpt = torch.load(os.path.join(phase_dir, "resume.pt"), map_location=device)
                    _mid = "resume.pt"
                raw_student.load_state_dict(resume_ckpt["model"])
                if is_main_process():
                    print(f"  resuming phase {pidx}: loaded {_mid} (after epoch {resume_ckpt['epoch'] + 1})")
            raw_student.train().requires_grad_(True)
            if is_distributed():
                student_ddp = DDP(raw_student, device_ids=[device.index] if hasattr(device, "index") else None)
            else:
                student_ddp = raw_student

            meta = train_phase(
                args=args,
                teacher=teacher,
                student_ddp=student_ddp,
                raw_student=raw_student,
                vae=vae,
                dataloader=dataloader,
                sampler=sampler,
                phase=phase,
                num_bins=num_bins,
                alpha_bar=alpha_bar,
                in_channels=in_channels,
                compute_dtype=compute_dtype,
                phase_dir=phase_dir,
                use_wandb=use_wandb,
                resume_ckpt=resume_ckpt,
                augment_pipe=augment_pipe,
            )
            meta["arch_plan"] = os.path.abspath(args.arch_plan)
            meta["variant"] = args.variant
            meta["realized_params"] = int(realized)
            if is_main_process():
                save_json(os.path.join(phase_dir, "training_meta.json"), meta)
            all_meta.append(meta)

            del student_ddp, raw_student
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if is_distributed():
                dist.barrier()
            continue

        # Fresh student copy from teacher per phase (teacher-initialized path)
        if is_main_process():
            print(f"  initialising student from teacher")
        raw_student = copy.deepcopy(teacher)
        resume_ckpt = None
        if action == "resume":
            if args.full_ckpt:
                _last_pt = os.path.join(phase_dir, "last.pt")
                resume_ckpt = validate_full_ckpt_payload(
                    load_full_ckpt(_last_pt, map_location=device), _last_pt)
                _mid = "last.pt"
            else:
                resume_ckpt = torch.load(os.path.join(phase_dir, "resume.pt"), map_location=device)
                _mid = "resume.pt"
            raw_student.load_state_dict(resume_ckpt["model"])
            if is_main_process():
                print(f"  resuming phase {pidx}: loaded {_mid} (after epoch {resume_ckpt['epoch'] + 1})")
        raw_student.train().requires_grad_(True)
        if is_distributed():
            student_ddp = DDP(raw_student, device_ids=[device.index] if hasattr(device, "index") else None)
        else:
            student_ddp = raw_student

        meta = train_phase(
            args=args,
            teacher=teacher,
            student_ddp=student_ddp,
            raw_student=raw_student,
            vae=vae,
            dataloader=dataloader,
            sampler=sampler,
            phase=phase,
            num_bins=num_bins,
            alpha_bar=alpha_bar,
            in_channels=in_channels,
            compute_dtype=compute_dtype,
            phase_dir=phase_dir,
            use_wandb=use_wandb,
            resume_ckpt=resume_ckpt,
        )
        if is_main_process():
            save_json(os.path.join(phase_dir, "training_meta.json"), meta)
        all_meta.append(meta)

        # Free student memory before the next phase
        del student_ddp, raw_student
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if is_distributed():
            dist.barrier()

    if is_main_process():
        elapsed = time.perf_counter() - t0
        if args.arch_plan:
            # Record the arch-allocation provenance alongside the per-phase metas so
            # eval can rebuild each NarrowDiT (also mirrored in each phase's arch_cfg.json).
            composite = {
                "mode": "arch_plan",
                "variant": args.variant,
                "arch_plan": os.path.abspath(args.arch_plan),
                "num_phases": total_num_phases,
                "phases": [
                    {"phase": m.get("phase_index"),
                     "bins": [m.get("phase_start"), m.get("phase_end")],
                     "realized_params": m.get("realized_params")}
                    for m in all_meta
                ],
                "per_phase_meta": all_meta,
            }
            save_json(os.path.join(args.output_dir, "all_phases_meta.json"), composite)
        else:
            save_json(os.path.join(args.output_dir, "all_phases_meta.json"), all_meta)
        print(f"\nTotal runtime: {elapsed:.1f}s ({elapsed / 60:.1f} min). Output: {args.output_dir}")
        if use_wandb:
            wandb.log({"runtime_seconds": elapsed})
            wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    main()
