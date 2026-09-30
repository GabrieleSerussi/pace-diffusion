#!/usr/bin/env python3
"""
Sample images from a DiT (teacher or composite-students chain) and compute FID.

Supports two model families:
  --model_type dit_xl    : DiT-XL/2 and the DiT-B/2 latent teachers (eps-prediction,
                           VAE latents).
  --model_type dit_micro : CIFAR-trained pixel-space DiTs (raw RGB; x_0-prediction
                           or, with --diffusion edm, the EDM preconditioned denoiser).

Supports two sampling modes:
  --mode teacher
      Sample with a single (teacher) checkpoint at all timesteps.
  --mode composite
      Phase-routed sampling: a ``timestep_grouping.json`` defines bin boundaries;
      at each denoising step, the current timestep is mapped to a bin, the bin
      to a phase, and the matching ``student_phase_{i}.pt`` is used for the
      forward pass. Switching happens on every step.

Multi-GPU via ``torchrun``: each rank generates a disjoint slice of the total
sample count. After sampling, rank 0 can compute a quick FID via torch-fidelity
(optional, ``pip install torch-fidelity``) against a reference directory of real
images; the paper's DiT FIDs use the ADM evaluator through
``scripts/eval_composite_curve.py`` on the saved samples (``--skip_fid``).

The facebookresearch/DiT checkout is taken from ``--dit_repo`` or ``$DIT_REPO``.
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from PIL import Image as PILImage
from tqdm.auto import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from evaluate_parameters_edm import (
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
from evaluate_parameters_dit import load_dit_network
from evaluate_parameters_dit_micro import load_dit_micro_network

_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from pace.external_repos import configure_dit_repo, import_dit_module


# ---------------------------------------------------------------------------
# Phase routing
# ---------------------------------------------------------------------------

def load_phase_boundaries(grouping_json_path: str) -> Tuple[List[int], int]:
    with open(grouping_json_path) as f:
        g = json.load(f)
    boundaries = list(g["boundaries"])
    num_bins = int(g["num_timesteps"])
    if boundaries[0] != 0 or boundaries[-1] != num_bins:
        raise ValueError(f"Boundaries {boundaries} don't span [0, {num_bins}].")
    return boundaries, num_bins


def load_phase_boundaries_from_arch_cfgs(
    student_dir: str, num_timesteps: Optional[int] = None,
) -> Tuple[List[int], int]:
    """Derive composite phase boundaries from the staged per-phase ``arch_cfg.json``
    ``bins`` (used by ``--arch_plan`` runs, which don't pass a ``timestep_grouping.json``).

    Each phase's ``bins`` is a ``[start, end)`` pair; they must contiguously tile
    ``[0, num_bins]`` (exactly as the trainer's ``load_phases_from_arch_plan`` asserts).
    Returns ``(boundaries, num_bins)`` in the same contract as ``load_phase_boundaries``:
    ``boundaries`` is the sorted list of phase edges and ``num_bins`` its last element.
    For the ``global`` variant this yields a single phase ``[0, num_bins]``.

    ``num_timesteps`` (default ``None``, byte-identical to every call before this
    parameter existed -- still raises ``FileNotFoundError`` when omitted) is an
    opt-in fallback for a genuinely STOCK/plain single-phase composite dir: a
    ``phase_0/`` with no ``arch_cfg.json`` and no ``phase_1`` (no ``--arch_plan``
    was used, so there is nothing to derive per-phase bins from -- e.g. a stock
    fine-tuned DiT-XL/2 full checkpoint). When set, that case degrades to ONE
    phase spanning the WHOLE timestep range ``[0, num_timesteps)``, identical to
    what a single-phase ``timestep_grouping.json`` would describe (and
    numerically identical to ``--mode teacher``'s "same net at every timestep",
    since a single phase's ``CompositeRoutedModel`` always routes to that one
    student). A dir with a
    ``phase_1`` present still raises: silently collapsing an ambiguous
    multi-phase composite (missing ``--grouping_json``, no arch_cfg.json either)
    down to one phase would be guessing, not a safe default.
    """
    pidx = 0
    ranges: List[Tuple[int, int]] = []
    while os.path.exists(os.path.join(student_dir, f"phase_{pidx}", "arch_cfg.json")):
        with open(os.path.join(student_dir, f"phase_{pidx}", "arch_cfg.json")) as f:
            ac = json.load(f)
        s, e = int(ac["bins"][0]), int(ac["bins"][1])
        ranges.append((s, e))
        pidx += 1
    if not ranges:
        if (num_timesteps is not None
                and os.path.exists(os.path.join(student_dir, "phase_0"))
                and not os.path.exists(os.path.join(student_dir, "phase_1"))):
            return [0, num_timesteps], num_timesteps
        raise FileNotFoundError(f"no phase_*/arch_cfg.json under {student_dir}")
    ranges.sort(key=lambda r: r[0])
    boundaries = [ranges[0][0]]
    for s, e in ranges:
        if s != boundaries[-1]:
            raise ValueError(
                f"arch_cfg bins {ranges} don't contiguously tile from {boundaries[-1]}"
            )
        boundaries.append(e)
    num_bins = boundaries[-1]
    if boundaries[0] != 0:
        raise ValueError(f"arch_cfg bins {ranges} don't start at 0")
    return boundaries, num_bins


def timestep_to_phase_index(
    t: int, boundaries: List[int], num_bins: int, num_timesteps: int,
) -> int:
    """Map an integer DDPM timestep in [0, num_timesteps) to its phase index.

    Bin numbering convention (matches the analysis pipeline): bin 0 = highest
    noise (t near num_timesteps-1), bin num_bins-1 = lowest noise (t near 0).
    """
    t_clamped = max(0, min(num_timesteps - 1, int(t)))
    # FLOOR of the schedule fraction, not round: the trainer's bin b spans the
    # timestep RANGE t in [round(t_max*(1 - (b+1)/nb)), round(t_max*(1 - b/nb))]
    # (see train_phase_students.sample_phase_timesteps), so the bin containing t
    # is floor((1 - t/t_max) * nb). Rounding instead would shift every bin edge
    # by half a bin and route timesteps to the neighbouring phase's specialist
    # (regression: tests/test_phase_routing_roundtrip.py). Matches the sibling
    # router _sigma_to_phase (int() truncation).
    bin_idx = math.floor((1.0 - t_clamped / (num_timesteps - 1)) * num_bins)
    bin_idx = max(0, min(num_bins - 1, bin_idx))
    for phase_idx in range(len(boundaries) - 1):
        if boundaries[phase_idx] <= bin_idx < boundaries[phase_idx + 1]:
            return phase_idx
    return len(boundaries) - 2  # fallback to last phase


class CompositeRoutedModel(nn.Module):
    """At each forward call, route to whichever phase student covers the given
    timestep. All timesteps in a batch are assumed equal (DDPM/DDIM both
    broadcast a single t value across the batch within one step)."""

    def __init__(
        self,
        students: Dict[int, nn.Module],
        boundaries: List[int],
        num_bins: int,
        num_timesteps: int,
    ):
        super().__init__()
        self.students = students
        self.boundaries = boundaries
        self.num_bins = num_bins
        self.num_timesteps = num_timesteps
        self.available_phases = sorted(students.keys())

    def _pick_student(self, t: torch.Tensor) -> nn.Module:
        t_int = int(t.flatten()[0].item())
        pidx = timestep_to_phase_index(t_int, self.boundaries, self.num_bins, self.num_timesteps)
        if pidx not in self.students:
            pidx = min(self.available_phases, key=lambda p: abs(p - pidx))
        return self.students[pidx]

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        return self._pick_student(t)(x, t, y)

    def forward_with_cfg(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, cfg_scale: float):
        """Classifier-free guidance variant: delegates to the chosen student's
        own forward_with_cfg (DiT-XL/2 provides this)."""
        return self._pick_student(t).forward_with_cfg(x, t, y, cfg_scale)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _student_state_from_payload(payload):
    """Extract the weights to load from a saved phase checkpoint payload.

    With opt-in full checkpoints (``--full_ckpt``), a phase's ``student.pt`` /
    ``curve/step_<k>.pt`` may now be one of three shapes:
      - a full-ckpt / curve dict ``{"model": sd, "ema": sd_or_None, ...}`` -> prefer
        ``ema`` when present (mirrors upstream eval, which scores the EMA weights),
        else fall back to ``model``;
      - a dict carrying only ``{"model": sd}`` (ema absent/None) -> use ``model``;
      - a bare ``state_dict`` (LEGACY, non-full runs) -> return unchanged, byte-identical
        to the pre-Task-6 path.
    """
    if isinstance(payload, dict) and "model" in payload:
        ema = payload.get("ema")
        return ema if ema is not None else payload["model"]
    return payload


def load_dit_xl_from_state_dict_or_path(path: str, device, dtype) -> nn.Module:
    """Load a DiT-XL/2 student .pt -- our trainer saves a raw state_dict,
    so we instantiate the architecture and load_state_dict."""
    DiT_models = import_dit_module("models").DiT_models
    model = DiT_models["DiT-XL/2"](input_size=32, num_classes=1000).to(device)
    state = torch.load(path, map_location="cpu", weights_only=False)
    # Full-ckpt runs (--full_ckpt) stage {"model", "ema", ...}; prefer EMA (mirrors
    # load_narrow_dit_from_dir / _student_state_from_payload below -- this loader
    # predates that helper and had its own ad hoc "model"-only extraction, which
    # silently scored raw (non-EMA) weights for every stock-DiT-XL/2 full-ckpt eval).
    state = _student_state_from_payload(state)
    model.load_state_dict(state)
    model.eval().requires_grad_(False)
    return model


def load_dit_micro_from_state_dict_or_path(
    path: str, device, num_heads: int,
) -> nn.Module:
    """Wrap load_dit_micro_network so it accepts either:
    - the upstream packed checkpoint (dict with 'model' key), or
    - our trainer's plain state_dict.
    """
    from evaluate_parameters_dit_micro import DiTMicro
    state = torch.load(path, map_location="cpu", weights_only=False)
    packed = isinstance(state, dict) and ("model" in state or "ema" in state)
    if packed:
        # Upstream-format teacher (packed ckpt): delegate to original loader.
        return load_dit_micro_network(path, device=device, num_heads=num_heads)
    # Plain state_dict (our student): instantiate + load.
    model = DiTMicro(num_heads=num_heads).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    extra_unexpected = [k for k in unexpected if not k.endswith("pos_embed")]
    extra_missing = [k for k in missing if not k.endswith("pos_embed")]
    if extra_missing or extra_unexpected:
        raise RuntimeError(
            f"Student state-dict mismatch loading {path}. "
            f"Missing (excl pos_embed): {extra_missing[:3]}; "
            f"Unexpected (excl pos_embed): {extra_unexpected[:3]}."
        )
    model.eval().requires_grad_(False)
    return model


def load_narrow_dit_from_dir(phase_dir: str, device, attn_impl: str = "manual",
                              ckpt_path: Optional[str] = None) -> nn.Module:
    """Rebuild a width-allocation NarrowDiT phase-student from its staged dir.

    The arch-plan trainer (``--arch_plan``) writes ``arch_cfg.json`` (with the full
    NarrowDiT kwargs under ``cfg``) next to the phase checkpoint. Instantiate that
    exact architecture and load the plain state_dict from ``student.pt``. The result
    is a bare F-space net with the same ``forward(x, t, y)`` signature as the
    DiTMicro teacher, so it plugs into the identical EDM
    precond/sampling path downstream.

    ``attn_impl`` (default ``"manual"``, byte-identical to every call before this
    parameter existed) selects NarrowAttention's execution path ("manual"/"sdpa");
    see ``pace.dit_arch_alloc.build_narrow_dit`` / ``NarrowAttention``. Same
    weights either way -- FID evals may use either (validated within noise);
    only throughput claims require sdpa.

    ``ckpt_path`` (default ``None``, byte-identical to every call before this
    parameter existed -- loads ``phase_dir/student.pt``): an explicit absolute
    checkpoint path to load INSTEAD, while still rebuilding the architecture from
    THIS ``phase_dir``'s ``arch_cfg.json`` (for example an earlier
    ``curve/step_<N>.pt`` of the same run).
    """
    from pace.dit_arch_alloc import build_narrow_dit
    with open(os.path.join(phase_dir, "arch_cfg.json")) as f:
        arch_cfg = json.load(f)
    model = build_narrow_dit(arch_cfg["cfg"], attn_impl=attn_impl).to(device)
    ckpt = ckpt_path if ckpt_path is not None else os.path.join(phase_dir, "student.pt")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    # Full-ckpt runs stage a dict {"model", "ema", ...}; prefer EMA. Legacy runs stage
    # a bare state_dict (returned unchanged). See _student_state_from_payload.
    state = _student_state_from_payload(payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    extra_unexpected = [k for k in unexpected if not k.endswith("pos_embed")]
    extra_missing = [k for k in missing if not k.endswith("pos_embed")]
    if extra_missing or extra_unexpected:
        raise RuntimeError(
            f"NarrowDiT state-dict mismatch loading {ckpt}. "
            f"Missing (excl pos_embed): {extra_missing[:3]}; "
            f"Unexpected (excl pos_embed): {extra_unexpected[:3]}."
        )
    model.eval().requires_grad_(False)
    return model


def build_phase_student_from_dir(phase_dir: str, model_type: str, device, dtype, num_heads: int,
                                  attn_impl: str = "manual") -> nn.Module:
    """Reconstruct one composite phase-student from its staged dir.

    Branches on which config file the dir carries:
      - ``arch_cfg.json``  -> width-allocation NarrowDiT (``--arch_plan`` runs),
        rebuilt via ``build_narrow_dit(cfg)`` + ``load_state_dict``.
      - otherwise -> a teacher-architecture student (stock DiT-XL/2 or DiT-Micro).

    The returned model is a bare F-space net (same forward signature in every
    case), so all downstream compositing/sampling stays identical.

    ``attn_impl`` (default ``"manual"``, byte-identical to every call before this
    parameter existed) is forwarded to ``load_narrow_dit_from_dir`` ONLY -- the
    stock (non-NarrowDiT) DiT-XL/DiT-Micro classes have no such switch.
    """
    ckpt = os.path.join(phase_dir, "student.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"missing student checkpoint: {ckpt}")

    arch_cfg_path = os.path.join(phase_dir, "arch_cfg.json")
    if os.path.exists(arch_cfg_path):
        # arch_cfg.json is self-contained (cfg carries in_channels/learn_sigma/num_classes/
        # hidden_size/depth/per_block), so build_narrow_dit works for dit_micro AND dit_xl.
        # DEFENSIVE: this arch_cfg.json check PRECEDES the teacher-architecture path
        # below, so every --arch_plan (--full_ckpt) phase dir is routed to
        # load_narrow_dit_from_dir (which prefers EMA).
        return load_narrow_dit_from_dir(phase_dir, device, attn_impl=attn_impl)

    # Teacher-architecture student (a stock DiT-XL/2 or DiT-Micro fine-tune).
    if model_type == "dit_xl":
        return load_dit_xl_from_state_dict_or_path(ckpt, device, dtype)
    return load_dit_micro_from_state_dict_or_path(ckpt, device, num_heads)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def build_dit_xl_diffusion(num_steps: int):
    """DDIM-respaced diffusion for DiT-XL/2 (eps-prediction, learned sigma)."""
    create_diffusion = import_dit_module("diffusion").create_diffusion
    return create_diffusion(timestep_respacing=f"ddim{num_steps}")


def build_dit_micro_diffusion(num_steps: int):
    """DDIM-respaced diffusion for DiT-Micro (x_0-prediction, fixed sigma)."""
    create_diffusion = import_dit_module("diffusion").create_diffusion
    return create_diffusion(
        timestep_respacing=f"ddim{num_steps}",
        predict_xstart=True,
        learn_sigma=False,
    )


@torch.no_grad()
def sample_dit_xl(
    model_obj,           # the (composite or single) DiT model
    diffusion,           # SpacedDiffusion
    batch_size: int,
    num_classes: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    cfg_scale: float = 1.0,
    sampler: str = "ddim",
    class_idx_override: Optional[int] = None,
) -> torch.Tensor:
    """Sample a batch of DiT-XL/2 latents.

    sampler: "ddim" (deterministic, fast) or "ddpm" (stochastic, p_sample_loop, paper protocol)
    cfg_scale=1.0 means no CFG; >1 uses the model's forward_with_cfg with the null class
    class_idx_override: if set, every sample uses this fixed label index instead of a
      uniform-random real class -- the unconditional-eval convention for a checkpoint
      trained via ``--force_label``/``--unconditional`` (see
      ``evaluate_parameters_dit.py``'s own ``--class_idx_override``, whose probe-time
      convention this mirrors: 1000 for a DiT-XL/2 fine-tune's trained null/CFG row,
      0 for a from-scratch NarrowDiT's single unconditional class). Default ``None``
      is byte-identical to every call before this parameter existed (uniform-random
      real class per sample).
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(batch_size, 4, 32, 32, device=device, dtype=torch.float32, generator=gen)
    if class_idx_override is not None:
        y = torch.full((batch_size,), class_idx_override, dtype=torch.long, device=device)
    else:
        y = torch.randint(0, num_classes, (batch_size,), device=device, generator=gen)

    if cfg_scale > 1.0 + 1e-6:
        # CFG path: double the latent batch and pair each class with the null class.
        y_null = torch.full_like(y, num_classes)
        y_in = torch.cat([y, y_null], dim=0)
        z_in = torch.cat([z, z], dim=0)
        model_fn = model_obj.forward_with_cfg
        model_kwargs = dict(y=y_in, cfg_scale=cfg_scale)
        shape = z_in.shape
        x_T = z_in
    else:
        model_fn = model_obj.forward
        model_kwargs = dict(y=y)
        shape = z.shape
        x_T = z

    autocast_enabled = dtype != torch.float32
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
        if sampler == "ddpm":
            latents = diffusion.p_sample_loop(
                model_fn, shape, x_T, clip_denoised=False,
                model_kwargs=model_kwargs, progress=False, device=device,
            )
        else:
            latents = diffusion.ddim_sample_loop(
                model_fn, shape, x_T, clip_denoised=False,
                model_kwargs=model_kwargs, progress=False, device=device,
            )

    if cfg_scale > 1.0 + 1e-6:
        latents = latents[: batch_size]  # drop the unconditional half
    return latents  # (B, 4, 32, 32)


@torch.no_grad()
def sample_dit_micro(
    model_fn,
    diffusion,
    batch_size: int,
    num_classes: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    """Sample a batch of raw 32x32 RGB images via DDIM (x_0-prediction)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(batch_size, 3, 32, 32, device=device, dtype=torch.float32, generator=gen)
    y = torch.randint(0, num_classes, (batch_size,), device=device, generator=gen)

    autocast_enabled = dtype != torch.float32
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
        samples = diffusion.ddim_sample_loop(
            model_fn, x.shape, x, clip_denoised=True,
            model_kwargs=dict(y=y), progress=False, device=device,
        )
    return samples  # (B, 3, 32, 32) in [-1, 1]


# ---------------------------------------------------------------------------
# EDM sampling (for the normalcomputing DiT-Micro preconditioned denoiser)
# ---------------------------------------------------------------------------

def _edm_denoise(net, x, sigma, y, sigma_data=0.5):
    s = sigma
    c_in = 1.0 / math.sqrt(s ** 2 + sigma_data ** 2)
    c_skip = sigma_data ** 2 / (s ** 2 + sigma_data ** 2)
    c_out = s * sigma_data / math.sqrt(s ** 2 + sigma_data ** 2)
    c_noise = 0.25 * math.log(s)
    t = torch.full((x.shape[0],), c_noise, device=x.device)
    return c_skip * x + c_out * net(c_in * x, t, y)


def _sigma_to_phase(sigma, boundaries, num_bins, sigma_min, sigma_max, rho):
    """Map a sigma to its phase index via the descending Karras schedule fraction."""
    smin_r, smax_r = sigma_min ** (1.0 / rho), sigma_max ** (1.0 / rho)
    f = (sigma ** (1.0 / rho) - smax_r) / (smin_r - smax_r)  # 0=high noise .. 1=low noise
    bin_idx = int(min(max(f * num_bins, 0), num_bins - 1))
    for p in range(len(boundaries) - 1):
        if boundaries[p] <= bin_idx < boundaries[p + 1]:
            return p
    return len(boundaries) - 2


@torch.no_grad()
def sample_dit_micro_edm(
    pick_net, batch_size, num_classes, device, dtype, seed,
    steps=50, sigma_min=0.002, sigma_max=80.0, rho=7.0, sigma_data=0.5, cfg_scale=2.0,
):
    """EDM Heun sampler. ``pick_net(sigma)`` returns the network to use at that sigma
    (a single model for teacher; the phase-routed student for composite)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    i = torch.arange(steps, device=device, dtype=torch.float64)
    sig = (sigma_max ** (1 / rho) + i / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    sig = torch.cat([sig, sig.new_zeros(1)])
    y = torch.randint(0, num_classes, (batch_size,), device=device, generator=gen)
    yn = torch.full_like(y, num_classes)  # null class index = num_classes (10)
    autocast_enabled = dtype != torch.float32

    def D(x, s):
        net = pick_net(s)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
            if cfg_scale == 1.0:
                return _edm_denoise(net, x, s, y, sigma_data).float()
            dc = _edm_denoise(net, x, s, y, sigma_data).float()
            dn = _edm_denoise(net, x, s, yn, sigma_data).float()
            return dn + cfg_scale * (dc - dn)

    x = torch.randn(batch_size, 3, 32, 32, device=device, generator=gen) * sig[0]
    for k in range(steps):
        s, s1 = sig[k].item(), sig[k + 1].item()
        d = (x - D(x, s)) / s
        xn = x + (s1 - s) * d
        if s1 > 0:
            d2 = (xn - D(xn, s1)) / s1
            xn = x + (s1 - s) * 0.5 * (d + d2)
        x = xn
    return x.clamp(-1, 1)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_images_to_dir(images: torch.Tensor, out_dir: str, start_idx: int) -> int:
    """Save a (B, 3, H, W) tensor in [-1, 1] as PNGs. Returns the next start index."""
    os.makedirs(out_dir, exist_ok=True)
    imgs = (images.clamp(-1, 1) + 1.0) * 127.5
    imgs = imgs.to(torch.uint8).cpu().numpy()
    for i in range(imgs.shape[0]):
        arr = imgs[i].transpose(1, 2, 0)
        PILImage.fromarray(arr).save(os.path.join(out_dir, f"img_{start_idx + i:06d}.png"))
    return start_idx + imgs.shape[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", required=True, choices=["dit_xl", "dit_micro"])
    p.add_argument("--dit_repo", default=None,
                   help="facebookresearch/DiT checkout (default: $DIT_REPO).")
    p.add_argument("--mode", required=True, choices=["teacher", "composite"])
    p.add_argument("--output_dir", required=True)

    # checkpoints
    p.add_argument("--teacher_checkpoint", default=None,
                   help="Teacher .pt path. Required for --mode teacher and also used for "
                        "DiT-XL/2 model construction in composite mode if needed.")
    p.add_argument("--student_dir", default=None,
                   help="Path to a students directory containing phase_{0,1}/student.pt files. "
                        "Required for --mode composite.")
    p.add_argument("--grouping_json", default=None,
                   help="timestep_grouping.json. Required for --mode composite.")

    # sampling
    p.add_argument("--num_samples", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_steps", type=int, default=50, help="Sampler step count.")
    p.add_argument("--num_timesteps", type=int, default=1000,
                   help="Underlying diffusion process length (matches training).")
    p.add_argument("--sampler", default="ddim", choices=["ddim", "ddpm"],
                   help="(ddpm-family models) ddim fast/deterministic; ddpm is the DiT paper's protocol.")
    p.add_argument("--diffusion", default="ddpm", choices=["ddpm", "edm"],
                   help="Forward process. 'edm' uses the EDM Heun sampler (pixel-space CIFAR-10 DiTs).")
    p.add_argument("--sigma_min", type=float, default=0.002)
    p.add_argument("--sigma_max", type=float, default=80.0)
    p.add_argument("--sigma_data", type=float, default=0.5)
    p.add_argument("--rho", type=float, default=7.0)
    p.add_argument("--cfg_scale", type=float, default=1.0,
                   help="Classifier-free guidance scale. 1.0 disables CFG. DiT paper uses 1.5; EDM DiT-Micro best at 2.0.")
    p.add_argument("--class_idx_override", type=int, default=None,
                   help="dit_xl only: force every sample to this fixed label index instead "
                        "of a uniform-random real class. Use the same index the checkpoint "
                        "was trained/probed with (e.g. 1000 for a DiT-XL/2 fine-tune's "
                        "trained null/CFG row via --force_label 1000, 0 for a from-scratch "
                        "NarrowDiT's single unconditional class via --unconditional) -- "
                        "mirrors evaluate_parameters_dit.py's own --class_idx_override "
                        "convention. Default None is byte-identical to every run before "
                        "this flag existed (uniform-random real class per sample).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--device", default="cuda")

    # DiT-Micro specifics
    p.add_argument("--num_heads", type=int, default=3)

    # NarrowDiT attention execution path (arch_cfg.json / --arch_plan students only;
    # see pace.dit_arch_alloc.NarrowAttention). Default "manual" is byte-identical
    # to every eval run before this flag existed. "sdpa" is a same-weights/same-math
    # execution-path change (fused F.scaled_dot_product_attention) -- FID evals may use
    # either (validated within noise); only throughput claims require sdpa.
    p.add_argument("--attn_impl", choices=["manual", "sdpa"], default="manual",
                   help="NarrowAttention execution path for --mode composite arch_plan "
                        "students. Default 'manual' unchanged; 'sdpa' uses the fused "
                        "kernel (same weights/math).")

    # FID
    p.add_argument("--reference_dir", default=None,
                   help="Directory of real images for FID. If omitted, FID is skipped.")
    p.add_argument("--skip_fid", action="store_true")

    return p.parse_args()


def shard_count(total: int, world_size: int, rank: int) -> int:
    """Per-rank count for a balanced shard of ``total`` samples."""
    base = total // world_size
    remainder = total - base * world_size
    return base + (1 if rank < remainder else 0)


def main():
    t0 = time.perf_counter()
    args = parse_args()

    configure_dit_repo(args.dit_repo)
    if args.class_idx_override is not None and args.model_type != "dit_xl":
        raise SystemExit("--class_idx_override is only meaningful for --model_type dit_xl "
                          "(dit_micro's classes are real CIFAR-10 categories, not an "
                          "unconditional-forcing index)")
    if args.class_idx_override is not None and args.class_idx_override < 0:
        raise SystemExit("--class_idx_override must be >= 0")

    device, rank, world_size = init_distributed(args.device)
    args.device = device
    device = torch.device(device) if isinstance(device, str) else device
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
    if is_distributed():
        dist.barrier()
    set_seed(args.seed + rank)
    dtype = choose_dtype(args.dtype)

    if is_main_process():
        print(f"world_size={world_size}, model_type={args.model_type}, mode={args.mode}, dtype={args.dtype}")

    # --- Load model(s) ---
    if args.mode == "teacher":
        if args.teacher_checkpoint is None:
            raise ValueError("--teacher_checkpoint required for --mode teacher")
        if is_main_process():
            print(f"loading teacher: {args.teacher_checkpoint}")
        if args.model_type == "dit_xl":
            model = load_dit_xl_from_state_dict_or_path(args.teacher_checkpoint, device, dtype)
        else:
            model = load_dit_micro_from_state_dict_or_path(args.teacher_checkpoint, device, args.num_heads)
        model_obj = model
        edm_pick_net = (lambda m: (lambda s: m))(model)  # teacher: same net at every sigma
    else:
        # composite: load student_phase_0/student.pt and student_phase_1/student.pt
        if args.student_dir is None:
            raise ValueError("--student_dir required for --mode composite")
        # Phase boundaries: normally from a timestep_grouping.json, but --arch_plan
        # runs stage per-phase arch_cfg.json (with their own bins) and may not pass a
        # grouping, so fall back to boundaries derived from those bins -- and, failing
        # that too, a genuinely stock/plain single-phase dir falls back further to one
        # phase spanning [0, num_timesteps) (see load_phase_boundaries_from_arch_cfgs).
        if args.grouping_json is not None:
            boundaries, num_bins = load_phase_boundaries(args.grouping_json)
        else:
            boundaries, num_bins = load_phase_boundaries_from_arch_cfgs(
                args.student_dir, args.num_timesteps)
        students: Dict[int, nn.Module] = {}
        num_phases = len(boundaries) - 1
        for pidx in range(num_phases):
            pdir = os.path.join(args.student_dir, f"phase_{pidx}")
            student = build_phase_student_from_dir(
                pdir, args.model_type, device, dtype, args.num_heads,
                attn_impl=args.attn_impl,
            )
            if is_main_process():
                kind = "NarrowDiT (arch_plan)" if os.path.exists(
                    os.path.join(pdir, "arch_cfg.json")) else "teacher-architecture"
                print(f"loading student phase {pidx}: {os.path.join(pdir, 'student.pt')} [{kind}]")
            students[pidx] = student
        composite = CompositeRoutedModel(students, boundaries, num_bins, args.num_timesteps).to(device)
        composite.eval().requires_grad_(False)
        model_obj = composite
        # EDM composite routing: pick the phase student whose bin range contains sigma.
        edm_pick_net = (lambda sd, bnd, nb: (
            lambda s: sd[_sigma_to_phase(s, bnd, nb, args.sigma_min, args.sigma_max, args.rho)]
        ))(students, boundaries, num_bins)

    # --- Build diffusion + sampler config ---
    if args.model_type == "dit_xl":
        diffusion = build_dit_xl_diffusion(args.num_steps)
        sample_fn = sample_dit_xl
        num_classes = 1000
        # VAE for latent->image decode (only DiT-XL/2)
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    else:
        diffusion = None if args.diffusion == "edm" else build_dit_micro_diffusion(args.num_steps)
        sample_fn = sample_dit_micro
        num_classes = 10
        vae = None

    # --- Shard the work ---
    local_count = shard_count(args.num_samples, world_size, rank)
    local_seed_base = args.seed * 1_000_000 + rank * 100_000

    # PNGs are written to a single shared dir; each rank writes with a unique prefix
    # (image index range) to avoid collisions.
    samples_dir = os.path.join(args.output_dir, "samples")
    if is_main_process():
        os.makedirs(samples_dir, exist_ok=True)
    if is_distributed():
        dist.barrier()

    # Per-rank starting index in the global PNG dir: sum of all earlier ranks' shards
    start_idx_global = sum(shard_count(args.num_samples, world_size, r) for r in range(rank))

    if is_main_process():
        print(f"sampling {args.num_samples} images total ({local_count} per rank)")
        pred_kind = {"dit_xl": "eps-prediction"}.get(args.model_type, "x0-prediction")
        print(f"  diffusion: {args.num_steps} {args.sampler.upper()} steps, cfg_scale={args.cfg_scale}, "
              f"{pred_kind}")
        print(f"  output PNG dir: {samples_dir}")

    # --- Sample + save ---
    generated = 0
    next_index = start_idx_global
    progress = tqdm(
        total=local_count, desc=f"rank {rank} sampling",
        disable=not is_main_process(), dynamic_ncols=True, leave=False,
    )
    while generated < local_count:
        bs = min(args.batch_size, local_count - generated)
        if args.model_type == "dit_xl":
            out = sample_dit_xl(
                model_obj=model_obj, diffusion=diffusion, batch_size=bs,
                num_classes=num_classes, device=device, dtype=dtype,
                seed=local_seed_base + generated,
                cfg_scale=args.cfg_scale, sampler=args.sampler,
                class_idx_override=args.class_idx_override,
            )
        elif args.diffusion == "edm":
            out = sample_dit_micro_edm(
                pick_net=edm_pick_net, batch_size=bs, num_classes=num_classes,
                device=device, dtype=dtype, seed=local_seed_base + generated,
                steps=args.num_steps, sigma_min=args.sigma_min, sigma_max=args.sigma_max,
                rho=args.rho, sigma_data=args.sigma_data, cfg_scale=args.cfg_scale,
            )
        else:
            out = sample_dit_micro(
                model_fn=model_obj.forward, diffusion=diffusion, batch_size=bs,
                num_classes=num_classes, device=device, dtype=dtype,
                seed=local_seed_base + generated,
            )
        if vae is not None:
            # latents -> images
            with torch.no_grad():
                imgs = vae.decode(out / 0.18215).sample
        else:
            imgs = out

        next_index = save_images_to_dir(imgs, samples_dir, next_index)
        generated += bs
        progress.update(bs)

        del out, imgs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    progress.close()

    if is_distributed():
        dist.barrier()
    if is_main_process():
        n_files = len(os.listdir(samples_dir))
        print(f"sampling done. {n_files} PNGs in {samples_dir}")

    # --- FID compute (rank 0 only) ---
    if is_main_process() and args.reference_dir and not args.skip_fid:
        print(f"\ncomputing FID against {args.reference_dir} ...")
        try:
            try:
                from torch_fidelity import calculate_metrics
            except ImportError as exc:
                raise ImportError(
                    "the quick FID needs torch-fidelity (pip install torch-fidelity); "
                    "or pass --skip_fid and use scripts/eval_composite_curve.py"
                ) from exc
            metrics = calculate_metrics(
                input1=samples_dir,
                input2=args.reference_dir,
                cuda=torch.cuda.is_available(),
                fid=True,
                isc=True,
                kid=False,  # KID is slow; skip for speed
                verbose=True,
            )
            metrics_path = os.path.join(args.output_dir, "fidelity_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"FID = {metrics.get('frechet_inception_distance', float('nan')):.4f}")
            print(f"IS  = {metrics.get('inception_score_mean', float('nan')):.4f}")
            print(f"saved metrics: {metrics_path}")
        except Exception as e:
            print(f"FID computation failed: {type(e).__name__}: {e}")

    if is_main_process():
        elapsed = time.perf_counter() - t0
        save_json(os.path.join(args.output_dir, "run_meta.json"), {
            "args": vars(args), "world_size": world_size,
            "num_samples": args.num_samples, "elapsed_sec": elapsed,
        })
        print(f"\nTotal runtime: {elapsed:.1f}s ({elapsed / 60:.1f} min). Output: {args.output_dir}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
