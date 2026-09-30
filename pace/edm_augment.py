"""Non-leaky augmentation pipeline (Karras EDM), ported for the DiT teachers.

Karras et al. ("Elucidating the Design Space of Diffusion-Based Generative
Models", EDM) show that at CIFAR-10 scale two regularizers are load-bearing:
dropout inside the network and a **non-leaky augmentation pipeline**. The
pipeline is "non-leaky" because the per-sample augmentation *parameters* are fed
back to the network as a conditioning vector (``augment_labels``), so the model
can tell an augmented sample from a real one and the augmentation distribution
provably does not leak into the learned data distribution.

This module is a faithful port of NVlabs/edm ``training/augment.py`` (fetched
from the upstream ``main`` branch), with two deliberate deviations, both of
which are no-ops for EDM's own CIFAR-10 configuration:

  * ``torch_utils.persistence`` / ``torch_utils.misc.constant`` are replaced by
    a plain local ``_constant`` helper (upstream's version only adds a pickling
    decorator and a constant cache -- no semantic difference).
  * the **colour** augmentation group (brightness / contrast / lumaflip / hue /
    saturation) is omitted: EDM's CIFAR-10 recipe enables none of them
    (``train.py``: ``xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1,
    translate_frac=1``), so porting them would be dead code. The pixel-blitting
    and geometric groups -- x-flip, y-flip, integer (90 degree) rotation,
    integer translation, isotropic scaling, fractional rotation, anisotropic
    scaling, fractional translation -- are ported verbatim.

Everything is disabled by default (all probability multipliers 0), so an
``AugmentPipe()`` with no arguments is the identity and returns a width-0 label
vector. Use :func:`cifar10_augment_pipe` for EDM's CIFAR-10 configuration.

The ``augment_labels`` width for a given configuration is
:attr:`AugmentPipe.label_dim`; for the CIFAR-10 configuration it is
:data:`CIFAR10_AUGMENT_DIM` == 9, matching EDM's ``network_kwargs.augment_dim = 9``.

Licence: this file is adapted from NVlabs/edm (Copyright (c) 2022, NVIDIA
CORPORATION & AFFILIATES), which is licensed under CC BY-NC-SA 4.0; that
licence applies to this file (see ``THIRD_PARTY_NOTICES.md``).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers for constructing transformation matrices (upstream augment.py).
# ---------------------------------------------------------------------------

def _constant(value, shape=None, dtype=None, device=None) -> torch.Tensor:
    """``torch_utils.misc.constant`` without the memo cache."""
    value = np.asarray(value)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device("cpu")
    tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
    if shape is not None:
        tensor, _ = torch.broadcast_tensors(
            tensor, torch.empty(tuple(shape), device=tensor.device)
        )
    return tensor.contiguous()


def matrix(*rows, device=None):
    assert all(len(row) == len(rows[0]) for row in rows)
    elems = [x for row in rows for x in row]
    ref = [x for x in elems if isinstance(x, torch.Tensor)]
    if len(ref) == 0:
        return _constant(np.asarray(rows), device=device)
    assert device is None or device == ref[0].device
    elems = [
        x if isinstance(x, torch.Tensor)
        else _constant(x, shape=ref[0].shape, device=ref[0].device)
        for x in elems
    ]
    return torch.stack(elems, dim=-1).reshape(ref[0].shape + (len(rows), -1))


def translate2d(tx, ty, **kwargs):
    return matrix(
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1],
        **kwargs)


def scale2d(sx, sy, **kwargs):
    return matrix(
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1],
        **kwargs)


def rotate2d(theta, **kwargs):
    return matrix(
        [torch.cos(theta), torch.sin(-theta), 0],
        [torch.sin(theta), torch.cos(theta),  0],
        [0,                0,                 1],
        **kwargs)


def translate2d_inv(tx, ty, **kwargs):
    return translate2d(-tx, -ty, **kwargs)


def scale2d_inv(sx, sy, **kwargs):
    return scale2d(1 / sx, 1 / sy, **kwargs)


def rotate2d_inv(theta, **kwargs):
    return rotate2d(-theta, **kwargs)


# Coefficients of the sym6 wavelet decomposition low-pass filter -- the only
# entry of upstream's ``wavelets`` dict that ``AugmentPipe`` actually uses (as
# the up/downsampling filter of the geometric branch).
WAVELET_SYM6 = [
    0.015404109327027373, 0.0034907120842174702, -0.11799011114819057,
    -0.048311742585633, 0.4910559419267466, 0.787641141030194,
    0.3379294217276218, -0.07263752278646252, -0.021060292512300564,
    0.04472490177066578, 0.0017677118642428036, -0.007800708325034148,
]


# ---------------------------------------------------------------------------
# Augmentation pipeline.
# ---------------------------------------------------------------------------

class AugmentPipe:
    """Non-leaky augmentation pipe: ``images -> (augmented_images, augment_labels)``.

    All augmentations are disabled by default; enable one by setting its
    probability multiplier to a positive value. The *effective* per-sample
    probability of an augmentation is ``min(multiplier * p, 1)``; upstream's
    CIFAR-10 config exploits that by passing ``xflip=1e8`` so x-flip is applied
    with probability 1 (i.e. always a fresh coin flip) while the remaining five
    augmentations fire with probability ``p``.

    Input images are expected in EDM's convention: float NCHW in [-1, 1]
    (3-channel RGB or 1-channel L). The returned ``augment_labels`` is
    ``(N, label_dim)`` float32 and encodes the parameters actually drawn for
    each sample -- feed it to the network's augment-conditioning embedding.
    """

    def __init__(
        self,
        p: float = 1,
        # Pixel blitting.
        xflip: float = 0,
        yflip: float = 0,
        rotate_int: float = 0,
        translate_int: float = 0,
        translate_int_max: float = 0.125,
        # Geometric transformations.
        scale: float = 0,
        rotate_frac: float = 0,
        aniso: float = 0,
        translate_frac: float = 0,
        scale_std: float = 0.2,
        rotate_frac_max: float = 1,
        aniso_std: float = 0.2,
        aniso_rotate_prob: float = 0.5,
        translate_frac_std: float = 0.125,
    ):
        self.p                  = float(p)                  # Overall multiplier for augmentation probability.

        # Pixel blitting.
        self.xflip              = float(xflip)              # Probability multiplier for x-flip.
        self.yflip              = float(yflip)              # Probability multiplier for y-flip.
        self.rotate_int         = float(rotate_int)         # Probability multiplier for integer (90 deg) rotation.
        self.translate_int      = float(translate_int)      # Probability multiplier for integer translation.
        self.translate_int_max  = float(translate_int_max)  # Range of integer translation, relative to image dimensions.

        # Geometric transformations.
        self.scale              = float(scale)              # Probability multiplier for isotropic scaling.
        self.rotate_frac        = float(rotate_frac)        # Probability multiplier for fractional rotation.
        self.aniso              = float(aniso)              # Probability multiplier for anisotropic scaling.
        self.translate_frac     = float(translate_frac)     # Probability multiplier for fractional translation.
        self.scale_std          = float(scale_std)          # Log2 standard deviation of isotropic scaling.
        self.rotate_frac_max    = float(rotate_frac_max)    # Range of fractional rotation, 1 = full circle.
        self.aniso_std          = float(aniso_std)          # Log2 standard deviation of anisotropic scaling.
        self.aniso_rotate_prob  = float(aniso_rotate_prob)  # Probability of anisotropic scaling w.r.t. a rotated frame.
        self.translate_frac_std = float(translate_frac_std) # Std of fractional translation, relative to image dimensions.

    # -- introspection ------------------------------------------------------

    @property
    def label_dim(self) -> int:
        """Width of the ``augment_labels`` vector this configuration emits.

        This is what the network's ``augment_dim`` must be set to. Derived from
        the enabled multipliers (NOT hardcoded), so it always matches the
        vector actually returned by ``__call__``.
        """
        dim = 0
        dim += 1 if self.xflip > 0 else 0
        dim += 1 if self.yflip > 0 else 0
        dim += 2 if self.rotate_int > 0 else 0
        dim += 2 if self.translate_int > 0 else 0
        dim += 1 if self.scale > 0 else 0
        dim += 2 if self.rotate_frac > 0 else 0
        dim += 2 if self.aniso > 0 else 0
        dim += 2 if self.translate_frac > 0 else 0
        return dim

    # -- the pipe -----------------------------------------------------------

    def __call__(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        N, C, H, W = images.shape
        device = images.device
        labels = [torch.zeros([N, 0], device=device)]

        # ---------------
        # Pixel blitting.
        # ---------------

        if self.xflip > 0:
            w = torch.randint(2, [N, 1, 1, 1], device=device)
            w = torch.where(torch.rand([N, 1, 1, 1], device=device) < self.xflip * self.p, w, torch.zeros_like(w))
            images = torch.where(w == 1, images.flip(3), images)
            labels += [w]

        if self.yflip > 0:
            w = torch.randint(2, [N, 1, 1, 1], device=device)
            w = torch.where(torch.rand([N, 1, 1, 1], device=device) < self.yflip * self.p, w, torch.zeros_like(w))
            images = torch.where(w == 1, images.flip(2), images)
            labels += [w]

        if self.rotate_int > 0:
            w = torch.randint(4, [N, 1, 1, 1], device=device)
            w = torch.where(torch.rand([N, 1, 1, 1], device=device) < self.rotate_int * self.p, w, torch.zeros_like(w))
            images = torch.where((w == 1) | (w == 2), images.flip(3), images)
            images = torch.where((w == 2) | (w == 3), images.flip(2), images)
            images = torch.where((w == 1) | (w == 3), images.transpose(2, 3), images)
            labels += [(w == 1) | (w == 2), (w == 2) | (w == 3)]

        if self.translate_int > 0:
            w = torch.rand([2, N, 1, 1, 1], device=device) * 2 - 1
            w = torch.where(torch.rand([1, N, 1, 1, 1], device=device) < self.translate_int * self.p, w, torch.zeros_like(w))
            tx = w[0].mul(W * self.translate_int_max).round().to(torch.int64)
            ty = w[1].mul(H * self.translate_int_max).round().to(torch.int64)
            b, c, y, x = torch.meshgrid(*(torch.arange(x, device=device) for x in images.shape), indexing='ij')
            x = W - 1 - (W - 1 - (x - tx) % (W * 2 - 2)).abs()
            y = H - 1 - (H - 1 - (y + ty) % (H * 2 - 2)).abs()
            images = images.flatten()[(((b * C) + c) * H + y) * W + x]
            labels += [tx.div(W * self.translate_int_max), ty.div(H * self.translate_int_max)]

        # ------------------------------------------------
        # Select parameters for geometric transformations.
        # ------------------------------------------------

        I_3 = torch.eye(3, device=device)
        G_inv = I_3

        if self.scale > 0:
            w = torch.randn([N], device=device)
            w = torch.where(torch.rand([N], device=device) < self.scale * self.p, w, torch.zeros_like(w))
            s = w.mul(self.scale_std).exp2()
            G_inv = G_inv @ scale2d_inv(s, s)
            labels += [w]

        if self.rotate_frac > 0:
            w = (torch.rand([N], device=device) * 2 - 1) * (np.pi * self.rotate_frac_max)
            w = torch.where(torch.rand([N], device=device) < self.rotate_frac * self.p, w, torch.zeros_like(w))
            G_inv = G_inv @ rotate2d_inv(-w)
            labels += [w.cos() - 1, w.sin()]

        if self.aniso > 0:
            w = torch.randn([N], device=device)
            r = (torch.rand([N], device=device) * 2 - 1) * np.pi
            w = torch.where(torch.rand([N], device=device) < self.aniso * self.p, w, torch.zeros_like(w))
            r = torch.where(torch.rand([N], device=device) < self.aniso_rotate_prob, r, torch.zeros_like(r))
            s = w.mul(self.aniso_std).exp2()
            G_inv = G_inv @ rotate2d_inv(r) @ scale2d_inv(s, 1 / s) @ rotate2d_inv(-r)
            labels += [w * r.cos(), w * r.sin()]

        if self.translate_frac > 0:
            w = torch.randn([2, N], device=device)
            w = torch.where(torch.rand([1, N], device=device) < self.translate_frac * self.p, w, torch.zeros_like(w))
            G_inv = G_inv @ translate2d_inv(w[0].mul(W * self.translate_frac_std), w[1].mul(H * self.translate_frac_std))
            labels += [w[0], w[1]]

        # ----------------------------------
        # Execute geometric transformations.
        # ----------------------------------

        if G_inv is not I_3:
            cx = (W - 1) / 2
            cy = (H - 1) / 2
            cp = matrix([-cx, -cy, 1], [cx, -cy, 1], [cx, cy, 1], [-cx, cy, 1], device=device)  # [idx, xyz]
            cp = G_inv @ cp.t()  # [batch, xyz, idx]
            Hz = np.asarray(WAVELET_SYM6, dtype=np.float32)
            Hz_pad = len(Hz) // 4
            margin = cp[:, :2, :].permute(1, 0, 2).flatten(1)  # [xy, batch * idx]
            margin = torch.cat([-margin, margin]).max(dim=1).values  # [x0, y0, x1, y1]
            margin = margin + _constant([Hz_pad * 2 - cx, Hz_pad * 2 - cy] * 2, device=device)
            margin = margin.max(_constant([0, 0] * 2, device=device))
            margin = margin.min(_constant([W - 1, H - 1] * 2, device=device))
            mx0, my0, mx1, my1 = margin.ceil().to(torch.int32)

            # Pad image and adjust origin.
            images = torch.nn.functional.pad(input=images, pad=[mx0, mx1, my0, my1], mode='reflect')
            G_inv = translate2d((mx0 - mx1) / 2, (my0 - my1) / 2) @ G_inv

            # Upsample.
            conv_weight = _constant(Hz[None, None, ::-1], dtype=images.dtype, device=images.device).tile([images.shape[1], 1, 1])
            conv_pad = (len(Hz) + 1) // 2
            images = torch.stack([images, torch.zeros_like(images)], dim=4).reshape(N, C, images.shape[2], -1)[:, :, :, :-1]
            images = torch.nn.functional.conv2d(images, conv_weight.unsqueeze(2), groups=images.shape[1], padding=[0, conv_pad])
            images = torch.stack([images, torch.zeros_like(images)], dim=3).reshape(N, C, -1, images.shape[3])[:, :, :-1, :]
            images = torch.nn.functional.conv2d(images, conv_weight.unsqueeze(3), groups=images.shape[1], padding=[conv_pad, 0])
            G_inv = scale2d(2, 2, device=device) @ G_inv @ scale2d_inv(2, 2, device=device)
            G_inv = translate2d(-0.5, -0.5, device=device) @ G_inv @ translate2d_inv(-0.5, -0.5, device=device)

            # Execute transformation.
            shape = [N, C, (H + Hz_pad * 2) * 2, (W + Hz_pad * 2) * 2]
            G_inv = scale2d(2 / images.shape[3], 2 / images.shape[2], device=device) @ G_inv @ scale2d_inv(2 / shape[3], 2 / shape[2], device=device)
            grid = torch.nn.functional.affine_grid(theta=G_inv[:, :2, :], size=shape, align_corners=False)
            images = torch.nn.functional.grid_sample(images, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

            # Downsample and crop.
            conv_weight = _constant(Hz[None, None, :], dtype=images.dtype, device=images.device).tile([images.shape[1], 1, 1])
            conv_pad = (len(Hz) - 1) // 2
            images = torch.nn.functional.conv2d(images, conv_weight.unsqueeze(2), groups=images.shape[1], stride=[1, 2], padding=[0, conv_pad])[:, :, :, Hz_pad: -Hz_pad]
            images = torch.nn.functional.conv2d(images, conv_weight.unsqueeze(3), groups=images.shape[1], stride=[2, 1], padding=[conv_pad, 0])[:, :, Hz_pad: -Hz_pad, :]

        labels = torch.cat([x.to(torch.float32).reshape(N, -1) for x in labels], dim=1)
        return images, labels


# ---------------------------------------------------------------------------
# EDM's CIFAR-10 configuration.
# ---------------------------------------------------------------------------

# Upstream NVlabs/edm train.py:
#     c.augment_kwargs = EasyDict(class_name='training.augment.AugmentPipe', p=opts.augment)
#     c.augment_kwargs.update(xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
#     c.network_kwargs.augment_dim = 9
# with ``--augment`` defaulting to 0.12. ``xflip=1e8`` makes xflip*p >= 1, i.e.
# x-flip is ALWAYS a fresh 50/50 coin flip (it subsumes a plain --hflip data
# augmentation); the other five fire with probability p each.
CIFAR10_AUGMENT_KWARGS = dict(
    xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1,
)
# 1 (xflip) + 1 (yflip) + 1 (scale) + 2 (rotate_frac) + 2 (aniso) + 2 (translate_frac)
CIFAR10_AUGMENT_DIM = 9


def cifar10_augment_pipe(p: float) -> AugmentPipe:
    """EDM's CIFAR-10 augmentation pipe at overall probability ``p`` (paper: 0.12)."""
    return AugmentPipe(p=float(p), **CIFAR10_AUGMENT_KWARGS)


def build_augment_pipe(p: Optional[float]) -> Optional[AugmentPipe]:
    """``None`` when ``p`` is falsy/<=0 -- so the caller can skip the pipe
    entirely and consume ZERO RNG, keeping augment-off runs byte-identical to
    before this module existed. Otherwise EDM's CIFAR-10 pipe at probability p."""
    if not p or float(p) <= 0.0:
        return None
    return cifar10_augment_pipe(float(p))
