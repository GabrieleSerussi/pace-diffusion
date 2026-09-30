"""Equivalence tests for NarrowAttention's opt-in ``attn_impl`` ('manual' vs 'sdpa') switch.

Perf fix: NarrowDiT's ``NarrowAttention`` computes attention manually (explicit
``(q @ kT) * scale -> softmax -> @ v``), while timm's ``Attention`` (``fused_attn=True``)
calls ``F.scaled_dot_product_attention`` -- measured ~41-44% faster at D=1024. This
adds an opt-in fused-SDPA execution path to ``NarrowAttention``/``NarrowDiTBlock``/
``NarrowDiT`` (default ``"manual"``, BYTE-IDENTICAL -- the hard repo rule). These CPU-only
tests confirm ``"sdpa"`` reproduces ``"manual"``'s output given IDENTICAL weights (same
weights, same math -- an execution-path-only change), and that no new parameters /
state_dict keys were introduced by the switch.
"""
import json
import os
import sys

import pytest
import torch
from pace.dit_arch_alloc import (  # noqa: E402
    MICRO_TEACHER_CFG,
    NarrowAttention,
    NarrowDiT,
    _uniform_per_block,
    build_narrow_dit,
    count_dit_params,
    pad_attention_heads,
)

# Per-block widths of a DiT-Micro layerwise_capacity phase (phase 2 of a
# four-phase plan): the nontrivial mixed-A-width bottleneck case this fix targets
# (a uniform-width cfg would not exercise cross-block attn_inner variation).
_MIXED_ATTN_INNER = [96, 96, 120, 72, 72, 48, 120, 96]
_MIXED_MLP_HIDDEN = [768, 704, 768, 512, 640, 384, 768, 704]


def _load_mixed_width_phase_cfg():
    """A DiT-Micro-shaped NarrowDiT cfg with mixed per-block widths."""
    cfg = {key: MICRO_TEACHER_CFG[key] for key in (
        "hidden_size", "depth", "patch_size", "in_channels", "num_classes", "input_size", "learn_sigma")}
    cfg["per_block"] = [
        {"num_heads": int(MICRO_TEACHER_CFG["num_heads"]), "attn_inner": a, "mlp_hidden": m}
        for a, m in zip(_MIXED_ATTN_INNER, _MIXED_MLP_HIDDEN)
    ]
    widths = [pb["attn_inner"] for pb in cfg["per_block"]]
    assert len(set(widths)) > 1, "expected a mixed-width phase for a nontrivial test"
    return cfg


# --- attn_impl plumbing basics -------------------------------------------------

def test_attn_impl_default_is_manual():
    attn = NarrowAttention(hidden_size=48, num_heads=4, attn_inner=64)
    assert attn.attn_impl == "manual"


def test_attn_impl_rejects_invalid_value():
    with pytest.raises(ValueError):
        NarrowAttention(hidden_size=48, num_heads=4, attn_inner=64, attn_impl="flash")


@pytest.mark.external_dit
def test_build_narrow_dit_attn_impl_default_none_leaves_cfg_untouched():
    """build_narrow_dit's own ``attn_impl=None`` default must not mutate the caller's
    cfg dict (no 'attn_impl' key added) and must build a manual-path model -- the
    pre-existing default behavior, unchanged by this parameter's addition."""
    cfg = dict(MICRO_TEACHER_CFG)
    cfg["per_block"] = _uniform_per_block(
        cfg["hidden_size"], cfg["depth"], cfg["num_heads"], cfg["mlp_ratio"])
    cfg_before = dict(cfg)
    model = build_narrow_dit(cfg)
    assert cfg == cfg_before
    assert model.attn_impl == "manual"
    assert all(block.attn.attn_impl == "manual" for block in model.blocks)


@pytest.mark.external_dit
def test_build_narrow_dit_attn_impl_override_sdpa():
    cfg = dict(MICRO_TEACHER_CFG)
    cfg["per_block"] = _uniform_per_block(
        cfg["hidden_size"], cfg["depth"], cfg["num_heads"], cfg["mlp_ratio"])
    model = build_narrow_dit(cfg, attn_impl="sdpa")
    assert model.attn_impl == "sdpa"
    assert all(block.attn.attn_impl == "sdpa" for block in model.blocks)


# --- Test (a): random-init NarrowAttention manual vs sdpa, eval mode, fp32 -----

def test_narrow_attention_manual_vs_sdpa_random_init_allclose():
    torch.manual_seed(0)
    manual = NarrowAttention(hidden_size=48, num_heads=4, attn_inner=64, attn_impl="manual").eval()
    sdpa = NarrowAttention(hidden_size=48, num_heads=4, attn_inner=64, attn_impl="sdpa").eval()
    sdpa.load_state_dict(manual.state_dict())  # identical weights; only impl differs

    x = torch.randn(3, 17, 48, dtype=torch.float32)  # (B, N, D); nonsquare N catches shape bugs
    with torch.no_grad():
        out_manual = manual(x)
        out_sdpa = sdpa(x)
    assert out_manual.shape == out_sdpa.shape
    torch.testing.assert_close(out_manual, out_sdpa, atol=1e-5, rtol=1e-5)


def test_narrow_attention_manual_vs_sdpa_multiple_shapes():
    """A few more (hidden_size, num_heads, attn_inner) combos, including attn_inner
    != hidden_size (the actual NarrowDiT bottleneck case motivating this fix) and
    attn_inner > hidden_size."""
    torch.manual_seed(1)
    configs = [
        dict(hidden_size=32, num_heads=2, attn_inner=32),    # A == D (uniform)
        dict(hidden_size=96, num_heads=3, attn_inner=48),    # A < D (bottleneck)
        dict(hidden_size=96, num_heads=6, attn_inner=120),   # A > D
    ]
    for cfg in configs:
        manual = NarrowAttention(**cfg, attn_impl="manual").eval()
        sdpa = NarrowAttention(**cfg, attn_impl="sdpa").eval()
        sdpa.load_state_dict(manual.state_dict())
        x = torch.randn(2, 9, cfg["hidden_size"], dtype=torch.float32)
        with torch.no_grad():
            out_manual = manual(x)
            out_sdpa = sdpa(x)
        torch.testing.assert_close(out_manual, out_sdpa, atol=1e-5, rtol=1e-5, msg=lambda m, c=cfg: f"{c}: {m}")


# --- Test (b): full NarrowDiT forward, real mixed-width per-block cfg ---------

@pytest.mark.external_dit
def test_narrow_dit_manual_vs_sdpa_real_mixed_width_plan():
    cfg = _load_mixed_width_phase_cfg()

    torch.manual_seed(2)
    manual = build_narrow_dit(cfg, attn_impl="manual").eval()
    sdpa = build_narrow_dit(cfg, attn_impl="sdpa").eval()
    sdpa.load_state_dict(manual.state_dict())

    B = 2
    x = torch.randn(B, cfg["in_channels"], cfg["input_size"], cfg["input_size"], dtype=torch.float32)
    t = torch.rand(B, dtype=torch.float32)
    y = torch.randint(0, cfg["num_classes"], (B,))
    with torch.no_grad():
        out_manual = manual(x, t, y)
        out_sdpa = sdpa(x, t, y)
    assert out_manual.shape == out_sdpa.shape
    torch.testing.assert_close(out_manual, out_sdpa, atol=1e-5, rtol=1e-5)


@pytest.mark.external_dit
def test_narrow_dit_manual_vs_sdpa_forward_with_cfg_real_mixed_width_plan():
    """Also check the CFG sampling path (forward_with_cfg), used by DDIM/DDPM
    composite sampling with cfg_scale > 1 -- exercises the same NarrowAttention
    code twice per call (conditional + null-class halves)."""
    cfg = _load_mixed_width_phase_cfg()

    torch.manual_seed(3)
    manual = build_narrow_dit(cfg, attn_impl="manual").eval()
    sdpa = build_narrow_dit(cfg, attn_impl="sdpa").eval()
    sdpa.load_state_dict(manual.state_dict())

    B = 4  # must be even (forward_with_cfg splits the batch in half)
    x = torch.randn(B, cfg["in_channels"], cfg["input_size"], cfg["input_size"], dtype=torch.float32)
    t = torch.rand(B, dtype=torch.float32)
    y = torch.randint(0, cfg["num_classes"], (B,))
    with torch.no_grad():
        out_manual = manual.forward_with_cfg(x, t, y, cfg_scale=2.0)
        out_sdpa = sdpa.forward_with_cfg(x, t, y, cfg_scale=2.0)
    torch.testing.assert_close(out_manual, out_sdpa, atol=1e-5, rtol=1e-5)


# --- Test (c): state_dict keys identical between impls (no new params) --------

def test_narrow_attention_state_dict_keys_identical():
    manual = NarrowAttention(hidden_size=48, num_heads=4, attn_inner=64, attn_impl="manual")
    sdpa = NarrowAttention(hidden_size=48, num_heads=4, attn_inner=64, attn_impl="sdpa")
    assert set(manual.state_dict().keys()) == set(sdpa.state_dict().keys())
    n_manual = sum(p.numel() for p in manual.parameters())
    n_sdpa = sum(p.numel() for p in sdpa.parameters())
    assert n_manual == n_sdpa


@pytest.mark.external_dit
def test_narrow_dit_state_dict_keys_identical_real_plan():
    cfg = _load_mixed_width_phase_cfg()
    manual = build_narrow_dit(cfg, attn_impl="manual")
    sdpa = build_narrow_dit(cfg, attn_impl="sdpa")
    assert set(manual.state_dict().keys()) == set(sdpa.state_dict().keys())
    assert count_dit_params(manual) == count_dit_params(sdpa)


@pytest.mark.external_dit
def test_narrow_dit_state_dict_keys_identical_uniform_teacher_cfg():
    """Also check the uniform (A==D) teacher-shaped case, not just the mixed-width
    student case above."""
    cfg = dict(MICRO_TEACHER_CFG)
    cfg["per_block"] = _uniform_per_block(
        cfg["hidden_size"], cfg["depth"], cfg["num_heads"], cfg["mlp_ratio"])
    manual = build_narrow_dit(cfg, attn_impl="manual")
    sdpa = build_narrow_dit(cfg, attn_impl="sdpa")
    assert set(manual.state_dict().keys()) == set(sdpa.state_dict().keys())
    assert count_dit_params(manual) == count_dit_params(sdpa)


# --- pad_attention_heads: inference-only, exact head_dim padding ---------------

def _rand_attention(hidden_size, num_heads, attn_inner, attn_impl):
    torch.manual_seed(0)
    attn = NarrowAttention(hidden_size=hidden_size, num_heads=num_heads,
                           attn_inner=attn_inner, attn_impl=attn_impl).eval()
    with torch.no_grad():  # non-trivial biases: a wrong padded slot would show up
        for p in attn.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    return attn


@pytest.mark.parametrize("attn_impl", ["manual", "sdpa"])
def test_pad_attention_heads_is_exact_and_keeps_original_scale(attn_impl):
    # head_dim 34 (= 68 / 2) is not a multiple of 8 -> padded up to 40.
    attn = _rand_attention(hidden_size=48, num_heads=2, attn_inner=68, attn_impl=attn_impl)
    x = torch.randn(3, 5, 48)
    with torch.no_grad():
        ref = attn(x)
    assert pad_attention_heads(attn, multiple=8) == 1
    assert (attn.head_dim, attn.attn_inner, attn.num_heads) == (40, 80, 2)
    assert attn.scale == 34 ** -0.5  # softmax scale stays the ORIGINAL head_dim ** -0.5
    assert attn.qkv.out_features == 3 * 80 and attn.proj.in_features == 80
    with torch.no_grad():
        out = attn(x)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_pad_attention_heads_skips_modules_already_aligned():
    attn = _rand_attention(hidden_size=48, num_heads=4, attn_inner=64, attn_impl="manual")
    before = {k: v.clone() for k, v in attn.state_dict().items()}
    assert pad_attention_heads(attn, multiple=8) == 0
    assert attn.head_dim == 16
    for k, v in attn.state_dict().items():
        assert torch.equal(v, before[k])
