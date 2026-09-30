"""Reconstruction smoke test for the composite FID-curve eval on NarrowDiT
(width-allocation, --arch_plan) phase-students.

Mirrors the composite reconstruction path exercised by
``scripts/eval_composite_curve.py`` -> ``scripts/evaluate_students.py``: a phase
dir carrying ``arch_cfg.json`` (instead of ``prune_config.json``) must rebuild the
student as a NarrowDiT via ``build_narrow_dit(cfg)`` + ``load_state_dict`` from
``student.pt``. We test the importable per-phase builder
``build_phase_student_from_dir`` that ``evaluate_students.main()`` now delegates to,
so no sampling / GPU is needed.
"""
import pytest
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pace.dit_arch_alloc import (  # noqa: E402
    MICRO_TEACHER_CFG,
    build_narrow_dit,
    build_plan,
)
from evaluate_students import (  # noqa: E402
    build_phase_student_from_dir,
    load_narrow_dit_from_dir,
    load_phase_boundaries_from_arch_cfgs,
)
from eval_composite_curve import discover_steps  # noqa: E402

MICRO_GLOBAL_PARAMS = 5_498_892


def _make_global_plan(n_bins=10):
    """A minimal Micro ``global`` plan (single phase over [0, n_bins]).

    ``global`` ignores per-head importance detail, so a flat results_dict suffices;
    this reproduces what a DiT-Micro plans file written by
    ``scripts/dit_arch_to_plans.py`` carries for the ``global`` variant.
    """
    rd = {
        "n_eff": [1.0] * n_bins,
        "group_names": [f"blocks.{b}.attn.head_{h}" for b in range(8) for h in range(3)],
        "relative_delta_stack": np.ones((24, n_bins)).tolist(),
    }
    return build_plan(rd, [(0, n_bins // 2), (n_bins // 2, n_bins)], MICRO_TEACHER_CFG, "global")


def _stage_phase_dir(tmp_path):
    """Build a global NarrowDiT, save its state_dict + arch_cfg.json into phase_0."""
    plan = _make_global_plan()
    assert len(plan["phases"]) == 1  # global -> single phase
    phase = plan["phases"][0]
    cfg = phase["cfg"]
    model = build_narrow_dit(cfg)
    phase_dir = os.path.join(str(tmp_path), "phase_0")
    os.makedirs(phase_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(phase_dir, "student.pt"))
    with open(os.path.join(phase_dir, "arch_cfg.json"), "w") as f:
        json.dump(
            {
                "variant": "global",
                "arch_plan": "synthetic",
                "phase": 0,
                "bins": phase["bins"],
                "realized_params": int(phase["realized_params"]),
                "cfg": cfg,
            },
            f,
        )
    return phase_dir, phase


@pytest.mark.external_dit
def test_load_narrow_dit_from_dir_param_count_and_forward(tmp_path):
    phase_dir, _ = _stage_phase_dir(tmp_path)
    model = load_narrow_dit_from_dir(phase_dir, device="cpu")
    # count_dit_params counts only requires_grad params; the reconstructed model is
    # in eval()/requires_grad_(False), and pos_embed is a frozen (non-trainable)
    # buffer even before eval. Count trainable params by matching the trainer's
    # count_dit_params semantics: exclude pos_embed (the only always-frozen param).
    n = sum(p.numel() for name, p in model.named_parameters() if not name.endswith("pos_embed"))
    assert n == MICRO_GLOBAL_PARAMS
    x = torch.randn(2, 3, 32, 32)
    t = torch.zeros(2)
    y = torch.zeros(2, dtype=torch.long)
    out = model(x, t, y)
    assert tuple(out.shape) == (2, 3, 32, 32)


@pytest.mark.external_dit
def test_build_phase_student_from_dir_routes_to_narrow_dit(tmp_path):
    # The dispatcher evaluate_students.main() uses must pick the NarrowDiT path when
    # arch_cfg.json is present (no prune_config.json needed).
    phase_dir, _ = _stage_phase_dir(tmp_path)
    assert not os.path.exists(os.path.join(phase_dir, "prune_config.json"))
    model = build_phase_student_from_dir(
        phase_dir, model_type="dit_micro", device="cpu", dtype=torch.float32, num_heads=3,
    )
    # count_dit_params counts only requires_grad params; the reconstructed model is
    # in eval()/requires_grad_(False), and pos_embed is a frozen (non-trainable)
    # buffer even before eval. Count trainable params by matching the trainer's
    # count_dit_params semantics: exclude pos_embed (the only always-frozen param).
    n = sum(p.numel() for name, p in model.named_parameters() if not name.endswith("pos_embed"))
    assert n == MICRO_GLOBAL_PARAMS
    out = model(torch.randn(2, 3, 32, 32), torch.zeros(2), torch.zeros(2, dtype=torch.long))
    assert tuple(out.shape) == (2, 3, 32, 32)


@pytest.mark.external_dit
def test_phase_boundaries_from_arch_cfgs_global(tmp_path):
    # Composite routing falls back to per-phase arch_cfg bins when no grouping_json.
    _stage_phase_dir(tmp_path)
    boundaries, num_bins = load_phase_boundaries_from_arch_cfgs(str(tmp_path))
    assert boundaries == [0, 10]  # global -> single phase spanning [0, num_bins]
    assert num_bins == 10


# ---------------------------------------------------------------------------
# Full-ckpt consumption: EMA-first weight extraction from student.pt payloads
# ``_stage_phase_dir`` writes a BARE state_dict; these tests overwrite student.pt
# with the full-ckpt / curve dict shapes of --full_ckpt runs.
# ---------------------------------------------------------------------------

def _bare_sd(phase_dir):
    return torch.load(os.path.join(phase_dir, "student.pt"), map_location="cpu")


def _plus_one(sd):
    """A weight set distinct from ``sd`` (every float tensor shifted by +1.0)."""
    return {k: (v + 1.0 if v.is_floating_point() else v.clone()) for k, v in sd.items()}


def _assert_weights_match(model, expect_sd, differ_sd=None):
    """Every non-pos_embed float param of ``model`` equals ``expect_sd`` (and, when
    given, differs from ``differ_sd``). Returns the number of params checked."""
    loaded = model.state_dict()
    checked = 0
    for k, v in expect_sd.items():
        if k.endswith("pos_embed") or not v.is_floating_point():
            continue
        assert torch.allclose(loaded[k], v), f"{k}: loaded weights != expected"
        if differ_sd is not None:
            assert not torch.allclose(loaded[k], differ_sd[k]), f"{k}: loaded the wrong set"
        checked += 1
    assert checked > 0
    return checked


@pytest.mark.external_dit
def test_load_narrow_dit_prefers_ema(tmp_path):
    # Full-ckpt / curve dict {"model": sd_model, "ema": sd_ema}: EMA must win.
    phase_dir, _ = _stage_phase_dir(tmp_path)
    sd_model = _bare_sd(phase_dir)
    sd_ema = _plus_one(sd_model)
    torch.save({"model": sd_model, "ema": sd_ema},
               os.path.join(phase_dir, "student.pt"))
    model = load_narrow_dit_from_dir(phase_dir, device="cpu")
    _assert_weights_match(model, sd_ema, differ_sd=sd_model)


@pytest.mark.external_dit
def test_load_narrow_dit_model_only_dict(tmp_path):
    # {"model": sd} (ema absent) and {"model": sd, "ema": None} both load sd.
    phase_dir, _ = _stage_phase_dir(tmp_path)
    sd_model = _bare_sd(phase_dir)
    torch.save({"model": sd_model}, os.path.join(phase_dir, "student.pt"))
    _assert_weights_match(load_narrow_dit_from_dir(phase_dir, device="cpu"), sd_model)
    torch.save({"model": sd_model, "ema": None}, os.path.join(phase_dir, "student.pt"))
    _assert_weights_match(load_narrow_dit_from_dir(phase_dir, device="cpu"), sd_model)


@pytest.mark.external_dit
def test_load_narrow_dit_bare_state_dict_legacy(tmp_path):
    # Legacy runs save a BARE state_dict (no wrapping dict) -- still loads unchanged.
    phase_dir, _ = _stage_phase_dir(tmp_path)  # already a bare state_dict
    sd = _bare_sd(phase_dir)
    _assert_weights_match(load_narrow_dit_from_dir(phase_dir, device="cpu"), sd)


# ---------------------------------------------------------------------------
# Full-ckpt consumption: discover_steps reads curve checkpoints from the
# ``phase_<p>/curve/`` subdir, and still supports the legacy top-level layout.
# ---------------------------------------------------------------------------

def _touch(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").close()


def test_discover_steps_curve_subdir(tmp_path):
    d = str(tmp_path)
    for p in range(2):
        _touch(os.path.join(d, f"phase_{p}", "arch_cfg.json"))
        for k in (0, 2000):
            _touch(os.path.join(d, f"phase_{p}", "curve", f"step_{k}.pt"))
    assert discover_steps(d, 2) == [0, 2000]


def test_discover_steps_legacy_top_level_still_works(tmp_path):
    d = str(tmp_path)
    for p in range(2):
        for k in (0, 2000):
            _touch(os.path.join(d, f"phase_{p}", f"step_{k}.pt"))
    assert discover_steps(d, 2) == [0, 2000]
