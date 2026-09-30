"""Graduation-and-launch QUALITY GATE glue for evaluate_students.py: measuring FID
for the structured teachers (2 stock-DiT-XL/2 fine-tunes, 2 from-scratch NarrowDiT
DiT-B pretrains -- both unconditional, trained via --force_label/--unconditional)
found two real gaps, both opt-in / default-off / byte-identical-when-unused:

1. ``sample_dit_xl`` always drew ``y = torch.randint(0, num_classes, ...)`` (a
   uniform-random REAL class) -- for a checkpoint trained exclusively through one
   fixed forced label (1000 = the DiT-XL/2 fine-tunes' trained null/CFG row, 0 =
   the DiT-B pretrains' single unconditional class), that silently samples classes
   the model was never trained to condition on. Added ``class_idx_override``
   (forwarded from a new opt-in CLI flag ``--class_idx_override``, mirroring
   ``evaluate_parameters_dit.py``'s own flag of the same name/convention): when
   set, every sample uses that one fixed label instead.
2. ``load_dit_xl_from_state_dict_or_path`` (the ``--mode teacher`` / composite
   stock-DiT-XL/2 loader) only ever read ``payload["model"]`` from a ``--full_ckpt``
   payload, silently scoring raw (non-EMA) weights instead of preferring EMA like
   every other loader in this file (``load_narrow_dit_from_dir``,
   ``load_dit_micro_from_state_dict_or_path``, via ``_student_state_from_payload``).
   Rewired to use that same existing helper.

Covers: class_idx_override forces a fixed label (vs. the untouched default-random
byte-identical-to-before behavior), the CLI default/guards, and EMA-preferring
extraction (present/None/absent) in load_dit_xl_from_state_dict_or_path.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import evaluate_students as script  # noqa: E402
from evaluate_students import (  # noqa: E402
    load_dit_xl_from_state_dict_or_path,
    sample_dit_xl,
)

_BASE_ARGV = ["evaluate_students.py", "--model_type", "dit_xl", "--mode", "teacher",
              "--output_dir", "/tmp/_unused_evaluate_students_test"]


class _StubDiffusion:
    """Records the y passed via model_kwargs; returns x_T unchanged (never calls
    model_fn) -- sample_dit_xl's y-construction is the only thing under test."""

    def __init__(self, captured):
        self.captured = captured

    def ddim_sample_loop(self, model_fn, shape, x_T, clip_denoised, model_kwargs, progress, device):
        self.captured.append(model_kwargs["y"].clone())
        return x_T

    def p_sample_loop(self, model_fn, shape, x_T, clip_denoised, model_kwargs, progress, device):
        self.captured.append(model_kwargs["y"].clone())
        return x_T


class _DummyModel:
    def forward(self, *a, **kw):
        raise AssertionError("stub diffusion should never call the model")

    def forward_with_cfg(self, *a, **kw):
        raise AssertionError("stub diffusion should never call the model")


def test_sample_dit_xl_class_idx_override_forces_fixed_label():
    captured = []
    sample_dit_xl(
        model_obj=_DummyModel(), diffusion=_StubDiffusion(captured), batch_size=8,
        num_classes=1000, device=torch.device("cpu"), dtype=torch.float32, seed=0,
        cfg_scale=1.0, sampler="ddim", class_idx_override=1000,
    )
    y = captured[0]
    assert y.shape == (8,)
    assert torch.equal(y, torch.full((8,), 1000, dtype=torch.long))


def test_sample_dit_xl_class_idx_override_zero_for_unconditional_narrow_dit():
    captured = []
    sample_dit_xl(
        model_obj=_DummyModel(), diffusion=_StubDiffusion(captured), batch_size=4,
        num_classes=1, device=torch.device("cpu"), dtype=torch.float32, seed=0,
        cfg_scale=1.0, sampler="ddpm", class_idx_override=0,
    )
    y = captured[0]
    assert torch.equal(y, torch.zeros(4, dtype=torch.long))


def test_sample_dit_xl_default_none_matches_pre_existing_random_behavior():
    """class_idx_override=None must be byte-identical to the code path that
    existed before this parameter did (uniform-random real class, drawn from the
    SAME generator right after z -- order matters for reproducibility)."""
    captured = []
    sample_dit_xl(
        model_obj=_DummyModel(), diffusion=_StubDiffusion(captured), batch_size=8,
        num_classes=1000, device=torch.device("cpu"), dtype=torch.float32, seed=0,
        cfg_scale=1.0, sampler="ddim", class_idx_override=None,
    )
    y = captured[0]

    gen = torch.Generator(device="cpu").manual_seed(0)
    _z_expected = torch.randn(8, 4, 32, 32, device="cpu", dtype=torch.float32, generator=gen)
    y_expected = torch.randint(0, 1000, (8,), device="cpu", generator=gen)
    assert torch.equal(y, y_expected)


def test_cli_class_idx_override_defaults_to_none(monkeypatch):
    monkeypatch.setattr(sys, "argv", list(_BASE_ARGV))
    args = script.parse_args()
    assert args.class_idx_override is None


def test_cli_class_idx_override_parses_int(monkeypatch):
    monkeypatch.setattr(sys, "argv", list(_BASE_ARGV) + ["--class_idx_override", "1000"])
    args = script.parse_args()
    assert args.class_idx_override == 1000


def test_main_rejects_class_idx_override_for_non_dit_xl(monkeypatch):
    argv = ["evaluate_students.py", "--model_type", "dit_micro", "--mode", "teacher",
            "--output_dir", "/tmp/_unused", "--class_idx_override", "0"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="only meaningful for --model_type dit_xl"):
        script.main()


def test_main_rejects_negative_class_idx_override(monkeypatch):
    argv = list(_BASE_ARGV) + ["--class_idx_override", "-1"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="must be >= 0"):
        script.main()


# ---------------------------------------------------------------------------
# load_dit_xl_from_state_dict_or_path: EMA-preferring extraction
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_dit_xl_factory(monkeypatch):
    """load_dit_xl_from_state_dict_or_path hardcodes DiT_models["DiT-XL/2"]
    (675M params) -- substitute the small DiT-S/2 preset for this test only
    (same forward/state_dict-key surface, this repo's own established pattern
    for CPU-testing the DiT-family constructor, see tests/test_tp_head_pruning.py),
    restored automatically by monkeypatch after the test."""
    from models import DiT_models  # noqa
    monkeypatch.setitem(DiT_models, "DiT-XL/2", DiT_models["DiT-S/2"])
    return lambda: DiT_models["DiT-XL/2"](input_size=32, num_classes=1000)


@pytest.mark.external_dit
def test_load_dit_xl_prefers_ema_when_present(tmp_path, tiny_dit_xl_factory):
    torch.manual_seed(0)
    model_sd = tiny_dit_xl_factory().state_dict()
    ema_sd = {k: v + 1.0 for k, v in model_sd.items()}
    path = os.path.join(str(tmp_path), "student.pt")
    torch.save({"model": model_sd, "ema": ema_sd}, path)

    loaded = load_dit_xl_from_state_dict_or_path(path, device="cpu", dtype=torch.float32)
    for k, v in loaded.state_dict().items():
        assert torch.allclose(v, ema_sd[k]), f"{k} did not load EMA weights"


@pytest.mark.external_dit
def test_load_dit_xl_falls_back_to_model_when_ema_is_none(tmp_path, tiny_dit_xl_factory):
    torch.manual_seed(0)
    model_sd = tiny_dit_xl_factory().state_dict()
    path = os.path.join(str(tmp_path), "student.pt")
    torch.save({"model": model_sd, "ema": None}, path)

    loaded = load_dit_xl_from_state_dict_or_path(path, device="cpu", dtype=torch.float32)
    for k, v in loaded.state_dict().items():
        assert torch.allclose(v, model_sd[k]), f"{k} did not fall back to model weights"


@pytest.mark.external_dit
def test_load_dit_xl_accepts_bare_state_dict_legacy(tmp_path, tiny_dit_xl_factory):
    """No {"model", "ema"} wrapping (e.g. the officially-released DiT-XL/2
    weights) -- must load unchanged, exactly like before this fix."""
    torch.manual_seed(0)
    model_sd = tiny_dit_xl_factory().state_dict()
    path = os.path.join(str(tmp_path), "bare.pt")
    torch.save(model_sd, path)

    loaded = load_dit_xl_from_state_dict_or_path(path, device="cpu", dtype=torch.float32)
    for k, v in loaded.state_dict().items():
        assert torch.allclose(v, model_sd[k])
