"""Single-phase fallback in ``load_phase_boundaries_from_arch_cfgs``.

Evaluating a STOCK/plain fine-tuned DiT-XL/2 full checkpoint (no ``--arch_plan``,
so no ``arch_cfg.json``) via ``evaluate_students.py --mode composite`` (the path
``eval_composite_curve.py`` always drives) used to hard-crash with
``FileNotFoundError: no phase_*/arch_cfg.json under <dir>``, because that
function only knew how to derive phase boundaries from ``--arch_plan``-staged
``arch_cfg.json`` files, and such a run has neither those nor a
``--grouping_json``.

``load_phase_boundaries_from_arch_cfgs`` now takes an opt-in ``num_timesteps``
fallback -- when there is exactly one phase dir (``phase_0``, no ``phase_1``)
and it carries no ``arch_cfg.json``, treat it as ONE phase spanning the whole
``[0, num_timesteps)`` range (identical to what a single-phase
``timestep_grouping.json`` would describe, and numerically identical to
``--mode teacher``'s "same net at every timestep"). Omitting ``num_timesteps``
preserves the old hard-crash byte-for-byte for any other existing caller.
"""
import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from evaluate_students import (  # noqa: E402
    CompositeRoutedModel,
    build_phase_student_from_dir,
    load_phase_boundaries_from_arch_cfgs,
)


def _touch(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").close()


# ---------------------------------------------------------------------------
# Pure boundary-derivation fallback (no model construction)
# ---------------------------------------------------------------------------

def test_raises_without_num_timesteps_backward_compat(tmp_path):
    # Omitting num_timesteps (every pre-existing caller) keeps the old hard
    # crash byte-for-byte, even for the single-phase stock case.
    _touch(os.path.join(str(tmp_path), "phase_0", "student.pt"))
    with pytest.raises(FileNotFoundError, match="no phase_\\*/arch_cfg.json"):
        load_phase_boundaries_from_arch_cfgs(str(tmp_path))


def test_falls_back_to_whole_range_for_stock_single_phase(tmp_path):
    _touch(os.path.join(str(tmp_path), "phase_0", "student.pt"))
    boundaries, num_bins = load_phase_boundaries_from_arch_cfgs(str(tmp_path), num_timesteps=1000)
    assert boundaries == [0, 1000]
    assert num_bins == 1000


def test_still_raises_for_multi_phase_without_grouping_or_arch_cfg(tmp_path):
    # phase_1 also present, still no arch_cfg.json anywhere, no grouping_json:
    # genuinely ambiguous (which phase covers what?) -- must NOT silently guess.
    _touch(os.path.join(str(tmp_path), "phase_0", "student.pt"))
    _touch(os.path.join(str(tmp_path), "phase_1", "student.pt"))
    with pytest.raises(FileNotFoundError, match="no phase_\\*/arch_cfg.json"):
        load_phase_boundaries_from_arch_cfgs(str(tmp_path), num_timesteps=1000)


def test_arch_cfg_present_takes_precedence_over_fallback(tmp_path):
    # A real --arch_plan dir must still use its own bins, never the fallback,
    # regardless of whether num_timesteps is also passed.
    phase_dir = os.path.join(str(tmp_path), "phase_0")
    os.makedirs(phase_dir, exist_ok=True)
    json.dump({"bins": [0, 7], "cfg": {}}, open(os.path.join(phase_dir, "arch_cfg.json"), "w"))
    open(os.path.join(phase_dir, "student.pt"), "w").close()
    boundaries, num_bins = load_phase_boundaries_from_arch_cfgs(str(tmp_path), num_timesteps=1000)
    assert (boundaries, num_bins) == ([0, 7], 7)


# ---------------------------------------------------------------------------
# End-to-end: the reconstructed single-phase composite model is numerically
# identical to calling the (would-be teacher-mode) model directly -- confirms
# the fallback isn't just "doesn't crash" but actually routes every timestep
# to the one available student, matching --mode teacher's semantics.
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_dit_xl_factory(monkeypatch):
    """Substitute the small DiT-S/2 preset for DiT-XL/2 (675M params) so this
    test builds/loads a real (but CPU-fast) stock DiT-family model -- same
    pattern as tests/test_evaluate_students.py's tiny_dit_xl_factory."""
    from models import DiT_models  # noqa
    monkeypatch.setitem(DiT_models, "DiT-XL/2", DiT_models["DiT-S/2"])
    return lambda: DiT_models["DiT-XL/2"](input_size=32, num_classes=1000)


@pytest.mark.external_dit
def test_stock_single_phase_composite_matches_teacher_mode_forward(tmp_path, tiny_dit_xl_factory):
    torch.manual_seed(0)
    model = tiny_dit_xl_factory()
    phase_dir = os.path.join(str(tmp_path), "phase_0")
    os.makedirs(phase_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(phase_dir, "student.pt"))
    assert not os.path.exists(os.path.join(phase_dir, "arch_cfg.json"))
    assert not os.path.exists(os.path.join(phase_dir, "prune_config.json"))

    boundaries, num_bins = load_phase_boundaries_from_arch_cfgs(str(tmp_path), num_timesteps=1000)
    student = build_phase_student_from_dir(
        phase_dir, model_type="dit_xl", device="cpu", dtype=torch.float32, num_heads=3,
    )
    composite = CompositeRoutedModel({0: student}, boundaries, num_bins, num_timesteps=1000)

    x = torch.randn(2, 4, 32, 32)
    y = torch.zeros(2, dtype=torch.long)
    for t_val in (0, 1, 500, 999):
        t = torch.full((2,), t_val, dtype=torch.long)
        expected = student(x, t, y)
        actual = composite(x, t, y)
        assert torch.allclose(actual, expected), f"t={t_val}: composite diverged from the lone student"
