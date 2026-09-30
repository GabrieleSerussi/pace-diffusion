"""--teacher_arch_cfg: opt-in NarrowDiT-format teacher for train_phase_students.py.

train_phase_students.py's ``load_teacher_for_model_type`` historically only
knew how to build a teacher for ``--model_type dit_micro`` via
``load_dit_micro_network``, which hardcodes the legacy third-party
``normalcomputing/dit-cifar10-32x32-class`` shape (hidden_size=192, depth=8,
``nn.MultiheadAttention``). This repo's own trained checkpoints (teacher_v2/
v3/v4, or any width-allocation student used as a teacher) are
``pace.dit_arch_alloc.NarrowDiT`` instances instead -- e.g.
teacher_v4_smicro is D=384/depth12 with ``augment_dim=9`` (Karras' non-leaky
augmentation conditioning). Loading such a checkpoint through the legacy path
would build the WRONG architecture entirely and fail to load the state dict.

Covers:
  - ``load_teacher_for_model_type`` dispatches to ``load_narrow_dit_network``
    when ``--teacher_arch_cfg`` is set, and to the legacy
    ``load_dit_micro_network`` when it is not (regression: every pre-existing
    ``--teacher_checkpoint`` call site is unaffected).
  - End-to-end: ``train_phase()`` runs a real KD step (``--kd_weight > 0``)
    against a NarrowDiT teacher built WITH ``augment_dim > 0``, producing a
    finite, nonzero KD loss and no shape errors -- confirming
    ``model_forward_for_loss``'s teacher call (which never passes
    ``augment_labels``) is compatible with an augment-conditioned teacher.
"""
import pytest
import json
import os
import sys
import types

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, _SCRIPTS)
import train_phase_students as tps  # noqa: E402
from train_phase_students import load_teacher_for_model_type, train_phase  # noqa: E402
from pace.dit_arch_alloc import NarrowDiT, build_narrow_dit  # noqa: E402


def _args(**overrides):
    base = dict(
        model_type="dit_micro",
        diffusion="edm",
        edm_loss_space="f",
        num_epochs=1,
        lr=1e-4,
        weight_decay=0.01,
        lr_min_ratio=0.01,
        grad_clip=1.0,
        gt_weight=0.25,
        kd_weight=1.0,
        sigma_min=0.002,
        sigma_max=80.0,
        sigma_data=0.5,
        rho=7.0,
        sigma_sampling="loguniform",
        num_timesteps=1000,
        log_every=50,
        device="cpu",
        ckpt_every=0,
        arch_plan="dummy_plan.json",
        cfg_label_drop=0.0,
        use_wandb=False,
        wandb_project=None,
        wandb_run_name=None,
        num_heads=3,
        teacher_arch_cfg=None,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _tiny_teacher_cfg(augment_dim=0):
    depth = 2
    return {
        "hidden_size": 24, "depth": depth, "patch_size": 2, "in_channels": 3,
        "num_classes": 10, "input_size": 8, "learn_sigma": False,
        "per_block": [{"num_heads": 3, "attn_inner": 24, "mlp_hidden": 48} for _ in range(depth)],
        "augment_dim": augment_dim, "dropout": 0.0,
    }


def _save_teacher(tmp_path, augment_dim=9):
    cfg = _tiny_teacher_cfg(augment_dim=augment_dim)
    model = build_narrow_dit(cfg)
    ckpt_path = tmp_path / "teacher_student.pt"
    torch.save(model.state_dict(), ckpt_path)
    arch_cfg_path = tmp_path / "teacher_arch_cfg.json"
    json.dump({"cfg": cfg}, open(arch_cfg_path, "w"))
    return model, str(ckpt_path), str(arch_cfg_path)


def _build_student():
    """Fresh, differently-weighted NarrowDiT student (no augment -- students
    never use augmentation, see --augment_prob's --kd_weight 0 requirement)."""
    cfg = _tiny_teacher_cfg(augment_dim=0)
    return build_narrow_dit(cfg).train().requires_grad_(True)


def _build_loader():
    images = torch.randn(4, 3, 8, 8)
    labels = torch.randint(0, 10, (4,))
    ds = TensorDataset(images, labels)
    return DataLoader(ds, batch_size=2, shuffle=False)


# ---------------------------------------------------------------------------
# load_teacher_for_model_type dispatch
# ---------------------------------------------------------------------------

@pytest.mark.external_dit
def test_dispatches_to_narrow_dit_when_teacher_arch_cfg_set(tmp_path):
    _, ckpt_path, arch_cfg_path = _save_teacher(tmp_path, augment_dim=9)
    args = _args(teacher_checkpoint=ckpt_path, teacher_arch_cfg=arch_cfg_path)
    teacher = load_teacher_for_model_type(args, device="cpu", dtype=torch.float32)
    assert isinstance(teacher, NarrowDiT)
    assert teacher.augment_dim == 9
    assert teacher.aug_embedder is not None


def test_legacy_path_unchanged_without_teacher_arch_cfg(monkeypatch):
    """teacher_arch_cfg=None (every pre-existing call site) must still take the
    legacy load_dit_micro_network path, never load_narrow_dit_network."""
    from evaluate_parameters_dit_micro import DiTMicro
    sentinel = DiTMicro()
    calls = {"legacy": 0, "narrow": 0}

    def _fake_legacy(checkpoint_path, device, num_heads):
        calls["legacy"] += 1
        return sentinel

    def _fake_narrow(*a, **kw):
        calls["narrow"] += 1
        raise AssertionError("load_narrow_dit_network must not be called when teacher_arch_cfg is None")

    monkeypatch.setattr(tps, "load_dit_micro_network", _fake_legacy)
    monkeypatch.setattr(tps, "load_narrow_dit_network", _fake_narrow)

    args = _args(teacher_checkpoint="unused.pt", teacher_arch_cfg=None)
    teacher = load_teacher_for_model_type(args, device="cpu", dtype=torch.float32)
    assert calls == {"legacy": 1, "narrow": 0}
    assert teacher is sentinel


# ---------------------------------------------------------------------------
# End-to-end: KD training step against an augment-conditioned NarrowDiT teacher
# ---------------------------------------------------------------------------

@pytest.mark.external_dit
def test_train_phase_kd_against_augment_conditioned_narrow_dit_teacher(tmp_path):
    teacher, ckpt_path, arch_cfg_path = _save_teacher(tmp_path, augment_dim=9)
    teacher.eval().requires_grad_(False)
    student = _build_student()
    loader = _build_loader()
    alpha_bar = torch.linspace(0.99, 0.01, 1000)
    phase = {"index": 0, "start": 0, "end": 20}
    args = _args(
        teacher_checkpoint=ckpt_path, teacher_arch_cfg=arch_cfg_path,
        kd_weight=1.0, gt_weight=0.25,
    )
    phase_dir = tmp_path / "phase_0"
    phase_dir.mkdir()

    meta = train_phase(
        args=args,
        teacher=teacher,
        student_ddp=student,
        raw_student=student,
        vae=None,
        dataloader=loader,
        sampler=None,
        phase=phase,
        num_bins=20,
        alpha_bar=alpha_bar,
        in_channels=3,
        compute_dtype=torch.float32,
        phase_dir=str(phase_dir),
        use_wandb=False,
        resume_ckpt=None,
    )

    assert len(meta["loss_history"]) == 1
    assert all(torch.isfinite(torch.tensor(v)) for v in meta["loss_gt_history"])
    assert all(torch.isfinite(torch.tensor(v)) for v in meta["loss_kd_history"])
    # KD loss must actually be exercised (nonzero): teacher and student start
    # with different weights, so a zero KD loss would indicate the teacher
    # forward silently didn't run / returned the student's own prediction.
    assert any(v > 0.0 for v in meta["loss_kd_history"])
    assert os.path.exists(os.path.join(str(phase_dir), "student.pt"))
