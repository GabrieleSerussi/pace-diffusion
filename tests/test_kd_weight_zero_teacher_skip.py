"""
--kd_weight 0 teacher skip: when --arch_plan is set (from-scratch NarrowDiT
students) and --kd_weight <= 0, the teacher should never be loaded or forward-
passed -- data-loss-only (--gt_weight only) training. The KD forward was
already gated on kd_weight>0 everywhere in train_phase/validate; this only
changes whether main() bothers to load the teacher checkpoint at all.

Covers:
  * should_skip_teacher(args): the decision helper factored out of main().
  * get_in_channels(args, None): the teacher=None fallback returns the same
    fixed per-model_type constant as the teacher-backed path.
  * train_phase() actually runs end-to-end with teacher=None, kd_weight=0,
    producing an all-zero loss_kd_history and a valid loss_gt_history -- and,
    combined with --cfg_label_drop > 0, correctly falls back to the student's
    OWN y_embedder.num_classes for the CFG null-class index.
"""
import copy
import os
import sys
import types

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, _SCRIPTS)
from evaluate_parameters_dit_micro import DiTMicro
from train_phase_students import train_phase, get_in_channels, should_skip_teacher


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
        gt_weight=1.0,
        kd_weight=0.0,
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
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _build_student(num_classes: int = 10) -> DiTMicro:
    """Tiny DiT-Micro student with initialised attention weights. DiTMicro's
    HookableMultiheadSelfAttention allocates in_proj_weight/bias via torch.empty
    (uninitialized memory) -- must be explicitly initialised or the forward
    pass produces NaN/Inf (mirrors test_resume_integration.py::_build_teacher)."""
    model = DiTMicro(num_heads=3, num_classes=num_classes)
    for block in model.blocks:
        nn.init.normal_(block.attn.in_proj_weight, std=0.02)
        nn.init.zeros_(block.attn.in_proj_bias)
    return model.train()


def _build_loader():
    images = torch.randn(4, 3, 32, 32)
    labels = torch.randint(0, 10, (4,))
    ds = TensorDataset(images, labels)
    return DataLoader(ds, batch_size=2, shuffle=False)


# ---------------------------------------------------------------------------
# should_skip_teacher
# ---------------------------------------------------------------------------

def test_should_skip_teacher_true_only_for_arch_plan_and_kd_zero():
    assert should_skip_teacher(_args(arch_plan="p.json", kd_weight=0.0)) is True
    # kd_weight negative also counts as "no KD" (<=0)
    assert should_skip_teacher(_args(arch_plan="p.json", kd_weight=-1.0)) is True


def test_should_skip_teacher_false_when_kd_weight_positive():
    assert should_skip_teacher(_args(arch_plan="p.json", kd_weight=1.0)) is False


def test_should_skip_teacher_false_when_no_arch_plan():
    # Prune/head-pruning paths always need the teacher as weight-init source,
    # regardless of kd_weight -- must NOT skip even with kd_weight==0.
    assert should_skip_teacher(_args(arch_plan=None, kd_weight=0.0)) is False


# ---------------------------------------------------------------------------
# get_in_channels(args, None) fallback
# ---------------------------------------------------------------------------

def test_get_in_channels_none_teacher_matches_constants():
    assert get_in_channels(_args(model_type="dit_micro"), None) == 3
    assert get_in_channels(_args(model_type="dit_xl"), None) == 4


def test_get_in_channels_none_matches_teacher_backed_value():
    teacher = DiTMicro(num_heads=3)
    args = _args(model_type="dit_micro")
    assert get_in_channels(args, None) == get_in_channels(args, teacher)


# ---------------------------------------------------------------------------
# train_phase end-to-end with teacher=None
# ---------------------------------------------------------------------------

def test_train_phase_runs_with_teacher_none_and_kd_weight_zero(tmp_path):
    student = _build_student()
    loader = _build_loader()
    alpha_bar = torch.linspace(0.99, 0.01, 1000)
    phase = {"index": 0, "start": 0, "end": 20}
    args = _args(kd_weight=0.0, gt_weight=1.0)

    meta = train_phase(
        args=args,
        teacher=None,
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
        phase_dir=str(tmp_path),
        use_wandb=False,
        resume_ckpt=None,
    )

    assert len(meta["loss_history"]) == 1
    # KD loss is identically zero throughout: kd_weight<=0 -> the else-branch
    # (torch.zeros) runs every step, never the teacher forward.
    assert all(v == 0.0 for v in meta["loss_kd_history"])
    assert meta["loss_gt_history"][0] > 0.0
    assert os.path.exists(os.path.join(str(tmp_path), "student.pt"))


def test_train_phase_cfg_label_drop_falls_back_to_student_num_classes_when_teacher_none(tmp_path):
    """--cfg_label_drop > 0 with teacher=None must use raw_student.y_embedder.num_classes
    (not crash on teacher.y_embedder) -- and use the SAME null-class index the
    teacher would have had (both share num_classes=10 for CIFAR-10)."""
    student = _build_student(num_classes=10)
    loader = _build_loader()
    alpha_bar = torch.linspace(0.99, 0.01, 1000)
    phase = {"index": 0, "start": 0, "end": 20}
    args = _args(kd_weight=0.0, gt_weight=1.0, cfg_label_drop=0.5)

    # Must not raise (AttributeError on NoneType.y_embedder would fail this).
    meta = train_phase(
        args=args,
        teacher=None,
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
        phase_dir=str(tmp_path),
        use_wandb=False,
        resume_ckpt=None,
    )
    assert len(meta["loss_history"]) == 1
