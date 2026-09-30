"""
CPU integration test for epoch-level checkpoint-resume load path in train_phase.

Verifies that when resume_ckpt={"model", "optimizer", "scheduler", "epoch": 0} is
supplied and num_epochs=3, train_phase runs only epochs 1 and 2 (2 epochs total),
producing len(meta["loss_history"]) == 2.
"""

import copy
import os
import sys
import types

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, _SCRIPTS)
from evaluate_parameters_dit_micro import DiTMicro
from train_phase_students import train_phase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_teacher():
    """Tiny DiT-Micro teacher with initialised attention weights."""
    model = DiTMicro(num_heads=3)
    for block in model.blocks:
        nn.init.normal_(block.attn.in_proj_weight, std=0.02)
        nn.init.zeros_(block.attn.in_proj_bias)
    return model.eval()


def _build_loader():
    """4 CIFAR-shaped images, batch size 2 -> 2 batches per epoch."""
    images = torch.randn(4, 3, 32, 32)
    labels = torch.zeros(4, dtype=torch.long)
    ds = TensorDataset(images, labels)
    return DataLoader(ds, batch_size=2, shuffle=False)


def _make_args(num_epochs: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        model_type="dit_micro",
        diffusion="edm",
        edm_loss_space="f",
        num_epochs=num_epochs,
        lr=1e-4,
        weight_decay=0.01,
        lr_min_ratio=0.01,
        grad_clip=1.0,
        gt_weight=1.0,
        kd_weight=1.0,
        sigma_min=0.002,
        sigma_max=80.0,
        sigma_data=0.5,
        rho=7.0,
        num_timesteps=1000,
        log_every=50,
        device="cpu",
        ckpt_every=0,
        # wandb
        use_wandb=False,
        wandb_project=None,
        wandb_run_name=None,
    )


def _build_resume_ckpt(student: nn.Module, loader: DataLoader, num_epochs: int) -> dict:
    """Construct a minimal resume_ckpt for epoch=0 (epoch 0 already done)."""
    opt = torch.optim.AdamW(student.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=num_epochs * len(loader)
    )
    return {
        "model": student.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "epoch": 0,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_resume_runs_remaining_epochs(tmp_path):
    """With resume_ckpt epoch=0 and num_epochs=3, only epochs 1 and 2 run
    -> len(loss_history) == 2, student.pt written, resume.pt removed."""
    teacher = _build_teacher()
    student = copy.deepcopy(teacher).train()
    loader = _build_loader()
    alpha_bar = torch.linspace(0.99, 0.01, 1000)
    phase = {"index": 0, "start": 0, "end": 20}
    args = _make_args(num_epochs=3)

    resume_ckpt = _build_resume_ckpt(student, loader, num_epochs=3)

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
        phase_dir=str(tmp_path),
        use_wandb=False,
        resume_ckpt=resume_ckpt,
    )

    # Core assertion: only 2 epochs actually ran (epochs 1 and 2; epoch 0 skipped)
    assert len(meta["loss_history"]) == 2, (
        f"Expected 2 epochs in loss_history (resume from epoch 0, num_epochs=3), "
        f"got {len(meta['loss_history'])}: {meta['loss_history']}"
    )

    # Phase-completion artifacts
    student_pt = os.path.join(str(tmp_path), "student.pt")
    resume_pt = os.path.join(str(tmp_path), "resume.pt")
    assert os.path.exists(student_pt), "student.pt not written at phase end"
    assert not os.path.exists(resume_pt), "resume.pt should be removed after phase completes"


def test_fresh_run_all_epochs(tmp_path):
    """Sanity: fresh run (no resume_ckpt) with num_epochs=2 runs all 2 epochs."""
    teacher = _build_teacher()
    student = copy.deepcopy(teacher).train()
    loader = _build_loader()
    alpha_bar = torch.linspace(0.99, 0.01, 1000)
    phase = {"index": 0, "start": 0, "end": 20}
    args = _make_args(num_epochs=2)

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
        phase_dir=str(tmp_path),
        use_wandb=False,
        resume_ckpt=None,
    )

    assert len(meta["loss_history"]) == 2, (
        f"Expected 2 epochs in fresh run loss_history, got {len(meta['loss_history'])}"
    )
    assert os.path.exists(os.path.join(str(tmp_path), "student.pt"))
