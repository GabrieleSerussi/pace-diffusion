"""--teacher_arch_cfg extended to --model_type dit_xl (graduation-and-launch task).

load_teacher_for_model_type's dit_xl branch historically ALWAYS called
load_dit_network (a hardcoded DiT_models[--dit_model] reconstruction), with no
way to load a NarrowDiT-format teacher -- unlike the dit_micro branch, which
already supported this via --teacher_arch_cfg (tests/test_teacher_arch_cfg.py).
This is a real gap for KD-training narrower --arch_plan students distilled
from a from-scratch --arch_plan "global" teacher built under the dit_xl
(DDPM/VAE-latent) family itself -- e.g. the objective-ablation DiT-B/2
pretrains (hidden_size=768/depth=12/
num_classes=1/learn_sigma=True): its state_dict shape depends on arch_cfg.json,
which load_dit_network's stock reconstruction cannot represent (wrong shape,
guaranteed crash).

Covers (mirrors test_teacher_arch_cfg.py's dit_micro coverage exactly, for
dit_xl instead):
  - load_teacher_for_model_type dispatches to load_narrow_dit_network when
    --teacher_arch_cfg is set for --model_type dit_xl.
  - Legacy path (--teacher_arch_cfg=None) is unaffected: still calls
    load_dit_network, never load_narrow_dit_network (every existing
    --teacher_checkpoint, e.g. a real stock DiT-XL/2 fine-tune, unaffected).
  - End-to-end: train_phase() runs a real KD step against a NarrowDiT teacher
    built with the DiT-B pretrain's own convention (in_channels=4, num_classes=1,
    learn_sigma=True) and --unconditional (forces the image_folder sentinel
    label -1 to 0), producing a finite, nonzero KD loss.
"""
import json
import os
import sys
import types

import pytest

import torch
from torch.utils.data import DataLoader, TensorDataset

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, _SCRIPTS)
import train_phase_students as tps  # noqa: E402
from train_phase_students import load_teacher_for_model_type, train_phase  # noqa: E402
from pace.dit_arch_alloc import NarrowDiT, build_narrow_dit  # noqa: E402


def _args(**overrides):
    base = dict(
        model_type="dit_xl",
        dit_model="DiT-XL/2",
        image_size=256,
        diffusion="ddpm",
        num_epochs=1,
        lr=1e-4,
        weight_decay=0.01,
        lr_min_ratio=0.01,
        grad_clip=1.0,
        gt_weight=0.25,
        kd_weight=1.0,
        num_timesteps=1000,
        log_every=50,
        device="cpu",
        ckpt_every=0,
        arch_plan="dummy_plan.json",
        cfg_label_drop=0.0,
        unconditional=False,
        use_wandb=False,
        wandb_project=None,
        wandb_run_name=None,
        teacher_arch_cfg=None,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _ditb_like_cfg():
    """Mirrors DITB_TEACHER_CFG's convention (scripts/dit_arch_to_plans.py):
    in_channels=4 (VAE latents), num_classes=1 (single unconditional class),
    learn_sigma=True -- shrunk to a tiny depth/width for a fast CPU test."""
    depth = 2
    return {
        "hidden_size": 32, "depth": depth, "patch_size": 2, "in_channels": 4,
        "num_classes": 1, "input_size": 8, "learn_sigma": True,
        "per_block": [{"num_heads": 4, "attn_inner": 32, "mlp_hidden": 64} for _ in range(depth)],
    }


def _save_teacher(tmp_path):
    cfg = _ditb_like_cfg()
    model = build_narrow_dit(cfg)
    ckpt_path = tmp_path / "teacher_student.pt"
    torch.save(model.state_dict(), ckpt_path)
    arch_cfg_path = tmp_path / "teacher_arch_cfg.json"
    json.dump({"cfg": cfg}, open(arch_cfg_path, "w"))
    return model, str(ckpt_path), str(arch_cfg_path)


def _build_student():
    return build_narrow_dit(_ditb_like_cfg()).train().requires_grad_(True)


def _build_loader():
    # image_folder's sentinel label for an unconditional dataset is -1
    # (ImageFolderFlat); --unconditional forces it to 0 before any embedding lookup.
    latents = torch.randn(4, 4, 8, 8)
    labels = torch.full((4,), -1, dtype=torch.long)
    ds = TensorDataset(latents, labels)
    return DataLoader(ds, batch_size=2, shuffle=False)


# ---------------------------------------------------------------------------
# load_teacher_for_model_type dispatch (dit_xl)
# ---------------------------------------------------------------------------

@pytest.mark.external_dit
def test_dispatches_to_narrow_dit_when_teacher_arch_cfg_set_for_dit_xl(tmp_path):
    _, ckpt_path, arch_cfg_path = _save_teacher(tmp_path)
    args = _args(teacher_checkpoint=ckpt_path, teacher_arch_cfg=arch_cfg_path)
    teacher = load_teacher_for_model_type(args, device="cpu", dtype=torch.float32)
    assert isinstance(teacher, NarrowDiT)
    assert teacher.in_channels == 4
    assert teacher.learn_sigma is True


def test_legacy_path_unchanged_without_teacher_arch_cfg_for_dit_xl(monkeypatch):
    """teacher_arch_cfg=None (every pre-existing dit_xl caller, e.g. the real
    stock finetune_xl_{bedroom,ffhq}_ddpm checkpoints) must still take the
    legacy load_dit_network path, never load_narrow_dit_network."""
    sentinel = object()
    calls = {"legacy": 0, "narrow": 0}

    def _fake_legacy(checkpoint_path, dit_model, input_size, num_classes, device, dtype):
        calls["legacy"] += 1
        return sentinel

    def _fake_narrow(*a, **kw):
        calls["narrow"] += 1
        raise AssertionError("load_narrow_dit_network must not be called when teacher_arch_cfg is None")

    monkeypatch.setattr(tps, "load_dit_network", _fake_legacy)
    monkeypatch.setattr(tps, "load_narrow_dit_network", _fake_narrow)

    args = _args(teacher_checkpoint="unused.pt", teacher_arch_cfg=None)
    teacher = load_teacher_for_model_type(args, device="cpu", dtype=torch.float32)
    assert calls == {"legacy": 1, "narrow": 0}
    assert teacher is sentinel


# ---------------------------------------------------------------------------
# End-to-end: KD training step against a DiT-B-like NarrowDiT teacher
# ---------------------------------------------------------------------------

@pytest.mark.external_dit
def test_train_phase_kd_against_ditb_like_narrow_dit_teacher(tmp_path):
    teacher, ckpt_path, arch_cfg_path = _save_teacher(tmp_path)
    teacher.eval().requires_grad_(False)
    student = _build_student()
    loader = _build_loader()
    alpha_bar = torch.linspace(0.9999, 0.02, 1000)
    phase = {"index": 0, "start": 0, "end": 20}
    args = _args(
        teacher_checkpoint=ckpt_path, teacher_arch_cfg=arch_cfg_path,
        kd_weight=1.0, gt_weight=0.25, unconditional=True,
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
        in_channels=4,
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
