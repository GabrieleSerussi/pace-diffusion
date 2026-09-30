"""
--unconditional: opt-in y=0 (single-class) label forcing in model_forward_for_loss,
for --model_type dit_xl --diffusion ddpm unconditional pretraining on an
image_folder dataset (whose labels are the sentinel -1, see ImageFolderFlat in
evaluate_parameters_edm.py). Default OFF (unconditional=False) is byte-identical:
labels pass through model_forward_for_loss untouched.

Covers:
  * parse_args(): --unconditional defaults to False.
  * model_forward_for_loss: forces labels to zero (ddpm branch) when
    args.unconditional is True; a no-op when False/absent (regression
    protection -- every existing dit_xl/dit_micro run is unaffected).
  * main()'s --unconditional / --cfg_label_drop mutual-exclusion guard.
  * train_phase() end-to-end with a real (tiny) NarrowDiT built the same way
    --arch_plan builds one (num_classes=1, learn_sigma=True), model_type=dit_xl,
    diffusion=ddpm, teacher=None (kd_weight=0), and an image_folder-shaped
    dataloader (labels == -1) -- must not crash indexing into the single-row
    LabelEmbedder.
"""
import os
import sys
import types

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, _SCRIPTS)
from train_phase_students import model_forward_for_loss, parse_args, train_phase
from pace.dit_arch_alloc import build_narrow_dit


def _args(**overrides):
    base = dict(
        model_type="dit_xl",
        diffusion="ddpm",
        num_timesteps=1000,
        num_epochs=1,
        lr=1e-4,
        weight_decay=0.01,
        lr_min_ratio=0.01,
        grad_clip=1.0,
        gt_weight=1.0,
        kd_weight=0.0,
        log_every=50,
        device="cpu",
        ckpt_every=0,
        arch_plan="dummy_plan.json",
        variant="global",
        cfg_label_drop=0.0,
        unconditional=True,
        use_wandb=False,
        wandb_project=None,
        wandb_run_name=None,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# parse_args default
# ---------------------------------------------------------------------------

def test_unconditional_defaults_to_false():
    argv = sys.argv
    sys.argv = ["train_phase_students.py",
                "--model_type", "dit_micro", "--teacher_checkpoint", "x",
                "--output_dir", "o", "--dataset", "cifar10", "--grouping_json", "g"]
    try:
        args = parse_args()
    finally:
        sys.argv = argv
    assert args.unconditional is False


# ---------------------------------------------------------------------------
# model_forward_for_loss: labels forced to zero (ddpm branch)
# ---------------------------------------------------------------------------

class _CapturingModel(nn.Module):
    """Records the labels tensor it was called with; returns zeros shaped like
    a learn_sigma=True DiT-XL/2-family output (2*in_channels)."""

    def __init__(self, in_channels=4):
        super().__init__()
        self.in_channels = in_channels
        self.seen_labels = None
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x, t, y):
        self.seen_labels = y.clone()
        b, c, h, w = x.shape
        return torch.zeros(b, 2 * self.in_channels, h, w) + self.dummy


def _ddpm_inputs(batch=4):
    clean = torch.randn(batch, 4, 8, 8)
    timesteps = torch.randint(0, 1000, (batch,))
    labels = torch.full((batch,), -1, dtype=torch.long)  # ImageFolderFlat sentinel
    alpha_bar = torch.linspace(0.999, 0.001, 1000)
    return clean, timesteps, labels, alpha_bar


def test_unconditional_true_forces_labels_to_zero_before_ddpm_forward():
    model = _CapturingModel()
    clean, timesteps, labels, alpha_bar = _ddpm_inputs()
    args = _args(unconditional=True)
    model_forward_for_loss(
        model=model, args=args, clean=clean, timesteps=timesteps, labels=labels,
        alpha_bar=alpha_bar, in_channels=4, compute_dtype=torch.float32,
    )
    assert model.seen_labels is not None
    assert torch.equal(model.seen_labels, torch.zeros_like(labels))


def test_unconditional_false_leaves_labels_untouched():
    model = _CapturingModel()
    clean, timesteps, labels, alpha_bar = _ddpm_inputs()
    args = _args(unconditional=False)
    model_forward_for_loss(
        model=model, args=args, clean=clean, timesteps=timesteps, labels=labels,
        alpha_bar=alpha_bar, in_channels=4, compute_dtype=torch.float32,
    )
    # Byte-identical to before this flag existed: still the raw -1 sentinel.
    assert torch.equal(model.seen_labels, labels)


def test_unconditional_defaults_to_false_when_attr_missing():
    """getattr fallback for args namespaces built before --unconditional existed
    (e.g. a saved cfg_payload from an older full checkpoint)."""
    model = _CapturingModel()
    clean, timesteps, labels, alpha_bar = _ddpm_inputs()
    args = _args()
    del args.unconditional
    model_forward_for_loss(
        model=model, args=args, clean=clean, timesteps=timesteps, labels=labels,
        alpha_bar=alpha_bar, in_channels=4, compute_dtype=torch.float32,
    )
    assert torch.equal(model.seen_labels, labels)


# ---------------------------------------------------------------------------
# main() validation
# ---------------------------------------------------------------------------

_BASE_ARGV = [
    "train_phase_students.py",
    "--model_type", "dit_xl",
    "--teacher_checkpoint", "/nonexistent.pt",
    "--output_dir", "/tmp/_unused_unconditional_test",
    "--dataset", "image_folder",
    "--image_root", "/nonexistent_root",
    "--diffusion", "ddpm",
    "--arch_plan", "/nonexistent_plan.json",
    "--variant", "global",
    "--kd_weight", "0",
]


def _main_with(extra):
    import train_phase_students
    argv = sys.argv
    sys.argv = _BASE_ARGV + extra
    try:
        with pytest.raises(SystemExit) as ei:
            train_phase_students.main()
        return str(ei.value)
    finally:
        sys.argv = argv


def test_unconditional_rejects_cfg_label_drop():
    assert "incompatible with --cfg_label_drop" in _main_with(
        ["--unconditional", "--cfg_label_drop", "0.5"])


def test_unconditional_alone_passes_the_early_validation_gate():
    # --unconditional alone must clear our new guard: it proceeds past
    # main()'s CLI-validation block and fails later, on the nonexistent
    # --arch_plan file (FileNotFoundError, not our guard's SystemExit).
    import train_phase_students
    argv = sys.argv
    sys.argv = _BASE_ARGV + ["--unconditional"]
    try:
        with pytest.raises(BaseException) as ei:
            train_phase_students.main()
        if isinstance(ei.value, SystemExit):
            assert "incompatible with --cfg_label_drop" not in str(ei.value)
    finally:
        sys.argv = argv


# ---------------------------------------------------------------------------
# End-to-end: train_phase() with a fresh unconditional NarrowDiT, teacher=None,
# image_folder-shaped (label==-1) batches.
# ---------------------------------------------------------------------------

def _build_tiny_unconditional_student():
    cfg = dict(
        hidden_size=32, depth=2, patch_size=2, in_channels=4, num_classes=1,
        input_size=8, learn_sigma=True,
        per_block=[{"num_heads": 2, "attn_inner": 32, "mlp_hidden": 128} for _ in range(2)],
    )
    return build_narrow_dit(cfg)


def _build_loader():
    latents = torch.randn(8, 4, 8, 8)
    labels = torch.full((8,), -1, dtype=torch.long)
    ds = TensorDataset(latents, labels)
    return DataLoader(ds, batch_size=4, shuffle=False)


@pytest.mark.external_dit
def test_train_phase_runs_unconditional_dit_xl_ddpm_with_sentinel_labels(tmp_path):
    student = _build_tiny_unconditional_student()
    loader = _build_loader()
    alpha_bar = torch.linspace(0.999, 0.001, 1000)
    phase = {"index": 0, "start": 0, "end": 1}
    args = _args(kd_weight=0.0, gt_weight=1.0, unconditional=True)

    meta = train_phase(
        args=args,
        teacher=None,
        student_ddp=student,
        raw_student=student,
        vae=None,
        dataloader=loader,
        sampler=None,
        phase=phase,
        num_bins=1,
        alpha_bar=alpha_bar,
        in_channels=4,
        compute_dtype=torch.float32,
        phase_dir=str(tmp_path),
        use_wandb=False,
        resume_ckpt=None,
    )
    assert len(meta["loss_history"]) == 1
    assert all(v == 0.0 for v in meta["loss_kd_history"])  # kd_weight == 0
    assert meta["loss_gt_history"][0] > 0.0
    assert os.path.exists(os.path.join(str(tmp_path), "student.pt"))
