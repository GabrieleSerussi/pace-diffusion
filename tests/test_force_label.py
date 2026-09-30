"""
--force_label: opt-in, generalizes --unconditional (tests/test_unconditional_ddpm.py)
from a hardcoded index-0 row to an ARBITRARY fixed row index, so a fine-tune of a
REAL class-conditional checkpoint (e.g. DiT-XL/2, 1001-row y_embedder: 1000 real
ImageNet classes + 1 trained null/CFG row at index 1000) can be driven through its
own already-trained "unconditional" row instead of repurposing an arbitrary real
class row (index 0).

Covers:
  * parse_args(): --force_label defaults to None.
  * model_forward_for_loss: --force_label forces labels to the given index (ddpm
    branch); takes precedence over --unconditional when both happen to be set;
    a no-op (labels pass through untouched) when None/absent.
  * main()'s mutual-exclusion guards: --force_label + --cfg_label_drop>0,
    --force_label + --unconditional, --force_label < 0.
  * train_phase() end-to-end with a REAL-SHAPED (1001-row) LabelEmbedder --
    num_classes=1000 (matching the stock DiT-XL/2 checkpoint), force_label=1000
    (the null row) -- regression protection against an off-by-one/out-of-range
    index into the embedding table (the num_classes=1 case in
    test_unconditional_ddpm.py has only 2 rows total and would not catch this).
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
        unconditional=False,
        force_label=None,
        use_wandb=False,
        wandb_project=None,
        wandb_run_name=None,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# parse_args default
# ---------------------------------------------------------------------------

def test_force_label_defaults_to_none():
    argv = sys.argv
    sys.argv = ["train_phase_students.py",
                "--model_type", "dit_micro", "--teacher_checkpoint", "x",
                "--output_dir", "o", "--dataset", "cifar10", "--grouping_json", "g"]
    try:
        args = parse_args()
    finally:
        sys.argv = argv
    assert args.force_label is None


# ---------------------------------------------------------------------------
# model_forward_for_loss: labels forced to the given index (ddpm branch)
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


def test_force_label_forces_labels_to_given_index():
    model = _CapturingModel()
    clean, timesteps, labels, alpha_bar = _ddpm_inputs()
    args = _args(force_label=1000)
    model_forward_for_loss(
        model=model, args=args, clean=clean, timesteps=timesteps, labels=labels,
        alpha_bar=alpha_bar, in_channels=4, compute_dtype=torch.float32,
    )
    assert model.seen_labels is not None
    assert torch.equal(model.seen_labels, torch.full_like(labels, 1000))


def test_force_label_takes_precedence_over_unconditional():
    model = _CapturingModel()
    clean, timesteps, labels, alpha_bar = _ddpm_inputs()
    args = _args(force_label=1000, unconditional=True)
    model_forward_for_loss(
        model=model, args=args, clean=clean, timesteps=timesteps, labels=labels,
        alpha_bar=alpha_bar, in_channels=4, compute_dtype=torch.float32,
    )
    # force_label (1000) wins, NOT unconditional's hardcoded 0.
    assert torch.equal(model.seen_labels, torch.full_like(labels, 1000))


def test_force_label_none_leaves_labels_untouched():
    model = _CapturingModel()
    clean, timesteps, labels, alpha_bar = _ddpm_inputs()
    args = _args(force_label=None, unconditional=False)
    model_forward_for_loss(
        model=model, args=args, clean=clean, timesteps=timesteps, labels=labels,
        alpha_bar=alpha_bar, in_channels=4, compute_dtype=torch.float32,
    )
    assert torch.equal(model.seen_labels, labels)


def test_force_label_defaults_to_none_when_attr_missing():
    """getattr fallback for args namespaces built before --force_label existed."""
    model = _CapturingModel()
    clean, timesteps, labels, alpha_bar = _ddpm_inputs()
    args = _args()
    del args.force_label
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
    "--output_dir", "/tmp/_unused_force_label_test",
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


def test_force_label_rejects_cfg_label_drop():
    assert "incompatible with --cfg_label_drop" in _main_with(
        ["--force_label", "1000", "--cfg_label_drop", "0.5"])


def test_force_label_rejects_combination_with_unconditional():
    assert "mutually exclusive" in _main_with(
        ["--force_label", "1000", "--unconditional"])


def test_force_label_rejects_negative():
    assert "must be >= 0" in _main_with(["--force_label", "-1"])


def test_force_label_alone_passes_the_early_validation_gate():
    # --force_label alone must clear our new guards: it proceeds past main()'s
    # CLI-validation block and fails later, on the nonexistent --arch_plan file
    # (FileNotFoundError, not one of our guards' SystemExit).
    import train_phase_students
    argv = sys.argv
    sys.argv = _BASE_ARGV + ["--force_label", "1000"]
    try:
        with pytest.raises(BaseException) as ei:
            train_phase_students.main()
        if isinstance(ei.value, SystemExit):
            msg = str(ei.value)
            assert "incompatible with --cfg_label_drop" not in msg
            assert "mutually exclusive" not in msg
            assert "must be >= 0" not in msg
    finally:
        sys.argv = argv


# ---------------------------------------------------------------------------
# End-to-end: train_phase() with a REAL-SHAPED (1001-row) LabelEmbedder,
# force_label=1000 (the null row), teacher=None, image_folder-shaped
# (label==-1) batches. Regression test for the null-row index actually being
# in-range (num_classes=1000 -> table has 1001 rows, indices 0..1000).
# ---------------------------------------------------------------------------

def _build_tiny_1000class_student():
    cfg = dict(
        hidden_size=32, depth=2, patch_size=2, in_channels=4, num_classes=1000,
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
def test_train_phase_runs_force_label_into_real_sized_null_row(tmp_path):
    student = _build_tiny_1000class_student()
    assert student.y_embedder.embedding_table.weight.shape[0] == 1001  # 1000 + null row
    loader = _build_loader()
    alpha_bar = torch.linspace(0.999, 0.001, 1000)
    phase = {"index": 0, "start": 0, "end": 1}
    args = _args(kd_weight=0.0, gt_weight=1.0, force_label=1000)

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
