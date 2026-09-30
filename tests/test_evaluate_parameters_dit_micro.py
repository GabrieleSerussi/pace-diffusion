"""Tests for evaluate_parameters_dit_micro.py's opt-in NarrowDiT support.

This importance-measurement script was originally written only for the
third-party ``normalcomputing/dit-cifar10-32x32-class`` checkpoint (hardcoded
hidden_size=192/depth=8, reconstructed ``nn.MultiheadAttention``). This repo's
own trained checkpoints (teacher_v2/v3/v4, and every width-allocation student)
are ``pace.dit_arch_alloc.NarrowDiT`` instances instead: arbitrary
(hidden_size, depth), timm-style qkv/proj attention, and an optional
``augment_dim``/``dropout`` (teacher_v4's Karras regularizers). The new
``--arch_cfg`` flag makes those checkpoints loadable by this script too.

Covers:
  - ``load_narrow_dit_network``: strict round-trip load of a NarrowDiT
    checkpoint, with and without ``augment_dim > 0`` (the "does the evaluator
    handle an augment-conditioned teacher" question -- forward is called with
    no ``augment_labels``, exactly like every real eval/KD call site, and must
    match the pre-save model's output bit-for-bit).
  - A checkpoint loaded against a MISMATCHED arch_cfg raises, rather than
    silently dropping weights.
  - ``DiTMicroEvaluator(..., arch_cfg_path=...)`` builds a ``NarrowDiT`` (not
    the legacy ``DiTMicro``); ``arch_cfg_path=None`` (every pre-existing call
    site) is unchanged -- still builds the legacy model, byte-identical to
    before this option existed.
  - ``_make_ablation_context`` dispatches NarrowDiT attention-head targets to
    the timm-style hooks from ``evaluate_parameters_dit.py`` (NarrowAttention's
    ``qkv``/``proj``/``num_heads``/``head_dim`` layout already satisfies their
    duck-typed interface), for all three ablation modes.
  - ``collect_attention_head_groups_dit`` produces the expected groups for a
    NarrowDiT model.
  - End-to-end: a full ``evaluate()`` pass (baseline + zero/random_same_norm/
    permutation ablation, with ``--corruption_order level_major
    --permutation_group_by_level true``) runs without error on a NarrowDiT
    teacher and returns finite losses.
"""

import json
import os
import sys

import pytest
import torch
from torch.utils.data import DataLoader, Dataset
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import evaluate_parameters_dit_micro as edm  # noqa: E402
from evaluate_parameters_dit_micro import (  # noqa: E402
    DiTMicro,
    DiTMicroEvaluator,
    load_narrow_dit_network,
)
from evaluate_parameters_dit import (  # noqa: E402
    TransformerHeadPermutationHook,
    TransformerHeadRandomSameNormHook,
    TransformerHeadZeroHook,
    collect_attention_head_groups_dit,
)
from evaluate_parameters_edm import (  # noqa: E402
    SigmaCorruptionDataset,
    collate_corruption,
)
from pace.dit_arch_alloc import NarrowDiT, build_narrow_dit  # noqa: E402

_MHA_HOOK_TYPES = (edm.MhaHeadZeroHook, edm.MhaHeadRandomSameNormHook, edm.MhaHeadPermutationHook)
_TRANSFORMER_HOOK_TYPES = (TransformerHeadZeroHook, TransformerHeadRandomSameNormHook,
                           TransformerHeadPermutationHook)


def _tiny_cfg(augment_dim: int = 0, dropout: float = 0.0):
    depth = 2
    return {
        "hidden_size": 24,
        "depth": depth,
        "patch_size": 2,
        "in_channels": 3,
        "num_classes": 10,
        "input_size": 8,
        "learn_sigma": False,
        "per_block": [{"num_heads": 3, "attn_inner": 24, "mlp_hidden": 48} for _ in range(depth)],
        "augment_dim": augment_dim,
        "dropout": dropout,
    }


def _save_narrow_dit(tmp_path, cfg):
    """Save a NarrowDiT checkpoint + arch_cfg.json in the REAL on-disk schema:
    arch_cfg.json nests the NarrowDiT kwargs under a "cfg" key (matching
    train_phase_students.py's phase save and scripts/evaluate_students.py's
    load_narrow_dit_from_dir, which reads ``arch_cfg["cfg"]`` -- NOT a flat
    kwargs dict)."""
    model = build_narrow_dit(cfg)
    ckpt_path = tmp_path / "student.pt"
    torch.save(model.state_dict(), ckpt_path)
    arch_cfg_path = tmp_path / "arch_cfg.json"
    json.dump({"cfg": cfg}, open(arch_cfg_path, "w"))
    return model, str(ckpt_path), str(arch_cfg_path)


# ---------------------------------------------------------------------------
# load_narrow_dit_network
# ---------------------------------------------------------------------------

class TestLoadNarrowDitNetwork:
    @pytest.mark.external_dit
    def test_round_trip_no_augment(self, tmp_path):
        cfg = _tiny_cfg()
        model, ckpt_path, arch_cfg_path = _save_narrow_dit(tmp_path, cfg)
        model.eval()

        loaded = load_narrow_dit_network(ckpt_path, arch_cfg_path, device="cpu")
        assert isinstance(loaded, NarrowDiT)
        assert not any(p.requires_grad for p in loaded.parameters())
        assert not loaded.training

        x = torch.randn(2, 3, 8, 8)
        t = torch.rand(2)
        y = torch.randint(0, 10, (2,))
        with torch.no_grad():
            out_orig = model(x, t, y)
            out_loaded = loaded(x, t, y)
        assert torch.allclose(out_orig, out_loaded)

    @pytest.mark.external_dit
    def test_round_trip_with_augment_dim(self, tmp_path):
        """teacher_v4-shaped checkpoint: augment_dim=9 (non-leaky augmentation
        conditioning) + dropout. Confirms the aug_embedder module strict-loads,
        and that calling forward with NO augment_labels (every real eval/KD call
        site) reproduces the pre-save model's output exactly -- the "zero
        augment vector or None" question resolves to: None is fine, the
        conditioning term is simply omitted (aug_embedder is bias-free, so a
        zero vector would give the identical zero contribution anyway)."""
        cfg = _tiny_cfg(augment_dim=9, dropout=0.1)
        model, ckpt_path, arch_cfg_path = _save_narrow_dit(tmp_path, cfg)
        assert model.aug_embedder is not None
        model.eval()

        loaded = load_narrow_dit_network(ckpt_path, arch_cfg_path, device="cpu")
        assert loaded.aug_embedder is not None
        assert loaded.augment_dim == 9

        x = torch.randn(2, 3, 8, 8)
        t = torch.rand(2)
        y = torch.randint(0, 10, (2,))
        with torch.no_grad():
            out_orig = model(x, t, y)  # augment_labels defaults to None
            out_loaded = loaded(x, t, y)
        assert torch.allclose(out_orig, out_loaded)
        assert torch.isfinite(out_loaded).all()

    @pytest.mark.external_dit
    def test_mismatched_arch_cfg_raises(self, tmp_path):
        cfg = _tiny_cfg()
        _, ckpt_path, _ = _save_narrow_dit(tmp_path, cfg)
        wrong_cfg = dict(
            cfg, hidden_size=48,
            per_block=[{"num_heads": 3, "attn_inner": 48, "mlp_hidden": 96} for _ in range(2)],
        )
        wrong_arch_cfg_path = tmp_path / "wrong_arch_cfg.json"
        json.dump({"cfg": wrong_cfg}, open(wrong_arch_cfg_path, "w"))
        with pytest.raises(RuntimeError):
            load_narrow_dit_network(ckpt_path, str(wrong_arch_cfg_path), device="cpu")


# ---------------------------------------------------------------------------
# DiTMicroEvaluator dispatch: arch_cfg_path set vs None (legacy, unchanged)
# ---------------------------------------------------------------------------

class TestDiTMicroEvaluatorNarrowDitDispatch:
    @pytest.mark.external_dit
    def test_arch_cfg_builds_narrow_dit(self, tmp_path):
        cfg = _tiny_cfg()
        _, ckpt_path, arch_cfg_path = _save_narrow_dit(tmp_path, cfg)
        ev = DiTMicroEvaluator(
            checkpoint_path=ckpt_path, device="cpu", dtype=torch.float32,
            num_timesteps=1000, num_timestep_levels=8, arch_cfg_path=arch_cfg_path,
        )
        assert isinstance(ev.net, NarrowDiT)
        assert len(ev.net.blocks) == 2
        assert ev.net.num_heads == 3

    def test_legacy_path_unchanged_without_arch_cfg(self, monkeypatch):
        """arch_cfg_path=None (every pre-existing call site) must still take the
        legacy load_dit_micro_network path, never load_narrow_dit_network --
        the byte-identical-by-default guarantee."""
        sentinel = DiTMicro()  # a real nn.Module; never actually checkpoint-loaded here
        calls = {"legacy": 0, "narrow": 0}

        def _fake_legacy(checkpoint_path, device, num_heads):
            calls["legacy"] += 1
            return sentinel

        def _fake_narrow(*args, **kwargs):
            calls["narrow"] += 1
            raise AssertionError("load_narrow_dit_network must not be called when arch_cfg_path is None")

        monkeypatch.setattr(edm, "load_dit_micro_network", _fake_legacy)
        monkeypatch.setattr(edm, "load_narrow_dit_network", _fake_narrow)

        ev = DiTMicroEvaluator(
            checkpoint_path="unused.pt", device="cpu", dtype=torch.float32,
            num_timesteps=1000, num_timestep_levels=8, arch_cfg_path=None,
        )
        assert calls == {"legacy": 1, "narrow": 0}
        assert ev.net is sentinel


# ---------------------------------------------------------------------------
# Ablation-hook dispatch: NarrowDiT -> timm-style hooks, legacy -> MHA hooks
# ---------------------------------------------------------------------------

class TestAblationHookDispatch:
    def _narrow_evaluator(self, tmp_path):
        cfg = _tiny_cfg()
        _, ckpt_path, arch_cfg_path = _save_narrow_dit(tmp_path, cfg)
        return DiTMicroEvaluator(
            checkpoint_path=ckpt_path, device="cpu", dtype=torch.float32,
            num_timesteps=1000, num_timestep_levels=8, arch_cfg_path=arch_cfg_path,
        )

    @pytest.mark.parametrize("mode,expected", [
        ("zero", TransformerHeadZeroHook),
        ("random_same_norm", TransformerHeadRandomSameNormHook),
        ("permutation", TransformerHeadPermutationHook),
    ])
    @pytest.mark.external_dit
    def test_narrow_dit_dispatches_to_transformer_hooks(self, tmp_path, mode, expected):
        ev = self._narrow_evaluator(tmp_path)
        attn = ev.net.blocks[0].attn
        ctx = ev._make_ablation_context((attn, 0), mode, ablation_random_seed=0)
        assert isinstance(ctx, expected)
        assert not isinstance(ctx, _MHA_HOOK_TYPES)

    def test_legacy_dispatches_to_mha_hooks(self, monkeypatch):
        sentinel = DiTMicro()
        monkeypatch.setattr(edm, "load_dit_micro_network", lambda *a, **kw: sentinel)
        ev = DiTMicroEvaluator(
            checkpoint_path="unused.pt", device="cpu", dtype=torch.float32,
            num_timesteps=1000, num_timestep_levels=8, arch_cfg_path=None,
        )
        attn = ev.net.blocks[0].attn
        ctx = ev._make_ablation_context((attn, 0), "permutation", ablation_random_seed=0)
        assert isinstance(ctx, edm.MhaHeadPermutationHook)
        assert not isinstance(ctx, _TRANSFORMER_HOOK_TYPES)

    @pytest.mark.external_dit
    def test_permutation_context_exposes_set_permutation_groups_others_dont(self, tmp_path):
        ev = self._narrow_evaluator(tmp_path)
        attn = ev.net.blocks[0].attn
        perm_ctx = ev._make_ablation_context((attn, 0), "permutation", ablation_random_seed=0)
        zero_ctx = ev._make_ablation_context((attn, 0), "zero", ablation_random_seed=None)
        assert hasattr(perm_ctx, "set_permutation_groups")
        assert not hasattr(zero_ctx, "set_permutation_groups")


@pytest.mark.external_dit
def test_collect_attention_head_groups_dit_on_narrow_dit(tmp_path):
    cfg = _tiny_cfg()
    model = build_narrow_dit(cfg)
    groups = collect_attention_head_groups_dit(model)
    assert len(groups) == 2 * 3  # depth=2, num_heads=3
    assert set(groups.keys()) == {
        f"blocks.{b}.attn.head_{h}" for b in range(2) for h in range(3)
    }
    for mod, head_idx in groups.values():
        assert mod is model.blocks[0].attn or mod is model.blocks[1].attn
        assert 0 <= head_idx < 3


# ---------------------------------------------------------------------------
# End-to-end: evaluate() on a NarrowDiT teacher, all 3 ablation modes,
# level_major + permutation_group_by_level (the corrected protocol).
# ---------------------------------------------------------------------------

class _TinyImageDataset(Dataset):
    """4 synthetic (image, label) pairs -- enough to exercise level-major
    ordering (one full batch == one level == all 4 images) without touching
    real CIFAR-10 data on disk."""

    def __init__(self, n=4, image_size=8):
        gen = torch.Generator().manual_seed(0)
        self.images = torch.rand(n, 3, image_size, image_size, generator=gen) * 2 - 1
        self.labels = [i % 10 for i in range(n)]

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


@pytest.mark.external_dit
@pytest.mark.parametrize("ablation_mode", ["zero", "random_same_norm", "permutation"])
def test_evaluate_end_to_end_on_narrow_dit(tmp_path, ablation_mode):
    cfg = _tiny_cfg()
    _, ckpt_path, arch_cfg_path = _save_narrow_dit(tmp_path, cfg)
    ev = DiTMicroEvaluator(
        checkpoint_path=ckpt_path, device="cpu", dtype=torch.float32,
        num_timesteps=1000, num_timestep_levels=4, arch_cfg_path=arch_cfg_path,
    )
    image_dataset = _TinyImageDataset(n=4, image_size=8)
    corruption_dataset = SigmaCorruptionDataset(
        image_dataset=image_dataset,
        sigma_values=ev.timestep_values.float(),
        samples_per_image=1,
        seed=0,
        order="level_major",
    )
    # level-major + batch_size == len(image_dataset): every batch is exactly
    # one full timestep-level (the corrected protocol's precondition).
    dataloader = DataLoader(
        corruption_dataset, batch_size=4, shuffle=False, collate_fn=collate_corruption,
    )

    attn = ev.net.blocks[0].attn
    stats = ev.evaluate(
        dataloader=dataloader, num_bins=4,
        ablate_target=(attn, 0), ablation_mode=ablation_mode, ablation_random_seed=0,
        permutation_group_by_level=True,
    )
    mean = stats.mean()
    assert mean.shape == (4,)
    assert torch.isfinite(mean).all()
