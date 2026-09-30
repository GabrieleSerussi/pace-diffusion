"""Tests for evaluate_parameters_dit.py — focused on the DiT head ablation hooks.

The interesting new piece is ``TransformerHeadRandomSameNormHook``: ablating a
single attention head BEFORE the output projection by replacing its head_dim
slice with Gaussian noise whose per-example L2 norm matches the original
slice's. The tests below verify:
  - Only the target head's slice changes; other heads' slices are untouched.
  - Per-example L2 norm of the head's pre-proj slice is preserved.
  - The replacement is non-zero in general (i.e. actually random noise, not zeros).
  - Removing the hook restores the original behaviour.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from evaluate_parameters_dit import (  # noqa: E402
    TransformerHeadRandomSameNormHook,
    TransformerHeadZeroHook,
    make_ddpm_alpha_schedule,
    make_timestep_bin_labels,
    make_timestep_schedule,
)


class _MockAttn(torch.nn.Module):
    """Minimal attention module matching the surface used by the hooks: ``proj``,
    ``num_heads``, ``head_dim``. Identity ``proj`` so output == input."""

    def __init__(self, hidden_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.proj = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        torch.nn.init.eye_(self.proj.weight)

    def forward(self, x):
        return self.proj(x)


# ---------------------------------------------------------------------------
# DDPM schedule sanity
# ---------------------------------------------------------------------------

class TestDdpmSchedule:
    def test_alpha_bar_monotonic(self):
        a = make_ddpm_alpha_schedule(num_timesteps=1000)
        assert a.shape == (1000,)
        assert (a[1:] - a[:-1] <= 0).all()
        assert 0.0 <= a.min().item() <= a.max().item() <= 1.0

    def test_timestep_schedule_descending(self):
        ts = make_timestep_schedule(num_levels=10, device=torch.device("cpu"))
        assert ts.shape == (10,)
        assert ts[0].item() == 999 and ts[-1].item() == 0

    def test_bin_labels_count(self):
        ts = make_timestep_schedule(num_levels=20, device=torch.device("cpu"))
        labels = make_timestep_bin_labels(ts, num_bins=5)
        assert len(labels) == 5


# ---------------------------------------------------------------------------
# TransformerHeadZeroHook
# ---------------------------------------------------------------------------

class TestTransformerHeadZeroHook:
    def test_zeros_only_target_head(self):
        attn = _MockAttn(hidden_dim=64, num_heads=4)
        x = torch.ones(1, 10, 64)
        with TransformerHeadZeroHook(attn, head_idx=0):
            out = attn(x)
        assert (out[..., :16] == 0).all()
        assert (out[..., 16:] != 0).any()

    def test_each_head_distinct(self):
        attn = _MockAttn(hidden_dim=64, num_heads=4)
        x = torch.randn(2, 8, 64)
        outs = []
        for h in range(4):
            with TransformerHeadZeroHook(attn, head_idx=h):
                outs.append(attn(x).clone())
        for i in range(4):
            for j in range(i + 1, 4):
                assert not torch.allclose(outs[i], outs[j])

    def test_restored_after_exit(self):
        attn = _MockAttn(hidden_dim=64, num_heads=4)
        x = torch.randn(1, 5, 64)
        baseline = attn(x).clone()
        with TransformerHeadZeroHook(attn, head_idx=2):
            pass
        assert torch.allclose(baseline, attn(x))


# ---------------------------------------------------------------------------
# TransformerHeadRandomSameNormHook
# ---------------------------------------------------------------------------

class TestTransformerHeadRandomSameNormHook:
    def test_other_heads_untouched(self):
        attn = _MockAttn(hidden_dim=64, num_heads=4)
        x = torch.randn(2, 16, 64)
        baseline = attn(x).clone()
        with TransformerHeadRandomSameNormHook(attn, head_idx=1, random_seed=123):
            out = attn(x)
        # With identity proj the output equals the (possibly modified) input.
        # Heads 0, 2, 3 are untouched.
        for h in [0, 2, 3]:
            start, end = h * 16, (h + 1) * 16
            assert torch.allclose(out[..., start:end], baseline[..., start:end])
        # Head 1 has been replaced; should differ from the original (extreme luck excluded).
        assert not torch.allclose(out[..., 16:32], baseline[..., 16:32])

    def test_preserves_per_example_l2_norm(self):
        attn = _MockAttn(hidden_dim=64, num_heads=4)
        x = torch.randn(3, 12, 64)
        head_idx = 2
        start, end = head_idx * 16, (head_idx + 1) * 16

        original_slice = x[..., start:end].clone()
        original_norms = torch.linalg.vector_norm(
            original_slice.reshape(original_slice.shape[0], -1), ord=2, dim=1,
        )

        with TransformerHeadRandomSameNormHook(attn, head_idx=head_idx, random_seed=7):
            out = attn(x)

        new_slice = out[..., start:end]
        new_norms = torch.linalg.vector_norm(
            new_slice.reshape(new_slice.shape[0], -1), ord=2, dim=1,
        )
        assert torch.allclose(original_norms, new_norms, atol=1e-4, rtol=1e-4), \
            f"Per-example norms differ: orig={original_norms}, new={new_norms}"

    def test_replacement_is_random_not_zero(self):
        attn = _MockAttn(hidden_dim=64, num_heads=4)
        x = torch.randn(2, 8, 64)
        with TransformerHeadRandomSameNormHook(attn, head_idx=0, random_seed=42):
            out = attn(x)
        head_slice = out[..., :16]
        assert head_slice.abs().sum().item() > 0, "Replacement should not be all zeros"
        # Different seeds should produce different replacements.
        with TransformerHeadRandomSameNormHook(attn, head_idx=0, random_seed=43):
            out_other = attn(x)
        assert not torch.allclose(head_slice, out_other[..., :16])

    def test_restored_after_exit(self):
        attn = _MockAttn(hidden_dim=64, num_heads=4)
        x = torch.randn(1, 5, 64)
        baseline = attn(x).clone()
        with TransformerHeadRandomSameNormHook(attn, head_idx=3, random_seed=99):
            pass
        assert torch.allclose(baseline, attn(x))

    def test_zero_input_slice_remains_zero(self):
        attn = _MockAttn(hidden_dim=64, num_heads=4)
        x = torch.randn(2, 8, 64)
        x[..., 16:32] = 0  # Zero out head 1's slice entirely.
        with TransformerHeadRandomSameNormHook(attn, head_idx=1, random_seed=5):
            out = attn(x)
        # If the original slice has zero norm, the replacement is zero
        # (random_same_norm_like does not invent magnitude).
        assert (out[..., 16:32] == 0).all()
