"""Regression tests for the period-``batch_size`` n_eff artifact.

Root cause: permutation-importance shuffles
a head's activation along the batch dim. With image-major sample order and
``batch_size < num_timestep_levels`` each batch is a narrow t-window of ONE
image, so the permutation swaps activations ACROSS t-levels. The measured
per-level importance then peaks at the edges of every ``batch_size``-wide window
=> a spurious period-``batch_size`` ripple in n_eff (aliased to period-5-bins for
64 levels / 20 bins). Micro (batch_size=256, every batch spans all levels) was
immune, which is exactly the fingerprint reproduced below.

The fix: order samples level-major AND restrict the permutation to same-level
examples so t is held fixed. These tests would have caught the bug.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from evaluate_parameters_edm import (  # noqa: E402
    BinStats,
    SigmaCorruptionDataset,
    compute_usage_metrics,
    level_to_bin,
    permute_along_batch_like,
    permute_along_batch_within_groups,
)


# --------------------------------------------------------------------------- #
# 1. Pure permutation-helper semantics                                        #
# --------------------------------------------------------------------------- #
def test_within_group_permutation_keeps_partners_in_group():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    b = 48
    group_ids = torch.arange(b) % 4          # 4 levels, 12 members each
    # rows tagged by group so we can check the partner's group after permuting
    tensor = group_ids.float().reshape(b, 1).clone()
    out = permute_along_batch_within_groups(tensor, group_ids, generator=gen)
    # every row still carries a value from its OWN group
    assert torch.equal(out.reshape(-1).long(), group_ids), "within-group permute must not cross groups"
    # it is a genuine permutation of the multiset (marginal per group unchanged)
    for g in group_ids.unique():
        assert torch.equal(
            torch.sort(tensor[group_ids == g].reshape(-1)).values,
            torch.sort(out[group_ids == g].reshape(-1)).values,
        )


def test_full_batch_permutation_crosses_groups():
    """Control: the legacy full-batch permutation mixes groups (that IS the bug)."""
    gen = torch.Generator().manual_seed(0)
    b = 48
    group_ids = torch.arange(b) % 4
    tensor = torch.arange(b).float().reshape(b, 1)  # distinct rows
    out = permute_along_batch_like(tensor, generator=gen)
    moved = (out.reshape(-1) != tensor.reshape(-1))
    assert moved.any(), "sanity: permutation should move at least one row"


def test_singleton_groups_are_identity():
    gen = torch.Generator().manual_seed(0)
    b = 8
    group_ids = torch.arange(b)             # every element its own group
    tensor = torch.randn(b, 3)
    out = permute_along_batch_within_groups(tensor, group_ids, generator=gen)
    assert torch.equal(out, tensor), "singleton groups cannot permute -> identity"


# --------------------------------------------------------------------------- #
# 2. Dataset ordering                                                         #
# --------------------------------------------------------------------------- #
class _DummyImages:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return torch.zeros(1), 0


def _batch_levels(ds, batch_size):
    levels = [s[1] for s in ds.samples]
    return [levels[i:i + batch_size] for i in range(0, len(levels), batch_size)]


def test_orders_produce_same_triples():
    sig = torch.linspace(0, 1, 64)
    a = SigmaCorruptionDataset(_DummyImages(50), sig, 1, seed=0, order="image_major")
    b = SigmaCorruptionDataset(_DummyImages(50), sig, 1, seed=0, order="level_major")
    assert set(a.samples) == set(b.samples), "reordering must not change the (image,level,noise) set"


def test_level_major_batches_are_single_level():
    sig = torch.linspace(0, 1, 64)
    ds = SigmaCorruptionDataset(_DummyImages(1000), sig, 1, seed=0, order="level_major")
    n_pure = sum(len(set(b)) == 1 for b in _batch_levels(ds, 16))
    # with 1000 images/level and batch 16, only the level-boundary batches mix levels
    assert n_pure / (len(ds.samples) // 16) > 0.95


def test_sample_index_addresses_samples_in_both_orders():
    sig = torch.linspace(0, 1, 8)
    for order in ("image_major", "level_major"):
        ds = SigmaCorruptionDataset(_DummyImages(5), sig, 1, seed=0, sigma_stride=3, order=order)
        for image_idx in range(ds.num_images):
            for position, sigma_idx in enumerate(ds.sigma_indices):
                image, level, _noise_seed = ds.samples[ds.sample_index(image_idx, position)]
                assert (image, level) == (image_idx, sigma_idx), order


def test_image_major_batches_span_many_levels():
    sig = torch.linspace(0, 1, 64)
    ds = SigmaCorruptionDataset(_DummyImages(1000), sig, 1, seed=0, order="image_major")
    # legacy: batch_size 16 < 64 levels -> each batch is 16 DISTINCT consecutive levels
    assert all(len(set(b)) == 16 for b in _batch_levels(ds, 16))


# --------------------------------------------------------------------------- #
# 3. End-to-end artifact reproduction and fix                                 #
# --------------------------------------------------------------------------- #
def _measure_per_level_importance(order, within_group, n_img=200, n_lev=64, bs=16, seed=0):
    """Faithful mini-measurement: smooth-in-t per-head activation with a per-image
    offset; importance = mean squared change under the (legacy or fixed) permutation.
    Returns per-level importance averaged over the single 'head'."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_lev)
    base = np.sin(2 * np.pi * t)                       # SMOOTH in t (<=1 cycle): no level-scale structure
    img_off = rng.normal(0, 1.0, n_img)                # per-image variation (what same-t permutation probes)

    ds = SigmaCorruptionDataset(_DummyImages(n_img), torch.linspace(0, 1, n_lev), 1, seed=seed, order=order)
    triples = ds.samples
    gen = torch.Generator().manual_seed(1234)

    imp = np.zeros(n_lev)
    cnt = np.zeros(n_lev)
    for i in range(0, len(triples), bs):
        batch = triples[i:i + bs]
        levs = torch.tensor([lv for (_, lv, _) in batch])
        acts = torch.tensor([base[lv] + img_off[im] for (im, lv, _) in batch], dtype=torch.float32).reshape(-1, 1)
        if within_group:
            corrupt = permute_along_batch_within_groups(acts, levs, generator=gen)
        else:
            corrupt = permute_along_batch_like(acts, generator=gen)
        d = ((acts - corrupt) ** 2).reshape(-1).numpy()
        for j, (_, lv, _) in enumerate(batch):
            imp[lv] += d[j]
            cnt[lv] += 1
    return imp / np.maximum(cnt, 1)


def _edge_vs_mid(imp, period=16):
    edges = imp[np.arange(len(imp)) % period == 0]        # levels 0,16,32,48
    mids = imp[np.arange(len(imp)) % period == period // 2]  # levels 8,24,40,56
    return edges.mean(), mids.mean()


def test_legacy_permutation_reproduces_period_batchsize_artifact():
    imp = _measure_per_level_importance(order="image_major", within_group=False, bs=16)
    edge, mid = _edge_vs_mid(imp, period=16)
    # batch-window edges get inflated importance -> the period-16 ripple
    assert edge > 1.5 * mid, f"expected batch-edge inflation (bug), got edge={edge:.3f} mid={mid:.3f}"


def test_fixed_permutation_removes_artifact():
    imp = _measure_per_level_importance(order="level_major", within_group=True, bs=16)
    # same-level permutation only probes the (t-independent) per-image variation -> FLAT in t
    cv = imp.std() / imp.mean()
    assert cv < 0.15, f"fixed measurement should be flat across levels, got CV={cv:.3f}"
    edge, mid = _edge_vs_mid(imp, period=16)
    assert 0.7 < edge / mid < 1.3, f"no period-16 structure expected, got edge/mid={edge/mid:.3f}"


# --------------------------------------------------------------------------- #
# 4. n_eff aggregation is unbiased for equal importance (task-requested check) #
# --------------------------------------------------------------------------- #
def _neff_for_equal_importance(n_levels, n_bins, n_groups=64):
    """Equal importance at every (group, level): n_eff MUST be flat (== n_groups)
    regardless of whether n_bins divides n_levels (rules out a binning-mass bug)."""
    base = BinStats(num_bins=n_bins)
    gstats = [BinStats(num_bins=n_bins) for _ in range(n_groups)]
    for lev in range(n_levels):
        bid = level_to_bin(torch.tensor([lev]), n_levels, n_bins)
        base.update(bid, torch.zeros(1))
        for g in range(n_groups):
            gstats[g].update(bid, torch.ones(1))     # identical importance everywhere
    bmean = base.mean()
    delta = torch.stack([torch.clamp(gs.mean() - bmean, min=0.0) for gs in gstats])
    um = compute_usage_metrics(delta, bmean, {f"h{i}": 1 for i in range(n_groups)},
                               [f"h{i}" for i in range(n_groups)], compute_group_correlation=False)
    return um["n_eff"].numpy()


def test_neff_flat_for_equal_importance_regardless_of_bin_ratio():
    for n_bins in (16, 20, 32, 7):        # 20 and 7 do NOT divide 64
        neff = _neff_for_equal_importance(64, n_bins)
        assert np.allclose(neff, 64.0, atol=1e-6), (
            f"n_eff must be flat (=64) for equal importance at n_bins={n_bins}; got {neff}"
        )
