"""
--latent_cache_dir: opt-in PERSISTENT (cross-restart) VAE latent cache, so
PerRankLatentCache's ~10-15 min (FFHQ) / ~50 min (Bedroom) re-encode no
longer repeats on every preemption+requeue (the datasets are static).

Covers, CPU-only with a tiny fake VAE + fake dataset (no network, no real
stabilityai/sd-vae-ft-mse download):

  * VAE ``encode()`` is deterministic in eval mode; only ``.sample()`` (the
    ``mean + std*eps`` reparameterization) is stochastic -- confirming WHY
    caching mu/logvar (not an already-sampled latent) is the right choice.
  * write -> load round-trip: mu/logvar survive the fp16 disk round-trip:
    a near-deterministic posterior (clamped-min logvar -> std ~ 0) recovers
    the encoder's mean up to fp16 precision.
  * sampling distribution equivalence: with a non-trivial logvar, repeated
    PersistentLatentCache loads reproduce the SAME Gaussian(mu, std) the
    on-the-fly VAE sample() would have produced -- fresh per construction,
    exactly like PerRankLatentCache's own "sample once per process start,
    reused for the whole run" cadence.
  * fingerprint mismatch (image count, VAE id, ...) -> cache not trusted.
  * a write a preemption interrupted (no DONE marker) -> not trusted, full
    rebuild, never a partial/incremental resume.
  * on-disk layout is canonical-order / world-size-agnostic: a cache BUILT
    with one shard count loads correctly under a DIFFERENT rank/world_size.
  * --hflip: REFUSED together with the persistent cache (RuntimeError at
    construction) -- latent-space flipping is a measured-bad approximation of
    pixel-space hflip; hflip=False (the default) is untouched.
  * default OFF (--latent_cache_dir unset) parses to None, and the ephemeral
    PerRankLatentCache class this leaves untouched still samples fresh
    latents on every construction (the exact behavior this feature is
    eliminating the COST of, not the fact of, when opted in).
"""
import json
import os
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from train_phase_students import (  # noqa: E402
    _VAE_SCALE_FACTOR,
    _dataset_identity_path,
    _fingerprint_hash,
    _latent_cache_done_path,
    _latent_cache_meta_path,
    _latent_cache_shard_paths,
    _atomic_write_json,
    build_persistent_latent_cache,
    latent_cache_fingerprint,
    latent_cache_is_complete,
    parse_args,
    PerRankLatentCache,
    PersistentLatentCache,
    resolve_latent_cache_dir,
    trusted_latent_cache_fingerprint,
)


# ---------------------------------------------------------------------------
# Fakes: tiny CPU-only VAE + image dataset, no network / real checkpoint.
# ---------------------------------------------------------------------------

class _LinearVAE(nn.Module):
    """Deterministic fake encoder (fixed random 1x1 conv + adaptive avg pool
    to a small latent grid) with a controllable constant logvar, exposing the
    same ``encode(imgs).latent_dist`` contract (a real
    diffusers.DiagonalGaussianDistribution) as AutoencoderKL."""

    def __init__(self, out_channels=4, latent_hw=4, logvar_value=0.0):
        super().__init__()
        self.proj = nn.Conv2d(3, out_channels, kernel_size=1, bias=False)
        with torch.no_grad():
            g = torch.Generator().manual_seed(1234)
            self.proj.weight.copy_(torch.randn(self.proj.weight.shape, generator=g) * 0.3)
        self.latent_hw = latent_hw
        self.logvar_value = logvar_value

    def encode(self, imgs):
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        feat = self.proj(imgs)
        mu = F.adaptive_avg_pool2d(feat, self.latent_hw)
        logvar = torch.full_like(mu, self.logvar_value)
        parameters = torch.cat([mu, logvar], dim=1)
        return types.SimpleNamespace(latent_dist=DiagonalGaussianDistribution(parameters))


class _AvgPoolVAE(nn.Module):
    """EXACTLY horizontal-flip-equivariant fake encoder: non-overlapping
    average pooling with no learned parameters and no padding, so
    avg_pool2d(flip(x)) == flip(avg_pool2d(x)) bit-for-bit whenever width is
    an exact multiple of the pool factor. Isolates whether
    PersistentLatentCache's "flip the sampled latent" --hflip shortcut is
    correct FOR a flip-equivariant VAE (real VAEs are only approximately
    so -- see PersistentLatentCache's docstring)."""

    def __init__(self, channels=4, factor=4, logvar_value=-30.0):
        super().__init__()
        self.channels = channels
        self.factor = factor
        self.logvar_value = logvar_value

    def encode(self, imgs):
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        pooled = F.avg_pool2d(imgs, kernel_size=self.factor, stride=self.factor)
        reps = -(-self.channels // imgs.shape[1])
        mu = pooled.repeat(1, reps, 1, 1)[:, : self.channels]
        logvar = torch.full_like(mu, self.logvar_value)
        parameters = torch.cat([mu, logvar], dim=1)
        return types.SimpleNamespace(latent_dist=DiagonalGaussianDistribution(parameters))


class _FakeImageDataset(torch.utils.data.Dataset):
    """Mirrors ImageFolderFlat's (image, -1) contract."""

    def __init__(self, n, size=8, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.imgs = torch.randn(n, 3, size, size, generator=g)

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        return self.imgs[idx], -1


class _IndexLabeledDataset(_FakeImageDataset):
    """Label = canonical dataset index -- lets a test verify EXACT round-robin
    coverage (no duplicates, no gaps) instead of just a count."""

    def __getitem__(self, idx):
        return self.imgs[idx], idx


def _fingerprint_for(ds, tmp_path, image_size=8, dataset="image_folder"):
    args = types.SimpleNamespace(dataset=dataset, image_root=str(tmp_path / "imgs"),
                                  data_root=str(tmp_path / "imgs"))
    return latent_cache_fingerprint(args, ds, image_size)


# ---------------------------------------------------------------------------
# VAE encode is deterministic in eval mode; only .sample() is stochastic.
# ---------------------------------------------------------------------------

def test_vae_encode_is_deterministic_only_sample_is_stochastic():
    torch.manual_seed(0)
    ds = _FakeImageDataset(n=3, size=8, seed=11)
    vae = _LinearVAE(logvar_value=0.3).eval()
    with torch.no_grad():
        d1 = vae.encode(ds.imgs).latent_dist
        d2 = vae.encode(ds.imgs).latent_dist
    assert torch.equal(d1.mean, d2.mean)
    assert torch.equal(d1.logvar, d2.logvar)
    assert not torch.equal(d1.sample(), d2.sample())


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------

def test_fingerprint_contains_required_dimensions(tmp_path):
    ds = _FakeImageDataset(n=5, size=8)
    fp = _fingerprint_for(ds, tmp_path, image_size=32)
    assert fp["image_count"] == 5
    assert fp["vae_id"] == "stabilityai/sd-vae-ft-mse"
    assert fp["image_size"] == 32
    assert fp["dataset_path"] == os.path.abspath(str(tmp_path / "imgs"))
    assert fp["dataset_kind"] == "image_folder"


def test_dataset_identity_path_uses_image_root_for_image_folder():
    args = types.SimpleNamespace(dataset="image_folder", image_root="relative/path")
    assert _dataset_identity_path(args) == os.path.abspath("relative/path")


def test_dataset_identity_path_uses_data_root_for_other_dataset_kinds():
    args = types.SimpleNamespace(dataset="imagenet1k_parquet", data_root="some/root")
    assert _dataset_identity_path(args) == os.path.abspath("some/root")


# ---------------------------------------------------------------------------
# Missing / fingerprint-mismatched / partial (no DONE marker) caches: never used.
# ---------------------------------------------------------------------------

def test_missing_cache_dir_is_not_complete(tmp_path):
    ds = _FakeImageDataset(n=5, size=8)
    fp = _fingerprint_for(ds, tmp_path)
    assert latent_cache_is_complete(str(tmp_path / "nope"), fp) is False


def test_fingerprint_mismatch_is_not_complete(tmp_path):
    ds = _FakeImageDataset(n=9, size=8)
    fp = _fingerprint_for(ds, tmp_path)
    vae = _LinearVAE()
    cache_dir = str(tmp_path / "cache")
    build_persistent_latent_cache(ds, vae, cache_dir, fp, rank=0, world_size=1,
                                   device=torch.device("cpu"), encode_batch_size=4, num_workers=0)
    assert latent_cache_is_complete(cache_dir, fp) is True

    other_ds = _FakeImageDataset(n=10, size=8)   # different image_count
    other_fp = _fingerprint_for(other_ds, tmp_path)
    assert latent_cache_is_complete(cache_dir, other_fp) is False

    vae_mismatch_fp = dict(fp)
    vae_mismatch_fp["vae_id"] = "some/other-vae"
    assert latent_cache_is_complete(cache_dir, vae_mismatch_fp) is False

    res_mismatch_fp = dict(fp)
    res_mismatch_fp["image_size"] = 999
    assert latent_cache_is_complete(cache_dir, res_mismatch_fp) is False


def test_partial_write_without_done_marker_is_not_complete_and_triggers_rebuild(tmp_path):
    ds = _FakeImageDataset(n=6, size=8)
    fp = _fingerprint_for(ds, tmp_path)
    cache_dir = str(tmp_path / "cache")
    os.makedirs(cache_dir)
    # Simulate a write a preemption interrupted: meta.json + shard files
    # present (as build_persistent_latent_cache writes them, in order), but
    # the DONE marker (written LAST) never landed.
    _atomic_write_json(_latent_cache_meta_path(cache_dir),
                        {"fingerprint": fp, "num_shards": 1, "shard_sizes": [6]})
    mu_path, logvar_path, labels_path = _latent_cache_shard_paths(cache_dir, 0)
    np.save(mu_path, np.zeros((6, 4, 4, 4), dtype=np.float16))
    np.save(logvar_path, np.zeros((6, 4, 4, 4), dtype=np.float16))
    np.save(labels_path, np.full((6,), -1, dtype=np.int64))

    assert os.path.isfile(_latent_cache_meta_path(cache_dir))
    assert not os.path.isfile(_latent_cache_done_path(cache_dir))
    assert latent_cache_is_complete(cache_dir, fp) is False

    # This is exactly the conditional train_phase_students.main() uses to
    # decide whether to load or (re)build -- a partial cache must route to
    # rebuild, never attempt to load it as-is.
    decision = "load" if latent_cache_is_complete(cache_dir, fp) else "rebuild"
    assert decision == "rebuild"


# ---------------------------------------------------------------------------
# Hardening item 7: fingerprint-keyed cache subdirectory, so a mismatched
# config (e.g. a smoke run with a reduced image_count) can never clobber an
# existing cache of a different shape at the same --latent_cache_dir.
# ---------------------------------------------------------------------------

def test_resolve_latent_cache_dir_fresh_path_uses_fingerprint_subdir(tmp_path):
    # No cache exists yet at all -- brand-new caches always land in a
    # fingerprint-keyed subdir, never directly at the flat base path.
    ds = _FakeImageDataset(n=5, size=8)
    fp = _fingerprint_for(ds, tmp_path)
    base = str(tmp_path / "cache")
    resolved = resolve_latent_cache_dir(base, fp)
    assert resolved == os.path.join(base, _fingerprint_hash(fp))
    assert resolved != base


def test_resolve_latent_cache_dir_recognizes_existing_flat_legacy_cache(tmp_path):
    # Backward compatibility: a cache already built (by code predating this
    # feature) directly at the flat path, matching the CURRENT fingerprint,
    # must be recognized and reused in place -- no migration required.
    ds = _FakeImageDataset(n=5, size=8)
    fp = _fingerprint_for(ds, tmp_path)
    vae = _LinearVAE()
    base = str(tmp_path / "cache")
    build_persistent_latent_cache(ds, vae, base, fp, rank=0, world_size=1,
                                   device=torch.device("cpu"), encode_batch_size=4, num_workers=0)
    assert resolve_latent_cache_dir(base, fp) == base


def test_resolve_latent_cache_dir_mismatched_fingerprint_never_touches_existing_cache(tmp_path):
    # THE incident this hardening item exists to prevent: a full-size cache
    # already lives at the flat base path; a differently-fingerprinted run
    # (e.g. a smoke test with a smaller image_count) must resolve to a
    # DIFFERENT directory, never rebuild/overwrite the existing one.
    full_ds = _FakeImageDataset(n=1000, size=8, seed=42)
    full_fp = _fingerprint_for(full_ds, tmp_path)
    vae = _LinearVAE()
    base = str(tmp_path / "cache")
    build_persistent_latent_cache(full_ds, vae, base, full_fp, rank=0, world_size=1,
                                   device=torch.device("cpu"), encode_batch_size=64, num_workers=0)
    mu_path, _, _ = _latent_cache_shard_paths(base, 0)
    original_mtime = os.path.getmtime(mu_path)
    original_meta = json.load(open(_latent_cache_meta_path(base)))

    smoke_ds = _FakeImageDataset(n=8, size=8, seed=99)  # different image_count -> different fingerprint
    smoke_fp = _fingerprint_for(smoke_ds, tmp_path)
    assert smoke_fp != full_fp

    resolved = resolve_latent_cache_dir(base, smoke_fp)
    assert resolved != base, "a mismatched fingerprint must NOT resolve to the existing cache's path"

    # Simulate what main() does next: build at the resolved (subdir) path.
    build_persistent_latent_cache(smoke_ds, vae, resolved, smoke_fp, rank=0, world_size=1,
                                   device=torch.device("cpu"), encode_batch_size=4, num_workers=0)

    # The ORIGINAL full-size cache must be completely untouched.
    assert os.path.getmtime(mu_path) == original_mtime
    assert json.load(open(_latent_cache_meta_path(base))) == original_meta
    assert latent_cache_is_complete(base, full_fp) is True
    # And the smoke cache is independently valid at its own subdir.
    assert latent_cache_is_complete(resolved, smoke_fp) is True


def test_fingerprint_hash_is_stable_and_key_order_independent():
    fp1 = {"a": 1, "b": 2}
    fp2 = {"b": 2, "a": 1}
    assert _fingerprint_hash(fp1) == _fingerprint_hash(fp2)


def test_fingerprint_hash_differs_for_different_fingerprints():
    assert _fingerprint_hash({"a": 1}) != _fingerprint_hash({"a": 2})


# ---------------------------------------------------------------------------
# Write -> load round-trip identity (mu/logvar survive the fp16 disk
# round-trip; near-deterministic posterior isolates this from sampling noise).
# ---------------------------------------------------------------------------

def test_persistent_cache_recovers_mean_up_to_fp16_precision_when_std_near_zero(tmp_path):
    torch.manual_seed(0)
    ds = _FakeImageDataset(n=13, size=8, seed=1)
    fp = _fingerprint_for(ds, tmp_path)
    vae = _LinearVAE(logvar_value=-30.0)  # min-clamped logvar -> std = exp(-15) ~ 3e-7, negligible
    cache_dir = str(tmp_path / "cache")
    build_persistent_latent_cache(ds, vae, cache_dir, fp, rank=0, world_size=1,
                                   device=torch.device("cpu"), encode_batch_size=4, num_workers=0)
    assert latent_cache_is_complete(cache_dir, fp)

    cached = PersistentLatentCache(cache_dir=cache_dir, rank=0, world_size=1)

    with torch.no_grad():
        expected = (vae.encode(ds.imgs).latent_dist.mean * _VAE_SCALE_FACTOR).to(dtype=torch.float16)

    assert torch.allclose(cached.latents.float(), expected.float(), atol=2e-3)
    assert torch.equal(cached.labels, torch.full((13,), -1, dtype=torch.long))
    assert cached.latents.dtype == torch.float16


# ---------------------------------------------------------------------------
# Sampling distribution equivalence: repeated loads reproduce the SAME
# Gaussian(mu, std) the on-the-fly VAE .sample() would have -- proving mu/
# logvar caching preserves stochasticity rather than silently freezing it.
# ---------------------------------------------------------------------------

def test_persistent_cache_sampling_matches_vae_posterior_distribution(tmp_path):
    ds = _FakeImageDataset(n=1, size=8, seed=2)
    fp = _fingerprint_for(ds, tmp_path)
    vae = _LinearVAE(logvar_value=0.0)  # std = 1
    cache_dir = str(tmp_path / "cache")
    build_persistent_latent_cache(ds, vae, cache_dir, fp, rank=0, world_size=1,
                                   device=torch.device("cpu"), encode_batch_size=4, num_workers=0)

    with torch.no_grad():
        expected_mean = vae.encode(ds.imgs).latent_dist.mean[0] * _VAE_SCALE_FACTOR

    torch.manual_seed(123)
    draws = torch.stack([
        PersistentLatentCache(cache_dir=cache_dir, rank=0, world_size=1).latents[0].float()
        for _ in range(400)
    ])
    empirical_mean = draws.mean(dim=0)
    empirical_std = draws.std(dim=0)
    # z = (mu + std_true*eps) * scale with std_true=1 -> std(z) == scale exactly.
    assert torch.allclose(empirical_mean, expected_mean, atol=0.15)
    assert torch.allclose(empirical_std, torch.full_like(empirical_std, _VAE_SCALE_FACTOR), atol=0.05)


def test_persistent_cache_draws_fresh_sample_every_construction(tmp_path):
    # Mirrors PerRankLatentCache's own cadence: fresh z each process start.
    ds = _FakeImageDataset(n=4, size=8, seed=3)
    fp = _fingerprint_for(ds, tmp_path)
    vae = _LinearVAE(logvar_value=0.0)
    cache_dir = str(tmp_path / "cache")
    build_persistent_latent_cache(ds, vae, cache_dir, fp, rank=0, world_size=1,
                                   device=torch.device("cpu"), encode_batch_size=4, num_workers=0)
    c1 = PersistentLatentCache(cache_dir=cache_dir, rank=0, world_size=1)
    c2 = PersistentLatentCache(cache_dir=cache_dir, rank=0, world_size=1)
    assert not torch.equal(c1.latents, c2.latents)


# ---------------------------------------------------------------------------
# Canonical order / world-size-agnostic: a cache built with one shard count
# loads correctly (every image exactly once) under a DIFFERENT world_size.
# ---------------------------------------------------------------------------

def test_persistent_cache_is_world_size_agnostic_between_build_and_load(tmp_path):
    n = 23
    ds = _IndexLabeledDataset(n=n, size=8, seed=4)
    fp = _fingerprint_for(ds, tmp_path)
    vae = _LinearVAE(logvar_value=-30.0)
    cache_dir = str(tmp_path / "cache")
    build_world_size = 3
    for r in range(build_world_size):
        build_persistent_latent_cache(ds, vae, cache_dir, fp, rank=r, world_size=build_world_size,
                                       device=torch.device("cpu"), encode_batch_size=4, num_workers=0)
    assert latent_cache_is_complete(cache_dir, fp)

    load_world_size = 2   # deliberately different from build_world_size
    all_labels = torch.cat([
        PersistentLatentCache(cache_dir=cache_dir, rank=r, world_size=load_world_size).labels
        for r in range(load_world_size)
    ])
    assert sorted(all_labels.tolist()) == list(range(n))


def test_persistent_cache_matches_direct_encode_per_index_after_world_size_change(tmp_path):
    # Stronger than the count check above: EVERY canonical index's recovered
    # mean matches a direct encode of that exact image (near-zero std isolates
    # this from sampling noise), regardless of which rank/world_size loaded it.
    n = 17
    ds = _IndexLabeledDataset(n=n, size=8, seed=5)
    fp = _fingerprint_for(ds, tmp_path)
    vae = _LinearVAE(logvar_value=-30.0)
    cache_dir = str(tmp_path / "cache")
    build_persistent_latent_cache(ds, vae, cache_dir, fp, rank=0, world_size=1,
                                   device=torch.device("cpu"), encode_batch_size=8, num_workers=0)

    with torch.no_grad():
        expected = (vae.encode(ds.imgs).latent_dist.mean * _VAE_SCALE_FACTOR).to(torch.float16)

    load_world_size = 4
    for r in range(load_world_size):
        c = PersistentLatentCache(cache_dir=cache_dir, rank=r, world_size=load_world_size)
        for local_i, canonical_i in enumerate(c.labels.tolist()):
            assert torch.allclose(c.latents[local_i].float(), expected[canonical_i].float(), atol=2e-3)


# ---------------------------------------------------------------------------
# --hflip: REFUSED on the persistent cache. The cache holds post-encode
# mu/logvar only, and flipping the sampled latent is a measured-bad
# approximation of pixel-space hflip (the sd-vae is not reflection-
# equivariant), so constructing the cache with hflip=True raises -- even for
# a VAE that happens to be exactly flip-equivariant, and before any latent is
# handed to training. hflip=False stays untouched.
# ---------------------------------------------------------------------------

def test_persistent_cache_hflip_is_refused_even_for_equivariant_vae(tmp_path):
    torch.manual_seed(7)
    size, factor = 8, 4  # size % factor == 0 -> exactly flip-equivariant pooling
    ds = _FakeImageDataset(n=6, size=size, seed=6)
    fp = _fingerprint_for(ds, tmp_path, image_size=size)
    vae = _AvgPoolVAE(channels=4, factor=factor, logvar_value=-30.0)
    cache_dir = str(tmp_path / "cache")
    build_persistent_latent_cache(ds, vae, cache_dir, fp, rank=0, world_size=1,
                                   device=torch.device("cpu"), encode_batch_size=4, num_workers=0)

    with pytest.raises(RuntimeError, match="--hflip is incompatible with --latent_cache_dir"):
        PersistentLatentCache(cache_dir=cache_dir, rank=0, world_size=1, hflip=True)


def test_persistent_cache_hflip_refusal_leaves_cache_loadable_without_hflip(tmp_path):
    torch.manual_seed(0)
    ds = _FakeImageDataset(n=64, size=8, seed=8)
    fp = _fingerprint_for(ds, tmp_path)
    vae = _LinearVAE(logvar_value=-30.0)
    cache_dir = str(tmp_path / "cache")
    build_persistent_latent_cache(ds, vae, cache_dir, fp, rank=0, world_size=1,
                                   device=torch.device("cpu"), encode_batch_size=8, num_workers=0)
    with torch.no_grad():
        mu = (vae.encode(ds.imgs).latent_dist.mean * _VAE_SCALE_FACTOR).to(torch.float16)

    with pytest.raises(RuntimeError):
        PersistentLatentCache(cache_dir=cache_dir, rank=0, world_size=1, hflip=True)
    # The refusal happens at construction time and never touches the on-disk
    # cache: the same cache still loads (unflipped) with the default hflip=False.
    assert latent_cache_is_complete(cache_dir, fp)
    cached = PersistentLatentCache(cache_dir=cache_dir, rank=0, world_size=1)
    assert torch.allclose(cached.latents.float(), mu.float(), atol=2e-3)


def test_persistent_cache_hflip_default_false_matches_plain_encode(tmp_path):
    ds = _FakeImageDataset(n=10, size=8, seed=9)
    fp = _fingerprint_for(ds, tmp_path)
    vae = _LinearVAE(logvar_value=-30.0)
    cache_dir = str(tmp_path / "cache")
    build_persistent_latent_cache(ds, vae, cache_dir, fp, rank=0, world_size=1,
                                   device=torch.device("cpu"), encode_batch_size=4, num_workers=0)
    with torch.no_grad():
        expected = (vae.encode(ds.imgs).latent_dist.mean * _VAE_SCALE_FACTOR).to(torch.float16)
    cached = PersistentLatentCache(cache_dir=cache_dir, rank=0, world_size=1)  # hflip default False
    assert torch.allclose(cached.latents.float(), expected.float(), atol=2e-3)


# ---------------------------------------------------------------------------
# Default OFF (--latent_cache_dir unset) is byte-identical: parses to None,
# and PerRankLatentCache (the path taken when it's None) is untouched -- it
# still samples a fresh latent on every construction, exactly as before this
# feature existed. This feature removes the COST of that on requeue, opt-in,
# never the fact of it when not opted in.
# ---------------------------------------------------------------------------

def test_latent_cache_dir_defaults_to_none(monkeypatch, tmp_path):
    argv = [
        "prog", "--model_type", "dit_xl", "--teacher_checkpoint", "x",
        "--output_dir", str(tmp_path), "--dataset", "image_folder",
        "--image_root", str(tmp_path), "--grouping_json", "x.json",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = parse_args()
    assert args.latent_cache_dir is None


def test_per_rank_latent_cache_still_samples_fresh_each_construction():
    # Regression guard: the ephemeral (default OFF) path must remain exactly
    # as stochastic/expensive as before -- this feature only lets callers
    # OPT OUT of paying that cost on every restart, it must not change it.
    torch.manual_seed(0)
    ds = _FakeImageDataset(n=4, size=8, seed=10)
    vae = _LinearVAE(logvar_value=0.0)
    c1 = PerRankLatentCache(image_dataset=ds, vae=vae, rank=0, world_size=1,
                             device=torch.device("cpu"), encode_batch_size=4, num_workers=0)
    c2 = PerRankLatentCache(image_dataset=ds, vae=vae, rank=0, world_size=1,
                             device=torch.device("cpu"), encode_batch_size=4, num_workers=0)
    assert not torch.equal(c1.latents, c2.latents)


def test_trusted_fingerprint_from_complete_cache(tmp_path):
    """--latent_cache_trust: the fingerprint comes from meta.json of the single
    complete cache (flat dir or fingerprint-keyed subdir); an incomplete cache
    (no DONE marker) is never trusted."""
    fp = {"version": 1, "dataset_kind": "imagenet1k_parquet", "dataset_path": "/x",
          "image_count": 3, "vae_id": "v", "image_size": 256}
    sub = tmp_path / _fingerprint_hash(fp)
    sub.mkdir()
    with open(sub / "meta.json", "w") as f:
        json.dump({"fingerprint": fp}, f)
    (sub / "DONE").write_text("ok")
    assert trusted_latent_cache_fingerprint(str(tmp_path)) == fp
    assert resolve_latent_cache_dir(str(tmp_path), fp) == str(sub)
    assert latent_cache_is_complete(str(sub), fp)
    (sub / "DONE").unlink()
    with pytest.raises(FileNotFoundError):
        trusted_latent_cache_fingerprint(str(tmp_path))
