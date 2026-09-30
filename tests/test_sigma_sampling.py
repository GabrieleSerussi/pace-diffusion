import os, sys, types
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from train_phase_students import (
    sample_phase_sigmas,
    sample_phase_sigmas_lognormal,
    draw_phase_sigmas,
    _sigma_at_fraction,
)

SIGMA_MIN, SIGMA_MAX, RHO = 0.002, 80.0, 7.0
NUM_BINS = 20


def _args(sigma_sampling="loguniform", p_mean=-1.2, p_std=1.2):
    return types.SimpleNamespace(
        sigma_sampling=sigma_sampling, p_mean=p_mean, p_std=p_std,
        sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, rho=RHO,
    )


def test_lognormal_global_phase_bounds():
    # global phase = full [0, num_bins] range -> [lo, hi] == [sigma_min, sigma_max].
    torch.manual_seed(0)
    sigma = sample_phase_sigmas_lognormal(
        batch_size=4096, phase_start=0, phase_end=NUM_BINS, num_bins=NUM_BINS,
        sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, rho=RHO,
        p_mean=-1.2, p_std=1.2, device=torch.device("cpu"),
    )
    assert sigma.shape == (4096,)
    assert bool((sigma >= SIGMA_MIN).all())
    assert bool((sigma <= SIGMA_MAX).all())
    # Not degenerate (a real spread of values, not all clamped to one edge).
    assert sigma.std() > 0


def test_lognormal_deterministic_given_rng_state():
    torch.manual_seed(1234)
    state = torch.get_rng_state()
    out1 = sample_phase_sigmas_lognormal(
        batch_size=256, phase_start=0, phase_end=NUM_BINS, num_bins=NUM_BINS,
        sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, rho=RHO,
        p_mean=-1.2, p_std=1.2, device=torch.device("cpu"),
    )
    torch.set_rng_state(state)
    out2 = sample_phase_sigmas_lognormal(
        batch_size=256, phase_start=0, phase_end=NUM_BINS, num_bins=NUM_BINS,
        sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, rho=RHO,
        p_mean=-1.2, p_std=1.2, device=torch.device("cpu"),
    )
    assert torch.equal(out1, out2)


def test_lognormal_respects_narrow_phase_truncation():
    # A narrow low-noise phase near the end of the bin range: [lo, hi] is a small
    # sub-interval of [sigma_min, sigma_max]. With EDM's default P_mean=-1.2 most
    # mass sits well above this band, so this genuinely exercises the
    # reject-and-redraw loop (not just the initial global clip).
    phase_start, phase_end = NUM_BINS - 1, NUM_BINS  # last bin only
    lo = _sigma_at_fraction(phase_end / NUM_BINS, SIGMA_MIN, SIGMA_MAX, RHO)
    hi = _sigma_at_fraction(phase_start / NUM_BINS, SIGMA_MIN, SIGMA_MAX, RHO)
    lo, hi = min(lo, hi), max(lo, hi)
    assert hi < SIGMA_MAX  # confirm this is a genuine sub-range, not the full one

    torch.manual_seed(7)
    sigma = sample_phase_sigmas_lognormal(
        batch_size=2048, phase_start=phase_start, phase_end=phase_end, num_bins=NUM_BINS,
        sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, rho=RHO,
        p_mean=-1.2, p_std=1.2, device=torch.device("cpu"),
    )
    assert bool((sigma >= lo - 1e-6).all())
    assert bool((sigma <= hi + 1e-6).all())


def test_lognormal_matches_loguniform_bounds_for_a_middle_phase():
    # Cross-check against the existing (unchanged) log-uniform sampler's own bin
    # math: both must agree on the [lo, hi] bounds for the same phase.
    phase_start, phase_end = 8, 16
    torch.manual_seed(0)
    uni = sample_phase_sigmas(
        batch_size=4096, phase_start=phase_start, phase_end=phase_end, num_bins=NUM_BINS,
        sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, rho=RHO, device=torch.device("cpu"),
    )
    lognorm = sample_phase_sigmas_lognormal(
        batch_size=4096, phase_start=phase_start, phase_end=phase_end, num_bins=NUM_BINS,
        sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, rho=RHO,
        p_mean=-1.2, p_std=1.2, device=torch.device("cpu"),
    )
    lo, hi = float(uni.min()), float(uni.max())
    assert bool((lognorm >= lo - 1e-4).all())
    assert bool((lognorm <= hi + 1e-4).all())


def test_draw_phase_sigmas_default_is_byte_identical_to_loguniform():
    # Default --sigma_sampling (loguniform, or the attribute absent entirely via
    # getattr fallback) must reproduce sample_phase_sigmas exactly, consuming the
    # SAME RNG draws -> existing studies stay byte-identical.
    torch.manual_seed(42)
    direct = sample_phase_sigmas(
        batch_size=64, phase_start=0, phase_end=NUM_BINS, num_bins=NUM_BINS,
        sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, rho=RHO, device=torch.device("cpu"),
    )
    torch.manual_seed(42)
    via_dispatch = draw_phase_sigmas(
        _args(sigma_sampling="loguniform"), batch_size=64,
        phase_start=0, phase_end=NUM_BINS, num_bins=NUM_BINS, device=torch.device("cpu"),
    )
    assert torch.equal(direct, via_dispatch)

    # args with no sigma_sampling attribute at all (legacy call sites / old
    # checkpoints' saved args) must fall back to the same default behavior.
    torch.manual_seed(42)
    legacy_args = types.SimpleNamespace(sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, rho=RHO)
    via_missing_attr = draw_phase_sigmas(
        legacy_args, batch_size=64,
        phase_start=0, phase_end=NUM_BINS, num_bins=NUM_BINS, device=torch.device("cpu"),
    )
    assert torch.equal(direct, via_missing_attr)


def test_draw_phase_sigmas_lognormal_routes_to_lognormal_sampler():
    torch.manual_seed(3)
    expected = sample_phase_sigmas_lognormal(
        batch_size=32, phase_start=0, phase_end=NUM_BINS, num_bins=NUM_BINS,
        sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, rho=RHO,
        p_mean=-1.0, p_std=0.9, device=torch.device("cpu"),
    )
    torch.manual_seed(3)
    got = draw_phase_sigmas(
        _args(sigma_sampling="lognormal", p_mean=-1.0, p_std=0.9), batch_size=32,
        phase_start=0, phase_end=NUM_BINS, num_bins=NUM_BINS, device=torch.device("cpu"),
    )
    assert torch.equal(expected, got)
