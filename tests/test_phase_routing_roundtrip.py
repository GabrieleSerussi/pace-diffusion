"""Train <-> eval phase-routing ROUND-TRIP consistency, for the DDPM and EDM families.

The invariant under test is the one that ties training to sampling: a noise level
that the trainer handed to phase ``p``'s specialist must, at eval time, be routed
back to phase ``p``'s specialist. If the two sides disagree, every affected
sampling step is evaluated by the *wrong* specialist -- silently, with no crash
and no log line, only a worse FID.

  family  trainer draw                    eval router
  ------  ------------------------------  -------------------------------------
  DDPM    ``sample_phase_timesteps``      ``evaluate_students.timestep_to_phase_index``
  EDM     ``sample_phase_sigmas``         ``evaluate_students._sigma_to_phase``

This is the regression that catches the DDPM router's ``round()``-instead-of-
``floor()`` bug (a systematic half-bin shift: 47.5% of individual timesteps got
the wrong bin, and wherever that crossed a phase boundary the DiT-XL composite
sampler used the neighbouring phase's specialist).

Known, benign trainer-side wrinkle: ``sample_phase_timesteps`` draws from the
CLOSED range ``[t_low, t_high]``, so consecutive phases share exactly one
timestep at each bin edge (see
``test_ddpm_bin_edge_timestep_is_shared_by_both_neighbours``). The router must
pick one of the two -- it cannot satisfy both -- so those single shared values
are the only tolerated ambiguity anywhere below.

CPU-only; no checkpoints / GPU needed.
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from evaluate_students import (  # noqa: E402
    _sigma_to_phase,
    timestep_to_phase_index,
)
from train_phase_students import (  # noqa: E402
    _sigma_at_fraction,
    sample_phase_sigmas,
    sample_phase_sigmas_lognormal,
    sample_phase_timesteps,
)

CPU = torch.device("cpu")
NUM_BINS = 20
NUM_TIMESTEPS = 1000
SIGMA_MIN, SIGMA_MAX, RHO = 0.002, 80.0, 7.0

# Boundary sets exercised everywhere: the single-phase "global" control (routing
# immune), the real DiT-XL 2-phase split, a 4-phase split, and two extreme
# 1-bin-wide-first/last-phase splits.
BOUNDARY_SETS = [
    [0, NUM_BINS],           # global: one phase, must be routing-immune
    [0, 14, NUM_BINS],       # the DiT-XL production split
    [0, 4, 8, 16, NUM_BINS],
    [0, 1, NUM_BINS],        # phase 0 = highest-noise bin only
    [0, 19, NUM_BINS],       # phase 1 = lowest-noise bin only
]


# ---------------------------------------------------------------------------
# Helpers: re-derivation of the trainer's per-phase noise-level support
# ---------------------------------------------------------------------------

def _trainer_t_bounds(phase_start, phase_end, num_bins=NUM_BINS, num_timesteps=NUM_TIMESTEPS):
    """Inclusive ``[t_low, t_high]`` support of ``sample_phase_timesteps``.

    Mirrors the trainer formula; ``test_trainer_t_bounds_helper_matches_sampler``
    keeps this copy honest against the real sampler.
    """
    t_max = num_timesteps - 1
    t_low = max(0, int(round(t_max * (1.0 - phase_end / num_bins))))
    t_high = max(t_low + 1, min(t_max, int(round(t_max * (1.0 - phase_start / num_bins)))))
    return t_low, t_high


def _bins_trained_on_t(t, num_bins=NUM_BINS, num_timesteps=NUM_TIMESTEPS):
    """Every bin whose single-bin phase would have been trained on timestep ``t``."""
    out = set()
    for b in range(num_bins):
        lo, hi = _trainer_t_bounds(b, b + 1, num_bins, num_timesteps)
        if lo <= t <= hi:
            out.add(b)
    return out


def _bin_of_t(t, num_bins=NUM_BINS, num_timesteps=NUM_TIMESTEPS):
    """The router's BIN for ``t``: with one phase per bin, phase index == bin index."""
    per_bin = list(range(num_bins + 1))
    return timestep_to_phase_index(t, per_bin, num_bins, num_timesteps)


def _phase_of_bin(b, boundaries):
    for p in range(len(boundaries) - 1):
        if boundaries[p] <= b < boundaries[p + 1]:
            return p
    raise AssertionError(f"bin {b} outside {boundaries}")


def _shared_edge_timesteps(boundaries, num_bins=NUM_BINS, num_timesteps=NUM_TIMESTEPS):
    """The (one per phase edge) timesteps that two adjacent phases both train on."""
    return {
        _trainer_t_bounds(b, b, num_bins, num_timesteps)[0]
        for b in boundaries[1:-1]
    } | {
        int(round((num_timesteps - 1) * (1.0 - b / num_bins))) for b in boundaries[1:-1]
    }


# ---------------------------------------------------------------------------
# DDPM: the exhaustive, fully deterministic core property
# ---------------------------------------------------------------------------

def test_trainer_t_bounds_helper_matches_sampler():
    """The re-derived support must equal what ``sample_phase_timesteps`` actually draws."""
    torch.manual_seed(0)
    for ps, pe in [(0, NUM_BINS), (0, 14), (14, NUM_BINS), (0, 1), (19, NUM_BINS), (4, 8)]:
        t = sample_phase_timesteps(200_000, ps, pe, NUM_BINS, NUM_TIMESTEPS, CPU)
        lo, hi = _trainer_t_bounds(ps, pe)
        assert int(t.min()) == lo, (ps, pe, int(t.min()), lo)
        assert int(t.max()) == hi, (ps, pe, int(t.max()), hi)


def test_ddpm_every_timestep_routes_to_a_bin_that_was_trained_on_it():
    """EXHAUSTIVE: for all 1000 timesteps, the router's bin must be a bin whose
    trainer range actually contains that timestep.

    This is the tightest statement of the bug: a half-bin-shifted router sends
    e.g. t=960 to bin 1, but bin 1 was only ever trained on t in [899, 949].
    """
    bad = [t for t in range(NUM_TIMESTEPS) if _bin_of_t(t) not in _bins_trained_on_t(t)]
    assert bad == [], (
        f"{len(bad)}/{NUM_TIMESTEPS} timesteps route to a bin that was never "
        f"trained on them, e.g. t={bad[:8]} -> bins {[_bin_of_t(t) for t in bad[:8]]}"
    )


def test_ddpm_router_is_floor_not_round_at_bin_interior():
    """The concrete defect witness: t=960 sits in bin 0's trained range [949, 999];
    floor puts it in bin 0, round puts it in bin 1."""
    f = 1.0 - 960 / (NUM_TIMESTEPS - 1)
    assert math.floor(f * NUM_BINS) == 0
    assert round(f * NUM_BINS) == 1          # what the buggy router computed
    assert _bin_of_t(960) == 0
    # ... and the mis-binning becomes a mis-ROUTING as soon as bin 0/1 are split.
    assert timestep_to_phase_index(960, [0, 1, NUM_BINS], NUM_BINS, NUM_TIMESTEPS) == 0


def test_ddpm_bin_edge_timestep_is_shared_by_both_neighbours():
    """Documents the one tolerated ambiguity: ``sample_phase_timesteps`` uses a
    CLOSED range, so each bin edge timestep is trained by both adjacent bins and
    the router necessarily picks one."""
    shared = 0
    for b in range(1, NUM_BINS):
        t_edge = _trainer_t_bounds(b, b + 1)[1]      # bin b's high end
        assert t_edge == _trainer_t_bounds(b - 1, b)[0]  # == bin b-1's low end
        assert _bins_trained_on_t(t_edge) >= {b - 1, b}
        assert _bin_of_t(t_edge) in (b - 1, b)
        shared += 1
    assert shared == NUM_BINS - 1


def test_ddpm_extreme_timesteps():
    assert _bin_of_t(NUM_TIMESTEPS - 1) == 0              # highest noise -> bin 0
    assert _bin_of_t(NUM_TIMESTEPS + 500) == 0            # clamped above
    assert _bin_of_t(0) == NUM_BINS - 1                   # lowest noise -> last bin
    assert _bin_of_t(-7) == NUM_BINS - 1                  # clamped below


def test_ddpm_roundtrip_all_boundary_sets():
    """THE round-trip property: a timestep the trainer drew for phase p must route
    back to phase p (bin-edge ties excepted)."""
    torch.manual_seed(1234)
    for boundaries in BOUNDARY_SETS:
        shared = _shared_edge_timesteps(boundaries)
        for p in range(len(boundaries) - 1):
            t = sample_phase_timesteps(
                20_000, boundaries[p], boundaries[p + 1], NUM_BINS, NUM_TIMESTEPS, CPU,
            )
            wrong = {}
            for t_int in sorted(set(int(v) for v in t.tolist())):
                if t_int in shared:
                    continue
                got = timestep_to_phase_index(t_int, boundaries, NUM_BINS, NUM_TIMESTEPS)
                if got != p:
                    wrong[t_int] = got
            assert not wrong, (
                f"boundaries={boundaries} phase={p}: {len(wrong)} trained timesteps "
                f"route elsewhere, e.g. {dict(list(wrong.items())[:6])}"
            )


def test_ddpm_global_single_phase_is_routing_immune():
    """One phase => every timestep routes to phase 0 no matter what the binning
    does. (This is why the ``global`` DiT-XL variant is a valid positive control
    for a routing change: its FID must not move.)"""
    for t in range(0, NUM_TIMESTEPS, 7):
        assert timestep_to_phase_index(t, [0, NUM_BINS], NUM_BINS, NUM_TIMESTEPS) == 0
    for t in (-1, 0, 1, 499, 998, 999, 10_000):
        assert timestep_to_phase_index(t, [0, NUM_BINS], NUM_BINS, NUM_TIMESTEPS) == 0


def test_ddpm_roundtrip_other_bin_counts():
    """Same property at other bin counts / horizons (nothing may be hard-coded to 20/1000)."""
    torch.manual_seed(7)
    for num_bins, num_timesteps, boundaries in [
        (8, 1000, [0, 3, 8]),
        (10, 250, [0, 5, 10]),
        (4, 100, [0, 1, 2, 4]),
    ]:
        shared = _shared_edge_timesteps(boundaries, num_bins, num_timesteps)
        for p in range(len(boundaries) - 1):
            t = sample_phase_timesteps(
                20_000, boundaries[p], boundaries[p + 1], num_bins, num_timesteps, CPU,
            )
            for t_int in sorted(set(int(v) for v in t.tolist())):
                if t_int in shared:
                    continue
                got = timestep_to_phase_index(t_int, boundaries, num_bins, num_timesteps)
                assert got == p, (num_bins, num_timesteps, boundaries, p, t_int, got)


# ---------------------------------------------------------------------------
# EDM (sigma) round-trip
# ---------------------------------------------------------------------------

def _sig_phase(sigma, boundaries, num_bins=NUM_BINS):
    return _sigma_to_phase(float(sigma), boundaries, num_bins, SIGMA_MIN, SIGMA_MAX, RHO)


def test_edm_roundtrip_loguniform_all_boundary_sets():
    torch.manual_seed(0)
    for boundaries in BOUNDARY_SETS:
        for p in range(len(boundaries) - 1):
            sig = sample_phase_sigmas(
                8192, boundaries[p], boundaries[p + 1], NUM_BINS,
                SIGMA_MIN, SIGMA_MAX, RHO, CPU,
            )
            got = [_sig_phase(s, boundaries) for s in sig.tolist()]
            assert set(got) == {p}, (
                f"boundaries={boundaries} phase={p}: routed to {sorted(set(got))}"
            )


def test_edm_roundtrip_lognormal_all_boundary_sets():
    """The log-normal (EDM-native) draw is phase-truncated the same way, so it must
    round-trip identically -- except for stragglers that the sampler CLAMPS onto the
    phase's exact sigma edge after exhausting its resample budget (see
    ``test_edm_bin_edge_sigmas_are_an_unresolvable_float_tie``)."""
    torch.manual_seed(0)
    for boundaries in BOUNDARY_SETS:
        for p in range(len(boundaries) - 1):
            sig = sample_phase_sigmas_lognormal(
                4096, boundaries[p], boundaries[p + 1], NUM_BINS,
                SIGMA_MIN, SIGMA_MAX, RHO, -1.2, 1.2, CPU,
            )
            # The clamped stragglers land exactly on an edge, in float32 (the dtype
            # the sampler works in), so compare against the float32 image of each edge.
            edges = {
                float(torch.tensor(
                    _sigma_at_fraction(b / NUM_BINS, SIGMA_MIN, SIGMA_MAX, RHO),
                    dtype=torch.float32,
                ))
                for b in (boundaries[p], boundaries[p + 1])
            }
            got = [_sig_phase(s, boundaries) for s in sig.tolist() if s not in edges]
            assert set(got) == {p}, (
                f"boundaries={boundaries} phase={p}: routed to {sorted(set(got))}"
            )


def test_edm_extreme_sigmas():
    boundaries = [0, 14, NUM_BINS]
    assert _sig_phase(SIGMA_MAX, boundaries) == 0            # highest noise
    assert _sig_phase(SIGMA_MAX * 10, boundaries) == 0       # clamped above
    assert _sig_phase(SIGMA_MIN, boundaries) == 1            # lowest noise
    assert _sig_phase(SIGMA_MIN / 10, boundaries) == 1       # clamped below


def test_edm_bin_edge_sigmas_are_an_unresolvable_float_tie():
    """The EDM analogue of the DDPM shared-bin-edge tie: an exact bin edge sigma
    inverts to a schedule fraction of b/num_bins +/- 1e-15, so ``int()`` lands on
    b or b-1 depending on the last bit. Measure-zero (log-uniform draws never hit
    it) and NOT a systematic shift -- unlike the DDPM ``round()`` bug, which moved
    a whole half-bin of interior values."""
    per_bin = list(range(NUM_BINS + 1))
    for b in range(1, NUM_BINS):
        s_edge = _sigma_at_fraction(b / NUM_BINS, SIGMA_MIN, SIGMA_MAX, RHO)
        assert _sig_phase(s_edge, per_bin) in (b - 1, b)
    # Strictly INSIDE a bin there is no ambiguity at all: mid-bin sigmas must land
    # on their own bin for every bin.
    for b in range(NUM_BINS):
        s_mid = _sigma_at_fraction((b + 0.5) / NUM_BINS, SIGMA_MIN, SIGMA_MAX, RHO)
        assert _sig_phase(s_mid, per_bin) == b


def test_edm_global_single_phase_is_routing_immune():
    for s in [SIGMA_MIN, 0.01, 0.5, 5.0, SIGMA_MAX]:
        assert _sig_phase(s, [0, NUM_BINS]) == 0


# ---------------------------------------------------------------------------
# What the bug cost the DiT-XL composite sampler, as an executable statement
# ---------------------------------------------------------------------------

def test_dit_xl_production_split_switches_at_the_trainer_boundary():
    """For the production DiT-XL split (bins [0,14) / [14,20)) the router must hand
    over to the low-noise specialist at the trainer's bin-14 edge (t_max*0.3 =
    299.7, i.e. the last phase-1 timestep is 299 and t=300 -- the shared edge --
    goes to phase 0), NOT half a bin early at t=324.

    The buggy ``round()`` router put t in [300, 324] on phase 1: 25 of 1000
    timesteps, which at 250 DDPM steps is ~7 sampling steps evaluated by the wrong
    specialist on every single image.
    """
    boundaries = [0, 14, NUM_BINS]
    r = [timestep_to_phase_index(t, boundaries, NUM_BINS, NUM_TIMESTEPS)
         for t in range(NUM_TIMESTEPS)]
    assert max(t for t in range(NUM_TIMESTEPS) if r[t] == 1) == 299
    assert min(t for t in range(NUM_TIMESTEPS) if r[t] == 0) == 300
    assert all(v == 1 for v in r[:300])
    assert all(v == 0 for v in r[300:])
    # The window the bug got wrong, stated positively: all of it is phase 0 now.
    assert all(r[t] == 0 for t in range(300, 325))
