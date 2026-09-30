"""Tests for scripts/bench_throughput.py's opt-in ``--model_spec``/``--rounds`` path
(which measures the run-to-run variance of throughput within one job).

CPU-only; no GPU needed: everything exercised here is either pure Python (JSON
parsing, the median/min/max/spread arithmetic, the composite-steps/sec formula) or
model *construction* (``build_spec_entry``, which is CPU-safe by design -- only
``time_forward``'s ``torch.cuda.synchronize`` actually requires CUDA, and that full
round-robin timing loop needs a GPU job instead). Building NarrowDiT students needs
facebookresearch/DiT (``external_dit``).
"""
import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import bench_throughput  # noqa: E402


def _tiny_phase_cfg(attn_inner):
    return {
        "hidden_size": 16, "depth": 1, "patch_size": 4, "in_channels": 3,
        "num_classes": 2, "input_size": 8, "learn_sigma": False,
        "per_block": [{"num_heads": 2, "attn_inner": attn_inner, "mlp_hidden": 16}],
    }


def _write_tiny_two_phase_plan(tmp_path):
    """A minimal 2-phase 'variant' plan (bins tile [0, 20], weights 0.6/0.4),
    mirroring the real plan files' shape closely enough to exercise
    build_spec_entry's arch_plan branch end-to-end on CPU."""
    plan = {
        "toy_variant": {
            "phases": [
                {"bins": [0, 12], "cfg": _tiny_phase_cfg(attn_inner=16)},
                {"bins": [12, 20], "cfg": _tiny_phase_cfg(attn_inner=32)},
            ],
        },
    }
    path = tmp_path / "toy_plan.json"
    path.write_text(json.dumps(plan))
    return str(path)


# ---------------------------------------------------------------------------
# load_model_spec_entries
# ---------------------------------------------------------------------------

def test_load_model_spec_entries_valid_list(tmp_path):
    spec = [
        {"label": "a", "arch_plan": "some_plan.json", "variant": "global"},
        {"label": "b", "arch_plan": "other_plan.json", "variant": "layerwise_capacity"},
    ]
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    entries = bench_throughput.load_model_spec_entries(str(path))
    assert entries == spec  # order preserved -- it IS the round-robin order


@pytest.mark.parametrize("bad_spec", [
    [],  # empty
    [{"arch_plan": "x.json", "variant": "global"}],  # missing label
    [{"label": "a", "variant": "global"}],  # missing arch_plan
    [{"label": "a", "arch_plan": "x.json"}],  # missing variant
    [{"label": "a"}],  # neither arch_plan nor variant
])
def test_load_model_spec_entries_rejects_malformed(tmp_path, bad_spec):
    path = tmp_path / "bad_spec.json"
    path.write_text(json.dumps(bad_spec))
    with pytest.raises(SystemExit):
        bench_throughput.load_model_spec_entries(str(path))


def test_load_model_spec_entries_rejects_non_list(tmp_path):
    path = tmp_path / "not_a_list.json"
    path.write_text(json.dumps({"label": "a"}))
    with pytest.raises(SystemExit):
        bench_throughput.load_model_spec_entries(str(path))


# ---------------------------------------------------------------------------
# build_spec_entry (CPU-safe: no timing, just construction)
# ---------------------------------------------------------------------------

@pytest.mark.external_dit
def test_build_spec_entry_arch_plan_two_phase_cpu(tmp_path):
    plan_path = _write_tiny_two_phase_plan(tmp_path)
    entry = {"label": "toy", "arch_plan": plan_path, "variant": "toy_variant"}
    built = bench_throughput.build_spec_entry(
        entry, attn_impl="manual", batch=2, num_bins=20, dev=torch.device("cpu"),
    )
    assert built["label"] == "toy"
    assert len(built["phases"]) == 2

    ph0, ph1 = built["phases"]
    assert ph0["bins"] == [0, 12] and ph1["bins"] == [12, 20]
    # weight = bin_width / num_bins, exactly bench_throughput's --arch_plan formula.
    assert ph0["weight"] == pytest.approx(12 / 20)
    assert ph1["weight"] == pytest.approx(8 / 20)
    assert built["total_params"] == ph0["params"] + ph1["params"]
    # Different attn_inner (16 vs 32) -> different param counts (not a copy-paste bug).
    assert ph0["params"] != ph1["params"]

    # Forward pass actually runs on CPU (construction is exercised for real, not just
    # inspected) and produces the right output shape.
    with torch.no_grad():
        out = ph0["model"](ph0["x"], ph0["t"], ph0["y"])
    assert tuple(out.shape) == (2, 3, 8, 8)
    assert torch.isfinite(out).all()




# ---------------------------------------------------------------------------
# composite_steps_per_sec / summarize_rounds (pure Python -- no torch needed)
# ---------------------------------------------------------------------------

def test_composite_steps_per_sec_matches_known_dit_xl_measurement():
    """Regression pin against a REAL measured value (a DiT-XL/2 throughput_sdpa.json
    record, blockwise_capacity: bins [0,14] w=0.7 lat=18.644ms,
    [14,20] w=0.3 lat=21.55ms -> steps/sec 51.24) -- confirms the factored-out
    composite formula agrees with the original --arch_plan loop's inline
    ``weighted_latency += w * lat; sps = 1/weighted_latency``."""
    weights = [0.7, 0.3]
    latencies_sec = [18.644e-3, 21.55e-3]
    sps = bench_throughput.composite_steps_per_sec(weights, latencies_sec)
    assert sps == pytest.approx(51.24, abs=0.01)


def test_composite_steps_per_sec_single_phase_is_plain_inverse_latency():
    assert bench_throughput.composite_steps_per_sec([1.0], [0.01]) == pytest.approx(100.0)


def test_summarize_rounds_median_min_max_spread():
    summary = bench_throughput.summarize_rounds([100.0, 90.0, 110.0])
    assert summary["median"] == pytest.approx(100.0)
    assert summary["min"] == pytest.approx(90.0)
    assert summary["max"] == pytest.approx(110.0)
    # spread = (max - min) / median * 100 = 20%
    assert summary["spread_pct_of_median"] == pytest.approx(20.0)


def test_summarize_rounds_identical_values_has_zero_spread():
    summary = bench_throughput.summarize_rounds([42.0, 42.0, 42.0])
    assert summary["spread_pct_of_median"] == 0.0


# ---------------------------------------------------------------------------
# argparse-level guards (reachable without CUDA -- checked before main() ever
# touches torch.device("cuda"))
# ---------------------------------------------------------------------------

def test_model_spec_requires_infer_mode(tmp_path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps([{"label": "a", "arch_plan": "p.json", "variant": "global"}]))
    argv = [
        "bench_throughput.py",
        "--model_spec", str(spec_path),
        "--mode", "train",
        "--out", str(tmp_path / "out.json"),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        with pytest.raises(SystemExit):
            bench_throughput.main()
    finally:
        sys.argv = old_argv


def test_model_spec_rejects_zero_rounds(tmp_path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps([{"label": "a", "arch_plan": "p.json", "variant": "global"}]))
    argv = [
        "bench_throughput.py",
        "--model_spec", str(spec_path),
        "--rounds", "0",
        "--out", str(tmp_path / "out.json"),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        with pytest.raises(SystemExit):
            bench_throughput.main()
    finally:
        sys.argv = old_argv


def test_arch_plan_required_unless_model_spec_given(tmp_path):
    argv = ["bench_throughput.py", "--out", str(tmp_path / "out.json")]
    old_argv = sys.argv
    sys.argv = argv
    try:
        with pytest.raises(SystemExit):
            bench_throughput.main()
    finally:
        sys.argv = old_argv


# ---------------------------------------------------------------------------
# nvidia_smi_snapshot: must never raise, regardless of whether nvidia-smi/a GPU
# is actually present on the box running the test suite.
# ---------------------------------------------------------------------------

def test_nvidia_smi_snapshot_never_raises():
    snap = bench_throughput.nvidia_smi_snapshot()
    assert snap is None or isinstance(snap, dict)
