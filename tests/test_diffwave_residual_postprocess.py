import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pace.diffwave_residual_postprocess import (
    RESIDUAL_ANALYSIS_FORMAT,
    RESIDUAL_METRICS_FORMAT,
    average_rank_columns,
    compute_residual_filter_view,
    postprocess_residual_filter_results,
    timestep_pearson_correlation,
)


def _fixture_results() -> dict:
    names = ["stem.f0", "res.0.f0", "res.0.f1", "head.f0"]
    signed = np.asarray(
        [
            [10.0, 10.0, 10.0],
            [1.0, -2.0, 1.0],
            [3.0, 2.0, -1.0],
            [100.0, 100.0, 100.0],
        ],
        dtype=np.float64,
    )
    positive = np.maximum(signed, 0.0)
    return {
        "format": "diffdist_diffwave_parameter_analysis_v1",
        "ablation_protocol": {
            "format": "test_protocol_v1",
            "protocol_id": "paired_test",
        },
        "group_names": names,
        "group_allocatable": {
            "stem.f0": False,
            "res.0.f0": True,
            "res.0.f1": True,
            "head.f0": False,
        },
        "group_param_counts": {
            "stem.f0": 1,
            "res.0.f0": 2,
            "res.0.f1": 4,
            "head.f0": 1,
        },
        "group_edm_proxy_param_counts": {
            "stem.f0": 1,
            "res.0.f0": 3,
            "res.0.f1": 5,
            "head.f0": 1,
        },
        "group_module_paths": {name: name.rsplit(".", 1)[0] for name in names},
        "group_structural_keys": {name: name.rsplit(".", 1)[0] for name in names},
        "group_stage_keys": {
            "stem.f0": "stem",
            "res.0.f0": "residual_block_00",
            "res.0.f1": "residual_block_00",
            "head.f0": "head",
        },
        "group_allocation_structural_keys": {
            "stem.f0": None,
            "res.0.f0": "residual_block_00",
            "res.0.f1": "residual_block_00",
            "head.f0": None,
        },
        "group_filter_indices": {
            "stem.f0": 0,
            "res.0.f0": 0,
            "res.0.f1": 1,
            "head.f0": 0,
        },
        "baseline_mean": [2.0, 4.0, 1.0],
        "baseline_stderr": [0.1, 0.1, 0.1],
        "baseline_count": [4, 4, 4],
        "signed_delta_stack": signed.tolist(),
        "delta_stack": positive.tolist(),
        "positive_delta_mass": positive.sum(axis=0).tolist(),
        "timestep_bin_labels": ["2", "1", "0"],
        "model_info": {"model_family": "diffwave", "total_parameter_count": 8},
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_average_rank_columns_uses_average_tie_ranks():
    values = np.asarray(
        [
            [1.0, 4.0],
            [1.0, 2.0],
            [3.0, 2.0],
            [5.0, 1.0],
        ]
    )

    ranked = average_rank_columns(values)

    assert np.array_equal(
        ranked,
        np.asarray(
            [
                [1.5, 4.0],
                [1.5, 2.5],
                [3.0, 2.5],
                [4.0, 1.0],
            ]
        ),
    )
    assert np.array_equal(
        timestep_pearson_correlation(ranked),
        timestep_pearson_correlation(average_rank_columns(values)),
    )


def test_residual_view_recomputes_metrics_only_over_allocatable_filters():
    results = _fixture_results()

    view = compute_residual_filter_view(
        results,
        expected_allocatable_count=2,
        top_filter_count=2,
    )

    assert view.group_names == ("res.0.f0", "res.0.f1")
    assert np.array_equal(view.source_group_indices, np.asarray([1, 2]))
    assert np.array_equal(
        view.signed_delta_stack,
        np.asarray([[1.0, -2.0, 1.0], [3.0, 2.0, -1.0]]),
    )
    assert np.array_equal(
        view.delta_stack,
        np.asarray([[1.0, 0.0, 1.0], [3.0, 2.0, 0.0]]),
    )
    assert np.allclose(view.weights, np.asarray([[0.25, 0.0, 1.0], [0.75, 1.0, 0.0]]))
    assert np.allclose(view.n_eff, np.asarray([1.6, 1.0, 1.0]))
    assert np.allclose(view.n_eff_fraction, np.asarray([0.8, 0.5, 0.5]))
    assert np.allclose(view.p_eff, np.asarray([3.5, 4.0, 2.0]))
    assert np.allclose(view.p_eff_edm_proxy, np.asarray([4.5, 5.0, 3.0]))
    assert np.allclose(view.signed_delta_positive_fraction, [1.0, 0.5, 0.5])
    assert np.allclose(view.signed_delta_negative_fraction, [0.0, 0.5, 0.5])
    assert np.array_equal(
        view.timestep_correlation_spearman,
        timestep_pearson_correlation(view.ranked_relative_delta_stack),
    )
    assert view.top_filter_indices.tolist() == [1, 0]


def test_residual_view_enforces_scope_count_and_positive_source_consistency():
    results = _fixture_results()
    with pytest.raises(ValueError, match="selected 2 filters; expected 3"):
        compute_residual_filter_view(results, expected_allocatable_count=3)

    results["delta_stack"][1][0] = 99.0
    with pytest.raises(ValueError, match="not the positive part"):
        compute_residual_filter_view(results, expected_allocatable_count=2)


def test_postprocessor_is_non_mutating_and_hash_binds_all_artifacts(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_path = source_dir / "results.json"
    source_path.write_text(json.dumps(_fixture_results(), indent=2) + "\n")
    source_bytes = source_path.read_bytes()
    output_dir = tmp_path / "derived"

    success = postprocess_residual_filter_results(
        source_path,
        output_dir=output_dir,
        expected_allocatable_count=2,
        top_filter_count=2,
        save_plots=True,
        invocation_argv=["postprocess", "--results", str(source_path)],
    )

    assert source_path.read_bytes() == source_bytes
    assert success["source_profile"]["results_sha256"] == hashlib.sha256(
        source_bytes
    ).hexdigest()
    assert success["selected_group_count"] == 2
    assert success["artifact_count"] == 7
    assert (output_dir / "_SUCCESS.json").is_file()
    for filename, digest in success["artifact_sha256"].items():
        assert _sha256(output_dir / filename) == digest

    derived = json.loads((output_dir / "results.json").read_text())
    assert derived["format"] == RESIDUAL_ANALYSIS_FORMAT
    assert derived["scope"]["selection_field"] == "group_allocatable"
    assert derived["scope"]["selected_source_group_indices"] == [1, 2]
    assert derived["group_names"] == ["res.0.f0", "res.0.f1"]
    assert all(derived["group_allocatable"].values())
    assert derived["C_timesteps"] == derived["C_timesteps_pearson"]
    assert derived["top_filters"][0]["group_name"] == "res.0.f1"

    metrics = torch.load(output_dir / "metrics.pt", map_location="cpu", weights_only=True)
    assert metrics["format"] == RESIDUAL_METRICS_FORMAT
    assert metrics["signed_delta_stack"].shape == (2, 3)
    assert torch.allclose(
        metrics["C_timesteps_spearman"],
        torch.corrcoef(metrics["ranked_relative_delta_stack"].T),
        atol=1e-15,
        rtol=1e-15,
    )


def test_default_output_is_beside_resolved_symlink_source(tmp_path):
    real_dir = tmp_path / "large-volume" / "profile"
    real_dir.mkdir(parents=True)
    (real_dir / "results.json").write_text(json.dumps(_fixture_results()))
    link = tmp_path / "per_filter"
    link.symlink_to(real_dir, target_is_directory=True)

    success = postprocess_residual_filter_results(
        link,
        expected_allocatable_count=2,
        top_filter_count=1,
        save_plots=False,
    )

    assert Path(success["output_dir"]) == real_dir / "residual_only"
    assert not (tmp_path / "residual_only").exists()
