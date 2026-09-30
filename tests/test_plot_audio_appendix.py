from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.paper.plot_audio_appendix import _validate_metrics, _validate_source_inputs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_fixture(tmp_path: Path) -> tuple[Path, Path, dict, dict, dict, dict]:
    residual_root = tmp_path / "residual_only"
    residual_root.mkdir()
    results_path = residual_root / "results.json"
    results_path.write_text("{}\n", encoding="utf-8")
    results_hash = _sha256(results_path)

    stages = [f"residual_block_{index:02d}" for index in range(36)]
    phase_budgets = [3.0, 6.0, 9.0]
    layer_budgets = [[budget / 36.0] * 36 for budget in phase_budgets]
    allocation = {
        "student_variant": "combined_layerwise",
        "allocation_group_scope": "allocatable",
        "score_reduction": "mean",
        "timestep_blocks": [[0, 13], [13, 17], [17, 20]],
        "source_profile": {"results_sha256": results_hash},
        "score_sources": {
            "block_capacity_scores": {
                "metric": "delta_p_eff_geomean",
                "group_scope": "allocatable",
                "reduction": "mean",
                "reduction_protocol": "duration_neutral_mean",
                "p_eff_source": "recomputed_from_allocatable_groups",
            },
            "layer_capacity_scores": {
                "score_source": "delta_p_eff_geomean",
                "requested_group_scope": "allocatable",
                "group_scope": "all",
                "reduction": "mean",
                "reduction_protocol": "duration_neutral_mean",
                "structural_aggregation": {
                    "applied": True,
                    "output_stage_keys": stages,
                },
            },
        },
        "target_budget_plan": {
            "total_student_system_budget": 18.0,
            "block_budgets": phase_budgets,
            "layer_names": stages,
            "layer_budgets": layer_budgets,
        },
    }
    allocation_path = (
        residual_root
        / "allocation_geomean_mean_pearson_k3_min2"
        / "combined_layerwise.json"
    )
    allocation_path.parent.mkdir()
    allocation_path.write_text(json.dumps(allocation) + "\n", encoding="utf-8")
    allocation_relative = allocation_path.relative_to(residual_root).as_posix()

    results = {
        "format": "diffdist_diffwave_residual_filter_analysis_v1",
        "group_names": [f"filter_{index}" for index in range(36_864)],
    }
    decision = {
        "format": "diffdist_diffwave_per_filter_decision_v2",
        "status": "passed",
        "provenance": {
            "residual_results_sha256": results_hash,
            "input_artifact_sha256": {
                allocation_relative: _sha256(allocation_path),
            },
        },
        "canonical_artifacts": {
            "pearson_k3_layerwise_allocation": allocation_relative,
        },
        "groupings": {"pearson_fixed_k3": {"boundaries": [0, 13, 17, 20]}},
        "scope": {"allocation_parameter_budget": 18.0},
        "allocations": {
            "pearson_fixed_k3": {"block_budgets": phase_budgets},
        },
    }
    stability = {
        "format": "diffdist_diffwave_per_filter_stability_v1",
        "passed": True,
        "group_count": 256,
        "num_bins": 20,
        "inputs": [
            {"seed": seed, "pfi_seed": seed}
            for seed in range(3)
        ],
        "pairs": [
            {"first": "seed0", "second": "seed1"},
            {"first": "seed0", "second": "seed2"},
            {"first": "seed1", "second": "seed2"},
        ],
    }
    return results_path, allocation_path, results, decision, allocation, stability


def test_source_validation_accepts_bound_duration_neutral_allocation(
    tmp_path: Path,
) -> None:
    _validate_source_inputs(*_source_fixture(tmp_path))


def test_source_validation_rejects_mislabeled_sum_allocation(tmp_path: Path) -> None:
    fixture = list(_source_fixture(tmp_path))
    fixture[4] = {**fixture[4], "score_reduction": "sum"}

    with pytest.raises(ValueError, match="duration-neutral mean"):
        _validate_source_inputs(*fixture)


def test_source_validation_rejects_stale_allocation_file(tmp_path: Path) -> None:
    fixture = _source_fixture(tmp_path)
    fixture[1].write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not bound to the supplied layerwise"):
        _validate_source_inputs(*fixture)


def test_portable_metrics_bind_effective_support_to_residual_catalog() -> None:
    root = Path(__file__).resolve().parents[1]
    metrics = json.loads(
        (root / "artifacts" / "audio" / "audio_appendix_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    _validate_metrics(metrics)

    inconsistent = deepcopy(metrics)
    inconsistent["n_eff_fraction"][0] *= 2
    with pytest.raises(ValueError, match="n_eff_fraction is inconsistent"):
        _validate_metrics(inconsistent)
