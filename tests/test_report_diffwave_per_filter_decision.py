from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.paper.report_diffwave_per_filter_decision import build_report


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_tree(tmp_path: Path) -> Path:
    root = tmp_path / "residual_only"
    results = {
        "scope": {
            "source_group_count": 6,
            "selected_group_count": 4,
            "excluded_group_count": 2,
        },
        "timestep_bin_members": [[3], [2], [1], [0]],
        "n_eff": [2.0, 3.0, 4.0, 5.0],
        "positive_delta_mass_fraction_of_source": [0.1, 0.2, 0.3, 0.4],
        "total_positive_delta_mass_fraction_of_source": 0.25,
        "source_analysis_basis_sha256": "source-basis",
        "profile_fingerprint_sha256": "profile-fingerprint",
    }
    results_path = root / "results.json"
    _write(results_path, results)
    results_sha = _sha256(results_path)
    _write(
        root / "_SUCCESS.json",
        {"passed": True, "results_sha256": results_sha},
    )

    grouping_specs = {
        "grouping_spearman_auto": (
            "matrix_spearman_cross_penalty",
            "spearman",
            [0, 4],
            "maximize_cross_penalty_total_score",
        ),
        "grouping_pearson_auto": (
            "matrix_correlation_cross_penalty",
            "pearson",
            [0, 2, 4],
            "maximize_cross_penalty_total_score",
        ),
        "grouping_pearson_k3_min2": (
            "matrix_correlation_cross_penalty",
            "pearson",
            [0, 1, 2, 4],
            "fixed",
        ),
        "grouping_spearman_k3_min2": (
            "matrix_spearman_cross_penalty",
            "spearman",
            [0, 1, 2, 4],
            "fixed",
        ),
    }
    grouping_paths: dict[str, Path] = {}
    for directory, (objective, metric, boundaries, policy) in grouping_specs.items():
        path = root / directory / "timestep_grouping.json"
        grouping_paths[directory] = path
        _write(
            path,
            {
                "builtin_cost": objective,
                "pairwise_metric": metric,
                "min_block_size": 1,
                "num_blocks": len(boundaries) - 1,
                "boundaries": boundaries,
                "total_score": 1.0,
                "cross_block_lambda": (
                    0.0 if directory == "grouping_pearson_k3_min2" else 0.02
                ),
                "num_blocks_selection": {
                    "policy": policy,
                    "candidates": [
                        {"num_blocks": len(boundaries) - 1, "total_score": 1.0}
                    ],
                },
                "source_profile": {"results_sha256": results_sha},
            },
        )

    allocation_specs = {
        "allocation_geomean_mean_spearman_auto": "grouping_spearman_auto",
        "allocation_geomean_mean_pearson_auto": "grouping_pearson_auto",
        "allocation_geomean_mean_pearson_k3_min2": "grouping_pearson_k3_min2",
        "allocation_geomean_mean_spearman_k3_min2": "grouping_spearman_k3_min2",
    }
    for directory, grouping_dir in allocation_specs.items():
        boundaries = json.loads(grouping_paths[grouping_dir].read_text())["boundaries"]
        count = len(boundaries) - 1
        _write(
            root / directory / "combined_blockwise.json",
            {
                "student_variant": "combined_blockwise",
                "allocation_group_scope": "allocatable",
                "score_reduction": "mean",
                "timestep_grouping_path": str(grouping_paths[grouping_dir]),
                "timestep_blocks": [
                    [start, end]
                    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
                ],
                "score_sources": {
                    "block_capacity_scores": {
                        "metric": "delta_p_eff_geomean",
                        "group_scope": "allocatable",
                        "reduction": "mean",
                        "reduction_protocol": "duration_neutral_mean",
                        "p_eff_source": "recomputed_from_allocatable_groups",
                    }
                },
                "target_budget_plan": {
                    "total_student_system_budget": 12.0,
                    "block_budgets": [12.0 / count] * count,
                },
                "source_profile": {"results_sha256": results_sha},
            },
        )
    pearson_grouping = grouping_paths["grouping_pearson_k3_min2"]
    pearson_allocation_dir = root / "allocation_geomean_mean_pearson_k3_min2"
    blockwise = json.loads((pearson_allocation_dir / "combined_blockwise.json").read_text())
    layerwise = json.loads(json.dumps(blockwise))
    layerwise["student_variant"] = "combined_layerwise"
    layerwise["timestep_grouping_path"] = str(pearson_grouping)
    layerwise["target_budget_plan"]["layer_budgets"] = [[1.0 / 9.0] * 36] * 3
    _write(pearson_allocation_dir / "combined_layerwise.json", layerwise)
    _write(
        pearson_allocation_dir / "summary.json",
        {
            "student_variants": ["combined_blockwise", "combined_layerwise"],
            "source_profile": {"results_sha256": results_sha},
        },
    )
    for filename in (
        "timestep_grouping_matrix.png",
        "timestep_grouping_n_eff.png",
        "timestep_grouping_cost_curve.png",
    ):
        path = root / "grouping_pearson_k3_min2" / filename
        path.write_bytes(b"synthetic png fixture")
    return root


def test_build_report_makes_fixed_pearson_k3_canonical(
    tmp_path: Path,
) -> None:
    report = build_report(_fixture_tree(tmp_path))

    assert report["status"] == "passed"
    assert report["decision"]["phase_specialization_supported"] is True
    assert report["decision"]["three_phase_structure_supported"] is True
    assert report["decision"]["exact_student_count_optimality_established"] is False
    assert report["decision"]["canonical_grouping"] == "pearson_fixed_k3"
    assert report["decision"]["canonical_num_phases"] == 3
    assert report["decision"]["canonical_allocation"] == "pearson_fixed_k3"
    assert report["allocations"]["spearman_fixed_k3_sensitivity"][
        "block_budget_fractions"
    ] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert report["groupings"]["pearson_auto"]["phases"][0]["label"] == "t=3..2"
    assert report["groupings"]["pearson_fixed_k3"]["cross_block_lambda"] == 0.0
    layerwise = report["canonical_layerwise_allocation"]
    assert layerwise["num_phases"] == 3
    assert layerwise["num_residual_blocks"] == 36
    assert layerwise["phase_budget_sums"] == pytest.approx([4.0, 4.0, 4.0])


def test_build_report_rejects_all_group_allocation_scope(tmp_path: Path) -> None:
    root = _fixture_tree(tmp_path)
    path = (
        root
        / "allocation_geomean_mean_spearman_auto"
        / "combined_blockwise.json"
    )
    payload = json.loads(path.read_text())
    payload["allocation_group_scope"] = "all"
    _write(path, payload)

    with pytest.raises(ValueError, match="allocation_group_scope='allocatable'"):
        build_report(root)
