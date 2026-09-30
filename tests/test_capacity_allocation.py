import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pace.capacity_allocation import (
    CapacityAllocationConfig,
    ModelBudget,
    TimestepBlock,
    aggregate_group_scores_for_allocation,
    compute_delta_p_eff_geomean_metric,
    compute_delta_param_geomean_layer_scores,
    compute_original_model_budget,
    discover_evaluation_output,
    load_block_capacity_scores_from_results,
    load_capacity_scores,
    load_layer_capacity_scores_from_results,
    load_layer_capacity_scores_with_metadata_from_results,
    load_timestep_blocks_from_grouping,
    make_blockwise_capacity_budgets,
    make_global_budget,
    make_layerwise_capacity_budgets,
    make_shuffled_capacity_budgets,
    make_uniform_block_budgets,
    make_variant_capacity_budgets,
    normalize_capacity_budgets,
    results_group_expansion_weights,
    validate_budgets,
    validate_capacity_scores,
    validate_total_stored_budget,
)


def _canonical_name_digest(names):
    payload = json.dumps(sorted(names), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stratified_sampling(group_names, expansion_weights):
    module_counts = {}
    population_names = []
    probabilities = {}
    expansions = {}
    for name, raw_weight in zip(group_names, expansion_weights):
        module_name, _filter_index = name.rsplit(".filter_", 1)
        population_count = int(raw_weight)
        module_counts[module_name] = {
            "population_filter_count": population_count,
            "selected_filter_count": 1,
            "inclusion_probability": 1.0 / population_count,
            "expansion_weight": float(population_count),
        }
        probabilities[name] = 1.0 / population_count
        expansions[name] = float(population_count)
        population_names.extend(
            f"{module_name}.filter_{index}" for index in range(population_count)
        )
    return {
        "format": "diffdist_edm_filter_sampling_protocol_v1",
        "protocol_id": "per_module_hash_stratified_filter_sampling_v1",
        "mode": "stratified_module",
        "seed": 0,
        "filters_per_module": 1,
        "population_group_count": len(population_names),
        "selected_group_count": len(group_names),
        "population_module_count": len(module_counts),
        "selected_module_count": len(module_counts),
        "module_counts": module_counts,
        "population_sha256": _canonical_name_digest(population_names),
        "selection_sha256": _canonical_name_digest(group_names),
        "selected_group_names": list(group_names),
        "selected_group_inclusion_probabilities": probabilities,
        "selected_group_expansion_weights": expansions,
    }


class FakeParameter:
    def __init__(self, numel):
        self._numel = int(numel)

    def numel(self):
        return self._numel


class FakeModel:
    def parameters(self):
        return [FakeParameter(12), FakeParameter(3), FakeParameter(6), FakeParameter(2)]


def test_original_model_budget_counting_from_module(caplog):
    with caplog.at_level(logging.INFO, logger="pace.capacity_allocation"):
        budget = compute_original_model_budget(FakeModel())

    assert budget == ModelBudget(parameters=23, flops=None)
    assert "original_model_budget parameters=23" in caplog.text


def test_load_capacity_scores_json(tmp_path):
    path = tmp_path / "scores.json"
    path.write_text(json.dumps({"capacity_scores": [1.0, 2.0, 3.0]}))

    scores = load_capacity_scores(path)

    assert np.allclose(scores, np.array([1.0, 2.0, 3.0]))


def test_load_capacity_scores_yaml(tmp_path):
    path = tmp_path / "scores.yaml"
    path.write_text("capacity_scores:\n  - 4.0\n  - 5.0\n")

    scores = load_capacity_scores(path)

    assert np.allclose(scores, np.array([4.0, 5.0]))


def test_load_capacity_scores_npy(tmp_path):
    path = tmp_path / "scores.npy"
    np.save(path, np.array([6.0, 7.0], dtype=np.float64))

    scores = load_capacity_scores(path)

    assert np.allclose(scores, np.array([6.0, 7.0]))


def test_load_capacity_scores_pt(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "scores.pt"
    torch.save({"capacity_scores": torch.tensor([8.0, 9.0])}, path)

    scores = load_capacity_scores(path)

    assert np.allclose(scores, np.array([8.0, 9.0]))


def test_capacity_config_supports_requested_fields():
    config = CapacityAllocationConfig.from_mapping(
        {
            "student_variant": "blockwise_capacity",
            "timestep_blocks": [[0, 200], [200, 500], [500, 1000]],
            "block_capacity_scores_path": "block.json",
            "layer_capacity_scores_path": "layer.json",
            "match_original_total_budget": True,
            "original_model_config_path": "teacher.json",
            "original_model_checkpoint_path": "teacher.pt",
            "allocation_group_scope": "allocatable",
            "allocation_alpha": 0.5,
            "min_width_multiplier": 0.25,
            "max_width_multiplier": 2.0,
            "shuffle_seed": 123,
            "budget_tolerance": 0.05,
            "rounding_rules": {
                "channel_divisibility": 8,
                "hidden_size_divisibility": 16,
                "attention_head_divisibility": 2,
            },
        }
    )

    assert config.student_variant == "blockwise_capacity"
    assert [(block.start, block.end) for block in config.timestep_blocks] == [
        (0, 200),
        (200, 500),
        (500, 1000),
    ]
    assert config.match_original_total_budget is True
    assert config.allocation_group_scope == "allocatable"
    assert config.allocation_alpha == 0.5
    assert config.rounding_rules.channel_divisibility == 8


def test_capacity_config_can_defer_blocks_to_eval_output_dir():
    config = CapacityAllocationConfig.from_mapping(
        {
            "student_variant": "blockwise_capacity",
            "eval_output_dir": "out_eval_edm_cifar10",
            "match_original_total_budget": True,
        }
    )

    assert config.shuffle_seed == 3

    assert config.eval_output_dir == "out_eval_edm_cifar10"
    assert config.timestep_blocks == []


def test_eval_output_layout_and_result_score_loading(tmp_path):
    out_dir = tmp_path / "out_eval_edm_cifar10"
    grouping_dir = out_dir / "grouping"
    grouping_dir.mkdir(parents=True)
    results_path = out_dir / "results.json"
    grouping_path = grouping_dir / "timestep_grouping.json"
    checkpoint_path = out_dir / "checkpoint_random_same_norm_rank0.pt"

    results_path.write_text(
        json.dumps(
            {
                "group_param_counts": {"g0": 3, "g1": 5},
                "n_eff": [1.0, 3.0, 5.0, 7.0],
                "p_eff": [10.0, 10.0, 10.0, 10.0],
                "relative_delta_stack": [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 7.0, 11.0, 13.0],
                ],
            }
        )
    )
    grouping_path.write_text(json.dumps({"boundaries": [0, 2, 4]}))
    checkpoint_path.write_text("placeholder")

    layout = discover_evaluation_output(out_dir)
    blocks = load_timestep_blocks_from_grouping(grouping_path)
    block_scores, metric_name = load_block_capacity_scores_from_results(
        results_path,
        blocks,
        metric="auto",
        reduction="mean",
    )
    layer_scores = load_layer_capacity_scores_from_results(
        results_path,
        blocks,
        score_source="relative_delta_stack",
        reduction="mean",
    )

    assert layout.results_json_path == str(results_path)
    assert layout.timestep_grouping_path == str(grouping_path)
    assert layout.checkpoint_paths == (str(checkpoint_path),)
    assert [(block.start, block.end) for block in blocks] == [(0, 2), (2, 4)]
    assert metric_name == "n_eff"
    assert np.allclose(block_scores, np.array([2.0, 6.0]))
    assert np.allclose(layer_scores, np.array([[1.5, 6.0], [3.5, 12.0]]))

    max_scores, _ = load_block_capacity_scores_from_results(
        results_path,
        blocks,
        metric="n_eff",
        reduction="max",
    )
    q90_scores, _ = load_block_capacity_scores_from_results(
        results_path,
        blocks,
        metric="n_eff",
        reduction="q90",
    )
    q90_layer_scores = load_layer_capacity_scores_from_results(
        results_path,
        blocks,
        score_source="relative_delta_stack",
        reduction="q90",
    )

    assert np.allclose(max_scores, np.array([3.0, 7.0]))
    assert np.allclose(q90_scores, np.array([2.8, 6.8]))
    assert np.allclose(q90_layer_scores, np.array([[1.9, 6.8], [3.9, 12.8]]))


def test_explicit_stage_metadata_aggregates_filters_and_excludes_fixed_groups(tmp_path):
    names = ["stem.filter_0"]
    names.extend(
        f"residual_layer.residual_blocks.{block}.conv.filter_{channel}"
        for block in range(36)
        for channel in range(2)
    )
    names.append("head.filter_0")
    stage_keys = {
        name: (
            "stem"
            if name.startswith("stem")
            else "head"
            if name.startswith("head")
            else f"residual_block_{int(name.split('.')[2]):02d}"
        )
        for name in names
    }
    results = {
        "group_names": names,
        "group_structural_keys": {name: name.rsplit(".filter_", 1)[0] for name in names},
        "group_stage_keys": stage_keys,
        "group_allocatable": {
            name: stage_keys[name].startswith("residual_block_") for name in names
        },
        "group_param_counts": {name: 3 for name in names},
        "relative_delta_stack": [
            [float(index + 1), float(index + 2), float(index + 3), float(index + 4)]
            for index in range(len(names))
        ],
        "model_info": {
            "total_parameter_count": 323,
            "allocation_unassigned_parameter_count": 101,
        },
    }
    matrix = np.asarray(results["relative_delta_stack"], dtype=np.float64)

    aggregated, report = aggregate_group_scores_for_allocation(results, matrix)

    assert aggregated.shape == (36, 4)
    assert report["output_stage_keys"] == [f"residual_block_{block:02d}" for block in range(36)]
    assert report["allocatable_group_count"] == 72
    assert report["excluded_group_count"] == 2
    assert report["allocatable_parameter_count"] == 216
    assert report["excluded_analyzed_parameter_count"] == 6
    assert report["unassigned_parameter_count"] == 101
    assert report["accounted_parameter_count"] == 323
    assert report["total_teacher_parameter_count"] == 323
    assert report["parameter_accounting_matches_teacher"] is True
    assert np.allclose(aggregated[0], matrix[1] + matrix[2])
    assert np.allclose(aggregated[-1], matrix[-3] + matrix[-2])

    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(results))
    grouping_path = tmp_path / "grouping.json"
    grouping_path.write_text(json.dumps({"boundaries": [0, 2, 4]}))
    blocks = load_timestep_blocks_from_grouping(grouping_path)
    scores, loaded_report = load_layer_capacity_scores_with_metadata_from_results(
        results_path,
        blocks,
        reduction="mean",
    )
    assert scores.shape == (2, 36)
    assert loaded_report == report
    assert np.allclose(scores[0, 0], np.mean((matrix[1] + matrix[2])[:2]))
    assert compute_original_model_budget(results).parameters == 216


def test_explicit_allocation_keys_can_select_rows_without_boolean_metadata():
    results = {
        "group_names": ["stem.f0", "block.f0", "block.f1", "head.f0"],
        "group_structural_keys": {
            "stem.f0": "stem",
            "block.f0": "block.conv",
            "block.f1": "block.conv",
            "head.f0": "head",
        },
        "group_stage_keys": {
            "stem.f0": "stem",
            "block.f0": "residual_block_00",
            "block.f1": "residual_block_00",
            "head.f0": "head",
        },
        "group_allocation_structural_keys": {
            "block.f0": "residual_block_00",
            "block.f1": "residual_block_00",
        },
        "group_param_counts": {"stem.f0": 2, "block.f0": 3, "block.f1": 4, "head.f0": 5},
    }
    matrix = np.arange(8, dtype=np.float64).reshape(4, 2)

    aggregated, report = aggregate_group_scores_for_allocation(results, matrix)

    assert np.array_equal(aggregated, (matrix[1] + matrix[2]).reshape(1, 2))
    assert report["key_source"] == "group_allocation_structural_keys"
    assert report["excluded_reasons"] == {"missing_allocation_structural_key": 2}
    assert compute_original_model_budget(results).parameters == 7


def test_dry_run_layerwise_allocation_reports_structural_filter_scope(tmp_path):
    out_dir = tmp_path / "out_eval_diffwave"
    grouping_dir = out_dir / "grouping"
    grouping_dir.mkdir(parents=True)
    names = ["stem.f0", "b0.f0", "b0.f1", "b1.f0", "head.f0"]
    stage_keys = {
        "stem.f0": "stem",
        "b0.f0": "residual_block_00",
        "b0.f1": "residual_block_00",
        "b1.f0": "residual_block_01",
        "head.f0": "head",
    }
    (out_dir / "results.json").write_text(
        json.dumps(
            {
                "group_names": names,
                "group_structural_keys": {name: name.rsplit(".", 1)[0] for name in names},
                "group_stage_keys": stage_keys,
                "group_allocatable": {
                    name: stage_keys[name].startswith("residual_block_") for name in names
                },
                "group_param_counts": {
                    "stem.f0": 2,
                    "b0.f0": 3,
                    "b0.f1": 4,
                    "b1.f0": 5,
                    "head.f0": 6,
                },
                "n_eff": [1.0, 1.0, 2.0, 2.0],
                "relative_delta_stack": [
                    [100.0, 100.0, 100.0, 100.0],
                    [1.0, 1.0, 2.0, 2.0],
                    [3.0, 3.0, 4.0, 4.0],
                    [5.0, 5.0, 6.0, 6.0],
                    [200.0, 200.0, 200.0, 200.0],
                ],
                "model_info": {
                    "total_parameter_count": 40,
                    "allocation_unassigned_parameter_count": 20,
                },
            }
        )
    )
    (grouping_dir / "timestep_grouping.json").write_text(
        json.dumps({"boundaries": [0, 2, 4]})
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/dry_run_capacity_allocation.py",
            "--eval-output-dir",
            str(out_dir),
            "--student-variant",
            "layerwise_capacity",
            # Legacy n_eff scores (the fixture has no delta_stack); the paper
            # defaults are covered by test_dry_run_cli_defaults_to_paper_rule.
            "--allocation-metric",
            "auto",
            "--score-reduction",
            "mean",
            "--layer-score-source",
            "relative_delta_stack",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["original_model_budget"]["parameters"] == 12
    assert payload["total_student_system_budget"] == 12
    assert payload["target_budget_plan"]["layer_names"] == [
        "residual_block_00",
        "residual_block_01",
    ]
    assert np.asarray(payload["target_budget_plan"]["layer_budgets"]).shape == (2, 2)
    assert payload["allocation_scope"]["excluded_group_names"] == ["stem.f0", "head.f0"]
    assert payload["allocation_scope"]["excluded_analyzed_parameter_count"] == 8
    assert payload["allocation_scope"]["unassigned_parameter_count"] == 20


def test_delta_p_eff_geomean_metric_and_layer_scores():
    results = {
        "group_names": ["g0", "g1"],
        "group_param_counts": {"g0": 4.0, "g1": 9.0},
        "delta_stack": [
            [1.0, 4.0, 9.0],
            [3.0, 5.0, 7.0],
        ],
        "p_eff": [16.0, 25.0, 36.0],
    }

    metric = compute_delta_p_eff_geomean_metric(results)
    layer_scores = compute_delta_param_geomean_layer_scores(results)

    assert np.allclose(metric, np.sqrt(np.array([4.0, 9.0, 16.0]) * np.array([16.0, 25.0, 36.0])))
    assert np.allclose(
        layer_scores,
        np.array(
            [
                [2.0, 4.0, 6.0],
                [np.sqrt(27.0), np.sqrt(45.0), np.sqrt(63.0)],
            ]
        ),
    )


def test_sampled_profile_uses_ht_expansion_for_totals_and_layer_mass(tmp_path):
    group_names = ["m0.filter_0", "m1.filter_0"]
    sampling = _stratified_sampling(group_names, [2, 3])
    results = {
        "group_names": group_names,
        "group_param_counts": {group_names[0]: 10.0, group_names[1]: 20.0},
        "delta_stack": [[2.0, 4.0], [1.0, 3.0]],
        # HT ratio: sum(e*d*p) / sum(e*d).
        "p_eff": [100.0 / 7.0, 260.0 / 17.0],
        "relative_delta_stack": [[1.0, 2.0], [3.0, 5.0]],
        "filter_sampling": sampling,
        "group_sampling_weights": {group_names[0]: 2.0, group_names[1]: 3.0},
    }

    assert results_group_expansion_weights(results).tolist() == [2.0, 3.0]
    assert compute_original_model_budget(results) == ModelBudget(parameters=80, flops=None)
    # sum(e*d) * p_eff is exactly [100, 260].
    assert compute_delta_p_eff_geomean_metric(results) == pytest.approx(np.sqrt([100.0, 260.0]))
    assert compute_delta_param_geomean_layer_scores(results) == pytest.approx(
        np.asarray(
            [
                [2.0 * np.sqrt(20.0), 2.0 * np.sqrt(40.0)],
                [3.0 * np.sqrt(20.0), 3.0 * np.sqrt(60.0)],
            ]
        )
    )

    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(results))
    layer_scores = load_layer_capacity_scores_from_results(
        results_path,
        [TimestepBlock(0, 2)],
        score_source="relative_delta_stack",
        reduction="mean",
    )
    # Raw rows remain [1,2] and [3,5]; expansion is an allocation contribution.
    assert np.allclose(layer_scores, np.asarray([[3.0, 12.0]]))


def test_sampling_weight_alias_mismatch_is_rejected():
    group_names = ["m0.filter_0"]
    results = {
        "group_names": group_names,
        "filter_sampling": _stratified_sampling(group_names, [2]),
        "group_sampling_weights": {group_names[0]: 3.0},
    }
    with pytest.raises(ValueError, match="does not match"):
        results_group_expansion_weights(results)


def test_delta_p_eff_geomean_metric_requires_inputs():
    with pytest.raises(KeyError, match="delta_stack"):
        compute_delta_p_eff_geomean_metric({"p_eff": [1.0]})
    with pytest.raises(KeyError, match="p_eff"):
        compute_delta_p_eff_geomean_metric({"delta_stack": [[1.0]]})


def test_delta_p_eff_geomean_metric_recomputes_allocatable_scope():
    results = {
        "group_names": ["fixed_head", "residual_a", "residual_b"],
        "group_allocatable": {
            "fixed_head": False,
            "residual_a": True,
            "residual_b": True,
        },
        "group_param_counts": {
            "fixed_head": 1000.0,
            "residual_a": 4.0,
            "residual_b": 9.0,
        },
        "delta_stack": [
            [100.0, 100.0],
            [1.0, 4.0],
            [3.0, 0.0],
        ],
        # Deliberately unrelated to the allocatable groups: this stored value
        # must only be used by the historical all-groups protocol.
        "p_eff": [1000.0, 1000.0],
    }

    all_groups = compute_delta_p_eff_geomean_metric(results, group_scope="all")
    allocatable = compute_delta_p_eff_geomean_metric(
        results,
        group_scope="allocatable",
    )

    assert np.allclose(all_groups, np.sqrt(np.array([104.0, 104.0]) * 1000.0))
    # Recomputed p_eff is [31 / 4, 16 / 4], using only residual_a/b.
    assert np.allclose(allocatable, np.array([np.sqrt(31.0), 4.0]))

    without_stored_p_eff = dict(results)
    without_stored_p_eff.pop("p_eff")
    assert np.allclose(
        compute_delta_p_eff_geomean_metric(
            without_stored_p_eff,
            group_scope="allocatable",
        ),
        allocatable,
    )


def test_delta_p_eff_geomean_allocatable_scope_requires_metadata():
    with pytest.raises(KeyError, match="group_allocatable"):
        compute_delta_p_eff_geomean_metric(
            {
                "group_names": ["g0"],
                "group_param_counts": {"g0": 1},
                "delta_stack": [[1.0]],
            },
            group_scope="allocatable",
        )


def test_combined_score_loading_uses_sum_reduction(tmp_path):
    out_dir = tmp_path / "out_eval"
    grouping_dir = out_dir / "grouping"
    grouping_dir.mkdir(parents=True)
    results_path = out_dir / "results.json"
    grouping_path = grouping_dir / "timestep_grouping.json"
    results_path.write_text(
        json.dumps(
            {
                "group_names": ["g0", "g1"],
                "group_param_counts": {"g0": 4.0, "g1": 9.0},
                "delta_stack": [[1.0, 4.0, 9.0, 16.0], [3.0, 5.0, 7.0, 9.0]],
                "p_eff": [16.0, 25.0, 36.0, 49.0],
            }
        )
    )
    grouping_path.write_text(json.dumps({"boundaries": [0, 2, 4]}))
    blocks = load_timestep_blocks_from_grouping(grouping_path)

    block_scores, metric_name = load_block_capacity_scores_from_results(
        results_path,
        blocks,
        metric="delta_p_eff_geomean",
        reduction="sum",
    )
    layer_scores = load_layer_capacity_scores_from_results(
        results_path,
        blocks,
        score_source="delta_p_eff_geomean",
        reduction="sum",
    )

    assert metric_name == "delta_p_eff_geomean"
    expected_metric = np.sqrt(np.array([4.0, 9.0, 16.0, 25.0]) * np.array([16.0, 25.0, 36.0, 49.0]))
    assert block_scores == pytest.approx([expected_metric[:2].sum(), expected_metric[2:].sum()])
    expected_layer = np.sqrt(
        np.array([[1.0, 4.0, 9.0, 16.0], [3.0, 5.0, 7.0, 9.0]])
        * np.array([[4.0], [9.0]])
    )
    assert np.allclose(layer_scores, np.stack([expected_layer[:, :2].sum(axis=1), expected_layer[:, 2:].sum(axis=1)]))


def test_dry_run_cli_accepts_eval_output_dir(tmp_path):
    out_dir = tmp_path / "out_eval_edm_cifar10"
    grouping_dir = out_dir / "grouping"
    grouping_dir.mkdir(parents=True)
    (out_dir / "results.json").write_text(
        json.dumps(
            {
                "group_param_counts": {"g0": 3, "g1": 5},
                "n_eff": [1.0, 3.0, 5.0, 7.0],
                "relative_delta_stack": [[1.0, 2.0, 3.0, 4.0]],
            }
        )
    )
    (grouping_dir / "timestep_grouping.json").write_text(json.dumps({"boundaries": [0, 2, 4]}))

    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/dry_run_capacity_allocation.py",
            "--eval-output-dir",
            str(out_dir),
            "--score-reduction",
            "q90",
            "--allocation-metric",
            "auto",
            "--layer-score-source",
            "relative_delta_stack",
            *[
                argument
                for variant in (
                    "global",
                    "uniform_blockwise",
                    "shuffled_capacity",
                    "blockwise_capacity",
                    "layerwise_capacity",
                    "reversed_layerwise_capacity",
                )
                for argument in ("--student-variant", variant)
            ],
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["eval_output"]["results_json_path"] == str(out_dir / "results.json")
    assert payload["student_variants"] == [
        "global",
        "uniform_blockwise",
        "shuffled_capacity",
        "blockwise_capacity",
        "layerwise_capacity",
        "reversed_layerwise_capacity",
    ]
    assert payload["allocation_group_scope"] == "all"
    assert payload["score_reduction"] == "q90"
    allocation_dir = out_dir / "allocation_results"
    assert payload["allocation_results_dir"] == str(allocation_dir)
    assert payload["allocation_summary_path"] == str(allocation_dir / "summary.json")
    assert (allocation_dir / "summary.json").is_file()
    for variant in payload["student_variants"]:
        assert (allocation_dir / f"{variant}.json").is_file()

    blockwise_payload = json.loads((allocation_dir / "blockwise_capacity.json").read_text())
    shuffled_payload = json.loads((allocation_dir / "shuffled_capacity.json").read_text())
    assert payload["shuffle_seed"] == 3
    assert shuffled_payload["shuffle_seed"] == 3
    assert blockwise_payload["timestep_blocks"] == [[0, 2], [2, 4]]
    assert blockwise_payload["score_sources"]["block_capacity_scores"]["metric"] == "n_eff"
    assert blockwise_payload["score_sources"]["block_capacity_scores"]["reduction"] == "q90"
    assert blockwise_payload["target_budget_plan"]["block_budgets"] == pytest.approx([7.0 / 3.0, 17.0 / 3.0])


def test_dry_run_cli_student_variant_writes_single_variant_payload(tmp_path):
    out_dir = tmp_path / "out_eval_edm_cifar10"
    grouping_dir = out_dir / "grouping"
    grouping_dir.mkdir(parents=True)
    (out_dir / "results.json").write_text(
        json.dumps(
            {
                "group_param_counts": {"g0": 3, "g1": 5},
                "n_eff": [1.0, 3.0, 5.0, 7.0],
                "relative_delta_stack": [[1.0, 2.0, 3.0, 4.0]],
            }
        )
    )
    (grouping_dir / "timestep_grouping.json").write_text(json.dumps({"boundaries": [0, 2, 4]}))

    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/dry_run_capacity_allocation.py",
            "--eval-output-dir",
            str(out_dir),
            "--student-variant",
            "blockwise_capacity",
            "--allocation-metric",
            "auto",
            "--score-reduction",
            "mean",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["student_variant"] == "blockwise_capacity"
    assert payload["allocation_results_dir"] == str(out_dir / "allocation_results")
    assert payload["allocation_results_path"] == str(out_dir / "allocation_results" / "blockwise_capacity.json")
    assert (out_dir / "allocation_results" / "summary.json").is_file()
    assert (out_dir / "allocation_results" / "blockwise_capacity.json").is_file()


def test_dry_run_cli_defaults_to_paper_rule(tmp_path):
    """Without flags the CLI runs the Section 3.4 rule and the four paper variants."""
    out_dir = tmp_path / "out_eval_edm_cifar10"
    grouping_dir = out_dir / "grouping"
    grouping_dir.mkdir(parents=True)
    (out_dir / "results.json").write_text(
        json.dumps(
            {
                "num_parameters": 100.0,
                "group_names": ["g0", "g1"],
                "group_param_counts": {"g0": 4.0, "g1": 9.0},
                "delta_stack": [[1.0, 4.0, 9.0, 16.0], [3.0, 5.0, 7.0, 9.0]],
                "p_eff": [16.0, 25.0, 36.0, 49.0],
            }
        )
    )
    (grouping_dir / "timestep_grouping.json").write_text(json.dumps({"boundaries": [0, 2, 4]}))

    completed = subprocess.run(
        [sys.executable, "scripts/dry_run_capacity_allocation.py", "--eval-output-dir", str(out_dir)],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["student_variants"] == [
        "global",
        "uniform_blockwise",
        "combined_blockwise",
        "combined_layerwise",
    ]
    assert summary["score_reduction"] == "sum"
    combined = summary["variants"]["combined_blockwise"]
    assert combined["allocation_rule"]["allocation_metric"] == "delta_p_eff_geomean"
    assert combined["allocation_rule"]["score_reduction"] == "sum"
    layerwise = summary["variants"]["combined_layerwise"]
    assert layerwise["allocation_rule"]["layer_score_source"] == "delta_p_eff_geomean"
    # Q_k = sum_t sqrt(sum_g Delta_{g,t} * p_eff_t) over each phase.
    q = np.sqrt(np.array([4.0, 9.0, 16.0, 25.0]) * np.array([16.0, 25.0, 36.0, 49.0]))
    phase_scores = np.array([q[:2].sum(), q[2:].sum()])
    budgets = combined["target_budget_plan"]["block_budgets"]
    assert np.allclose(budgets, 100.0 * phase_scores / phase_scores.sum())


def test_dry_run_cli_combined_variant_writes_isolated_allocation_dir(tmp_path):
    out_dir = tmp_path / "out_eval_edm_cifar10"
    allocation_dir = tmp_path / "allocation_results_delta_peff"
    grouping_dir = out_dir / "grouping"
    grouping_dir.mkdir(parents=True)
    (out_dir / "results.json").write_text(
        json.dumps(
            {
                "num_parameters": 100.0,
                "group_names": ["g0", "g1"],
                "group_param_counts": {"g0": 4.0, "g1": 9.0},
                "delta_stack": [[1.0, 4.0, 9.0, 16.0], [3.0, 5.0, 7.0, 9.0]],
                "p_eff": [16.0, 25.0, 36.0, 49.0],
            }
        )
    )
    (grouping_dir / "timestep_grouping.json").write_text(json.dumps({"boundaries": [0, 2, 4]}))

    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/dry_run_capacity_allocation.py",
            "--eval-output-dir",
            str(out_dir),
            "--allocation-results-dir",
            str(allocation_dir),
            "--student-variant",
            "combined_blockwise",
            "--allocation-metric",
            "delta_p_eff_geomean",
            "--score-reduction",
            "sum",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    expected_metric = np.sqrt(np.array([4.0, 9.0, 16.0, 25.0]) * np.array([16.0, 25.0, 36.0, 49.0]))
    expected_scores = np.array([expected_metric[:2].sum(), expected_metric[2:].sum()])
    expected_budgets = expected_scores / expected_scores.sum() * 100.0

    assert payload["student_variant"] == "combined_blockwise"
    assert payload["allocation_results_dir"] == str(allocation_dir)
    assert payload["score_sources"]["block_capacity_scores"]["metric"] == "delta_p_eff_geomean"
    assert payload["score_sources"]["block_capacity_scores"]["reduction"] == "sum"
    assert payload["allocation_group_scope"] == "all"
    assert payload["score_reduction"] == "sum"
    assert payload["score_sources"]["block_capacity_scores"]["group_scope"] == "all"
    assert payload["score_sources"]["block_capacity_scores"]["p_eff_source"] == "results.p_eff"
    assert payload["score_sources"]["block_capacity_scores"]["reduction_protocol"] == "edm_exact_sum"
    assert payload["target_budget_plan"]["block_budgets"] == pytest.approx(expected_budgets.tolist())
    assert (allocation_dir / "combined_blockwise.json").is_file()
    assert not (out_dir / "allocation_results" / "combined_blockwise.json").exists()


def test_dry_run_cli_combined_metric_rejects_non_geometric_reduction(tmp_path):
    out_dir = tmp_path / "out_eval_edm_cifar10"
    grouping_dir = out_dir / "grouping"
    grouping_dir.mkdir(parents=True)
    (out_dir / "results.json").write_text(
        json.dumps(
            {
                "num_parameters": 100.0,
                "group_names": ["g0"],
                "group_param_counts": {"g0": 4.0},
                "delta_stack": [[1.0, 2.0]],
                "p_eff": [3.0, 4.0],
            }
        )
    )
    (grouping_dir / "timestep_grouping.json").write_text(json.dumps({"boundaries": [0, 1, 2]}))

    repo_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/dry_run_capacity_allocation.py",
            "--eval-output-dir",
            str(out_dir),
            "--student-variant",
            "combined_blockwise",
            "--allocation-metric",
            "delta_p_eff_geomean",
            "--score-reduction",
            "q90",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "delta_p_eff_geomean score reduction must be one of" in completed.stderr


def test_dry_run_cli_combined_layerwise_allocatable_mean_is_duration_neutral(tmp_path):
    out_dir = tmp_path / "out_eval_diffwave"
    grouping_dir = out_dir / "grouping"
    grouping_dir.mkdir(parents=True)
    (out_dir / "results.json").write_text(
        json.dumps(
            {
                "group_names": ["fixed_head", "residual"],
                "group_stage_keys": {
                    "fixed_head": "head",
                    "residual": "residual_block_00",
                },
                "group_allocatable": {
                    "fixed_head": False,
                    "residual": True,
                },
                "group_param_counts": {
                    "fixed_head": 100,
                    "residual": 4,
                },
                "delta_stack": [
                    [100.0, 100.0, 100.0, 100.0],
                    [1.0, 1.0, 1.0, 4.0],
                ],
                "p_eff": [100.0, 100.0, 100.0, 100.0],
                "model_info": {
                    "total_parameter_count": 104,
                    "allocation_unassigned_parameter_count": 0,
                },
            }
        )
    )
    (grouping_dir / "timestep_grouping.json").write_text(
        json.dumps({"boundaries": [0, 3, 4]})
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/dry_run_capacity_allocation.py",
            "--eval-output-dir",
            str(out_dir),
            "--student-variant",
            "combined_layerwise",
            "--allocation-metric",
            "delta_p_eff_geomean",
            "--allocation-group-scope",
            "allocatable",
            "--score-reduction",
            "mean",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(completed.stdout)

    # Allocatable per-timestep scores are [2, 2, 2, 4]. Taking phase means
    # yields [2, 4]; a sum would instead be [6, 4] and favor the long phase.
    assert payload["original_model_budget"]["parameters"] == 4
    assert payload["target_budget_plan"]["block_budgets"] == pytest.approx(
        [4.0 / 3.0, 8.0 / 3.0]
    )
    assert payload["allocation_group_scope"] == "allocatable"
    assert payload["score_reduction"] == "mean"
    score_source = payload["score_sources"]["block_capacity_scores"]
    assert score_source["group_scope"] == "allocatable"
    assert score_source["p_eff_source"] == "recomputed_from_allocatable_groups"
    assert score_source["reduction_protocol"] == "duration_neutral_mean"
    assert payload["target_budget_plan"]["layer_names"] == ["residual_block_00"]
    assert np.asarray(payload["target_budget_plan"]["layer_budgets"]).shape == (2, 1)
    assert np.asarray(payload["target_budget_plan"]["layer_budgets"]).sum(axis=1) == pytest.approx(
        payload["target_budget_plan"]["block_budgets"]
    )


def test_original_budget_can_be_inferred_from_results_group_param_counts():
    budget = compute_original_model_budget({"group_param_counts": {"g0": 12, "g1": 8}})

    assert budget == ModelBudget(parameters=20, flops=None)


def test_normalization_uses_original_model_budget():
    original_budget = compute_original_model_budget({"num_parameters": 100})

    budgets = normalize_capacity_budgets(
        [1.0, 2.0, 3.0],
        total_budget=original_budget.parameters,
        alpha=1.0,
    )

    assert np.allclose(budgets, np.array([100.0 / 6.0, 200.0 / 6.0, 300.0 / 6.0]))
    assert budgets.sum() == pytest.approx(original_budget.parameters)


def test_global_budget_uses_full_reference_budget():
    budgets = make_global_budget(total_budget=100.0)

    assert np.allclose(budgets, np.array([100.0]))


def test_uniform_block_budgets_split_reference_budget_across_blocks():
    budgets = make_uniform_block_budgets(num_blocks=3, total_budget=100.0)

    assert np.allclose(budgets, np.array([100.0 / 3.0, 100.0 / 3.0, 100.0 / 3.0]))
    assert budgets.sum() == pytest.approx(100.0)


def test_invalid_negative_scores():
    with pytest.raises(ValueError, match="nonnegative"):
        validate_capacity_scores([1.0, -1.0, 2.0])


def test_all_zero_scores_raise_unless_uniform_fallback_enabled():
    with pytest.raises(ValueError, match="all zero"):
        make_blockwise_capacity_budgets([0.0, 0.0], total_budget=100.0)

    budgets = make_blockwise_capacity_budgets(
        [0.0, 0.0],
        total_budget=100.0,
        allow_uniform_if_all_zero=True,
    )

    assert np.allclose(budgets, np.array([50.0, 50.0]))


def test_mismatched_number_of_blocks():
    with pytest.raises(ValueError, match="must match number of timestep blocks"):
        validate_capacity_scores([1.0, 2.0], expected_num_blocks=3)


def test_layerwise_sum_preservation():
    block_budgets = np.array([100.0, 200.0])
    layer_scores = np.array(
        [
            [1.0, 1.0, 2.0],
            [3.0, 1.0, 0.0],
        ]
    )

    layer_budgets = make_layerwise_capacity_budgets(block_budgets, layer_scores)

    assert layer_budgets.shape == (2, 3)
    assert np.allclose(layer_budgets.sum(axis=1), block_budgets)


def test_layerwise_scores_must_match_blocks_and_layers():
    with pytest.raises(ValueError, match="block dimension"):
        validate_capacity_scores(
            np.ones((2, 3)),
            expected_num_blocks=3,
            expected_num_layers=3,
            name="layer_capacity_scores",
        )

    with pytest.raises(ValueError, match="layer dimension"):
        validate_capacity_scores(
            np.ones((2, 3)),
            expected_num_blocks=2,
            expected_num_layers=4,
            name="layer_capacity_scores",
        )


def test_shuffled_budget_reproducibility_and_same_budget_set():
    scores = [1.0, 2.0, 3.0, 4.0]
    reference = make_blockwise_capacity_budgets(scores, total_budget=100.0, alpha=1.0)

    shuffled_a = make_shuffled_capacity_budgets(scores, total_budget=100.0, alpha=1.0, seed=7)
    shuffled_b = make_shuffled_capacity_budgets(scores, total_budget=100.0, alpha=1.0, seed=7)

    assert np.allclose(shuffled_a, shuffled_b)
    assert sorted(shuffled_a.tolist()) == pytest.approx(sorted(reference.tolist()))
    assert shuffled_a.sum() == pytest.approx(100.0)


def test_budget_matching_between_blockwise_and_layerwise_capacity():
    block_budgets = make_blockwise_capacity_budgets([1.0, 3.0], total_budget=100.0)
    layer_budgets = make_layerwise_capacity_budgets(
        block_budgets,
        np.array(
            [
                [1.0, 2.0],
                [4.0, 1.0],
            ]
        ),
    )

    assert np.allclose(layer_budgets.sum(axis=1), block_budgets)
    assert layer_budgets.sum() == pytest.approx(block_budgets.sum())
    assert layer_budgets.sum(axis=1).sum() == pytest.approx(100.0)


def test_variant_budget_plans_implement_revised_definitions():
    total_budget = 120.0
    num_blocks = 3
    block_scores = np.array([1.0, 2.0, 3.0])
    layer_scores = np.array(
        [
            [1.0, 1.0],
            [1.0, 3.0],
            [4.0, 2.0],
        ]
    )

    global_plan = make_variant_capacity_budgets(
        "global",
        num_blocks=num_blocks,
        total_budget=total_budget,
    )
    uniform_plan = make_variant_capacity_budgets(
        "uniform_blockwise",
        num_blocks=num_blocks,
        total_budget=total_budget,
    )
    blockwise_plan = make_variant_capacity_budgets(
        "blockwise_capacity",
        num_blocks=num_blocks,
        total_budget=total_budget,
        block_capacity_scores=block_scores,
    )
    layerwise_plan = make_variant_capacity_budgets(
        "layerwise_capacity",
        num_blocks=num_blocks,
        total_budget=total_budget,
        block_capacity_scores=block_scores,
        layer_capacity_scores=layer_scores,
    )
    shuffled_plan = make_variant_capacity_budgets(
        "shuffled_capacity",
        num_blocks=num_blocks,
        total_budget=total_budget,
        block_capacity_scores=block_scores,
        seed=11,
    )
    reversed_layerwise_plan = make_variant_capacity_budgets(
        "reversed_layerwise_capacity",
        num_blocks=num_blocks,
        total_budget=total_budget,
        block_capacity_scores=block_scores,
        layer_capacity_scores=layer_scores,
    )
    combined_blockwise_plan = make_variant_capacity_budgets(
        "combined_blockwise",
        num_blocks=num_blocks,
        total_budget=total_budget,
        block_capacity_scores=block_scores,
    )
    combined_layerwise_plan = make_variant_capacity_budgets(
        "combined_layerwise",
        num_blocks=num_blocks,
        total_budget=total_budget,
        block_capacity_scores=block_scores,
        layer_capacity_scores=layer_scores,
    )

    assert global_plan.block_budgets == [120.0]
    assert global_plan.num_students == 1
    assert uniform_plan.block_budgets == pytest.approx([40.0, 40.0, 40.0])
    assert blockwise_plan.block_budgets == pytest.approx([20.0, 40.0, 60.0])
    assert layerwise_plan.block_budgets == pytest.approx(blockwise_plan.block_budgets)
    assert np.asarray(layerwise_plan.layer_budgets).sum(axis=1).tolist() == pytest.approx(
        blockwise_plan.block_budgets
    )
    assert sorted(shuffled_plan.block_budgets) == pytest.approx(sorted(blockwise_plan.block_budgets))
    assert reversed_layerwise_plan.block_budgets == pytest.approx([60.0, 40.0, 20.0])
    assert np.asarray(reversed_layerwise_plan.layer_budgets).sum(axis=1).tolist() == pytest.approx(
        reversed_layerwise_plan.block_budgets
    )
    assert combined_blockwise_plan.block_budgets == pytest.approx(blockwise_plan.block_budgets)
    assert combined_layerwise_plan.block_budgets == pytest.approx(blockwise_plan.block_budgets)
    assert np.asarray(combined_layerwise_plan.layer_budgets).sum(axis=1).tolist() == pytest.approx(
        combined_layerwise_plan.block_budgets
    )
    for plan in [
        global_plan,
        uniform_plan,
        blockwise_plan,
        layerwise_plan,
        shuffled_plan,
        reversed_layerwise_plan,
        combined_blockwise_plan,
        combined_layerwise_plan,
    ]:
        assert plan.total_stored_budget == pytest.approx(total_budget)
        assert plan.total_student_system_budget == pytest.approx(total_budget)


def test_config_rejects_disabling_original_total_budget_match():
    with pytest.raises(ValueError, match="match_original_total_budget must be true"):
        CapacityAllocationConfig.from_mapping(
            {
                "student_variant": "global",
                "timestep_blocks": [[0, 10]],
                "match_original_total_budget": False,
            }
        )


@pytest.mark.parametrize("variant", ["combined_blockwise", "combined_layerwise"])
def test_combined_config_requires_geomean_and_accepts_both_supported_reductions(variant):
    base = {
        "student_variant": variant,
        "timestep_blocks": [[0, 3], [3, 20]],
        "match_original_total_budget": True,
    }
    with pytest.raises(ValueError, match="requires allocation_metric='delta_p_eff_geomean'"):
        CapacityAllocationConfig.from_mapping(
            base | {"allocation_metric": "n_eff", "score_reduction": "q90"}
        )
    combined_sum = CapacityAllocationConfig.from_mapping(
        base | {"allocation_metric": "delta_p_eff_geomean", "score_reduction": "sum"}
    )
    combined_mean = CapacityAllocationConfig.from_mapping(
        base | {"allocation_metric": "delta_p_eff_geomean", "score_reduction": "mean"}
    )
    global_control = CapacityAllocationConfig.from_mapping(
        base | {"student_variant": "global", "allocation_metric": "delta_p_eff_geomean", "score_reduction": "sum"}
    )
    uniform_control = CapacityAllocationConfig.from_mapping(
        base
        | {
            "student_variant": "uniform_blockwise",
            "allocation_metric": "delta_p_eff_geomean",
            "score_reduction": "sum",
        }
    )
    assert combined_sum.student_variant == variant
    assert combined_mean.student_variant == variant
    assert combined_mean.score_reduction == "mean"
    assert global_control.student_variant == "global"
    assert uniform_control.student_variant == "uniform_blockwise"


def test_validate_total_stored_budget_accepts_within_tolerance():
    report = validate_total_stored_budget(
        student_budgets=[33.0, 33.0, 34.0],
        original_model_budget=100.0,
        tolerance=0.0,
    )

    assert report.within_tolerance is True
    assert report.realized_budgets == [100.0]


def test_validate_total_stored_budget_fails_when_mismatch_exceeds_tolerance():
    with pytest.raises(ValueError, match="budget_tolerance"):
        validate_total_stored_budget(
            student_budgets=[20.0, 20.0, 20.0],
            original_model_budget=100.0,
            tolerance=0.05,
        )


def test_validate_budgets_reports_rounding_mismatch():
    report = validate_budgets(
        [100.0, 200.0],
        [103.0, 190.0],
        budget_tolerance=0.05,
        fail_on_mismatch=False,
    )

    assert report.within_tolerance is True
    assert report.realized_budgets == [103.0, 190.0]

    with pytest.raises(ValueError, match="budget_tolerance"):
        validate_budgets([100.0], [120.0], budget_tolerance=0.05)
