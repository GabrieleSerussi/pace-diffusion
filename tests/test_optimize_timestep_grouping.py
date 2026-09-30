import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.optimize_timestep_grouping import (
    _compute_pairwise_similarity_matrix,
    compute_total_cost_curve,
    optimal_timestep_partition,
    select_num_blocks_by_total_score,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_min_block_size_prevents_singleton_phases_and_default_is_unchanged():
    def singleton_reward(start: int, end: int) -> float:
        return -100.0 if end - start == 1 else 0.0

    default_result = optimal_timestep_partition(6, 3, cost_fn=singleton_reward)
    explicit_default = optimal_timestep_partition(6, 3, cost_fn=singleton_reward, min_block_size=1)
    constrained = optimal_timestep_partition(6, 3, cost_fn=singleton_reward, min_block_size=2)

    assert default_result.boundaries == explicit_default.boundaries
    assert any(block.size == 1 for block in default_result.blocks)
    assert constrained.boundaries == [0, 2, 4, 6]
    assert constrained.min_block_size == 2
    assert all(block.size >= 2 for block in constrained.blocks)


def test_min_block_size_rejects_infeasible_partition_and_curve():
    interval_costs = np.zeros((7, 7), dtype=np.float64)
    with pytest.raises(ValueError, match="min_block_size=3"):
        optimal_timestep_partition(6, 3, interval_costs=interval_costs, min_block_size=3)
    with pytest.raises(ValueError, match="cannot exceed 2"):
        compute_total_cost_curve(
            6,
            interval_costs,
            max_blocks=3,
            min_block_size=3,
        )


def test_spearman_similarity_ranks_each_timestep_profile_with_average_ties():
    matrix = np.asarray(
        [
            [1.0, 1.0, 4.0],
            [2.0, 4.0, 3.0],
            [3.0, 9.0, 2.0],
            [4.0, 16.0, 1.0],
        ]
    )

    spearman = _compute_pairwise_similarity_matrix(matrix, metric="spearman")
    pearson = _compute_pairwise_similarity_matrix(matrix, metric="pearson")

    assert spearman[0, 1] == pytest.approx(1.0)
    assert spearman[0, 2] == pytest.approx(-1.0)
    assert pearson[0, 1] < 0.99

    tied = np.asarray([[1.0, 10.0], [1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    tied_spearman = _compute_pairwise_similarity_matrix(tied, metric="spearman")
    assert tied_spearman[0, 1] == pytest.approx(1.0)


def test_num_block_selection_maximizes_score_and_breaks_ties_toward_smaller_k():
    assert select_num_blocks_by_total_score([1, 2, 3], [0.1, 0.8, 0.7]) == 2
    assert select_num_blocks_by_total_score([3, 1, 2], [0.8, 0.8, 0.7]) == 1


def test_cli_uses_scoped_spearman_matrix_minimum_size_and_automatic_k(tmp_path):
    matrix_path = tmp_path / "analysis.json"
    output_path = tmp_path / "grouping.json"
    identical_rank_profiles = np.tile(np.arange(1.0, 7.0)[:, None], (1, 6))
    matrix_path.write_text(json.dumps({"residual_only_signed_delta": identical_rank_profiles.tolist()}))

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "optimize_timestep_grouping.py"),
            "--matrix",
            str(matrix_path),
            "--matrix_key",
            "residual_only_signed_delta",
            "--builtin_cost",
            "matrix_spearman_cross_penalty",
            "--min-block-size",
            "2",
            "--select-num-blocks",
            "--max-num-blocks",
            "3",
            "--output",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload == json.loads(output_path.read_text())
    assert payload["array_key"] == "residual_only_signed_delta"
    assert payload["pairwise_metric"] == "spearman"
    assert payload["min_block_size"] == 2
    assert payload["num_blocks"] == 1
    assert payload["num_blocks_selection"]["policy"] == "maximize_cross_penalty_total_score"
    assert [item["num_blocks"] for item in payload["num_blocks_selection"]["candidates"]] == [1, 2, 3]
