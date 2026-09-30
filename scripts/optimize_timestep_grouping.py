"""Optimize contiguous timestep groupings with dynamic programming.

This script can be used as a small library or as a CLI utility.

Programmatic usage:
    from scripts.optimize_timestep_grouping import optimal_timestep_partition

    def interval_cost(start: int, end: int) -> float:
        # Cost for the half-open interval [start, end).
        ...

    result = optimal_timestep_partition(
        num_timesteps=1000,
        num_blocks=16,
        cost_fn=interval_cost,
    )

Paper settings (Section 3.3) are the defaults: the ``matrix_correlation_cross_penalty``
objective J(B) on ``relative_delta_stack`` with lambda_sep = 0.02 and automatic
selection of the number of phases (``--num_blocks auto``, ties toward smaller K):

    python scripts/optimize_timestep_grouping.py --matrix results.json --output grouping.json

Profiles may be plain ``.json`` or gzip-compressed ``.json.gz`` files.

CLI examples:
    python scripts/optimize_timestep_grouping.py --values results.json --array_key baseline_mean --num_blocks 8
    python scripts/optimize_timestep_grouping.py --interval_costs results.json --array_key C_noise_levels --num_blocks 8 --plot_output partition_overlay.png
    python scripts/optimize_timestep_grouping.py --builtin_cost matrix_correlation --matrix results.json --matrix_key row_normalized_relative_delta_stack --num_blocks 8 --plot_matrix results.json --plot_matrix_key C_noise_levels
    python scripts/optimize_timestep_grouping.py --builtin_cost matrix_spearman_cross_penalty --matrix scoped_results.json --matrix_key residual_only_signed_delta --select-num-blocks --max-num-blocks 4 --min-block-size 2
    python scripts/optimize_timestep_grouping.py --cost_function my_module:interval_cost --matrix results.json --matrix_key relative_delta_stack --num_blocks 8
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pace.profile_provenance import (
    provenance_from_results,
    provenance_from_results_path,
)
from pace.filter_sampling import (
    aligned_group_expansion_weights,
    expansion_weighting_summary,
)
from pace.jsonio import is_json_path, load_json

IntervalCostFn = Callable[[int, int], float]


@dataclass
class PartitionBlock:
    start: int
    end: int
    size: int
    cost: float


@dataclass
class PartitionResult:
    num_timesteps: int
    num_blocks: int
    total_cost: float
    boundaries: list[int]
    blocks: list[PartitionBlock]
    min_block_size: int = 1

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["blocks"] = [asdict(block) for block in self.blocks]
        return payload



def make_sse_cost(values: Sequence[float]) -> IntervalCostFn:
    """Return a segment cost that measures within-segment SSE."""
    values_array = np.asarray(values, dtype=np.float64)
    prefix_sum = np.zeros(len(values_array) + 1, dtype=np.float64)
    prefix_sum_sq = np.zeros(len(values_array) + 1, dtype=np.float64)
    prefix_sum[1:] = np.cumsum(values_array)
    prefix_sum_sq[1:] = np.cumsum(values_array ** 2)

    def cost(start: int, end: int) -> float:
        if not (0 <= start < end <= len(values_array)):
            raise ValueError(f"Invalid interval [{start}, {end}) for {len(values_array)} values")
        count = end - start
        total = prefix_sum[end] - prefix_sum[start]
        total_sq = prefix_sum_sq[end] - prefix_sum_sq[start]
        return float(total_sq - (total * total) / count)

    return cost



def _prepare_timestep_feature_matrix(matrix: np.ndarray, axis: int = -1) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim == 0:
        raise ValueError("matrix-based costs require an array with at least one timestep axis")

    resolved_axis = axis if axis >= 0 else arr.ndim + axis
    if resolved_axis < 0 or resolved_axis >= arr.ndim:
        raise ValueError(f"matrix axis {axis} is out of bounds for shape {arr.shape}")

    arr = np.moveaxis(arr, resolved_axis, -1)
    num_timesteps = arr.shape[-1]
    return arr.reshape(-1, num_timesteps)



def _make_2d_prefix_sums(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    prefix = np.zeros((arr.shape[0] + 1, arr.shape[1] + 1), dtype=np.float64)
    prefix[1:, 1:] = np.cumsum(np.cumsum(arr, axis=0), axis=1)
    return prefix



def _square_block_sum(prefix: np.ndarray, start: int, end: int) -> float:
    return float(prefix[end, end] - prefix[start, end] - prefix[end, start] + prefix[start, start])



def _rank_feature_columns(features: np.ndarray) -> np.ndarray:
    """Return average ranks down each feature column (ties share a rank)."""
    if not np.all(np.isfinite(features)):
        raise ValueError("Spearman similarity requires finite matrix values")

    ranks = np.empty_like(features, dtype=np.float64)
    for column_index in range(features.shape[1]):
        column = features[:, column_index]
        _, inverse, counts = np.unique(column, return_inverse=True, return_counts=True)
        starts = np.cumsum(counts) - counts
        average_ranks = starts + (counts + 1.0) / 2.0
        ranks[:, column_index] = average_ranks[inverse]
    return ranks



def _compute_pairwise_similarity_matrix(
    matrix: np.ndarray,
    axis: int = -1,
    metric: str = "cosine",
    eps: float = 1e-12,
    feature_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    features = _prepare_timestep_feature_matrix(matrix, axis=axis)
    weights = _normalize_feature_weights(feature_weights, features.shape[0])
    if metric == "spearman":
        features = _rank_feature_columns(features)
    if metric in {"correlation", "pearson", "spearman"}:
        if weights is None:
            features = features - features.mean(axis=0, keepdims=True)
        else:
            weighted_mean = np.average(features, axis=0, weights=weights)
            features = (features - weighted_mean.reshape(1, -1)) * np.sqrt(weights).reshape(-1, 1)
    elif metric != "cosine":
        raise ValueError(f"Unsupported pairwise metric: {metric}")
    elif weights is not None:
        features = features * np.sqrt(weights).reshape(-1, 1)

    norms = np.linalg.norm(features, axis=0, keepdims=True)
    normalized = np.divide(features, np.maximum(norms, eps), out=np.zeros_like(features), where=norms > eps)
    similarity = normalized.T @ normalized
    similarity = np.clip(similarity, -1.0, 1.0)
    similarity = 0.5 * (similarity + similarity.T)
    return similarity


def _normalize_feature_weights(
    feature_weights: Optional[np.ndarray],
    expected_features: int,
) -> Optional[np.ndarray]:
    """Validate expansion weights, retaining the exact legacy arithmetic path."""

    if feature_weights is None:
        return None
    weights = np.asarray(feature_weights, dtype=np.float64)
    if weights.ndim != 1 or weights.shape[0] != expected_features:
        raise ValueError(
            "feature_weights must be 1D with one entry per matrix feature row, "
            f"got shape {weights.shape} for {expected_features} rows"
        )
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("feature_weights must contain only positive finite values")
    # Unit expansion is mathematically and operationally exhaustive.  Taking
    # the old branch avoids even roundoff-level changes to legacy results.
    return None if np.all(weights == 1.0) else weights



def infer_pairwise_metric_from_builtin_cost(builtin_cost: Optional[str]) -> Optional[str]:
    if builtin_cost in {"matrix_cosine", "matrix_cosine_cross_penalty"}:
        return "cosine"
    if builtin_cost in {"matrix_correlation", "matrix_correlation_cross_penalty"}:
        return "pearson"
    if builtin_cost in {"matrix_spearman", "matrix_spearman_cross_penalty"}:
        return "spearman"
    return None



def recompute_builtin_plot_matrix(
    matrix: np.ndarray,
    builtin_cost: Optional[str],
    axis: int = -1,
    plot_matrix_source: str = "recomputed_similarity",
    feature_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    metric = infer_pairwise_metric_from_builtin_cost(builtin_cost)
    if metric is None:
        raise ValueError(
            f"Cannot recompute a plot matrix for builtin_cost={builtin_cost!r}. "
            "Use a pairwise built-in cost such as matrix_correlation or matrix_cosine."
        )

    similarity = _compute_pairwise_similarity_matrix(
        matrix=matrix,
        axis=axis,
        metric=metric,
        feature_weights=feature_weights,
    )
    if plot_matrix_source == "recomputed_similarity":
        return similarity
    if plot_matrix_source == "recomputed_distance":
        distance = 1.0 - similarity
        distance = 0.5 * (distance + distance.T)
        np.fill_diagonal(distance, 0.0)
        return distance
    raise ValueError(f"Unsupported plot_matrix_source: {plot_matrix_source}")



def make_matrix_sse_cost(
    matrix: np.ndarray,
    axis: int = -1,
    feature_weights: Optional[np.ndarray] = None,
) -> IntervalCostFn:
    """Return a segment cost that measures within-segment SSE for timestep feature vectors."""
    features = _prepare_timestep_feature_matrix(matrix, axis=axis)
    weights = _normalize_feature_weights(feature_weights, features.shape[0])
    if weights is not None:
        features = features * np.sqrt(weights).reshape(-1, 1)
    num_timesteps = features.shape[-1]
    prefix_sum = np.zeros((features.shape[0], num_timesteps + 1), dtype=np.float64)
    prefix_sum_sq = np.zeros((features.shape[0], num_timesteps + 1), dtype=np.float64)
    prefix_sum[:, 1:] = np.cumsum(features, axis=1)
    prefix_sum_sq[:, 1:] = np.cumsum(features ** 2, axis=1)

    def cost(start: int, end: int) -> float:
        if not (0 <= start < end <= num_timesteps):
            raise ValueError(f"Invalid interval [{start}, {end}) for {num_timesteps} timesteps")
        count = end - start
        sums = prefix_sum[:, end] - prefix_sum[:, start]
        sums_sq = prefix_sum_sq[:, end] - prefix_sum_sq[:, start]
        return float(sums_sq.sum() - np.square(sums).sum() / count)

    return cost



def make_matrix_pairwise_distance_cost(
    matrix: np.ndarray,
    axis: int = -1,
    metric: str = "cosine",
    eps: float = 1e-12,
    normalization: str = "size",
    feature_weights: Optional[np.ndarray] = None,
) -> IntervalCostFn:
    """Return a block cost from pairwise distances between timestep feature vectors."""
    similarity = _compute_pairwise_similarity_matrix(
        matrix=matrix,
        axis=axis,
        metric=metric,
        eps=eps,
        feature_weights=feature_weights,
    )
    distance = 1.0 - similarity
    distance = 0.5 * (distance + distance.T)
    np.fill_diagonal(distance, 0.0)

    prefix = _make_2d_prefix_sums(distance)

    def cost(start: int, end: int) -> float:
        if not (0 <= start < end <= distance.shape[0]):
            raise ValueError(f"Invalid interval [{start}, {end}) for {distance.shape[0]} timesteps")
        block_size = end - start
        block_sum = _square_block_sum(prefix, start, end)
        total = 0.5 * block_sum
        if normalization == "size":
            total /= block_size
        elif normalization != "none":
            raise ValueError(f"Unsupported pairwise normalization: {normalization}")
        return float(max(0.0, total))

    return cost



def make_matrix_cosine_cost(
    matrix: np.ndarray,
    axis: int = -1,
    normalization: str = "size",
    feature_weights: Optional[np.ndarray] = None,
) -> IntervalCostFn:
    return make_matrix_pairwise_distance_cost(
        matrix=matrix,
        axis=axis,
        metric="cosine",
        normalization=normalization,
        feature_weights=feature_weights,
    )



def make_matrix_correlation_cost(
    matrix: np.ndarray,
    axis: int = -1,
    normalization: str = "size",
    feature_weights: Optional[np.ndarray] = None,
) -> IntervalCostFn:
    return make_matrix_pairwise_distance_cost(
        matrix=matrix,
        axis=axis,
        metric="pearson",
        normalization=normalization,
        feature_weights=feature_weights,
    )



def make_matrix_spearman_cost(
    matrix: np.ndarray,
    axis: int = -1,
    normalization: str = "size",
    feature_weights: Optional[np.ndarray] = None,
) -> IntervalCostFn:
    """Return a rank-correlation segment cost robust to feature magnitudes."""
    return make_matrix_pairwise_distance_cost(
        matrix=matrix,
        axis=axis,
        metric="spearman",
        normalization=normalization,
        feature_weights=feature_weights,
    )



def build_pairwise_cross_block_score_table(
    matrix: np.ndarray,
    axis: int = -1,
    metric: str = "correlation",
    cross_block_lambda: float = 0.0,
    within_block_normalization: str = "size",
    eps: float = 1e-12,
    feature_weights: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float]:
    """Return an additive score table equivalent to a cross-block-penalized objective.

    The objective uses only off-diagonal pairs for the within-block reward:
        sum_k reward(B_k) - lambda * sum_{i,j in different blocks}(S_ij)

    where reward(B_k) can be normalized by the number of off-diagonal pairs, by
    block size, or left unnormalized. Excluding the diagonal avoids the degenerate
    singleton preference caused by S_ii = 1. Since the cross-block sum equals a
    partition-independent constant minus the within-block off-diagonal sum, the DP
    can maximize the equivalent additive objective:
        const + sum_k [reward(B_k) + lambda * within_offdiag_sum(B_k)]

    This helper returns the block-score table for that reduced objective together
    with the off-diagonal similarity constant needed to reconstruct the original score.
    """
    if cross_block_lambda < 0:
        raise ValueError(f"cross_block_lambda must be non-negative, got {cross_block_lambda}")

    similarity = _compute_pairwise_similarity_matrix(
        matrix=matrix,
        axis=axis,
        metric=metric,
        eps=eps,
        feature_weights=feature_weights,
    )
    num_timesteps = similarity.shape[0]

    offdiag_similarity = similarity.copy()
    np.fill_diagonal(offdiag_similarity, 0.0)
    offdiag_prefix = _make_2d_prefix_sums(offdiag_similarity)
    total_offdiag_similarity = float(offdiag_similarity.sum())

    score_table = np.full((num_timesteps + 1, num_timesteps + 1), -np.inf, dtype=np.float64)
    for start in range(num_timesteps):
        for end in range(start + 1, num_timesteps + 1):
            block_size = end - start
            within_offdiag_sum = _square_block_sum(offdiag_prefix, start, end)
            if block_size <= 1:
                reward = 0.0
            elif within_block_normalization == "pairs":
                reward = within_offdiag_sum / float(block_size * (block_size - 1))
            elif within_block_normalization == "size":
                reward = within_offdiag_sum / float(block_size)
            elif within_block_normalization == "none":
                reward = within_offdiag_sum
            else:
                raise ValueError(f"Unsupported cross-block reward normalization: {within_block_normalization}")
            score_table[start, end] = reward + (cross_block_lambda * within_offdiag_sum)
    return score_table, total_offdiag_similarity



def build_interval_cost_table(num_timesteps: int, cost_fn: IntervalCostFn) -> np.ndarray:
    """Precompute interval costs so the DP only needs table lookups."""
    if num_timesteps <= 0:
        raise ValueError(f"num_timesteps must be positive, got {num_timesteps}")

    interval_costs = np.full((num_timesteps + 1, num_timesteps + 1), np.inf, dtype=np.float64)
    for start in range(num_timesteps):
        for end in range(start + 1, num_timesteps + 1):
            interval_costs[start, end] = float(cost_fn(start, end))
    return interval_costs



def infer_num_timesteps_from_interval_costs(interval_costs: np.ndarray) -> int:
    """Infer the underlying timestep count from a square interval-cost table."""
    costs = np.asarray(interval_costs, dtype=np.float64)
    if costs.ndim != 2 or costs.shape[0] != costs.shape[1]:
        raise ValueError(f"interval_costs must be a square 2D array, got shape {costs.shape}")
    if costs.shape[0] <= 1:
        raise ValueError(f"interval_costs must contain at least two rows, got shape {costs.shape}")

    diagonal = np.diag(costs)
    if np.any(np.isfinite(diagonal)):
        return int(costs.shape[0])
    return int(costs.shape[0] - 1)



def normalize_interval_costs(
    interval_costs: np.ndarray,
    num_timesteps: Optional[int] = None,
) -> tuple[np.ndarray, int]:
    """Normalize interval costs to an (N + 1, N + 1) table.

    Supported input layouts:
    - (N, N): cost[start, end - 1] gives the cost of [start, end).
    - (N + 1, N + 1): cost[start, end] gives the cost of [start, end).
    """
    costs = np.asarray(interval_costs, dtype=np.float64)
    if costs.ndim != 2 or costs.shape[0] != costs.shape[1]:
        raise ValueError(f"interval_costs must be a square 2D array, got shape {costs.shape}")

    if num_timesteps is None:
        num_timesteps = infer_num_timesteps_from_interval_costs(costs)

    if costs.shape == (num_timesteps + 1, num_timesteps + 1):
        return costs.copy(), num_timesteps

    if costs.shape == (num_timesteps, num_timesteps):
        table = np.full((num_timesteps + 1, num_timesteps + 1), np.inf, dtype=np.float64)
        for start in range(num_timesteps):
            table[start, start + 1 : num_timesteps + 1] = costs[start, start:num_timesteps]
        return table, num_timesteps

    raise ValueError(
        "interval_costs shape does not match num_timesteps. "
        f"Got shape {costs.shape} and num_timesteps={num_timesteps}."
    )



def infer_num_timesteps_from_array(array: np.ndarray, axis: int = -1) -> int:
    """Infer the timestep count from an arbitrary array or matrix."""
    arr = np.asarray(array)
    if arr.ndim == 0:
        raise ValueError("Cannot infer num_timesteps from a scalar")
    if arr.ndim == 1:
        return int(arr.shape[0])

    resolved_axis = axis if axis >= 0 else arr.ndim + axis
    if resolved_axis < 0 or resolved_axis >= arr.ndim:
        raise ValueError(f"matrix_axis={axis} is out of bounds for array with shape {arr.shape}")
    return int(arr.shape[resolved_axis])



def plot_partitioned_matrix(
    matrix: np.ndarray,
    boundaries: Sequence[int],
    out_path: str,
    title: str = "Timestep Correlation Matrix with Partition Blocks",
    labels: Optional[Sequence[str]] = None,
) -> None:
    """Plot a square matrix and draw one square overlay per partition block."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"plot matrix must be square, got shape {arr.shape}")

    num_timesteps = arr.shape[0]
    if list(boundaries) != sorted(boundaries):
        raise ValueError(f"boundaries must be sorted, got {boundaries}")
    if boundaries[0] != 0 or boundaries[-1] != num_timesteps:
        raise ValueError(
            f"boundaries must span [0, {num_timesteps}], got {boundaries[0]}..{boundaries[-1]}"
        )

    if labels is not None and len(labels) != num_timesteps:
        raise ValueError(
            f"plot labels length ({len(labels)}) does not match matrix size ({num_timesteps})"
        )

    finite_values = arr[np.isfinite(arr)]
    if finite_values.size == 0:
        raise ValueError("plot matrix does not contain any finite values")

    use_correlation_scale = float(finite_values.min()) >= -1.01 and float(finite_values.max()) <= 1.01
    figsize = max(6.0, min(14.0, 0.45 * num_timesteps))
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    im = ax.imshow(
        arr,
        aspect="equal",
        interpolation="nearest",
        cmap="coolwarm" if use_correlation_scale else "viridis",
        vmin=-1.0 if use_correlation_scale else None,
        vmax=1.0 if use_correlation_scale else None,
    )

    if labels is None:
        labels = [str(i) for i in range(num_timesteps)]
    tick_count = num_timesteps if num_timesteps <= 30 else min(12, num_timesteps)
    tick_positions = np.unique(np.linspace(0, num_timesteps - 1, num=tick_count, dtype=int))
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels([labels[idx] for idx in tick_positions], rotation=90, fontsize=8)
    ax.set_yticklabels([labels[idx] for idx in tick_positions], fontsize=8)

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        size = end - start
        outer = Rectangle(
            (start - 0.5, start - 0.5),
            size,
            size,
            fill=False,
            edgecolor="black",
            linewidth=2.5,
        )
        inner = Rectangle(
            (start - 0.5, start - 0.5),
            size,
            size,
            fill=False,
            edgecolor="white",
            linewidth=1.2,
        )
        ax.add_patch(outer)
        ax.add_patch(inner)

    ax.set_xlim(-0.5, num_timesteps - 0.5)
    ax.set_ylim(num_timesteps - 0.5, -0.5)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Timestep")
    ax.set_title(title)
    colorbar = fig.colorbar(im, ax=ax)
    colorbar.set_label("Correlation" if use_correlation_scale else "Value")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_partitioned_series(
    values: Sequence[float],
    boundaries: Sequence[int],
    out_path: str,
    title: str = "Metric Curve with Partition Blocks",
    ylabel: str = "Value",
    labels: Optional[Sequence[str]] = None,
) -> None:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"plot series must be 1D, got shape {arr.shape}")

    num_timesteps = arr.shape[0]
    if list(boundaries) != sorted(boundaries):
        raise ValueError(f"boundaries must be sorted, got {boundaries}")
    if boundaries[0] != 0 or boundaries[-1] != num_timesteps:
        raise ValueError(
            f"boundaries must span [0, {num_timesteps}], got {boundaries[0]}..{boundaries[-1]}"
        )
    if labels is not None and len(labels) != num_timesteps:
        raise ValueError(
            f"plot labels length ({len(labels)}) does not match series length ({num_timesteps})"
        )

    x = np.arange(num_timesteps)
    figsize = (max(8.0, min(16.0, 0.35 * num_timesteps)), 4.8)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, arr, color="tab:blue", linewidth=2.0)

    finite_values = arr[np.isfinite(arr)]
    if finite_values.size == 0:
        ymin, ymax = 0.0, 1.0
    else:
        ymin = float(finite_values.min())
        ymax = float(finite_values.max())
        if math.isclose(ymin, ymax):
            pad = max(1e-6, abs(ymin) * 0.05, 0.05)
            ymin -= pad
            ymax += pad
        else:
            pad = 0.08 * (ymax - ymin)
            ymin -= pad
            ymax += pad

    phase_colors = ["#f3f6fb", "#ebf0f8"]
    label_y = ymin + 0.5 * (ymax - ymin)
    for block_idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        ax.axvspan(start - 0.5, end - 0.5, color=phase_colors[block_idx % len(phase_colors)], alpha=0.8, zorder=0)
        center = (start + end - 1) / 2.0
        ax.text(
            center,
            label_y,
            f"B{block_idx}",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="#1f2933",
            zorder=4,
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": "#4b5563",
                "linewidth": 0.8,
                "alpha": 0.92,
            },
        )
    for boundary in boundaries[1:-1]:
        ax.axvline(boundary - 0.5, color="black", linewidth=1.5, alpha=0.75)

    if labels is None:
        labels = [str(i) for i in range(num_timesteps)]
    tick_count = num_timesteps if num_timesteps <= 30 else min(12, num_timesteps)
    tick_positions = np.unique(np.linspace(0, num_timesteps - 1, num=tick_count, dtype=int))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([labels[idx] for idx in tick_positions], rotation=90, fontsize=8)
    ax.set_xlim(-0.5, num_timesteps - 0.5)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Timestep")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def compute_total_cost_curve(
    num_timesteps: int,
    interval_costs: np.ndarray,
    max_blocks: int,
    min_blocks: int = 1,
    min_block_size: int = 1,
) -> tuple[list[int], list[float]]:
    """Compute the optimal total DP cost for each block count in a sweep."""
    if min_block_size < 1:
        raise ValueError(f"min_block_size must be at least 1, got {min_block_size}")
    if min_blocks < 1:
        raise ValueError(f"min_blocks must be at least 1, got {min_blocks}")
    if max_blocks < min_blocks:
        raise ValueError(f"max_blocks ({max_blocks}) must be >= min_blocks ({min_blocks})")
    max_feasible_blocks = num_timesteps // min_block_size
    if max_blocks > max_feasible_blocks:
        raise ValueError(
            f"max_blocks ({max_blocks}) cannot exceed {max_feasible_blocks} when "
            f"num_timesteps={num_timesteps} and min_block_size={min_block_size}"
        )

    block_counts: list[int] = []
    total_costs: list[float] = []
    for num_blocks in range(min_blocks, max_blocks + 1):
        result = optimal_timestep_partition(
            num_timesteps=num_timesteps,
            num_blocks=num_blocks,
            interval_costs=interval_costs,
            min_block_size=min_block_size,
        )
        block_counts.append(num_blocks)
        total_costs.append(result.total_cost)
    return block_counts, total_costs



def cross_penalty_total_scores(
    total_costs: Sequence[float],
    cross_block_lambda: float,
    total_cross_block_similarity: float,
) -> list[float]:
    """Convert reduced DP costs into comparable cross-penalty objective scores."""
    if cross_block_lambda < 0:
        raise ValueError(f"cross_block_lambda must be non-negative, got {cross_block_lambda}")
    return [
        float(-cost - (cross_block_lambda * total_cross_block_similarity))
        for cost in total_costs
    ]



def select_num_blocks_by_total_score(
    block_counts: Sequence[int],
    total_scores: Sequence[float],
) -> int:
    """Select the score-maximizing K, preferring the smaller K on an exact tie."""
    if len(block_counts) != len(total_scores):
        raise ValueError("block_counts and total_scores must have the same length")
    if not block_counts:
        raise ValueError("At least one candidate block count is required")
    candidates = [(int(count), float(score)) for count, score in zip(block_counts, total_scores)]
    if any(count < 1 for count, _ in candidates):
        raise ValueError("Candidate block counts must be positive")
    if any(not math.isfinite(score) for _, score in candidates):
        raise ValueError("Candidate total scores must be finite")
    return min(candidates, key=lambda item: (-item[1], item[0]))[0]



def plot_total_cost_curve(
    block_counts: Sequence[int],
    total_costs: Sequence[float],
    out_path: str,
    title: str = "Total DP Cost vs. Number of Blocks",
    ylabel: str = "Total DP Cost",
    selected_num_blocks: Optional[int] = None,
) -> None:
    """Plot the optimal total DP value as a function of block count."""
    if len(block_counts) != len(total_costs):
        raise ValueError("block_counts and total_costs must have the same length")
    if not block_counts:
        raise ValueError("cost-curve plot requires at least one point")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(block_counts, total_costs, marker="o", linewidth=1.8)
    if selected_num_blocks is not None and selected_num_blocks in block_counts:
        selected_idx = list(block_counts).index(selected_num_blocks)
        ax.scatter(
            [block_counts[selected_idx]],
            [total_costs[selected_idx]],
            color="crimson",
            s=50,
            zorder=3,
            label=f"selected M={selected_num_blocks}",
        )
        ax.legend()
    ax.set_xlabel("Number of Blocks")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)



def optimal_timestep_partition(
    num_timesteps: int,
    num_blocks: int,
    cost_fn: Optional[IntervalCostFn] = None,
    interval_costs: Optional[np.ndarray] = None,
    min_block_size: int = 1,
) -> PartitionResult:
    """Find the minimum-cost partition of ordered timesteps into contiguous blocks.

    The cost for a block covering timesteps [start, end) is provided either through
    ``cost_fn(start, end)`` or by passing a precomputed interval cost table.
    """
    if num_timesteps <= 0:
        raise ValueError(f"num_timesteps must be positive, got {num_timesteps}")
    if num_blocks <= 0:
        raise ValueError(f"num_blocks must be positive, got {num_blocks}")
    if min_block_size <= 0:
        raise ValueError(f"min_block_size must be positive, got {min_block_size}")
    if num_blocks > num_timesteps:
        raise ValueError(f"num_blocks ({num_blocks}) cannot exceed num_timesteps ({num_timesteps})")
    if num_blocks * min_block_size > num_timesteps:
        raise ValueError(
            f"Cannot partition {num_timesteps} timesteps into {num_blocks} blocks with "
            f"min_block_size={min_block_size}"
        )
    if (cost_fn is None) == (interval_costs is None):
        raise ValueError("Provide exactly one of cost_fn or interval_costs")

    if interval_costs is None:
        interval_costs = build_interval_cost_table(num_timesteps=num_timesteps, cost_fn=cost_fn)
    else:
        interval_costs, inferred_timesteps = normalize_interval_costs(interval_costs, num_timesteps=num_timesteps)
        if inferred_timesteps != num_timesteps:
            raise ValueError(
                f"Normalized interval costs imply {inferred_timesteps} timesteps, "
                f"but num_timesteps={num_timesteps}"
            )

    dp = np.full((num_blocks + 1, num_timesteps + 1), np.inf, dtype=np.float64)
    backpointers = np.full((num_blocks + 1, num_timesteps + 1), -1, dtype=np.int64)
    dp[0, 0] = 0.0

    for block_count in range(1, num_blocks + 1):
        min_end = block_count * min_block_size
        max_end = num_timesteps - (num_blocks - block_count) * min_block_size
        for end in range(min_end, max_end + 1):
            best_cost = math.inf
            best_start = -1
            min_start = (block_count - 1) * min_block_size
            for start in range(min_start, end - min_block_size + 1):
                prev_cost = dp[block_count - 1, start]
                segment_cost = interval_costs[start, end]
                if not math.isfinite(prev_cost) or not math.isfinite(segment_cost):
                    continue
                candidate = prev_cost + segment_cost
                if candidate < best_cost:
                    best_cost = candidate
                    best_start = start
            dp[block_count, end] = best_cost
            backpointers[block_count, end] = best_start

    total_cost = float(dp[num_blocks, num_timesteps])
    if not math.isfinite(total_cost):
        raise RuntimeError("Failed to find a finite partition cost. Check the provided cost function.")

    boundaries = [num_timesteps]
    current_end = num_timesteps
    for block_count in range(num_blocks, 0, -1):
        start = int(backpointers[block_count, current_end])
        if start < 0:
            raise RuntimeError("Backtracking failed while reconstructing the optimal partition")
        boundaries.append(start)
        current_end = start
    boundaries.reverse()

    blocks = [
        PartitionBlock(
            start=start,
            end=end,
            size=end - start,
            cost=float(interval_costs[start, end]),
        )
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    return PartitionResult(
        num_timesteps=num_timesteps,
        num_blocks=num_blocks,
        min_block_size=min_block_size,
        total_cost=total_cost,
        boundaries=boundaries,
        blocks=blocks,
    )



def load_value(path: str, key: Optional[str] = None) -> Any:
    source_path = Path(path)
    suffix = source_path.suffix.lower()
    if is_json_path(source_path):
        # Plain or gzip-compressed JSON (released profiles use ``.json.gz``).
        payload = load_json(source_path)
        if key is None:
            return payload
        if not isinstance(payload, dict):
            raise ValueError(f"JSON file {path} is not an object, so a key cannot be selected")
        if key not in payload:
            raise ValueError(f"Key {key!r} was not found in {path}")
        return payload[key]
    if suffix == ".npy":
        if key is not None:
            raise ValueError(f"Cannot use a key with .npy input {path}")
        return np.load(source_path)
    if suffix == ".npz":
        data = np.load(source_path)
        chosen_key = key or data.files[0]
        return data[chosen_key]
    raise ValueError(f"Unsupported file format for {path}. Expected .json, .json.gz, .npy, or .npz")



def load_numeric_array(path: str, array_key: Optional[str] = None) -> np.ndarray:
    return np.asarray(load_value(path, key=array_key), dtype=np.float64)


def load_matrix_feature_expansion(
    path: str,
    *,
    matrix: np.ndarray,
    matrix_key: Optional[str],
    matrix_axis: int,
) -> tuple[Optional[np.ndarray], Optional[dict[str, Any]]]:
    """Load HT row weights when a JSON matrix contains sampled group rows."""

    source_path = Path(path)
    if not is_json_path(source_path):
        return None, None
    payload = load_value(path, key=None)
    if not isinstance(payload, Mapping):
        return None, None
    weighting = expansion_weighting_summary(payload)
    if weighting is None:
        return None, None
    raw_names = payload.get("group_names")
    if not isinstance(raw_names, Sequence) or isinstance(raw_names, (str, bytes)):
        raise ValueError("sampled profile matrix is missing group_names")
    names = [str(name) for name in raw_names]
    num_feature_rows = _prepare_timestep_feature_matrix(matrix, axis=matrix_axis).shape[0]
    group_matrix_keys = {
        "delta_stack",
        "signed_delta_stack",
        "relative_delta_stack",
        "row_normalized_relative_delta_stack",
    }
    if matrix_key not in group_matrix_keys:
        return None, {
            **weighting,
            "application": "not_applied_matrix_is_not_group_by_timestep",
            "matrix_key": matrix_key,
        }
    if num_feature_rows != len(names):
        raise ValueError(
            f"sampled profile matrix {matrix_key!r} has {num_feature_rows} feature rows, "
            f"but group_names has {len(names)} entries"
        )
    weights = np.asarray(aligned_group_expansion_weights(payload, names), dtype=np.float64)
    return weights, {
        **weighting,
        "application": "weighted_group_rows_for_timestep_similarity",
        "matrix_key": matrix_key,
    }



def load_plot_labels(path: str, key: Optional[str] = None) -> list[str]:
    payload = load_value(path, key=None)
    if key is None:
        if isinstance(payload, dict):
            for candidate in ("sigma_bin_labels", "timestep_bin_labels"):
                if candidate in payload:
                    return normalize_plot_labels(payload[candidate])
        raise ValueError(f"Could not infer plot labels from {path}")

    if not isinstance(payload, dict):
        return normalize_plot_labels(load_value(path, key=key))

    if key in payload:
        return normalize_plot_labels(payload[key])

    fallback_keys = {
        "sigma_bin_labels": ["timestep_bin_labels"],
        "timestep_bin_labels": ["sigma_bin_labels"],
    }
    for fallback_key in fallback_keys.get(key, []):
        if fallback_key in payload:
            return normalize_plot_labels(payload[fallback_key])

    raise ValueError(f"Key {key!r} was not found in {path}")



def normalize_plot_labels(labels: Any) -> list[str]:
    label_array = np.asarray(labels, dtype=object)
    if label_array.ndim != 1:
        raise ValueError(f"plot labels must be 1D, got shape {label_array.shape}")
    return [str(value) for value in label_array.tolist()]



def load_callable(spec: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise ValueError("cost function spec must look like module_or_file.py:function_name")

    module_spec, function_name = spec.rsplit(":", 1)
    module: Any
    candidate_path = Path(module_spec)
    if candidate_path.exists() or module_spec.endswith(".py"):
        module_path = candidate_path.resolve()
        spec_obj = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if spec_obj is None or spec_obj.loader is None:
            raise ValueError(f"Could not import module from {module_path}")
        module = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(module)
    else:
        module = importlib.import_module(module_spec)

    try:
        return getattr(module, function_name)
    except AttributeError as exc:
        raise ValueError(f"Function {function_name!r} was not found in {module_spec!r}") from exc



def supports_matrix_keyword(fn: Callable[..., Any]) -> bool:
    signature = inspect.signature(fn)
    if "matrix" in signature.parameters:
        return True
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())



def build_result_payload(
    result: PartitionResult,
    array_source: Optional[str] = None,
    array_key: Optional[str] = None,
    matrix_shape: Optional[Sequence[int]] = None,
    plot_output: Optional[str] = None,
    plot_matrix_source: Optional[str] = None,
    builtin_cost: Optional[str] = None,
    pairwise_metric: Optional[str] = None,
    pairwise_normalization: Optional[str] = None,
    cross_block_lambda: Optional[float] = None,
    cross_block_reward_normalization: Optional[str] = None,
    total_reduced_score: Optional[float] = None,
    total_score: Optional[float] = None,
    block_reduced_scores: Optional[Sequence[float]] = None,
    cost_curve_output: Optional[str] = None,
    cost_curve_block_counts: Optional[Sequence[int]] = None,
    cost_curve_total_costs: Optional[Sequence[float]] = None,
    cost_curve_total_scores: Optional[Sequence[float]] = None,
    plot_series_output: Optional[str] = None,
    plot_series_key: Optional[str] = None,
    num_blocks_selection: Optional[Mapping[str, Any]] = None,
    profile_provenance: Optional[Mapping[str, Any]] = None,
    matrix_row_weighting: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    payload = result.to_dict()
    payload["intervals"] = [f"[{block.start}, {block.end})" for block in result.blocks]
    if block_reduced_scores is not None:
        if len(block_reduced_scores) != len(payload["blocks"]):
            raise ValueError("block_reduced_scores length must match the number of blocks")
        for block_payload, reduced_score in zip(payload["blocks"], block_reduced_scores):
            block_payload["reduced_score"] = float(reduced_score)
    if array_source is not None:
        payload["array_source"] = array_source
    if array_key is not None:
        payload["array_key"] = array_key
    if matrix_shape is not None:
        payload["matrix_shape"] = list(matrix_shape)
    if plot_output is not None:
        payload["plot_output"] = plot_output
    if plot_matrix_source is not None:
        payload["plot_matrix_source"] = plot_matrix_source
    if plot_series_output is not None:
        payload["plot_series_output"] = plot_series_output
    if plot_series_key is not None:
        payload["plot_series_key"] = plot_series_key
    if builtin_cost is not None:
        payload["builtin_cost"] = builtin_cost
    if pairwise_metric is not None:
        payload["pairwise_metric"] = pairwise_metric
    if pairwise_normalization is not None:
        payload["pairwise_normalization"] = pairwise_normalization
    if cross_block_lambda is not None:
        payload["cross_block_lambda"] = float(cross_block_lambda)
    if cross_block_reward_normalization is not None:
        payload["cross_block_reward_normalization"] = cross_block_reward_normalization
    if total_reduced_score is not None:
        payload["total_reduced_score"] = float(total_reduced_score)
    if total_score is not None:
        payload["total_score"] = float(total_score)
    if cost_curve_output is not None:
        payload["cost_curve_output"] = cost_curve_output
    if cost_curve_block_counts is not None and cost_curve_total_costs is not None:
        payload["cost_curve"] = {
            "num_blocks": list(cost_curve_block_counts),
            "total_costs": [float(value) for value in cost_curve_total_costs],
        }
        if cost_curve_total_scores is not None:
            payload["cost_curve"]["total_scores"] = [float(value) for value in cost_curve_total_scores]
    if num_blocks_selection is not None:
        payload["num_blocks_selection"] = dict(num_blocks_selection)
    if profile_provenance is not None:
        payload.update(profile_provenance)
    if matrix_row_weighting is not None:
        payload["matrix_row_weighting"] = dict(matrix_row_weighting)
    return payload



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_timesteps", type=int, default=None, help="Optional explicit number of ordered timesteps/items to partition")
    parser.add_argument("--num_blocks", type=str, default=None,
                        help="Number of contiguous blocks, or 'auto' to pick the block count "
                             "that maximizes the total DP score (argmax of the score curve; "
                             "ties go to the smaller K). Default: 'auto' (the paper setting), "
                             "or 4 as the upper bound of --select-num-blocks.")
    parser.add_argument(
        "--min-block-size",
        "--min_block_size",
        dest="min_block_size",
        type=int,
        default=1,
        help="Minimum number of ordered timesteps/items in every block (default: 1)",
    )
    parser.add_argument(
        "--select-num-blocks",
        "--select_num_blocks",
        dest="select_num_blocks",
        action="store_true",
        help=(
            "Select K by maximizing the comparable total score of a *_cross_penalty objective. "
            "By default K remains fixed at --num_blocks."
        ),
    )
    parser.add_argument(
        "--max-num-blocks",
        "--max_num_blocks",
        dest="max_num_blocks",
        type=int,
        default=None,
        help="Largest K considered by --select-num-blocks (default: --num_blocks)",
    )
    parser.add_argument("--values", type=str, default=None, help="Optional .json/.npy/.npz file of scalar values. For results.json, also pass --array_key.")
    parser.add_argument("--interval_costs", type=str, default=None, help="Optional .json/.npy/.npz file of precomputed interval costs. For results.json, also pass --array_key.")
    parser.add_argument("--cost_function", type=str, default=None, help="Optional module:function or path.py:function callback that returns cost(start, end)")
    parser.add_argument(
        "--builtin_cost",
        type=str,
        default=None,
        choices=[
            "matrix_sse",
            "matrix_cosine",
            "matrix_correlation",
            "matrix_spearman",
            "matrix_cosine_cross_penalty",
            "matrix_correlation_cross_penalty",
            "matrix_spearman_cross_penalty",
        ],
        help=(
            "Built-in segment objective computed from --matrix. Default when no --values, "
            "--interval_costs or --cost_function is given: matrix_correlation_cross_penalty "
            "(the paper objective J(B)). matrix_correlation is the within-block distance cost "
            "without the cross-phase penalty."
        ),
    )
    parser.add_argument("--matrix", type=str, default=None, help="Optional .json/.npy/.npz matrix passed to --cost_function or used by --builtin_cost, and used to infer num_timesteps when omitted.")
    parser.add_argument("--matrix_key", type=str, default="relative_delta_stack", help="Optional key to load from --matrix when it is a JSON object or NPZ file.")
    parser.add_argument("--matrix_axis", type=int, default=-1, help="Axis of --matrix that represents timesteps when inferring num_timesteps.")
    parser.add_argument("--pairwise_normalization", type=str, default="size", choices=["size", "none"], help="Normalization for pairwise built-in costs like cosine and correlation")
    parser.add_argument("--cross_block_lambda", type=float, default=0.02, help="Lambda for *_cross_penalty objectives. Larger values penalize leaving high-similarity pairs across block boundaries.")
    parser.add_argument("--cross_block_reward_normalization", type=str, default="size", choices=["pairs", "size", "none"], help="Normalization used by the within-block reward term of *_cross_penalty objectives.")
    parser.add_argument("--cost_kwargs_json", type=str, default=None, help="Optional JSON object passed as keyword arguments to --cost_function")
    parser.add_argument("--array_key", type=str, default=None, help="Optional key to load from --values or --interval_costs when the file is JSON/NPZ.")
    parser.add_argument("--plot_output", type=str, default=None, help="Optional image path for plotting a matrix with partition block overlays")
    parser.add_argument("--plot_matrix_source", type=str, default="file", choices=["auto", "file", "recomputed_similarity", "recomputed_distance"], help="Choose whether to plot a file-backed matrix or a matrix recomputed from the built-in optimization features.")
    parser.add_argument("--plot_matrix", type=str, default=None, help="Optional .json/.npy/.npz matrix to visualize. If omitted, the script reuses the interval or input matrix when possible.")
    parser.add_argument("--plot_matrix_key", type=str, default="C_noise_levels", help="Optional key to load from --plot_matrix when it is a JSON object or NPZ file.")
    parser.add_argument("--plot_labels_key", type=str, default=None, help="Optional key, usually from results.json, that provides axis labels for the visualization.")
    parser.add_argument("--plot_title", type=str, default="Timestep Correlation Matrix with Partition Blocks", help="Title for the visualization plot")
    parser.add_argument("--plot_series_output", type=str, default=None, help="Optional image path for plotting a 1D metric curve with partition overlays")
    parser.add_argument("--plot_series", type=str, default=None, help="Optional .json/.npy/.npz series to visualize. If omitted, the script reuses --values when available.")
    parser.add_argument("--plot_series_key", type=str, default="n_eff", help="Optional key to load from --plot_series when it is a JSON object or NPZ file.")
    parser.add_argument("--plot_series_title", type=str, default="Metric Curve with Partition Blocks", help="Title for the 1D metric visualization")
    parser.add_argument("--plot_series_ylabel", type=str, default="Value", help="Y-axis label for the 1D metric visualization")
    parser.add_argument("--cost_curve_output", type=str, default=None, help="Optional image path for plotting total DP cost as a function of num_blocks")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save the resulting partition as JSON")
    args = parser.parse_args()
    if args.builtin_cost is None and args.values is None and args.interval_costs is None and args.cost_function is None:
        args.builtin_cost = "matrix_correlation_cross_penalty"
    if args.num_blocks is None:
        args.num_blocks = "4" if args.select_num_blocks else "auto"
    return args



def main() -> None:
    args = parse_args()
    sources = [
        args.values is not None,
        args.interval_costs is not None,
        args.cost_function is not None,
        args.builtin_cost is not None,
    ]
    if sum(sources) != 1:
        raise ValueError("Provide exactly one of --values, --interval_costs, --cost_function, or --builtin_cost")

    matrix = None
    inferred_from_matrix = None
    matrix_feature_weights = None
    matrix_row_weighting = None
    if args.matrix is not None:
        matrix = load_numeric_array(args.matrix, array_key=args.matrix_key)
        inferred_from_matrix = infer_num_timesteps_from_array(matrix, axis=args.matrix_axis)
        matrix_feature_weights, matrix_row_weighting = load_matrix_feature_expansion(
            args.matrix,
            matrix=matrix,
            matrix_key=args.matrix_key,
            matrix_axis=args.matrix_axis,
        )

    default_plot_matrix = None
    default_plot_source = None
    if matrix is not None and matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
        default_plot_matrix = matrix
        default_plot_source = args.matrix

    array_source = None
    array_key = None
    num_timesteps: Optional[int] = None
    resolved_interval_costs: Optional[np.ndarray] = None
    score_table: Optional[np.ndarray] = None
    total_cross_block_similarity: Optional[float] = None

    if args.values is not None:
        values = load_numeric_array(args.values, array_key=args.array_key)
        if values.ndim != 1:
            raise ValueError(f"--values must resolve to a 1D array, got shape {values.shape}")
        num_timesteps = args.num_timesteps if args.num_timesteps is not None else int(values.shape[0])
        if num_timesteps != int(values.shape[0]):
            raise ValueError(
                f"--num_timesteps ({num_timesteps}) does not match the number of values ({values.shape[0]})"
            )
        resolved_interval_costs = build_interval_cost_table(
            num_timesteps=num_timesteps,
            cost_fn=make_sse_cost(values),
        )
        array_source = args.values
        array_key = args.array_key
    elif args.interval_costs is not None:
        raw_interval_costs = load_numeric_array(args.interval_costs, array_key=args.array_key)
        requested_num_timesteps = args.num_timesteps if args.num_timesteps is not None else None
        resolved_interval_costs, num_timesteps = normalize_interval_costs(
            raw_interval_costs,
            num_timesteps=requested_num_timesteps,
        )
        array_source = args.interval_costs
        array_key = args.array_key
        if raw_interval_costs.ndim == 2 and raw_interval_costs.shape[0] == raw_interval_costs.shape[1]:
            default_plot_matrix = raw_interval_costs
            default_plot_source = args.interval_costs
    elif args.builtin_cost is not None:
        if matrix is None:
            raise ValueError("--matrix is required when using --builtin_cost")
        num_timesteps = args.num_timesteps if args.num_timesteps is not None else inferred_from_matrix
        if num_timesteps is None:
            raise ValueError("--num_timesteps is required when using --builtin_cost unless --matrix is provided")
        if args.builtin_cost == "matrix_sse":
            builtin_cost_fn = make_matrix_sse_cost(
                matrix,
                axis=args.matrix_axis,
                feature_weights=matrix_feature_weights,
            )
            resolved_interval_costs = build_interval_cost_table(
                num_timesteps=num_timesteps,
                cost_fn=builtin_cost_fn,
            )
        elif args.builtin_cost == "matrix_cosine":
            builtin_cost_fn = make_matrix_cosine_cost(
                matrix,
                axis=args.matrix_axis,
                normalization=args.pairwise_normalization,
                feature_weights=matrix_feature_weights,
            )
            resolved_interval_costs = build_interval_cost_table(
                num_timesteps=num_timesteps,
                cost_fn=builtin_cost_fn,
            )
        elif args.builtin_cost == "matrix_correlation":
            builtin_cost_fn = make_matrix_correlation_cost(
                matrix,
                axis=args.matrix_axis,
                normalization=args.pairwise_normalization,
                feature_weights=matrix_feature_weights,
            )
            resolved_interval_costs = build_interval_cost_table(
                num_timesteps=num_timesteps,
                cost_fn=builtin_cost_fn,
            )
        elif args.builtin_cost == "matrix_spearman":
            builtin_cost_fn = make_matrix_spearman_cost(
                matrix,
                axis=args.matrix_axis,
                normalization=args.pairwise_normalization,
            )
            resolved_interval_costs = build_interval_cost_table(
                num_timesteps=num_timesteps,
                cost_fn=builtin_cost_fn,
            )
        elif args.builtin_cost in {
            "matrix_cosine_cross_penalty",
            "matrix_correlation_cross_penalty",
            "matrix_spearman_cross_penalty",
        }:
            score_metric = infer_pairwise_metric_from_builtin_cost(args.builtin_cost)
            if score_metric is None:
                raise RuntimeError(f"Could not infer pairwise metric for {args.builtin_cost}")
            score_table, total_cross_block_similarity = build_pairwise_cross_block_score_table(
                matrix,
                axis=args.matrix_axis,
                metric=score_metric,
                cross_block_lambda=args.cross_block_lambda,
                within_block_normalization=args.cross_block_reward_normalization,
                feature_weights=matrix_feature_weights,
            )
            resolved_interval_costs = np.full(score_table.shape, np.inf, dtype=np.float64)
            finite_mask = np.isfinite(score_table)
            resolved_interval_costs[finite_mask] = -score_table[finite_mask]
        else:
            raise ValueError(f"Unsupported builtin cost: {args.builtin_cost}")
        array_source = args.matrix
        array_key = args.matrix_key
    else:
        num_timesteps = args.num_timesteps if args.num_timesteps is not None else inferred_from_matrix
        if num_timesteps is None:
            raise ValueError("--num_timesteps is required when using --cost_function unless --matrix is provided")
        raw_cost_fn = load_callable(args.cost_function)
        cost_kwargs = json.loads(args.cost_kwargs_json) if args.cost_kwargs_json else {}
        can_pass_matrix = matrix is not None and supports_matrix_keyword(raw_cost_fn)

        def wrapped_cost(start: int, end: int) -> float:
            kwargs = dict(cost_kwargs)
            if can_pass_matrix:
                kwargs["matrix"] = matrix
            return float(raw_cost_fn(start, end, **kwargs))

        resolved_interval_costs = build_interval_cost_table(
            num_timesteps=num_timesteps,
            cost_fn=wrapped_cost,
        )
        array_source = args.matrix
        array_key = args.matrix_key

    if num_timesteps is None or resolved_interval_costs is None:
        raise RuntimeError("Failed to resolve the DP problem definition")

    # --num_blocks: an explicit integer, or 'auto' = the block count with the highest total
    # DP score over every feasible K (lowest total cost when no score objective is active).
    # --select-num-blocks is the cross-penalty-scored selection over 1..--max-num-blocks.
    num_blocks_auto = str(args.num_blocks).lower() == "auto"
    if num_blocks_auto and args.select_num_blocks:
        raise ValueError("--num_blocks auto and --select-num-blocks are mutually exclusive")
    requested_num_blocks = None if num_blocks_auto else int(args.num_blocks)

    if args.min_block_size < 1:
        raise ValueError(f"--min-block-size must be at least 1, got {args.min_block_size}")
    max_feasible_blocks = num_timesteps // args.min_block_size
    if args.max_num_blocks is not None and not args.select_num_blocks:
        raise ValueError("--max-num-blocks requires --select-num-blocks")

    cost_curve_block_counts: Optional[list[int]] = None
    cost_curve_total_costs: Optional[list[float]] = None
    cost_curve_total_scores: Optional[list[float]] = None
    if args.select_num_blocks:
        if score_table is None or total_cross_block_similarity is None:
            raise ValueError(
                "--select-num-blocks requires a *_cross_penalty built-in cost so candidate "
                "total scores are comparable"
            )
        selection_max_blocks = args.max_num_blocks if args.max_num_blocks is not None else requested_num_blocks
        if selection_max_blocks < 1:
            raise ValueError(f"--max-num-blocks must be at least 1, got {selection_max_blocks}")
        cost_curve_block_counts, cost_curve_total_costs = compute_total_cost_curve(
            num_timesteps=num_timesteps,
            interval_costs=resolved_interval_costs,
            max_blocks=selection_max_blocks,
            min_block_size=args.min_block_size,
        )
        cost_curve_total_scores = cross_penalty_total_scores(
            cost_curve_total_costs,
            cross_block_lambda=args.cross_block_lambda,
            total_cross_block_similarity=total_cross_block_similarity,
        )
        selected_num_blocks = select_num_blocks_by_total_score(
            cost_curve_block_counts,
            cost_curve_total_scores,
        )
        num_blocks_selection: dict[str, Any] = {
            "policy": "maximize_cross_penalty_total_score",
            "selected_num_blocks": selected_num_blocks,
            "max_num_blocks": selection_max_blocks,
            "tie_break": "smallest_num_blocks",
            "candidates": [
                {
                    "num_blocks": count,
                    "total_cost": float(cost),
                    "total_score": float(score),
                }
                for count, cost, score in zip(
                    cost_curve_block_counts,
                    cost_curve_total_costs,
                    cost_curve_total_scores,
                )
            ],
        }
    elif num_blocks_auto:
        cost_curve_block_counts, cost_curve_total_costs = compute_total_cost_curve(
            num_timesteps=num_timesteps,
            interval_costs=resolved_interval_costs,
            max_blocks=max_feasible_blocks,
            min_blocks=1,
            min_block_size=args.min_block_size,
        )
        if score_table is not None:
            if total_cross_block_similarity is None:
                raise RuntimeError("Cross-block similarity constant is missing for the selected score objective")
            cost_curve_total_scores = cross_penalty_total_scores(
                cost_curve_total_costs,
                cross_block_lambda=args.cross_block_lambda,
                total_cross_block_similarity=total_cross_block_similarity,
            )
            selected_num_blocks = int(cost_curve_block_counts[int(np.argmax(cost_curve_total_scores))])  # highest score
        else:
            selected_num_blocks = int(cost_curve_block_counts[int(np.argmin(cost_curve_total_costs))])  # lowest cost
        print(f"[auto] selected num_blocks={selected_num_blocks} (max total DP score over 1..{max_feasible_blocks})")
        num_blocks_selection = {
            "policy": "auto_max_total_dp_score",
            "selected_num_blocks": selected_num_blocks,
            "max_num_blocks": max_feasible_blocks,
        }
    else:
        selected_num_blocks = requested_num_blocks
        num_blocks_selection = {
            "policy": "fixed",
            "requested_num_blocks": requested_num_blocks,
            "selected_num_blocks": selected_num_blocks,
        }

    result = optimal_timestep_partition(
        num_timesteps=num_timesteps,
        num_blocks=selected_num_blocks,
        interval_costs=resolved_interval_costs,
        min_block_size=args.min_block_size,
    )

    total_reduced_score: Optional[float] = None
    total_score: Optional[float] = None
    block_reduced_scores: Optional[list[float]] = None
    if score_table is not None:
        total_reduced_score = float(-result.total_cost)
        block_reduced_scores = [float(score_table[block.start, block.end]) for block in result.blocks]
        if total_cross_block_similarity is None:
            raise RuntimeError("Cross-block similarity constant is missing for the selected score objective")
        total_score = float(total_reduced_score - (args.cross_block_lambda * total_cross_block_similarity))

    if args.cost_curve_output is not None:
        if cost_curve_block_counts is None or cost_curve_total_costs is None:
            cost_curve_block_counts, cost_curve_total_costs = compute_total_cost_curve(
                num_timesteps=num_timesteps,
                interval_costs=resolved_interval_costs,
                max_blocks=max_feasible_blocks,
                min_block_size=args.min_block_size,
            )
        curve_title = "Total DP Cost vs. Number of Blocks"
        curve_ylabel = "Total DP Cost"
        curve_values = cost_curve_total_costs
        if score_table is not None:
            if total_cross_block_similarity is None:
                raise RuntimeError("Cross-block similarity constant is missing for the selected score objective")
            if cost_curve_total_scores is None:
                cost_curve_total_scores = cross_penalty_total_scores(
                    cost_curve_total_costs,
                    cross_block_lambda=args.cross_block_lambda,
                    total_cross_block_similarity=total_cross_block_similarity,
                )
            curve_title = "Total DP Score vs. Number of Blocks"
            curve_ylabel = "Total DP Score"
            curve_values = cost_curve_total_scores
        plot_total_cost_curve(
            block_counts=cost_curve_block_counts,
            total_costs=curve_values,
            out_path=args.cost_curve_output,
            title=curve_title,
            ylabel=curve_ylabel,
            selected_num_blocks=result.num_blocks,
        )

    if args.plot_output is not None:
        plot_matrix = None
        plot_source = None
        if args.plot_matrix_source == "file":
            if args.plot_matrix is not None:
                plot_matrix = load_numeric_array(args.plot_matrix, array_key=args.plot_matrix_key)
                plot_source = args.plot_matrix
            else:
                plot_matrix = default_plot_matrix
                plot_source = default_plot_source
        elif args.plot_matrix_source in {"recomputed_similarity", "recomputed_distance"}:
            if matrix is None:
                raise ValueError(f"--plot_matrix_source {args.plot_matrix_source!r} requires --matrix")
            plot_matrix = recompute_builtin_plot_matrix(
                matrix=matrix,
                builtin_cost=args.builtin_cost,
                axis=args.matrix_axis,
                plot_matrix_source=args.plot_matrix_source,
                feature_weights=matrix_feature_weights,
            )
            plot_source = args.matrix
        elif args.plot_matrix_source == "auto":
            if args.plot_matrix is not None:
                plot_matrix = load_numeric_array(args.plot_matrix, array_key=args.plot_matrix_key)
                plot_source = args.plot_matrix
            elif default_plot_matrix is not None:
                plot_matrix = default_plot_matrix
                plot_source = default_plot_source
            elif matrix is not None and infer_pairwise_metric_from_builtin_cost(args.builtin_cost) is not None:
                plot_matrix = recompute_builtin_plot_matrix(
                    matrix=matrix,
                    builtin_cost=args.builtin_cost,
                    axis=args.matrix_axis,
                    plot_matrix_source="recomputed_similarity",
                    feature_weights=matrix_feature_weights,
                )
                plot_source = args.matrix
            else:
                raise ValueError(
                    "No plot matrix is available. Pass --plot_matrix for a file-backed matrix or use a pairwise built-in cost with --plot_matrix_source recomputed_similarity."
                )
        else:
            raise ValueError(f"Unsupported --plot_matrix_source: {args.plot_matrix_source}")

        if plot_matrix is None:
            raise ValueError("Failed to resolve a plot matrix")
        plot_labels = None
        if args.plot_labels_key is not None:
            if plot_source is None:
                raise ValueError("--plot_labels_key requires a file-backed plot matrix source")
            plot_labels = load_plot_labels(plot_source, key=args.plot_labels_key)
        plot_partitioned_matrix(
            matrix=plot_matrix,
            boundaries=result.boundaries,
            out_path=args.plot_output,
            title=args.plot_title,
            labels=plot_labels,
        )

    if args.plot_series_output is not None:
        plot_series = None
        plot_series_source = None
        if args.plot_series is not None:
            plot_series = load_numeric_array(args.plot_series, array_key=args.plot_series_key)
            plot_series_source = args.plot_series
        elif args.values is not None:
            plot_series = load_numeric_array(args.values, array_key=args.array_key)
            plot_series_source = args.values
        else:
            raise ValueError(
                "No plot series is available. Pass --plot_series for a file-backed metric curve or use --values."
            )

        if plot_series.ndim != 1:
            raise ValueError(f"--plot_series must resolve to a 1D array, got shape {plot_series.shape}")
        if plot_series.shape[0] != num_timesteps:
            raise ValueError(
                f"plot series length ({plot_series.shape[0]}) does not match num_timesteps ({num_timesteps})"
            )

        plot_labels = None
        if args.plot_labels_key is not None:
            if plot_series_source is None:
                raise ValueError("--plot_labels_key requires a file-backed plot series source")
            plot_labels = load_plot_labels(plot_series_source, key=args.plot_labels_key)

        plot_partitioned_series(
            values=plot_series,
            boundaries=result.boundaries,
            out_path=args.plot_series_output,
            title=args.plot_series_title,
            ylabel=args.plot_series_ylabel,
            labels=plot_labels,
        )

    profile_provenance = provenance_from_results(None, None)
    provenance_source = array_source or args.matrix
    if provenance_source is not None and is_json_path(provenance_source):
        raw_profile_source = load_value(provenance_source, key=None)
        if isinstance(raw_profile_source, Mapping):
            profile_provenance = provenance_from_results_path(provenance_source)
        # Arbitrary JSON arrays remain supported by this general-purpose
        # optimizer; only object-shaped evaluator results carry a profile.

    payload = build_result_payload(
        result,
        array_source=array_source,
        array_key=array_key,
        matrix_shape=None if matrix is None else matrix.shape,
        plot_output=args.plot_output,
        plot_matrix_source=args.plot_matrix_source if args.plot_output is not None else None,
        builtin_cost=args.builtin_cost,
        pairwise_metric=infer_pairwise_metric_from_builtin_cost(args.builtin_cost),
        pairwise_normalization=args.pairwise_normalization if args.builtin_cost in {"matrix_cosine", "matrix_correlation", "matrix_spearman"} else None,
        cross_block_lambda=args.cross_block_lambda if args.builtin_cost in {"matrix_cosine_cross_penalty", "matrix_correlation_cross_penalty", "matrix_spearman_cross_penalty"} else None,
        cross_block_reward_normalization=args.cross_block_reward_normalization if args.builtin_cost in {"matrix_cosine_cross_penalty", "matrix_correlation_cross_penalty", "matrix_spearman_cross_penalty"} else None,
        total_reduced_score=total_reduced_score,
        total_score=total_score,
        block_reduced_scores=block_reduced_scores,
        cost_curve_output=args.cost_curve_output,
        cost_curve_block_counts=cost_curve_block_counts,
        cost_curve_total_costs=cost_curve_total_costs,
        cost_curve_total_scores=cost_curve_total_scores,
        plot_series_output=args.plot_series_output,
        plot_series_key=args.plot_series_key if args.plot_series_output is not None else None,
        num_blocks_selection=num_blocks_selection,
        profile_provenance=profile_provenance,
        matrix_row_weighting=matrix_row_weighting,
    )
    print(json.dumps(payload, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
