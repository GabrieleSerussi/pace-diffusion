"""Capacity-score loading and fixed-budget allocation helpers.

The main experiment uses the original model budget as the fixed stored-budget
reference.  The ``global`` variant uses one student with that full budget.
Blockwise variants split the same total budget across one student per timestep
block, so the sum of stored student budgets matches the original model budget.
"""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence

import numpy as np

from pace.filter_sampling import (
    aligned_group_expansion_weights,
    group_expansion_weight_map,
    normalize_filter_sampling,
)
from pace.jsonio import find_json_file, is_json_path, load_json

try:
    import torch
except ImportError:  # pragma: no cover - exercised only in minimal environments.
    torch = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)
DEFAULT_SHUFFLE_SEED = 3

StudentVariant = Literal[
    "global",
    "uniform_blockwise",
    "shuffled_capacity",
    "blockwise_capacity",
    "layerwise_capacity",
    "reversed_layerwise_capacity",
    "combined_blockwise",
    "combined_layerwise",
]

AllocationGroupScope = Literal["all", "allocatable"]

SUPPORTED_STUDENT_VARIANTS: tuple[str, ...] = (
    "global",
    "uniform_blockwise",
    "shuffled_capacity",
    "blockwise_capacity",
    "layerwise_capacity",
    "reversed_layerwise_capacity",
    "combined_blockwise",
    "combined_layerwise",
)

# The four allocation variants compared in the paper (Section 3.4, Table 2):
# Global, Uniform blockwise, Phase-aware blockwise and Phase-aware layerwise.
PAPER_STUDENT_VARIANTS: tuple[str, ...] = (
    "global",
    "uniform_blockwise",
    "combined_blockwise",
    "combined_layerwise",
)

# The older capacity-control variants, kept for comparison runs.
LEGACY_STUDENT_VARIANTS: tuple[str, ...] = (
    "global",
    "uniform_blockwise",
    "shuffled_capacity",
    "blockwise_capacity",
    "layerwise_capacity",
    "reversed_layerwise_capacity",
)

DEFAULT_STUDENT_VARIANTS: tuple[str, ...] = PAPER_STUDENT_VARIANTS

# Paper values of the allocation rule (Section 3.4, Eq. 1): q_t is the
# geometric mean of the sensitivity-weighted parameter cost, summed per phase.
PAPER_ALLOCATION_METRIC = "delta_p_eff_geomean"
PAPER_SCORE_REDUCTION = "sum"
PAPER_LAYER_SCORE_SOURCE = "delta_p_eff_geomean"

SUPPORTED_SCORE_REDUCTIONS: tuple[str, ...] = ("mean", "sum", "max", "q90")

SUPPORTED_ALLOCATION_GROUP_SCOPES: tuple[str, ...] = ("all", "allocatable")

GEOMETRIC_SCORE_REDUCTION_PROTOCOLS: Mapping[str, str] = {
    "mean": "duration_neutral_mean",
    "sum": "edm_exact_sum",
}

SCORE_KEYS: tuple[str, ...] = (
    "capacity_score",
    "capacity_scores",
    "block_capacity_score",
    "block_capacity_scores",
    "layer_capacity_score",
    "layer_capacity_scores",
    "blockwise_capacity",
    "layerwise_capacity",
    "scores",
    "values",
    # Backward-compatible keys for analysis outputs in this repository.
    "n_eff",
    "p_eff",
)

PARAMETER_BUDGET_KEYS: tuple[str, ...] = (
    "parameters",
    "num_parameters",
    "parameter_count",
    "param_count",
    "n_params",
    "params",
)

ALLOCATION_PARAMETER_BUDGET_KEYS: tuple[str, ...] = (
    "allocation_parameter_count",
    "allocatable_parameter_count",
    "allocation_analyzed_parameter_count",
)

FLOP_BUDGET_KEYS: tuple[str, ...] = (
    "flops",
    "num_flops",
    "flop_count",
    "total_flops",
)


@dataclass(frozen=True)
class TimestepBlock:
    """Half-open interval of timestep or noise-level bins."""

    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class RoundingRules:
    """Architecture-specific divisibility constraints used by builders."""

    channel_divisibility: int = 1
    hidden_size_divisibility: int = 1
    attention_head_divisibility: int = 1
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CapacityAllocationConfig:
    """Configuration surface for adaptive-capacity student allocation."""

    student_variant: StudentVariant
    timestep_blocks: list[TimestepBlock] = field(default_factory=list)
    eval_output_dir: Optional[str] = None
    allocation_results_dir: Optional[str] = None
    timestep_grouping_path: Optional[str] = None
    block_capacity_scores_path: Optional[str] = None
    layer_capacity_scores_path: Optional[str] = None
    match_original_total_budget: bool = True
    original_model_config_path: Optional[str] = None
    original_model_checkpoint_path: Optional[str] = None
    allocation_metric: str = "auto"
    allocation_group_scope: AllocationGroupScope = "all"
    score_reduction: str = "mean"
    layer_score_source: str = "relative_delta_stack"
    allocation_alpha: float = 1.0
    min_width_multiplier: Optional[float] = None
    max_width_multiplier: Optional[float] = None
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED
    budget_tolerance: float = 0.05
    rounding_rules: RoundingRules = field(default_factory=RoundingRules)
    allow_uniform_if_all_zero: bool = False

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "CapacityAllocationConfig":
        variant = str(config.get("student_variant", "global"))
        if variant not in SUPPORTED_STUDENT_VARIANTS:
            raise ValueError(
                f"student_variant must be one of {SUPPORTED_STUDENT_VARIANTS}, got {variant!r}"
            )

        eval_output_dir = _optional_str(
            config.get("eval_output_dir", config.get("evaluation_output_dir"))
        )
        raw_blocks = config.get("timestep_blocks")
        if raw_blocks is None and eval_output_dir is None and config.get("timestep_grouping_path") is None:
            raise ValueError(
                "timestep_blocks is required unless eval_output_dir or timestep_grouping_path is provided"
            )
        score_reduction = str(config.get("score_reduction", "mean"))
        if score_reduction not in SUPPORTED_SCORE_REDUCTIONS:
            raise ValueError(
                f"score_reduction must be one of {SUPPORTED_SCORE_REDUCTIONS}, got {score_reduction!r}"
            )
        allocation_group_scope = str(config.get("allocation_group_scope", "all"))
        if allocation_group_scope not in SUPPORTED_ALLOCATION_GROUP_SCOPES:
            raise ValueError(
                "allocation_group_scope must be one of "
                f"{SUPPORTED_ALLOCATION_GROUP_SCOPES}, got {allocation_group_scope!r}"
            )
        allocation_metric = str(config.get("allocation_metric", "auto"))
        if allocation_metric == "delta_p_eff_geomean":
            geometric_score_reduction_protocol(score_reduction)
        if (
            variant in {"combined_blockwise", "combined_layerwise"}
            and allocation_metric != "delta_p_eff_geomean"
        ):
            raise ValueError(
                f"{variant} requires allocation_metric='delta_p_eff_geomean'; "
                "the score reduction must use a supported geometric protocol"
            )

        rounding_payload = config.get("rounding_rules", {}) or {}
        if isinstance(rounding_payload, RoundingRules):
            rounding_rules = rounding_payload
        elif isinstance(rounding_payload, Mapping):
            known = {
                key: rounding_payload[key]
                for key in (
                    "channel_divisibility",
                    "hidden_size_divisibility",
                    "attention_head_divisibility",
                )
                if key in rounding_payload
            }
            extra = {
                key: value
                for key, value in rounding_payload.items()
                if key not in known
            }
            rounding_rules = RoundingRules(**known, extra=extra)
        else:
            raise TypeError("rounding_rules must be a mapping")

        if "match_original_total_budget" not in config:
            raise ValueError("match_original_total_budget is required and must be true")
        match_original_total_budget = bool(config["match_original_total_budget"])
        if not match_original_total_budget:
            raise ValueError("match_original_total_budget must be true for the main fixed-budget experiment")

        shuffle_seed = config.get("shuffle_seed", DEFAULT_SHUFFLE_SEED)
        if isinstance(shuffle_seed, bool) or not isinstance(shuffle_seed, int):
            raise ValueError(f"shuffle_seed must be an integer, got {shuffle_seed!r}")
        if shuffle_seed < 0:
            raise ValueError(f"shuffle_seed must be non-negative, got {shuffle_seed}")

        return cls(
            student_variant=variant,  # type: ignore[arg-type]
            timestep_blocks=[] if raw_blocks is None else normalize_timestep_blocks(raw_blocks),
            eval_output_dir=eval_output_dir,
            allocation_results_dir=_optional_str(config.get("allocation_results_dir")),
            timestep_grouping_path=_optional_str(config.get("timestep_grouping_path")),
            block_capacity_scores_path=_optional_str(config.get("block_capacity_scores_path")),
            layer_capacity_scores_path=_optional_str(config.get("layer_capacity_scores_path")),
            match_original_total_budget=match_original_total_budget,
            original_model_config_path=_optional_str(config.get("original_model_config_path")),
            original_model_checkpoint_path=_optional_str(config.get("original_model_checkpoint_path")),
            allocation_metric=str(config.get("allocation_metric", "auto")),
            allocation_group_scope=allocation_group_scope,  # type: ignore[arg-type]
            score_reduction=score_reduction,
            layer_score_source=str(config.get("layer_score_source", "relative_delta_stack")),
            allocation_alpha=float(config.get("allocation_alpha", 1.0)),
            min_width_multiplier=_optional_float(config.get("min_width_multiplier")),
            max_width_multiplier=_optional_float(config.get("max_width_multiplier")),
            shuffle_seed=shuffle_seed,
            budget_tolerance=float(config.get("budget_tolerance", 0.05)),
            rounding_rules=rounding_rules,
            allow_uniform_if_all_zero=bool(config.get("allow_uniform_if_all_zero", False)),
        )


@dataclass(frozen=True)
class ModelBudget:
    """Original-model reference budget."""

    parameters: int
    flops: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BudgetValidationReport:
    """Target-vs-realized budget validation summary."""

    target_budgets: list[float]
    realized_budgets: Optional[list[float]]
    absolute_mismatch: Optional[list[float]]
    relative_mismatch: Optional[list[float]]
    within_tolerance: bool
    budget_tolerance: float
    budget_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CapacityBudgetPlan:
    """Target budgets for one student-variant configuration."""

    student_variant: StudentVariant
    total_student_system_budget: float
    block_budgets: list[float]
    layer_budgets: Optional[list[list[float]]] = None

    @property
    def total_budget(self) -> float:
        return self.total_student_system_budget

    @property
    def num_students(self) -> int:
        return len(self.block_budgets)

    @property
    def total_stored_budget(self) -> float:
        return float(sum(self.block_budgets))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "num_students": self.num_students,
            "total_stored_budget": self.total_stored_budget,
        }


@dataclass(frozen=True)
class EvaluationOutputLayout:
    """Files discovered inside an evaluate_parameters* output directory."""

    output_dir: str
    results_json_path: Optional[str]
    timestep_grouping_path: Optional[str]
    checkpoint_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "results_json_path": self.results_json_path,
            "timestep_grouping_path": self.timestep_grouping_path,
            "checkpoint_paths": list(self.checkpoint_paths),
            "num_checkpoint_dumps": len(self.checkpoint_paths),
        }


def _optional_str(value: Any) -> Optional[str]:
    return None if value is None else str(value)


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def normalize_timestep_blocks(blocks: Sequence[Sequence[int] | TimestepBlock]) -> list[TimestepBlock]:
    normalized: list[TimestepBlock] = []
    for index, block in enumerate(blocks):
        if isinstance(block, TimestepBlock):
            item = block
        else:
            if len(block) != 2:
                raise ValueError(f"timestep block {index} must contain [start, end], got {block}")
            item = TimestepBlock(start=int(block[0]), end=int(block[1]))
        if item.start < 0 or item.end <= item.start:
            raise ValueError(f"invalid timestep block {index}: [{item.start}, {item.end})")
        if normalized and item.start != normalized[-1].end:
            raise ValueError(
                "timestep_blocks must be contiguous and sorted; "
                f"block {index} starts at {item.start}, previous ended at {normalized[-1].end}"
            )
        normalized.append(item)
    if not normalized:
        raise ValueError("timestep_blocks must contain at least one block")
    return normalized


def discover_evaluation_output(output_dir: str | Path) -> EvaluationOutputLayout:
    """Discover standard dump files under an evaluate_parameters* output directory."""

    root = Path(output_dir)
    if not root.is_dir():
        raise ValueError(f"eval_output_dir must be a directory, got {root}")

    # ``.json.gz`` variants are accepted for released artifacts; plain files win.
    results_path = find_json_file(root, "results") or root / "results.json"
    grouping_candidates = (
        find_json_file(root / "grouping", "timestep_grouping"),
        find_json_file(root, "timestep_grouping"),
    )
    grouping_path = next((path for path in grouping_candidates if path is not None), None)

    checkpoint_paths = sorted(
        {
            str(path)
            for pattern in ("checkpoint_rank*.pt", "checkpoint_*_rank*.pt")
            for path in root.glob(pattern)
            if path.is_file()
        }
    )
    return EvaluationOutputLayout(
        output_dir=str(root),
        results_json_path=str(results_path) if results_path.is_file() else None,
        timestep_grouping_path=str(grouping_path) if grouping_path is not None else None,
        checkpoint_paths=tuple(checkpoint_paths),
    )


def load_timestep_blocks_from_grouping(path: str | Path) -> list[TimestepBlock]:
    grouping = _load_structured_file(Path(path))
    if not isinstance(grouping, Mapping):
        raise ValueError(f"timestep grouping file must contain a mapping, got {type(grouping).__name__}")
    return timestep_blocks_from_grouping(grouping)


def timestep_blocks_from_grouping(grouping: Mapping[str, Any]) -> list[TimestepBlock]:
    """Extract contiguous timestep blocks from optimize_timestep_grouping output."""

    boundaries = grouping.get("boundaries")
    if boundaries is not None:
        boundary_values = [int(value) for value in boundaries]
        if len(boundary_values) < 2:
            raise ValueError(f"grouping boundaries must contain at least two entries, got {boundary_values}")
        return normalize_timestep_blocks(
            [[start, end] for start, end in zip(boundary_values[:-1], boundary_values[1:])]
        )

    raw_blocks = grouping.get("blocks", grouping.get("timestep_blocks"))
    if raw_blocks is None:
        raise ValueError("grouping file must contain boundaries, blocks, or timestep_blocks")
    blocks: list[list[int]] = []
    for index, block in enumerate(raw_blocks):
        if isinstance(block, Mapping):
            try:
                blocks.append([int(block["start"]), int(block["end"])])
            except KeyError as exc:
                raise ValueError(f"grouping block {index} must contain start and end") from exc
        else:
            if len(block) != 2:
                raise ValueError(f"grouping block {index} must contain [start, end], got {block}")
            blocks.append([int(block[0]), int(block[1])])
    return normalize_timestep_blocks(blocks)


def load_block_capacity_scores_from_results(
    results_path: str | Path,
    timestep_blocks: Sequence[TimestepBlock],
    *,
    metric: str = "auto",
    reduction: str = "mean",
    group_scope: AllocationGroupScope = "all",
) -> tuple[np.ndarray, str]:
    """Load timestep metrics from results.json and aggregate them to block scores."""

    results = _load_results_mapping(results_path)
    metric_name, values = resolve_results_metric(
        results,
        metric,
        group_scope=group_scope,
    )
    return aggregate_timestep_values(values, timestep_blocks, reduction=reduction), metric_name


def load_layer_capacity_scores_from_results(
    results_path: str | Path,
    timestep_blocks: Sequence[TimestepBlock],
    *,
    score_source: str = "relative_delta_stack",
    reduction: str = "mean",
) -> np.ndarray:
    """Aggregate a results matrix to ``[num_blocks, num_allocation_stages]``.

    Historical results do not carry structural allocation metadata and retain
    the old behavior: every analysis group is treated as one layer.  Newer
    fine-grained results can provide ``group_stage_keys`` (or an explicit
    ``group_allocation_structural_keys`` mapping) and ``group_allocatable``.
    In that case filter rows are summed into the declared allocation stages and
    non-allocatable groups are omitted.
    """

    scores, _ = load_layer_capacity_scores_with_metadata_from_results(
        results_path,
        timestep_blocks,
        score_source=score_source,
        reduction=reduction,
    )
    return scores


def load_layer_capacity_scores_with_metadata_from_results(
    results_path: str | Path,
    timestep_blocks: Sequence[TimestepBlock],
    *,
    score_source: str = "relative_delta_stack",
    reduction: str = "mean",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load layer scores and describe any explicit structural aggregation."""

    results = _load_results_mapping(results_path)
    if score_source == "delta_p_eff_geomean":
        matrix = compute_delta_param_geomean_layer_scores(results, apply_expansion=False)
    elif score_source not in results:
        raise KeyError(f"results.json does not contain layer score source {score_source!r}")
    else:
        matrix = np.asarray(results[score_source], dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{score_source} must be 2D [num_groups, num_timesteps], got shape {matrix.shape}")
    expansion = results_group_expansion_weights(
        results,
        expected_num_groups=matrix.shape[0],
    )
    matrix = matrix * expansion.reshape(-1, 1)
    matrix, aggregation = aggregate_group_scores_for_allocation(results, matrix)
    _validate_timestep_coverage(matrix.shape[1], timestep_blocks, name=score_source)

    scores = np.zeros((len(timestep_blocks), matrix.shape[0]), dtype=np.float64)
    for block_index, block in enumerate(timestep_blocks):
        block_values = matrix[:, block.start:block.end]
        scores[block_index] = reduce_capacity_values(block_values, reduction=reduction, axis=1)
    return scores, aggregation


def aggregate_group_scores_for_allocation(
    results: Mapping[str, Any],
    matrix: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sum fine-grained score rows into explicit allocation stages.

    The function is deliberately metadata-driven.  It never infers DiffWave
    stages from names, which keeps existing EDM and residual-block artifacts
    unchanged.  The supported metadata contract is:

    - ``group_names`` fixes matrix row order;
    - ``group_structural_keys`` identifies the analyzed module;
    - ``group_stage_keys`` identifies its architecture stage;
    - ``group_allocatable`` can exclude fixed groups; and
    - ``group_allocation_structural_keys`` (``group_allocation_keys`` is an
      accepted compatibility alias) can directly select and name allocatable
      rows.  Missing/null entries in that explicit mapping are excluded.

    Returns the possibly aggregated matrix and a JSON-serializable report.
    """

    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"group score matrix must be 2D, got shape {arr.shape}")

    raw_names = results.get("group_names")
    if raw_names is None:
        return arr, _unaggregated_group_report(arr.shape[0])
    if not isinstance(raw_names, Sequence) or isinstance(raw_names, (str, bytes)):
        raise ValueError("group_names must be a sequence")
    group_names = [str(name) for name in raw_names]
    if len(group_names) != arr.shape[0]:
        raise ValueError(
            "group_names length must match score matrix group dimension, "
            f"got {len(group_names)} and {arr.shape[0]}"
        )
    if len(set(group_names)) != len(group_names):
        raise ValueError("group_names must be unique for structural aggregation")

    structural_keys = _optional_group_mapping(results, "group_structural_keys", group_names)
    stage_keys = _optional_group_mapping(results, "group_stage_keys", group_names)
    allocation_key_field = next(
        (
            field
            for field in ("group_allocation_structural_keys", "group_allocation_keys")
            if results.get(field) is not None
        ),
        None,
    )
    allocation_keys = (
        _optional_group_mapping(
            results,
            allocation_key_field,
            group_names,
            allow_partial=True,
            allow_null=True,
        )
        if allocation_key_field is not None
        else None
    )
    allocatable = _optional_group_bool_mapping(results, "group_allocatable", group_names)

    if allocation_keys is not None:
        key_source = str(allocation_key_field)
    elif stage_keys is not None:
        key_source = "group_stage_keys"
    elif structural_keys is not None and allocatable is not None:
        # Structural keys alone are meaningful only when the producer also
        # explicitly opts groups in/out.  Otherwise old artifacts remain raw.
        key_source = "group_structural_keys"
    else:
        return arr, _unaggregated_group_report(len(group_names))

    param_counts = results.get("group_param_counts")
    if param_counts is not None and not isinstance(param_counts, Mapping):
        raise ValueError("group_param_counts must be a mapping")

    output_keys: list[str] = []
    output_indices: dict[str, int] = {}
    included_rows: list[tuple[int, str]] = []
    excluded_rows: list[tuple[int, str, str]] = []
    for index, name in enumerate(group_names):
        is_allocatable = True if allocatable is None else bool(allocatable[name])
        if allocation_keys is not None:
            raw_key = allocation_keys.get(name)
            if raw_key is None or not str(raw_key).strip():
                is_allocatable = False
                reason = "missing_allocation_structural_key"
            else:
                reason = "group_allocatable_false"
                allocation_key = str(raw_key)
        else:
            reason = "group_allocatable_false"
            source_mapping = stage_keys if stage_keys is not None else structural_keys
            assert source_mapping is not None
            allocation_key = str(source_mapping[name])

        if not is_allocatable:
            stage_key = "unassigned"
            if stage_keys is not None and stage_keys.get(name) is not None:
                stage_key = str(stage_keys[name])
            excluded_rows.append((index, stage_key, reason))
            continue
        if not allocation_key.strip():
            raise ValueError(f"{key_source}[{name!r}] must be a non-empty string")
        if allocation_key not in output_indices:
            output_indices[allocation_key] = len(output_keys)
            output_keys.append(allocation_key)
        included_rows.append((index, allocation_key))

    if not included_rows:
        raise ValueError("structural allocation metadata excludes every analysis group")

    aggregated = np.zeros((len(output_keys), arr.shape[1]), dtype=np.float64)
    stage_group_counts = {key: 0 for key in output_keys}
    stage_param_counts = {key: 0 for key in output_keys}
    for row_index, key in included_rows:
        aggregated[output_indices[key]] += arr[row_index]
        stage_group_counts[key] += 1
        if param_counts is not None:
            if group_names[row_index] not in param_counts:
                raise ValueError(f"group_param_counts is missing {group_names[row_index]!r}")
            stage_param_counts[key] += int(param_counts[group_names[row_index]])

    excluded_stage_counts: dict[str, int] = {}
    excluded_stage_params: dict[str, int] = {}
    excluded_reasons: dict[str, int] = {}
    excluded_names: list[str] = []
    for row_index, stage_key, reason in excluded_rows:
        name = group_names[row_index]
        excluded_names.append(name)
        excluded_stage_counts[stage_key] = excluded_stage_counts.get(stage_key, 0) + 1
        excluded_reasons[reason] = excluded_reasons.get(reason, 0) + 1
        if param_counts is not None:
            if name not in param_counts:
                raise ValueError(f"group_param_counts is missing {name!r}")
            excluded_stage_params[stage_key] = excluded_stage_params.get(stage_key, 0) + int(param_counts[name])

    allocatable_parameter_count = (
        int(sum(stage_param_counts.values())) if param_counts is not None else None
    )
    excluded_parameter_count = (
        int(sum(excluded_stage_params.values())) if param_counts is not None else None
    )
    unassigned_parameter_count = _find_allocation_unassigned_parameter_count(results)
    total_teacher_parameter_count = _find_total_teacher_parameter_count(results)
    accounted_parameter_count = (
        allocatable_parameter_count + excluded_parameter_count + unassigned_parameter_count
        if allocatable_parameter_count is not None
        and excluded_parameter_count is not None
        and unassigned_parameter_count is not None
        else None
    )
    report = {
        "applied": True,
        "key_source": key_source,
        "structural_key_source": "group_structural_keys" if structural_keys is not None else None,
        "stage_key_source": "group_stage_keys" if stage_keys is not None else None,
        "input_group_count": len(group_names),
        "allocatable_group_count": len(included_rows),
        "excluded_group_count": len(excluded_rows),
        "output_stage_count": len(output_keys),
        "output_stage_keys": output_keys,
        "stages": [
            {
                "stage_key": key,
                "group_count": stage_group_counts[key],
                "parameter_count": stage_param_counts[key] if param_counts is not None else None,
            }
            for key in output_keys
        ],
        "excluded_stages": [
            {
                "stage_key": key,
                "group_count": excluded_stage_counts[key],
                "parameter_count": excluded_stage_params.get(key) if param_counts is not None else None,
            }
            for key in excluded_stage_counts
        ],
        "excluded_reasons": excluded_reasons,
        "excluded_group_names": excluded_names,
        "allocatable_parameter_count": allocatable_parameter_count,
        "excluded_analyzed_parameter_count": excluded_parameter_count,
        "unassigned_parameter_count": unassigned_parameter_count,
        "accounted_parameter_count": accounted_parameter_count,
        "total_teacher_parameter_count": total_teacher_parameter_count,
        "parameter_accounting_matches_teacher": (
            accounted_parameter_count == total_teacher_parameter_count
            if accounted_parameter_count is not None and total_teacher_parameter_count is not None
            else None
        ),
        "aggregation": "sum_of_individual_group_scores",
    }
    return aggregated, report


def describe_group_allocation_scope(results: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return structural allocation metadata without loading a score matrix."""

    raw_names = results.get("group_names")
    if not isinstance(raw_names, Sequence) or isinstance(raw_names, (str, bytes)):
        return None
    dummy = np.zeros((len(raw_names), 1), dtype=np.float64)
    _, report = aggregate_group_scores_for_allocation(results, dummy)
    return report if report.get("applied") else None


def _unaggregated_group_report(group_count: int) -> dict[str, Any]:
    return {
        "applied": False,
        "key_source": None,
        "input_group_count": int(group_count),
        "allocatable_group_count": int(group_count),
        "excluded_group_count": 0,
        "output_stage_count": int(group_count),
        "output_stage_keys": None,
        "aggregation": "none",
    }


def _optional_group_mapping(
    results: Mapping[str, Any],
    field: str | None,
    group_names: Sequence[str],
    *,
    allow_partial: bool = False,
    allow_null: bool = False,
) -> dict[str, Any] | None:
    if field is None or results.get(field) is None:
        return None
    raw = results[field]
    if not isinstance(raw, Mapping):
        raise ValueError(f"{field} must be a mapping keyed by group name")
    expected_names = set(group_names)
    extra = sorted(str(key) for key in raw if str(key) not in expected_names)
    if extra:
        raise ValueError(f"{field} contains unknown group names: {extra[:5]}")
    if not allow_partial:
        missing = [name for name in group_names if name not in raw]
        if missing:
            raise ValueError(f"{field} is missing group names: {missing[:5]}")
    normalized: dict[str, Any] = {}
    for name in group_names:
        if name not in raw:
            continue
        value = raw[name]
        if value is None and allow_null:
            normalized[name] = None
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field}[{name!r}] must be a non-empty string")
        normalized[name] = value
    return normalized


def _optional_group_bool_mapping(
    results: Mapping[str, Any],
    field: str,
    group_names: Sequence[str],
) -> dict[str, bool] | None:
    raw = results.get(field)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"{field} must be a mapping keyed by group name")
    missing = [name for name in group_names if name not in raw]
    if missing:
        raise ValueError(f"{field} is missing group names: {missing[:5]}")
    expected_names = set(group_names)
    extra = sorted(str(key) for key in raw if str(key) not in expected_names)
    if extra:
        raise ValueError(f"{field} contains unknown group names: {extra[:5]}")
    normalized: dict[str, bool] = {}
    for name in group_names:
        value = raw[name]
        if not isinstance(value, bool):
            raise ValueError(f"{field}[{name!r}] must be boolean")
        normalized[name] = value
    return normalized


def _find_allocation_unassigned_parameter_count(results: Mapping[str, Any]) -> int | None:
    for field_name in (
        "allocation_unassigned_parameter_count",
        "shared_unassigned_parameter_count",
        "unassigned_parameter_count",
    ):
        value = results.get(field_name)
        if value is not None and not isinstance(value, Mapping):
            return int(value)
    model_info = results.get("model_info")
    if isinstance(model_info, Mapping):
        return _find_allocation_unassigned_parameter_count(model_info)
    return None


def _find_total_teacher_parameter_count(results: Mapping[str, Any]) -> int | None:
    for field_name in (
        "total_teacher_parameter_count",
        "total_parameter_count",
        "num_parameters",
        "parameter_count",
    ):
        value = results.get(field_name)
        if value is not None and not isinstance(value, Mapping):
            return int(value)
    model_info = results.get("model_info")
    if isinstance(model_info, Mapping):
        return _find_total_teacher_parameter_count(model_info)
    return None


def resolve_results_metric(
    results: Mapping[str, Any],
    requested_metric: str,
    *,
    group_scope: AllocationGroupScope = "all",
) -> tuple[str, np.ndarray]:
    if requested_metric == "delta_p_eff_geomean":
        return requested_metric, compute_delta_p_eff_geomean_metric(
            results,
            group_scope=group_scope,
        )
    if requested_metric == "auto":
        for metric_name in ("n_eff", "p_eff"):
            if metric_name in results:
                return metric_name, np.asarray(results[metric_name], dtype=np.float64)
        raise KeyError("results.json does not contain 'n_eff' or 'p_eff'")
    if requested_metric not in results:
        raise KeyError(f"results.json does not contain allocation metric {requested_metric!r}")
    return requested_metric, np.asarray(results[requested_metric], dtype=np.float64)


def compute_delta_p_eff_geomean_metric(
    results: Mapping[str, Any],
    *,
    group_scope: AllocationGroupScope = "all",
) -> np.ndarray:
    """Return ``sqrt(HT-total(delta) * p_eff)`` per timestep.

    ``group_scope='all'`` uses every sampled delta row and the stored
    ``p_eff``.  ``group_scope='allocatable'`` selects rows marked true in
    ``group_allocatable`` and recomputes ``p_eff`` from those same rows.  Both
    paths apply the profile's Horvitz--Thompson expansion weights when the
    source profile used stratified filter sampling.
    """

    if "delta_stack" not in results:
        raise KeyError("results.json does not contain 'delta_stack'")

    delta_stack = np.asarray(results["delta_stack"], dtype=np.float64)
    if delta_stack.ndim != 2:
        raise ValueError(f"delta_stack must be 2D [num_groups, num_timesteps], got shape {delta_stack.shape}")
    _validate_score_values(delta_stack, name="delta_stack")

    expansion = results_group_expansion_weights(
        results,
        expected_num_groups=delta_stack.shape[0],
    )
    normalized_scope = _normalize_allocation_group_scope(group_scope)
    if normalized_scope == "allocatable":
        delta_stack, p_eff, expansion = _allocatable_delta_and_p_eff(
            results,
            delta_stack,
            expansion,
        )
    else:
        if "p_eff" not in results:
            raise KeyError("results.json does not contain 'p_eff'")
        p_eff = np.asarray(results["p_eff"], dtype=np.float64)

    if p_eff.ndim != 1:
        raise ValueError(f"p_eff must be 1D [num_timesteps], got shape {p_eff.shape}")
    if delta_stack.shape[1] != p_eff.shape[0]:
        raise ValueError(
            "delta_stack timestep dimension must match p_eff length, "
            f"got {delta_stack.shape[1]} and {p_eff.shape[0]}"
        )
    _validate_score_values(p_eff, name="p_eff")
    delta_total = (delta_stack * expansion.reshape(-1, 1)).sum(axis=0)
    return np.sqrt(delta_total * p_eff)


def _allocatable_delta_and_p_eff(
    results: Mapping[str, Any],
    delta_stack: np.ndarray,
    expansion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_group_names = results.get("group_names")
    if raw_group_names is None:
        raise KeyError("results.json does not contain 'group_names'")
    if not isinstance(raw_group_names, Sequence) or isinstance(raw_group_names, (str, bytes)):
        raise ValueError("group_names must be a sequence")
    group_names = [str(name) for name in raw_group_names]
    if len(group_names) != delta_stack.shape[0]:
        raise ValueError(
            "group_names length must match delta_stack group dimension, "
            f"got {len(group_names)} and {delta_stack.shape[0]}"
        )

    allocatable = _optional_group_bool_mapping(results, "group_allocatable", group_names)
    if allocatable is None:
        raise KeyError(
            "group_scope='allocatable' requires results.json field 'group_allocatable'"
        )
    param_counts = results.get("group_param_counts")
    if not isinstance(param_counts, Mapping):
        raise KeyError(
            "group_scope='allocatable' requires results.json field 'group_param_counts'"
        )

    selected_indices = [
        index
        for index, name in enumerate(group_names)
        if allocatable[name]
    ]
    if not selected_indices:
        raise ValueError("group_allocatable excludes every analysis group")
    missing_counts = [name for name in group_names if allocatable[name] and name not in param_counts]
    if missing_counts:
        raise KeyError(f"group_param_counts is missing allocatable groups: {missing_counts[:5]}")

    scoped_delta = delta_stack[selected_indices]
    scoped_expansion = expansion[selected_indices]
    scoped_parameters = np.asarray(
        [param_counts[group_names[index]] for index in selected_indices],
        dtype=np.float64,
    )
    _validate_score_values(scoped_parameters, name="allocatable_group_param_counts")
    expanded_delta = scoped_delta * scoped_expansion.reshape(-1, 1)
    positive_delta_mass = expanded_delta.sum(axis=0)
    weighted_parameter_mass = (
        expanded_delta * scoped_parameters.reshape(-1, 1)
    ).sum(axis=0)
    scoped_p_eff = np.divide(
        weighted_parameter_mass,
        positive_delta_mass,
        out=np.zeros_like(weighted_parameter_mass),
        where=positive_delta_mass > 0,
    )
    return scoped_delta, scoped_p_eff, scoped_expansion


def geometric_score_reduction_protocol(reduction: str) -> str:
    """Name the supported phase-reduction protocol for geometric scores."""

    try:
        return GEOMETRIC_SCORE_REDUCTION_PROTOCOLS[reduction]
    except KeyError as exc:
        supported = tuple(GEOMETRIC_SCORE_REDUCTION_PROTOCOLS)
        raise ValueError(
            "delta_p_eff_geomean score reduction must be one of "
            f"{supported}, got {reduction!r}"
        ) from exc


def _normalize_allocation_group_scope(group_scope: str) -> AllocationGroupScope:
    normalized_scope = str(group_scope)
    if normalized_scope not in SUPPORTED_ALLOCATION_GROUP_SCOPES:
        raise ValueError(
            "allocation group scope must be one of "
            f"{SUPPORTED_ALLOCATION_GROUP_SCOPES}, got {normalized_scope!r}"
        )
    return normalized_scope  # type: ignore[return-value]


def compute_delta_param_geomean_layer_scores(
    results: Mapping[str, Any],
    *,
    apply_expansion: bool = True,
) -> np.ndarray:
    """Return per-group ``sqrt(delta * params)`` contribution scores.

    A sampled group represents ``1 / pi`` population groups.  Expansion is
    therefore applied outside the square root: the additive population score
    is ``sum_i e_i * sqrt(delta_i * p_i)``, not
    ``sum_i sqrt(e_i * delta_i * p_i)``.
    """

    if "delta_stack" not in results:
        raise KeyError("results.json does not contain 'delta_stack'")
    if "group_names" not in results:
        raise KeyError("results.json does not contain 'group_names'")
    if "group_param_counts" not in results:
        raise KeyError("results.json does not contain 'group_param_counts'")

    delta_stack = np.asarray(results["delta_stack"], dtype=np.float64)
    if delta_stack.ndim != 2:
        raise ValueError(f"delta_stack must be 2D [num_groups, num_timesteps], got shape {delta_stack.shape}")
    group_names = list(results["group_names"])
    if len(group_names) != delta_stack.shape[0]:
        raise ValueError(
            "group_names length must match delta_stack group dimension, "
            f"got {len(group_names)} and {delta_stack.shape[0]}"
        )
    param_counts = results["group_param_counts"]
    if not isinstance(param_counts, Mapping):
        raise ValueError("group_param_counts must be a mapping")
    p = np.asarray([param_counts[name] for name in group_names], dtype=np.float64)
    _validate_score_values(delta_stack, name="delta_stack")
    _validate_score_values(p, name="group_param_counts")
    scores = np.sqrt(delta_stack * p.reshape(-1, 1))
    if apply_expansion:
        expansion = results_group_expansion_weights(results, expected_num_groups=delta_stack.shape[0])
        scores *= expansion.reshape(-1, 1)
    return scores


def results_group_expansion_weights(
    results: Mapping[str, Any],
    *,
    expected_num_groups: int | None = None,
) -> np.ndarray:
    """Load selected-group expansion weights while preserving legacy units."""

    if normalize_filter_sampling(results) is None and group_expansion_weight_map(results) is None:
        count = expected_num_groups
        if count is None:
            raw_names = results.get("group_names")
            if isinstance(raw_names, Sequence) and not isinstance(raw_names, (str, bytes)):
                count = len(raw_names)
            else:
                counts = results.get("group_param_counts")
                count = len(counts) if isinstance(counts, Mapping) else 0
        return np.ones(int(count), dtype=np.float64)

    raw_names = results.get("group_names")
    if isinstance(raw_names, Sequence) and not isinstance(raw_names, (str, bytes)):
        names = [str(name) for name in raw_names]
    else:
        counts = results.get("group_param_counts")
        if isinstance(counts, Mapping):
            names = [str(name) for name in counts]
        elif group_expansion_weight_map(results) is not None:
            raise ValueError("sampled results require group_names to align expansion weights")
        else:
            count = 0 if expected_num_groups is None else int(expected_num_groups)
            return np.ones(count, dtype=np.float64)

    if expected_num_groups is not None and len(names) != expected_num_groups:
        raise ValueError(
            "group_names length must match the group dimension, "
            f"got {len(names)} and {expected_num_groups}"
        )
    weights = np.asarray(aligned_group_expansion_weights(results, names), dtype=np.float64)
    _validate_score_values(weights, name="group_sampling_weights")
    return weights


def expansion_weighted_parameter_total(results: Mapping[str, Any]) -> float:
    """Estimate the population parameter total represented by selected groups."""

    counts_obj = results.get("group_param_counts")
    if not isinstance(counts_obj, Mapping):
        raise ValueError("group_param_counts must be a mapping")
    raw_names = results.get("group_names")
    names = (
        [str(name) for name in raw_names]
        if isinstance(raw_names, Sequence) and not isinstance(raw_names, (str, bytes))
        else [str(name) for name in counts_obj]
    )
    try:
        counts = np.asarray([counts_obj[name] for name in names], dtype=np.float64)
    except KeyError as exc:
        raise ValueError(f"group_param_counts is missing selected group {exc.args[0]!r}") from exc
    _validate_score_values(counts, name="group_param_counts")
    expansion = np.asarray(aligned_group_expansion_weights(results, names), dtype=np.float64)
    return float(np.dot(expansion, counts))


def aggregate_timestep_values(
    values: Any,
    timestep_blocks: Sequence[TimestepBlock],
    *,
    reduction: str = "mean",
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"timestep metric values must be 1D, got shape {arr.shape}")
    _validate_score_values(arr, name="timestep_metric")
    _validate_timestep_coverage(arr.shape[0], timestep_blocks, name="timestep_metric")

    scores = np.zeros(len(timestep_blocks), dtype=np.float64)
    for index, block in enumerate(timestep_blocks):
        block_values = arr[block.start:block.end]
        scores[index] = float(reduce_capacity_values(block_values, reduction=reduction))
    return scores


def reduce_capacity_values(
    values: np.ndarray,
    *,
    reduction: str,
    axis: Optional[int] = None,
) -> np.ndarray | np.float64:
    if reduction == "mean":
        return np.mean(values, axis=axis)
    if reduction == "sum":
        return np.sum(values, axis=axis)
    if reduction == "max":
        return np.max(values, axis=axis)
    if reduction == "q90":
        return np.quantile(values, 0.9, axis=axis)
    raise ValueError(f"score_reduction must be one of {SUPPORTED_SCORE_REDUCTIONS}, got {reduction!r}")


def _load_results_mapping(path: str | Path) -> Mapping[str, Any]:
    payload = _load_structured_file(Path(path))
    if not isinstance(payload, Mapping):
        raise ValueError(f"results file must contain a mapping, got {type(payload).__name__}")
    return payload


def _validate_timestep_coverage(
    num_timesteps: int,
    timestep_blocks: Sequence[TimestepBlock],
    *,
    name: str,
) -> None:
    if not timestep_blocks:
        raise ValueError("timestep_blocks must contain at least one block")
    last_end = timestep_blocks[-1].end
    if last_end > num_timesteps:
        raise ValueError(
            f"{name} has {num_timesteps} timesteps, but timestep blocks extend to {last_end}"
        )


def compute_original_model_budget(model_or_config: Any) -> ModelBudget:
    """Count the original model's parameter budget.

    Supports a live ``torch.nn.Module``, a mapping with budget metadata, a JSON/YAML
    config path containing budget metadata, or a PyTorch checkpoint/state dict.
    FLOPs are reported only when they are already present as metadata.
    """

    if isinstance(model_or_config, ModelBudget):
        budget = model_or_config
    elif _is_torch_module(model_or_config):
        budget = ModelBudget(
            parameters=_count_parameters_iterable(model_or_config.parameters()),
            flops=_extract_flops_from_object(model_or_config),
        )
    elif isinstance(model_or_config, Mapping):
        budget = _budget_from_mapping(model_or_config)
    elif isinstance(model_or_config, (str, Path)):
        budget = _budget_from_path(Path(model_or_config))
    elif hasattr(model_or_config, "parameters"):
        budget = ModelBudget(
            parameters=_count_parameters_iterable(model_or_config.parameters()),
            flops=_extract_flops_from_object(model_or_config),
        )
    else:
        raise TypeError(
            "model_or_config must be a torch.nn.Module, mapping, config path, checkpoint path, "
            f"or object with parameters(); got {type(model_or_config).__name__}"
        )

    if budget.parameters <= 0:
        raise ValueError(f"original model parameter budget must be positive, got {budget.parameters}")

    LOGGER.info(
        "original_model_budget parameters=%s flops=%s",
        budget.parameters,
        "unavailable" if budget.flops is None else budget.flops,
    )
    return budget


def _budget_from_path(path: Path) -> ModelBudget:
    suffix = path.suffix.lower()
    if is_json_path(path) or suffix in {".json", ".yaml", ".yml"}:
        return _budget_from_mapping(_load_structured_file(path))
    if suffix in {".pt", ".pth"}:
        payload = _torch_load(path)
        if _is_torch_module(payload):
            return compute_original_model_budget(payload)
        if isinstance(payload, Mapping):
            try:
                return _budget_from_mapping(payload)
            except ValueError:
                return ModelBudget(parameters=_count_tensors_in_mapping(payload))
        if _is_torch_tensor(payload):
            return ModelBudget(parameters=int(payload.numel()))
        raise TypeError(f"Unsupported checkpoint payload type in {path}: {type(payload).__name__}")
    raise ValueError(f"Unsupported original model budget file format: {path}")


def _budget_from_mapping(payload: Mapping[str, Any]) -> ModelBudget:
    allocation_parameters = _find_numeric_key(
        payload,
        ALLOCATION_PARAMETER_BUDGET_KEYS,
        required=False,
    )
    if allocation_parameters is None:
        allocation_parameters = _allocation_parameter_count_from_group_metadata(payload)
    if allocation_parameters is not None:
        flops = _find_numeric_key(payload, FLOP_BUDGET_KEYS, required=False)
        return ModelBudget(
            parameters=int(allocation_parameters),
            flops=None if flops is None else float(flops),
        )

    parameters = _find_numeric_key(payload, PARAMETER_BUDGET_KEYS)
    flops = _find_numeric_key(payload, FLOP_BUDGET_KEYS, required=False)
    if parameters is not None:
        return ModelBudget(parameters=int(parameters), flops=None if flops is None else float(flops))

    group_param_counts = payload.get("group_param_counts")
    if isinstance(group_param_counts, Mapping):
        total_parameters = int(round(expansion_weighted_parameter_total(payload)))
        if total_parameters > 0:
            return ModelBudget(
                parameters=total_parameters,
                flops=None if flops is None else float(flops),
            )
    elif isinstance(group_param_counts, Sequence) and not isinstance(group_param_counts, (str, bytes)):
        total_parameters = sum(int(value) for value in group_param_counts)
        if total_parameters > 0:
            return ModelBudget(
                parameters=total_parameters,
                flops=None if flops is None else float(flops),
            )

    for nested_key in ("original_model_budget", "model_budget", "budget", "model_info", "config"):
        nested = payload.get(nested_key)
        if isinstance(nested, Mapping):
            try:
                return _budget_from_mapping(nested)
            except ValueError:
                continue

    for state_key in ("state_dict", "model_state_dict", "model", "ema"):
        state = payload.get(state_key)
        if _is_torch_module(state):
            return compute_original_model_budget(state)
        if isinstance(state, Mapping):
            return ModelBudget(parameters=_count_tensors_in_mapping(state), flops=None if flops is None else float(flops))

    raise ValueError(
        "Could not infer original model parameter budget from mapping. "
        f"Expected one of {PARAMETER_BUDGET_KEYS}, group_param_counts, or a checkpoint state dict."
    )


def _allocation_parameter_count_from_group_metadata(payload: Mapping[str, Any]) -> int | None:
    group_names = payload.get("group_names")
    param_counts = payload.get("group_param_counts")
    if not isinstance(group_names, Sequence) or isinstance(group_names, (str, bytes)):
        return None
    if not isinstance(param_counts, Mapping):
        return None
    if not any(
        payload.get(field) is not None
        for field in (
            "group_allocatable",
            "group_allocation_structural_keys",
            "group_allocation_keys",
        )
    ):
        return None
    dummy = np.zeros((len(group_names), 1), dtype=np.float64)
    _, report = aggregate_group_scores_for_allocation(payload, dummy)
    count = report.get("allocatable_parameter_count")
    return None if count is None else int(count)


def _find_numeric_key(
    payload: Mapping[str, Any],
    keys: Sequence[str],
    required: bool = True,
) -> Optional[float]:
    for key in keys:
        if key in payload and payload[key] is not None:
            value = payload[key]
            if isinstance(value, Mapping):
                continue
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Budget field {key!r} must be numeric, got {value!r}") from exc
    if required:
        return None
    return None


def _extract_flops_from_object(model: Any) -> Optional[float]:
    for attr in FLOP_BUDGET_KEYS:
        if hasattr(model, attr):
            value = getattr(model, attr)
            if callable(value):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _is_torch_module(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.nn.Module)


def _is_torch_tensor(value: Any) -> bool:
    return torch is not None and torch.is_tensor(value)


def _count_parameters_iterable(parameters: Any) -> int:
    return sum(_parameter_numel(parameter) for parameter in parameters)


def _parameter_numel(parameter: Any) -> int:
    if hasattr(parameter, "numel"):
        return int(parameter.numel())
    return int(np.asarray(parameter).size)


def _count_tensors_in_mapping(payload: Mapping[str, Any]) -> int:
    total = 0
    for value in payload.values():
        if _is_torch_tensor(value):
            total += int(value.numel())
        elif isinstance(value, np.ndarray):
            total += int(value.size)
        elif isinstance(value, Mapping):
            total += _count_tensors_in_mapping(value)
    if total <= 0:
        raise ValueError("Checkpoint mapping did not contain any tensors to count")
    return total


def load_capacity_scores(path: str | Path) -> np.ndarray:
    """Load capacity scores from JSON, YAML, NPY/NPZ, or PyTorch files."""

    source_path = Path(path)
    suffix = source_path.suffix.lower()
    if is_json_path(source_path) or suffix in {".json", ".yaml", ".yml"}:
        payload = _load_structured_file(source_path)
    elif suffix == ".npy":
        payload = np.load(source_path, allow_pickle=False)
    elif suffix == ".npz":
        loaded = np.load(source_path, allow_pickle=False)
        chosen_key = _choose_mapping_key({key: loaded[key] for key in loaded.files}, SCORE_KEYS)
        payload = loaded[chosen_key]
    elif suffix in {".pt", ".pth"}:
        payload = _torch_load(source_path)
    else:
        raise ValueError(f"Unsupported capacity score file format: {source_path}")

    scores = np.asarray(_extract_numeric_payload(payload, source_path=source_path), dtype=np.float64)
    _validate_score_values(scores, name="capacity_scores")
    return scores


def _load_structured_file(path: Path) -> Any:
    if is_json_path(path):
        # Plain or gzip-compressed JSON (released artifacts use ``.json.gz``).
        return load_json(path)
    return _load_yaml_text(path.read_text())


def _load_yaml_text(text: str) -> Any:
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        return _parse_minimal_yaml(text)
    return yaml.safe_load(text)


def _parse_minimal_yaml(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    lines = [
        line.rstrip()
        for line in stripped.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if all(line.lstrip().startswith("- ") for line in lines):
        return [_parse_yaml_value(line.lstrip()[2:].strip()) for line in lines]

    mapping: dict[str, Any] = {}
    index = 0
    while index < len(lines):
        line = lines[index]
        if ":" not in line or line.startswith(" "):
            raise ValueError("Unsupported YAML structure; install PyYAML for full YAML support")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            mapping[key] = _parse_yaml_value(value)
            index += 1
            continue

        index += 1
        items: list[Any] = []
        while index < len(lines) and lines[index].startswith(" "):
            child = lines[index].strip()
            if not child.startswith("- "):
                raise ValueError("Unsupported YAML mapping structure; install PyYAML for full YAML support")
            items.append(_parse_yaml_value(child[2:].strip()))
            index += 1
        mapping[key] = items
    return mapping


def _parse_yaml_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _torch_load(path: Path) -> Any:
    if torch is None:
        raise ImportError("PyTorch is required to load .pt/.pth capacity score or checkpoint files")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_numeric_payload(payload: Any, source_path: Path) -> np.ndarray:
    if _is_torch_tensor(payload):
        return payload.detach().cpu().numpy()
    if isinstance(payload, np.ndarray):
        return payload
    if isinstance(payload, (list, tuple)):
        return np.asarray(payload, dtype=np.float64)
    if isinstance(payload, Mapping):
        key = _choose_mapping_key(payload, SCORE_KEYS)
        return _extract_numeric_payload(payload[key], source_path=source_path)
    raise TypeError(
        f"Could not extract numeric capacity scores from {source_path}: "
        f"unsupported payload type {type(payload).__name__}"
    )


def _choose_mapping_key(payload: Mapping[str, Any], candidate_keys: Sequence[str]) -> str:
    for key in candidate_keys:
        if key in payload:
            return key
    if len(payload) == 1:
        return next(iter(payload.keys()))
    raise KeyError(
        f"Could not infer capacity score key. Expected one of {candidate_keys}; "
        f"available keys: {sorted(str(key) for key in payload.keys())}"
    )


def validate_capacity_scores(
    scores: Any,
    *,
    expected_num_blocks: Optional[int] = None,
    expected_num_layers: Optional[int] = None,
    name: str = "capacity_scores",
    allow_all_zero: bool = False,
) -> np.ndarray:
    """Validate finite, nonnegative block or layer capacity scores.

    Layer scores use shape ``[num_blocks, num_layers]``.
    """

    arr = np.asarray(scores, dtype=np.float64)
    _validate_score_values(arr, name=name)

    if expected_num_layers is None:
        if arr.ndim != 1:
            raise ValueError(f"{name} must be 1D with one score per timestep block, got shape {arr.shape}")
        if expected_num_blocks is not None and arr.shape[0] != expected_num_blocks:
            raise ValueError(
                f"{name} length ({arr.shape[0]}) must match number of timestep blocks "
                f"({expected_num_blocks})"
            )
    else:
        if arr.ndim != 2:
            raise ValueError(
                f"{name} must be 2D with shape [num_blocks, num_layers], got shape {arr.shape}"
            )
        if expected_num_blocks is not None and arr.shape[0] != expected_num_blocks:
            raise ValueError(
                f"{name} block dimension ({arr.shape[0]}) must match number of timestep blocks "
                f"({expected_num_blocks})"
            )
        if arr.shape[1] != expected_num_layers:
            raise ValueError(
                f"{name} layer dimension ({arr.shape[1]}) must match expected layers/modules/stages "
                f"({expected_num_layers})"
            )

    if np.all(arr == 0) and not allow_all_zero:
        raise ValueError(
            f"{name} are all zero; cannot derive adaptive capacity budgets. "
            "Use the uniform variant or explicitly allow uniform fallback."
        )
    return arr


def _validate_score_values(arr: np.ndarray, *, name: str) -> None:
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(arr < 0):
        raise ValueError(f"{name} must be nonnegative")


def normalize_capacity_budgets(
    scores: Any,
    total_budget: float,
    alpha: float = 1.0,
    *,
    allow_uniform_if_all_zero: bool = False,
) -> np.ndarray:
    """Convert capacity scores to per-block budgets that sum to ``total_budget``."""

    if total_budget <= 0 or not np.isfinite(total_budget):
        raise ValueError(f"total_budget must be positive and finite, got {total_budget}")
    if alpha < 0 or not np.isfinite(alpha):
        raise ValueError(f"alpha must be nonnegative and finite, got {alpha}")

    arr = validate_capacity_scores(scores, allow_all_zero=allow_uniform_if_all_zero)
    if np.all(arr == 0) and allow_uniform_if_all_zero:
        return make_uniform_block_budgets(num_blocks=arr.shape[0], total_budget=total_budget)

    if alpha == 0:
        transformed = np.ones_like(arr, dtype=np.float64)
    else:
        transformed = np.power(arr, alpha)

    total_score = float(transformed.sum())
    if total_score <= 0:
        raise ValueError("capacity scores produced zero total allocation mass")
    return transformed / total_score * float(total_budget)


def make_uniform_block_budgets(num_blocks: int, total_budget: float) -> np.ndarray:
    if num_blocks <= 0:
        raise ValueError(f"num_blocks must be positive, got {num_blocks}")
    if total_budget <= 0 or not np.isfinite(total_budget):
        raise ValueError(f"total_budget must be positive and finite, got {total_budget}")
    return np.full(num_blocks, float(total_budget) / num_blocks, dtype=np.float64)


def make_global_budget(total_budget: float) -> np.ndarray:
    """Return the single-student global budget."""

    if total_budget <= 0 or not np.isfinite(total_budget):
        raise ValueError(f"total_budget must be positive and finite, got {total_budget}")
    return np.asarray([float(total_budget)], dtype=np.float64)


def make_blockwise_capacity_budgets(
    block_capacity_scores: Any,
    total_budget: float,
    alpha: float = 1.0,
    *,
    allow_uniform_if_all_zero: bool = False,
) -> np.ndarray:
    return normalize_capacity_budgets(
        block_capacity_scores,
        total_budget=total_budget,
        alpha=alpha,
        allow_uniform_if_all_zero=allow_uniform_if_all_zero,
    )


def make_shuffled_capacity_budgets(
    block_capacity_scores: Any,
    total_budget: float,
    alpha: float = 1.0,
    seed: Optional[int] = None,
    *,
    allow_uniform_if_all_zero: bool = False,
) -> np.ndarray:
    block_budgets = make_blockwise_capacity_budgets(
        block_capacity_scores,
        total_budget=total_budget,
        alpha=alpha,
        allow_uniform_if_all_zero=allow_uniform_if_all_zero,
    )
    rng = np.random.default_rng(seed)
    return rng.permutation(block_budgets)


def make_layerwise_capacity_budgets(
    block_budgets: Any,
    layer_capacity_scores: Any,
    alpha: float = 1.0,
    *,
    allow_uniform_if_all_zero: bool = False,
) -> np.ndarray:
    """Split each block budget across layers/modules/stages.

    ``layer_capacity_scores`` must have shape ``[num_blocks, num_layers]``.
    The returned array has the same shape, and each row sums to the matching
    value in ``block_budgets``.
    """

    if alpha < 0 or not np.isfinite(alpha):
        raise ValueError(f"alpha must be nonnegative and finite, got {alpha}")

    block_arr = np.asarray(block_budgets, dtype=np.float64)
    validate_budgets(block_arr, budget_name="block_budgets", fail_on_mismatch=False)
    if block_arr.ndim != 1:
        raise ValueError(f"block_budgets must be 1D with one budget per timestep block, got {block_arr.shape}")

    raw_scores = np.asarray(layer_capacity_scores, dtype=np.float64)
    if raw_scores.ndim != 2:
        raise ValueError(
            f"layer_capacity_scores must be 2D with shape [num_blocks, num_layers], got {raw_scores.shape}"
        )
    score_arr = validate_capacity_scores(
        raw_scores,
        expected_num_blocks=block_arr.shape[0],
        expected_num_layers=raw_scores.shape[1],
        name="layer_capacity_scores",
        allow_all_zero=allow_uniform_if_all_zero,
    )

    transformed = np.ones_like(score_arr, dtype=np.float64) if alpha == 0 else np.power(score_arr, alpha)
    row_sums = transformed.sum(axis=1, keepdims=True)
    zero_rows = np.flatnonzero(row_sums[:, 0] <= 0)
    if zero_rows.size:
        if allow_uniform_if_all_zero:
            transformed[zero_rows] = 1.0
            row_sums = transformed.sum(axis=1, keepdims=True)
        else:
            raise ValueError(
                "layer_capacity_scores contain all-zero rows for timestep blocks "
                f"{zero_rows.tolist()}; cannot derive layerwise budgets"
            )

    weights = transformed / row_sums
    layer_budgets = weights * block_arr.reshape(-1, 1)

    row_budget_sums = layer_budgets.sum(axis=1)
    if not np.allclose(row_budget_sums, block_arr, rtol=1e-10, atol=1e-8):
        raise RuntimeError("layerwise capacity budgets do not sum to their block budgets")
    return layer_budgets


def make_variant_capacity_budgets(
    student_variant: StudentVariant,
    *,
    num_blocks: int,
    total_budget: float,
    block_capacity_scores: Any = None,
    layer_capacity_scores: Any = None,
    alpha: float = 1.0,
    seed: Optional[int] = None,
    allow_uniform_if_all_zero: bool = False,
) -> CapacityBudgetPlan:
    """Build target stored budgets for one of the supported student variants."""

    if student_variant not in SUPPORTED_STUDENT_VARIANTS:
        raise ValueError(
            f"student_variant must be one of {SUPPORTED_STUDENT_VARIANTS}, got {student_variant!r}"
        )
    if num_blocks <= 0:
        raise ValueError(f"num_blocks must be positive, got {num_blocks}")

    layer_budgets: Optional[np.ndarray] = None
    if student_variant == "global":
        block_budgets = make_global_budget(total_budget)
    elif student_variant == "uniform_blockwise":
        block_budgets = make_uniform_block_budgets(num_blocks=num_blocks, total_budget=total_budget)
    elif student_variant in {"blockwise_capacity", "combined_blockwise"}:
        if block_capacity_scores is None:
            raise ValueError(f"block_capacity_scores are required for {student_variant}")
        validate_capacity_scores(
            block_capacity_scores,
            expected_num_blocks=num_blocks,
            name="block_capacity_scores",
            allow_all_zero=allow_uniform_if_all_zero,
        )
        block_budgets = make_blockwise_capacity_budgets(
            block_capacity_scores,
            total_budget=total_budget,
            alpha=alpha,
            allow_uniform_if_all_zero=allow_uniform_if_all_zero,
        )
    elif student_variant == "shuffled_capacity":
        if block_capacity_scores is None:
            raise ValueError("block_capacity_scores are required for shuffled_capacity")
        validate_capacity_scores(
            block_capacity_scores,
            expected_num_blocks=num_blocks,
            name="block_capacity_scores",
            allow_all_zero=allow_uniform_if_all_zero,
        )
        block_budgets = make_shuffled_capacity_budgets(
            block_capacity_scores,
            total_budget=total_budget,
            alpha=alpha,
            seed=seed,
            allow_uniform_if_all_zero=allow_uniform_if_all_zero,
        )
    else:
        if block_capacity_scores is None:
            raise ValueError(f"block_capacity_scores are required for {student_variant}")
        if layer_capacity_scores is None:
            raise ValueError(f"layer_capacity_scores are required for {student_variant}")
        validate_capacity_scores(
            block_capacity_scores,
            expected_num_blocks=num_blocks,
            name="block_capacity_scores",
            allow_all_zero=allow_uniform_if_all_zero,
        )
        block_budgets = make_blockwise_capacity_budgets(
            block_capacity_scores,
            total_budget=total_budget,
            alpha=alpha,
            allow_uniform_if_all_zero=allow_uniform_if_all_zero,
        )
        if student_variant == "reversed_layerwise_capacity":
            block_budgets = block_budgets[::-1].copy()
        layer_score_array = np.asarray(layer_capacity_scores, dtype=np.float64)
        if layer_score_array.ndim != 2:
            raise ValueError(
                "layer_capacity_scores are required with shape [num_blocks, num_layers] "
                f"for {student_variant}, got {layer_score_array.shape}"
            )
        validate_capacity_scores(
            layer_score_array,
            expected_num_blocks=num_blocks,
            expected_num_layers=layer_score_array.shape[1],
            name="layer_capacity_scores",
            allow_all_zero=allow_uniform_if_all_zero,
        )
        layer_budgets = make_layerwise_capacity_budgets(
            block_budgets,
            layer_score_array,
            alpha=alpha,
            allow_uniform_if_all_zero=allow_uniform_if_all_zero,
        )

    validate_budgets(block_budgets, budget_name=f"{student_variant}_target_parameters")
    validate_total_stored_budget(
        student_budgets=block_budgets,
        original_model_budget=total_budget,
        tolerance=1e-10,
    )
    return CapacityBudgetPlan(
        student_variant=student_variant,
        total_student_system_budget=float(total_budget),
        block_budgets=block_budgets.tolist(),
        layer_budgets=None if layer_budgets is None else layer_budgets.tolist(),
    )


def validate_total_stored_budget(
    student_budgets: Any,
    original_model_budget: Any,
    tolerance: float,
) -> BudgetValidationReport:
    """Validate total stored student budget against the original model budget."""

    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError(f"tolerance must be nonnegative and finite, got {tolerance}")
    original_parameters = _as_parameter_budget(original_model_budget)
    student_budget_array = np.asarray(student_budgets, dtype=np.float64)
    if student_budget_array.size == 0:
        raise ValueError("student_budgets must not be empty")
    if not np.all(np.isfinite(student_budget_array)):
        raise ValueError("student_budgets must be finite")
    if np.any(student_budget_array < 0):
        raise ValueError("student_budgets must be nonnegative")

    total_student_system_budget = float(student_budget_array.sum())
    report = validate_budgets(
        [float(original_parameters)],
        [total_student_system_budget],
        budget_tolerance=float(tolerance),
        budget_name="total_stored_parameters",
        fail_on_mismatch=True,
    )
    LOGGER.info(
        "total_student_system_budget=%s original_model_budget=%s tolerance=%s",
        total_student_system_budget,
        original_parameters,
        tolerance,
    )
    return report


def _as_parameter_budget(value: Any) -> float:
    if isinstance(value, ModelBudget):
        return float(value.parameters)
    if isinstance(value, Mapping):
        return float(compute_original_model_budget(value).parameters)
    if isinstance(value, (str, Path)) or _is_torch_module(value) or hasattr(value, "parameters"):
        return float(compute_original_model_budget(value).parameters)
    try:
        parameters = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "original_model_budget must be a positive numeric parameter count, ModelBudget, "
            "model/config mapping, path, or model object"
        ) from exc
    if parameters <= 0 or not np.isfinite(parameters):
        raise ValueError(f"original_model_budget must be positive, got {parameters}")
    return parameters


def validate_budgets(
    target_budgets: Any,
    realized_budgets: Any = None,
    *,
    budget_tolerance: float = 0.05,
    budget_name: str = "parameters",
    fail_on_mismatch: bool = True,
) -> BudgetValidationReport:
    """Validate target budgets and optional realized budgets after rounding."""

    if budget_tolerance < 0 or not np.isfinite(budget_tolerance):
        raise ValueError(f"budget_tolerance must be nonnegative and finite, got {budget_tolerance}")

    target = np.asarray(target_budgets, dtype=np.float64)
    if target.size == 0:
        raise ValueError("target_budgets must not be empty")
    if not np.all(np.isfinite(target)):
        raise ValueError("target_budgets must be finite")
    if np.any(target < 0):
        raise ValueError("target_budgets must be nonnegative")

    realized_list: Optional[list[float]] = None
    absolute_list: Optional[list[float]] = None
    relative_list: Optional[list[float]] = None
    within_tolerance = True

    if realized_budgets is not None:
        realized = np.asarray(realized_budgets, dtype=np.float64)
        if realized.shape != target.shape:
            raise ValueError(
                f"realized_budgets shape {realized.shape} must match target_budgets shape {target.shape}"
            )
        if not np.all(np.isfinite(realized)):
            raise ValueError("realized_budgets must be finite")
        if np.any(realized < 0):
            raise ValueError("realized_budgets must be nonnegative")

        absolute = realized - target
        relative = np.zeros_like(target, dtype=np.float64)
        positive_target = target > 0
        relative[positive_target] = absolute[positive_target] / target[positive_target]
        relative[~positive_target] = np.where(realized[~positive_target] == 0, 0.0, np.inf)
        within_tolerance = bool(np.all(np.abs(relative) <= budget_tolerance))

        LOGGER.info(
            "budget_validation %s target=%s realized=%s relative_mismatch=%s tolerance=%s",
            budget_name,
            target.tolist(),
            realized.tolist(),
            relative.tolist(),
            budget_tolerance,
        )
        if not within_tolerance:
            message = (
                f"Realized {budget_name} budgets differ from targets by more than "
                f"budget_tolerance={budget_tolerance}: relative mismatch {relative.tolist()}"
            )
            if fail_on_mismatch:
                raise ValueError(message)
            LOGGER.warning(message)

        realized_list = realized.tolist()
        absolute_list = absolute.tolist()
        relative_list = relative.tolist()
    else:
        LOGGER.info("budget_validation %s target=%s", budget_name, target.tolist())

    return BudgetValidationReport(
        target_budgets=target.tolist(),
        realized_budgets=realized_list,
        absolute_mismatch=absolute_list,
        relative_mismatch=relative_list,
        within_tolerance=within_tolerance,
        budget_tolerance=float(budget_tolerance),
        budget_name=budget_name,
    )
