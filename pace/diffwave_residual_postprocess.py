"""Residual-only postprocessing for completed DiffWave per-filter profiles.

The original evaluator intentionally reports every analyzed convolutional
filter.  That is useful for measurement completeness, but fixed stem/head
filters must not participate in student-capacity decisions.  This module
builds a separate, source-hash-bound view containing only groups for which
``group_allocatable`` is exactly ``True``.

No model evaluation is performed and the source profile is never modified.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import torch

from pace.profile_provenance import provenance_from_results

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESIDUAL_ANALYSIS_FORMAT = "diffdist_diffwave_residual_filter_analysis_v1"
RESIDUAL_METRICS_FORMAT = "diffdist_diffwave_residual_filter_metrics_v1"
SUCCESS_FORMAT = "diffdist_diffwave_residual_filter_success_v1"
DEFAULT_EXPECTED_ALLOCATABLE_FILTERS = 36_864


class _NumpyJSONEncoder(json.JSONEncoder):
    """Encode arrays lazily so large matrices need not be duplicated eagerly."""

    def default(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        return super().default(value)


@dataclass(frozen=True)
class ResidualFilterView:
    """Validated residual-only matrices and their derived usage metrics."""

    group_names: tuple[str, ...]
    source_group_indices: np.ndarray
    signed_delta_stack: np.ndarray
    delta_stack: np.ndarray
    relative_delta_stack: np.ndarray
    row_normalized_relative_delta_stack: np.ndarray
    ranked_relative_delta_stack: np.ndarray
    weights: np.ndarray
    n_eff: np.ndarray
    n_eff_fraction: np.ndarray
    p_eff: np.ndarray
    p_eff_edm_proxy: np.ndarray
    positive_delta_mass: np.ndarray
    source_positive_delta_mass: np.ndarray
    positive_delta_mass_fraction_of_source: np.ndarray
    signed_delta_positive_fraction: np.ndarray
    signed_delta_negative_fraction: np.ndarray
    signed_delta_zero_fraction: np.ndarray
    timestep_correlation_pearson: np.ndarray
    timestep_correlation_spearman: np.ndarray
    top_filter_indices: np.ndarray


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _require_group_mapping(
    results: Mapping[str, Any],
    key: str,
    group_names: Sequence[str],
) -> Mapping[str, Any]:
    value = results.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"source results must contain a {key!r} group mapping")
    missing = [name for name in group_names if name not in value]
    if missing:
        raise ValueError(f"{key} is missing {len(missing)} groups: {missing[:5]}")
    return value


def _numeric_matrix(
    value: Any,
    *,
    key: str,
    expected_rows: int,
    expected_columns: int,
) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a rectangular numeric matrix") from exc
    if matrix.shape != (expected_rows, expected_columns):
        raise ValueError(
            f"{key} has shape {matrix.shape}; expected "
            f"{(expected_rows, expected_columns)}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"{key} contains non-finite values")
    return matrix


def average_rank_columns(matrix: np.ndarray) -> np.ndarray:
    """Return one-based, average-tie ranks independently for every column."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"rank input must be two-dimensional, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("rank input contains non-finite values")

    ranks = np.empty_like(values, dtype=np.float64)
    for column_index in range(values.shape[1]):
        column = values[:, column_index]
        order = np.argsort(column, kind="stable")
        sorted_values = column[order]
        boundaries = np.flatnonzero(
            np.concatenate(
                (
                    np.array([True]),
                    sorted_values[1:] != sorted_values[:-1],
                    np.array([True]),
                )
            )
        )
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            # Ranks are one-based.  The average of [start + 1, ..., end] is
            # (start + 1 + end) / 2.
            ranks[order[start:end], column_index] = (start + end + 1.0) / 2.0
    return ranks


def timestep_pearson_correlation(matrix: np.ndarray) -> np.ndarray:
    """Correlate timestep columns across filters with finite constant handling."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"correlation input must be two-dimensional, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("correlation input contains non-finite values")
    num_timesteps = values.shape[1]
    if num_timesteps == 0:
        raise ValueError("correlation input must contain at least one timestep")
    if values.shape[0] < 2:
        return np.zeros((num_timesteps, num_timesteps), dtype=np.float64)

    centered = values - values.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    denominator = np.outer(norms, norms)
    correlation = np.divide(
        centered.T @ centered,
        denominator,
        out=np.zeros((num_timesteps, num_timesteps), dtype=np.float64),
        where=denominator > 0,
    )
    correlation = np.clip(correlation, -1.0, 1.0)
    correlation = 0.5 * (correlation + correlation.T)
    nonconstant = norms > 0
    correlation[np.diag_indices(num_timesteps)] = nonconstant.astype(np.float64)
    return correlation


def _top_filter_indices(relative_delta_stack: np.ndarray, count: int) -> np.ndarray:
    if count < 0:
        raise ValueError(f"top_filter_count must be non-negative, got {count}")
    count = min(count, relative_delta_stack.shape[0])
    scores = relative_delta_stack.sum(axis=1)
    # np.lexsort uses the last key as primary: descending score, then canonical
    # residual-only group index for deterministic ties.
    order = np.lexsort((np.arange(scores.size, dtype=np.int64), -scores))
    return order[:count].astype(np.int64, copy=False)


def compute_residual_filter_view(
    results: Mapping[str, Any],
    *,
    expected_allocatable_count: int | None = None,
    top_filter_count: int = 256,
) -> ResidualFilterView:
    """Select allocatable filters and recompute allocation-driving metrics."""

    raw_group_names = results.get("group_names")
    if not _is_sequence(raw_group_names):
        raise ValueError("source results must contain a group_names sequence")
    group_names = tuple(str(name) for name in raw_group_names)
    if not group_names or len(group_names) != len(set(group_names)):
        raise ValueError("source group_names must be non-empty and unique")

    baseline_mean = np.asarray(results.get("baseline_mean"), dtype=np.float64)
    if baseline_mean.ndim != 1 or baseline_mean.size == 0:
        raise ValueError("source baseline_mean must be a non-empty numeric vector")
    if not np.isfinite(baseline_mean).all():
        raise ValueError("source baseline_mean contains non-finite values")
    num_timesteps = int(baseline_mean.size)

    allocatable = _require_group_mapping(results, "group_allocatable", group_names)
    invalid_flags = [
        name for name in group_names if not isinstance(allocatable[name], (bool, np.bool_))
    ]
    if invalid_flags:
        raise ValueError(
            "group_allocatable values must be booleans; invalid groups: "
            f"{invalid_flags[:5]}"
        )
    source_indices = np.asarray(
        [index for index, name in enumerate(group_names) if allocatable[name] is True],
        dtype=np.int64,
    )
    # np.bool_(True) is intentionally accepted above but does not satisfy
    # ``is True``. JSON inputs always use bool; support in-memory numpy fixtures
    # as well without treating integers as flags.
    if any(isinstance(allocatable[name], np.bool_) for name in group_names):
        source_indices = np.asarray(
            [index for index, name in enumerate(group_names) if bool(allocatable[name])],
            dtype=np.int64,
        )
    if source_indices.size == 0:
        raise ValueError("group_allocatable selects no filters")
    if expected_allocatable_count is not None:
        if expected_allocatable_count <= 0:
            raise ValueError("expected_allocatable_count must be positive or None")
        if source_indices.size != expected_allocatable_count:
            raise ValueError(
                "residual-only scope selected "
                f"{source_indices.size:,} filters; expected "
                f"{expected_allocatable_count:,}"
            )

    full_signed = _numeric_matrix(
        results.get("signed_delta_stack"),
        key="signed_delta_stack",
        expected_rows=len(group_names),
        expected_columns=num_timesteps,
    )
    signed = np.ascontiguousarray(full_signed[source_indices])
    positive = np.maximum(signed, 0.0)

    if results.get("delta_stack") is not None:
        full_source_positive = _numeric_matrix(
            results.get("delta_stack"),
            key="delta_stack",
            expected_rows=len(group_names),
            expected_columns=num_timesteps,
        )
        selected_source_positive = full_source_positive[source_indices]
        if not np.array_equal(selected_source_positive, positive):
            raise ValueError(
                "source delta_stack is not the positive part of signed_delta_stack "
                "for the allocatable scope"
            )

    positive_mass = positive.sum(axis=0, dtype=np.float64)
    empty_bins = np.flatnonzero(positive_mass <= 0).tolist()
    if empty_bins:
        raise ValueError(
            "residual-only positive ablation-delta mass is zero for timestep bins "
            f"{empty_bins}"
        )

    relative = np.nan_to_num(
        positive / (baseline_mean[np.newaxis, :] + 1e-12),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    row_sums = relative.sum(axis=1, keepdims=True, dtype=np.float64)
    row_normalized = np.nan_to_num(
        relative / (row_sums + 1e-12),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    weights = np.nan_to_num(
        positive / (positive_mass[np.newaxis, :] + 1e-12),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    n_eff = 1.0 / (np.square(weights).sum(axis=0, dtype=np.float64) + 1e-12)
    n_eff_fraction = n_eff / float(source_indices.size)

    parameter_counts = _require_group_mapping(results, "group_param_counts", group_names)
    selected_names = tuple(group_names[index] for index in source_indices)
    try:
        counts = np.asarray([int(parameter_counts[name]) for name in selected_names], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("group_param_counts values must be integers") from exc
    if np.any(counts <= 0):
        raise ValueError("allocatable group_param_counts values must be positive")
    p_eff = (weights * counts[:, np.newaxis]).sum(axis=0, dtype=np.float64)

    proxy_mapping = results.get("group_edm_proxy_param_counts")
    if proxy_mapping is None:
        proxy_counts = counts
    else:
        if not isinstance(proxy_mapping, Mapping):
            raise ValueError("group_edm_proxy_param_counts must be a mapping")
        missing_proxy = [name for name in selected_names if name not in proxy_mapping]
        if missing_proxy:
            raise ValueError(
                "group_edm_proxy_param_counts is missing groups: "
                f"{missing_proxy[:5]}"
            )
        try:
            proxy_counts = np.asarray(
                [int(proxy_mapping[name]) for name in selected_names],
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("group_edm_proxy_param_counts values must be integers") from exc
        if np.any(proxy_counts <= 0):
            raise ValueError("allocatable group_edm_proxy_param_counts values must be positive")
    p_eff_edm_proxy = (weights * proxy_counts[:, np.newaxis]).sum(
        axis=0,
        dtype=np.float64,
    )

    raw_source_mass = results.get("positive_delta_mass")
    if raw_source_mass is None:
        if results.get("delta_stack") is None:
            source_positive_mass = np.maximum(full_signed, 0.0).sum(
                axis=0,
                dtype=np.float64,
            )
        else:
            source_positive_mass = full_source_positive.sum(axis=0, dtype=np.float64)
    else:
        source_positive_mass = np.asarray(raw_source_mass, dtype=np.float64)
        if source_positive_mass.shape != (num_timesteps,):
            raise ValueError(
                "positive_delta_mass has shape "
                f"{source_positive_mass.shape}; expected {(num_timesteps,)}"
            )
        if not np.isfinite(source_positive_mass).all() or np.any(source_positive_mass <= 0):
            raise ValueError("source positive_delta_mass must be finite and positive")
    positive_mass_fraction = np.divide(
        positive_mass,
        source_positive_mass,
        out=np.zeros_like(positive_mass),
        where=source_positive_mass > 0,
    )

    positive_fraction = (signed > 0).mean(axis=0, dtype=np.float64)
    negative_fraction = (signed < 0).mean(axis=0, dtype=np.float64)
    zero_fraction = (signed == 0).mean(axis=0, dtype=np.float64)
    ranked = average_rank_columns(relative)
    pearson = timestep_pearson_correlation(relative)
    spearman = timestep_pearson_correlation(ranked)
    top_indices = _top_filter_indices(relative, top_filter_count)

    arrays = (
        signed,
        positive,
        relative,
        row_normalized,
        ranked,
        weights,
        n_eff,
        n_eff_fraction,
        p_eff,
        p_eff_edm_proxy,
        positive_mass,
        source_positive_mass,
        positive_mass_fraction,
        positive_fraction,
        negative_fraction,
        zero_fraction,
        pearson,
        spearman,
    )
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("residual-only postprocessing produced non-finite metrics")

    return ResidualFilterView(
        group_names=selected_names,
        source_group_indices=source_indices,
        signed_delta_stack=signed,
        delta_stack=positive,
        relative_delta_stack=relative,
        row_normalized_relative_delta_stack=row_normalized,
        ranked_relative_delta_stack=ranked,
        weights=weights,
        n_eff=n_eff,
        n_eff_fraction=n_eff_fraction,
        p_eff=p_eff,
        p_eff_edm_proxy=p_eff_edm_proxy,
        positive_delta_mass=positive_mass,
        source_positive_delta_mass=source_positive_mass,
        positive_delta_mass_fraction_of_source=positive_mass_fraction,
        signed_delta_positive_fraction=positive_fraction,
        signed_delta_negative_fraction=negative_fraction,
        signed_delta_zero_fraction=zero_fraction,
        timestep_correlation_pearson=pearson,
        timestep_correlation_spearman=spearman,
        top_filter_indices=top_indices,
    )


def _selected_mapping(
    results: Mapping[str, Any],
    key: str,
    selected_names: Sequence[str],
    *,
    required: bool = False,
) -> dict[str, Any] | None:
    raw = results.get(key)
    if raw is None and not required:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"source {key} must be a mapping")
    missing = [name for name in selected_names if name not in raw]
    if missing:
        raise ValueError(f"source {key} is missing groups: {missing[:5]}")
    return {name: raw[name] for name in selected_names}


def _timestep_labels(results: Mapping[str, Any], num_timesteps: int) -> list[str]:
    for key in ("timestep_bin_labels", "bin_labels"):
        raw = results.get(key)
        if raw is not None:
            if not _is_sequence(raw) or len(raw) != num_timesteps:
                raise ValueError(f"source {key} must contain {num_timesteps} labels")
            return [str(label) for label in raw]
    return [str(index) for index in range(num_timesteps)]


def _top_filter_metadata(
    results: Mapping[str, Any],
    view: ResidualFilterView,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parameter_counts = _selected_mapping(
        results,
        "group_param_counts",
        view.group_names,
        required=True,
    )
    assert parameter_counts is not None
    proxy_counts = _selected_mapping(
        results,
        "group_edm_proxy_param_counts",
        view.group_names,
    )
    optional_mappings = {
        key: _selected_mapping(results, key, view.group_names)
        for key in (
            "group_module_paths",
            "group_structural_keys",
            "group_stage_keys",
            "group_allocation_structural_keys",
            "group_filter_indices",
        )
    }
    scores = view.relative_delta_stack.sum(axis=1, dtype=np.float64)
    delta_mass = view.delta_stack.sum(axis=1, dtype=np.float64)
    total_delta_mass = float(delta_mass.sum())
    records: list[dict[str, Any]] = []
    for rank, group_index_value in enumerate(view.top_filter_indices, start=1):
        group_index = int(group_index_value)
        name = view.group_names[group_index]
        record: dict[str, Any] = {
            "rank": rank,
            "group_index": group_index,
            "source_group_index": int(view.source_group_indices[group_index]),
            "group_name": name,
            "sum_positive_relative_delta": float(scores[group_index]),
            "sum_positive_delta": float(delta_mass[group_index]),
            "positive_mass_fraction_within_residual": (
                float(delta_mass[group_index] / total_delta_mass)
                if total_delta_mass > 0
                else 0.0
            ),
            "parameter_count": int(parameter_counts[name]),
        }
        if proxy_counts is not None:
            record["edm_proxy_parameter_count"] = int(proxy_counts[name])
        for mapping_key, mapping in optional_mappings.items():
            if mapping is not None:
                record[mapping_key.removeprefix("group_")] = mapping[name]
        records.append(record)

    compatibility = {
        "selection": "sum_positive_relative_delta_descending_then_group_index",
        "count": len(records),
        "indices": [record["group_index"] for record in records],
        "source_indices": [record["source_group_index"] for record in records],
        "names": [record["group_name"] for record in records],
        "scores": [record["sum_positive_relative_delta"] for record in records],
    }
    return compatibility, records


def build_residual_results_payload(
    source_results: Mapping[str, Any],
    view: ResidualFilterView,
    *,
    source_path: Path,
    invocation_argv: Sequence[str] | None = None,
    artifact_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a JSON-compatible, allocation-tool-friendly derived profile."""

    provenance = provenance_from_results(source_results, source_path)
    source_group_names = source_results["group_names"]
    num_timesteps = int(view.delta_stack.shape[1])
    labels = _timestep_labels(source_results, num_timesteps)
    selected_param_counts = _selected_mapping(
        source_results,
        "group_param_counts",
        view.group_names,
        required=True,
    )
    assert selected_param_counts is not None
    selected_proxy_counts = _selected_mapping(
        source_results,
        "group_edm_proxy_param_counts",
        view.group_names,
    )
    top_filter_plot, top_filters = _top_filter_metadata(source_results, view)

    source_model_info = source_results.get("model_info")
    source_model_info = dict(source_model_info) if isinstance(source_model_info, Mapping) else {}
    model_info = {
        key: source_model_info[key]
        for key in (
            "teacher",
            "checkpoint_format",
            "checkpoint_preset",
            "checkpoint_sha256",
            "model_family",
            "total_parameter_count",
        )
        if key in source_model_info
    }
    model_info.update(
        {
            "grouping": "per_filter_residual_only",
            "analyzed_group_parameter_count": int(
                sum(int(value) for value in selected_param_counts.values())
            ),
            "allocatable_residual_filter_parameter_count": int(
                sum(int(value) for value in selected_param_counts.values())
            ),
        }
    )

    invocation = None
    if invocation_argv is not None:
        argv = [str(value) for value in invocation_argv]
        invocation = {
            "argv": argv,
            "command": " ".join(shlex.quote(value) for value in argv),
        }

    payload: dict[str, Any] = {
        "format": RESIDUAL_ANALYSIS_FORMAT,
        "description": (
            "Allocation-relevant DiffWave per-filter metrics recomputed only "
            "over source groups where group_allocatable is true."
        ),
        **provenance,
        "source_format": source_results.get("format"),
        "source_analysis_basis_sha256": source_results.get("analysis_basis_sha256"),
        "profile_fingerprint": source_results.get("profile_fingerprint"),
        "profile_fingerprint_sha256": source_results.get("profile_fingerprint_sha256"),
        "invocation": invocation,
        "scope": {
            "selection_field": "group_allocatable",
            "selection_value": True,
            "source_group_count": len(source_group_names),
            "selected_group_count": len(view.group_names),
            "excluded_group_count": len(source_group_names) - len(view.group_names),
            "selected_source_group_indices": view.source_group_indices.tolist(),
            "allocatable_parameter_count": int(
                sum(int(value) for value in selected_param_counts.values())
            ),
            "positive_matrix_definition": "maximum(selected signed_delta_stack, 0)",
            "n_eff_fraction_denominator": len(view.group_names),
            "correlation_features": "relative_delta_stack columns across selected filters",
            "spearman_rank_method": "columnwise one-based average ranks for ties",
        },
        "model_info": model_info,
        "diffusion_axis": source_results.get("diffusion_axis"),
        "timestep_values": source_results.get("timestep_values"),
        "timestep_bin_members": source_results.get("timestep_bin_members"),
        "timestep_bin_labels": labels,
        "group_names": list(view.group_names),
        "group_param_counts": selected_param_counts,
        "group_edm_proxy_param_counts": (
            selected_proxy_counts if selected_proxy_counts is not None else dict(selected_param_counts)
        ),
        "group_module_paths": _selected_mapping(
            source_results, "group_module_paths", view.group_names
        ),
        "group_structural_keys": _selected_mapping(
            source_results, "group_structural_keys", view.group_names
        ),
        "group_stage_keys": _selected_mapping(
            source_results, "group_stage_keys", view.group_names
        ),
        "group_allocation_structural_keys": _selected_mapping(
            source_results, "group_allocation_structural_keys", view.group_names
        ),
        "group_allocatable": {name: True for name in view.group_names},
        "group_filter_indices": _selected_mapping(
            source_results, "group_filter_indices", view.group_names
        ),
        "full_group_count": len(view.group_names),
        "baseline_mean": np.asarray(source_results["baseline_mean"], dtype=np.float64),
        "baseline_stderr": np.asarray(
            source_results.get("baseline_stderr", np.zeros(num_timesteps)),
            dtype=np.float64,
        ),
        "baseline_count": np.asarray(
            source_results.get("baseline_count", np.zeros(num_timesteps, dtype=np.int64)),
            dtype=np.int64,
        ),
        "signed_delta_stack": view.signed_delta_stack,
        "delta_stack": view.delta_stack,
        "relative_delta_stack": view.relative_delta_stack,
        "row_normalized_relative_delta_stack": view.row_normalized_relative_delta_stack,
        "ranked_relative_delta_stack": view.ranked_relative_delta_stack,
        "weights": view.weights,
        "n_eff": view.n_eff,
        "n_eff_fraction": view.n_eff_fraction,
        "p_eff": view.p_eff,
        "p_eff_edm_proxy": view.p_eff_edm_proxy,
        "positive_delta_mass": view.positive_delta_mass,
        "source_positive_delta_mass": view.source_positive_delta_mass,
        "positive_delta_mass_fraction_of_source": (
            view.positive_delta_mass_fraction_of_source
        ),
        "total_positive_delta_mass": float(view.positive_delta_mass.sum()),
        "total_source_positive_delta_mass": float(view.source_positive_delta_mass.sum()),
        "total_positive_delta_mass_fraction_of_source": float(
            view.positive_delta_mass.sum() / view.source_positive_delta_mass.sum()
        ),
        "signed_delta_positive_fraction": view.signed_delta_positive_fraction,
        "signed_delta_negative_fraction": view.signed_delta_negative_fraction,
        "signed_delta_zero_fraction": view.signed_delta_zero_fraction,
        # Keep C_timesteps as the historical Pearson-compatible key while
        # exposing both choices explicitly for robust downstream grouping.
        "C_timesteps": view.timestep_correlation_pearson,
        "C_timesteps_pearson": view.timestep_correlation_pearson,
        "C_timesteps_spearman": view.timestep_correlation_spearman,
        "C_groups": None,
        "group_correlation": {
            "computed": False,
            "reason": "disabled_for_large_residual_per_filter_scope",
            "num_groups": len(view.group_names),
        },
        "top_filter_plot": top_filter_plot,
        "top_filters": top_filters,
        "artifacts": dict(artifact_paths or {}),
    }
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any], *, compact: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                cls=_NumpyJSONEncoder,
                allow_nan=False,
                indent=None if compact else 2,
                separators=(",", ":") if compact else None,
            )
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _tensor(value: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(value))


def _metrics_payload(
    results_payload: Mapping[str, Any],
    view: ResidualFilterView,
) -> dict[str, Any]:
    return {
        "format": RESIDUAL_METRICS_FORMAT,
        "source_profile": results_payload["source_profile"],
        "ablation_protocol": results_payload["ablation_protocol"],
        "scope": results_payload["scope"],
        "group_names": list(view.group_names),
        "source_group_indices": _tensor(view.source_group_indices),
        "baseline_mean": torch.as_tensor(results_payload["baseline_mean"], dtype=torch.float64),
        "signed_delta_stack": _tensor(view.signed_delta_stack),
        "delta_stack": _tensor(view.delta_stack),
        "relative_delta_stack": _tensor(view.relative_delta_stack),
        "row_normalized_relative_delta_stack": _tensor(
            view.row_normalized_relative_delta_stack
        ),
        "ranked_relative_delta_stack": _tensor(view.ranked_relative_delta_stack),
        "weights": _tensor(view.weights),
        "n_eff": _tensor(view.n_eff),
        "n_eff_fraction": _tensor(view.n_eff_fraction),
        "p_eff": _tensor(view.p_eff),
        "p_eff_edm_proxy": _tensor(view.p_eff_edm_proxy),
        "positive_delta_mass": _tensor(view.positive_delta_mass),
        "source_positive_delta_mass": _tensor(view.source_positive_delta_mass),
        "positive_delta_mass_fraction_of_source": _tensor(
            view.positive_delta_mass_fraction_of_source
        ),
        "signed_delta_positive_fraction": _tensor(view.signed_delta_positive_fraction),
        "signed_delta_negative_fraction": _tensor(view.signed_delta_negative_fraction),
        "signed_delta_zero_fraction": _tensor(view.signed_delta_zero_fraction),
        "C_timesteps": _tensor(view.timestep_correlation_pearson),
        "C_timesteps_pearson": _tensor(view.timestep_correlation_pearson),
        "C_timesteps_spearman": _tensor(view.timestep_correlation_spearman),
        "top_filter_indices": _tensor(view.top_filter_indices),
    }


def _sparse_positions(count: int, maximum: int) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.unique(np.linspace(0, count - 1, maximum, dtype=np.int64))


def _plot_heatmap(
    matrix: np.ndarray,
    *,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    path: Path,
    title: str,
    colorbar_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    row_positions = _sparse_positions(len(row_labels), 40)
    column_positions = _sparse_positions(len(column_labels), 25)
    height = max(5.0, min(28.0, 0.11 * len(row_labels) + 3.0))
    figure, axis = plt.subplots(figsize=(13, height))
    image = axis.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    axis.set_xticks(column_positions, [column_labels[index] for index in column_positions])
    axis.tick_params(axis="x", labelrotation=90, labelsize=8)
    axis.set_yticks(row_positions, [row_labels[index] for index in row_positions])
    axis.tick_params(axis="y", labelsize=7)
    axis.set_xlabel("Timestep bin")
    axis.set_ylabel("Residual filter")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label=colorbar_label)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _save_plots(
    output_dir: Path,
    results: Mapping[str, Any],
    view: ResidualFilterView,
) -> list[Path]:
    labels = _timestep_labels(results, view.delta_stack.shape[1])
    paths: list[Path] = []
    if view.top_filter_indices.size:
        top_names = [view.group_names[int(index)] for index in view.top_filter_indices]
        relative_path = output_dir / "top_filter_relative_delta_heatmap.png"
        _plot_heatmap(
            view.relative_delta_stack[view.top_filter_indices],
            row_labels=top_names,
            column_labels=labels,
            path=relative_path,
            title="Top residual filters: positive relative PFI delta",
            colorbar_label="Relative delta",
        )
        paths.append(relative_path)
        normalized_path = output_dir / "top_filter_row_normalized_heatmap.png"
        _plot_heatmap(
            view.row_normalized_relative_delta_stack[view.top_filter_indices],
            row_labels=top_names,
            column_labels=labels,
            path=normalized_path,
            title="Top residual filters: row-normalized timestep profile",
            colorbar_label="Row fraction",
            vmin=0.0,
            vmax=1.0,
        )
        paths.append(normalized_path)

    for matrix, filename, title in (
        (
            view.timestep_correlation_pearson,
            "timestep_correlation_pearson.png",
            "Residual-filter timestep Pearson correlation",
        ),
        (
            view.timestep_correlation_spearman,
            "timestep_correlation_spearman.png",
            "Residual-filter timestep Spearman correlation",
        ),
    ):
        path = output_dir / filename
        _plot_heatmap(
            matrix,
            row_labels=labels,
            column_labels=labels,
            path=path,
            title=title,
            colorbar_label="Correlation",
            vmin=-1.0,
            vmax=1.0,
        )
        paths.append(path)

    series_path = output_dir / "residual_usage_summary.png"
    x = np.arange(len(labels))
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(x, view.n_eff, marker="o", linewidth=1)
    axes[0].set_ylabel("N_eff")
    axes[0].set_title("Residual-only allocation metrics")
    axes[1].plot(x, view.n_eff_fraction, marker="o", linewidth=1)
    axes[1].set_ylabel("N_eff / residual filters")
    axes[2].plot(
        x,
        view.positive_delta_mass_fraction_of_source,
        marker="o",
        linewidth=1,
    )
    axes[2].set_ylabel("Residual / all positive mass")
    positions = _sparse_positions(len(labels), 25)
    axes[2].set_xticks(positions, [labels[index] for index in positions], rotation=90)
    axes[2].set_xlabel("Timestep bin")
    figure.tight_layout()
    figure.savefig(series_path, dpi=150)
    plt.close(figure)
    paths.append(series_path)
    return paths


def resolve_source_results_path(path: str | Path) -> Path:
    source = Path(path).expanduser().resolve()
    if source.is_dir():
        source = source / "results.json"
    if not source.is_file():
        raise FileNotFoundError(f"DiffWave per-filter results do not exist: {source}")
    return source


def postprocess_residual_filter_results(
    source: str | Path,
    *,
    output_dir: str | Path | None = None,
    expected_allocatable_count: int | None = DEFAULT_EXPECTED_ALLOCATABLE_FILTERS,
    top_filter_count: int = 256,
    save_plots: bool = True,
    invocation_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Create source-bound residual-only JSON/PT/plot artifacts atomically."""

    source_path = resolve_source_results_path(source)
    destination = (
        source_path.parent / "residual_only"
        if output_dir is None
        else Path(output_dir).expanduser().resolve()
    )
    destination.mkdir(parents=True, exist_ok=True)
    results_path = destination / "results.json"
    if results_path.resolve() == source_path:
        raise ValueError("derived results path must differ from the source results path")

    try:
        source_results = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read source results {source_path}: {exc}") from exc
    if not isinstance(source_results, Mapping):
        raise ValueError(f"source results must contain a JSON object: {source_path}")

    view = compute_residual_filter_view(
        source_results,
        expected_allocatable_count=expected_allocatable_count,
        top_filter_count=top_filter_count,
    )
    artifact_names = {
        "results": "results.json",
        "metrics": "metrics.pt",
    }
    if save_plots:
        artifact_names.update(
            {
                "top_filter_relative_delta_heatmap": (
                    "top_filter_relative_delta_heatmap.png"
                ),
                "top_filter_row_normalized_heatmap": (
                    "top_filter_row_normalized_heatmap.png"
                ),
                "timestep_correlation_pearson": "timestep_correlation_pearson.png",
                "timestep_correlation_spearman": "timestep_correlation_spearman.png",
                "residual_usage_summary": "residual_usage_summary.png",
            }
        )
    results_payload = build_residual_results_payload(
        source_results,
        view,
        source_path=source_path,
        invocation_argv=invocation_argv,
        artifact_paths=artifact_names,
    )

    metrics_path = destination / "metrics.pt"
    _atomic_torch_save(metrics_path, _metrics_payload(results_payload, view))
    generated_plots = _save_plots(destination, source_results, view) if save_plots else []
    _atomic_write_json(results_path, results_payload, compact=True)

    artifact_paths = [results_path, metrics_path, *generated_plots]
    artifact_sha256 = {
        path.name: _file_sha256(path) for path in sorted(artifact_paths, key=lambda item: item.name)
    }
    source_profile = results_payload["source_profile"]
    success = {
        "format": SUCCESS_FORMAT,
        "passed": True,
        "source_profile": source_profile,
        "ablation_protocol": results_payload["ablation_protocol"],
        "output_dir": str(destination),
        "results_path": str(results_path),
        "results_sha256": artifact_sha256[results_path.name],
        "metrics_path": str(metrics_path),
        "metrics_sha256": artifact_sha256[metrics_path.name],
        "selected_group_count": len(view.group_names),
        "artifact_count": len(artifact_sha256),
        "artifact_sha256": artifact_sha256,
    }
    _atomic_write_json(destination / "_SUCCESS.json", success, compact=False)
    return success
