#!/usr/bin/env python3
"""Validate and summarize the corrected DiffWave per-filter decision artifacts.

This report makes the fixed Pearson K=3 design canonical, while retaining
auto-K and Spearman partitions as diagnostics.  It performs no teacher
inference and never modifies the source or residual-only results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPORT_FORMAT = "diffdist_diffwave_per_filter_decision_v2"


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {label} at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_source_hash(
    payload: Mapping[str, Any],
    *,
    expected: str,
    label: str,
) -> None:
    source = payload.get("source_profile")
    if not isinstance(source, Mapping):
        raise ValueError(f"{label} does not contain source_profile provenance")
    actual = source.get("results_sha256")
    if actual != expected:
        raise ValueError(
            f"{label} is bound to residual results SHA-256 {actual!r}, "
            f"expected {expected!r}"
        )


def _phase_ranges(
    boundaries: Sequence[int],
    timestep_bin_members: Sequence[Sequence[int]],
) -> list[dict[str, Any]]:
    ranges: list[dict[str, Any]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        if not (0 <= start < end <= len(timestep_bin_members)):
            raise ValueError(f"invalid grouping interval [{start}, {end})")
        raw_timesteps = [
            int(timestep)
            for members in timestep_bin_members[start:end]
            for timestep in members
        ]
        if not raw_timesteps:
            raise ValueError(f"grouping interval [{start}, {end}) is empty")
        ranges.append(
            {
                "bin_interval": [int(start), int(end)],
                "bin_count": int(end - start),
                "raw_timestep_count": len(raw_timesteps),
                "first_raw_timestep": raw_timesteps[0],
                "last_raw_timestep": raw_timesteps[-1],
                "label": f"t={raw_timesteps[0]}..{raw_timesteps[-1]}",
            }
        )
    return ranges


def _grouping_summary(
    payload: Mapping[str, Any],
    *,
    expected_builtin_cost: str,
    timestep_bin_members: Sequence[Sequence[int]],
) -> dict[str, Any]:
    if payload.get("builtin_cost") != expected_builtin_cost:
        raise ValueError(
            f"grouping uses {payload.get('builtin_cost')!r}; "
            f"expected {expected_builtin_cost!r}"
        )
    boundaries = [int(value) for value in payload.get("boundaries", [])]
    num_blocks = int(payload.get("num_blocks", 0))
    if len(boundaries) != num_blocks + 1:
        raise ValueError("grouping boundaries do not match num_blocks")
    selection = payload.get("num_blocks_selection")
    candidates: list[dict[str, Any]] = []
    if isinstance(selection, Mapping):
        raw_candidates = selection.get("candidates", [])
        if isinstance(raw_candidates, Sequence):
            candidates = [
                {
                    "num_blocks": int(candidate["num_blocks"]),
                    "total_score": float(candidate["total_score"]),
                }
                for candidate in raw_candidates
                if isinstance(candidate, Mapping)
            ]
    return {
        "pairwise_metric": payload.get("pairwise_metric"),
        "objective": payload.get("builtin_cost"),
        "cross_block_lambda": (
            None
            if payload.get("cross_block_lambda") is None
            else float(payload["cross_block_lambda"])
        ),
        "min_block_size": int(payload.get("min_block_size", 1)),
        "num_blocks": num_blocks,
        "boundaries": boundaries,
        "phases": _phase_ranges(boundaries, timestep_bin_members),
        "total_score": float(payload["total_score"]),
        "selection": selection,
        "candidate_scores": candidates,
    }


def _allocation_summary(
    payload: Mapping[str, Any],
    *,
    expected_grouping: Path,
) -> dict[str, Any]:
    if payload.get("student_variant") != "combined_blockwise":
        raise ValueError("allocation input must be the combined_blockwise artifact")
    if payload.get("allocation_group_scope") != "allocatable":
        raise ValueError("allocation must use allocation_group_scope='allocatable'")
    if payload.get("score_reduction") != "mean":
        raise ValueError("allocation must use duration-neutral score_reduction='mean'")
    grouping_path = Path(str(payload.get("timestep_grouping_path"))).resolve()
    if grouping_path != expected_grouping.resolve():
        raise ValueError(
            f"allocation grouping path is {grouping_path}, expected {expected_grouping.resolve()}"
        )
    score_sources = payload.get("score_sources")
    if not isinstance(score_sources, Mapping):
        raise ValueError("allocation does not contain score_sources")
    block_source = score_sources.get("block_capacity_scores")
    if not isinstance(block_source, Mapping):
        raise ValueError("allocation does not describe its block capacity scores")
    expected_fields = {
        "metric": "delta_p_eff_geomean",
        "group_scope": "allocatable",
        "reduction": "mean",
        "reduction_protocol": "duration_neutral_mean",
        "p_eff_source": "recomputed_from_allocatable_groups",
    }
    for key, expected in expected_fields.items():
        if block_source.get(key) != expected:
            raise ValueError(
                f"allocation score source {key} is {block_source.get(key)!r}, "
                f"expected {expected!r}"
            )
    plan = payload.get("target_budget_plan")
    if not isinstance(plan, Mapping):
        raise ValueError("allocation does not contain target_budget_plan")
    budgets = [float(value) for value in plan.get("block_budgets", [])]
    if not budgets or not all(math.isfinite(value) and value > 0 for value in budgets):
        raise ValueError("allocation block budgets must be finite and positive")
    total_budget = float(plan.get("total_student_system_budget", sum(budgets)))
    if not math.isclose(sum(budgets), total_budget, rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError("allocation block budgets do not sum to the total budget")
    return {
        "metric": "sqrt(positive_delta_mass * residual_p_eff)",
        "group_scope": "allocatable_residual_filters",
        "phase_reduction": "duration_neutral_mean",
        "total_residual_parameter_budget": total_budget,
        "block_budgets": budgets,
        "block_budget_fractions": [value / total_budget for value in budgets],
        "timestep_blocks": payload.get("timestep_blocks"),
    }


def build_report(residual_dir: str | Path) -> dict[str, Any]:
    root = Path(residual_dir).expanduser().resolve()
    paths = {
        "residual_results": root / "results.json",
        "residual_success": root / "_SUCCESS.json",
        "spearman_auto_grouping": root
        / "grouping_spearman_auto"
        / "timestep_grouping.json",
        "pearson_auto_grouping": root
        / "grouping_pearson_auto"
        / "timestep_grouping.json",
        "pearson_k3_grouping": root
        / "grouping_pearson_k3_min2"
        / "timestep_grouping.json",
        "pearson_k3_grouping_matrix": root
        / "grouping_pearson_k3_min2"
        / "timestep_grouping_matrix.png",
        "pearson_k3_grouping_n_eff": root
        / "grouping_pearson_k3_min2"
        / "timestep_grouping_n_eff.png",
        "pearson_k3_grouping_cost_curve": root
        / "grouping_pearson_k3_min2"
        / "timestep_grouping_cost_curve.png",
        "spearman_k3_grouping": root
        / "grouping_spearman_k3_min2"
        / "timestep_grouping.json",
        "spearman_auto_allocation": root
        / "allocation_geomean_mean_spearman_auto"
        / "combined_blockwise.json",
        "pearson_auto_allocation": root
        / "allocation_geomean_mean_pearson_auto"
        / "combined_blockwise.json",
        "pearson_k3_allocation": root
        / "allocation_geomean_mean_pearson_k3_min2"
        / "combined_blockwise.json",
        "pearson_k3_layerwise_allocation": root
        / "allocation_geomean_mean_pearson_k3_min2"
        / "combined_layerwise.json",
        "pearson_k3_allocation_summary": root
        / "allocation_geomean_mean_pearson_k3_min2"
        / "summary.json",
        "spearman_k3_allocation": root
        / "allocation_geomean_mean_spearman_k3_min2"
        / "combined_blockwise.json",
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")

    results = _load_json(paths["residual_results"], "residual results")
    success = _load_json(paths["residual_success"], "residual success marker")
    residual_sha256 = _sha256(paths["residual_results"])
    if success.get("passed") is not True or success.get("results_sha256") != residual_sha256:
        raise ValueError("residual success marker does not validate residual results.json")

    raw_bin_members = results.get("timestep_bin_members")
    if not isinstance(raw_bin_members, Sequence) or not raw_bin_members:
        raise ValueError("residual results do not contain timestep_bin_members")
    timestep_bin_members = [list(map(int, members)) for members in raw_bin_members]

    grouping_payloads = {
        key: _load_json(paths[key], key)
        for key in (
            "spearman_auto_grouping",
            "pearson_auto_grouping",
            "pearson_k3_grouping",
            "spearman_k3_grouping",
        )
    }
    for key, payload in grouping_payloads.items():
        _require_source_hash(payload, expected=residual_sha256, label=key)

    groupings = {
        "spearman_auto": _grouping_summary(
            grouping_payloads["spearman_auto_grouping"],
            expected_builtin_cost="matrix_spearman_cross_penalty",
            timestep_bin_members=timestep_bin_members,
        ),
        "pearson_auto": _grouping_summary(
            grouping_payloads["pearson_auto_grouping"],
            expected_builtin_cost="matrix_correlation_cross_penalty",
            timestep_bin_members=timestep_bin_members,
        ),
        "pearson_fixed_k3": _grouping_summary(
            grouping_payloads["pearson_k3_grouping"],
            expected_builtin_cost="matrix_correlation_cross_penalty",
            timestep_bin_members=timestep_bin_members,
        ),
        "spearman_fixed_k3_sensitivity": _grouping_summary(
            grouping_payloads["spearman_k3_grouping"],
            expected_builtin_cost="matrix_spearman_cross_penalty",
            timestep_bin_members=timestep_bin_members,
        ),
    }
    if groupings["pearson_fixed_k3"]["num_blocks"] != 3:
        raise ValueError("the fixed Pearson grouping must contain three phases")
    if groupings["pearson_fixed_k3"]["cross_block_lambda"] != 0.0:
        raise ValueError(
            "the canonical fixed Pearson grouping must use cross_block_lambda=0"
        )
    if groupings["spearman_fixed_k3_sensitivity"]["num_blocks"] != 3:
        raise ValueError("the fixed-K sensitivity grouping must contain three phases")

    allocation_specs = {
        "spearman_auto": ("spearman_auto_allocation", "spearman_auto_grouping"),
        "pearson_auto": ("pearson_auto_allocation", "pearson_auto_grouping"),
        "pearson_fixed_k3": (
            "pearson_k3_allocation",
            "pearson_k3_grouping",
        ),
        "spearman_fixed_k3_sensitivity": (
            "spearman_k3_allocation",
            "spearman_k3_grouping",
        ),
    }
    allocations: dict[str, Any] = {}
    for output_key, (allocation_key, grouping_key) in allocation_specs.items():
        payload = _load_json(paths[allocation_key], allocation_key)
        _require_source_hash(payload, expected=residual_sha256, label=allocation_key)
        allocations[output_key] = _allocation_summary(
            payload,
            expected_grouping=paths[grouping_key],
        )

    pearson_layerwise = _load_json(
        paths["pearson_k3_layerwise_allocation"],
        "pearson_k3_layerwise_allocation",
    )
    _require_source_hash(
        pearson_layerwise,
        expected=residual_sha256,
        label="pearson_k3_layerwise_allocation",
    )
    if pearson_layerwise.get("student_variant") != "combined_layerwise":
        raise ValueError("canonical layerwise artifact has the wrong student variant")
    if pearson_layerwise.get("allocation_group_scope") != "allocatable":
        raise ValueError("canonical layerwise allocation must use allocatable groups")
    if pearson_layerwise.get("score_reduction") != "mean":
        raise ValueError("canonical layerwise allocation must use mean reduction")
    if Path(str(pearson_layerwise.get("timestep_grouping_path"))).resolve() != paths[
        "pearson_k3_grouping"
    ].resolve():
        raise ValueError("canonical layerwise allocation uses the wrong grouping")
    layerwise_plan = pearson_layerwise.get("target_budget_plan")
    if not isinstance(layerwise_plan, Mapping):
        raise ValueError("canonical layerwise allocation has no target budget plan")
    raw_layer_budgets = layerwise_plan.get("layer_budgets")
    if not isinstance(raw_layer_budgets, Sequence) or len(raw_layer_budgets) != 3:
        raise ValueError("canonical layerwise allocation must contain three phase rows")
    layer_budgets = [[float(value) for value in row] for row in raw_layer_budgets]
    if any(len(row) != 36 for row in layer_budgets):
        raise ValueError("canonical layerwise allocation must contain 36 residual blocks")
    if not all(math.isfinite(value) and value >= 0 for row in layer_budgets for value in row):
        raise ValueError("canonical layerwise budgets must be finite and non-negative")
    canonical_block_budgets = allocations["pearson_fixed_k3"]["block_budgets"]
    for index, (row, block_budget) in enumerate(
        zip(layer_budgets, canonical_block_budgets, strict=True)
    ):
        if not math.isclose(sum(row), block_budget, rel_tol=1e-12, abs_tol=1e-6):
            raise ValueError(
                f"canonical layerwise phase {index} does not sum to its block budget"
            )

    pearson_summary = _load_json(
        paths["pearson_k3_allocation_summary"],
        "pearson_k3_allocation_summary",
    )
    _require_source_hash(
        pearson_summary,
        expected=residual_sha256,
        label="pearson_k3_allocation_summary",
    )
    if pearson_summary.get("student_variants") != [
        "combined_blockwise",
        "combined_layerwise",
    ]:
        raise ValueError("canonical allocation summary does not contain both variants")

    spearman_k = int(groupings["spearman_auto"]["num_blocks"])
    pearson_k = int(groupings["pearson_auto"]["num_blocks"])
    n_eff = [float(value) for value in results.get("n_eff", [])]
    residual_mass_fraction = [
        float(value)
        for value in results.get("positive_delta_mass_fraction_of_source", [])
    ]
    if not n_eff or not residual_mass_fraction:
        raise ValueError("residual results are missing decision metrics")

    return {
        "format": REPORT_FORMAT,
        "status": "passed",
        "scope": {
            "source_group_count": int(results["scope"]["source_group_count"]),
            "residual_filter_count": int(results["scope"]["selected_group_count"]),
            "excluded_fixed_filter_count": int(results["scope"]["excluded_group_count"]),
            "allocation_parameter_budget": allocations["pearson_fixed_k3"][
                "total_residual_parameter_budget"
            ],
        },
        "residual_metrics": {
            "n_eff_min": min(n_eff),
            "n_eff_max": max(n_eff),
            "positive_delta_mass_fraction_of_all_filters_min": min(
                residual_mass_fraction
            ),
            "positive_delta_mass_fraction_of_all_filters_max": max(
                residual_mass_fraction
            ),
            "positive_delta_mass_fraction_of_all_filters_total": float(
                results["total_positive_delta_mass_fraction_of_source"]
            ),
        },
        "groupings": groupings,
        "allocations": allocations,
        "canonical_layerwise_allocation": {
            "num_phases": len(layer_budgets),
            "num_residual_blocks": len(layer_budgets[0]),
            "phase_budget_sums": [sum(row) for row in layer_budgets],
        },
        "canonical_artifacts": {
            key: str(path.relative_to(root))
            for key, path in paths.items()
            if key.startswith("pearson_k3_")
        },
        "decision": {
            "phase_specialization_supported": True,
            "three_phase_structure_supported": True,
            "exact_student_count_optimality_established": False,
            "student_count_policy": "fixed_k3_consistent_with_edm_methodology",
            "canonical_grouping": "pearson_fixed_k3",
            "canonical_num_phases": 3,
            "canonical_allocation": "pearson_fixed_k3",
            "spearman_fixed_k3_role": "boundary_sensitivity",
            "auto_k_role": "uncalibrated_diagnostic_only",
            "auto_k_diagnostic": {
                "spearman_num_phases": spearman_k,
                "pearson_num_phases": pearson_k,
            },
            "reason": (
                "The residual Pearson matrix supports a fixed three-phase design. "
                "K is specified by the requested three-student architecture, as in "
                "the EDM workflow; the analysis optimizes its contiguous boundaries."
            ),
        },
        "provenance": {
            "residual_results_sha256": residual_sha256,
            "input_artifact_sha256": {
                str(path.relative_to(root)): _sha256(path)
                for path in paths.values()
            },
            "source_analysis_basis_sha256": results.get(
                "source_analysis_basis_sha256"
            ),
            "profile_fingerprint_sha256": results.get(
                "profile_fingerprint_sha256"
            ),
        },
        "limitations": [
            "K=3 is a design choice; this profile does not establish that three students outperform K=1 or K=2.",
            "Spearman and penalized auto-K outputs are retained as grouping sensitivities, not as overrides of the fixed Pearson K=3 design.",
            "This is postprocessing of the completed 100-utterance, four-samples-per-timestep profile; no new PFI observations were collected.",
            "No student is instantiated or trained by these dry-run allocations.",
        ],
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--residual-dir",
        default="out_eval_diffwave_sc09/per_filter_exact_timestep_v1/profile_pool100_n4_t200_bins20_seed0/residual_only",
        help="Residual-only profile containing the standard grouping/allocation subdirectories.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: <residual-dir>/decision_report.json).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    residual_dir = Path(args.residual_dir).expanduser().resolve()
    output = (
        residual_dir / "decision_report.json"
        if args.output is None
        else Path(args.output).expanduser().resolve()
    )
    report = build_report(residual_dir)
    _atomic_write_json(output, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "decision": report["decision"],
                "canonical_block_budgets": report["allocations"]["pearson_fixed_k3"][
                    "block_budgets"
                ],
                "spearman_k3_sensitivity_block_budgets": report["allocations"][
                    "spearman_fixed_k3_sensitivity"
                ]["block_budgets"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
