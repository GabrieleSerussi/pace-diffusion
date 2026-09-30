#!/usr/bin/env python3
"""Generate the paper-ready figures for the DiffWave/SC09 appendix.

By default the script renders from a compact, versioned metrics snapshot so
figure generation does not depend on the external experiment volume. Passing
``--refresh-from-profile`` first validates the finalized source hashes and
recomputes that snapshot from the residual-only production artifacts. Neither
mode performs model inference or modifies an evaluation artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS_FORMAT = "diffdist_audio_appendix_metrics_v1"
PHASE_BOUNDARIES = (0, 13, 17, 20)
PHASE_MATH_LABELS = (r"$t=199$--$70$", r"$t=69$--$30$", r"$t=29$--$0$")
PHASE_SHORT_LABELS = ("Early", "Middle", "Late")
PHASE_COLORS = ("#dbeafe", "#fef3c7", "#dcfce7")
OKABE_ITO = ("#0072b2", "#d55e00", "#009e73", "#cc79a7")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load {label} from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _array(
    payload: Mapping[str, Any], key: str, expected_shape: tuple[int, ...]
) -> np.ndarray:
    values = np.asarray(payload.get(key), dtype=np.float64)
    if values.shape != expected_shape or not np.isfinite(values).all():
        raise ValueError(
            f"{key} must be a finite array with shape {expected_shape}; "
            f"found {values.shape}"
        )
    return values


def _off_diagonal_mean(matrix: np.ndarray) -> float:
    if matrix.shape[0] < 2:
        raise ValueError("a phase must contain at least two bins")
    return float(matrix[~np.eye(matrix.shape[0], dtype=bool)].mean())


def _validate_source_inputs(
    residual_results_path: Path,
    allocation_path: Path,
    results: Mapping[str, Any],
    decision: Mapping[str, Any],
    allocation: Mapping[str, Any],
    stability: Mapping[str, Any],
) -> None:
    actual_hash = _sha256(residual_results_path)
    decision_provenance = decision.get("provenance")
    if not isinstance(decision_provenance, Mapping):
        raise ValueError("decision report does not contain provenance")
    expected_hash = decision_provenance.get("residual_results_sha256")
    if actual_hash != expected_hash:
        raise ValueError(
            "decision report is not bound to the supplied residual results: "
            f"expected {expected_hash!r}, found {actual_hash!r}"
        )
    try:
        allocation_relative = allocation_path.resolve().relative_to(
            residual_results_path.parent.resolve()
        ).as_posix()
    except ValueError as exc:
        raise ValueError("canonical allocation must be inside the residual profile") from exc
    canonical_artifacts = decision.get("canonical_artifacts")
    if not isinstance(canonical_artifacts, Mapping) or canonical_artifacts.get(
        "pearson_k3_layerwise_allocation"
    ) != allocation_relative:
        raise ValueError("decision report identifies a different layerwise allocation")
    artifact_hashes = decision_provenance.get("input_artifact_sha256")
    if not isinstance(artifact_hashes, Mapping):
        raise ValueError("decision report does not contain input artifact hashes")
    expected_allocation_hash = artifact_hashes.get(allocation_relative)
    actual_allocation_hash = _sha256(allocation_path)
    if actual_allocation_hash != expected_allocation_hash:
        raise ValueError(
            "decision report is not bound to the supplied layerwise allocation: "
            f"expected {expected_allocation_hash!r}, found {actual_allocation_hash!r}"
        )
    if decision.get("status") != "passed":
        raise ValueError("audio decision report did not pass validation")
    if decision.get("format") != "diffdist_diffwave_per_filter_decision_v2":
        raise ValueError("unexpected audio decision report format")
    if (
        stability.get("passed") is not True
        or stability.get("format") != "diffdist_diffwave_per_filter_stability_v1"
    ):
        raise ValueError("audio stability report did not pass validation")
    if results.get("format") != "diffdist_diffwave_residual_filter_analysis_v1":
        raise ValueError("unexpected residual analysis format")
    if len(results.get("group_names", [])) != 36_864:
        raise ValueError("the canonical residual view must contain 36,864 filters")
    if decision.get("groupings", {}).get("pearson_fixed_k3", {}).get(
        "boundaries"
    ) != list(PHASE_BOUNDARIES):
        raise ValueError("canonical Pearson K=3 boundaries have changed")
    if allocation.get("student_variant") != "combined_layerwise":
        raise ValueError("expected the canonical combined_layerwise allocation")
    allocation_source = allocation.get("source_profile")
    if not isinstance(allocation_source, Mapping) or allocation_source.get(
        "results_sha256"
    ) != actual_hash:
        raise ValueError("layerwise allocation is not bound to the residual results")
    if allocation.get("allocation_group_scope") != "allocatable":
        raise ValueError("layerwise allocation must use allocatable groups")
    if allocation.get("score_reduction") != "mean":
        raise ValueError("layerwise allocation must use duration-neutral mean reduction")
    if allocation.get("timestep_blocks") != [[0, 13], [13, 17], [17, 20]]:
        raise ValueError("layerwise allocation does not use canonical phase boundaries")
    score_sources = allocation.get("score_sources")
    if not isinstance(score_sources, Mapping):
        raise ValueError("layerwise allocation does not contain score sources")
    block_source = score_sources.get("block_capacity_scores")
    layer_source = score_sources.get("layer_capacity_scores")
    if not isinstance(block_source, Mapping) or not isinstance(layer_source, Mapping):
        raise ValueError("layerwise allocation is missing block or layer score metadata")
    expected_block_protocol = {
        "metric": "delta_p_eff_geomean",
        "group_scope": "allocatable",
        "reduction": "mean",
        "reduction_protocol": "duration_neutral_mean",
        "p_eff_source": "recomputed_from_allocatable_groups",
    }
    for key, expected in expected_block_protocol.items():
        if block_source.get(key) != expected:
            raise ValueError(
                f"block score protocol {key} is {block_source.get(key)!r}; "
                f"expected {expected!r}"
            )
    expected_layer_protocol = {
        "score_source": "delta_p_eff_geomean",
        # The source is already the residual-only view, so every remaining
        # group is allocatable and the loader records its effective scope as
        # "all" while retaining the requested allocatable scope separately.
        "requested_group_scope": "allocatable",
        "group_scope": "all",
        "reduction": "mean",
        "reduction_protocol": "duration_neutral_mean",
    }
    for key, expected in expected_layer_protocol.items():
        if layer_source.get(key) != expected:
            raise ValueError(
                f"layer score protocol {key} is {layer_source.get(key)!r}; "
                f"expected {expected!r}"
            )
    expected_stages = [f"residual_block_{index:02d}" for index in range(36)]
    structural_aggregation = layer_source.get("structural_aggregation")
    if not isinstance(structural_aggregation, Mapping) or structural_aggregation.get(
        "applied"
    ) is not True:
        raise ValueError("layer scores must use structural residual-block aggregation")
    allocation_stages = structural_aggregation.get("output_stage_keys")
    if allocation_stages != expected_stages:
        raise ValueError("layerwise allocation residual-block ordering has changed")
    plan = allocation.get("target_budget_plan")
    if not isinstance(plan, Mapping):
        raise ValueError("layerwise allocation does not contain a target budget plan")
    if plan.get("layer_names") != expected_stages:
        raise ValueError("target budget layer ordering has changed")
    phase_budgets = np.asarray(plan.get("block_budgets"), dtype=np.float64)
    layer_budgets = np.asarray(plan.get("layer_budgets"), dtype=np.float64)
    total_budget = float(plan.get("total_student_system_budget", 0.0))
    if (
        phase_budgets.shape != (3,)
        or layer_budgets.shape != (3, 36)
        or not np.isfinite(phase_budgets).all()
        or not np.isfinite(layer_budgets).all()
        or not np.isfinite(total_budget)
        or np.any(phase_budgets <= 0)
        or np.any(layer_budgets < 0)
        or total_budget <= 0
    ):
        raise ValueError("canonical allocation budgets have invalid shape or values")
    if not np.allclose(
        layer_budgets.sum(axis=1), phase_budgets, rtol=1e-12, atol=1e-5
    ) or not np.isclose(phase_budgets.sum(), total_budget, rtol=1e-12, atol=1e-5):
        raise ValueError("canonical layerwise budgets do not preserve phase/system totals")
    decision_allocation = decision.get("allocations", {}).get("pearson_fixed_k3", {})
    expected_budgets = np.asarray(
        decision_allocation.get("block_budgets"), dtype=np.float64
    )
    expected_total = float(
        decision.get("scope", {}).get("allocation_parameter_budget", 0.0)
    )
    if expected_budgets.shape != (3,) or not np.allclose(
        phase_budgets, expected_budgets, rtol=1e-12, atol=1e-5
    ) or not np.isclose(total_budget, expected_total, rtol=1e-12, atol=1e-5):
        raise ValueError("layerwise budgets disagree with the canonical decision report")
    stability_inputs = stability.get("inputs")
    expected_seeds = [0, 1, 2]
    if not isinstance(stability_inputs, Sequence) or len(stability_inputs) != 3:
        raise ValueError("stability audit must bind exactly three input profiles")
    for expected_seed, record in zip(expected_seeds, stability_inputs, strict=True):
        if not isinstance(record, Mapping) or (
            record.get("seed"), record.get("pfi_seed")
        ) != (expected_seed, expected_seed):
            raise ValueError("stability audit seed/PFI-seed settings have changed")
    pairs = stability.get("pairs")
    expected_pairs = [
        ("seed0", "seed1"),
        ("seed0", "seed2"),
        ("seed1", "seed2"),
    ]
    actual_pairs = (
        [(pair.get("first"), pair.get("second")) for pair in pairs]
        if isinstance(pairs, Sequence)
        and all(isinstance(pair, Mapping) for pair in pairs)
        else []
    )
    if (
        stability.get("group_count") != 256
        or stability.get("num_bins") != 20
        or actual_pairs != expected_pairs
    ):
        raise ValueError("stability audit must contain three pairs over 256 filters")


def _phase_shares(positive: np.ndarray, selector: np.ndarray) -> list[float]:
    shares: list[float] = []
    for start, end in zip(
        PHASE_BOUNDARIES[:-1], PHASE_BOUNDARIES[1:], strict=True
    ):
        phase = positive[:, start:end]
        denominator = float(phase.sum())
        if denominator <= 0:
            raise ValueError("each phase must contain positive residual PFI mass")
        shares.append(float(phase[selector].sum() / denominator))
    return shares


def _stability_catalog_composition(
    stability_path: Path,
    stability: Mapping[str, Any],
) -> dict[str, int]:
    """Validate the report's three source profiles and count catalog scopes."""

    inputs = stability.get("inputs")
    if not isinstance(inputs, Sequence) or len(inputs) != 3:
        raise ValueError("stability report must bind exactly three source profiles")
    reference_names: list[str] | None = None
    reference_stages: list[str] | None = None
    for seed, record in enumerate(inputs):
        if not isinstance(record, Mapping):
            raise ValueError("stability input records must be mappings")
        source_path = stability_path.parent / f"seed{seed}" / "results.json"
        reported_path = Path(str(record.get("path", ""))).expanduser().resolve()
        if reported_path != source_path.resolve():
            raise ValueError("stability report input paths have changed")
        if _sha256(source_path) != record.get("sha256"):
            raise ValueError("stability report is not bound to its source profiles")
        payload = _load_json(source_path, f"stability seed {seed}")
        config = payload.get("config")
        if not isinstance(config, Mapping) or (
            config.get("seed"),
            config.get("pfi_seed"),
            config.get("max_groups"),
            config.get("max_groups_mode"),
        ) != (seed, seed, 256, "stratified"):
            raise ValueError("stability source profile protocol has changed")
        names = [str(value) for value in payload.get("group_names", [])]
        stage_mapping = payload.get("group_stage_keys")
        if len(names) != 256 or not isinstance(stage_mapping, Mapping):
            raise ValueError("stability source profile has invalid group metadata")
        stages = [str(stage_mapping[name]) for name in names]
        if reference_names is None:
            reference_names, reference_stages = names, stages
        elif names != reference_names or stages != reference_stages:
            raise ValueError("stability source profiles use different filter subsets")
    assert reference_stages is not None
    residual_count = sum(stage.startswith("residual_block_") for stage in reference_stages)
    fixed_count = len(reference_stages) - residual_count
    if (residual_count, fixed_count) != (251, 5):
        raise ValueError("canonical stability subset composition has changed")
    return {
        "residual_filters": residual_count,
        "fixed_stem_head_filters": fixed_count,
    }


def _extract_metrics(
    results: Mapping[str, Any],
    decision: Mapping[str, Any],
    allocation: Mapping[str, Any],
    stability: Mapping[str, Any],
    stability_composition: Mapping[str, int],
) -> dict[str, Any]:
    labels = [str(value) for value in results["timestep_bin_labels"]]
    if len(labels) != 20:
        raise ValueError("the audio appendix expects exactly 20 timestep bins")
    group_names = [str(value) for value in results["group_names"]]
    positive = _array(results, "delta_stack", (len(group_names), 20))
    correlation = _array(results, "C_timesteps_pearson", (20, 20))

    module_mapping = results.get("group_module_paths")
    stage_mapping = results.get("group_stage_keys")
    if not isinstance(module_mapping, Mapping) or not isinstance(stage_mapping, Mapping):
        raise ValueError("group module and stage metadata must be name-keyed mappings")
    modules = np.asarray([str(module_mapping[name]) for name in group_names])
    stages = np.asarray([str(stage_mapping[name]) for name in group_names])
    selectors = {
        "dilated_convolution": np.char.find(modules, ".dilated_conv_layer.conv") >= 0,
        "skip_projection": np.char.find(modules, ".skip_conv") >= 0,
        "residual_projection": np.char.find(modules, ".res_conv") >= 0,
        "residual_block_00": stages == "residual_block_00",
    }
    if not np.all(
        selectors["dilated_convolution"]
        | selectors["skip_projection"]
        | selectors["residual_projection"]
    ):
        raise ValueError("residual filters contain an unexpected module role")

    phase_similarity = np.empty((3, 3), dtype=np.float64)
    for first in range(3):
        first_slice = slice(PHASE_BOUNDARIES[first], PHASE_BOUNDARIES[first + 1])
        for second in range(3):
            second_slice = slice(
                PHASE_BOUNDARIES[second], PHASE_BOUNDARIES[second + 1]
            )
            submatrix = correlation[first_slice, second_slice]
            phase_similarity[first, second] = (
                _off_diagonal_mean(submatrix)
                if first == second
                else float(submatrix.mean())
            )

    plan = allocation["target_budget_plan"]
    phase_budgets = np.asarray(plan["block_budgets"], dtype=np.float64)
    layer_budgets = np.asarray(plan["layer_budgets"], dtype=np.float64)
    total_budget = float(plan["total_student_system_budget"])
    if phase_budgets.shape != (3,) or layer_budgets.shape != (3, 36):
        raise ValueError("canonical allocation must contain 3 phases by 36 blocks")

    stability_pairs = []
    for pair in stability["pairs"]:
        stability_pairs.append(
            {
                "label": f"{pair['first']}--{pair['second']}",
                "filter_importance_spearman": pair["filter_importance_spearman"],
                "stage_signed_spearman": pair["aggregates"]["stage"][
                    "flattened_signed_spearman"
                ],
                "filter_signed_spearman": pair["signed_delta_flattened_spearman"],
                "filter_sign_agreement": pair["signed_delta_sign_agreement"],
                "top_quartile_jaccard": pair["top_quartile_filter_jaccard"],
            }
        )

    phase_shares = {
        key: _phase_shares(positive, selector)
        for key, selector in selectors.items()
    }
    phase_shares["residual_block_00_overall"] = float(
        positive[selectors["residual_block_00"]].sum() / positive.sum()
    )
    groupings = decision["groupings"]

    return {
        "format": METRICS_FORMAT,
        "provenance": {
            "residual_results_sha256": decision["provenance"][
                "residual_results_sha256"
            ],
            "profile_fingerprint_sha256": decision["provenance"][
                "profile_fingerprint_sha256"
            ],
            "teacher_checkpoint_sha256": results["profile_fingerprint"]["teacher"][
                "checkpoint_sha256"
            ],
            "decision_report_format": decision["format"],
            "stability_report_format": stability["format"],
        },
        "scope": {
            "source_filter_count": decision["scope"]["source_group_count"],
            "residual_filter_count": decision["scope"]["residual_filter_count"],
            "excluded_fixed_filter_count": decision["scope"][
                "excluded_fixed_filter_count"
            ],
            "allocatable_residual_parameter_budget": decision["scope"][
                "allocation_parameter_budget"
            ],
        },
        "timestep_bin_labels": labels,
        "n_eff": results["n_eff"],
        "n_eff_fraction": results["n_eff_fraction"],
        "residual_positive_mass_fraction": results[
            "positive_delta_mass_fraction_of_source"
        ],
        "total_residual_positive_mass_fraction": results[
            "total_positive_delta_mass_fraction_of_source"
        ],
        "phase_boundaries": list(PHASE_BOUNDARIES),
        "phase_labels": ["t=199--70", "t=69--30", "t=29--0"],
        "timestep_correlation_pearson": correlation.tolist(),
        "phase_similarity": phase_similarity.tolist(),
        "phase_similarity_diagonal": (
            "mean_off_diagonal_bin_pair_correlation_within_phase"
        ),
        "phase_similarity_off_diagonal": (
            "mean_bin_pair_correlation_across_phases"
        ),
        "phase_raw_positive_mass_shares": phase_shares,
        "grouping_diagnostics": {
            "pearson_auto": {
                "num_phases": groupings["pearson_auto"]["num_blocks"],
                "boundaries": groupings["pearson_auto"]["boundaries"],
                "cross_block_lambda": groupings["pearson_auto"][
                    "cross_block_lambda"
                ],
            },
            "spearman_auto": {
                "num_phases": groupings["spearman_auto"]["num_blocks"],
                "boundaries": groupings["spearman_auto"]["boundaries"],
                "cross_block_lambda": groupings["spearman_auto"][
                    "cross_block_lambda"
                ],
            },
        },
        "allocation": {
            "total_residual_parameter_budget": total_budget,
            "phase_budgets": phase_budgets.tolist(),
            "phase_budget_fractions": (phase_budgets / total_budget).tolist(),
            "residual_block_00_within_phase_budget_fractions": (
                layer_budgets[:, 0] / phase_budgets
            ).tolist(),
            "metric": "sqrt(positive_delta_mass * residual_p_eff)",
            "phase_reduction": "duration_neutral_mean",
        },
        "stability": {
            "group_count": stability["group_count"],
            "num_bins": stability["num_bins"],
            "catalog_composition": dict(stability_composition),
            "pairs": stability_pairs,
        },
    }


def _validate_metrics(metrics: Mapping[str, Any]) -> None:
    if metrics.get("format") != METRICS_FORMAT:
        raise ValueError(f"unexpected appendix metrics format: {metrics.get('format')!r}")
    provenance = metrics.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("appendix metrics are missing provenance")
    for key in (
        "residual_results_sha256",
        "profile_fingerprint_sha256",
        "teacher_checkpoint_sha256",
    ):
        if not _is_sha256(provenance.get(key)):
            raise ValueError(f"appendix provenance {key} is not a SHA-256 digest")
    if (
        provenance.get("decision_report_format")
        != "diffdist_diffwave_per_filter_decision_v2"
        or provenance.get("stability_report_format")
        != "diffdist_diffwave_per_filter_stability_v1"
    ):
        raise ValueError("appendix provenance report formats have changed")
    scope = metrics.get("scope")
    expected_scope = {
        "source_filter_count": 37_377,
        "residual_filter_count": 36_864,
        "excluded_fixed_filter_count": 513,
        "allocatable_residual_parameter_budget": 18_948_096,
    }
    if not isinstance(scope, Mapping) or any(
        scope.get(key) != value for key, value in expected_scope.items()
    ):
        raise ValueError("appendix scope does not match the canonical audio profile")
    if metrics.get("phase_boundaries") != list(PHASE_BOUNDARIES):
        raise ValueError("appendix metrics do not use the canonical phase boundaries")
    labels = metrics.get("timestep_bin_labels")
    expected_labels = [
        f"{199 - 10 * index}-{max(0, 190 - 10 * index)}"
        for index in range(20)
    ]
    if not isinstance(labels, Sequence) or list(labels) != expected_labels:
        raise ValueError("appendix metrics must contain the canonical 20 timestep bins")
    n_eff = _array(metrics, "n_eff", (20,))
    n_eff_fraction = _array(metrics, "n_eff_fraction", (20,))
    if np.any(n_eff <= 0) or np.any(n_eff > expected_scope["residual_filter_count"]):
        raise ValueError("effective filter support lies outside the residual catalog")
    if not np.allclose(
        n_eff_fraction,
        n_eff / expected_scope["residual_filter_count"],
        rtol=1e-12,
        atol=1e-15,
    ):
        raise ValueError("n_eff_fraction is inconsistent with n_eff")
    residual_mass = _array(metrics, "residual_positive_mass_fraction", (20,))
    total_residual_mass = float(
        metrics.get("total_residual_positive_mass_fraction", float("nan"))
    )
    if (
        np.any((residual_mass < 0) | (residual_mass > 1))
        or not np.isfinite(total_residual_mass)
        or not 0 <= total_residual_mass <= 1
    ):
        raise ValueError("residual positive-mass fractions must lie in [0, 1]")
    timestep_correlation = _array(
        metrics, "timestep_correlation_pearson", (20, 20)
    )
    if (
        np.any((timestep_correlation < -1) | (timestep_correlation > 1))
        or not np.allclose(
            timestep_correlation,
            timestep_correlation.T,
            rtol=0.0,
            atol=1e-12,
        )
        or not np.allclose(
            np.diag(timestep_correlation), 1.0, rtol=0.0, atol=1e-12
        )
    ):
        raise ValueError("timestep correlation must be a symmetric correlation matrix")
    similarity = _array(metrics, "phase_similarity", (3, 3))
    if np.any((similarity < -1) | (similarity > 1)) or not np.allclose(
        similarity, similarity.T, rtol=0.0, atol=1e-12
    ):
        raise ValueError("phase similarity must be symmetric")
    if (
        metrics.get("phase_similarity_diagonal")
        != "mean_off_diagonal_bin_pair_correlation_within_phase"
        or metrics.get("phase_similarity_off_diagonal")
        != "mean_bin_pair_correlation_across_phases"
    ):
        raise ValueError("phase-similarity aggregation labels have changed")
    shares = metrics.get("phase_raw_positive_mass_shares")
    if not isinstance(shares, Mapping):
        raise ValueError("appendix metrics are missing phase role shares")
    role_shares = np.vstack(
        [
            np.asarray(shares[key], dtype=np.float64)
            for key in (
                "dilated_convolution",
                "skip_projection",
                "residual_projection",
            )
        ]
    )
    if (
        role_shares.shape != (3, 3)
        or not np.isfinite(role_shares).all()
        or np.any((role_shares < 0) | (role_shares > 1))
        or not np.allclose(role_shares.sum(axis=0), 1.0, rtol=0.0, atol=5e-4)
    ):
        raise ValueError("phase convolution-role shares must sum to one")
    block_zero = np.asarray(shares.get("residual_block_00"), dtype=np.float64)
    block_zero_overall = float(
        shares.get("residual_block_00_overall", float("nan"))
    )
    if (
        block_zero.shape != (3,)
        or not np.isfinite(block_zero).all()
        or np.any((block_zero < 0) | (block_zero > 1))
        or not np.isfinite(block_zero_overall)
        or not 0 <= block_zero_overall <= 1
    ):
        raise ValueError("block-0 raw mass shares must be three fractions")
    grouping = metrics.get("grouping_diagnostics")
    expected_grouping = {
        "pearson_auto": (2, [0, 17, 20], 0.02),
        "spearman_auto": (1, [0, 20], 0.02),
    }
    if not isinstance(grouping, Mapping):
        raise ValueError("appendix metrics are missing automatic-K diagnostics")
    for key, (num_phases, boundaries, penalty) in expected_grouping.items():
        record = grouping.get(key)
        if not isinstance(record, Mapping) or (
            record.get("num_phases") != num_phases
            or record.get("boundaries") != boundaries
            or not np.isclose(
                float(record.get("cross_block_lambda", float("nan"))), penalty
            )
        ):
            raise ValueError(f"{key} diagnostic no longer matches the appendix")
    allocation = metrics.get("allocation")
    if not isinstance(allocation, Mapping):
        raise ValueError("appendix metrics are missing allocation data")
    budgets = np.asarray(allocation.get("phase_budgets"), dtype=np.float64)
    fractions = np.asarray(allocation.get("phase_budget_fractions"), dtype=np.float64)
    total = float(allocation.get("total_residual_parameter_budget", 0.0))
    if (
        budgets.shape != (3,)
        or fractions.shape != (3,)
        or not np.isfinite(budgets).all()
        or not np.isfinite(fractions).all()
        or np.any(budgets <= 0)
        or np.any((fractions <= 0) | (fractions > 1))
    ):
        raise ValueError("allocation must contain three phase budgets")
    if (
        allocation.get("metric")
        != "sqrt(positive_delta_mass * residual_p_eff)"
        or allocation.get("phase_reduction") != "duration_neutral_mean"
    ):
        raise ValueError("allocation metric or phase reduction has changed")
    if not np.isclose(
        total,
        expected_scope["allocatable_residual_parameter_budget"],
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("allocation total does not match the residual budget")
    if not np.isclose(budgets.sum(), total, rtol=1e-12, atol=1e-5):
        raise ValueError("phase budgets do not sum to the residual total")
    if not np.allclose(fractions, budgets / total, rtol=1e-12, atol=1e-12):
        raise ValueError("phase budget fractions do not match absolute budgets")
    block_fraction = np.asarray(
        allocation.get("residual_block_00_within_phase_budget_fractions"),
        dtype=np.float64,
    )
    if block_fraction.shape != (3,) or np.any(
        (block_fraction < 0) | (block_fraction > 1)
    ):
        raise ValueError("block-0 budget shares must be three fractions")
    stability = metrics.get("stability")
    expected_composition = {
        "residual_filters": 251,
        "fixed_stem_head_filters": 5,
    }
    if (
        not isinstance(stability, Mapping)
        or stability.get("group_count") != 256
        or stability.get("num_bins") != 20
        or stability.get("catalog_composition") != expected_composition
    ):
        raise ValueError("appendix metrics must contain three stability pairs")
    pairs = stability.get("pairs")
    expected_pair_labels = ["seed0--seed1", "seed0--seed2", "seed1--seed2"]
    if not isinstance(pairs, Sequence) or [
        pair.get("label") if isinstance(pair, Mapping) else None for pair in pairs
    ] != expected_pair_labels:
        raise ValueError("appendix stability pair identities have changed")
    signed_metrics = (
        "filter_importance_spearman",
        "stage_signed_spearman",
        "filter_signed_spearman",
    )
    fraction_metrics = ("filter_sign_agreement", "top_quartile_jaccard")
    for pair in pairs:
        assert isinstance(pair, Mapping)
        for key in signed_metrics:
            value = float(pair.get(key, float("nan")))
            if not np.isfinite(value) or not -1 <= value <= 1:
                raise ValueError(f"stability metric {key} must lie in [-1, 1]")
        for key in fraction_metrics:
            value = float(pair.get(key, float("nan")))
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"stability metric {key} must lie in [0, 1]")


def _phase_background(ax: plt.Axes) -> None:
    for start, end, color in zip(
        PHASE_BOUNDARIES[:-1], PHASE_BOUNDARIES[1:], PHASE_COLORS, strict=True
    ):
        ax.axvspan(start - 0.5, end - 0.5, color=color, alpha=0.62, linewidth=0)
    for boundary in PHASE_BOUNDARIES[1:-1]:
        ax.axvline(boundary - 0.5, color="0.25", linewidth=0.8, linestyle="--")


def _save_figure(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    # A raster preview is useful in environments without a local PDF viewer.
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _phase_tick_labels() -> list[str]:
    labels = []
    for name, phase in zip(PHASE_SHORT_LABELS, PHASE_MATH_LABELS, strict=True):
        labels.append(f"{name}\n{phase}")
    return labels


def plot_sensitivity(metrics: Mapping[str, Any], output: Path) -> None:
    labels = [str(value) for value in metrics["timestep_bin_labels"]]
    x = np.arange(20)
    n_eff_fraction = 100.0 * _array(metrics, "n_eff_fraction", (20,))
    residual_mass_fraction = 100.0 * _array(
        metrics, "residual_positive_mass_fraction", (20,)
    )
    shares = metrics["phase_raw_positive_mass_shares"]
    role_values = 100.0 * np.vstack(
        [
            np.asarray(shares[key], dtype=np.float64)
            for key in (
                "dilated_convolution",
                "skip_projection",
                "residual_projection",
            )
        ]
    )
    block_zero = 100.0 * np.asarray(shares["residual_block_00"], dtype=np.float64)

    fig, (ax_curve, ax_role) = plt.subplots(
        1,
        2,
        figsize=(5.5, 2.65),
        gridspec_kw={"width_ratios": (1.48, 0.92), "wspace": 0.38},
    )
    _phase_background(ax_curve)
    ax_curve.plot(
        x,
        n_eff_fraction,
        color=OKABE_ITO[0],
        marker="o",
        markersize=2.8,
        linewidth=1.3,
        label=r"$N_{\mathrm{eff}}/36{,}864$",
    )
    ax_curve.plot(
        x,
        residual_mass_fraction,
        color=OKABE_ITO[1],
        marker="s",
        markersize=2.6,
        linewidth=1.3,
        label="Residual share of all positive PFI",
    )
    ticks = np.asarray([0, 3, 6, 9, 12, 15, 19])
    ax_curve.set_xlim(-0.5, 19.5)
    ax_curve.set_ylim(0.0, 12.2)
    ax_curve.set_xticks(ticks, [labels[index] for index in ticks])
    ax_curve.tick_params(axis="x", rotation=42)
    ax_curve.set_xlabel(r"Diffusion timestep bin (high noise $\rightarrow$ low noise)")
    ax_curve.set_ylabel("Percentage (%)")
    ax_curve.set_title("(a) Support and allocatable mass", loc="left")
    ax_curve.grid(axis="y", color="white", linewidth=0.8)
    ax_curve.legend(loc="upper left", frameon=False, fontsize=6.6)

    bar_x = np.arange(3)
    bottom = np.zeros(3, dtype=np.float64)
    role_labels = ("Dilated", "Skip", "Residual")
    for values, label, color in zip(
        role_values, role_labels, OKABE_ITO[:3], strict=True
    ):
        ax_role.bar(
            bar_x,
            values,
            bottom=bottom,
            width=0.68,
            color=color,
            label=label,
            linewidth=0,
        )
        bottom += values
    for index, value in enumerate(block_zero):
        ax_role.text(
            index,
            102.0,
            f"block 0\n{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=5.9,
            linespacing=0.9,
        )
    ax_role.set_ylim(0.0, 116.0)
    ax_role.set_yticks(np.arange(0, 101, 20))
    # The phase ranges are already encoded by panel (a)'s shaded boundaries;
    # short labels keep this narrower panel legible at the ICLR text width.
    ax_role.set_xticks(bar_x, PHASE_SHORT_LABELS)
    ax_role.set_ylabel("Share of residual positive PFI (%)")
    ax_role.set_title("(b) Pathway shift", loc="left")
    ax_role.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.31),
        ncol=3,
        frameon=False,
        fontsize=6.3,
        columnspacing=0.8,
        handlelength=1.5,
    )
    ax_role.grid(axis="y", color="0.9", linewidth=0.7)

    _save_figure(fig, output)


def plot_phase_and_allocation(metrics: Mapping[str, Any], output: Path) -> None:
    correlation = _array(metrics, "timestep_correlation_pearson", (20, 20))
    labels = [str(value) for value in metrics["timestep_bin_labels"]]
    allocation = metrics["allocation"]
    phase_fractions = 100.0 * np.asarray(
        allocation["phase_budget_fractions"], dtype=np.float64
    )

    fig, (ax_corr, ax_budget) = plt.subplots(
        1,
        2,
        figsize=(5.5, 2.7),
        gridspec_kw={"width_ratios": (1.25, 1.0), "wspace": 0.52},
    )
    corr_image = ax_corr.imshow(
        correlation,
        origin="upper",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        rasterized=True,
    )
    for boundary in PHASE_BOUNDARIES[1:-1]:
        position = boundary - 0.5
        ax_corr.axvline(position, color="white", linewidth=1.0, linestyle="--")
        ax_corr.axhline(position, color="white", linewidth=1.0, linestyle="--")
    ticks = np.asarray([0, 4, 8, 12, 16, 19])
    tick_labels = [labels[index] for index in ticks]
    ax_corr.set_xticks(ticks, tick_labels, rotation=48, ha="right")
    ax_corr.set_yticks(ticks, tick_labels)
    ax_corr.set_xlabel("Timestep bin")
    ax_corr.set_ylabel("Timestep bin")
    ax_corr.set_title("(a) Residual-filter PFI correlation", loc="left")
    colorbar = fig.colorbar(corr_image, ax=ax_corr, fraction=0.046, pad=0.03)
    colorbar.set_label(r"Pearson $\rho$", fontsize=6.7)
    colorbar.ax.tick_params(labelsize=5.8)

    bar_x = np.arange(3)
    bars = ax_budget.bar(
        bar_x,
        phase_fractions,
        width=0.62,
        color=OKABE_ITO[:3],
        linewidth=0,
    )
    for bar, fraction in zip(bars, phase_fractions, strict=True):
        ax_budget.text(
            bar.get_x() + bar.get_width() / 2,
            fraction + 1.0,
            f"{fraction:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7.0,
        )
    ax_budget.set_ylim(0.0, 52.0)
    ax_budget.set_xticks(bar_x, _phase_tick_labels())
    ax_budget.set_ylabel("Relative capacity demand (%)")
    ax_budget.set_title(r"(b) Demand for fixed $K=3$", loc="left")
    ax_budget.grid(axis="y", color="0.9", linewidth=0.7)
    ax_budget.set_axisbelow(True)

    _save_figure(fig, output)


def plot_stability(metrics: Mapping[str, Any], output: Path) -> None:
    pairs = metrics["stability"]["pairs"]
    pair_labels = [str(pair["label"]) for pair in pairs]
    metric_specs: Sequence[tuple[str, str]] = (
        ("Overall filter\n" r"importance $\rho$", "filter_importance_spearman"),
        ("Stage-level signed\n" r"PFI $\rho$", "stage_signed_spearman"),
        ("Filter-level signed\n" r"PFI $\rho$", "filter_signed_spearman"),
        ("Filter-level sign\nagreement", "filter_sign_agreement"),
        ("Top-quartile\nJaccard", "top_quartile_jaccard"),
    )
    values = np.asarray(
        [[float(pair[key]) for _, key in metric_specs] for pair in pairs],
        dtype=np.float64,
    )
    if values.shape != (3, 5) or not np.isfinite(values).all():
        raise ValueError("stability metrics must form a finite 3-by-5 matrix")

    fig, ax = plt.subplots(figsize=(5.5, 2.25))
    x = np.arange(len(metric_specs))
    markers = ("o", "s", "^")
    for pair_index, (label, marker, color) in enumerate(
        zip(pair_labels, markers, OKABE_ITO[:3], strict=True)
    ):
        ax.plot(
            x,
            values[pair_index],
            color=color,
            marker=marker,
            linewidth=1.1,
            markersize=4.2,
            label=label,
        )
    ax.set_xlim(-0.25, len(metric_specs) - 0.75)
    ax.set_ylim(0.35, 0.95)
    ax.set_xticks(x, [label for label, _ in metric_specs])
    ax.set_ylabel("Agreement across seed settings")
    ax.grid(axis="y", color="0.88", linewidth=0.7)
    ax.legend(ncol=3, loc="upper center", frameon=False, fontsize=7)
    ax.set_title("Three-seed audit on the same stratified 256-filter subset", loc="left")

    _save_figure(fig, output)


def _summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    labels = [str(value) for value in metrics["timestep_bin_labels"]]
    n_eff = _array(metrics, "n_eff", (20,))
    n_eff_fraction = _array(metrics, "n_eff_fraction", (20,))
    residual_fraction = _array(metrics, "residual_positive_mass_fraction", (20,))
    stability_pairs = metrics["stability"]["pairs"]

    def bounds(key: str) -> list[float]:
        values = [float(pair[key]) for pair in stability_pairs]
        return [min(values), max(values)]

    return {
        "metrics_format": metrics["format"],
        "source_results_sha256": metrics["provenance"]["residual_results_sha256"],
        "n_eff": {
            "min": float(n_eff.min()),
            "min_bin": labels[int(n_eff.argmin())],
            "max": float(n_eff.max()),
            "max_bin": labels[int(n_eff.argmax())],
            "fraction_range": [
                float(n_eff_fraction.min()),
                float(n_eff_fraction.max()),
            ],
        },
        "residual_positive_mass_fraction": {
            "per_bin_range": [
                float(residual_fraction.min()),
                float(residual_fraction.max()),
            ],
            "overall": float(metrics["total_residual_positive_mass_fraction"]),
        },
        "phase_similarity": metrics["phase_similarity"],
        "phase_budget_fractions": metrics["allocation"]["phase_budget_fractions"],
        "stability_ranges": {
            "filter_importance_spearman": bounds("filter_importance_spearman"),
            "stage_signed_spearman": bounds("stage_signed_spearman"),
            "filter_signed_spearman": bounds("filter_signed_spearman"),
            "filter_sign_agreement": bounds("filter_sign_agreement"),
            "top_quartile_jaccard": bounds("top_quartile_jaccard"),
        },
    }


def _refresh_from_profile(args: argparse.Namespace, metrics_path: Path) -> None:
    residual_dir = Path(args.residual_dir).expanduser().resolve()
    results_path = residual_dir / "results.json"
    decision_path = residual_dir / "decision_report.json"
    allocation_path = (
        residual_dir
        / "allocation_geomean_mean_pearson_k3_min2"
        / "combined_layerwise.json"
    )
    stability_path = (
        Path(args.stability_report).expanduser().resolve()
        if args.stability_report
        else residual_dir.parent.parent / "stability" / "stability_report.json"
    )
    results = _load_json(results_path, "residual results")
    decision = _load_json(decision_path, "decision report")
    allocation = _load_json(allocation_path, "canonical layerwise allocation")
    stability = _load_json(stability_path, "stability report")
    _validate_source_inputs(
        results_path,
        allocation_path,
        results,
        decision,
        allocation,
        stability,
    )
    stability_composition = _stability_catalog_composition(
        stability_path,
        stability,
    )
    metrics = _extract_metrics(
        results,
        decision,
        allocation,
        stability,
        stability_composition,
    )
    _validate_metrics(metrics)
    _atomic_write_json(metrics_path, metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        default="artifacts/audio/audio_appendix_metrics.json",
        help="portable, versioned metrics snapshot (default: the released Appendix B snapshot)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/figures/audio",
        help="directory for publication PDFs and raster previews",
    )
    parser.add_argument(
        "--refresh-from-profile",
        action="store_true",
        help="validate the production artifacts and refresh --metrics before plotting",
    )
    parser.add_argument(
        "--residual-dir",
        default="out_eval_diffwave_sc09/per_filter_exact_timestep_v1/profile_pool100_n4_t200_bins20_seed0/residual_only",
        help="finalized residual-only profile directory used during refresh",
    )
    parser.add_argument(
        "--stability-report",
        default=None,
        help="stability report used during refresh (inferred if omitted)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics).expanduser().resolve()
    if args.refresh_from_profile:
        _refresh_from_profile(args, metrics_path)
    metrics = _load_json(metrics_path, "audio appendix metrics")
    _validate_metrics(metrics)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.2,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.7,
            "legend.fontsize": 6.8,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.format": "pdf",
        }
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    plot_phase_and_allocation(metrics, output_dir / "audio_phases.pdf")
    print(json.dumps(_summary(metrics), indent=2))


if __name__ == "__main__":
    main()
