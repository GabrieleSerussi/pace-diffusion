#!/usr/bin/env python3
"""Dry-run adaptive-capacity budget allocation from a config or out_eval folder.

This is the U-Net allocator of the paper (Section 3.4).  The defaults are the
paper settings: timestep demand ``q_t = sqrt(sum_g Delta_{g,t} * p_eff_t)``
(``--allocation-metric delta_p_eff_geomean``), summed within each phase
(``--score-reduction sum``), layer scores ``r_{b,g}`` from the same geometric
rule (``--layer-score-source delta_p_eff_geomean``), and the four paper variants
``global``, ``uniform_blockwise``, ``combined_blockwise`` (Phase-aware
blockwise) and ``combined_layerwise`` (Phase-aware layerwise).

The profile is ``<eval-output-dir>/results.json`` (or ``results.json.gz``), or
any file passed with ``--results-json`` (for example a released
``artifacts/profiles/*.json.gz``).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pace.capacity_allocation import (
    CapacityAllocationConfig,
    DEFAULT_SHUFFLE_SEED,
    DEFAULT_STUDENT_VARIANTS,
    EvaluationOutputLayout,
    PAPER_ALLOCATION_METRIC,
    PAPER_LAYER_SCORE_SOURCE,
    PAPER_SCORE_REDUCTION,
    SUPPORTED_ALLOCATION_GROUP_SCOPES,
    SUPPORTED_SCORE_REDUCTIONS,
    SUPPORTED_STUDENT_VARIANTS,
    compute_original_model_budget,
    describe_group_allocation_scope,
    discover_evaluation_output,
    geometric_score_reduction_protocol,
    load_block_capacity_scores_from_results,
    load_capacity_scores,
    load_layer_capacity_scores_with_metadata_from_results,
    load_timestep_blocks_from_grouping,
    make_variant_capacity_budgets,
    validate_total_stored_budget,
)
from pace.profile_provenance import (
    provenance_from_artifact,
    provenance_from_results,
    validate_matching_source_profile,
)
from pace.filter_sampling import expansion_weighting_summary
from pace.jsonio import load_json


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text()
    if config_path.suffix.lower() == ".json":
        return json.loads(text)
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("PyYAML is required to load YAML config files") from exc
        payload = yaml.safe_load(text)
        if not isinstance(payload, dict):
            raise ValueError(f"Config file must contain a mapping, got {type(payload).__name__}")
        return payload
    raise ValueError(f"Unsupported config file format: {config_path}")


def resolve_path(base_dir: Path, maybe_path: str | None) -> str | None:
    if maybe_path is None:
        return None
    path = Path(maybe_path)
    if path.is_absolute():
        return str(path)
    base_candidate = (base_dir / path).resolve()
    if base_candidate.exists():
        return str(base_candidate)
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return str(cwd_candidate)
    return str(base_candidate)


def missing_results_message(layout: EvaluationOutputLayout | None, purpose: str) -> str:
    message = f"{purpose} requires results.json or an explicit score path"
    if layout is not None and layout.checkpoint_paths:
        message += (
            f"; found {len(layout.checkpoint_paths)} checkpoint dump(s) in {layout.output_dir}, "
            "but results.json is required to derive capacity scores"
        )
    return message


def load_scores_if_needed(
    config: CapacityAllocationConfig,
    base_dir: Path,
    timestep_blocks,
    layout: EvaluationOutputLayout | None,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    block_scores = None
    layer_scores = None
    score_sources: dict[str, Any] = {}
    if config.student_variant in {
        "blockwise_capacity",
        "combined_blockwise",
        "layerwise_capacity",
        "combined_layerwise",
        "reversed_layerwise_capacity",
        "shuffled_capacity",
    }:
        if config.block_capacity_scores_path is not None:
            block_path = resolve_path(base_dir, config.block_capacity_scores_path)
            block_scores = load_capacity_scores(block_path)
            score_sources["block_capacity_scores"] = {
                "kind": "explicit_path",
                "path": block_path,
                "requested_group_scope": config.allocation_group_scope,
                "group_scope": "precomputed",
            }
        elif layout is not None and layout.results_json_path is not None:
            effective_reduction = config.score_reduction
            if config.allocation_metric == "delta_p_eff_geomean":
                geometric_score_reduction_protocol(effective_reduction)
            block_scores, metric_name = load_block_capacity_scores_from_results(
                layout.results_json_path,
                timestep_blocks,
                metric=config.allocation_metric,
                reduction=effective_reduction,
                group_scope=config.allocation_group_scope,
            )
            score_sources["block_capacity_scores"] = {
                "kind": "eval_output_results",
                "path": layout.results_json_path,
                "metric": metric_name,
                "reduction": effective_reduction,
                "requested_group_scope": config.allocation_group_scope,
                "group_scope": (
                    config.allocation_group_scope
                    if metric_name == "delta_p_eff_geomean"
                    else "precomputed_metric"
                ),
            }
            if metric_name == "delta_p_eff_geomean":
                score_sources["block_capacity_scores"].update(
                    {
                        "reduction_protocol": geometric_score_reduction_protocol(
                            effective_reduction
                        ),
                        "p_eff_source": (
                            "recomputed_from_allocatable_groups"
                            if config.allocation_group_scope == "allocatable"
                            else "results.p_eff"
                        ),
                    }
                )
        else:
            raise ValueError(missing_results_message(layout, f"{config.student_variant}"))

    if config.student_variant in {"layerwise_capacity", "reversed_layerwise_capacity", "combined_layerwise"}:
        if config.layer_capacity_scores_path is not None:
            layer_path = resolve_path(base_dir, config.layer_capacity_scores_path)
            layer_scores = load_capacity_scores(layer_path)
            score_sources["layer_capacity_scores"] = {
                "kind": "explicit_path",
                "path": layer_path,
                "requested_group_scope": config.allocation_group_scope,
                "group_scope": "precomputed",
            }
        elif layout is not None and layout.results_json_path is not None:
            effective_layer_source = config.layer_score_source
            if config.student_variant == "combined_layerwise" and effective_layer_source == "relative_delta_stack":
                effective_layer_source = "delta_p_eff_geomean"
            effective_reduction = config.score_reduction
            if effective_layer_source == "delta_p_eff_geomean":
                geometric_score_reduction_protocol(effective_reduction)
            layer_scores, structural_aggregation = load_layer_capacity_scores_with_metadata_from_results(
                layout.results_json_path,
                timestep_blocks,
                score_source=effective_layer_source,
                reduction=effective_reduction,
            )
            score_sources["layer_capacity_scores"] = {
                "kind": "eval_output_results",
                "path": layout.results_json_path,
                "score_source": effective_layer_source,
                "reduction": effective_reduction,
                "requested_group_scope": config.allocation_group_scope,
                "group_scope": (
                    "allocatable"
                    if structural_aggregation.get("excluded_group_count", 0) > 0
                    else "all"
                ),
            }
            if effective_layer_source == "delta_p_eff_geomean":
                score_sources["layer_capacity_scores"]["reduction_protocol"] = (
                    geometric_score_reduction_protocol(effective_reduction)
                )
            if structural_aggregation.get("applied"):
                score_sources["layer_capacity_scores"]["structural_aggregation"] = structural_aggregation
        else:
            raise ValueError(missing_results_message(layout, config.student_variant))
    if layout is not None and layout.results_json_path is not None:
        results_payload = load_json(layout.results_json_path)
        if isinstance(results_payload, dict):
            weighting = expansion_weighting_summary(results_payload)
            if weighting is not None:
                for source in score_sources.values():
                    if source.get("kind") == "eval_output_results":
                        source["filter_sampling_weighting"] = {
                            **weighting,
                            "application": (
                                "profile_metric_ht_estimate"
                                if "metric" in source and source.get("metric") in {"n_eff", "p_eff"}
                                else "selected_group_population_expansion"
                            ),
                        }
    return block_scores, layer_scores, score_sources


def raw_config_from_args(args: argparse.Namespace) -> tuple[dict[str, Any], Path | None, Path]:
    if args.config is None and args.eval_output_dir is None and getattr(args, "results_json", None) is None:
        raise ValueError("--config, --eval-output-dir or --results-json is required")

    config_path = Path(args.config).resolve() if args.config is not None else None
    raw_config = load_config(config_path) if config_path is not None else {}
    base_dir = config_path.parent if config_path is not None else Path.cwd()

    if args.eval_output_dir is not None:
        raw_config["eval_output_dir"] = args.eval_output_dir
    if args.allocation_results_dir is not None:
        raw_config["allocation_results_dir"] = args.allocation_results_dir
    if args.student_variant is not None:
        raw_config["student_variant"] = args.student_variant if len(args.student_variant) > 1 else args.student_variant[0]
    if args.timestep_grouping is not None:
        raw_config["timestep_grouping_path"] = args.timestep_grouping
    if args.allocation_metric is not None:
        raw_config["allocation_metric"] = args.allocation_metric
    if args.allocation_group_scope is not None:
        raw_config["allocation_group_scope"] = args.allocation_group_scope
    if args.score_reduction is not None:
        raw_config["score_reduction"] = args.score_reduction
    if args.layer_score_source is not None:
        raw_config["layer_score_source"] = args.layer_score_source
    if args.allocation_alpha is not None:
        raw_config["allocation_alpha"] = args.allocation_alpha
    if args.budget_tolerance is not None:
        raw_config["budget_tolerance"] = args.budget_tolerance
    if args.shuffle_seed is not None:
        raw_config["shuffle_seed"] = args.shuffle_seed

    raw_config.setdefault("match_original_total_budget", True)
    # Paper values (Section 3.4) unless the command line or a config sets them.
    raw_config.setdefault("allocation_metric", PAPER_ALLOCATION_METRIC)
    raw_config.setdefault("score_reduction", PAPER_SCORE_REDUCTION)
    raw_config.setdefault("layer_score_source", PAPER_LAYER_SCORE_SOURCE)
    return raw_config, config_path, base_dir


def config_allocation_dir(raw_config: dict[str, Any], args: argparse.Namespace) -> str | None:
    """Return the requested allocation output directory, if any."""

    if args.allocation_results_dir is not None:
        return str(args.allocation_results_dir)
    value = raw_config.get("allocation_results_dir")
    return None if value is None else str(value)


def selected_student_variants(raw_config: dict[str, Any], cli_student_variant: list[str] | None) -> list[str]:
    if cli_student_variant is not None:
        if isinstance(cli_student_variant, list):
            return [str(variant) for variant in cli_student_variant]
        return [str(cli_student_variant)]
    if "student_variant" in raw_config:
        raw_variant = raw_config["student_variant"]
        variants = [str(variant) for variant in raw_variant] if isinstance(raw_variant, list) else [str(raw_variant)]
        for variant in variants:
            if variant not in SUPPORTED_STUDENT_VARIANTS:
                raise ValueError(
                    f"student_variant must be one of {SUPPORTED_STUDENT_VARIANTS}, got {variant!r}"
                )
        return variants
    return list(DEFAULT_STUDENT_VARIANTS)


def resolve_timestep_blocks(
    config: CapacityAllocationConfig,
    base_dir: Path,
    layout: EvaluationOutputLayout | None,
) -> tuple[list, str | None]:
    timestep_blocks = list(config.timestep_blocks)
    grouping_source = None
    if config.timestep_grouping_path is not None:
        grouping_source = resolve_path(base_dir, config.timestep_grouping_path)
    elif layout is not None:
        grouping_source = layout.timestep_grouping_path

    if not timestep_blocks and grouping_source is not None:
        timestep_blocks = load_timestep_blocks_from_grouping(grouping_source)

    if not timestep_blocks:
        message = "timestep_blocks is required"
        if layout is not None and layout.checkpoint_paths and layout.timestep_grouping_path is None:
            message += (
                f"; found {len(layout.checkpoint_paths)} checkpoint dump(s) in {layout.output_dir}, "
                "but no grouping/timestep_grouping.json"
            )
        raise ValueError(message)
    return timestep_blocks, grouping_source


def build_variant_payload(
    *,
    config: CapacityAllocationConfig,
    config_path: Path | None,
    base_dir: Path,
    layout: EvaluationOutputLayout,
    timestep_blocks,
    grouping_source: str | None,
    original_model_budget,
    total_student_system_budget: float,
    profile_provenance: dict[str, Any] | None = None,
    allocation_scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    block_scores, layer_scores, score_sources = load_scores_if_needed(
        config,
        base_dir,
        timestep_blocks,
        layout,
    )
    plan = make_variant_capacity_budgets(
        config.student_variant,
        num_blocks=len(timestep_blocks),
        total_budget=total_student_system_budget,
        block_capacity_scores=block_scores,
        layer_capacity_scores=layer_scores,
        alpha=config.allocation_alpha,
        seed=config.shuffle_seed,
        allow_uniform_if_all_zero=False,
    )
    total_budget_report = validate_total_stored_budget(
        student_budgets=plan.block_budgets,
        original_model_budget=original_model_budget,
        tolerance=config.budget_tolerance,
    )

    plan_payload = plan.to_dict()
    layer_source = score_sources.get("layer_capacity_scores", {})
    structural_aggregation = layer_source.get("structural_aggregation")
    if isinstance(structural_aggregation, dict):
        plan_payload["layer_names"] = structural_aggregation["output_stage_keys"]
    layer_source_record = score_sources.get("layer_capacity_scores", {})
    effective_layer_score_source = (
        layer_source_record.get("score_source", config.layer_score_source)
        if isinstance(layer_source_record, dict)
        else config.layer_score_source
    )

    return {
        "config_path": None if config_path is None else str(config_path),
        "eval_output": layout.to_dict(),
        "student_variant": config.student_variant,
        "allocation_group_scope": config.allocation_group_scope,
        "score_reduction": config.score_reduction,
        "shuffle_seed": config.shuffle_seed if config.student_variant == "shuffled_capacity" else None,
        "allocation_rule": {
            "student_variant": config.student_variant,
            "allocation_metric": config.allocation_metric,
            "allocation_group_scope": config.allocation_group_scope,
            "score_reduction": config.score_reduction,
            "layer_score_source": effective_layer_score_source,
            "allocation_alpha": config.allocation_alpha,
            "shuffle_seed": (
                config.shuffle_seed if config.student_variant == "shuffled_capacity" else None
            ),
        },
        "timestep_grouping_path": grouping_source,
        "timestep_blocks": [[block.start, block.end] for block in timestep_blocks],
        "score_sources": score_sources,
        "original_model_budget": original_model_budget.to_dict(),
        "total_student_system_budget": total_student_system_budget,
        "target_budget_plan": plan_payload,
        "total_stored_budget_validation": total_budget_report.to_dict(),
        "rounding_rules": {
            "channel_divisibility": config.rounding_rules.channel_divisibility,
            "hidden_size_divisibility": config.rounding_rules.hidden_size_divisibility,
            "attention_head_divisibility": config.rounding_rules.attention_head_divisibility,
            "extra": dict(config.rounding_rules.extra),
        },
        **({"allocation_scope": allocation_scope} if allocation_scope is not None else {}),
        **(profile_provenance or provenance_from_results(None, None)),
    }


def write_allocation_results(
    *,
    layout: EvaluationOutputLayout,
    payloads: dict[str, dict[str, Any]],
    allocation_results_dir: str | None = None,
) -> dict[str, Any]:
    allocation_dir = Path(allocation_results_dir) if allocation_results_dir is not None else Path(layout.output_dir) / "allocation_results"
    allocation_dir.mkdir(parents=True, exist_ok=True)

    written_variant_paths: dict[str, str] = {}
    for variant, payload in payloads.items():
        output_path = allocation_dir / f"{variant}.json"
        payload["allocation_results_dir"] = str(allocation_dir)
        payload["allocation_results_path"] = str(output_path)
        written_variant_paths[variant] = str(output_path)
        output_path.write_text(json.dumps(payload, indent=2) + "\n")

    summary_path = allocation_dir / "summary.json"
    summary_provenance = provenance_from_artifact(next(iter(payloads.values())))
    summary = {
        "config_path": next(iter(payloads.values()))["config_path"],
        "eval_output": layout.to_dict(),
        "allocation_group_scope": next(iter(payloads.values()))[
            "allocation_group_scope"
        ],
        "score_reduction": next(iter(payloads.values()))["score_reduction"],
        "allocation_results_dir": str(allocation_dir),
        "allocation_summary_path": str(summary_path),
        "shuffle_seed": next(
            (payload["shuffle_seed"] for payload in payloads.values() if payload["shuffle_seed"] is not None),
            None,
        ),
        "student_variants": list(payloads.keys()),
        "written_variant_paths": written_variant_paths,
        "variants": payloads,
        **summary_provenance,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to an adaptive-capacity allocation config JSON/YAML file.")
    parser.add_argument("--eval-output-dir", default=None, help="Path to an evaluate_parameters* output directory, e.g. out_eval_edm_cifar10.")
    parser.add_argument(
        "--results-json",
        default=None,
        help=(
            "Explicit profile results (.json or .json.gz) instead of <eval-output-dir>/results.json, "
            "e.g. artifacts/profiles/cifar10_ddpmpp_random_same_norm.json.gz. Without "
            "--eval-output-dir, --allocation-results-dir is required."
        ),
    )
    parser.add_argument("--allocation-results-dir", default=None, help="Optional output directory for allocation JSONs. Defaults to <eval-output-dir>/allocation_results.")
    parser.add_argument("--student-variant", action="append", choices=SUPPORTED_STUDENT_VARIANTS, default=None, help="Student variant to dry-run. Repeat for multiple variants. Omit to run the four paper variants (global, uniform_blockwise, combined_blockwise, combined_layerwise) unless a config provides student_variant.")
    parser.add_argument("--timestep-grouping", default=None, help="Optional timestep_grouping.json path. Defaults to <eval-output-dir>/grouping/timestep_grouping.json.")
    parser.add_argument("--allocation-metric", default=None, help=f"Metric from results.json used for block budgets. Default: {PAPER_ALLOCATION_METRIC} (the paper's q_t); 'auto' uses n_eff then p_eff.")
    parser.add_argument(
        "--allocation-group-scope",
        choices=SUPPORTED_ALLOCATION_GROUP_SCOPES,
        default=None,
        help=(
            "Groups used by delta_p_eff_geomean phase scoring. 'all' preserves "
            "the historical stored-p_eff protocol; 'allocatable' recomputes "
            "p_eff from group_allocatable=true rows."
        ),
    )
    parser.add_argument(
        "--score-reduction",
        choices=SUPPORTED_SCORE_REDUCTIONS,
        default=None,
        help=(
            "How to aggregate timestep-level result values inside each block. "
            f"Default: {PAPER_SCORE_REDUCTION} (the paper's Q_k). For delta_p_eff_geomean, "
            "'mean' is duration-neutral (used by the audio appendix) and 'sum' "
            "preserves the exact historical EDM protocol."
        ),
    )
    parser.add_argument("--layer-score-source", default=None, help=f"Matrix key from results.json used for layerwise capacity scores. Default: {PAPER_LAYER_SCORE_SOURCE} (the paper's r_(b,g)).")
    parser.add_argument("--allocation-alpha", type=float, default=None, help="Power applied to capacity scores before budget normalization.")
    parser.add_argument("--budget-tolerance", type=float, default=None, help="Allowed relative mismatch in total stored budget validation.")
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help=f"Seed for the shuffled-capacity control (default: {DEFAULT_SHUFFLE_SEED}, unless set in --config).",
    )
    args = parser.parse_args()

    try:
        raw_config, config_path, base_dir = raw_config_from_args(args)
        student_variants = selected_student_variants(raw_config, args.student_variant)
    except ValueError as exc:
        parser.error(str(exc))

    if args.results_json is not None and args.eval_output_dir is None and "eval_output_dir" not in raw_config:
        if config_allocation_dir(raw_config, args) is None:
            parser.error("--results-json without --eval-output-dir requires --allocation-results-dir")
        # Use the allocation output directory's parent as the evaluation directory;
        # the profile itself comes from --results-json.
        allocation_dir = Path(resolve_path(base_dir, config_allocation_dir(raw_config, args)))
        allocation_dir.parent.mkdir(parents=True, exist_ok=True)
        raw_config["eval_output_dir"] = str(allocation_dir.parent)
    config = CapacityAllocationConfig.from_mapping(
        raw_config | {"student_variant": student_variants[0]}
    )
    if config.eval_output_dir is None:
        raise ValueError(
            "--eval-output-dir or eval_output_dir in config is required; "
            "allocation results are written under that directory"
        )
    layout = discover_evaluation_output(resolve_path(base_dir, config.eval_output_dir))
    if args.results_json is not None:
        results_path = Path(resolve_path(base_dir, args.results_json))
        if not results_path.is_file():
            parser.error(f"--results-json does not exist: {results_path}")
        layout = dataclasses.replace(layout, results_json_path=str(results_path))

    timestep_blocks, grouping_source = resolve_timestep_blocks(config, base_dir, layout)

    results_payload: dict[str, Any] | None = None
    if layout.results_json_path is not None:
        loaded_results = load_json(layout.results_json_path)
        if not isinstance(loaded_results, dict):
            raise ValueError(f"profile results must contain a JSON object: {layout.results_json_path}")
        results_payload = loaded_results
    profile_provenance = provenance_from_results(results_payload, layout.results_json_path)
    allocation_scope = (
        describe_group_allocation_scope(results_payload)
        if results_payload is not None
        else None
    )
    if grouping_source is not None:
        grouping_payload = load_json(grouping_source)
        if not isinstance(grouping_payload, dict):
            raise ValueError(f"timestep grouping must contain a JSON object: {grouping_source}")
        validate_matching_source_profile(
            profile_provenance["source_profile"],
            grouping_payload,
            context=f"timestep grouping {grouping_source}",
            expected_ablation_protocol=profile_provenance["ablation_protocol"],
            expected_filter_sampling=profile_provenance.get("filter_sampling"),
        )

    original_model_source = config.original_model_checkpoint_path or config.original_model_config_path
    if original_model_source is None and layout is not None:
        original_model_source = layout.results_json_path
    if original_model_source is None:
        raise ValueError(
            missing_results_message(
                layout,
                "original_model_config_path is required unless original_model_checkpoint_path is provided",
            )
        )
    original_model_source = resolve_path(base_dir, original_model_source)
    original_model_budget = compute_original_model_budget(original_model_source)
    total_student_system_budget = float(original_model_budget.parameters)

    payloads: dict[str, dict[str, Any]] = {}
    for student_variant in student_variants:
        variant_config = CapacityAllocationConfig.from_mapping(
            raw_config | {"student_variant": student_variant}
        )
        payloads[student_variant] = build_variant_payload(
            config=variant_config,
            config_path=config_path,
            base_dir=base_dir,
            layout=layout,
            timestep_blocks=timestep_blocks,
            grouping_source=grouping_source,
            original_model_budget=original_model_budget,
            total_student_system_budget=total_student_system_budget,
            profile_provenance=profile_provenance,
            allocation_scope=allocation_scope,
        )

    allocation_results_dir = resolve_path(base_dir, config.allocation_results_dir) if config.allocation_results_dir is not None else None
    summary = write_allocation_results(
        layout=layout,
        payloads=payloads,
        allocation_results_dir=allocation_results_dir,
    )
    payload = next(iter(payloads.values())) if len(payloads) == 1 else summary
    print(json.dumps(payload, indent=2) + "\n", end="")


if __name__ == "__main__":
    main()
