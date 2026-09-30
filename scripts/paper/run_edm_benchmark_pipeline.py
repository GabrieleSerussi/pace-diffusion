#!/usr/bin/env python3
"""Preflight and optionally execute the resumable EDM benchmark pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pace.benchmark_pipeline import (
    DEFAULT_SHARED_ROOT,
    PIPELINE_PRESETS,
    SUPPORTED_ALLOCATION_SCORE_REDUCTIONS,
    SUPPORTED_GROUPING_BUILTIN_COSTS,
    SUPPORTED_PIPELINE_VARIANTS,
    SUPPORTED_PROFILE_ABLATION_MODES,
    PreflightError,
    build_pipeline_actions,
    execute_pipeline_actions,
    parse_gpu_ids,
    pipeline_plan_payload,
    run_preflight,
)
from pace.external_repos import default_edm_root


STAGE_ORDER = ("prepare", "profile", "group", "allocate", "plans", "train", "benchmark")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--preset", choices=sorted(PIPELINE_PRESETS), required=True)
    parser.add_argument("--source-root", type=Path, required=True, help="Raw FFHQ source or flat LSUN Bedroom JPEG directory.")
    parser.add_argument("--shared-root", type=Path, default=DEFAULT_SHARED_ROOT, help="Root for every large dataset, cache, run, and metric artifact ($PACE_SHARED_ROOT or <repo>/shared).")
    parser.add_argument("--gpu-ids", default=None, help="Explicit comma-separated physical GPU IDs. Defaults to the preset's GPU list.")
    parser.add_argument("--min-free-memory-mib", type=int, default=None, help="Minimum free memory required on every selected GPU.")
    parser.add_argument("--min-free-storage-gib", type=int, default=None)
    parser.add_argument(
        "--variant",
        action="append",
        choices=SUPPORTED_PIPELINE_VARIANTS,
        help="Student variant; repeat as needed. Defaults to the dataset preset.",
    )
    parser.add_argument(
        "--num-timestep-blocks",
        type=int,
        default=None,
        help="Number of contiguous timestep groups selected from the grouping cost curve; defaults to the preset.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="Seed shared by shuffled allocation and architecture planning; defaults to the preset.",
    )
    parser.add_argument(
        "--grouping-builtin-cost",
        choices=SUPPORTED_GROUPING_BUILTIN_COSTS,
        default=None,
        help="Timestep grouping objective; defaults to the dataset preset.",
    )
    parser.add_argument(
        "--allocation-metric",
        default=None,
        help="Metric used for block allocation; defaults to the dataset preset.",
    )
    parser.add_argument(
        "--score-reduction",
        choices=SUPPORTED_ALLOCATION_SCORE_REDUCTIONS,
        default=None,
        help="Within-block reduction used for allocation scores; defaults to the dataset preset.",
    )
    parser.add_argument(
        "--profile-ablation-mode",
        choices=SUPPORTED_PROFILE_ABLATION_MODES,
        default="pfi",
        help="Teacher profile ablation; PFI uses its versioned isolated downstream tree.",
    )
    parser.add_argument("--ffhq-adm-reference", type=Path, default=None, help="Custom first-50k FFHQ ADM reference; defaults under shared references.")
    parser.add_argument("--nvlabs-edm-root", type=Path, default=default_edm_root(), help="Pinned NVLabs EDM checkout containing fid.py ($EDM_REPO or ../edm).")
    parser.add_argument("--adm-evaluator", type=Path, default=None, help="Pinned OpenAI evaluator.py path in the optional ADM environment.")
    parser.add_argument("--adm-python", type=Path, default=None, help="Python executable from the isolated ADM metric environment.")
    parser.add_argument(
        "--adm-detector",
        type=Path,
        default=None,
        help="Canonical classify_image_graph_def.pb; defaults to <shared-root>/references.",
    )
    parser.add_argument("--artifact-retention", choices=("keep", "keep_preview", "discard"), default="keep_preview")
    parser.add_argument("--preview-count", type=int, default=64)
    parser.add_argument("--stage", action="append", choices=STAGE_ORDER, help="Execute only selected stages; repeat to select several.")
    parser.add_argument("--force-action", action="append", default=[], help="Rerun an action ID even if state records it complete.")
    parser.add_argument("--skip-lsun-full-scan", action="store_true", help="Development-only: skip the million-file permission/name scan. Production execute still refuses this option.")
    parser.add_argument("--execute", action="store_true", help="Run commands after preflight. Without this flag the command is read-only and prints the complete plan.")
    parser.add_argument("--plan-output", type=Path, default=None, help="Optional JSON destination for the dry-run plan; should be on shared storage.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    preset = PIPELINE_PRESETS[args.preset]
    num_timestep_blocks = (
        preset.num_timestep_blocks if args.num_timestep_blocks is None else args.num_timestep_blocks
    )
    grouping_builtin_cost = (
        preset.grouping_builtin_cost
        if args.grouping_builtin_cost is None
        else args.grouping_builtin_cost
    )
    allocation_metric = (
        preset.allocation_metric if args.allocation_metric is None else args.allocation_metric
    )
    score_reduction = (
        preset.score_reduction
        if args.score_reduction is None
        else args.score_reduction
    )
    variants = tuple(args.variant or preset.default_variants)
    shuffle_seed = preset.shuffle_seed if args.shuffle_seed is None else args.shuffle_seed
    selected_stages = set(args.stage) if args.stage else None
    benchmark_requested = selected_stages is None or "benchmark" in selected_stages
    gpu_requested = selected_stages is None or bool({"profile", "train", "benchmark"}.intersection(selected_stages))
    try:
        gpu_ids = parse_gpu_ids(args.gpu_ids or preset.default_gpu_ids)
        full_gpu_suite_requested = selected_stages is None or bool(
            {"profile", "train"}.intersection(selected_stages)
        )
        if args.execute and args.skip_lsun_full_scan and preset.dataset == "lsun_bedroom":
            raise PreflightError("production LSUN execution requires the full permission/name scan")
        preflight = run_preflight(
            preset,
            source_root=args.source_root,
            shared_root=args.shared_root,
            gpu_ids=gpu_ids,
            min_free_memory_mib=args.min_free_memory_mib,
            min_free_storage_gib=args.min_free_storage_gib,
            scan_lsun=not args.skip_lsun_full_scan,
            benchmark_requested=benchmark_requested,
            ffhq_adm_reference=args.ffhq_adm_reference,
            nvlabs_edm_root=args.nvlabs_edm_root,
            adm_evaluator=args.adm_evaluator,
            adm_python=args.adm_python,
            adm_detector=args.adm_detector,
            gpu_requested=gpu_requested,
            required_gpu_count=(
                preset.required_gpu_count if full_gpu_suite_requested else len(gpu_ids)
            ),
        )
        layout, actions = build_pipeline_actions(
            preset,
            source_root=args.source_root,
            shared_root=args.shared_root,
            gpu_ids=gpu_ids,
            variants=variants,
            ffhq_adm_reference=args.ffhq_adm_reference,
            nvlabs_edm_root=args.nvlabs_edm_root,
            adm_evaluator=args.adm_evaluator,
            adm_python=args.adm_python,
            adm_detector=args.adm_detector,
            artifact_retention=args.artifact_retention,
            preview_count=args.preview_count,
            profile_ablation_mode=args.profile_ablation_mode,
            project_root=REPO_ROOT,
            num_timestep_blocks=num_timestep_blocks,
            grouping_builtin_cost=grouping_builtin_cost,
            allocation_metric=allocation_metric,
            score_reduction=score_reduction,
            shuffle_seed=shuffle_seed,
        )
    except PreflightError as exc:
        parser.error(str(exc))

    selected_stage_args = [item for stage in (args.stage or []) for item in ("--stage", stage)]
    variant_args = [item for variant in variants for item in ("--variant", variant)]
    resume_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--preset",
        args.preset,
        "--source-root",
        str(args.source_root.expanduser().resolve()),
        "--shared-root",
        str(args.shared_root.expanduser().resolve()),
        "--gpu-ids",
        ",".join(str(index) for index in gpu_ids),
        "--profile-ablation-mode",
        args.profile_ablation_mode,
        "--num-timestep-blocks",
        str(num_timestep_blocks),
        "--shuffle-seed",
        str(shuffle_seed),
        "--grouping-builtin-cost",
        grouping_builtin_cost,
        "--allocation-metric",
        allocation_metric,
        "--score-reduction",
        score_reduction,
        *variant_args,
        *selected_stage_args,
    ]
    if args.ffhq_adm_reference is not None:
        resume_command += ["--ffhq-adm-reference", str(args.ffhq_adm_reference.expanduser().resolve())]
    if args.nvlabs_edm_root is not None:
        resume_command += ["--nvlabs-edm-root", str(args.nvlabs_edm_root.expanduser().resolve())]
    if args.adm_evaluator is not None:
        resume_command += ["--adm-evaluator", str(args.adm_evaluator.expanduser().resolve())]
    if args.adm_python is not None:
        resume_command += [
            "--adm-python",
            str(Path(os.path.abspath(args.adm_python.expanduser()))),
        ]
    if args.adm_detector is not None:
        resume_command += ["--adm-detector", str(args.adm_detector.expanduser().resolve())]
    resume_command += [
        "--artifact-retention",
        args.artifact_retention,
        "--preview-count",
        str(args.preview_count),
        "--execute",
    ]
    payload = pipeline_plan_payload(preset, layout, actions, preflight)
    payload["driver"] = {
        "state_path": str(layout.run_root / "pipeline_state.json"),
        "recommended_log_path": str(layout.run_root / "pipeline_driver.log"),
        "resume_command": resume_command,
        "resume_command_shell": shlex.join(resume_command),
        "selected_stages": args.stage,
        "num_timestep_blocks": num_timestep_blocks,
        "grouping_builtin_cost": grouping_builtin_cost,
        "allocation_metric": allocation_metric,
        "score_reduction": score_reduction,
        "variants": list(variants),
        "shuffle_seed": shuffle_seed,
    }
    if args.plan_output is not None:
        destination = args.plan_output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    if not args.execute:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if not preflight["passed"]:
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise SystemExit("preflight failed; no pipeline commands were launched")
    result = execute_pipeline_actions(
        actions,
        state_path=layout.run_root / "pipeline_state.json",
        repo_root=REPO_ROOT,
        selected_stages=selected_stages,
        force_actions=set(args.force_action),
    )
    print(json.dumps({"dry_run": False, "preflight": preflight, "result": result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
