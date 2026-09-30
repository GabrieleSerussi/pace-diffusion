#!/usr/bin/env python3
"""Create an allocation-ready residual-only view of DiffWave filter PFI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pace.diffwave_residual_postprocess import (
    DEFAULT_EXPECTED_ALLOCATABLE_FILTERS,
    postprocess_residual_filter_results,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute DiffWave per-filter usage metrics over only groups where "
            "group_allocatable=true, without modifying the source profile."
        )
    )
    parser.add_argument(
        "--results",
        default="out_eval_diffwave_sc09/per_filter_exact_timestep_v1/profile_pool100_n4_t200_bins20_seed0/results.json",
        help="Source results.json or its containing profile directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Derived artifact directory. By default this is residual_only beside "
            "the resolved source results, including when --results traverses a symlink."
        ),
    )
    parser.add_argument(
        "--expected-allocatable-count",
        type=int,
        default=DEFAULT_EXPECTED_ALLOCATABLE_FILTERS,
        help=(
            "Require this many group_allocatable=true filters (default: 36864). "
            "Pass 0 to disable the production-count guard."
        ),
    )
    parser.add_argument(
        "--top-filter-count",
        type=int,
        default=256,
        help="Number of residual filters retained in top-filter metadata and heatmaps.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Write results.json and metrics.pt but skip PNG artifacts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    expected_count = (
        None if args.expected_allocatable_count == 0 else args.expected_allocatable_count
    )
    invocation = [sys.argv[0], *(argv if argv is not None else sys.argv[1:])]
    success = postprocess_residual_filter_results(
        args.results,
        output_dir=args.output_dir,
        expected_allocatable_count=expected_count,
        top_filter_count=args.top_filter_count,
        save_plots=not args.no_plots,
        invocation_argv=invocation,
    )
    print(json.dumps(success, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
