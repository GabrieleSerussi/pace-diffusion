#!/usr/bin/env python3
"""Build EDM U-Net student architecture plans from a profile and its allocations.

Each variant's phase budgets (and, for ``combined_layerwise``, its layer
budgets) are converted into realizable NarrowSongUNet students (Section 3.4
and Appendix A.4 of the paper).  The profile is ``<eval-output-dir>/results.json``
or ``--results-json`` (for example a released ``artifacts/profiles/*.json.gz``);
the allocations are ``<allocation-results-dir>/<variant>.json`` (or
``.json.gz``).  Building students needs the NVlabs/edm checkout
(``$EDM_REPO`` or ``../edm``).

Released plans of the trained students are in ``artifacts/plans/``; a rebuild
with the current code can differ from them (see ``REPRODUCING.md``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pace.edm_distillation import (
    DEFAULT_DISTILLATION_VARIANTS,
    SUPPORTED_DISTILLATION_VARIANTS,
    prepare_distillation_architectures,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-output-dir", default="out_eval_edm_cifar10")
    parser.add_argument("--output-dir", default="out_distill_edm_cifar10/smoke")
    parser.add_argument("--allocation-results-dir", default=None, help="Optional allocation_results directory to read instead of <eval-output-dir>/allocation_results.")
    parser.add_argument("--results-json", default=None, help="Explicit profile (.json or .json.gz) instead of <eval-output-dir>/results.json.")
    parser.add_argument(
        "--variant",
        action="append",
        choices=SUPPORTED_DISTILLATION_VARIANTS,
        help="Variant to prepare. Repeat to prepare multiple variants. Defaults to the four paper variants.",
    )
    parser.add_argument("--budget-tolerance", type=float, default=0.05)
    parser.add_argument("--shuffle-seed", type=int, default=3)
    parser.add_argument("--layerwise-search-steps", type=int, default=12)
    args = parser.parse_args()

    summary = prepare_distillation_architectures(
        eval_output_dir=args.eval_output_dir,
        output_dir=args.output_dir,
        allocation_results_dir=args.allocation_results_dir,
        variants=args.variant or DEFAULT_DISTILLATION_VARIANTS,
        budget_tolerance=args.budget_tolerance,
        shuffle_seed=args.shuffle_seed,
        layerwise_search_steps=args.layerwise_search_steps,
        results_json_path=args.results_json,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
