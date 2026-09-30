#!/usr/bin/env python3
"""Reproduce the phases, the Table 3 statistics and the U-Net phase budgets on a CPU.

The script runs the repository's phase discovery
(``scripts/optimize_timestep_grouping.py``) and U-Net allocator
(``scripts/dry_run_capacity_allocation.py``) on the released profiles in
``artifacts/``, and checks the results against the released groupings and
allocations and against the values printed in the paper. It needs neither a
GPU nor the external repositories, and runs in under a minute on a laptop.

Checks:

1. Phase discovery (Section 3.3). The Section 3.3 objective with lambda_sep =
   0.02 and automatic K selects the released phases of CIFAR-10, ImageNet-64,
   LSUN Bedroom and the archived DiT-Micro profile. On FFHQ-64 it selects a
   single phase; the released FFHQ-64 phases are the K = 3 optimum of the
   within-phase correlation cost (``--builtin_cost matrix_correlation``), which
   at fixed K is also the optimum of the objective with lambda_sep = 0; both
   are checked.
2. Table 3 and Appendix C. Mean within-phase and cross-phase correlations of
   the stored bin-by-bin Pearson matrix at the released phases, and a check
   that the stored matrix equals the similarity that phase discovery computes
   (Horvitz-Thompson weighted for LSUN Bedroom).
3. Phase budgets (Section 3.4). The four student variants are allocated again
   from the released profiles and groupings; the phase budgets must equal the
   released ones exactly, and the layer budgets of the layerwise student must
   agree to 1e-6 parameters (the summation order of floating-point values can
   differ in the last digits).
4. Appendix B. The released audio metrics snapshot passes the validation of
   ``scripts/paper/plot_audio_appendix.py``; its phases are the K = 3 optimum of the
   stored correlation matrix without the separation term, and its printed
   similarities and capacity shares are checked.

Usage::

    python reproduce/cpu_phases_and_budgets.py [--output-dir outputs/cpu_phases_and_budgets]

The exit status is 0 when every check passes and 1 otherwise.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pace.jsonio import find_json_file, load_json  # noqa: E402

VARIANTS = ("global", "uniform_blockwise", "combined_blockwise", "combined_layerwise")

# Values printed in the paper (Table 3, Appendix C, Section 3.4 budgets).
UNET = [
    {
        "dataset": "cifar10",
        "label": "CIFAR-10",
        "stem": "cifar10_ddpmpp_random_same_norm",
        "auto_boundaries": [0, 8, 20],
        "boundaries": [0, 8, 20],
        "within": [0.912, 0.923],
        "cross": {(0, 1): 0.363},
        "p_tot": 51_025_667,
        "block_budgets": [5_406_655.84, 45_619_011.16],
    },
    {
        "dataset": "imagenet64",
        "label": "ImageNet-64",
        "stem": "imagenet64_adm_random_same_norm",
        "auto_boundaries": [0, 16, 20],
        "boundaries": [0, 16, 20],
        "within": [0.904, 0.951],
        "cross": {(0, 1): 0.429},
        "p_tot": 266_818_179,
        "block_budgets": [183_586_906.62, 83_231_272.38],
    },
    {
        "dataset": "ffhq64",
        "label": "FFHQ-64",
        "stem": "ffhq64_ddpmpp_pfi",
        "auto_boundaries": [0, 20],
        "boundaries": [0, 3, 16, 20],
        "within": [0.917, 0.984, 0.982],
        "cross": {(1, 2): 0.884, (0, 2): 0.554},
        "p_tot": 56_301_955,
        "block_budgets": [5_795_995.43, 39_836_608.68, 10_669_350.89],
    },
    {
        "dataset": "lsun256",
        "label": "LSUN Bedroom",
        "stem": "lsun256_adm_pfi_stratified",
        "auto_boundaries": [0, 16, 20],
        "boundaries": [0, 16, 20],
        "within": [0.950, 0.940],
        "cross": {(0, 1): 0.547},
        "p_tot": 498_338_051,
        "block_budgets": [317_097_723.42, 181_240_327.58],
    },
]
DIT_MICRO = {
    "label": "DiT-Micro (Appendix C.2)",
    "profile": "artifacts/dit_micro/dit_micro_perm_results.json",
    "grouping": "artifacts/dit_micro/dit_micro_perm_4phase.json",
    "boundaries": [0, 4, 8, 16, 20],
    "within": [0.956, 0.832, 0.914, 0.928],
    "cross": {(0, 2): -0.146, (0, 3): -0.189},
}
AUDIO = {
    "metrics": "artifacts/audio/audio_appendix_metrics.json",
    "boundaries": [0, 13, 17, 20],
    "auto_boundaries": [0, 17, 20],
    "within": [0.9915, 0.9434, 0.9118],
    "cross": {(0, 1): 0.8153, (0, 2): 0.2683, (1, 2): 0.6926},
    "shares": [0.2339, 0.3042, 0.4619],
}


class Checker:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.count = 0

    def check(self, condition: bool, message: str) -> None:
        self.count += 1
        status = "ok  " if condition else "FAIL"
        print(f"  [{status}] {message}")
        if not condition:
            self.failures.append(message)


def run_script(args: list[str]) -> None:
    command = [sys.executable, *args]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout + completed.stderr)
        raise SystemExit(f"command failed: {' '.join(command)}")


def phase_statistics(matrix: np.ndarray, boundaries: list[int]) -> tuple[list[float], dict]:
    """Mean off-diagonal correlation within each phase, mean correlation across phases."""

    blocks = list(zip(boundaries[:-1], boundaries[1:]))
    within = []
    for start, end in blocks:
        block = matrix[start:end, start:end]
        size = end - start
        within.append(float((block.sum() - np.trace(block)) / (size * size - size)))
    cross = {}
    for x, y in itertools.combinations(range(len(blocks)), 2):
        (a, b), (c, d) = blocks[x], blocks[y]
        cross[(x, y)] = float(matrix[a:b, c:d].mean())
    return within, cross


def grouping(profile: str, output: Path, *extra: str) -> dict:
    run_script([
        "scripts/optimize_timestep_grouping.py",
        "--matrix", profile, "--matrix_key", "relative_delta_stack",
        *extra, "--output", str(output),
    ])
    return json.loads(output.read_text())


PAPER_OBJECTIVE = (
    "--builtin_cost", "matrix_correlation_cross_penalty", "--cross_block_lambda", "0.02",
    "--cross_block_reward_normalization", "size", "--num_blocks", "auto",
)


def check_phase_discovery(checker: Checker, out: Path) -> None:
    print("\n1. Phase discovery (Section 3.3 objective, lambda_sep = 0.02, automatic K)")
    for spec in UNET:
        profile = f"artifacts/profiles/{spec['stem']}.json.gz"
        released = load_json(ROOT / f"artifacts/groupings/{spec['stem']}.json")
        result = grouping(profile, out / "groupings" / f"{spec['dataset']}_auto.json", *PAPER_OBJECTIVE)
        if spec["auto_boundaries"] == spec["boundaries"]:
            checker.check(
                result["boundaries"] == released["boundaries"] == spec["boundaries"]
                and abs(result["total_score"] - released["total_score"]) < 1e-9,
                f"{spec['label']}: phases {result['boundaries']}, J(B) = {result['total_score']:.6f} "
                f"(released {released['boundaries']}, J(B) = {released['total_score']:.6f})",
            )
        else:
            checker.check(
                result["boundaries"] == spec["auto_boundaries"],
                f"{spec['label']}: automatic K selects {result['boundaries']} "
                f"(K = {len(result['boundaries']) - 1}); the released phases use another cost",
            )
            fixed = grouping(
                profile, out / "groupings" / f"{spec['dataset']}_matrix_correlation_k3.json",
                "--builtin_cost", "matrix_correlation", "--num_blocks", "3",
            )
            checker.check(
                fixed["boundaries"] == released["boundaries"] == spec["boundaries"]
                and abs(fixed["total_cost"] - released["total_cost"]) < 1e-9,
                f"{spec['label']}: matrix_correlation with K = 3 gives {fixed['boundaries']}, "
                f"cost {fixed['total_cost']:.6f} (released {released['boundaries']}, "
                f"cost {released['total_cost']:.6f})",
            )
            # At fixed K, matrix_correlation has the optimum of the Section 3.3
            # objective without the separation term (lambda_sep = 0).
            zero = grouping(
                profile, out / "groupings" / f"{spec['dataset']}_lambda0_k3.json",
                "--builtin_cost", "matrix_correlation_cross_penalty", "--cross_block_lambda", "0",
                "--num_blocks", "3",
            )
            checker.check(
                zero["boundaries"] == spec["boundaries"],
                f"{spec['label']}: the Section 3.3 objective with lambda_sep = 0 and K = 3 gives "
                f"{zero['boundaries']}",
            )
    released = load_json(ROOT / DIT_MICRO["grouping"])
    result = grouping(DIT_MICRO["profile"], out / "groupings" / "dit_micro_auto.json", *PAPER_OBJECTIVE)
    checker.check(
        result["boundaries"] == released["boundaries"] == DIT_MICRO["boundaries"]
        and abs(result["total_score"] - released["total_score"]) < 1e-9,
        f"{DIT_MICRO['label']}: phases {result['boundaries']}, J(B) = {result['total_score']:.6f} "
        f"(released J(B) = {released['total_score']:.6f})",
    )


def check_correlations(checker: Checker) -> None:
    print("\n2. Table 3 and Appendix C (within-phase and cross-phase correlation)")
    from scripts.optimize_timestep_grouping import (
        _compute_pairwise_similarity_matrix,
        load_matrix_feature_expansion,
    )

    def check_one(label, profile_path, matrix_key, boundaries, within_expected, cross_expected):
        profile = load_json(ROOT / profile_path)
        stored = np.asarray(profile[matrix_key], dtype=np.float64)
        features = np.asarray(profile["relative_delta_stack"], dtype=np.float64)
        weights, _ = load_matrix_feature_expansion(
            str(ROOT / profile_path), matrix=features, matrix_key="relative_delta_stack", matrix_axis=-1,
        )
        recomputed = _compute_pairwise_similarity_matrix(
            features, axis=-1, metric="correlation", feature_weights=weights,
        )
        deviation = float(np.abs(recomputed - stored).max())
        weighting = "Horvitz-Thompson weighted " if weights is not None else ""
        checker.check(
            deviation < 1e-9,
            f"{label}: stored {matrix_key} equals the {weighting}Pearson similarity of phase "
            f"discovery (max deviation {deviation:.1e})",
        )
        within, cross = phase_statistics(stored, boundaries)
        within_rounded = [round(value, 3) for value in within]
        cross_rounded = {pair: round(cross[pair], 3) for pair in cross_expected}
        checker.check(
            within_rounded == within_expected and cross_rounded == cross_expected,
            f"{label}: within {within_rounded}, cross "
            + ", ".join(f"{x + 1}-{y + 1} {value}" for (x, y), value in cross_rounded.items()),
        )

    for spec in UNET:
        check_one(spec["label"], f"artifacts/profiles/{spec['stem']}.json.gz", "C_noise_levels",
                  spec["boundaries"], spec["within"], spec["cross"])
    check_one(DIT_MICRO["label"], DIT_MICRO["profile"], "C_timesteps",
              DIT_MICRO["boundaries"], DIT_MICRO["within"], DIT_MICRO["cross"])


def check_budgets(checker: Checker, out: Path) -> None:
    print("\n3. Phase budgets (Section 3.4: delta_p_eff_geomean, summed within each phase)")
    for spec in UNET:
        target = out / "allocations" / spec["dataset"]
        run_script([
            "scripts/dry_run_capacity_allocation.py",
            "--results-json", f"artifacts/profiles/{spec['stem']}.json.gz",
            "--timestep-grouping", f"artifacts/groupings/{spec['stem']}.json",
            "--allocation-results-dir", str(target),
            "--allocation-metric", "delta_p_eff_geomean", "--score-reduction", "sum",
            "--layer-score-source", "delta_p_eff_geomean",
            *itertools.chain.from_iterable(("--student-variant", variant) for variant in VARIANTS),
        ])
        for variant in VARIANTS:
            new = json.loads((target / f"{variant}.json").read_text())["target_budget_plan"]
            old = load_json(find_json_file(ROOT / "artifacts" / "allocations" / spec["dataset"], variant))["target_budget_plan"]
            same_blocks = new["block_budgets"] == old["block_budgets"]
            same_total = new["total_student_system_budget"] == old["total_student_system_budget"] == spec["p_tot"]
            message = (f"{spec['label']} {variant}: phase budgets "
                       f"{[round(value, 2) for value in new['block_budgets']]}, P_tot {int(new['total_student_system_budget']):,}")
            if variant == "combined_blockwise":
                same_blocks = same_blocks and [round(value, 2) for value in new["block_budgets"]] == spec["block_budgets"]
            if variant == "combined_layerwise":
                deviation = max(
                    float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max())
                    for a, b in zip(new["layer_budgets"], old["layer_budgets"])
                )
                same_blocks = same_blocks and deviation <= 1e-6
                message += f", layer budgets within {deviation:.1e} of the released ones"
            checker.check(same_blocks and same_total, message)


def check_audio(checker: Checker) -> None:
    print("\n4. Appendix B (audio snapshot)")
    from scripts.optimize_timestep_grouping import optimal_timestep_partition
    from scripts.paper.plot_audio_appendix import _validate_metrics

    metrics = load_json(ROOT / AUDIO["metrics"])
    try:
        _validate_metrics(metrics)
        valid = True
    except Exception as exc:  # the validator raises ValueError or KeyError
        print(f"    {exc}")
        valid = False
    checker.check(valid, "the snapshot passes the validation of scripts/paper/plot_audio_appendix.py")

    matrix = np.asarray(metrics["timestep_correlation_pearson"], dtype=np.float64)
    size = matrix.shape[0]

    def within_cost(start: int, end: int) -> float:
        block = matrix[start:end, start:end]
        return -float((block.sum() - np.trace(block)) / (end - start))

    fixed = optimal_timestep_partition(num_timesteps=size, num_blocks=3, cost_fn=within_cost, min_block_size=2)
    checker.check(
        list(fixed.boundaries) == AUDIO["boundaries"] == metrics["phase_boundaries"],
        f"K = 3 optimum without the separation term (phases of at least two bins): {list(fixed.boundaries)}",
    )

    def objective(boundaries: list[int], lam: float) -> float:
        blocks = list(zip(boundaries[:-1], boundaries[1:]))
        within = sum(-within_cost(a, b) for a, b in blocks)
        cross = sum(matrix[a:b, c:d].sum() for (a, b), (c, d) in itertools.permutations(blocks, 2))
        return within - lam * cross

    candidates = []
    for num_phases in range(1, 6):
        for cuts in itertools.combinations(range(1, size), num_phases - 1):
            bounds = [0, *cuts, size]
            if all(b - a >= 2 for a, b in zip(bounds[:-1], bounds[1:])):
                candidates.append(bounds)
    best = max(candidates, key=lambda bounds: (objective(bounds, 0.02), -len(bounds)))
    checker.check(
        best == AUDIO["auto_boundaries"] == metrics["grouping_diagnostics"]["pearson_auto"]["boundaries"],
        f"Section 3.3 objective with lambda_sep = 0.02 and automatic K (at most five phases) selects {best}",
    )

    within, cross = phase_statistics(matrix, metrics["phase_boundaries"])
    within_rounded = [round(value, 4) for value in within]
    cross_rounded = {pair: round(cross[pair], 4) for pair in AUDIO["cross"]}
    checker.check(
        within_rounded == AUDIO["within"] and cross_rounded == AUDIO["cross"],
        f"phase similarities: within {within_rounded}, cross "
        + ", ".join(f"{x + 1}-{y + 1} {value}" for (x, y), value in cross_rounded.items()),
    )
    fractions = metrics["allocation"]["phase_budget_fractions"]
    budgets = np.asarray(metrics["allocation"]["phase_budgets"], dtype=np.float64)
    consistent = np.allclose(budgets / budgets.sum(), fractions, rtol=0, atol=1e-12)
    checker.check(
        [round(value, 4) for value in fractions] == AUDIO["shares"] and consistent,
        f"capacity shares {[round(value, 4) for value in fractions]} "
        f"({metrics['allocation']['phase_reduction']} of {metrics['allocation']['metric']})",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs" / "cpu_phases_and_budgets",
        help="directory for the regenerated groupings and allocations",
    )
    args = parser.parse_args()
    out = args.output_dir.resolve()
    (out / "groupings").mkdir(parents=True, exist_ok=True)
    checker = Checker()
    check_phase_discovery(checker, out)
    check_correlations(checker)
    check_budgets(checker, out)
    check_audio(checker)
    print()
    if checker.failures:
        print(f"{len(checker.failures)} of {checker.count} checks failed:")
        for failure in checker.failures:
            print(f"  {failure}")
        return 1
    print(f"All {checker.count} checks passed. Outputs are in {out}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
