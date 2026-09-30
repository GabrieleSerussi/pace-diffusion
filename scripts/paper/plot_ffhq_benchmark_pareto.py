#!/usr/bin/env python3
"""Plot FFHQ benchmark quality metrics against measured sampling throughput."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


VARIANTS = (
    "teacher",
    "global",
    "uniform_blockwise",
    "combined_blockwise",
    "combined_layerwise",
)

LABELS = {
    "teacher": "Teacher",
    "global": "Global",
    "uniform_blockwise": "Uniform blockwise",
    "combined_blockwise": "Combined blockwise",
    "combined_layerwise": "Combined layerwise",
}

COLORS = {
    "teacher": "#4D4D4D",
    "global": "#0072B2",
    "uniform_blockwise": "#D55E00",
    "combined_blockwise": "#009E73",
    "combined_layerwise": "#CC79A7",
}

ANNOTATION_OFFSETS = {
    "teacher": (6, 6),
    "global": (6, 7),
    "uniform_blockwise": (6, 6),
    "combined_blockwise": (6, 6),
    "combined_layerwise": (6, -13),
}

METRICS = {
    "fid_nvlabs_legacy": ("NVLabs FID (minimum of 3 runs)", False),
    "fid_adm": ("ADM FID", False),
    "sfid_adm": ("ADM sFID", False),
    "inception_score_adm": ("ADM Inception Score", True),
    "precision_adm": ("ADM Precision", True),
    "recall_adm": ("ADM Recall", True),
}


@dataclass(frozen=True)
class Result:
    variant: str
    throughput: float
    throughput_min: float
    throughput_max: float
    metrics: dict[str, float]


def load_result(benchmark_root: Path, variant: str) -> Result:
    nvlabs_root = benchmark_root / variant / "ffhq64_nvlabs_edm_fid_v1"
    summary = json.loads((nvlabs_root / "evaluation_summary.json").read_text())
    throughputs = []
    for run_index in range(3):
        run = json.loads((nvlabs_root / f"run{run_index}" / "evaluation_result.json").read_text())
        generation = run["execution"]["generation"]
        value = generation.get("generation_only_images_per_second")
        if generation.get("newly_generated") and value is not None:
            throughputs.append(float(value))
    if not throughputs:
        raise ValueError(f"{variant} has no fresh generation throughput measurements")

    adm_path = (
        benchmark_root
        / variant
        / "ffhq64_openai_adm_custom_first50k_v1"
        / "evaluation_result.json"
    )
    adm_metrics = json.loads(adm_path.read_text())["metrics"]
    metrics = {"fid_nvlabs_legacy": float(summary["aggregate"])}
    for metric in METRICS:
        if metric != "fid_nvlabs_legacy":
            metrics[metric] = float(adm_metrics[metric])
    return Result(
        variant=variant,
        throughput=sum(throughputs) / len(throughputs),
        throughput_min=min(throughputs),
        throughput_max=max(throughputs),
        metrics=metrics,
    )


def pareto_frontier(results: list[Result], metric: str, higher_is_better: bool) -> list[Result]:
    def no_worse(other: Result, candidate: Result) -> bool:
        quality_ok = (
            other.metrics[metric] >= candidate.metrics[metric]
            if higher_is_better
            else other.metrics[metric] <= candidate.metrics[metric]
        )
        return other.throughput >= candidate.throughput and quality_ok

    def strictly_better(other: Result, candidate: Result) -> bool:
        quality_better = (
            other.metrics[metric] > candidate.metrics[metric]
            if higher_is_better
            else other.metrics[metric] < candidate.metrics[metric]
        )
        return other.throughput > candidate.throughput or quality_better

    frontier = [
        candidate
        for candidate in results
        if not any(
            other is not candidate and no_worse(other, candidate) and strictly_better(other, candidate)
            for other in results
        )
    ]
    return sorted(frontier, key=lambda result: result.throughput)


def plot_metric(ax, results: list[Result], metric: str, *, show_legend: bool) -> None:
    label, higher_is_better = METRICS[metric]
    frontier = pareto_frontier(results, metric, higher_is_better)
    frontier_variants = {result.variant for result in frontier}
    ax.plot(
        [result.throughput for result in frontier],
        [result.metrics[metric] for result in frontier],
        color="#555555",
        linestyle="--",
        linewidth=1.25,
        alpha=0.75,
        zorder=1,
        label="Pareto frontier",
    )
    for result in results:
        xerr = [[result.throughput - result.throughput_min], [result.throughput_max - result.throughput]]
        ax.errorbar(
            result.throughput,
            result.metrics[metric],
            xerr=xerr,
            fmt="o",
            markersize=8.5,
            color=COLORS[result.variant],
            markeredgecolor="white" if result.variant in frontier_variants else COLORS[result.variant],
            markerfacecolor=COLORS[result.variant] if result.variant in frontier_variants else "white",
            markeredgewidth=1.0,
            capsize=2.5,
            linewidth=1.0,
            zorder=3,
            label=LABELS[result.variant],
        )
        ax.annotate(
            LABELS[result.variant],
            (result.throughput, result.metrics[metric]),
            xytext=ANNOTATION_OFFSETS[result.variant],
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_title(label)
    ax.set_xlabel("Sampling throughput (images/s on 4 GPUs; higher is better)")
    ax.set_ylabel(f"{label} ({'higher' if higher_is_better else 'lower'} is better)")
    ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), frameon=False, fontsize=8, loc="best")


def write_csv(results: list[Result], path: Path) -> None:
    frontier_sets = {
        metric: {result.variant for result in pareto_frontier(results, metric, higher)}
        for metric, (_label, higher) in METRICS.items()
    }
    fields = [
        "variant",
        "throughput_images_per_second",
        "throughput_min",
        "throughput_max",
        "speedup_vs_teacher",
    ]
    for metric in METRICS:
        fields.extend([metric, f"{metric}_pareto"])
    teacher_throughput = next(result.throughput for result in results if result.variant == "teacher")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {
                "variant": result.variant,
                "throughput_images_per_second": result.throughput,
                "throughput_min": result.throughput_min,
                "throughput_max": result.throughput_max,
                "speedup_vs_teacher": result.throughput / teacher_throughput,
            }
            for metric in METRICS:
                row[metric] = result.metrics[metric]
                row[f"{metric}_pareto"] = result.variant in frontier_sets[metric]
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = [load_result(args.benchmark_root, variant) for variant in VARIANTS]
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(8.4, 5.5), constrained_layout=True)
        plot_metric(ax, results, metric, show_legend=True)
        fig.suptitle("FFHQ 64×64: quality–throughput Pareto tradeoff", fontsize=13)
        fig.savefig(args.output_dir / f"{metric}_vs_throughput.png", dpi=220)
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), constrained_layout=True)
    for ax, metric in zip(axes.flat, METRICS):
        plot_metric(ax, results, metric, show_legend=False)
    fig.suptitle("FFHQ 64×64: quality–throughput Pareto tradeoffs", fontsize=16)
    fig.savefig(args.output_dir / "all_metrics_vs_throughput.png", dpi=220)
    plt.close(fig)
    write_csv(results, args.output_dir / "pareto_metrics.csv")

    payload = {
        result.variant: {
            "throughput_images_per_second": result.throughput,
            "throughput_range": [result.throughput_min, result.throughput_max],
            "metrics": result.metrics,
            "pareto_metrics": [
                metric
                for metric, (_label, higher) in METRICS.items()
                if result in pareto_frontier(results, metric, higher)
            ],
        }
        for result in results
    }
    (args.output_dir / "pareto_metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
