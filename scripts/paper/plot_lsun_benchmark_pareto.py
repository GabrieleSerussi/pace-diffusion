#!/usr/bin/env python3
"""Plot completed LSUN Bedroom ADM metrics against recorded sampling throughput."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_ffhq_benchmark_pareto import COLORS, LABELS, VARIANTS, Result, pareto_frontier


PROTOCOL = "lsun_bedroom256_openai_adm_v1"
MIXED_PROTOCOL = "lsun_bedroom256_openai_adm_mixed_fp16_v1"
METRICS = {
    "fid_adm": ("FID", False),
    "sfid_adm": ("sFID", False),
    "inception_score_adm": ("Inception Score", True),
    "precision_adm": ("Precision", True),
    "recall_adm": ("Recall", True),
}
SHORT_LABELS = {
    "teacher": "Teacher",
    "global": "Global",
    "uniform_blockwise": "Uniform",
    "combined_blockwise": "Blockwise",
    "combined_layerwise": "Layerwise",
}
OFFSETS = {
    "teacher": (-10, 10),
    "global": (8, 12),
    "uniform_blockwise": (8, 10),
    "combined_blockwise": (-8, -17),
    "combined_layerwise": (8, -17),
}
NOTES = [
    "50,000 samples per model; 40-step EDM Heun (79 NFE); identical image seeds.",
    "Teacher: mixed FP16. Students: FP32 EMA. This is not a precision-matched speed comparison.",
    "Each run used 2 GPUs, batch size 32 per GPU; hardware is recorded in provenance.",
    "Throughput excludes warmup, image writing, and metrics. Student timings cover only the final resumed segment; uneven remaining work across ranks is included.",
    "Single 50k evaluation per model; no confidence intervals or repeat-run error bars.",
    "Best-monitor EMA checkpoint steps are recorded in the tables. Student training budgets can differ.",
    "Frontier lines connect measured points only; they do not assert achievable intermediate models.",
]
MIXED_NOTES = [
    "50,000 samples per model; 40-step EDM Heun (79 NFE); identical image seeds.",
    "Teacher and students all use native mixed FP16; architecture-specific casting policies differ.",
    "Teacher uses its previous completed mixed-FP16 evaluation; original source protocol IDs and SHA256 hashes are preserved.",
    "Each run used 2 matched GPUs, batch size 32 per GPU; hardware is recorded in provenance.",
    "All timing measurements cover 50,000 freshly generated images, with no resumed sample subset. Throughput excludes warmup, image writing and metrics; slowest-rank time is used.",
    "One evaluation per model; no repeat-run uncertainty estimates. Near-equal throughput should not be interpreted as a significant speed difference.",
    "Best-monitor EMA checkpoint steps differ across students and are recorded in the tables.",
    "Frontier lines connect measured points only; they do not assert achievable intermediate models.",
]


def finite_number(value, name: str, *, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return number


def load_results(benchmark_root: Path) -> tuple[list[Result], dict[str, dict]]:
    return _load_result_files(
        {variant: benchmark_root / variant / PROTOCOL / "evaluation_result.json" for variant in VARIANTS},
        {variant: PROTOCOL for variant in VARIANTS},
    )


def load_selected_results(selection_path: Path) -> tuple[list[Result], dict[str, dict]]:
    """Load immutable source artifacts, not the selection file's metric copies."""
    selection = json.loads(selection_path.read_text())
    if selection.get("format") != "diffdist_lsun_mixed_fp16_comparison_selection_v1":
        raise ValueError("Unsupported comparison selection format")
    if selection.get("selection_status") != "complete":
        raise ValueError("Comparison selection is not complete")
    records = selection.get("models", {})
    if set(records) != set(VARIANTS):
        raise ValueError("Comparison selection must contain exactly the teacher and four students")
    paths, protocols = {}, {}
    for variant in VARIANTS:
        record = records[variant]
        path = Path(record["source_result"])
        if not path.is_absolute():
            path = selection_path.parent / path
        if hashlib.sha256(path.read_bytes()).hexdigest() != record.get("source_sha256"):
            raise ValueError(f"{variant}: selected source SHA256 mismatch")
        allowed = {PROTOCOL, MIXED_PROTOCOL} if variant == "teacher" else {MIXED_PROTOCOL}
        if record.get("source_protocol_id") not in allowed:
            raise ValueError(f"{variant}: unsupported selected source protocol")
        if record.get("inference_precision") != "mixed_fp16":
            raise ValueError(f"{variant}: selection must declare mixed_fp16")
        paths[variant], protocols[variant] = path, record["source_protocol_id"]
    results, metadata = _load_result_files(paths, protocols, mixed=True)
    for variant in VARIANTS:
        if metadata[variant]["source_sha256"] != records[variant]["source_sha256"]:
            raise ValueError(f"{variant}: selected source changed while loading")
        metadata[variant]["reused_completed_evaluation"] = bool(records[variant].get("reused_completed_evaluation"))
    return results, metadata


def _load_result_files(
    paths: dict[str, Path], protocols: dict[str, str], *, mixed: bool = False,
) -> tuple[list[Result], dict[str, dict]]:
    results: list[Result] = []
    metadata: dict[str, dict] = {}
    hardware_signatures = []
    comparison_signatures = []
    for variant in VARIANTS:
        path = paths[variant]
        raw = path.read_bytes()
        payload = json.loads(raw)
        if payload.get("status") != "complete":
            raise ValueError(f"{variant}: evaluation is not complete")
        if payload.get("benchmark_protocol_id") != protocols[variant]:
            raise ValueError(f"{variant}: benchmark protocol mismatch")
        model = payload["model"]
        if model.get("architecture_variant") != variant or model.get("weights") != "ema":
            raise ValueError(f"{variant}: model identity or EMA weights mismatch")
        expected_kind = "teacher_network" if variant == "teacher" else "distilled_checkpoint"
        if model.get("kind") != expected_kind:
            raise ValueError(f"{variant}: model kind must be {expected_kind}")
        expected_precision = "mixed_fp16" if mixed or variant == "teacher" else "fp32"
        if model.get("inference_precision") != expected_precision:
            raise ValueError(f"{variant}: expected inference precision {expected_precision}")
        sampling = payload["sampling"]
        for key, expected in (("num_samples", 50000), ("num_steps", 40), ("seed", 2100000)):
            if sampling.get(key) != expected:
                raise ValueError(f"{variant}: sampling {key} must be {expected}")
        if payload["metrics"].get("num_samples") != 50000:
            raise ValueError(f"{variant}: metrics require 50000 samples")
        execution = payload["execution"]
        if execution.get("world_size") != 2 or execution.get("batch_size_per_rank") != 32:
            raise ValueError(f"{variant}: require world_size=2 and batch_size_per_rank=32")
        generation = execution["generation"]
        if not generation.get("newly_generated") or generation.get("generated_images", 0) <= 0:
            raise ValueError(f"{variant}: no newly generated throughput measurement")
        generated = generation["generated_images"]
        reused = generation.get("reused_images", 0)
        if reused < 0 or generated + reused != 50000:
            raise ValueError(f"{variant}: generated and reused image counts must sum to 50000")
        if mixed and (generated != 50000 or reused != 0):
            raise ValueError(f"{variant}: selected comparison requires complete fresh 50000-image timing")
        throughput = finite_number(
            generation.get("model_only_samples_per_second"),
            f"{variant} throughput", positive=True,
        )
        metrics = {
            key: finite_number(payload["metrics"].get(key), f"{variant} {key}")
            for key in METRICS
        }
        hardware = execution.get("hardware", [])
        if mixed:
            if len(hardware) != 2 or any(
                any(row.get(key) is None for key in ("gpu_name", "gpu_total_memory_bytes", "torch_version"))
                for row in hardware
            ):
                raise ValueError(f"{variant}: selected comparison requires two recorded GPU devices")
            if sampling.get("model_precision") != "mixed_fp16":
                raise ValueError(f"{variant}: sampling precision must be mixed_fp16")
            reference, detector = payload.get("reference"), payload.get("detector")
            if not reference or not detector:
                raise ValueError(f"{variant}: selected comparison requires reference and detector provenance")
            if payload.get("metric_backend") != "openai_adm":
                raise ValueError(f"{variant}: expected OpenAI ADM metric backend")
            ranks = generation.get("ranks", [])
            if len(ranks) != 2 or any(row.get("quantizer") != "openai_x_plus_1_x127_5" for row in ranks):
                raise ValueError(f"{variant}: require two rank records with OpenAI truncation quantizer")
            comparison_signatures.append({
                "sampling": {k: v for k, v in sampling.items() if k != "protocol_inference_dtype"},
                "reference": {k: v for k, v in reference.items() if k != "protocol_id"},
                "detector": detector,
                "quantizer": ranks[0]["quantizer"],
            })
        if hardware:
            signature = sorted(
                (row.get("gpu_name"), row.get("gpu_total_memory_bytes"), row.get("torch_version"))
                for row in hardware
            )
            hardware_signatures.append(signature)
        results.append(Result(variant, throughput, throughput, throughput, metrics))
        metadata[variant] = {
            "source_result": str(path.resolve()),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "source_protocol_id": payload["benchmark_protocol_id"],
            "completed_at": payload.get("completed_at"),
            "checkpoint_step": model.get("checkpoint_step"),
            "weights": model["weights"],
            "inference_precision": model["inference_precision"],
            "world_size": execution["world_size"],
            "batch_size_per_rank": execution["batch_size_per_rank"],
            "generated_images_in_timing_segment": generated,
            "reused_images": reused,
            "hardware": hardware,
        }
    if hardware_signatures and any(s != hardware_signatures[0] for s in hardware_signatures):
        raise ValueError("Hardware or Torch versions differ across evaluations")
    if comparison_signatures and any(s != comparison_signatures[0] for s in comparison_signatures):
        raise ValueError("Sampling, reference, detector or quantizer differs across selected evaluations")
    return results, metadata


def frontier_sets(results: list[Result]) -> dict[str, dict[str, list[str]]]:
    students = [row for row in results if row.variant != "teacher"]
    return {
        key: {
            "overall": [r.variant for r in pareto_frontier(results, key, higher)],
            "students_only": [r.variant for r in pareto_frontier(students, key, higher)],
        }
        for key, (_, higher) in METRICS.items()
    }


def legend_handles() -> list[Line2D]:
    handles = [
        Line2D([], [], linestyle="none", marker="*" if v == "teacher" else "o",
               markersize=11 if v == "teacher" else 8, color=COLORS[v], label=LABELS[v])
        for v in VARIANTS
    ]
    handles.extend([
        Line2D([], [], color="#222222", marker="o", markerfacecolor="none",
               linewidth=1.2, label="Overall Pareto frontier"),
        Line2D([], [], color="#777777", linestyle="--", marker="o", markerfacecolor="none",
               linewidth=1.1, label="Students-only frontier"),
    ])
    return handles


def plot_metric(ax, results: list[Result], metric: str) -> None:
    label, higher = METRICS[metric]
    students = [r for r in results if r.variant != "teacher"]
    for subset, style, color in ((results, "-", "#222222"), (students, "--", "#777777")):
        frontier = pareto_frontier(subset, metric, higher)
        ax.plot([r.throughput for r in frontier], [r.metrics[metric] for r in frontier],
                linestyle=style, color=color, linewidth=1.2, alpha=0.8, zorder=1)
        ax.scatter([r.throughput for r in frontier], [r.metrics[metric] for r in frontier],
                   s=195, facecolors="none", edgecolors=color, linewidths=1.1, zorder=2)
    for result in results:
        v = result.variant
        ax.scatter(result.throughput, result.metrics[metric], color=COLORS[v],
                   marker="*" if v == "teacher" else "o", s=130 if v == "teacher" else 65,
                   edgecolors="white", linewidths=0.8, zorder=3)
        offset = OFFSETS[v]
        if metric == "recall_adm" and v == "combined_layerwise":
            offset = (8, 12)
        if metric == "recall_adm" and v == "combined_blockwise":
            offset = (-8, 12)
        ax.annotate(SHORT_LABELS[v], (result.throughput, result.metrics[metric]),
                    xytext=offset, textcoords="offset points", fontsize=8,
                    ha="right" if offset[0] < 0 else "left")
    ax.set_title(f"{label}  ({'higher' if higher else 'lower'} is better)", fontsize=11)
    ax.set_xlabel("Sampling throughput (images/s, 2 GPUs)")
    ax.set_ylabel(label)
    ax.margins(x=0.16, y=0.20)
    ax.grid(True, color="#DDDDDD", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def write_tables(
    results: list[Result], metadata: dict[str, dict], output: Path, *,
    protocol: str = PROTOCOL, notes: list[str] | None = None,
) -> None:
    fronts = frontier_sets(results)
    teacher_rate = next(r.throughput for r in results if r.variant == "teacher")
    rows = []
    for r in results:
        row = {
            "variant": r.variant,
            "checkpoint_step": metadata[r.variant]["checkpoint_step"],
            "inference_precision": metadata[r.variant]["inference_precision"],
            "throughput_images_per_second": r.throughput,
            "speedup_vs_teacher": r.throughput / teacher_rate,
            **r.metrics,
        }
        for key in METRICS:
            row[f"{key}_pareto_overall"] = r.variant in fronts[key]["overall"]
            row[f"{key}_pareto_students"] = r.variant in fronts[key]["students_only"]
        rows.append(row)
    with (output / "pareto_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"protocol": protocol, "notes": NOTES if notes is None else notes, "models": rows,
              "frontiers": fronts, "provenance": metadata}
    (output / "pareto_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def save_figure(fig, output: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--benchmark-root", type=Path, help="Original FP32-student protocol directory.")
    source.add_argument("--selection", type=Path, help="Verified mixed-FP16 comparison-source selection JSON.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    mixed = args.selection is not None
    results, metadata = load_selected_results(args.selection) if mixed else load_results(args.benchmark_root)
    precision_note = "Teacher and students: native mixed FP16." if mixed else "Teacher: mixed FP16; students: FP32 EMA."
    timing_note = "Full 50k-image timing; previous teacher evaluation." if mixed else "Timing uses the final resumed student segment."
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "pdf.fonttype": 42})
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(9.2, 5.3))
        fig.subplots_adjust(left=0.09, right=0.70, bottom=0.18, top=0.84)
        plot_metric(ax, results, metric)
        fig.suptitle("LSUN Bedroom 256: quality–throughput tradeoff", fontsize=14, y=0.97)
        fig.legend(handles=legend_handles(), loc="center left", bbox_to_anchor=(0.72, 0.51),
                   frameon=False, fontsize=8)
        fig.text(0.09, 0.035, "50k samples · 40-step Heun · " + precision_note + "\n" + timing_note
                 + " Single evaluation, no error bars.",
                 fontsize=8, color="#555555")
        save_figure(fig, args.output_dir, f"{metric}_vs_throughput")
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.3))
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.10, top=0.88, wspace=0.27, hspace=0.39)
    for ax, metric in zip(axes.flat, METRICS):
        plot_metric(ax, results, metric)
    notes_ax = axes.flat[-1]
    notes_ax.axis("off")
    notes_ax.legend(handles=legend_handles(), loc="upper left", frameon=False, fontsize=10)
    fronts = frontier_sets(results)
    overview = (
        "Teacher dominates all five metrics in this protocol."
        if all(front["overall"] == ["teacher"] for front in fronts.values())
        else "Solid outlines show the overall Pareto frontier."
    )
    notes_ax.text(0.02, 0.42,
                  overview + "\n"
                  "Dashed lines show the student-only tradeoffs.\n\n"
                  + precision_note + "\n"
                  "Different student training/checkpoint steps.\n"
                  + timing_note,
                  transform=notes_ax.transAxes, va="top", fontsize=8.8,
                  linespacing=1.35, color="#555555")
    fig.suptitle("LSUN Bedroom 256: quality–throughput Pareto tradeoffs", fontsize=18, y=0.97)
    fig.text(0.5, 0.915, "50,000 samples per model · 40-step EDM Heun · batch 32 per GPU · 2 GPUs per model",
             ha="center", fontsize=11, color="#555555")
    fig.text(0.055, 0.025,
             "Model-only timing excludes warmup, image writing and metrics. Rank imbalance is included. "
             "One evaluation per model; no repeat-run uncertainty estimates.", fontsize=9, color="#555555")
    save_figure(fig, args.output_dir, "all_metrics_vs_throughput")
    write_tables(results, metadata, args.output_dir,
                 protocol=MIXED_PROTOCOL if mixed else PROTOCOL, notes=MIXED_NOTES if mixed else NOTES)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()),
                      "frontiers": frontier_sets(results)}, indent=2))


if __name__ == "__main__":
    main()
