#!/usr/bin/env python3
"""Compare deterministic seed replicates for a DiffWave per-filter pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_result(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Stability input must contain a JSON object: {source}")
    if payload.get("config", {}).get("grouping") != "per_filter":
        raise ValueError(f"Stability input is not a per-filter profile: {source}")
    if not (source.parent / "_SUCCESS.json").is_file():
        raise ValueError(f"Stability input is incomplete (missing _SUCCESS.json): {source}")
    return source, payload


def finite_matrix(payload: Mapping[str, Any], key: str) -> np.ndarray:
    matrix = np.asarray(payload[key], dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError(f"{key} must be a finite two-dimensional matrix")
    return matrix


def safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape:
        raise ValueError("Stability vectors must have identical shapes")
    if first.size < 2:
        return 1.0
    value = float(spearmanr(first, second).statistic)
    return 0.0 if not np.isfinite(value) else value


def top_quartile_jaccard(first: np.ndarray, second: np.ndarray) -> float:
    count = max(1, int(np.ceil(first.size * 0.25)))
    first_top = set(np.argsort(-first, kind="stable")[:count].tolist())
    second_top = set(np.argsort(-second, kind="stable")[:count].tolist())
    return len(first_top & second_top) / len(first_top | second_top)


def pair_record(
    first_label: str,
    first: Mapping[str, Any],
    second_label: str,
    second: Mapping[str, Any],
) -> dict[str, Any]:
    first_signed = finite_matrix(first, "signed_delta_stack")
    second_signed = finite_matrix(second, "signed_delta_stack")
    first_relative = finite_matrix(first, "relative_delta_stack")
    second_relative = finite_matrix(second, "relative_delta_stack")
    per_bin = [
        safe_spearman(first_signed[:, index], second_signed[:, index])
        for index in range(first_signed.shape[1])
    ]
    first_importance = first_relative.sum(axis=1)
    second_importance = second_relative.sum(axis=1)
    aggregates: dict[str, Any] = {}
    for view in ("module", "stage"):
        first_view = first["per_filter_aggregates"][view]
        second_view = second["per_filter_aggregates"][view]
        if first_view["names"] != second_view["names"]:
            raise ValueError(f"{view} aggregate names differ across stability runs")
        first_matrix = np.asarray(first_view["signed_delta_stack"], dtype=np.float64)
        second_matrix = np.asarray(second_view["signed_delta_stack"], dtype=np.float64)
        aggregates[view] = {
            "flattened_signed_spearman": safe_spearman(
                first_matrix.reshape(-1), second_matrix.reshape(-1)
            ),
            "sign_agreement": float(
                np.mean(np.sign(first_matrix) == np.sign(second_matrix))
            ),
        }
    return {
        "first": first_label,
        "second": second_label,
        "signed_delta_flattened_spearman": safe_spearman(
            first_signed.reshape(-1), second_signed.reshape(-1)
        ),
        "signed_delta_per_bin_spearman": per_bin,
        "signed_delta_per_bin_median_spearman": float(np.median(per_bin)),
        "signed_delta_sign_agreement": float(
            np.mean(np.sign(first_signed) == np.sign(second_signed))
        ),
        "filter_importance_spearman": safe_spearman(
            first_importance, second_importance
        ),
        "top_quartile_filter_jaccard": top_quartile_jaccard(
            first_importance, second_importance
        ),
        "aggregates": aggregates,
    }


def compare_results(paths: Sequence[str | Path]) -> dict[str, Any]:
    if len(paths) != 3:
        raise ValueError("The stability audit requires exactly three seed results")
    loaded = [load_result(path) for path in paths]
    payloads = [item[1] for item in loaded]
    reference = payloads[0]
    invariant_keys = (
        "group_names",
        "group_module_paths",
        "group_filter_indices",
        "group_stage_keys",
        "timestep_bin_labels",
        "full_group_count",
    )
    for payload in payloads[1:]:
        for key in invariant_keys:
            if payload.get(key) != reference.get(key):
                raise ValueError(f"Stability runs differ in invariant field {key!r}")
        for section, key in (
            ("teacher", "checkpoint_sha256"),
            ("dataset", "population_fingerprint"),
            ("groups", "full_catalog_sha256"),
        ):
            if (
                payload["profile_fingerprint"][section].get(key)
                != reference["profile_fingerprint"][section].get(key)
            ):
                raise ValueError(f"Stability runs differ in {section}.{key}")
    if len(reference["group_names"]) != 256:
        raise ValueError("The stability audit requires exactly 256 stratified filters")
    labels = [f"seed{int(payload['config']['seed'])}" for payload in payloads]
    pairs = [
        pair_record(labels[first], payloads[first], labels[second], payloads[second])
        for first, second in combinations(range(3), 2)
    ]
    return {
        "format": "diffdist_diffwave_per_filter_stability_v1",
        "passed": True,
        "interpretation": (
            "This stratified subset audit quantifies seed sensitivity; it does not "
            "upgrade the single-seed full-profile filter rankings to replicated estimates."
        ),
        "inputs": [
            {
                "path": str(source),
                "sha256": file_sha256(source),
                "seed": int(payload["config"]["seed"]),
                "pfi_seed": int(payload["config"]["pfi_seed"]),
                "positive_fraction": payload["signed_delta_positive_fraction"],
                "negative_fraction": payload["signed_delta_negative_fraction"],
            }
            for source, payload in loaded
        ],
        "group_count": 256,
        "num_bins": len(reference["timestep_bin_labels"]),
        "pairs": pairs,
    }


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(dict(payload), indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_plot(path: Path, report: Mapping[str, Any]) -> None:
    labels = [f"{pair['first']}–{pair['second']}" for pair in report["pairs"]]
    values = [pair["signed_delta_per_bin_median_spearman"] for pair in report["pairs"]]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values)
    plt.ylim(-1.0, 1.0)
    plt.ylabel("Median bin-wise Spearman correlation")
    plt.title("DiffWave per-filter seed stability (256-filter subset)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plot-output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare_results(args.results)
    output = Path(args.output).expanduser().resolve()
    plot = Path(args.plot_output).expanduser().resolve()
    atomic_write_json(output, report)
    plot.parent.mkdir(parents=True, exist_ok=True)
    save_plot(plot, report)
    print(json.dumps({"report": str(output), "plot": str(plot), "passed": True}, indent=2))


if __name__ == "__main__":
    main()
