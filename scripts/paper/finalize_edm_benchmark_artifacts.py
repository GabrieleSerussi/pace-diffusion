#!/usr/bin/env python3
"""Validate a complete metric suite, then apply one guarded artifact-retention policy."""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
from pathlib import Path


def _load_complete_result(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete" or not isinstance(payload.get("metrics"), dict):
        raise ValueError(f"metric result is not complete: {path}")
    manifest_pointer = payload.get("benchmark_manifest")
    if not isinstance(manifest_pointer, dict):
        raise ValueError(f"metric result has no benchmark manifest: {path}")
    manifest_path = Path(str(manifest_pointer.get("path", ""))).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"declared benchmark manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("manifest_sha256") != manifest_pointer.get("manifest_sha256"):
        raise ValueError(f"benchmark manifest identity mismatch: {manifest_path}")
    if manifest.get("protocol", {}).get("identity") != payload.get("benchmark_protocol_id"):
        raise ValueError(f"benchmark manifest protocol mismatch: {manifest_path}")
    return payload


def _sample_pngs(samples_dir: Path) -> list[Path]:
    if not samples_dir.is_dir():
        raise FileNotFoundError(f"sample directory does not exist: {samples_dir}")
    return sorted(path for path in samples_dir.glob("seed*.png") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--result", action="append", type=Path, required=True, help="Required completed evaluation_result.json; repeat.")
    parser.add_argument("--summary", type=Path, required=True, help="Completed minimum-of-three NVLabs evaluation summary.")
    parser.add_argument("--sample-dir", action="append", type=Path, required=True, help="Explicit disposable sample root; repeat.")
    parser.add_argument("--metric-artifact", action="append", type=Path, default=[], help="Explicit disposable ADM sample NPZ; repeat.")
    parser.add_argument("--retention", choices=("keep", "keep_preview", "discard"), default="keep_preview")
    parser.add_argument("--preview-dir", type=Path, required=True)
    parser.add_argument("--preview-count", type=int, default=64)
    parser.add_argument("--expected-samples-per-dir", type=int, default=50_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.preview_count < 0 or args.expected_samples_per_dir <= 0:
        parser.error("--preview-count must be non-negative and --expected-samples-per-dir must be positive")

    if len(args.result) != 4:
        parser.error("FFHQ finalization requires exactly three NVLabs results and one ADM result")
    results = [_load_complete_result(path.expanduser().resolve()) for path in args.result]
    protocol_counts: dict[str, int] = {}
    for result in results:
        protocol = str(result.get("benchmark_protocol_id"))
        protocol_counts[protocol] = protocol_counts.get(protocol, 0) + 1
    expected_protocol_counts = {
        "ffhq64_nvlabs_edm_fid_v1": 3,
        "ffhq64_openai_adm_custom_first50k_v1": 1,
    }
    if protocol_counts != expected_protocol_counts:
        raise ValueError(
            f"unexpected FFHQ metric suite protocol counts: expected {expected_protocol_counts}, got {protocol_counts}"
        )
    for result in results:
        metrics = result["metrics"]
        if result["benchmark_protocol_id"] == "ffhq64_nvlabs_edm_fid_v1":
            if "fid_nvlabs_legacy" not in metrics:
                raise ValueError("NVLabs result is missing fid_nvlabs_legacy")
        else:
            missing_adm = {"fid_adm", "sfid_adm", "precision_adm", "recall_adm", "inception_score_adm"}.difference(metrics)
            if missing_adm:
                raise ValueError(f"ADM result is missing metrics: {sorted(missing_adm)}")

    nv_results = [
        result for result in results
        if result["benchmark_protocol_id"] == "ffhq64_nvlabs_edm_fid_v1"
    ]
    nv_seed_ranges = {
        (
            int(result.get("sampling", {}).get("seed", -1)),
            int(result.get("sampling", {}).get("num_samples", -1)),
        )
        for result in nv_results
    }
    expected_nv_seed_ranges = {(0, 50_000), (50_000, 50_000), (100_000, 50_000)}
    if nv_seed_ranges != expected_nv_seed_ranges:
        raise ValueError(
            "canonical FFHQ cleanup requires distinct seed/count pairs "
            f"{sorted(expected_nv_seed_ranges)}, got {sorted(nv_seed_ranges)}"
        )
    adm_result = next(
        result for result in results
        if result["benchmark_protocol_id"] == "ffhq64_openai_adm_custom_first50k_v1"
    )
    adm_sampling = adm_result.get("sampling", {})
    if (int(adm_sampling.get("seed", -1)), int(adm_sampling.get("num_samples", -1))) != (0, 50_000):
        raise ValueError("FFHQ ADM cleanup requires seed=0 and num_samples=50000")

    summary_path = args.summary.expanduser().resolve()
    summary = json.loads(summary_path.read_text())
    if (
        summary.get("protocol_id") != "ffhq64_nvlabs_edm_fid_v1"
        or summary.get("metric") != "fid_nvlabs_legacy"
        or summary.get("aggregation") != "minimum"
        or len(summary.get("runs", [])) != 3
    ):
        raise ValueError(f"invalid minimum-of-three NVLabs summary: {summary_path}")
    nv_result_paths = {
        str(path.expanduser().resolve())
        for path, result in zip(args.result, results)
        if result["benchmark_protocol_id"] == "ffhq64_nvlabs_edm_fid_v1"
    }
    summary_result_paths_list = [str(Path(run["path"]).expanduser().resolve()) for run in summary["runs"]]
    summary_result_paths = set(summary_result_paths_list)
    if len(summary_result_paths) != len(summary_result_paths_list):
        raise ValueError("minimum-of-three summary contains duplicate result paths")
    if summary_result_paths != nv_result_paths:
        raise ValueError("minimum-of-three summary does not reference exactly the validated NVLabs results")
    nv_values = {
        str(path.expanduser().resolve()): float(result["metrics"]["fid_nvlabs_legacy"])
        for path, result in zip(args.result, results)
        if result["benchmark_protocol_id"] == "ffhq64_nvlabs_edm_fid_v1"
    }
    for run in summary["runs"]:
        run_path = str(Path(run["path"]).expanduser().resolve())
        if float(run.get("value", float("nan"))) != nv_values[run_path]:
            raise ValueError(f"minimum-of-three summary value does not match validated result: {run_path}")
    expected_minimum = min(nv_values.values())
    if float(summary.get("aggregate", float("nan"))) != expected_minimum:
        raise ValueError(
            f"minimum-of-three summary aggregate must be {expected_minimum}, got {summary.get('aggregate')}"
        )
    expected_selected_index = [float(run["value"]) for run in summary["runs"]].index(expected_minimum)
    if summary.get("selected_run_index") != expected_selected_index:
        raise ValueError(
            "minimum-of-three summary selected_run_index does not identify its first minimum"
        )
    if len(args.sample_dir) != 4:
        parser.error("FFHQ finalization requires exactly four sample directories")
    sample_roots = [path.expanduser().resolve() for path in args.sample_dir]
    samples_by_root = {str(root): _sample_pngs(root) for root in sample_roots}
    bad_counts = {
        root: len(paths)
        for root, paths in samples_by_root.items()
        if len(paths) != args.expected_samples_per_dir
    }
    if bad_counts:
        raise ValueError(
            f"refusing cleanup because sample directories are incomplete; expected "
            f"{args.expected_samples_per_dir} PNGs each, got {bad_counts}"
        )
    metric_artifacts = [path.expanduser().resolve() for path in args.metric_artifact]
    if len(metric_artifacts) != 1:
        parser.error("FFHQ finalization requires exactly one packed ADM metric artifact")
    missing_metric_artifacts = [str(path) for path in metric_artifacts if not path.is_file()]
    if missing_metric_artifacts:
        raise FileNotFoundError(f"declared metric artifacts are missing: {missing_metric_artifacts}")
    previewed: list[str] = []
    deleted: list[str] = []
    if args.retention == "keep_preview":
        args.preview_dir.mkdir(parents=True, exist_ok=True)
        candidates = [path for root in sample_roots for path in samples_by_root[str(root)]]
        for index, source in enumerate(candidates[: args.preview_count]):
            destination = args.preview_dir / f"{index:03d}_{source.parent.name}_{source.name}"
            if destination.exists():
                if not filecmp.cmp(source, destination, shallow=False):
                    raise FileExistsError(f"preview destination exists with different content: {destination}")
            else:
                shutil.copy2(source, destination)
            previewed.append(str(destination.resolve()))

    if args.retention != "keep":
        for root in sample_roots:
            for path in samples_by_root[str(root)]:
                path.unlink()
                deleted.append(str(path))
        for artifact in metric_artifacts:
            if artifact.is_file():
                artifact.unlink()
                deleted.append(str(artifact))

    payload = {
        "cleanup_format": "diffdist_edm_benchmark_cleanup_v1",
        "status": "complete",
        "validated_results": [str(path.expanduser().resolve()) for path in args.result],
        "validated_summary": str(summary_path),
        "validated_result_count": len(results),
        "retention": args.retention,
        "previewed": previewed,
        "deleted_count": len(deleted),
        "deleted": deleted,
        "sample_roots": [str(path) for path in sample_roots],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, args.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
