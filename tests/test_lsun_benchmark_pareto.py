from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


pytest.importorskip("matplotlib")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import plot_lsun_benchmark_pareto as plotter


PROTOCOL = "lsun_bedroom256_openai_adm_v1"
MIXED_PROTOCOL = "lsun_bedroom256_openai_adm_mixed_fp16_v1"


def result_path(root: Path, variant: str) -> Path:
    return root / variant / PROTOCOL / "evaluation_result.json"


def write_result(root: Path, variant: str, payload: dict) -> None:
    path = result_path(root, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


@pytest.fixture
def benchmark(tmp_path: Path) -> Path:
    for index, variant in enumerate(plotter.VARIANTS):
        teacher = variant == "teacher"
        write_result(tmp_path, variant, {
            "status": "complete",
            "benchmark_protocol_id": PROTOCOL,
            "model": {
                "kind": "teacher_network" if teacher else "distilled_checkpoint",
                "architecture_variant": variant,
                "weights": "ema",
                "checkpoint_step": None if teacher else 90000,
                "inference_precision": "mixed_fp16" if teacher else "fp32",
            },
            "sampling": {"num_samples": 50000, "num_steps": 40, "seed": 2100000},
            "execution": {
                "world_size": 2,
                "batch_size_per_rank": 32,
                "total_wall_seconds": 100000,
                "generation": {
                    "newly_generated": True,
                    "generated_images": 100,
                    "reused_images": 49900,
                    "model_only_samples_per_second": 7.0 + index,
                },
            },
            "metrics": {"num_samples": 50000, **{key: 1.0 for key in plotter.METRICS}},
        })
    return tmp_path


def change_field(root: Path, keys: tuple[str, ...], value, *, variant: str = "global") -> None:
    payload = json.loads(result_path(root, variant).read_text())
    target = payload
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    write_result(root, variant, payload)


def point(variant: str, throughput: float, quality: float) -> plotter.Result:
    return plotter.Result(
        variant=variant,
        throughput=throughput,
        throughput_min=throughput,
        throughput_max=throughput,
        metrics={"quality": quality},
    )


@pytest.mark.parametrize("higher", [False, True])
def test_pareto_direction_and_exact_ties(higher: bool) -> None:
    sign = -1 if higher else 1
    points = [
        point("fast", 3.0, sign * 2.0),
        point("dominated", 1.0, sign * 3.0),
        point("best_quality", 1.0, sign * 1.0),
        point("fast_twin", 3.0, sign * 2.0),
        point("equal_quality_slower", 2.0, sign * 2.0),
        point("equal_speed_worse", 3.0, sign * 3.0),
    ]
    frontier = plotter.pareto_frontier(points, "quality", higher)
    assert [item.variant for item in frontier] == ["best_quality", "fast", "fast_twin"]


def test_metric_directions_are_canonical() -> None:
    assert {key: higher for key, (_, higher) in plotter.METRICS.items()} == {
        "fid_adm": False,
        "sfid_adm": False,
        "inception_score_adm": True,
        "precision_adm": True,
        "recall_adm": True,
    }


def test_load_complete_results_uses_fresh_model_rate_for_resumed_runs(benchmark: Path) -> None:
    results, metadata = plotter.load_results(benchmark)
    assert [result.variant for result in results] == list(plotter.VARIANTS)
    assert set(metadata) == set(plotter.VARIANTS)
    for index, result in enumerate(results):
        assert result.throughput == 7.0 + index
        assert result.throughput_min == result.throughput_max == result.throughput
        assert result.throughput != 50000 / 100000
        assert set(result.metrics) == set(plotter.METRICS)


@pytest.mark.parametrize(("keys", "value"), [
    (("status",), "running"),
    (("benchmark_protocol_id",), "ffhq64_openai_adm_custom_first50k_v1"),
    (("sampling", "num_samples"), 49999),
    (("sampling", "num_steps"), 18),
    (("sampling", "seed"), 42),
    (("metrics", "num_samples"), 49999),
    (("execution", "world_size"), 4),
    (("execution", "batch_size_per_rank"), 16),
    (("model", "weights"), "student"),
    (("model", "inference_precision"), "fp16"),
    (("model", "architecture_variant"), "combined_blockwise"),
    (("model", "kind"), "teacher_network"),
])
def test_load_rejects_incomplete_or_incomparable_results(
    benchmark: Path, keys: tuple[str, ...], value,
) -> None:
    change_field(benchmark, keys, value)
    with pytest.raises(ValueError):
        plotter.load_results(benchmark)


def test_load_rejects_wrong_teacher_precision(benchmark: Path) -> None:
    change_field(benchmark, ("model", "inference_precision"), "fp32", variant="teacher")
    with pytest.raises(ValueError):
        plotter.load_results(benchmark)


@pytest.mark.parametrize("value", [None, 0, -0.5, float("nan"), float("inf"), -float("inf")])
def test_load_rejects_invalid_throughput(benchmark: Path, value) -> None:
    change_field(benchmark, ("execution", "generation", "model_only_samples_per_second"), value)
    with pytest.raises(ValueError):
        plotter.load_results(benchmark)


def test_load_rejects_missing_throughput(benchmark: Path) -> None:
    payload = json.loads(result_path(benchmark, "global").read_text())
    del payload["execution"]["generation"]["model_only_samples_per_second"]
    write_result(benchmark, "global", payload)
    with pytest.raises(ValueError):
        plotter.load_results(benchmark)


@pytest.mark.parametrize(("key", "value"), [
    ("newly_generated", False),
    ("generated_images", 0),
    ("generated_images", 99),
    ("reused_images", -1),
])
def test_load_rejects_absent_fresh_timing_or_inconsistent_counts(
    benchmark: Path, key: str, value,
) -> None:
    change_field(benchmark, ("execution", "generation", key), value)
    with pytest.raises(ValueError):
        plotter.load_results(benchmark)


@pytest.mark.parametrize("metric", [
    "fid_adm", "sfid_adm", "inception_score_adm", "precision_adm", "recall_adm",
])
@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -float("inf")])
def test_load_rejects_nonfinite_metrics(benchmark: Path, metric: str, value) -> None:
    change_field(benchmark, ("metrics", metric), value)
    with pytest.raises(ValueError):
        plotter.load_results(benchmark)


@pytest.fixture
def selected_benchmark(benchmark: Path) -> Path:
    """Fresh mixed-FP16 students combined with the completed old-protocol teacher."""
    selection = {
        "format": "diffdist_lsun_mixed_fp16_comparison_selection_v1",
        "selection_status": "complete",
        "models": {},
    }
    for variant in plotter.VARIANTS:
        teacher = variant == "teacher"
        protocol = PROTOCOL if teacher else MIXED_PROTOCOL
        payload = json.loads(result_path(benchmark, variant).read_text())
        payload["benchmark_protocol_id"] = protocol
        payload["metric_backend"] = "openai_adm"
        payload["model"]["inference_precision"] = "mixed_fp16"
        payload["sampling"].update({
            "clip_denoised": True,
            "label_mode": "seed_modulo",
            "model_precision": "mixed_fp16",
            "nfe_per_image": 79,
            "protocol_inference_dtype": (
                "teacher_mixed_fp16_or_student_fp32_with_float64_sampler_state" if teacher
                else "teacher_mixed_fp16_and_student_mixed_fp16_with_float64_sampler_state"
            ),
            "rho": 7.0,
            "s_churn": 0.0,
            "s_max": None,
            "s_min": 0.0,
            "s_noise": 1.0,
            "sigma_max": 80.0,
            "sigma_min": 0.002,
        })
        payload["execution"]["generation"].update({
            "generated_images": 50000,
            "reused_images": 0,
            "assigned_images": 50000,
            "warmup_samples": 64,
            "ranks": [
                {
                    "rank": rank,
                    "assigned_count": 25000,
                    "generated_count": 25000,
                    "skipped_count": 0,
                    "warmup_samples": 32,
                    "world_size": 2,
                    "quantizer": "openai_x_plus_1_x127_5",
                }
                for rank in range(2)
            ],
        })
        payload["execution"]["hardware"] = [
            {
                "rank": rank,
                "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
                "gpu_total_memory_bytes": 101973491712,
                "torch_version": "2.10.0+cu128",
                "cuda_capability": [12, 0],
                "device": f"cuda:{rank}",
                "platform": "test-linux",
            }
            for rank in range(2)
        ]
        payload["reference"] = {
            "backend": "openai_adm",
            "protocol_id": protocol,
            "path": "/reference/VIRTUAL_lsun_bedroom256.npz",
            "size_bytes": 1054074338,
        }
        payload["detector"] = {
            "md5": "e6bb154e85f5d4331c22abcddb5dcf31",
            "path": "/reference/classify_image_graph_def.pb",
            "size_bytes": 95673916,
        }
        path = benchmark / "selected_sources" / variant / protocol / "evaluation_result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(payload).encode()
        path.write_bytes(raw)
        selection["models"][variant] = {
            "source_result": str(path),
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "source_protocol_id": protocol,
            "inference_precision": "mixed_fp16",
            "reused_completed_evaluation": teacher,
        }
    selection_path = benchmark / "selected_results.json"
    selection_path.write_text(json.dumps(selection))
    return selection_path


def change_selected_payload(
    selection_path: Path, keys: tuple[str | int, ...], value, *, variant: str = "global",
) -> None:
    selection = json.loads(selection_path.read_text())
    entry = selection["models"][variant]
    path = Path(entry["source_result"])
    payload = json.loads(path.read_text())
    target = payload
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    raw = json.dumps(payload).encode()
    path.write_bytes(raw)
    entry["source_sha256"] = hashlib.sha256(raw).hexdigest()
    selection_path.write_text(json.dumps(selection))


def test_selection_loads_prior_teacher_and_fresh_mixed_students(selected_benchmark: Path) -> None:
    results, metadata = plotter.load_selected_results(selected_benchmark)
    assert [row.variant for row in results] == list(plotter.VARIANTS)
    assert set(metadata) == set(plotter.VARIANTS)
    for variant in plotter.VARIANTS:
        teacher = variant == "teacher"
        assert metadata[variant]["source_protocol_id"] == (PROTOCOL if teacher else MIXED_PROTOCOL)
        assert metadata[variant]["reused_completed_evaluation"] is teacher
        assert metadata[variant]["inference_precision"] == "mixed_fp16"
        assert metadata[variant]["generated_images_in_timing_segment"] == 50000
        assert metadata[variant]["reused_images"] == 0


def test_selection_can_use_a_completed_new_protocol_teacher(selected_benchmark: Path) -> None:
    change_selected_payload(
        selected_benchmark, ("benchmark_protocol_id",), MIXED_PROTOCOL, variant="teacher",
    )
    change_selected_payload(
        selected_benchmark, ("reference", "protocol_id"), MIXED_PROTOCOL, variant="teacher",
    )
    change_selected_payload(
        selected_benchmark, ("sampling", "protocol_inference_dtype"),
        "teacher_mixed_fp16_and_student_mixed_fp16_with_float64_sampler_state", variant="teacher",
    )
    selection = json.loads(selected_benchmark.read_text())
    selection["models"]["teacher"]["source_protocol_id"] = MIXED_PROTOCOL
    selection["models"]["teacher"]["reused_completed_evaluation"] = False
    selected_benchmark.write_text(json.dumps(selection))
    _, metadata = plotter.load_selected_results(selected_benchmark)
    assert metadata["teacher"]["source_protocol_id"] == MIXED_PROTOCOL
    assert metadata["teacher"]["reused_completed_evaluation"] is False


@pytest.mark.parametrize(("keys", "value"), [
    (("format",), "unknown"),
    (("selection_status",), "incomplete"),
    (("models", "global", "source_sha256"), "0" * 64),
    (("models", "global", "source_protocol_id"), PROTOCOL),
    (("models", "global", "inference_precision"), "fp32"),
])
def test_selection_rejects_invalid_selection_metadata(
    selected_benchmark: Path, keys: tuple[str, ...], value,
) -> None:
    selection = json.loads(selected_benchmark.read_text())
    target = selection
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    selected_benchmark.write_text(json.dumps(selection))
    with pytest.raises(ValueError):
        plotter.load_selected_results(selected_benchmark)


def test_selection_rejects_a_missing_variant(selected_benchmark: Path) -> None:
    selection = json.loads(selected_benchmark.read_text())
    del selection["models"]["global"]
    selected_benchmark.write_text(json.dumps(selection))
    with pytest.raises(ValueError):
        plotter.load_selected_results(selected_benchmark)


@pytest.mark.parametrize(("keys", "value"), [
    (("status",), "running"),
    (("benchmark_protocol_id",), "unknown_protocol"),
    (("model", "inference_precision"), "fp32"),
    (("model", "architecture_variant"), "teacher"),
    (("model", "weights"), "student"),
    (("sampling", "rho"), 3.0),
    (("sampling", "sigma_max"), 90.0),
    (("metric_backend",), "other"),
    (("sampling", "clip_denoised"), False),
    (("sampling", "model_precision"), "fp32"),
    (("sampling", "seed"), 42),
    (("sampling", "num_samples"), 49999),
    (("metrics", "num_samples"), 49999),
    (("reference", "size_bytes"), 1),
    (("reference", "backend"), "other"),
    (("detector", "md5"), "0" * 32),
    (("execution", "hardware", 0, "gpu_name"), "Other GPU"),
    (("execution", "hardware", 0, "torch_version"), "1.0"),
    (("execution", "hardware"), []),
    (("execution", "world_size"), 1),
    (("execution", "batch_size_per_rank"), 16),
    (("execution", "generation", "newly_generated"), False),
    (("execution", "generation", "generated_images"), 40000),
    (("execution", "generation", "reused_images"), 10000),
    (("execution", "generation", "ranks"), []),
    (("execution", "generation", "ranks", 0, "quantizer"), "rounded_uint8"),
])
def test_selection_rejects_incompatible_source_results(
    selected_benchmark: Path, keys: tuple[str | int, ...], value,
) -> None:
    change_selected_payload(selected_benchmark, keys, value)
    with pytest.raises(ValueError):
        plotter.load_selected_results(selected_benchmark)


def test_selection_rejects_single_rank_hardware(selected_benchmark: Path) -> None:
    selection = json.loads(selected_benchmark.read_text())
    payload = json.loads(Path(selection["models"]["global"]["source_result"]).read_text())
    change_selected_payload(
        selected_benchmark, ("execution", "hardware"), payload["execution"]["hardware"][:1],
    )
    with pytest.raises(ValueError):
        plotter.load_selected_results(selected_benchmark)


def test_selection_report_preserves_protocol_and_reused_teacher_provenance(
    selected_benchmark: Path, tmp_path: Path,
) -> None:
    results, metadata = plotter.load_selected_results(selected_benchmark)
    plotter.write_tables(
        results, metadata, tmp_path, protocol=MIXED_PROTOCOL, notes=plotter.MIXED_NOTES,
    )
    report = json.loads((tmp_path / "pareto_metrics.json").read_text())
    assert report["protocol"] == MIXED_PROTOCOL
    assert report["notes"] == plotter.MIXED_NOTES
    assert report["provenance"]["teacher"]["source_protocol_id"] == PROTOCOL
    assert report["provenance"]["teacher"]["reused_completed_evaluation"] is True
    assert {row["inference_precision"] for row in report["models"]} == {"mixed_fp16"}


def test_legacy_report_defaults_are_unchanged(benchmark: Path, tmp_path: Path) -> None:
    results, metadata = plotter.load_results(benchmark)
    plotter.write_tables(results, metadata, tmp_path)
    report = json.loads((tmp_path / "pareto_metrics.json").read_text())
    assert report["protocol"] == PROTOCOL
    assert report["notes"] == plotter.NOTES
    assert {row["inference_precision"] for row in report["models"]} == {"fp32", "mixed_fp16"}


def test_selection_uses_source_metrics_not_selection_copies(selected_benchmark: Path) -> None:
    selection = json.loads(selected_benchmark.read_text())
    selection["models"]["global"]["metrics"] = {"fid_adm": 9999}
    selection["models"]["global"]["throughput_images_per_second_2_gpus"] = 9999
    selected_benchmark.write_text(json.dumps(selection))
    results, _ = plotter.load_selected_results(selected_benchmark)
    result = next(row for row in results if row.variant == "global")
    assert result.metrics["fid_adm"] == 1.0
    assert result.throughput != 9999


def test_selection_resolves_relative_paths_beside_manifest(selected_benchmark: Path) -> None:
    selection = json.loads(selected_benchmark.read_text())
    for entry in selection["models"].values():
        entry["source_result"] = str(Path(entry["source_result"]).relative_to(selected_benchmark.parent))
    selected_benchmark.write_text(json.dumps(selection))
    results, metadata = plotter.load_selected_results(selected_benchmark)
    assert len(results) == 5
    assert all(Path(row["source_result"]).is_absolute() for row in metadata.values())
