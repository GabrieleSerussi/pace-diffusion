import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import evaluate_edm_checkpoint as evaluate_module
from pace.edm_distillation import NarrowEDMPrecond, NarrowVPPrecond
from pace.evaluation_protocols import (
    LSUN_BEDROOM256_ADM_MIXED_FP16_PROTOCOL,
    LSUN_BEDROOM256_ADM_PROTOCOL,
)


def make_exhaustive_sampling() -> dict:
    names = ["m.filter_0"]
    digest = hashlib.sha256(
        json.dumps(names, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "format": "diffdist_edm_filter_sampling_protocol_v1",
        "protocol_id": "per_filter_exhaustive_v1",
        "mode": "exhaustive",
        "seed": 0,
        "filters_per_module": None,
        "population_group_count": 1,
        "selected_group_count": 1,
        "population_module_count": 1,
        "selected_module_count": 1,
        "module_counts": {
            "m": {
                "population_filter_count": 1,
                "selected_filter_count": 1,
                "inclusion_probability": 1.0,
                "expansion_weight": 1.0,
            }
        },
        "selection_sha256": digest,
        "population_sha256": digest,
    }


class TinyEvaluationNet(torch.nn.Module):
    img_resolution = 4
    img_channels = 3
    label_dim = 2

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))


def make_sampling_config() -> dict:
    return {
        "num_steps": 2,
        "sigma_min": 0.002,
        "sigma_max": 1.0,
        "rho": 7.0,
        "s_churn": 0.0,
        "s_min": 0.0,
        "s_max": None,
        "s_noise": 1.0,
        "label_mode": "seed_modulo",
    }


def make_args(**overrides) -> argparse.Namespace:
    values = {
        "num_samples": 8,
        "seed": 0,
        "num_steps": 2,
        "sigma_min": 0.002,
        "sigma_max": 1.0,
        "rho": 7.0,
        "s_churn": 0.0,
        "s_min": 0.0,
        "s_max": None,
        "s_noise": 1.0,
        "label_mode": "seed_modulo",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def write_reference_stats(path: Path) -> None:
    np.savez_compressed(
        path,
        mu=np.zeros(2),
        sigma=np.eye(2),
        metadata=json.dumps({"dataset": "tiny", "num_reference_images": 8}),
    )


@pytest.mark.parametrize(("weights", "expected"), [("ema", 2.0), ("student", 1.0)])
def test_load_distilled_checkpoint_selects_requested_weights(monkeypatch, tmp_path, weights, expected):
    monkeypatch.setattr(evaluate_module, "construct_student_from_plan", lambda _plan: TinyEvaluationNet())
    checkpoint = tmp_path / "student.pt"
    torch.save(
        {
            "snapshot_format": "test",
            "step": 17,
            "architecture_plan": {
                "variant": "tiny",
                "num_sigma_bins": 1,
                "source_profile": {
                    "format": "diffdist_edm_source_profile_v1",
                    "results_json_path": "/profiles/pfi/results.json",
                    "results_sha256": "a" * 64,
                    "profile_fingerprint": {"pfi_seed": 0},
                    "profile_fingerprint_sha256": "b" * 64,
                },
                "ablation_protocol": {
                    "protocol_id": "batch_local_exact_sigma_pfi_v1",
                    "ablation_mode": "pfi",
                },
                "filter_sampling": make_exhaustive_sampling(),
                "students": [
                    {
                        "timestep_block": [0, 1],
                        "full_parameter_count": 1,
                    }
                ],
            },
            "student_state_dict": {"weight": torch.tensor(1.0)},
            "ema_state_dict": {"weight": torch.tensor(2.0)},
        },
        checkpoint,
    )

    network, _plan, metadata = evaluate_module.load_distilled_checkpoint(
        checkpoint,
        weights=weights,
        device=torch.device("cpu"),
    )

    assert network.weight.item() == pytest.approx(expected)
    assert not network.training
    assert not network.weight.requires_grad
    assert metadata["checkpoint_step"] == 17
    assert metadata["architecture_variant"] == "tiny"
    assert metadata["average_active_parameters"] == pytest.approx(1.0)
    assert metadata["source_profile"]["results_sha256"] == "a" * 64
    assert metadata["ablation_protocol"]["protocol_id"] == "batch_local_exact_sigma_pfi_v1"
    assert metadata["filter_sampling"] == make_exhaustive_sampling()
    assert metadata["inference_precision"] == "fp32"
    assert metadata["parameter_dtype"] == "float32"


def tiny_precision_wrapper(wrapper_type=NarrowEDMPrecond, *, use_fp16=False):
    """Exercise the real wrapper type without loading EDM or allocating a UNet."""
    wrapper = wrapper_type.__new__(wrapper_type)
    torch.nn.Module.__init__(wrapper)
    wrapper.use_fp16 = use_fp16
    wrapper.register_parameter("weight", torch.nn.Parameter(torch.ones(())))
    return wrapper


def test_configure_student_precision_updates_every_expert_without_casting_weights():
    first = tiny_precision_wrapper()
    second = tiny_precision_wrapper(NarrowVPPrecond)
    routed = torch.nn.ModuleList([first, second])
    original_parameters = list(routed.parameters())

    # A CUDA device descriptor does not allocate CUDA tensors: configuration
    # changes native wrapper flags only, so this test also runs on CPU hosts.
    evaluate_module.configure_student_precision(
        routed, dtype=torch.float16, device=torch.device("cuda")
    )

    assert all(wrapper.use_fp16 for wrapper in routed)
    assert all(parameter.dtype is torch.float32 for parameter in routed.parameters())
    assert all(parameter.device.type == "cpu" for parameter in routed.parameters())
    assert all(
        actual is original
        for actual, original in zip(routed.parameters(), original_parameters)
    )

    evaluate_module.configure_student_precision(
        routed, dtype=torch.float32, device=torch.device("cpu")
    )
    assert not any(wrapper.use_fp16 for wrapper in routed)


def test_configure_student_precision_rejects_fp16_on_cpu():
    network = tiny_precision_wrapper()
    with pytest.raises(ValueError, match="CUDA"):
        evaluate_module.configure_student_precision(
            network, dtype=torch.float16, device=torch.device("cpu")
        )
    assert not network.use_fp16


def test_configure_student_precision_rejects_unsupported_fp16_network():
    network = TinyEvaluationNet()
    with pytest.raises(ValueError):
        evaluate_module.configure_student_precision(
            network, dtype=torch.float16, device=torch.device("cuda")
        )

    # Existing FP32 checkpoint types are still accepted.
    evaluate_module.configure_student_precision(
        network, dtype=torch.float32, device=torch.device("cpu")
    )


def test_load_distilled_checkpoint_records_native_mixed_fp16(monkeypatch, tmp_path):
    network = tiny_precision_wrapper()
    monkeypatch.setattr(evaluate_module, "construct_student_from_plan", lambda _plan: network)
    moves = []

    def fake_to(*args, **kwargs):
        moves.append((args, kwargs))
        return network

    monkeypatch.setattr(network, "to", fake_to)
    checkpoint = tmp_path / "student.pt"
    torch.save(
        {
            "step": 19,
            "architecture_plan": {
                "variant": "global",
                "num_sigma_bins": 1,
                "students": [{"timestep_block": [0, 1], "full_parameter_count": 1}],
            },
            "ema_state_dict": {"weight": torch.tensor(2.0)},
        },
        checkpoint,
    )

    loaded, _plan, metadata = evaluate_module.load_distilled_checkpoint(
        checkpoint,
        weights="ema",
        device=torch.device("cuda"),
        dtype=torch.float16,
    )

    assert loaded is network
    assert loaded.use_fp16
    assert loaded.weight.item() == pytest.approx(2.0)
    assert not loaded.training
    assert not loaded.weight.requires_grad
    assert loaded.weight.dtype is torch.float32
    assert moves == [((), {"device": torch.device("cuda"), "dtype": torch.float32})]
    assert metadata["inference_precision"] == "mixed_fp16"
    assert metadata["parameter_dtype"] == "float32"
    assert metadata["checkpoint_step"] == 19


@pytest.mark.parametrize(
    ("requested", "protocol", "device", "expected"),
    [
        ("auto", None, "cpu", torch.float32),
        ("auto", None, "cuda", torch.float32),
        ("auto", LSUN_BEDROOM256_ADM_PROTOCOL, "cuda", torch.float32),
        ("auto", LSUN_BEDROOM256_ADM_MIXED_FP16_PROTOCOL, "cuda", torch.float16),
        ("fp32", LSUN_BEDROOM256_ADM_MIXED_FP16_PROTOCOL, "cuda", torch.float32),
        ("fp16", None, "cuda", torch.float16),
    ],
)
def test_resolve_student_dtype(requested, protocol, device, expected):
    assert evaluate_module.resolve_student_dtype(
        argparse.Namespace(student_dtype=requested), protocol, device=torch.device(device)
    ) is expected


def test_resolve_student_dtype_legacy_namespace_defaults_to_fp32():
    assert evaluate_module.resolve_student_dtype(
        argparse.Namespace(), LSUN_BEDROOM256_ADM_PROTOCOL, device=torch.device("cuda")
    ) is torch.float32


def test_resolve_student_dtype_rejects_automatic_mixed_fp16_on_cpu():
    with pytest.raises(ValueError, match="CUDA"):
        evaluate_module.resolve_student_dtype(
            argparse.Namespace(student_dtype="auto"),
            LSUN_BEDROOM256_ADM_MIXED_FP16_PROTOCOL,
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize("requested", ["fp16", "bf16", "unknown"])
def test_resolve_student_dtype_rejects_invalid_request(requested):
    with pytest.raises(ValueError):
        evaluate_module.resolve_student_dtype(
            argparse.Namespace(student_dtype=requested), None, device=torch.device("cpu")
        )


@pytest.mark.parametrize(
    ("protocol", "kind", "precision"),
    [
        (LSUN_BEDROOM256_ADM_PROTOCOL, "distilled_checkpoint", "fp32"),
        (LSUN_BEDROOM256_ADM_PROTOCOL, "teacher_network", "mixed_fp16"),
        (LSUN_BEDROOM256_ADM_MIXED_FP16_PROTOCOL, "distilled_checkpoint", "mixed_fp16"),
        (LSUN_BEDROOM256_ADM_MIXED_FP16_PROTOCOL, "teacher_network", "mixed_fp16"),
    ],
)
def test_validate_model_precision_accepts_matching_protocol(protocol, kind, precision):
    evaluate_module.validate_model_precision(
        {"kind": kind, "inference_precision": precision}, protocol
    )


@pytest.mark.parametrize(
    ("protocol", "kind", "precision"),
    [
        (LSUN_BEDROOM256_ADM_PROTOCOL, "distilled_checkpoint", "mixed_fp16"),
        (LSUN_BEDROOM256_ADM_PROTOCOL, "teacher_network", "fp32"),
        (LSUN_BEDROOM256_ADM_MIXED_FP16_PROTOCOL, "distilled_checkpoint", "fp32"),
        (LSUN_BEDROOM256_ADM_MIXED_FP16_PROTOCOL, "teacher_network", "fp32"),
        (LSUN_BEDROOM256_ADM_MIXED_FP16_PROTOCOL, "reused_samples", "artifact_only_unknown"),
    ],
)
def test_validate_model_precision_rejects_mismatched_protocol(protocol, kind, precision):
    with pytest.raises(ValueError):
        evaluate_module.validate_model_precision(
            {"kind": kind, "inference_precision": precision}, protocol
        )


def test_validate_model_precision_does_not_relabel_reused_fp32_samples():
    metadata = {
        "kind": "distilled_checkpoint",
        "inference_precision": "fp32",
        "reuse_provenance": {"source": "original-fp32-evaluation.json"},
    }
    with pytest.raises(ValueError):
        evaluate_module.validate_model_precision(
            metadata, LSUN_BEDROOM256_ADM_MIXED_FP16_PROTOCOL
        )
    assert metadata["inference_precision"] == "fp32"


@pytest.mark.parametrize("requested", [None, "auto", "fp32", "fp16"])
def test_student_dtype_cli_accepts_native_precision_options(monkeypatch, tmp_path, requested):
    argv = [
        "evaluate_edm_checkpoint.py",
        "--checkpoint", str(tmp_path / "student.pt"),
        "--output-dir", str(tmp_path / "evaluation"),
        "--reference-stats", str(tmp_path / "reference.npz"),
        "--num-samples", "50000",
    ]
    if requested is not None:
        argv.extend(["--student-dtype", requested])
    monkeypatch.setattr(sys, "argv", argv)
    # Parsing should not initialize CUDA merely to exercise this CLI option.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert evaluate_module.parse_args().student_dtype == (requested or "auto")


def test_load_distilled_checkpoint_rejects_training_state(tmp_path):
    checkpoint = tmp_path / "training-state.pt"
    torch.save({"step": 4, "optimizer_state_dict": {}}, checkpoint)

    with pytest.raises(ValueError, match="architecture_plan"):
        evaluate_module.load_distilled_checkpoint(
            checkpoint,
            weights="ema",
            device=torch.device("cpu"),
        )


def test_reference_metadata_and_protocol_config_are_stable(tmp_path):
    reference = tmp_path / "reference.npz"
    write_reference_stats(reference)
    metadata = evaluate_module.read_reference_metadata(reference)
    model = {
        "kind": "distilled_checkpoint",
        "source": "/tmp/student.pt",
        "checkpoint_step": 12,
        "weights": "ema",
    }

    first = evaluate_module.build_evaluation_config(
        model_metadata=model,
        reference_metadata=metadata,
        args=make_args(),
    )
    second = evaluate_module.build_evaluation_config(
        model_metadata=model,
        reference_metadata=metadata,
        args=make_args(),
    )
    changed = evaluate_module.build_evaluation_config(
        model_metadata=model,
        reference_metadata=metadata,
        args=make_args(num_steps=3),
    )

    assert metadata["mu_shape"] == [2]
    assert metadata["sigma_shape"] == [2, 2]
    assert metadata["metadata"]["dataset"] == "tiny"
    assert first == second
    assert first["protocol_id"] != changed["protocol_id"]


def test_evaluation_config_surfaces_distilled_profile_provenance(tmp_path):
    reference = tmp_path / "reference.npz"
    write_reference_stats(reference)
    source_profile = {
        "format": "diffdist_edm_source_profile_v1",
        "results_json_path": "/profiles/pfi/results.json",
        "results_sha256": "a" * 64,
        "profile_fingerprint": {"pfi_seed": 0},
        "profile_fingerprint_sha256": "b" * 64,
    }
    ablation_protocol = {
        "protocol_id": "batch_local_exact_sigma_pfi_v1",
        "ablation_mode": "pfi",
    }
    filter_sampling = make_exhaustive_sampling()

    config = evaluate_module.build_evaluation_config(
        model_metadata={
            "kind": "distilled_checkpoint",
            "source": "/tmp/student.pt",
            "source_profile": source_profile,
            "ablation_protocol": ablation_protocol,
            "filter_sampling": filter_sampling,
        },
        reference_metadata=evaluate_module.read_reference_metadata(reference),
        args=make_args(),
    )

    assert config["source_profile"] == source_profile
    assert config["ablation_protocol"] == ablation_protocol
    assert config["model"]["source_profile"] == source_profile
    assert config["filter_sampling"] == filter_sampling
    assert config["model"]["filter_sampling"] == filter_sampling


def test_prepare_output_directory_rejects_mismatch_and_overwrites_samples(tmp_path):
    output_dir = tmp_path / "evaluation"
    config = {"config_format": "test", "protocol_id": "one"}
    evaluate_module.prepare_output_directory(output_dir, config, overwrite=False)
    samples = output_dir / "samples"
    samples.mkdir()
    (samples / "seed000000.png").write_bytes(b"old")

    with pytest.raises(ValueError, match="does not match"):
        evaluate_module.prepare_output_directory(
            output_dir,
            {"config_format": "test", "protocol_id": "two"},
            overwrite=False,
        )

    replacement = {"config_format": "test", "protocol_id": "two"}
    evaluate_module.prepare_output_directory(output_dir, replacement, overwrite=True)
    assert not samples.exists()
    assert json.loads((output_dir / "evaluation_config.json").read_text()) == replacement


def test_prepare_output_directory_accepts_json_equivalent_tuple_metadata(tmp_path):
    output_dir = tmp_path / "evaluation"
    config = {"config_format": "test", "array": {"shape": (2048,)}}

    evaluate_module.prepare_output_directory(output_dir, config, overwrite=False)
    samples = output_dir / "samples"
    samples.mkdir()
    (samples / "seed000000.png").write_bytes(b"preserve")

    evaluate_module.prepare_output_directory(output_dir, config, overwrite=False)

    assert (samples / "seed000000.png").read_bytes() == b"preserve"
    assert json.loads((output_dir / "evaluation_config.json").read_text())["array"]["shape"] == [2048]


def test_generate_samples_resumes_and_replaces_corrupt_images(monkeypatch, tmp_path):
    calls = []

    def fake_sampler(_net, latents, **_kwargs):
        calls.append(int(latents.shape[0]))
        return torch.zeros_like(latents)

    monkeypatch.setattr(evaluate_module, "edm_sampler", fake_sampler)
    samples_dir = tmp_path / "samples"
    seeds = [0, 1, 2]
    network = TinyEvaluationNet()

    first = evaluate_module.generate_samples(
        net=network,
        samples_dir=samples_dir,
        seeds=seeds,
        batch_size=2,
        sampling_config=make_sampling_config(),
        protocol_id="protocol",
        device=torch.device("cpu"),
        rank=0,
        world_size=1,
    )
    second = evaluate_module.generate_samples(
        net=network,
        samples_dir=samples_dir,
        seeds=seeds,
        batch_size=2,
        sampling_config=make_sampling_config(),
        protocol_id="protocol",
        device=torch.device("cpu"),
        rank=0,
        world_size=1,
    )
    (samples_dir / "seed000001.png").write_bytes(b"corrupt")
    third = evaluate_module.generate_samples(
        net=network,
        samples_dir=samples_dir,
        seeds=seeds,
        batch_size=2,
        sampling_config=make_sampling_config(),
        protocol_id="protocol",
        device=torch.device("cpu"),
        rank=0,
        world_size=1,
    )

    assert first["generated_count"] == 3
    assert first["skipped_count"] == 0
    assert second["generated_count"] == 0
    assert second["skipped_count"] == 3
    assert third["generated_count"] == 1
    assert third["skipped_count"] == 2
    assert calls == [2, 1, 1]
    evaluate_module.validate_complete_sample_set(samples_dir, seeds)


def test_generate_samples_partitions_seeds_by_rank(monkeypatch, tmp_path):
    monkeypatch.setattr(
        evaluate_module,
        "edm_sampler",
        lambda _net, latents, **_kwargs: torch.zeros_like(latents),
    )
    samples_dir = tmp_path / "samples"
    seeds = list(range(6))
    network = TinyEvaluationNet()

    rank_zero = evaluate_module.generate_samples(
        net=network,
        samples_dir=samples_dir,
        seeds=seeds,
        batch_size=2,
        sampling_config=make_sampling_config(),
        protocol_id="protocol",
        device=torch.device("cpu"),
        rank=0,
        world_size=2,
    )
    rank_one = evaluate_module.generate_samples(
        net=network,
        samples_dir=samples_dir,
        seeds=seeds,
        batch_size=2,
        sampling_config=make_sampling_config(),
        protocol_id="protocol",
        device=torch.device("cpu"),
        rank=1,
        world_size=2,
    )

    assert rank_zero["assigned_count"] == 3
    assert rank_one["assigned_count"] == 3
    assert len(list(samples_dir.glob("seed*.png"))) == 6
    evaluate_module.validate_complete_sample_set(samples_dir, seeds)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_generate_samples_rejects_nonfinite_images_before_writing(monkeypatch, tmp_path, nonfinite):
    def fake_sampler(_net, latents, **_kwargs):
        images = torch.zeros_like(latents)
        images[-1, 0, 0, 0] = nonfinite
        return images

    monkeypatch.setattr(evaluate_module, "edm_sampler", fake_sampler)
    samples_dir = tmp_path / "samples"
    secondary_dir = tmp_path / "secondary"
    with pytest.raises(RuntimeError, match="Non-finite generated samples"):
        evaluate_module.generate_samples(
            net=TinyEvaluationNet(),
            samples_dir=samples_dir,
            secondary_samples_dir=secondary_dir,
            secondary_quantizer=evaluate_module.SampleQuantizer.OPENAI_TRUNCATE,
            seeds=[0, 1],
            batch_size=2,
            sampling_config=make_sampling_config(),
            protocol_id="protocol",
            device=torch.device("cpu"),
            rank=0,
            world_size=1,
        )

    # Even the finite first image must not be saved from a failed batch.
    assert list(samples_dir.iterdir()) == []
    assert list(secondary_dir.iterdir()) == []


def test_generate_samples_rejects_nonfinite_warmup_before_generation(monkeypatch, tmp_path):
    calls = []

    def fake_sampler(_net, latents, **_kwargs):
        calls.append(int(latents.shape[0]))
        return torch.full_like(latents, float("nan")) if len(calls) == 1 else torch.zeros_like(latents)

    monkeypatch.setattr(evaluate_module, "edm_sampler", fake_sampler)
    samples_dir = tmp_path / "samples"
    with pytest.raises(RuntimeError, match="(?i)non-finite.*warmup"):
        evaluate_module.generate_samples(
            net=TinyEvaluationNet(),
            samples_dir=samples_dir,
            seeds=[0, 1],
            batch_size=2,
            sampling_config=make_sampling_config(),
            protocol_id="protocol",
            device=torch.device("cpu"),
            rank=0,
            world_size=1,
            warmup_samples=2,
        )

    assert calls == [2]
    assert list(samples_dir.iterdir()) == []


def test_dry_run_cli_does_not_load_checkpoint(tmp_path):
    checkpoint = tmp_path / "placeholder.pt"
    checkpoint.touch()
    reference = tmp_path / "reference.npz"
    write_reference_stats(reference)
    output_dir = tmp_path / "output"

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluate_edm_checkpoint.py"),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output_dir),
            "--reference-stats",
            str(reference),
            "--num-samples",
            "5000",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)

    assert payload["dry_run"] is True
    assert payload["seed_range"] == [0, 4999]
    assert payload["nfe_per_image"] == 35
    assert not output_dir.exists()
