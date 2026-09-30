import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

from pace.evaluation_protocols import (
    ArtifactRetention,
    FFHQ64_ADM_CUSTOM_PROTOCOL,
    FFHQ64_NVIDIA_PROTOCOL,
    LSUN_BEDROOM256_ADM_MIXED_FP16_PROTOCOL,
    LSUN_BEDROOM256_ADM_PROTOCOL,
    PROTOCOLS,
    ProtocolError,
    SampleQuantizer,
    ThroughputAccumulator,
    apply_artifact_retention,
    benchmark_protocol_records,
    build_benchmark_manifest,
    quantize_samples,
    resolve_benchmark_protocol,
    samples_to_layout,
    validate_adm_sample_npz,
    validate_untyped_cleanfid_reference,
)

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from sample_edm_distilled import edm_sampler


def test_protocol_identities_quantizers_and_sampling_are_explicit():
    assert FFHQ64_NVIDIA_PROTOCOL.metrics == ("fid_nvlabs_legacy",)
    assert FFHQ64_NVIDIA_PROTOCOL.sampling.seed_ranges == (
        (0, 49_999),
        (50_000, 99_999),
        (100_000, 149_999),
    )
    assert FFHQ64_NVIDIA_PROTOCOL.sampling.aggregation == "minimum_of_three_runs"
    assert not FFHQ64_NVIDIA_PROTOCOL.sampling.clip_denoised
    assert FFHQ64_ADM_CUSTOM_PROTOCOL.quantizer is SampleQuantizer.OPENAI_TRUNCATE
    assert FFHQ64_ADM_CUSTOM_PROTOCOL.sampling.seed_ranges == ((0, 49_999),)
    assert FFHQ64_ADM_CUSTOM_PROTOCOL.sampling.aggregation == "single_run"
    assert LSUN_BEDROOM256_ADM_PROTOCOL.sampling.seed_ranges == ((2_100_000, 2_149_999),)
    assert LSUN_BEDROOM256_ADM_PROTOCOL.sampling.clip_denoised
    assert LSUN_BEDROOM256_ADM_PROTOCOL.reference_image_count == 5_000
    assert FFHQ64_ADM_CUSTOM_PROTOCOL.reference_image_count == 10_000


def test_benchmark_manifest_preserves_teacher_checkpoint_identity():
    teacher = {
        "source": "https://example.invalid/teacher.pt",
        "format": "state_dict",
        "preset": "fixture",
        "model_config": {"label_dim": 0},
        "sampling": {"sampler": "heun"},
        "expected_sha256": "a" * 64,
        "expected_size_bytes": 123,
        "checkpoint_sha256": "b" * 64,
        "checkpoint_size_bytes": 123,
    }

    manifest = build_benchmark_manifest(LSUN_BEDROOM256_ADM_PROTOCOL, teacher=teacher)

    assert manifest["teacher"] == teacher


def test_lsun_mixed_fp16_protocol_changes_only_identity_precision_and_notes():
    original = LSUN_BEDROOM256_ADM_PROTOCOL
    matched = LSUN_BEDROOM256_ADM_MIXED_FP16_PROTOCOL

    assert original.identity == "lsun_bedroom256_openai_adm_v1"
    assert matched.identity == "lsun_bedroom256_openai_adm_mixed_fp16_v1"
    assert original.sampling.inference_dtype == "teacher_mixed_fp16_or_student_fp32_with_float64_sampler_state"
    assert matched.sampling.inference_dtype == "teacher_mixed_fp16_and_student_mixed_fp16_with_float64_sampler_state"
    assert PROTOCOLS[original.identity] is original
    assert PROTOCOLS[matched.identity] is matched
    assert matched.notes[: len(original.notes)] == original.notes
    assert len(matched.notes) > len(original.notes)
    assert dataclasses.replace(
        matched,
        identity=original.identity,
        notes=original.notes,
        sampling=dataclasses.replace(matched.sampling, inference_dtype=original.sampling.inference_dtype),
    ) == original


@pytest.mark.parametrize("dataset", ["lsun_bedroom", "lsun_bedroom256", "lsun_bedroom_256", "bedroom"])
@pytest.mark.parametrize("backend", ["auto", "openai_adm"])
def test_lsun_mixed_fp16_protocol_does_not_change_dataset_defaults(dataset, backend):
    assert resolve_benchmark_protocol(dataset, backend=backend, resolution=256) is LSUN_BEDROOM256_ADM_PROTOCOL
    assert benchmark_protocol_records(dataset) == (LSUN_BEDROOM256_ADM_PROTOCOL.to_dict(),)


def test_dataset_presets_publish_the_intended_protocol_records():
    ffhq = benchmark_protocol_records("ffhq64")
    bedroom = benchmark_protocol_records("lsun_bedroom")

    assert [record["identity"] for record in ffhq] == [
        FFHQ64_NVIDIA_PROTOCOL.identity,
        FFHQ64_ADM_CUSTOM_PROTOCOL.identity,
    ]
    assert [record["identity"] for record in bedroom] == [LSUN_BEDROOM256_ADM_PROTOCOL.identity]
    assert benchmark_protocol_records("cifar10") == ()


def test_exact_upstream_quantizers_and_layout_conversion():
    samples = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)
    np.testing.assert_array_equal(
        quantize_samples(samples, SampleQuantizer.NVLABS_ROUND),
        np.asarray([0, 128, 255], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        quantize_samples(samples, SampleQuantizer.OPENAI_TRUNCATE),
        np.asarray([0, 127, 255], dtype=np.uint8),
    )
    nchw = np.arange(24, dtype=np.uint8).reshape(1, 3, 2, 4)
    nhwc = samples_to_layout(nchw, source="NCHW", destination="NHWC")
    assert nhwc.shape == (1, 2, 4, 3)
    np.testing.assert_array_equal(samples_to_layout(nhwc, source="NHWC", destination="NCHW"), nchw)


def test_adm_sample_npz_schema_is_checked_from_headers(tmp_path):
    protocol = dataclasses.replace(LSUN_BEDROOM256_ADM_PROTOCOL, resolution=4, sample_count=2)
    valid = tmp_path / "valid.npz"
    np.savez(valid, arr_0=np.zeros((2, 4, 4, 3), dtype=np.uint8))
    assert validate_adm_sample_npz(valid, protocol)["count"] == 2

    bad = tmp_path / "bad.npz"
    np.savez(bad, arr_0=np.zeros((2, 3, 4, 3), dtype=np.uint8))
    with pytest.raises(ProtocolError, match="must have shape"):
        validate_adm_sample_npz(bad, protocol)


def test_legacy_cleanfid_reference_rejects_adm_arrays_but_accepts_custom_stats(tmp_path):
    custom = tmp_path / "custom.npz"
    np.savez(custom, mu=np.zeros(2), sigma=np.eye(2))
    metadata = validate_untyped_cleanfid_reference(custom)
    assert metadata["feature_space"] == "legacy_untyped_custom_cleanfid"

    adm = tmp_path / "adm.npz"
    np.savez(
        adm,
        arr_0=np.zeros((1, 4, 4, 3), dtype=np.uint8),
        mu=np.zeros(2),
        sigma=np.eye(2),
        mu_s=np.zeros(3),
        sigma_s=np.eye(3),
    )
    with pytest.raises(ProtocolError, match="cannot consume OpenAI ADM"):
        validate_untyped_cleanfid_reference(adm)


def test_retention_keeps_preview_only_after_success(tmp_path):
    root = tmp_path / "samples"
    preview = tmp_path / "preview"
    root.mkdir()
    paths = []
    for index in range(3):
        path = root / f"seed{index:06d}.png"
        path.write_bytes(bytes([index]))
        paths.append(path)
    outcome = apply_artifact_retention(
        root=root,
        samples=paths,
        retention=ArtifactRetention.KEEP_PREVIEW,
        evaluation_succeeded=True,
        preview_dir=preview,
        preview_count=1,
    )
    assert outcome == {"retention": "keep_preview", "deleted": 3, "previewed": 1, "skipped": 0}
    assert not any(path.exists() for path in paths)
    assert len(list(preview.iterdir())) == 1


def test_throughput_summary_uses_explicit_model_and_wall_labels():
    timing = ThroughputAccumulator(num_steps=40)
    timing.add(100, model_seconds=2.0, wall_seconds=4.0, write_seconds=1.0)
    summary = timing.summary()
    assert summary["model_samples_per_second"] == pytest.approx(50.0)
    assert summary["wall_samples_per_second"] == pytest.approx(25.0)
    assert summary["model_nfe_per_second"] == pytest.approx(3950.0)


class _ConstantDenoiser(torch.nn.Module):
    sigma_min = 0.0
    sigma_max = float("inf")

    @staticmethod
    def round_sigma(sigma):
        return torch.as_tensor(sigma)

    def forward(self, x, sigma, labels):
        return torch.full_like(x, 5.0)


def test_sampler_clips_both_euler_and_heun_denoised_predictions():
    latents = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    kwargs = dict(
        num_steps=2,
        sigma_min=0.1,
        sigma_max=1.0,
        rho=1.0,
        S_churn=0.0,
        randn_like=lambda value: torch.zeros_like(value),
    )
    unclipped = edm_sampler(_ConstantDenoiser(), latents, clip_denoised=False, **kwargs)
    clipped = edm_sampler(_ConstantDenoiser(), latents, clip_denoised=True, **kwargs)
    assert unclipped.item() == pytest.approx(5.0)
    assert clipped.item() == pytest.approx(1.0)
    assert not torch.equal(unclipped, clipped)
