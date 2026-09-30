"""Reproducible metric and sample-artifact protocols for EDM benchmarks.

The protocol identity is deliberately separate from the model loader.  Callers
may pass teacher metadata produced by any loader, but metric statistics are
validated against the feature extractor and artifact layout that created them.
This prevents a numerically plausible ``mu``/``sigma`` file from being used with
the wrong Inception implementation.
"""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import hashlib
import json
import os
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np


BENCHMARK_MANIFEST_FORMAT = "diffdist_edm_benchmark_manifest_v1"
ARTIFACT_MANIFEST_FORMAT = "diffdist_edm_artifact_manifest_v1"


class ProtocolError(ValueError):
    """Raised when artifacts do not satisfy a declared metric protocol."""


class MetricBackend(str, enum.Enum):
    NVLABS_EDM_FID = "nvlabs_edm_fid"
    OPENAI_ADM = "openai_adm"
    CLEANFID = "cleanfid"


class SampleQuantizer(str, enum.Enum):
    NVLABS_ROUND = "nvlabs_x127_5_plus_128"
    OPENAI_TRUNCATE = "openai_x_plus_1_x127_5"


class ArtifactRetention(str, enum.Enum):
    KEEP = "keep"
    KEEP_PREVIEW = "keep_preview"
    DISCARD = "discard"


@dataclass(frozen=True)
class RemoteArtifact:
    url: str
    size_bytes: int | None = None
    sha256: str | None = None
    md5: str | None = None


@dataclass(frozen=True)
class SamplingProtocol:
    sampler: str
    num_steps: int
    nfe_per_image: int
    sigma_min: float
    sigma_max: float
    rho: float
    s_churn: float
    s_min: float
    s_max: float | None
    s_noise: float
    seed_scheme: str
    seed_ranges: tuple[tuple[int, int], ...]
    aggregation: str
    inference_dtype: str
    clip_denoised: bool


@dataclass(frozen=True)
class BenchmarkProtocol:
    identity: str
    backend: MetricBackend
    dataset: str
    resolution: int
    metrics: tuple[str, ...]
    sample_count: int
    sample_format: str
    sample_layout: str
    quantizer: SampleQuantizer
    reference: RemoteArtifact | None
    detector: RemoteArtifact
    sampling: SamplingProtocol
    fid_mode: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)
    reference_image_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(dataclasses.asdict(self))


NVLABS_INCEPTION_URL = (
    "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/"
    "files/metrics/inception-2015-12-05.pkl"
)
OPENAI_ADM_INCEPTION_URL = (
    "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/"
    "ref_batches/classify_image_graph_def.pb"
)
FFHQ64_REFERENCE_URL = "https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/ffhq-64x64.npz"
LSUN_BEDROOM256_REFERENCE_URL = (
    "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/"
    "ref_batches/lsun/bedroom/VIRTUAL_lsun_bedroom256.npz"
)


FFHQ64_NVIDIA_PROTOCOL = BenchmarkProtocol(
    identity="ffhq64_nvlabs_edm_fid_v1",
    backend=MetricBackend.NVLABS_EDM_FID,
    dataset="ffhq",
    resolution=64,
    metrics=("fid_nvlabs_legacy",),
    sample_count=50_000,
    sample_format="png_directory_or_zip",
    sample_layout="HWC_uint8_per_png",
    quantizer=SampleQuantizer.NVLABS_ROUND,
    reference=RemoteArtifact(
        url=FFHQ64_REFERENCE_URL,
        size_bytes=33_571_316,
        sha256="c78359df3b784dd472ff1ec53beefd9aeae9dace7df592128ed6b9d0c9f09c47",
    ),
    detector=RemoteArtifact(url=NVLABS_INCEPTION_URL),
    sampling=SamplingProtocol(
        sampler="edm_heun",
        num_steps=40,
        nfe_per_image=79,
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0,
        s_churn=0.0,
        s_min=0.0,
        s_max=None,
        s_noise=1.0,
        seed_scheme="nvlabs_stacked_per_sample",
        seed_ranges=((0, 49_999), (50_000, 99_999), (100_000, 149_999)),
        aggregation="minimum_of_three_runs",
        inference_dtype="teacher_or_student_fp32_with_float64_sampler_state",
        clip_denoised=False,
    ),
    notes=(
        "Official NVLabs paper-comparison protocol; report every run as well as the minimum.",
        "The reference contains pool-feature mean and covariance only.",
    ),
)


LSUN_BEDROOM256_ADM_PROTOCOL = BenchmarkProtocol(
    identity="lsun_bedroom256_openai_adm_v1",
    backend=MetricBackend.OPENAI_ADM,
    dataset="lsun_bedroom",
    resolution=256,
    metrics=("fid_adm", "sfid_adm", "precision_adm", "recall_adm", "inception_score_adm"),
    sample_count=50_000,
    sample_format="npz_arr_0",
    sample_layout="NHWC_uint8",
    quantizer=SampleQuantizer.OPENAI_TRUNCATE,
    reference=RemoteArtifact(
        url=LSUN_BEDROOM256_REFERENCE_URL,
        size_bytes=1_054_074_338,
        md5="765035466f3da1de10de3b72ff1a361b",
    ),
    detector=RemoteArtifact(
        url=OPENAI_ADM_INCEPTION_URL,
        size_bytes=95_673_916,
        md5="e6bb154e85f5d4331c22abcddb5dcf31",
    ),
    sampling=SamplingProtocol(
        sampler="edm_heun",
        num_steps=40,
        nfe_per_image=79,
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0,
        s_churn=0.0,
        s_min=0.0,
        s_max=None,
        s_noise=1.0,
        seed_scheme="openai_determ_indiv_seed_42_num_samples_50000",
        seed_ranges=((2_100_000, 2_149_999),),
        aggregation="single_run",
        inference_dtype="teacher_mixed_fp16_or_student_fp32_with_float64_sampler_state",
        clip_denoised=True,
    ),
    notes=(
        "The seed range records the per-image torch.Generator seeds used by determ-indiv.",
        "The verified official reference stores full-dataset statistics and 5,000 real images for precision/recall.",
    ),
    reference_image_count=5_000,
)


LSUN_BEDROOM256_ADM_MIXED_FP16_PROTOCOL = dataclasses.replace(
    LSUN_BEDROOM256_ADM_PROTOCOL,
    identity="lsun_bedroom256_openai_adm_mixed_fp16_v1",
    sampling=dataclasses.replace(
        LSUN_BEDROOM256_ADM_PROTOCOL.sampling,
        inference_dtype="teacher_mixed_fp16_and_student_mixed_fp16_with_float64_sampler_state",
    ),
    notes=LSUN_BEDROOM256_ADM_PROTOCOL.notes
    + (
        "Precision-matched experiment: teacher and student denoisers both use mixed FP16; sampler state remains float64.",
        "This explicit opt-in protocol does not replace the original LSUN Bedroom protocol or its existing results.",
    ),
)


FFHQ64_ADM_CUSTOM_PROTOCOL = BenchmarkProtocol(
    identity="ffhq64_openai_adm_custom_first50k_v1",
    backend=MetricBackend.OPENAI_ADM,
    dataset="ffhq",
    resolution=64,
    metrics=("fid_adm", "sfid_adm", "precision_adm", "recall_adm", "inception_score_adm"),
    sample_count=50_000,
    sample_format="npz_arr_0",
    sample_layout="NHWC_uint8",
    quantizer=SampleQuantizer.OPENAI_TRUNCATE,
    reference=None,
    detector=RemoteArtifact(
        url=OPENAI_ADM_INCEPTION_URL,
        size_bytes=95_673_916,
        md5="e6bb154e85f5d4331c22abcddb5dcf31",
    ),
    sampling=dataclasses.replace(
        FFHQ64_NVIDIA_PROTOCOL.sampling,
        seed_ranges=((0, 49_999),),
        aggregation="single_run",
    ),
    notes=(
        "PACE custom ADM-suite protocol; it is not an NVLabs-published FFHQ metric protocol.",
        "Reference stats use the first 50,000 prepared FFHQ-64 images; arr_0 stores the first 10,000 for PR.",
        "ADM artifacts use OpenAI truncation; generate a separate NVLabs-rounded representation for canonical FID.",
        "Report this alongside, never in place of, canonical NVLabs FID.",
    ),
    reference_image_count=10_000,
)


def cleanfid_protocol(dataset: str, resolution: int, *, mode: str = "clean", sample_count: int = 5_000) -> BenchmarkProtocol:
    """Return an explicitly non-paper-comparable CleanFID monitoring protocol."""

    if mode not in {"clean", "legacy_tensorflow", "legacy_pytorch"}:
        raise ProtocolError(f"unsupported CleanFID mode: {mode}")
    if resolution <= 0 or sample_count < 2:
        raise ProtocolError("resolution must be positive and sample_count must be at least two")
    return BenchmarkProtocol(
        identity=f"{dataset}{resolution}_cleanfid_{mode}_monitor_v1",
        backend=MetricBackend.CLEANFID,
        dataset=dataset,
        resolution=resolution,
        metrics=(f"fid_clean_{mode}",),
        sample_count=sample_count,
        sample_format="png_directory",
        sample_layout="HWC_uint8_per_png",
        quantizer=SampleQuantizer.NVLABS_ROUND,
        reference=None,
        detector=RemoteArtifact(
            url="https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/"
            "inception-2015-12-05.pt"
        ),
        sampling=SamplingProtocol(
            sampler="edm_heun",
            num_steps=18,
            nfe_per_image=35,
            sigma_min=0.002,
            sigma_max=80.0,
            rho=7.0,
            s_churn=0.0,
            s_min=0.0,
            s_max=None,
            s_noise=1.0,
            seed_scheme="nvlabs_stacked_per_sample",
            seed_ranges=((0, sample_count - 1),),
            aggregation="single_run",
            inference_dtype="model_default",
            clip_denoised=False,
        ),
        fid_mode=mode,
        notes=("Monitoring protocol; do not compare directly with published NVLabs or ADM numbers.",),
    )


PROTOCOLS: dict[str, BenchmarkProtocol] = {
    FFHQ64_NVIDIA_PROTOCOL.identity: FFHQ64_NVIDIA_PROTOCOL,
    FFHQ64_ADM_CUSTOM_PROTOCOL.identity: FFHQ64_ADM_CUSTOM_PROTOCOL,
    LSUN_BEDROOM256_ADM_PROTOCOL.identity: LSUN_BEDROOM256_ADM_PROTOCOL,
    LSUN_BEDROOM256_ADM_MIXED_FP16_PROTOCOL.identity: LSUN_BEDROOM256_ADM_MIXED_FP16_PROTOCOL,
}


def benchmark_protocol_records(dataset: str) -> tuple[dict[str, Any], ...]:
    """Return the intended published protocols for a supported dataset preset.

    Profiling and training do not execute metrics, but carrying these immutable
    records forward prevents a later benchmark from silently changing the
    protocol that the production preset was designed to publish.  Unknown and
    legacy datasets intentionally return no records.
    """

    normalized = str(dataset).lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ffhq64": "ffhq",
        "ffhq_64": "ffhq",
        "lsun_bedroom256": "lsun_bedroom",
        "lsun_bedroom_256": "lsun_bedroom",
        "bedroom": "lsun_bedroom",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized == "ffhq":
        selected = (FFHQ64_NVIDIA_PROTOCOL, FFHQ64_ADM_CUSTOM_PROTOCOL)
    elif normalized == "lsun_bedroom":
        selected = (LSUN_BEDROOM256_ADM_PROTOCOL,)
    else:
        selected = ()
    return tuple(protocol.to_dict() for protocol in selected)


def resolve_benchmark_protocol(
    dataset: str,
    *,
    backend: str | MetricBackend = "auto",
    resolution: int | None = None,
) -> BenchmarkProtocol:
    normalized = dataset.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ffhq64": "ffhq",
        "ffhq_64": "ffhq",
        "lsun_bedroom256": "lsun_bedroom",
        "lsun_bedroom_256": "lsun_bedroom",
        "bedroom": "lsun_bedroom",
    }
    normalized = aliases.get(normalized, normalized)
    resolved_backend = backend.value if isinstance(backend, MetricBackend) else str(backend)
    if resolved_backend == "auto":
        if normalized == "ffhq" and resolution in {None, 64}:
            return FFHQ64_NVIDIA_PROTOCOL
        if normalized == "lsun_bedroom" and resolution in {None, 256}:
            return LSUN_BEDROOM256_ADM_PROTOCOL
        raise ProtocolError(f"no automatic benchmark protocol for dataset={dataset!r}, resolution={resolution!r}")
    if resolved_backend == MetricBackend.NVLABS_EDM_FID.value and normalized == "ffhq" and resolution in {None, 64}:
        return FFHQ64_NVIDIA_PROTOCOL
    if resolved_backend == MetricBackend.OPENAI_ADM.value and normalized == "ffhq" and resolution in {None, 64}:
        return FFHQ64_ADM_CUSTOM_PROTOCOL
    if resolved_backend == MetricBackend.OPENAI_ADM.value and normalized == "lsun_bedroom" and resolution in {None, 256}:
        return LSUN_BEDROOM256_ADM_PROTOCOL
    if resolved_backend == MetricBackend.CLEANFID.value:
        if resolution is None:
            raise ProtocolError("CleanFID requires an explicit resolution")
        return cleanfid_protocol(normalized, resolution)
    raise ProtocolError(
        f"backend {resolved_backend!r} is not canonical for dataset={dataset!r}, resolution={resolution!r}"
    )


def _load_feature_array(value: str | Path | np.ndarray, *, name: str) -> np.ndarray:
    if isinstance(value, (str, Path)):
        source = Path(value)
        if not source.is_file():
            raise ProtocolError(f"{name} feature file does not exist: {source}")
        loaded = np.load(source, mmap_mode="r", allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            try:
                if name not in loaded.files:
                    raise ProtocolError(f"{source} does not contain array {name!r}")
                array = np.asarray(loaded[name])
            finally:
                loaded.close()
            return array
        return np.asarray(loaded)
    return np.asarray(value)


def _sorted_image_paths(image_root: str | Path) -> list[Path]:
    root = Path(image_root)
    if not root.is_dir():
        raise ProtocolError(f"image root does not exist: {root}")
    suffixes = {".png", ".jpg", ".jpeg"}
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def build_ffhq64_adm_reference(
    destination: str | Path,
    *,
    prepared_image_root: str | Path,
    pool_features: str | Path | np.ndarray,
    spatial_features: str | Path | np.ndarray,
    stats_count: int = 50_000,
    precision_recall_count: int = 10_000,
) -> dict[str, Any]:
    """Build the approved custom FFHQ-64 ADM reference from fixed-order inputs.

    ``pool_features`` and ``spatial_features`` must be activations from the
    pinned OpenAI ADM GraphDef for the lexicographically first ``stats_count``
    prepared images.  Feature extraction stays an external integration hook so
    TensorFlow does not enter the core environment.
    """

    if stats_count != 50_000 or precision_recall_count != 10_000:
        raise ProtocolError("ffhq64_openai_adm_custom_first50k_v1 requires stats_count=50000 and PR count=10000")
    images = _sorted_image_paths(prepared_image_root)
    if len(images) < stats_count:
        raise ProtocolError(f"FFHQ ADM reference needs at least {stats_count} images, found {len(images)}")
    pool = _load_feature_array(pool_features, name="pool_features")
    spatial = _load_feature_array(spatial_features, name="spatial_features")
    for name, features in (("pool_features", pool), ("spatial_features", spatial)):
        if features.ndim != 2 or features.shape[0] < stats_count:
            raise ProtocolError(f"{name} must have shape [at least {stats_count}, D], got {features.shape}")
        if not np.issubdtype(features.dtype, np.floating):
            raise ProtocolError(f"{name} must be floating point, got {features.dtype}")
    if pool.shape[1] != 2048:
        raise ProtocolError(f"pool_features must have dimension 2048, got {pool.shape[1]}")

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow is a core dependency in this repository
        raise ProtocolError("Pillow is required to build the FFHQ ADM reference") from exc
    pr_images = np.empty((precision_recall_count, 64, 64, 3), dtype=np.uint8)
    selected_names: list[str] = []
    root = Path(prepared_image_root).resolve()
    for index, path in enumerate(images[:precision_recall_count]):
        try:
            with Image.open(path) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        except Exception as exc:
            raise ProtocolError(f"could not decode prepared FFHQ image {path}: {exc}") from exc
        if array.shape != (64, 64, 3):
            raise ProtocolError(f"prepared FFHQ image must be 64x64 RGB, got {array.shape}: {path}")
        pr_images[index] = array
        selected_names.append(str(path.resolve().relative_to(root)))

    pool_slice = np.asarray(pool[:stats_count], dtype=np.float64)
    spatial_slice = np.asarray(spatial[:stats_count], dtype=np.float64)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_name(f".{destination_path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            # Uncompressed NPZ preserves the streaming reader used by the ADM
            # evaluator and avoids an expensive, opaque compression step.
            np.savez(
                handle,
                arr_0=pr_images,
                mu=np.mean(pool_slice, axis=0),
                sigma=np.cov(pool_slice, rowvar=False),
                mu_s=np.mean(spatial_slice, axis=0),
                sigma_s=np.cov(spatial_slice, rowvar=False),
            )
        os.replace(temporary, destination_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    payload = {
        "protocol_id": FFHQ64_ADM_CUSTOM_PROTOCOL.identity,
        "reference_path": str(destination_path.resolve()),
        "reference_sha256": file_digest(destination_path),
        "prepared_image_root": str(root),
        "stats_count": stats_count,
        "precision_recall_count": precision_recall_count,
        "image_order": "lexicographic_relative_path",
        "first_pr_image": selected_names[0],
        "last_pr_image": selected_names[-1],
        "detector": _jsonable(dataclasses.asdict(FFHQ64_ADM_CUSTOM_PROTOCOL.detector)),
    }
    atomic_write_json(destination_path.with_suffix(destination_path.suffix + ".manifest.json"), payload)
    return payload


def _is_torch_tensor(value: Any) -> bool:
    return value.__class__.__module__.startswith("torch") and hasattr(value, "to")


def quantize_samples(images: Any, quantizer: SampleQuantizer | str) -> Any:
    """Quantize model outputs in ``[-1, 1]`` exactly like the named upstream.

    Both upstream implementations cast a clipped non-negative floating tensor
    to uint8, which truncates fractional values.  Their offsets differ by 0.5.
    The function preserves the input layout and returns the same array family
    (NumPy or Torch).
    """

    kind = SampleQuantizer(quantizer)
    if _is_torch_tensor(images):
        import torch

        values = images * 127.5 + (128.0 if kind is SampleQuantizer.NVLABS_ROUND else 127.5)
        return values.clamp(0, 255).to(torch.uint8)
    values = np.asarray(images)
    values = values * 127.5 + (128.0 if kind is SampleQuantizer.NVLABS_ROUND else 127.5)
    return np.clip(values, 0, 255).astype(np.uint8)


def samples_to_layout(images: Any, *, source: str, destination: str) -> Any:
    source = source.upper()
    destination = destination.upper()
    if source == destination:
        return images
    if {source, destination} != {"NCHW", "NHWC"}:
        raise ProtocolError(f"unsupported layout conversion: {source} -> {destination}")
    axes = (0, 2, 3, 1) if source == "NCHW" else (0, 3, 1, 2)
    if _is_torch_tensor(images):
        return images.permute(*axes).contiguous()
    return np.ascontiguousarray(np.transpose(np.asarray(images), axes))


@dataclass(frozen=True)
class NpyMember:
    name: str
    shape: tuple[int, ...]
    dtype: str
    fortran_order: bool


def inspect_npz_headers(path: str | Path) -> dict[str, NpyMember]:
    """Inspect NPZ array headers without inflating large ADM image arrays."""

    source = Path(path)
    try:
        archive = zipfile.ZipFile(source)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ProtocolError(f"could not open NPZ archive {source}: {exc}") from exc
    members: dict[str, NpyMember] = {}
    with archive:
        for member_name in archive.namelist():
            if not member_name.endswith(".npy"):
                continue
            key = member_name[:-4]
            with archive.open(member_name) as handle:
                try:
                    version = np.lib.format.read_magic(handle)
                    if version == (1, 0):
                        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
                    elif version in {(2, 0), (3, 0)}:
                        # NumPy 2.0 stopped exporting the private
                        # ``_read_array_header`` helper. Versions 2 and 3 use
                        # the same header-length layout; v3 only changes the
                        # header text encoding, and array descriptors emitted
                        # by NumPy remain compatible with the public v2 reader.
                        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
                    else:
                        raise ProtocolError(f"unsupported NPY version {version} in {member_name}")
                except Exception as exc:
                    raise ProtocolError(f"could not read {member_name} header in {source}: {exc}") from exc
            members[key] = NpyMember(
                name=key,
                shape=tuple(int(dim) for dim in shape),
                dtype=np.dtype(dtype).str,
                fortran_order=bool(fortran_order),
            )
    if not members:
        raise ProtocolError(f"NPZ archive contains no NPY members: {source}")
    return members


def file_digest(path: str | Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_members(headers: Mapping[str, NpyMember], required: set[str], *, context: str) -> None:
    missing = required.difference(headers)
    if missing:
        raise ProtocolError(f"{context} is missing required arrays: {sorted(missing)}")


def _validate_stats(headers: Mapping[str, NpyMember], mu_key: str, sigma_key: str, *, expected_dim: int | None) -> None:
    mu = headers[mu_key]
    sigma = headers[sigma_key]
    if len(mu.shape) != 1:
        raise ProtocolError(f"{mu_key} must be one-dimensional, got {mu.shape}")
    if sigma.shape != (mu.shape[0], mu.shape[0]):
        raise ProtocolError(f"{sigma_key} must have shape {(mu.shape[0], mu.shape[0])}, got {sigma.shape}")
    if expected_dim is not None and mu.shape != (expected_dim,):
        raise ProtocolError(f"{mu_key} must have shape {(expected_dim,)}, got {mu.shape}")
    if np.dtype(mu.dtype).kind != "f" or np.dtype(sigma.dtype).kind != "f":
        raise ProtocolError(f"{mu_key}/{sigma_key} must use floating-point dtypes")


def validate_reference_npz(
    path: str | Path,
    protocol: BenchmarkProtocol,
    *,
    verify_checksum: bool = True,
    strict_counts: bool = True,
) -> dict[str, Any]:
    """Validate a reference file against its declared feature-space protocol."""

    source = Path(path)
    if not source.is_file():
        raise ProtocolError(f"reference file does not exist: {source}")
    headers = inspect_npz_headers(source)
    if protocol.backend is MetricBackend.NVLABS_EDM_FID:
        _require_members(headers, {"mu", "sigma"}, context="NVLabs EDM reference")
        forbidden = {"arr_0", "mu_s", "sigma_s"}.intersection(headers)
        if forbidden:
            raise ProtocolError(f"NVLabs EDM reference unexpectedly contains ADM arrays: {sorted(forbidden)}")
        _validate_stats(headers, "mu", "sigma", expected_dim=2048)
    elif protocol.backend is MetricBackend.OPENAI_ADM:
        _require_members(headers, {"arr_0", "mu", "sigma", "mu_s", "sigma_s"}, context="ADM reference")
        _validate_stats(headers, "mu", "sigma", expected_dim=2048)
        _validate_stats(headers, "mu_s", "sigma_s", expected_dim=None)
        images = headers["arr_0"]
        expected_count = (
            protocol.reference_image_count
            if strict_counts and protocol.reference_image_count is not None
            else images.shape[0]
        )
        expected_shape = (expected_count, protocol.resolution, protocol.resolution, 3)
        if images.shape != expected_shape:
            raise ProtocolError(f"ADM reference arr_0 must have shape {expected_shape}, got {images.shape}")
        if np.dtype(images.dtype) != np.dtype(np.uint8):
            raise ProtocolError(f"ADM reference arr_0 must be uint8, got {images.dtype}")
    elif protocol.backend is MetricBackend.CLEANFID:
        _require_members(headers, {"mu", "sigma"}, context="CleanFID reference")
        forbidden = {"arr_0", "mu_s", "sigma_s"}.intersection(headers)
        if forbidden:
            raise ProtocolError(f"CleanFID reference unexpectedly contains ADM arrays: {sorted(forbidden)}")
        if source.stat().st_size == FFHQ64_NVIDIA_PROTOCOL.reference.size_bytes:
            actual = file_digest(source, "sha256") if verify_checksum else None
            if actual == FFHQ64_NVIDIA_PROTOCOL.reference.sha256:
                raise ProtocolError(
                    "canonical NVLabs FFHQ statistics are legacy TF-Inception features, not CleanFID-clean statistics"
                )
        _validate_stats(headers, "mu", "sigma", expected_dim=2048)
    else:  # pragma: no cover - enum exhaustiveness
        raise ProtocolError(f"unsupported backend: {protocol.backend}")

    reference = protocol.reference
    if reference is not None:
        if reference.size_bytes is not None and source.stat().st_size != reference.size_bytes:
            raise ProtocolError(
                f"reference size mismatch: expected {reference.size_bytes}, got {source.stat().st_size}"
            )
        if verify_checksum and reference.sha256:
            actual = file_digest(source, "sha256")
            if actual != reference.sha256:
                raise ProtocolError(f"reference SHA-256 mismatch: expected {reference.sha256}, got {actual}")
        if verify_checksum and reference.md5:
            actual = file_digest(source, "md5")
            if actual != reference.md5:
                raise ProtocolError(f"reference MD5 mismatch: expected {reference.md5}, got {actual}")
    return {
        "path": str(source.resolve()),
        "backend": protocol.backend.value,
        "protocol_id": protocol.identity,
        "size_bytes": source.stat().st_size,
        "arrays": {key: dataclasses.asdict(value) for key, value in sorted(headers.items())},
    }


def validate_untyped_cleanfid_reference(
    path: str | Path,
    *,
    verify_known_checksum: bool = True,
) -> dict[str, Any]:
    """Safety-check a legacy, untyped CleanFID reference.

    Older callers supplied custom ``mu``/``sigma`` files without feature-space
    metadata, so this compatibility path deliberately accepts arbitrary feature
    dimensions.  It still rejects the two reference families that are known to
    be incompatible: OpenAI ADM batches and the canonical NVLabs legacy
    TensorFlow-Inception FFHQ statistics.
    """

    source = Path(path)
    if not source.is_file():
        raise ProtocolError(f"CleanFID reference file does not exist: {source}")
    headers = inspect_npz_headers(source)
    _require_members(headers, {"mu", "sigma"}, context="CleanFID reference")
    forbidden = {"arr_0", "mu_s", "sigma_s"}.intersection(headers)
    if forbidden:
        raise ProtocolError(
            "CleanFID cannot consume OpenAI ADM feature/image arrays; "
            f"found {sorted(forbidden)} in {source}"
        )
    _validate_stats(headers, "mu", "sigma", expected_dim=None)

    official = FFHQ64_NVIDIA_PROTOCOL.reference
    assert official is not None
    if source.stat().st_size == official.size_bytes and verify_known_checksum:
        if file_digest(source, "sha256") == official.sha256:
            raise ProtocolError(
                "the canonical NVLabs FFHQ reference uses legacy TensorFlow-Inception features and "
                "is incompatible with CleanFID; select --metric-backend nvlabs_edm_fid"
            )
    return {
        "path": str(source.resolve()),
        "backend": MetricBackend.CLEANFID.value,
        "protocol_id": None,
        "size_bytes": source.stat().st_size,
        "arrays": {key: dataclasses.asdict(value) for key, value in sorted(headers.items())},
        "feature_space": "legacy_untyped_custom_cleanfid",
    }


def validate_adm_sample_npz(
    path: str | Path,
    protocol: BenchmarkProtocol = LSUN_BEDROOM256_ADM_PROTOCOL,
    *,
    strict_count: bool = True,
) -> dict[str, Any]:
    if protocol.backend is not MetricBackend.OPENAI_ADM:
        raise ProtocolError("ADM sample validation requires an OpenAI ADM protocol")
    headers = inspect_npz_headers(path)
    _require_members(headers, {"arr_0"}, context="ADM sample batch")
    images = headers["arr_0"]
    count = protocol.sample_count if strict_count else images.shape[0]
    expected = (count, protocol.resolution, protocol.resolution, 3)
    if images.shape != expected:
        raise ProtocolError(f"ADM sample arr_0 must have shape {expected}, got {images.shape}")
    if np.dtype(images.dtype) != np.dtype(np.uint8):
        raise ProtocolError(f"ADM sample arr_0 must be uint8, got {images.dtype}")
    return {"path": str(Path(path).resolve()), "count": images.shape[0], "layout": protocol.sample_layout}


def _jsonable(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.value
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_benchmark_manifest(
    protocol: BenchmarkProtocol,
    *,
    teacher: Mapping[str, Any],
    artifacts: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
    versions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a stable manifest while keeping the teacher-loader interface narrow."""

    if not teacher.get("source"):
        raise ProtocolError("teacher metadata must include a non-empty 'source'")
    teacher_payload = {
        key: _jsonable(teacher.get(key))
        for key in (
            "source",
            "format",
            "preset",
            "model_config",
            "sampling",
            "expected_sha256",
            "expected_size_bytes",
            "checkpoint_sha256",
            "checkpoint_size_bytes",
        )
        if key in teacher
    }
    base = {
        "manifest_format": BENCHMARK_MANIFEST_FORMAT,
        "protocol": protocol.to_dict(),
        "teacher": teacher_payload,
        "artifacts": _jsonable(artifacts or {}),
        "runtime": _jsonable(runtime or {}),
        "versions": dict(versions or {}),
    }
    return {**base, "manifest_sha256": canonical_digest(base)}


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass
class ThroughputAccumulator:
    num_steps: int
    samples: int = 0
    batches: int = 0
    model_seconds: float = 0.0
    wall_seconds: float = 0.0
    write_seconds: float = 0.0

    def add(self, sample_count: int, *, model_seconds: float, wall_seconds: float, write_seconds: float = 0.0) -> None:
        if sample_count < 0 or min(model_seconds, wall_seconds, write_seconds) < 0:
            raise ProtocolError("throughput counts and timings must be non-negative")
        self.samples += int(sample_count)
        self.batches += 1
        self.model_seconds += float(model_seconds)
        self.wall_seconds += float(wall_seconds)
        self.write_seconds += float(write_seconds)

    def summary(self) -> dict[str, float | int | None]:
        nfe = 2 * self.num_steps - 1
        return {
            "samples": self.samples,
            "batches": self.batches,
            "num_steps": self.num_steps,
            "nfe_per_image": nfe,
            "model_seconds": self.model_seconds,
            "wall_seconds": self.wall_seconds,
            "write_seconds": self.write_seconds,
            "model_samples_per_second": self.samples / self.model_seconds if self.model_seconds else None,
            "wall_samples_per_second": self.samples / self.wall_seconds if self.wall_seconds else None,
            "model_nfe_per_second": self.samples * nfe / self.model_seconds if self.model_seconds else None,
        }


@contextlib.contextmanager
def timed_region(synchronize: Callable[[], None] | None = None) -> Iterator[Callable[[], float]]:
    """Time a CUDA-safe region; the yielded callable reports elapsed seconds."""

    if synchronize is not None:
        synchronize()
    started = time.perf_counter()
    elapsed = 0.0

    def result() -> float:
        return elapsed

    try:
        yield result
    finally:
        if synchronize is not None:
            synchronize()
        elapsed = time.perf_counter() - started


def estimated_sample_bytes(protocol: BenchmarkProtocol) -> int:
    return protocol.sample_count * protocol.resolution * protocol.resolution * 3


def write_artifact_manifest(path: str | Path, *, root: str | Path, samples: Sequence[str | Path]) -> dict[str, Any]:
    root_path = Path(root).resolve()
    relative: list[str] = []
    for sample in samples:
        resolved = Path(sample).resolve()
        try:
            relative.append(str(resolved.relative_to(root_path)))
        except ValueError as exc:
            raise ProtocolError(f"sample path escapes artifact root: {sample}") from exc
    payload = {
        "manifest_format": ARTIFACT_MANIFEST_FORMAT,
        "root": str(root_path),
        "samples": relative,
    }
    atomic_write_json(path, payload)
    return payload


def apply_artifact_retention(
    *,
    root: str | Path,
    samples: Iterable[str | Path],
    retention: ArtifactRetention | str,
    evaluation_succeeded: bool,
    preview_dir: str | Path | None = None,
    preview_count: int = 64,
) -> dict[str, Any]:
    """Apply post-evaluation retention to an explicit, root-confined file list.

    No file is removed unless the caller confirms successful evaluation.  The
    function never recursively deletes ``root`` and rejects paths that escape it.
    """

    policy = ArtifactRetention(retention)
    root_path = Path(root).resolve()
    paths = sorted({Path(path).resolve() for path in samples})
    for path in paths:
        try:
            path.relative_to(root_path)
        except ValueError as exc:
            raise ProtocolError(f"refusing to retain/delete path outside {root_path}: {path}") from exc
    if policy is ArtifactRetention.KEEP or not evaluation_succeeded:
        return {"retention": policy.value, "deleted": 0, "previewed": 0, "skipped": len(paths)}
    if preview_count < 0:
        raise ProtocolError("preview_count must be non-negative")

    previewed = 0
    if policy is ArtifactRetention.KEEP_PREVIEW:
        if preview_dir is None:
            raise ProtocolError("keep_preview requires preview_dir")
        preview_root = Path(preview_dir).resolve()
        try:
            preview_root.relative_to(root_path)
        except ValueError:
            pass  # Keeping previews outside the disposable sample root is preferred.
        preview_root.mkdir(parents=True, exist_ok=True)
        for index, source in enumerate(path for path in paths if path.is_file()):
            if index >= preview_count:
                break
            destination = preview_root / source.name
            if destination.exists() and destination.resolve() != source:
                raise ProtocolError(f"preview destination already exists: {destination}")
            if destination.resolve() != source:
                shutil.copy2(source, destination)
            previewed += 1

    deleted = 0
    for path in paths:
        if path.is_file() or path.is_symlink():
            path.unlink()
            deleted += 1
    for directory in sorted({path.parent for path in paths}, key=lambda item: len(item.parts), reverse=True):
        if directory == root_path:
            continue
        with contextlib.suppress(OSError):
            directory.rmdir()
    return {"retention": policy.value, "deleted": deleted, "previewed": previewed, "skipped": 0}
