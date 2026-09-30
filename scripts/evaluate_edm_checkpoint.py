#!/usr/bin/env python3
"""Re-evaluate a distilled EDM checkpoint with a reproducible FID protocol."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pace.dataset_specs import dataset_spec
from pace.edm_distillation import NarrowEDMPrecond, construct_student_from_plan, load_edm_network
from pace.evaluation_protocols import (
    ArtifactRetention,
    FFHQ64_NVIDIA_PROTOCOL,
    LSUN_BEDROOM256_ADM_MIXED_FP16_PROTOCOL,
    LSUN_BEDROOM256_ADM_PROTOCOL,
    MetricBackend,
    PROTOCOLS,
    ProtocolError,
    SampleQuantizer,
    apply_artifact_retention,
    build_benchmark_manifest,
    quantize_samples,
    validate_adm_sample_npz,
    validate_reference_npz,
    validate_untyped_cleanfid_reference,
)
from pace.external_repos import default_edm_root
from pace.profile_provenance import provenance_from_artifact
from pace.teacher_models import (
    materialize_teacher_source,
    resolve_teacher_spec,
    teacher_model_metadata,
    teacher_preset_config,
)
from sample_edm_distilled import StackedRandomGenerator, edm_sampler, make_labels
from train_edm_distillation import cleanup_distributed, compute_fid_with_reference_stats, init_distributed


CONFIG_FORMAT = "diffdist_edm_checkpoint_evaluation_config_v1"
RESULT_FORMAT = "diffdist_edm_checkpoint_evaluation_result_v1"


def resolve_declared_protocol(args: argparse.Namespace):
    protocol = PROTOCOLS.get(args.protocol_id) if args.protocol_id is not None else None
    if args.metric_backend == MetricBackend.CLEANFID.value:
        if protocol is not None and protocol.backend is not MetricBackend.CLEANFID:
            raise ProtocolError(f"protocol {protocol.identity} does not use CleanFID")
        return protocol
    if protocol is None:
        raise ProtocolError(
            f"--protocol-id must name one of {sorted(PROTOCOLS)} when --metric-backend={args.metric_backend}"
        )
    if protocol.backend.value != args.metric_backend:
        raise ProtocolError(
            f"protocol {protocol.identity} requires backend {protocol.backend.value}, got {args.metric_backend}"
        )
    return protocol


def resolve_teacher_dtype(args: argparse.Namespace, protocol, *, device: torch.device) -> torch.dtype:
    requested = getattr(args, "teacher_dtype", "auto")
    if requested == "fp16":
        if device.type != "cuda":
            raise ValueError("--teacher-dtype fp16 requires a CUDA device")
        return torch.float16
    if requested == "fp32":
        return torch.float32
    if requested != "auto":
        raise ValueError(f"Unsupported teacher dtype {requested!r}")

    # The versioned benchmark protocol is the strongest source.  For model-only
    # runs without a declared benchmark, fall back to the structured built-in
    # preset. OpenAI's Bedroom UNet uses mixed FP16; FFHQ uses FP32.
    wants_mixed_fp16 = bool(
        protocol is not None and "teacher_mixed_fp16" in protocol.sampling.inference_dtype
    ) or getattr(args, "network_preset", None) in {
        "lsun_bedroom_256",
        "lsun_bedroom256",
        "bedroom256",
    }
    return torch.float16 if wants_mixed_fp16 and device.type == "cuda" else torch.float32


def resolve_student_dtype(args: argparse.Namespace, protocol, *, device: torch.device) -> torch.dtype:
    requested = getattr(args, "student_dtype", "auto")
    if requested == "auto":
        requested = (
            "fp16" if protocol is not None and "student_mixed_fp16" in protocol.sampling.inference_dtype
            else "fp32"
        )
    if requested == "fp16":
        if device.type != "cuda":
            raise ValueError("--student-dtype fp16 requires a CUDA device")
        return torch.float16
    if requested == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported student dtype {requested!r}")


def configure_student_precision(network: torch.nn.Module, *, dtype: torch.dtype, device: torch.device) -> None:
    """Select native EDM mixed precision without quantizing FP32 master weights.

    Each expert casts its UNet activations to half, while time embeddings,
    attention logits, preconditioning and denoiser outputs retain FP32.
    """
    if dtype not in {torch.float16, torch.float32}:
        raise ValueError(f"Unsupported student dtype {dtype}")
    if dtype == torch.float16 and device.type != "cuda":
        raise ValueError("--student-dtype fp16 requires a CUDA device")
    wrappers = [module for module in network.modules() if isinstance(module, NarrowEDMPrecond)]
    if dtype == torch.float16 and not wrappers:
        raise ValueError("Mixed FP16 requires a supported NarrowEDMPrecond student wrapper")
    for module in wrappers:
        module.use_fp16 = dtype == torch.float16


def validate_model_precision(model_metadata: dict[str, Any], protocol) -> None:
    """Prevent generation or sample reuse from mislabeling a precision protocol."""
    if protocol is None or protocol.identity not in {
        LSUN_BEDROOM256_ADM_PROTOCOL.identity,
        LSUN_BEDROOM256_ADM_MIXED_FP16_PROTOCOL.identity,
    }:
        return
    kind = model_metadata.get("kind")
    if kind not in {"distilled_checkpoint", "teacher_network"}:
        raise ValueError(f"protocol {protocol.identity} requires known model precision provenance")
    expected = (
        "mixed_fp16"
        if kind == "teacher_network" or protocol.identity == LSUN_BEDROOM256_ADM_MIXED_FP16_PROTOCOL.identity
        else "fp32"
    )
    observed = model_metadata.get("inference_precision")
    if observed != expected:
        raise ValueError(
            f"protocol {protocol.identity} requires {kind} inference_precision={expected}, got {observed!r}; "
            "use the matching precision protocol and fresh samples"
        )


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - upstream artifact identity, not a security primitive.
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_adm_detector(path: Path, protocol) -> dict[str, Any]:
    detector = path.expanduser().resolve()
    expected = protocol.detector
    if detector.name != "classify_image_graph_def.pb":
        raise ValueError(
            "OpenAI evaluator resolves its detector by basename; --adm-detector must be named "
            f"classify_image_graph_def.pb, got {detector.name!r}"
        )
    if not detector.is_file():
        raise FileNotFoundError(f"OpenAI ADM detector does not exist: {detector}")
    if expected.size_bytes is not None and detector.stat().st_size != expected.size_bytes:
        raise ValueError(
            f"OpenAI ADM detector size mismatch: expected {expected.size_bytes}, got {detector.stat().st_size}"
        )
    actual_md5 = file_md5(detector)
    if expected.md5 is not None and actual_md5 != expected.md5:
        raise ValueError(f"OpenAI ADM detector MD5 mismatch: expected {expected.md5}, got {actual_md5}")
    return {
        "path": str(detector),
        "size_bytes": int(detector.stat().st_size),
        "md5": actual_md5,
        "source_url": expected.url,
    }


def canonical_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def read_reference_metadata(path: Path) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as stats:
            missing = {"mu", "sigma"}.difference(stats.files)
            if missing:
                raise ValueError(f"reference statistics are missing keys: {sorted(missing)}")
            mu_shape = tuple(int(value) for value in stats["mu"].shape)
            sigma_shape = tuple(int(value) for value in stats["sigma"].shape)
            metadata_value = stats["metadata"].item() if "metadata" in stats.files else None
    except Exception as exc:
        raise ValueError(f"Could not read FID reference statistics {path}: {exc}") from exc

    metadata: Any = metadata_value
    if isinstance(metadata_value, bytes):
        metadata = metadata_value.decode("utf-8")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {"raw": metadata}
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {"value": metadata}
    return {
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "sha256": file_sha256(path),
        "mu_shape": list(mu_shape),
        "sigma_shape": list(sigma_shape),
        "metadata": metadata,
    }


def read_dataset_manifest_metadata(path: Path | None, *, dataset_id: str | None) -> dict[str, Any] | None:
    """Read a dataset manifest without instantiating or decoding its dataset."""

    if path is None:
        if dataset_id is None:
            return None
        return {"dataset_spec": dataclasses.asdict(dataset_spec(dataset_id)), "manifest": None}
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read dataset manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset manifest must contain a JSON object: {path}")
    manifest_dataset = payload.get("dataset_id") or payload.get("dataset")
    if dataset_id is not None and manifest_dataset not in {None, dataset_id}:
        raise ValueError(
            f"Dataset manifest {path} names dataset {manifest_dataset!r}, expected {dataset_id!r}"
        )
    resolved_dataset = dataset_id or (str(manifest_dataset) if manifest_dataset is not None else None)
    return {
        "dataset_spec": dataclasses.asdict(dataset_spec(resolved_dataset)) if resolved_dataset else None,
        "manifest": {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "size_bytes": int(path.stat().st_size),
            "format": payload.get("manifest_format") or payload.get("format"),
            "dataset_id": manifest_dataset,
            "protocol": payload.get("protocol"),
            "entries_sha256": payload.get("entries_sha256"),
            "source_listing_sha256": payload.get("source_listing_sha256"),
            "source_records_sha256": payload.get("source_records_sha256"),
        },
    }


def read_reuse_provenance(
    path: Path,
    *,
    samples_dir: Path,
    expected_seed: int,
    expected_count: int,
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read reuse provenance {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("status") != "complete":
        raise ValueError(f"Reuse provenance must be a completed evaluation result: {path}")
    if payload.get("benchmark_protocol_id") != FFHQ64_NVIDIA_PROTOCOL.identity:
        raise ValueError(
            "FFHQ ADM reused samples must originate from the canonical seed-0 NVLabs protocol run; "
            f"got {payload.get('benchmark_protocol_id')!r}"
        )
    sampling = payload.get("sampling")
    samples = payload.get("samples")
    model = payload.get("model")
    if not isinstance(sampling, dict) or not isinstance(samples, dict) or not isinstance(model, dict):
        raise ValueError(f"Reuse provenance is missing sampling/samples/model objects: {path}")
    if int(sampling.get("seed", -1)) != expected_seed or int(sampling.get("num_samples", -1)) != expected_count:
        raise ValueError(
            f"Reuse provenance seed/count mismatch: expected seed={expected_seed}, count={expected_count}; "
            f"got seed={sampling.get('seed')}, count={sampling.get('num_samples')}"
        )
    declared_directory = samples.get("secondary_directory")
    if declared_directory is None or Path(declared_directory).expanduser().resolve() != samples_dir.resolve():
        raise ValueError(
            f"Reuse provenance secondary directory {declared_directory!r} does not match --samples-dir {samples_dir}"
        )
    if samples.get("secondary_quantizer") != SampleQuantizer.OPENAI_TRUNCATE.value:
        raise ValueError(
            "Reuse provenance does not declare the OpenAI truncation quantizer required by the FFHQ ADM protocol"
        )
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "source_result_format": payload.get("result_format"),
        "source_benchmark_protocol_id": payload.get("benchmark_protocol_id"),
        "model": model,
        "teacher_spec": payload.get("teacher_spec"),
        "dataset": payload.get("dataset"),
        "sampling": sampling,
        "source_execution": payload.get("execution"),
    }


def _git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def backend_version_metadata(args: argparse.Namespace) -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pace_git_commit": _git_commit(REPO_ROOT) or "unknown",
    }
    if args.metric_backend == MetricBackend.NVLABS_EDM_FID.value:
        versions["nvlabs_edm_git_commit"] = _git_commit(args.nvlabs_edm_root) or "unknown"
    elif args.metric_backend == MetricBackend.OPENAI_ADM.value and args.adm_evaluator is not None:
        checkout = args.adm_evaluator.parent.parent
        versions["openai_guided_diffusion_git_commit"] = _git_commit(checkout) or "unknown"
        versions["adm_python"] = str(args.adm_python)
    else:
        try:
            from importlib.metadata import version

            versions["clean_fid"] = version("clean-fid")
        except Exception:
            versions["clean_fid"] = "unknown"
    return versions


def weighted_active_parameter_count(plan: dict[str, Any]) -> float | None:
    students = plan.get("students")
    num_bins = plan.get("num_sigma_bins")
    if not isinstance(students, list) or not students or not isinstance(num_bins, int) or num_bins <= 0:
        return None
    weighted_total = 0.0
    for student in students:
        block = student.get("timestep_block")
        count = student.get("full_parameter_count")
        if not isinstance(block, list) or len(block) != 2 or not isinstance(count, (int, float)):
            return None
        weighted_total += (int(block[1]) - int(block[0])) * float(count)
    return weighted_total / float(num_bins)


def load_distilled_checkpoint(
    path: Path,
    *,
    weights: str,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint must contain a dictionary payload: {path}")
    plan = payload.get("architecture_plan")
    if not isinstance(plan, dict) or not isinstance(plan.get("students"), list):
        raise ValueError(f"Checkpoint is missing a valid architecture_plan: {path}")
    if weights not in {"ema", "student"}:
        raise ValueError(f"Unsupported weights selection: {weights}")

    state_key = f"{weights}_state_dict"
    if state_key in payload:
        if not isinstance(payload[state_key], dict):
            raise ValueError(f"Checkpoint field {state_key} is not a state dictionary: {path}")
        network = construct_student_from_plan(plan)
        network.load_state_dict(payload[state_key], strict=True)
    else:
        network = payload.get(weights)
        if not isinstance(network, torch.nn.Module):
            raise ValueError(f"Checkpoint does not contain {state_key} or legacy '{weights}' weights: {path}")

    stat = path.stat()
    profile_provenance = provenance_from_artifact(payload, fallback=plan)
    metadata = {
        "kind": "distilled_checkpoint",
        "source": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "snapshot_format": payload.get("snapshot_format"),
        "checkpoint_step": int(payload["step"]) if isinstance(payload.get("step"), int) else None,
        "weights": weights,
        "architecture_variant": plan.get("variant"),
        "teacher": plan.get("teacher"),
        "dataset_spec": plan.get("dataset_spec") or plan.get("dataset_info", {}).get("dataset_spec"),
        "dataset_manifest_metadata": plan.get("dataset_info", {}).get("manifest_metadata"),
        "dataset_info": plan.get("dataset_info"),
        "average_active_parameters": weighted_active_parameter_count(plan),
        **profile_provenance,
    }
    # Retain FP32 master weights and preconditioning even for mixed inference.
    network = network.eval().requires_grad_(False).to(device=device, dtype=torch.float32)
    configure_student_precision(network, dtype=dtype, device=device)
    metadata["loaded_parameter_count"] = int(sum(parameter.numel() for parameter in network.parameters()))
    metadata["inference_precision"] = "mixed_fp16" if dtype == torch.float16 else "fp32"
    metadata["parameter_dtype"] = "float32"
    if dtype == torch.float16:
        metadata["mixed_precision_policy"] = "native_narrow_edm_use_fp16_v1"
    return network, plan, metadata


def _is_url(value: str) -> bool:
    return urllib.parse.urlparse(value).scheme in {"http", "https"}


def _broadcast_main_error(ctx: Any, error: str | None) -> None:
    if not ctx.enabled:
        if error is not None:
            raise RuntimeError(error)
        return
    payload: list[str | None] = [error if ctx.is_main else None]
    dist.broadcast_object_list(payload, src=0)
    if payload[0] is not None:
        raise RuntimeError(payload[0])


def load_teacher_network(
    source: str,
    *,
    output_dir: Path,
    device: torch.device,
    ctx: Any,
    network_format: str | None = None,
    network_preset: str | None = None,
    model_cache_dir: str | Path | None = None,
    trust_local_pickle: bool = False,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    spec = resolve_teacher_spec(source, network_format=network_format, preset=network_preset)
    local_path = materialize_teacher_source(spec, cache_dir=model_cache_dir)
    network = load_edm_network(
        spec,
        device=device,
        dtype=dtype,
        cache_dir=model_cache_dir,
        trust_local_pickle=trust_local_pickle,
    )
    resolved_spec = resolve_teacher_spec(getattr(network, "teacher_spec", spec.to_dict()))
    stat = local_path.stat()
    metadata = {
        "kind": "teacher_network",
        "source": source,
        "resolved_path": str(local_path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "checkpoint_step": None,
        "weights": "ema",
        "architecture_variant": "teacher",
        "average_active_parameters": int(sum(parameter.numel() for parameter in network.parameters())),
        "loaded_parameter_count": int(sum(parameter.numel() for parameter in network.parameters())),
        "inference_precision": "mixed_fp16" if dtype == torch.float16 else "fp32",
        "parameter_dtype": "float16" if dtype == torch.float16 else "float32",
        **teacher_model_metadata(network, resolved_spec),
    }
    return network, {"teacher": resolved_spec.to_dict()}, metadata


def build_evaluation_config(
    *,
    model_metadata: dict[str, Any],
    reference_metadata: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    benchmark_protocol = PROTOCOLS.get(getattr(args, "protocol_id", None))
    dataset_metadata = read_dataset_manifest_metadata(
        getattr(args, "dataset_manifest", None),
        dataset_id=benchmark_protocol.dataset if benchmark_protocol is not None else None,
    )
    detector_metadata = (
        validate_adm_detector(args.adm_detector, benchmark_protocol)
        if benchmark_protocol is not None
        and benchmark_protocol.backend is MetricBackend.OPENAI_ADM
        and getattr(args, "adm_detector", None) is not None
        else None
    )
    base = {
        "config_format": CONFIG_FORMAT,
        "model": model_metadata,
        "sampling": {
            "num_samples": int(args.num_samples),
            "seed": int(args.seed),
            "num_steps": int(args.num_steps),
            "nfe_per_image": int(2 * args.num_steps - 1),
            "sigma_min": float(args.sigma_min),
            "sigma_max": float(args.sigma_max),
            "rho": float(args.rho),
            "s_churn": float(args.s_churn),
            "s_min": float(args.s_min),
            "s_max": None if args.s_max is None else float(args.s_max),
            "s_noise": float(args.s_noise),
            "label_mode": str(args.label_mode),
            "clip_denoised": bool(
                PROTOCOLS[args.protocol_id].sampling.clip_denoised
                if getattr(args, "protocol_id", None) in PROTOCOLS
                else False
            ),
            "protocol_inference_dtype": (
                PROTOCOLS[args.protocol_id].sampling.inference_dtype
                if getattr(args, "protocol_id", None) in PROTOCOLS
                else None
            ),
            "model_precision": model_metadata.get("inference_precision"),
        },
        "reference": reference_metadata,
        "detector": detector_metadata,
        "metric_backend": str(getattr(args, "metric_backend", MetricBackend.CLEANFID.value)),
        "benchmark_protocol_id": getattr(args, "protocol_id", None),
        "benchmark_protocol": benchmark_protocol.to_dict() if benchmark_protocol is not None else None,
        "dataset": dataset_metadata,
        "teacher_spec": model_metadata.get("teacher"),
        "fid_mode": str(getattr(args, "fid_mode", "clean"))
        if getattr(args, "metric_backend", MetricBackend.CLEANFID.value) == MetricBackend.CLEANFID.value
        else None,
    }
    for key in ("source_profile", "ablation_protocol", "filter_sampling"):
        if isinstance(model_metadata.get(key), dict):
            base[key] = model_metadata[key]
    return {**base, "protocol_id": canonical_digest(base)}


def prepare_output_directory(
    output_dir: Path,
    config: dict[str, Any],
    *,
    samples_dir: Path | None = None,
    overwrite: bool,
    reuse_samples: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "evaluation_config.json"
    samples_dir = samples_dir or output_dir / "samples"
    result_path = output_dir / "evaluation_result.json"
    # Compare the JSON representation, not the Python objects. Metadata readers
    # may naturally return tuples (for example, NumPy array shapes), while the
    # same values are lists after an existing config is loaded from JSON.
    serialized_config = json.loads(json.dumps(config, allow_nan=False))

    if overwrite:
        if samples_dir.exists() and not reuse_samples:
            shutil.rmtree(samples_dir)
        for path in (config_path, result_path):
            if path.exists():
                path.unlink()

    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Existing evaluation config is invalid: {config_path}") from exc
        if existing != serialized_config:
            raise ValueError(
                f"Evaluation configuration does not match {config_path}. "
                "Choose another output directory or pass --overwrite."
            )
    elif samples_dir.exists() and any(samples_dir.iterdir()) and not reuse_samples:
        raise ValueError(
            f"Sample directory exists without an evaluation config: {samples_dir}. "
            "Choose another output directory or pass --overwrite."
        )
    else:
        atomic_write_json(config_path, serialized_config)


def _valid_sample(path: Path, *, resolution: int, channels: int) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            expected_mode = "RGB" if channels == 3 else "L"
            return image.size == (resolution, resolution) and image.mode == expected_mode
    except Exception:
        return False


def _save_quantized_png(image: torch.Tensor, path: Path, *, quantizer: SampleQuantizer) -> None:
    quantized = quantize_samples(image, quantizer).permute(1, 2, 0).cpu().numpy()
    if quantized.shape[2] == 1:
        pil_image = Image.fromarray(quantized[:, :, 0], mode="L")
    else:
        pil_image = Image.fromarray(quantized, mode="RGB")
    pil_image.save(path, format="PNG")


def write_adm_sample_npz(
    samples_dir: Path,
    destination: Path,
    seeds: Sequence[int],
    *,
    resolution: int,
) -> Path:
    """Stream PNG pixels into an uncompressed ADM-compatible ``arr_0`` NPZ."""

    import zipfile

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with zipfile.ZipFile(temporary, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            with archive.open("arr_0.npy", mode="w", force_zip64=True) as member:
                np.lib.format.write_array_header_2_0(
                    member,
                    {
                        "descr": np.lib.format.dtype_to_descr(np.dtype(np.uint8)),
                        "fortran_order": False,
                        "shape": (len(seeds), resolution, resolution, 3),
                    },
                )
                for seed in tqdm(seeds, desc="packing ADM sample batch", dynamic_ncols=True):
                    path = samples_dir / f"seed{seed:06d}.png"
                    with Image.open(path) as image:
                        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
                    if pixels.shape != (resolution, resolution, 3):
                        raise ValueError(f"unexpected sample shape {pixels.shape}: {path}")
                    member.write(pixels.tobytes(order="C"))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _run_checked(command: Sequence[str], *, cwd: Path, environment: dict[str, str] | None = None) -> str:
    try:
        result = subprocess.run(
            list(command),
            cwd=cwd,
            env=environment,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as exc:
        output = exc.stdout or exc.output or ""
        detail = output[-4000:].strip()
        raise RuntimeError(
            f"command failed with exit code {exc.returncode}: {' '.join(map(str, command))}"
            + (f"\n{detail}" if detail else "")
        ) from exc
    return result.stdout


def compute_nvlabs_edm_fid(
    *,
    samples_dir: Path,
    reference_stats: Path,
    num_samples: int,
    nvlabs_edm_root: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], str]:
    fid_script = nvlabs_edm_root / "fid.py"
    if not fid_script.is_file():
        raise FileNotFoundError(
            f"NVLabs FID backend requires a pinned EDM checkout containing fid.py: {nvlabs_edm_root}"
        )
    command = [
        sys.executable,
        str(fid_script),
        "calc",
        "--images",
        str(samples_dir),
        "--ref",
        str(reference_stats),
        "--num",
        str(num_samples),
    ]
    environment = os.environ.copy()
    for name in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "GROUP_RANK",
        "GROUP_WORLD_SIZE",
        "ROLE_NAME",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "TORCHELASTIC_ERROR_FILE",
        "TORCHELASTIC_MAX_RESTARTS",
        "TORCHELASTIC_RESTART_COUNT",
        "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_SIGNALS_TO_HANDLE",
        "TORCHELASTIC_USE_AGENT_STORE",
    ):
        environment.pop(name, None)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(nvlabs_edm_root), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    stdout = _run_checked(command, cwd=output_dir, environment=environment)
    candidates = [line.strip() for line in stdout.splitlines() if re.fullmatch(r"[-+0-9.eE]+", line.strip())]
    if not candidates:
        raise RuntimeError(f"could not parse FID from NVLabs output:\n{stdout[-4000:]}")
    return {"fid_nvlabs_legacy": float(candidates[-1]), "num_samples": num_samples}, stdout


def compute_openai_adm_metrics(
    *,
    sample_npz: Path,
    reference_stats: Path,
    adm_evaluator: Path,
    adm_python: Path,
    adm_detector: Path,
    output_dir: Path,
    protocol,
) -> tuple[dict[str, Any], str]:
    if not adm_python.is_file():
        raise FileNotFoundError(f"ADM backend Python executable does not exist: {adm_python}")
    if not os.access(adm_python, os.X_OK):
        raise PermissionError(f"ADM backend Python executable is not executable: {adm_python}")
    if not adm_evaluator.is_file():
        raise FileNotFoundError(f"ADM backend evaluator.py does not exist: {adm_evaluator}")
    validate_adm_detector(adm_detector, protocol)
    stdout = _run_checked(
        [str(adm_python), str(adm_evaluator), str(reference_stats), str(sample_npz)],
        cwd=adm_detector.parent,
    )
    labels = {
        "Inception Score": "inception_score_adm",
        "FID": "fid_adm",
        "sFID": "sfid_adm",
        "Precision": "precision_adm",
        "Recall": "recall_adm",
    }
    metrics: dict[str, Any] = {}
    for label, key in labels.items():
        matches = re.findall(rf"(?m)^{re.escape(label)}:\s*([-+0-9.eE]+)\s*$", stdout)
        if not matches:
            raise RuntimeError(f"could not parse {label} from ADM evaluator output:\n{stdout[-4000:]}")
        metrics[key] = float(matches[-1])
    metrics["num_samples"] = validate_adm_sample_npz(
        sample_npz,
        protocol,
        strict_count=False,
    )["count"]
    return metrics, stdout


def _make_random_labels(
    rnd: StackedRandomGenerator,
    *,
    batch_size: int,
    label_dim: int,
    device: torch.device,
) -> torch.Tensor | None:
    if label_dim <= 0:
        return None
    indices = torch.stack(
        [torch.randint(label_dim, size=[], generator=generator, device=device) for generator in rnd.generators]
    )
    labels = torch.zeros([batch_size, label_dim], device=device)
    labels.scatter_(1, indices.reshape(-1, 1), 1.0)
    return labels


def _make_evaluation_labels(
    rnd: StackedRandomGenerator,
    seeds: list[int],
    *,
    label_dim: int,
    label_mode: str,
    device: torch.device,
) -> torch.Tensor | None:
    if label_mode == "seed_modulo":
        return make_labels(seeds, label_dim, device)
    if label_mode == "random":
        return _make_random_labels(rnd, batch_size=len(seeds), label_dim=label_dim, device=device)
    raise ValueError(f"Unsupported label mode: {label_mode}")


def _cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def generate_samples(
    *,
    net: torch.nn.Module,
    samples_dir: Path,
    seeds: Sequence[int],
    batch_size: int,
    sampling_config: dict[str, Any],
    protocol_id: str,
    device: torch.device,
    rank: int,
    world_size: int,
    quantizer: SampleQuantizer = SampleQuantizer.NVLABS_ROUND,
    secondary_samples_dir: Path | None = None,
    secondary_quantizer: SampleQuantizer | None = None,
    warmup_samples: int = 0,
) -> dict[str, Any]:
    samples_dir.mkdir(parents=True, exist_ok=True)
    if secondary_samples_dir is not None:
        if secondary_quantizer is None:
            raise ValueError("secondary_samples_dir requires secondary_quantizer")
        secondary_samples_dir.mkdir(parents=True, exist_ok=True)
    resolution = int(net.img_resolution)
    channels = int(net.img_channels)
    label_dim = int(getattr(net, "label_dim", 0))
    assigned_seeds = list(seeds)[rank::world_size]
    generated_count = 0
    skipped_count = 0
    model_seconds = 0.0
    write_seconds = 0.0
    warmup_seconds = 0.0
    warmup_count = min(max(int(warmup_samples), 0), len(assigned_seeds), batch_size)
    if warmup_count:
        warmup_seeds = assigned_seeds[:warmup_count]
        warmup_random = StackedRandomGenerator(device, warmup_seeds)
        warmup_latents = warmup_random.randn(
            [warmup_count, channels, resolution, resolution],
            device=device,
            dtype=torch.float32,
        )
        warmup_labels = _make_evaluation_labels(
            warmup_random,
            warmup_seeds,
            label_dim=label_dim,
            label_mode=str(sampling_config["label_mode"]),
            device=device,
        )
        _cuda_synchronize(device)
        warmup_start = time.perf_counter()
        with torch.inference_mode():
            warmup_images = edm_sampler(
                net,
                warmup_latents,
                class_labels=warmup_labels,
                randn_like=warmup_random.randn_like,
                num_steps=int(sampling_config["num_steps"]),
                sigma_min=float(sampling_config["sigma_min"]),
                sigma_max=float(sampling_config["sigma_max"]),
                rho=float(sampling_config["rho"]),
                S_churn=float(sampling_config["s_churn"]),
                S_min=float(sampling_config["s_min"]),
                S_max=float("inf") if sampling_config["s_max"] is None else float(sampling_config["s_max"]),
                S_noise=float(sampling_config["s_noise"]),
                clip_denoised=bool(sampling_config.get("clip_denoised", False)),
            )
        _cuda_synchronize(device)
        warmup_seconds = time.perf_counter() - warmup_start
        if not torch.isfinite(warmup_images).all().item():
            raise RuntimeError(f"Non-finite warmup samples on rank {rank}, seeds {warmup_seeds}")
        del warmup_images
    wall_start = time.perf_counter()

    iterator = range(0, len(assigned_seeds), batch_size)
    for start in tqdm(
        iterator,
        desc=f"checkpoint samples rank {rank}",
        dynamic_ncols=True,
        disable=(world_size > 1 and rank != 0),
    ):
        batch_seeds = assigned_seeds[start : start + batch_size]
        pending_seeds: list[int] = []
        for sample_seed in batch_seeds:
            path = samples_dir / f"seed{sample_seed:06d}.png"
            secondary_path = (
                secondary_samples_dir / f"seed{sample_seed:06d}.png"
                if secondary_samples_dir is not None
                else None
            )
            primary_valid = _valid_sample(path, resolution=resolution, channels=channels)
            secondary_valid = secondary_path is None or _valid_sample(
                secondary_path, resolution=resolution, channels=channels
            )
            if primary_valid and secondary_valid:
                skipped_count += 1
            else:
                if path.exists():
                    path.unlink()
                if secondary_path is not None and secondary_path.exists():
                    secondary_path.unlink()
                pending_seeds.append(sample_seed)
        if not pending_seeds:
            continue

        rnd = StackedRandomGenerator(device, pending_seeds)
        latents = rnd.randn(
            [len(pending_seeds), channels, resolution, resolution],
            device=device,
            dtype=torch.float32,
        )
        labels = _make_evaluation_labels(
            rnd,
            pending_seeds,
            label_dim=label_dim,
            label_mode=str(sampling_config["label_mode"]),
            device=device,
        )
        _cuda_synchronize(device)
        model_start = time.perf_counter()
        with torch.inference_mode():
            images = edm_sampler(
                net,
                latents,
                class_labels=labels,
                randn_like=rnd.randn_like,
                num_steps=int(sampling_config["num_steps"]),
                sigma_min=float(sampling_config["sigma_min"]),
                sigma_max=float(sampling_config["sigma_max"]),
                rho=float(sampling_config["rho"]),
                S_churn=float(sampling_config["s_churn"]),
                S_min=float(sampling_config["s_min"]),
                S_max=float("inf") if sampling_config["s_max"] is None else float(sampling_config["s_max"]),
                S_noise=float(sampling_config["s_noise"]),
                clip_denoised=bool(sampling_config.get("clip_denoised", False)),
            )
        _cuda_synchronize(device)
        model_seconds += time.perf_counter() - model_start

        # Do not silently quantize NaN/Inf into apparently valid PNG files.
        # Keep validation outside the model-only throughput measurement.
        if not torch.isfinite(images).all().item():
            raise RuntimeError(f"Non-finite generated samples on rank {rank}, seeds {pending_seeds}")

        write_start = time.perf_counter()
        for sample_seed, image in zip(pending_seeds, images):
            path = samples_dir / f"seed{sample_seed:06d}.png"
            tmp_path = path.with_name(f".{path.stem}.tmp-rank{rank}-{os.getpid()}.png")
            try:
                _save_quantized_png(image, tmp_path, quantizer=quantizer)
                os.replace(tmp_path, path)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
            if secondary_samples_dir is not None:
                assert secondary_quantizer is not None
                secondary_path = secondary_samples_dir / f"seed{sample_seed:06d}.png"
                secondary_tmp = secondary_path.with_name(
                    f".{secondary_path.stem}.tmp-rank{rank}-{os.getpid()}.png"
                )
                try:
                    _save_quantized_png(image, secondary_tmp, quantizer=secondary_quantizer)
                    os.replace(secondary_tmp, secondary_path)
                finally:
                    if secondary_tmp.exists():
                        secondary_tmp.unlink()
            generated_count += 1
        write_seconds += time.perf_counter() - write_start

    row = {
        "rank": int(rank),
        "world_size": int(world_size),
        "assigned_count": len(assigned_seeds),
        "generated_count": int(generated_count),
        "skipped_count": int(skipped_count),
        "model_seconds": float(model_seconds),
        "write_seconds": float(write_seconds),
        "wall_seconds": float(time.perf_counter() - wall_start),
        "warmup_samples": warmup_count,
        "warmup_seconds": warmup_seconds,
        "quantizer": quantizer.value,
        "secondary_samples_dir": str(secondary_samples_dir) if secondary_samples_dir is not None else None,
        "secondary_quantizer": secondary_quantizer.value if secondary_quantizer is not None else None,
    }
    manifest_path = samples_dir / f"sample_manifest_world{world_size}_rank{rank}.json"
    atomic_write_json(manifest_path, {**row, "protocol_id": protocol_id})
    if secondary_samples_dir is not None:
        secondary_manifest = secondary_samples_dir / f"sample_manifest_world{world_size}_rank{rank}.json"
        atomic_write_json(
            secondary_manifest,
            {
                **row,
                "protocol_id": protocol_id,
                "artifact_role": "secondary_quantized_samples_for_followup_metric",
                "primary_samples_dir": str(samples_dir),
            },
        )
    return row


def validate_complete_sample_set(samples_dir: Path, seeds: Sequence[int]) -> None:
    expected = {f"seed{sample_seed:06d}.png" for sample_seed in seeds}
    actual = {path.name for path in samples_dir.glob("seed*.png")}
    missing = expected.difference(actual)
    extra = actual.difference(expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {len(missing)} images")
        if extra:
            details.append(f"found {len(extra)} unexpected images")
        raise ValueError(f"Incomplete FID sample set in {samples_dir}: {', '.join(details)}")


def local_hardware_metadata(ctx: Any) -> dict[str, Any]:
    metadata = {
        "rank": int(ctx.rank),
        "device": str(ctx.device),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
    }
    if ctx.device.type == "cuda":
        properties = torch.cuda.get_device_properties(ctx.device)
        metadata.update(
            {
                "gpu_name": properties.name,
                "gpu_total_memory_bytes": int(properties.total_memory),
                "cuda_capability": [int(properties.major), int(properties.minor)],
            }
        )
    return metadata


def gather_objects(ctx: Any, value: Any) -> list[Any]:
    if not ctx.enabled:
        return [value]
    gathered: list[Any] = [None] * ctx.world_size
    dist.all_gather_object(gathered, value)
    return gathered


def aggregate_generation_stats(rows: list[dict[str, Any]], *, nfe_per_image: int) -> dict[str, Any]:
    generated = sum(int(row["generated_count"]) for row in rows)
    skipped = sum(int(row["skipped_count"]) for row in rows)
    model_wall = max(float(row["model_seconds"]) for row in rows)
    generation_wall = max(float(row["wall_seconds"]) for row in rows)
    warmup_wall = max(float(row.get("warmup_seconds", 0.0)) for row in rows)
    warmup_samples = sum(int(row.get("warmup_samples", 0)) for row in rows)
    model_rate = generated / model_wall if generated > 0 and model_wall > 0 else None
    end_to_end_rate = generated / generation_wall if generated > 0 and generation_wall > 0 else None
    nfe_rate = generated * nfe_per_image / model_wall if generated > 0 and model_wall > 0 else None
    return {
        "assigned_images": sum(int(row["assigned_count"]) for row in rows),
        "generated_images": generated,
        "reused_images": skipped,
        "parallel_model_seconds": model_wall,
        "parallel_generation_wall_seconds": generation_wall,
        "parallel_warmup_seconds": warmup_wall,
        "warmup_samples": warmup_samples,
        "newly_generated": generated > 0,
        "model_only_samples_per_second": model_rate,
        "end_to_end_samples_per_second": end_to_end_rate,
        "nfe_per_second": nfe_rate,
        # Preserve the original field names for result readers written before
        # the benchmark protocol labels were finalized.
        "generation_only_images_per_second": model_rate,
        "generation_only_nfe_per_second": nfe_rate,
        "fresh_end_to_end_images_per_second": end_to_end_rate,
        "timing_protocol": "one synchronized untimed warm-up batch per rank; slowest-rank model/wall time",
        "ranks": rows,
    }


def parse_args() -> argparse.Namespace:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(description=__doc__)
    model_group = parser.add_mutually_exclusive_group(required=False)
    model_group.add_argument("--checkpoint", type=Path, help="PACE student snapshot (.pt).")
    model_group.add_argument(
        "--network-source",
        "--network-pkl",
        dest="network_pkl",
        help="Teacher checkpoint path/URL; --network-pkl is retained as a legacy alias.",
    )
    parser.add_argument("--network-format", default=None, help="Structured teacher checkpoint format.")
    parser.add_argument("--network-preset", default=None, help="Built-in teacher architecture/checkpoint preset.")
    parser.add_argument(
        "--teacher-dtype",
        choices=("auto", "fp32", "fp16"),
        default="auto",
        help="Teacher inference precision. Auto selects mixed FP16 for the built-in Bedroom protocol and FP32 otherwise.",
    )
    parser.add_argument(
        "--student-dtype",
        choices=("auto", "fp32", "fp16"),
        default="auto",
        help="Student inference precision. Auto follows the mixed-FP16 protocol, otherwise FP32; FP16 keeps master weights and preconditioning FP32.",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=None,
        help="Required cache directory for remote teacher checkpoints.",
    )
    parser.add_argument(
        "--trust-local-pickle",
        action="store_true",
        help="Allow a local/third-party NVLabs teacher pickle after verifying its origin.",
    )
    parser.add_argument("--weights", choices=("ema", "student"), default="ema")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=None,
        help="Metadata-only versioned dataset manifest embedded in the evaluation config/result.",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        help="Directory for generated PNGs. Defaults to <output-dir>/samples.",
    )
    parser.add_argument("--reference-stats", type=Path, required=True, help="Explicit CleanFID reference .npz file.")
    parser.add_argument(
        "--metric-backend",
        choices=tuple(backend.value for backend in MetricBackend),
        default=MetricBackend.CLEANFID.value,
    )
    parser.add_argument("--protocol-id", default=None, help="Explicit protocol identity from pace.evaluation_protocols.")
    parser.add_argument("--fid-mode", choices=("clean", "legacy_tensorflow", "legacy_pytorch"), default="clean")
    parser.add_argument(
        "--nvlabs-edm-root",
        type=Path,
        default=default_edm_root(),
        help="NVlabs/edm checkout with fid.py for --metric-backend nvlabs_edm (default: $EDM_REPO or ../edm).",
    )
    parser.add_argument("--adm-evaluator", type=Path, default=None)
    parser.add_argument("--adm-python", type=Path, default=None)
    parser.add_argument(
        "--adm-detector",
        type=Path,
        default=None,
        help="Canonical classify_image_graph_def.pb; the isolated evaluator runs in its parent directory.",
    )
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=18)
    parser.add_argument("--batch-size-per-rank", type=int, default=16)
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--warmup-samples-per-rank", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--label-mode", choices=("seed_modulo", "random"), default="seed_modulo")
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--s-churn", type=float, default=0.0)
    parser.add_argument("--s-min", type=float, default=0.0)
    parser.add_argument("--s-max", type=float, default=None, help="Defaults to infinity.")
    parser.add_argument("--s-noise", type=float, default=1.0)
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--distributed-timeout-minutes", type=float, default=180.0)
    parser.add_argument("--discard-samples", action="store_true")
    parser.add_argument("--artifact-retention", choices=tuple(policy.value for policy in ArtifactRetention), default=None)
    parser.add_argument("--preview-count", type=int, default=64)
    parser.add_argument("--preview-dir", type=Path, default=None)
    parser.add_argument("--secondary-samples-dir", type=Path, default=None)
    parser.add_argument("--secondary-quantizer", choices=tuple(item.value for item in SampleQuantizer), default=None)
    parser.add_argument("--reuse-samples", action="store_true", help="Skip model generation and evaluate a complete existing --samples-dir.")
    parser.add_argument(
        "--reuse-provenance",
        type=Path,
        default=None,
        help="Completed source evaluation_result.json proving model/dataset/quantizer identity for --reuse-samples.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    protocol = resolve_declared_protocol(args)
    if args.checkpoint is None and args.network_pkl is None and args.network_preset is None:
        raise ValueError("Specify --checkpoint, --network-pkl, or --network-preset")
    if args.checkpoint is not None and args.network_preset is not None:
        raise ValueError("--network-preset cannot be combined with --checkpoint")
    if args.network_pkl is None and args.network_preset is not None:
        args.network_pkl = teacher_preset_config(args.network_preset)["source"]
    if args.network_pkl is not None and _is_url(args.network_pkl) and args.model_cache_dir is None:
        raise ValueError("Remote teacher checkpoints require --model-cache-dir")
    if args.num_samples < 2:
        raise ValueError("--num-samples must be at least 2 for FID covariance estimation")
    if args.num_steps < 2:
        raise ValueError("--num-steps must be at least 2")
    if args.batch_size_per_rank <= 0:
        raise ValueError("--batch-size-per-rank must be positive")
    if args.feature_batch_size <= 0:
        raise ValueError("--feature-batch-size must be positive")
    if args.warmup_samples_per_rank < 0:
        raise ValueError("--warmup-samples-per-rank cannot be negative")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.sigma_min <= 0 or args.sigma_max <= 0 or args.sigma_min > args.sigma_max:
        raise ValueError("Require 0 < --sigma-min <= --sigma-max")
    if args.rho <= 0:
        raise ValueError("--rho must be positive")
    if args.preview_count < 0:
        raise ValueError("--preview-count cannot be negative")
    if args.discard_samples and args.artifact_retention not in {None, ArtifactRetention.DISCARD.value}:
        raise ValueError("--discard-samples conflicts with --artifact-retention")
    if args.artifact_retention is None:
        args.artifact_retention = ArtifactRetention.DISCARD.value if args.discard_samples else ArtifactRetention.KEEP.value
    if args.reuse_samples and args.samples_dir is None:
        raise ValueError("--reuse-samples requires --samples-dir")
    if args.reuse_provenance is not None and not args.reuse_samples:
        raise ValueError("--reuse-provenance requires --reuse-samples")
    if args.reuse_samples and protocol is not None and args.reuse_provenance is None:
        raise ValueError("versioned benchmark protocols require --reuse-provenance with --reuse-samples")
    if (args.secondary_samples_dir is None) != (args.secondary_quantizer is None):
        raise ValueError("--secondary-samples-dir and --secondary-quantizer must be provided together")
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.dataset_manifest is not None:
        args.dataset_manifest = args.dataset_manifest.expanduser().resolve()
        if not args.dataset_manifest.is_file():
            raise FileNotFoundError(f"Dataset manifest does not exist: {args.dataset_manifest}")
    if args.samples_dir is not None:
        args.samples_dir = args.samples_dir.expanduser().resolve()
        if args.output_dir == args.samples_dir or args.output_dir.is_relative_to(args.samples_dir):
            raise ValueError("--samples-dir cannot be --output-dir or one of its parents")
    if args.secondary_samples_dir is not None:
        args.secondary_samples_dir = args.secondary_samples_dir.expanduser().resolve()
    if args.preview_dir is not None:
        args.preview_dir = args.preview_dir.expanduser().resolve()
    if args.reuse_provenance is not None:
        args.reuse_provenance = args.reuse_provenance.expanduser().resolve()
        if not args.dry_run and not args.reuse_provenance.is_file():
            raise FileNotFoundError(f"Reuse provenance does not exist: {args.reuse_provenance}")
    args.reference_stats = args.reference_stats.expanduser().resolve()
    if not args.reference_stats.is_file():
        raise FileNotFoundError(f"FID reference statistics do not exist: {args.reference_stats}")
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.expanduser().resolve()
        if not args.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
    if args.model_cache_dir is not None:
        args.model_cache_dir = args.model_cache_dir.expanduser().resolve()
    if not args.reuse_samples:
        is_student = args.checkpoint is not None
        dtype_resolver = resolve_student_dtype if is_student else resolve_teacher_dtype
        requested_dtype = dtype_resolver(args, protocol, device=torch.device(args.device))
        validate_model_precision(
            {
                "kind": "distilled_checkpoint" if is_student else "teacher_network",
                "inference_precision": "mixed_fp16" if requested_dtype == torch.float16 else "fp32",
            },
            protocol,
        )
    args.nvlabs_edm_root = args.nvlabs_edm_root.expanduser().resolve()
    if args.adm_evaluator is not None:
        args.adm_evaluator = args.adm_evaluator.expanduser().resolve()
    if args.adm_python is not None:
        # Preserve a virtual environment's ``bin/python`` entry point instead
        # of resolving its symlink to the base interpreter.
        args.adm_python = Path(os.path.abspath(args.adm_python.expanduser()))
    if args.adm_detector is not None:
        args.adm_detector = args.adm_detector.expanduser().resolve()
    if protocol is not None:
        if args.num_samples != protocol.sample_count:
            raise ValueError(
                f"protocol {protocol.identity} requires --num-samples {protocol.sample_count}, got {args.num_samples}"
            )
        expected = protocol.sampling
        observed = {
            "num_steps": args.num_steps,
            "sigma_min": args.sigma_min,
            "sigma_max": args.sigma_max,
            "rho": args.rho,
            "s_churn": args.s_churn,
            "s_min": args.s_min,
            "s_max": args.s_max,
            "s_noise": args.s_noise,
        }
        for key, value in observed.items():
            expected_value = getattr(expected, key)
            if value != expected_value:
                raise ValueError(
                    f"protocol {protocol.identity} requires {key}={expected_value!r}, got {value!r}"
                )
        observed_seed_range = (args.seed, args.seed + args.num_samples - 1)
        if observed_seed_range not in expected.seed_ranges:
            raise ValueError(
                f"protocol {protocol.identity} requires one of seed ranges {list(expected.seed_ranges)}, "
                f"got {observed_seed_range}"
            )
        if expected.clip_denoised and args.network_preset not in {None, "lsun_bedroom_256"} and args.checkpoint is None:
            raise ValueError(f"protocol {protocol.identity} requires clip_denoised=True model semantics")
        validate_reference_npz(args.reference_stats, protocol, verify_checksum=True)
    elif args.metric_backend == MetricBackend.CLEANFID.value:
        validate_untyped_cleanfid_reference(args.reference_stats, verify_known_checksum=True)
    if args.metric_backend == MetricBackend.NVLABS_EDM_FID.value and not args.dry_run:
        if not (args.nvlabs_edm_root / "fid.py").is_file():
            raise FileNotFoundError(f"NVLabs EDM checkout is missing fid.py: {args.nvlabs_edm_root}")
    if args.metric_backend == MetricBackend.OPENAI_ADM.value and not args.dry_run:
        if args.adm_evaluator is None or args.adm_python is None or args.adm_detector is None:
            raise ValueError(
                "OpenAI ADM backend requires --adm-evaluator, --adm-python, and --adm-detector from the "
                "pinned isolated environment"
            )
        assert protocol is not None
        validate_adm_detector(args.adm_detector, protocol)


def dry_run_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dry_run": True,
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "network_pkl": args.network_pkl,
        "network_format": args.network_format,
        "network_preset": args.network_preset,
        "teacher_dtype": args.teacher_dtype,
        "student_dtype": getattr(args, "student_dtype", "auto"),
        "resolved_student_dtype": (
            "fp16"
            if resolve_student_dtype(args, resolve_declared_protocol(args), device=torch.device(args.device))
            == torch.float16
            else "fp32"
        ),
        "resolved_teacher_dtype": (
            "fp16"
            if resolve_teacher_dtype(args, resolve_declared_protocol(args), device=torch.device(args.device))
            == torch.float16
            else "fp32"
        ),
        "model_cache_dir": str(args.model_cache_dir) if args.model_cache_dir is not None else None,
        "trust_local_pickle": bool(args.trust_local_pickle),
        "weights": args.weights,
        "output_dir": str(args.output_dir),
        "dataset_manifest": str(args.dataset_manifest) if args.dataset_manifest is not None else None,
        "samples_dir": str(args.samples_dir or args.output_dir / "samples"),
        "reference_stats": str(args.reference_stats),
        "metric_backend": args.metric_backend,
        "benchmark_protocol_id": args.protocol_id,
        "fid_mode": args.fid_mode if args.metric_backend == MetricBackend.CLEANFID.value else None,
        "nvlabs_edm_root": str(args.nvlabs_edm_root),
        "adm_evaluator": str(args.adm_evaluator) if args.adm_evaluator is not None else None,
        "adm_python": str(args.adm_python) if args.adm_python is not None else None,
        "adm_detector": str(args.adm_detector) if args.adm_detector is not None else None,
        "num_samples": args.num_samples,
        "seed_range": [args.seed, args.seed + args.num_samples - 1],
        "num_steps": args.num_steps,
        "nfe_per_image": 2 * args.num_steps - 1,
        "batch_size_per_rank": args.batch_size_per_rank,
        "warmup_samples_per_rank": args.warmup_samples_per_rank,
        "label_mode": args.label_mode,
        "artifact_retention": args.artifact_retention,
        "preview_count": args.preview_count,
        "preview_dir": str(args.preview_dir) if args.preview_dir is not None else None,
        "secondary_samples_dir": str(args.secondary_samples_dir) if args.secondary_samples_dir is not None else None,
        "secondary_quantizer": args.secondary_quantizer,
        "reuse_samples": args.reuse_samples,
        "reuse_provenance": str(args.reuse_provenance) if args.reuse_provenance is not None else None,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    declared_protocol = resolve_declared_protocol(args)
    if args.dry_run:
        print(json.dumps(dry_run_payload(args), indent=2, sort_keys=True))
        return

    ctx = init_distributed(args.device, timeout_minutes=args.distributed_timeout_minutes)
    total_start = time.perf_counter()
    try:
        samples_dir = args.samples_dir or args.output_dir / "samples"
        if args.reuse_samples:
            network = None
            _plan = {}
            if args.reuse_provenance is not None:
                reuse_provenance = read_reuse_provenance(
                    args.reuse_provenance,
                    samples_dir=samples_dir,
                    expected_seed=args.seed,
                    expected_count=args.num_samples,
                )
                model_metadata = dict(reuse_provenance["model"])
                model_metadata["reuse_provenance"] = {
                    key: value for key, value in reuse_provenance.items() if key != "model"
                }
                if model_metadata.get("teacher") is None and reuse_provenance.get("teacher_spec") is not None:
                    model_metadata["teacher"] = reuse_provenance["teacher_spec"]
            else:
                reuse_provenance = None
                model_metadata = {
                    "kind": "reused_samples",
                    "source": str(samples_dir),
                    "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
                    "network_preset": args.network_preset,
                    "inference_precision": "artifact_only_unknown",
                    "parameter_dtype": None,
                }
        elif args.checkpoint is not None:
            network, _plan, model_metadata = load_distilled_checkpoint(
                args.checkpoint,
                weights=args.weights,
                device=ctx.device,
                dtype=resolve_student_dtype(args, declared_protocol, device=ctx.device),
            )
        else:
            teacher_dtype = resolve_teacher_dtype(args, declared_protocol, device=ctx.device)
            network, _plan, model_metadata = load_teacher_network(
                args.network_pkl,
                output_dir=args.output_dir,
                device=ctx.device,
                ctx=ctx,
                network_format=args.network_format,
                network_preset=args.network_preset,
                model_cache_dir=args.model_cache_dir,
                trust_local_pickle=args.trust_local_pickle,
                dtype=teacher_dtype,
            )

        validate_model_precision(model_metadata, declared_protocol)
        if ctx.is_main:
            print(f"Model inference precision: {model_metadata.get('inference_precision')}", flush=True)

        if declared_protocol is not None:
            reference_metadata = validate_reference_npz(args.reference_stats, declared_protocol, verify_checksum=True)
        else:
            validate_untyped_cleanfid_reference(args.reference_stats, verify_known_checksum=True)
            reference_metadata = read_reference_metadata(args.reference_stats)
        config = build_evaluation_config(
            model_metadata=model_metadata,
            reference_metadata=reference_metadata,
            args=args,
        )
        prepare_error = None
        if ctx.is_main:
            try:
                prepare_output_directory(
                    args.output_dir,
                    config,
                    samples_dir=samples_dir,
                    overwrite=args.overwrite,
                    reuse_samples=args.reuse_samples,
                )
            except Exception as exc:
                prepare_error = str(exc)
        _broadcast_main_error(ctx, prepare_error)
        if ctx.enabled:
            dist.barrier()

        all_seeds = list(range(args.seed, args.seed + args.num_samples))
        local_generation: dict[str, Any] | None = None
        local_error = None
        try:
            if args.reuse_samples:
                assigned = len(all_seeds[ctx.rank :: ctx.world_size])
                local_generation = {
                    "rank": ctx.rank,
                    "world_size": ctx.world_size,
                    "assigned_count": assigned,
                    "generated_count": 0,
                    "skipped_count": assigned,
                    "model_seconds": 0.0,
                    "write_seconds": 0.0,
                    "wall_seconds": 0.0,
                    "warmup_samples": 0,
                    "warmup_seconds": 0.0,
                    "quantizer": declared_protocol.quantizer.value if declared_protocol is not None else None,
                    "reused_complete_set": True,
                }
            else:
                primary_quantizer = (
                    declared_protocol.quantizer if declared_protocol is not None else SampleQuantizer.NVLABS_ROUND
                )
                local_generation = generate_samples(
                    net=network,
                    samples_dir=samples_dir,
                    seeds=all_seeds,
                    batch_size=args.batch_size_per_rank,
                    sampling_config=config["sampling"],
                    protocol_id=config["protocol_id"],
                    device=ctx.device,
                    rank=ctx.rank,
                    world_size=ctx.world_size,
                    quantizer=primary_quantizer,
                    secondary_samples_dir=args.secondary_samples_dir,
                    secondary_quantizer=SampleQuantizer(args.secondary_quantizer)
                    if args.secondary_quantizer is not None
                    else None,
                    warmup_samples=args.warmup_samples_per_rank,
                )
        except Exception as exc:
            local_error = f"rank {ctx.rank}: {type(exc).__name__}: {exc}"
        generation_errors = gather_objects(ctx, local_error)
        errors = [error for error in generation_errors if error is not None]
        if errors:
            raise RuntimeError("Sample generation failed: " + "; ".join(errors))
        assert local_generation is not None

        generation_rows = gather_objects(ctx, local_generation)
        hardware_rows = gather_objects(ctx, local_hardware_metadata(ctx))
        if ctx.enabled:
            dist.barrier()

        # NVLabs fid.py initializes its own NCCL process group. Running it while
        # this evaluator's sampling group is still alive can deadlock on the
        # rank-0 GPU. Sampling and validation are complete at this point, so
        # release the worker group and let rank 0 perform the metric alone.
        released_metric_workers = ctx.enabled and args.metric_backend == MetricBackend.NVLABS_EDM_FID.value
        if released_metric_workers:
            dist.destroy_process_group()
            if not ctx.is_main:
                return
            if ctx.device.type == "cuda":
                torch.cuda.empty_cache()

        metric_error = None
        result: dict[str, Any] | None = None
        if ctx.is_main:
            try:
                validate_complete_sample_set(samples_dir, all_seeds)
                if args.secondary_samples_dir is not None:
                    validate_complete_sample_set(args.secondary_samples_dir, all_seeds)
                metric_start = time.perf_counter()
                metric_stdout = None
                metric_artifact = None
                if args.metric_backend == MetricBackend.CLEANFID.value:
                    from cleanfid import fid

                    fid_value = compute_fid_with_reference_stats(
                        fid_module=fid,
                        sample_dir=samples_dir,
                        stats_path=args.reference_stats,
                        mode=args.fid_mode,
                        feature_batch_size=args.feature_batch_size,
                        num_workers=args.num_workers,
                        device=ctx.device,
                    )
                    metrics = {
                        f"fid_clean_{args.fid_mode}": float(fid_value),
                        "num_samples": int(args.num_samples),
                    }
                elif args.metric_backend == MetricBackend.NVLABS_EDM_FID.value:
                    metrics, metric_stdout = compute_nvlabs_edm_fid(
                        samples_dir=samples_dir,
                        reference_stats=args.reference_stats,
                        num_samples=args.num_samples,
                        nvlabs_edm_root=args.nvlabs_edm_root,
                        output_dir=args.output_dir,
                    )
                else:
                    assert args.adm_evaluator is not None and args.adm_python is not None and args.adm_detector is not None
                    sample_npz = args.output_dir / "adm_samples.npz"
                    write_adm_sample_npz(
                        samples_dir,
                        sample_npz,
                        all_seeds,
                        resolution=declared_protocol.resolution,
                    )
                    validate_adm_sample_npz(sample_npz, declared_protocol, strict_count=True)
                    metrics, metric_stdout = compute_openai_adm_metrics(
                        sample_npz=sample_npz,
                        reference_stats=args.reference_stats,
                        adm_evaluator=args.adm_evaluator,
                        adm_python=args.adm_python,
                        adm_detector=args.adm_detector,
                        output_dir=args.output_dir,
                        protocol=declared_protocol,
                    )
                    metric_artifact = str(sample_npz)
                metric_seconds = time.perf_counter() - metric_start
                if metric_stdout is not None:
                    (args.output_dir / "metric_backend.log").write_text(metric_stdout)
                generation_summary = aggregate_generation_stats(
                    generation_rows,
                    nfe_per_image=int(config["sampling"]["nfe_per_image"]),
                )
                result = {
                    "result_format": RESULT_FORMAT,
                    "status": "complete",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "protocol_id": config["protocol_id"],
                    "benchmark_protocol_id": args.protocol_id,
                    "benchmark_protocol": config["benchmark_protocol"],
                    "dataset": config["dataset"],
                    "teacher_spec": config["teacher_spec"],
                    "metric_backend": args.metric_backend,
                    "model": config["model"],
                    "sampling": config["sampling"],
                    "reference": config["reference"],
                    "detector": config["detector"],
                    "metrics": metrics,
                    "execution": {
                        "world_size": int(ctx.world_size),
                        "batch_size_per_rank": int(args.batch_size_per_rank),
                        "feature_batch_size": int(args.feature_batch_size),
                        "num_workers": int(args.num_workers),
                        "generation": generation_summary,
                        "metric_seconds": float(metric_seconds),
                        "total_wall_seconds": float(time.perf_counter() - total_start),
                        "hardware": hardware_rows,
                    },
                    "samples": {
                        "directory": str(samples_dir),
                        "secondary_directory": str(args.secondary_samples_dir)
                        if args.secondary_samples_dir is not None
                        else None,
                        "secondary_quantizer": args.secondary_quantizer,
                        "secondary_retention": "deferred_to_followup_metric"
                        if args.secondary_samples_dir is not None
                        else None,
                        "metric_artifact": metric_artifact,
                        "retention": args.artifact_retention,
                    },
                }
                for key in ("source_profile", "ablation_protocol", "filter_sampling"):
                    if key in config:
                        result[key] = config[key]
                atomic_write_json(args.output_dir / "evaluation_result.json", result)
                png_paths = [samples_dir / f"seed{seed:06d}.png" for seed in all_seeds]
                retention_result = apply_artifact_retention(
                    root=samples_dir,
                    samples=png_paths,
                    retention=args.artifact_retention,
                    evaluation_succeeded=True,
                    preview_dir=args.preview_dir or args.output_dir / "preview",
                    preview_count=args.preview_count,
                )
                metric_artifact_discarded = False
                if (
                    metric_artifact is not None
                    and args.artifact_retention != ArtifactRetention.KEEP.value
                    and Path(metric_artifact).is_file()
                ):
                    Path(metric_artifact).unlink()
                    metric_artifact_discarded = True
                result["samples"]["retention_result"] = retention_result
                result["samples"]["metric_artifact_discarded"] = metric_artifact_discarded
                if declared_protocol is not None:
                    teacher_payload = config.get("teacher_spec")
                    if not isinstance(teacher_payload, dict) or not teacher_payload.get("source"):
                        raise ProtocolError(
                            "versioned benchmark results require TeacherSpec provenance with a non-empty source"
                        )
                    benchmark_manifest = build_benchmark_manifest(
                        declared_protocol,
                        teacher=teacher_payload,
                        artifacts={
                            "model": result["model"],
                            "source_profile": result.get("source_profile"),
                            "ablation_protocol": result.get("ablation_protocol"),
                            "filter_sampling": result.get("filter_sampling"),
                            "dataset": result["dataset"],
                            "reference": result["reference"],
                            "detector": result["detector"],
                            "samples": result["samples"],
                            "metrics": result["metrics"],
                        },
                        runtime=result["execution"],
                        versions=backend_version_metadata(args),
                    )
                    benchmark_manifest_path = args.output_dir / "benchmark_manifest.json"
                    atomic_write_json(benchmark_manifest_path, benchmark_manifest)
                    result["benchmark_manifest"] = {
                        "path": str(benchmark_manifest_path),
                        "manifest_sha256": benchmark_manifest["manifest_sha256"],
                    }
                atomic_write_json(args.output_dir / "evaluation_result.json", result)
            except Exception as exc:
                metric_error = f"{type(exc).__name__}: {exc}"
        if released_metric_workers:
            if metric_error is not None:
                raise RuntimeError(metric_error)
        else:
            _broadcast_main_error(ctx, metric_error)
            if ctx.enabled:
                dist.barrier()
        if ctx.is_main:
            assert result is not None
            print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    finally:
        cleanup_distributed(ctx)


if __name__ == "__main__":
    main()
