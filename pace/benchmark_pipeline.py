"""Operational preflight and resumable launch planning for EDM benchmarks.

The ``ffhq64`` and ``lsun_bedroom256`` presets drive the full U-Net pipeline from
one command.  The reference commands of the paper runs are the plain scripts in
``reproduce/``; the LSUN profile additionally used module-stratified filter
sampling and a sigma stride of 4, which this driver does not generate.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .evaluation_protocols import (
    FFHQ64_ADM_CUSTOM_PROTOCOL,
    FFHQ64_NVIDIA_PROTOCOL,
    LSUN_BEDROOM256_ADM_PROTOCOL,
    atomic_write_json,
    canonical_digest,
    estimated_sample_bytes,
    file_digest,
    validate_reference_npz,
)


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Datasets, teacher caches, training checkpoints and benchmark samples live on a
# large disk.  Set $PACE_SHARED_ROOT (or pass --shared-root) to choose it.
DEFAULT_SHARED_ROOT = Path(os.environ.get("PACE_SHARED_ROOT") or DEFAULT_PROJECT_ROOT / "shared")
PIPELINE_STATE_FORMAT = "diffdist_edm_pipeline_state_v1"
PFI_PROFILE_PROTOCOL_ID = "batch_local_exact_sigma_pfi_v1"
PFI_OUTPUT_NAMESPACE = "pfi_batch_local_exact_sigma_v1"
SUPPORTED_PROFILE_ABLATION_MODES = ("pfi", "zero", "random_same_norm")
SUPPORTED_GROUPING_BUILTIN_COSTS = (
    "matrix_sse",
    "matrix_cosine",
    "matrix_correlation",
    "matrix_cosine_cross_penalty",
    "matrix_correlation_cross_penalty",
)
SUPPORTED_PIPELINE_VARIANTS = (
    "global",
    "uniform_blockwise",
    "blockwise_capacity",
    "shuffled_capacity",
    "layerwise_capacity",
    "reversed_layerwise_capacity",
    "combined_blockwise",
    "combined_layerwise",
)
SUPPORTED_ALLOCATION_SCORE_REDUCTIONS = ("mean", "sum", "max", "q90")


class PreflightError(RuntimeError):
    """Raised when production launch preconditions are not met."""


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status in {"pass", "warning"}


@dataclass(frozen=True)
class GPUStatus:
    index: int
    free_memory_mib: int
    total_memory_mib: int
    utilization_percent: int


@dataclass(frozen=True)
class PipelinePreset:
    name: str
    dataset: str
    resolution: int
    teacher_preset: str
    default_gpu_ids: tuple[int, ...]
    required_gpu_count: int
    min_free_memory_mib: int
    min_free_storage_gib: int
    expected_source_images: int
    profile_images: int
    profile_batch_size: int
    train_batch_size: int
    train_microbatch: int
    train_steps: int
    train_dtype: str
    num_timestep_blocks: int
    grouping_builtin_cost: str
    allocation_metric: str
    score_reduction: str
    default_variants: tuple[str, ...]
    shuffle_seed: int
    benchmark_protocols: tuple[str, ...]
    profile_ablation_mode: str = "pfi"


PIPELINE_PRESETS: dict[str, PipelinePreset] = {
    "ffhq64": PipelinePreset(
        name="ffhq64",
        dataset="ffhq",
        resolution=64,
        teacher_preset="ffhq_64_vp",
        default_gpu_ids=tuple(range(8)),
        required_gpu_count=8,
        min_free_memory_mib=90_000,
        min_free_storage_gib=100,
        expected_source_images=70_000,
        # The paper profile: 100 monitor images in one PFI batch per sigma level.
        profile_images=100,
        profile_batch_size=256,
        train_batch_size=512,
        train_microbatch=16,
        train_steps=50_000,
        train_dtype="fp32",
        num_timestep_blocks=3,
        grouping_builtin_cost="matrix_correlation",
        allocation_metric="delta_p_eff_geomean",
        score_reduction="sum",
        default_variants=(
            "global",
            "uniform_blockwise",
            "combined_blockwise",
            "combined_layerwise",
        ),
        shuffle_seed=3,
        profile_ablation_mode="pfi",
        benchmark_protocols=(FFHQ64_NVIDIA_PROTOCOL.identity, FFHQ64_ADM_CUSTOM_PROTOCOL.identity),
    ),
    "lsun_bedroom256": PipelinePreset(
        name="lsun_bedroom256",
        dataset="lsun_bedroom",
        resolution=256,
        teacher_preset="lsun_bedroom_256",
        default_gpu_ids=tuple(range(8)),
        required_gpu_count=8,
        min_free_memory_mib=90_000,
        min_free_storage_gib=150,
        expected_source_images=1_000_000,
        # The paper profile: 100 monitor images in one PFI batch per sigma level.
        profile_images=100,
        profile_batch_size=100,
        train_batch_size=64,
        train_microbatch=1,
        train_steps=50_000,
        train_dtype="fp16",
        # Automatic phase selection on the paper profile gives K = 2 ([0,16,20]).
        num_timestep_blocks=2,
        grouping_builtin_cost="matrix_correlation_cross_penalty",
        allocation_metric="delta_p_eff_geomean",
        score_reduction="sum",
        default_variants=(
            "global",
            "uniform_blockwise",
            "combined_blockwise",
            "combined_layerwise",
        ),
        shuffle_seed=3,
        profile_ablation_mode="pfi",
        benchmark_protocols=(LSUN_BEDROOM256_ADM_PROTOCOL.identity,),
    ),
}


def parse_gpu_ids(value: str | Sequence[int]) -> tuple[int, ...]:
    raw = value if not isinstance(value, str) else value.replace(",", " ").split()
    try:
        parsed = tuple(int(item) for item in raw)
    except (TypeError, ValueError) as exc:
        raise PreflightError(f"GPU IDs must be comma/space-separated integers, got {value!r}") from exc
    if not parsed or any(item < 0 for item in parsed) or len(set(parsed)) != len(parsed):
        raise PreflightError(f"GPU IDs must be unique non-negative integers, got {parsed}")
    return parsed


def query_gpu_status(command: Sequence[str] | None = None) -> dict[int, GPUStatus]:
    query = list(command or [
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    try:
        result = subprocess.run(query, check=True, text=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PreflightError(f"could not query GPUs with {shlex.join(query)}: {exc}") from exc
    statuses: dict[int, GPUStatus] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        pieces = [piece.strip() for piece in line.split(",")]
        if len(pieces) != 4:
            raise PreflightError(f"unexpected nvidia-smi row: {line!r}")
        status = GPUStatus(*(int(piece) for piece in pieces))
        statuses[status.index] = status
    if not statuses:
        raise PreflightError("nvidia-smi returned no GPUs")
    return statuses


def check_gpus(gpu_ids: Sequence[int], *, required_count: int, min_free_memory_mib: int) -> Check:
    try:
        statuses = query_gpu_status()
    except PreflightError as exc:
        return Check("gpus", "fail", str(exc))
    missing = [index for index in gpu_ids if index not in statuses]
    busy = [
        index
        for index in gpu_ids
        if index in statuses and statuses[index].free_memory_mib < min_free_memory_mib
    ]
    details = {
        "requested": list(gpu_ids),
        "required_count": required_count,
        "min_free_memory_mib": min_free_memory_mib,
        "observed": {str(index): dataclasses.asdict(statuses[index]) for index in gpu_ids if index in statuses},
        "missing": missing,
        "below_threshold": busy,
    }
    if len(gpu_ids) != required_count:
        return Check("gpus", "fail", f"preset requires exactly {required_count} GPUs; got {len(gpu_ids)}", details)
    if missing:
        return Check("gpus", "fail", f"requested GPU IDs do not exist: {missing}", details)
    if busy:
        return Check(
            "gpus",
            "fail",
            f"GPUs below {min_free_memory_mib} MiB free: {busy}; refusing to share occupied GPUs",
            details,
        )
    return Check("gpus", "pass", f"all requested GPUs have at least {min_free_memory_mib} MiB free", details)


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def check_storage(artifact_root: str | Path, *, min_free_storage_gib: int) -> Check:
    root = Path(artifact_root).expanduser().resolve()
    parent = _nearest_existing_parent(root)
    try:
        usage = shutil.disk_usage(parent)
    except OSError as exc:
        return Check("storage", "fail", f"cannot inspect storage for {root}: {exc}")
    free_gib = usage.free / 2**30
    details = {
        "artifact_root": str(root),
        "filesystem_probe": str(parent),
        "free_bytes": usage.free,
        "free_gib": free_gib,
        "minimum_free_gib": min_free_storage_gib,
    }
    if free_gib < min_free_storage_gib:
        return Check("storage", "fail", f"only {free_gib:.1f} GiB free; need {min_free_storage_gib} GiB", details)
    return Check("storage", "pass", f"{free_gib:.1f} GiB free at {parent}", details)


def _mode_readable(file_stat: os.stat_result) -> bool:
    mode = file_stat.st_mode
    if file_stat.st_uid == os.geteuid():
        return bool(mode & stat.S_IRUSR)
    groups = set(os.getgroups()) | {os.getegid()}
    if file_stat.st_gid in groups:
        return bool(mode & stat.S_IRGRP)
    return bool(mode & stat.S_IROTH)


def inspect_lsun_flat_directory(root: str | Path, *, expected_count: int = 1_000_000) -> dict[str, Any]:
    """Check the fixed LSUN source contract without decoding a million JPEGs."""

    source = Path(root)
    if not source.is_dir():
        raise PreflightError(f"LSUN source directory does not exist: {source}")
    count = 0
    numeric_indices: list[int] = []
    invalid_names: list[str] = []
    subdirectories: list[str] = []
    unreadable = 0
    modes: Counter[str] = Counter()
    with os.scandir(source) as entries:
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                if len(subdirectories) < 20:
                    subdirectories.append(entry.name)
                continue
            if not entry.is_file(follow_symlinks=False):
                continue
            count += 1
            path = Path(entry.path)
            stem = path.stem
            if path.suffix.lower() != ".jpg" or len(stem) != 7 or not stem.isdigit():
                if len(invalid_names) < 20:
                    invalid_names.append(entry.name)
            else:
                numeric_indices.append(int(stem))
            file_stat = entry.stat(follow_symlinks=False)
            modes[f"{stat.S_IMODE(file_stat.st_mode):03o}"] += 1
            if not _mode_readable(file_stat) or not os.access(path, os.R_OK):
                unreadable += 1
    contiguous = bool(numeric_indices) and len(set(numeric_indices)) == len(numeric_indices)
    if contiguous:
        contiguous = max(numeric_indices) - min(numeric_indices) + 1 == len(numeric_indices)
    return {
        "root": str(source.resolve()),
        "image_count": count,
        "expected_count": expected_count,
        "invalid_name_examples": invalid_names,
        "subdirectory_examples": subdirectories,
        "numeric_min": min(numeric_indices) if numeric_indices else None,
        "numeric_max": max(numeric_indices) if numeric_indices else None,
        "contiguous_numeric_names": contiguous,
        "unreadable_count": unreadable,
        "permission_modes": dict(sorted(modes.items())),
        "image_contract": "RGB; short side 256; variable width; deterministic center crop to 256",
    }


def check_dataset_source(preset: PipelinePreset, source_root: str | Path, *, scan_lsun: bool = True) -> Check:
    source = Path(source_root).expanduser()
    if not source.exists():
        return Check("dataset_source", "fail", f"dataset source does not exist: {source}")
    if preset.dataset != "lsun_bedroom":
        if not os.access(source, os.R_OK):
            return Check("dataset_source", "fail", f"dataset source is not readable: {source}")
        return Check(
            "dataset_source",
            "pass",
            "FFHQ source is present; the preparation stage performs full decode/count validation",
            {"source": str(source.resolve()), "expected_images": preset.expected_source_images},
        )
    if not scan_lsun:
        return Check("dataset_source", "warning", "LSUN full permission/name scan was explicitly skipped")
    try:
        details = inspect_lsun_flat_directory(source, expected_count=preset.expected_source_images)
    except PreflightError as exc:
        return Check("dataset_source", "fail", str(exc))
    failures: list[str] = []
    if details["image_count"] != preset.expected_source_images:
        failures.append(f"expected {preset.expected_source_images} files, found {details['image_count']}")
    if details["subdirectory_examples"]:
        failures.append("source contains subdirectories")
    if details["invalid_name_examples"]:
        failures.append("files are not contiguous seven-digit JPG names")
    if not details["contiguous_numeric_names"]:
        failures.append("numeric filenames are not contiguous")
    if details["unreadable_count"]:
        failures.append(f"{details['unreadable_count']} files are unreadable")
    if failures:
        return Check("dataset_source", "fail", "; ".join(failures), details)
    return Check("dataset_source", "pass", "LSUN flat source contract and permissions passed", details)


def check_benchmark_dependencies(
    preset: PipelinePreset,
    *,
    shared_root: str | Path,
    ffhq_adm_reference: str | Path | None,
    nvlabs_edm_root: str | Path | None,
    adm_evaluator: str | Path | None,
    adm_python: str | Path | None,
    adm_detector: str | Path | None,
) -> Check:
    """Validate external metric adapters and feature-space-specific references."""

    layout = pipeline_layout(preset, shared_root)
    missing: list[str] = []
    invalid: list[str] = []
    details: dict[str, Any] = {"dataset": preset.dataset}

    evaluator_path = Path(adm_evaluator).expanduser().resolve() if adm_evaluator is not None else None
    # Do not resolve interpreter symlinks: a venv commonly exposes ``bin/python``
    # as a symlink to the base interpreter, and replacing it with the resolved
    # target silently drops the venv's package environment.
    python_path = (
        Path(os.path.abspath(Path(adm_python).expanduser())) if adm_python is not None else None
    )
    details["adm_evaluator"] = str(evaluator_path) if evaluator_path is not None else None
    details["adm_python"] = str(python_path) if python_path is not None else None
    if evaluator_path is None or not evaluator_path.is_file():
        missing.append("--adm-evaluator must point to the pinned OpenAI evaluations/evaluator.py")
    if python_path is None or not python_path.is_file() or not os.access(python_path, os.X_OK):
        missing.append("--adm-python must point to an executable isolated ADM-environment Python")
    detector_path = (
        Path(adm_detector).expanduser().resolve()
        if adm_detector is not None
        else layout.reference_root / "classify_image_graph_def.pb"
    )
    details["adm_detector"] = str(detector_path)
    expected_detector = LSUN_BEDROOM256_ADM_PROTOCOL.detector
    if not detector_path.is_file():
        missing.append(f"OpenAI ADM detector is missing: {detector_path}")
    else:
        if detector_path.name != "classify_image_graph_def.pb":
            invalid.append(f"ADM detector must be named classify_image_graph_def.pb: {detector_path}")
        if expected_detector.size_bytes is not None and detector_path.stat().st_size != expected_detector.size_bytes:
            invalid.append(
                f"ADM detector size mismatch at {detector_path}: expected {expected_detector.size_bytes}, "
                f"got {detector_path.stat().st_size}"
            )
        if expected_detector.md5 is not None and file_digest(detector_path, "md5") != expected_detector.md5:
            invalid.append(f"ADM detector MD5 mismatch: {detector_path}")

    reference_specs: list[tuple[Path, Any]] = []
    if preset.dataset == "ffhq":
        edm_root = Path(nvlabs_edm_root).expanduser().resolve() if nvlabs_edm_root is not None else None
        details["nvlabs_edm_root"] = str(edm_root) if edm_root is not None else None
        if edm_root is None or not (edm_root / "fid.py").is_file():
            missing.append("--nvlabs-edm-root must point to a pinned NVLabs EDM checkout containing fid.py")
        reference_specs.append((layout.reference_root / "ffhq-64x64.npz", FFHQ64_NVIDIA_PROTOCOL))
        custom_reference = (
            Path(ffhq_adm_reference).expanduser().resolve()
            if ffhq_adm_reference is not None
            else layout.reference_root / "VIRTUAL_ffhq64_first50k_adm.npz"
        )
        reference_specs.append((custom_reference, FFHQ64_ADM_CUSTOM_PROTOCOL))
    else:
        reference_specs.append(
            (layout.reference_root / "VIRTUAL_lsun_bedroom256.npz", LSUN_BEDROOM256_ADM_PROTOCOL)
        )

    details["references"] = [str(path) for path, _protocol in reference_specs]
    for path, protocol in reference_specs:
        if not path.is_file():
            missing.append(f"reference file is missing: {path}")
            continue
        try:
            validate_reference_npz(path, protocol, verify_checksum=True)
        except Exception as exc:
            invalid.append(f"{path}: {exc}")

    if missing or invalid:
        details["missing"] = missing
        details["invalid"] = invalid
        return Check(
            "benchmark_dependencies",
            "fail",
            "; ".join(missing + [f"invalid reference {item}" for item in invalid]),
            details,
        )
    return Check(
        "benchmark_dependencies",
        "pass",
        "metric adapters and protocol-specific references passed",
        details,
    )


def heavy_cache_environment(shared_root: str | Path) -> dict[str, str]:
    root = Path(shared_root).expanduser().resolve()
    cache = root / "cache"
    return {
        "PACE_SHARED_ROOT": str(root),
        "HF_HOME": str(cache / "huggingface"),
        "TORCH_HOME": str(cache / "torch"),
        "DNNLIB_CACHE_DIR": str(cache / "dnnlib"),
        "XDG_CACHE_HOME": str(cache / "xdg"),
        "PIP_CACHE_DIR": str(cache / "pip"),
        "MPLCONFIGDIR": str(cache / "matplotlib"),
        "WANDB_DIR": str(root / "wandb"),
        "TMPDIR": str(root / "tmp"),
    }


@dataclass(frozen=True)
class PipelineAction:
    action_id: str
    stage: str
    command: tuple[str, ...]
    environment: Mapping[str, str]
    output_paths: tuple[str, ...] = ()
    resume_output_dir: str | None = None
    upstream_fingerprints: tuple[str, ...] = ()
    fingerprint_salt: str | None = None

    @property
    def fingerprint(self) -> str:
        return canonical_digest(
            {
                "command": self.command,
                "environment": self.environment,
                "upstream_fingerprints": self.upstream_fingerprints,
                "fingerprint_salt": self.fingerprint_salt,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "stage": self.stage,
            "command": list(self.command),
            "command_shell": shlex.join(self.command),
            "environment": dict(self.environment),
            "output_paths": list(self.output_paths),
            "resume_output_dir": self.resume_output_dir,
            "upstream_fingerprints": list(self.upstream_fingerprints),
            "fingerprint_salt": self.fingerprint_salt,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class PipelineLayout:
    shared_root: Path
    project_root: Path
    run_root: Path
    analysis_root: Path
    dataset_root: Path
    dataset_manifest: Path
    profile_root: Path
    plans_root: Path
    training_root: Path
    benchmark_root: Path
    model_cache: Path
    reference_root: Path
    profile_ablation_mode: str
    profile_protocol_id: str


def pipeline_layout(
    preset: PipelinePreset,
    shared_root: str | Path,
    *,
    profile_ablation_mode: str | None = None,
    project_root: str | Path | None = None,
) -> PipelineLayout:
    shared = Path(shared_root).expanduser().resolve()
    project = Path(project_root or DEFAULT_PROJECT_ROOT).expanduser().resolve()
    base_run = shared / "benchmarks" / preset.name
    resolved_ablation_mode = profile_ablation_mode or preset.profile_ablation_mode
    if resolved_ablation_mode not in SUPPORTED_PROFILE_ABLATION_MODES:
        raise PreflightError(
            f"profile ablation mode must be one of {SUPPORTED_PROFILE_ABLATION_MODES}, "
            f"got {resolved_ablation_mode!r}"
        )
    profile_protocol_id = (
        PFI_PROFILE_PROTOCOL_ID if resolved_ablation_mode == "pfi" else "legacy_unspecified"
    )
    # Versioned PFI outputs get isolated repo-local analysis and shared-storage
    # operational trees. Explicit historical modes retain their exact shared
    # root-level layout so existing resume commands and artifacts remain valid.
    run = base_run / PFI_OUTPUT_NAMESPACE if resolved_ablation_mode == "pfi" else base_run
    analysis_root = (
        project / f"out_eval_edm_{preset.name}" / PFI_OUTPUT_NAMESPACE
        if resolved_ablation_mode == "pfi"
        else run
    )
    dataset_root = shared / "datasets" / (
        "ffhq256_to64_lanczos_v1" if preset.dataset == "ffhq" else "lsun_bedroom256_source"
    )
    return PipelineLayout(
        shared_root=shared,
        project_root=project,
        run_root=run,
        analysis_root=analysis_root,
        dataset_root=dataset_root,
        dataset_manifest=base_run / "dataset" / "manifest.json",
        profile_root=analysis_root if resolved_ablation_mode == "pfi" else run / "profile",
        plans_root=analysis_root / "plans" if resolved_ablation_mode == "pfi" else run / "plans",
        training_root=run / "training",
        benchmark_root=run / "benchmark",
        model_cache=shared / "cache" / "teachers",
        reference_root=shared / "references",
        profile_ablation_mode=resolved_ablation_mode,
        profile_protocol_id=profile_protocol_id,
    )


def _python_prefix(gpu_ids: Sequence[int]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        str(len(gpu_ids)),
    ]


def _action(
    action_id: str,
    stage: str,
    command: Sequence[str | Path],
    environment: Mapping[str, str],
    *outputs: str | Path,
    resume_output_dir: str | Path | None = None,
    upstream_fingerprints: Sequence[str] = (),
    fingerprint_salt: str | None = None,
) -> PipelineAction:
    return PipelineAction(
        action_id=action_id,
        stage=stage,
        command=tuple(str(item) for item in command),
        environment=dict(environment),
        output_paths=tuple(str(Path(item)) for item in outputs),
        resume_output_dir=None if resume_output_dir is None else str(Path(resume_output_dir)),
        upstream_fingerprints=tuple(str(value) for value in upstream_fingerprints),
        fingerprint_salt=fingerprint_salt,
    )


def build_pipeline_actions(
    preset: PipelinePreset,
    *,
    source_root: str | Path,
    shared_root: str | Path,
    gpu_ids: Sequence[int],
    variants: Sequence[str] | None = None,
    ffhq_adm_reference: str | Path | None = None,
    nvlabs_edm_root: str | Path | None = None,
    adm_evaluator: str | Path | None = None,
    adm_python: str | Path | None = None,
    adm_detector: str | Path | None = None,
    artifact_retention: str = "keep_preview",
    preview_count: int = 64,
    profile_ablation_mode: str | None = None,
    project_root: str | Path | None = None,
    num_timestep_blocks: int | None = None,
    grouping_builtin_cost: str | None = None,
    allocation_metric: str | None = None,
    score_reduction: str | None = None,
    shuffle_seed: int | None = None,
) -> tuple[PipelineLayout, list[PipelineAction]]:
    resolved_variants = tuple(preset.default_variants if variants is None else variants)
    resolved_num_timestep_blocks = (
        preset.num_timestep_blocks if num_timestep_blocks is None else num_timestep_blocks
    )
    resolved_grouping_builtin_cost = (
        preset.grouping_builtin_cost if grouping_builtin_cost is None else grouping_builtin_cost
    )
    resolved_allocation_metric = (
        preset.allocation_metric if allocation_metric is None else allocation_metric
    )
    resolved_score_reduction = (
        preset.score_reduction if score_reduction is None else score_reduction
    )
    resolved_shuffle_seed = preset.shuffle_seed if shuffle_seed is None else shuffle_seed
    if not resolved_variants:
        raise PreflightError("at least one student variant is required")
    if len(set(resolved_variants)) != len(resolved_variants):
        raise PreflightError(f"student variants must be unique, got {resolved_variants}")
    unsupported_variants = [
        variant for variant in resolved_variants if variant not in SUPPORTED_PIPELINE_VARIANTS
    ]
    if unsupported_variants:
        raise PreflightError(f"unsupported student variants: {unsupported_variants}")
    if not 1 <= resolved_num_timestep_blocks <= 20:
        raise PreflightError(
            "num_timestep_blocks must be between 1 and the 20 profiled sigma bins, "
            f"got {resolved_num_timestep_blocks}"
        )
    if resolved_grouping_builtin_cost not in SUPPORTED_GROUPING_BUILTIN_COSTS:
        raise PreflightError(
            f"grouping_builtin_cost must be one of {SUPPORTED_GROUPING_BUILTIN_COSTS}, "
            f"got {resolved_grouping_builtin_cost!r}"
        )
    if resolved_score_reduction not in SUPPORTED_ALLOCATION_SCORE_REDUCTIONS:
        raise PreflightError(
            "score_reduction must be one of "
            f"{SUPPORTED_ALLOCATION_SCORE_REDUCTIONS}, got {resolved_score_reduction!r}"
        )
    if (
        resolved_allocation_metric == "delta_p_eff_geomean"
        and resolved_score_reduction != "sum"
    ):
        raise PreflightError(
            "allocation_metric='delta_p_eff_geomean' requires score_reduction='sum'"
        )
    combined_variants = sorted(
        set(resolved_variants).intersection({"combined_blockwise", "combined_layerwise"})
    )
    if combined_variants and (
        resolved_allocation_metric != "delta_p_eff_geomean"
        or resolved_score_reduction != "sum"
    ):
        raise PreflightError(
            f"combined variants {combined_variants} require "
            "allocation_metric='delta_p_eff_geomean' and score_reduction='sum'"
        )
    if resolved_shuffle_seed < 0:
        raise PreflightError(f"shuffle_seed must be non-negative, got {resolved_shuffle_seed}")

    layout = pipeline_layout(
        preset,
        shared_root,
        profile_ablation_mode=profile_ablation_mode,
        project_root=project_root,
    )
    environment = heavy_cache_environment(layout.shared_root) | {
        "CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in gpu_ids),
        "PYTHONUNBUFFERED": "1",
    }
    source = Path(source_root).expanduser().resolve()
    resolved_adm_detector = (
        Path(adm_detector).expanduser().resolve()
        if adm_detector is not None
        else layout.reference_root / "classify_image_graph_def.pb"
    )
    actions: list[PipelineAction] = []
    if preset.dataset == "ffhq":
        actions.append(
            _action(
                "prepare:ffhq64",
                "prepare",
                [
                    sys.executable,
                    "scripts/data/prepare_ffhq_dataset.py",
                    "--source",
                    source,
                    "--output-dir",
                    layout.dataset_root,
                    "--resolution",
                    "64",
                    "--workers",
                    "16",
                ],
                environment,
                layout.dataset_root / "dataset_manifest.json",
            )
        )
    else:
        # LSUN stays at its source location; the dataset loader performs the
        # deterministic 256 center crop.  Never copy a million JPEGs by default.
        layout = dataclasses.replace(layout, dataset_root=source)
    actions.append(
        _action(
            f"prepare:{preset.dataset}_manifest",
            "prepare",
            [
                sys.executable,
                "scripts/data/preflight_edm_dataset.py",
                "--dataset",
                preset.dataset,
                "--data-root",
                layout.dataset_root,
                "--dataset-split",
                "train",
                "--image-size",
                str(preset.resolution),
                "--dataset-manifest",
                layout.dataset_manifest,
            ],
            environment,
            layout.dataset_manifest,
        )
    )

    profile_command = _python_prefix(gpu_ids) + [
        "scripts/evaluate_parameters_edm.py",
        "--distributed-timeout-seconds",
        "86400",
        "--dataset",
        preset.dataset,
        "--data_root",
        str(layout.dataset_root),
        "--dataset-manifest",
        str(layout.dataset_manifest),
        "--dataset-split",
        "monitor",
        "--network-preset",
        preset.teacher_preset,
        "--model-cache-dir",
        str(layout.model_cache),
        "--output_dir",
        str(layout.profile_root),
        "--grouping",
        "per_filter",
        "--confirm-full-profile",
        "--ablation_mode",
        layout.profile_ablation_mode,
        "--batch_size",
        str(preset.profile_batch_size),
        "--num_workers",
        "16",
        "--max_images",
        str(preset.profile_images),
        "--num_bins",
        "20",
        "--num_sigma_levels",
        "256",
        "--image_size",
        str(preset.resolution),
        "--dtype",
        preset.train_dtype,
    ]
    if preset.dataset == "ffhq":
        profile_command += ["--ffhq-protocol", "ffhq256_numeric_v1"]
    else:
        profile_command += ["--lsun-monitor-size", "10000", "--lsun-monitor-seed", "12345"]
    if layout.profile_ablation_mode == "pfi":
        profile_command += ["--pfi_seed", "0"]
    profile_action = _action(
        "profile:teacher",
        "profile",
        profile_command,
        environment,
        layout.profile_root / "results.json",
    )
    actions.append(profile_action)

    grouping = layout.profile_root / "grouping" / "timestep_grouping.json"
    grouping_cost_curve = layout.profile_root / "grouping" / "timestep_grouping_cost_curve.png"
    grouping_action = _action(
        "group:timesteps",
        "group",
        [
                sys.executable,
                "scripts/optimize_timestep_grouping.py",
                "--matrix",
                layout.profile_root / "results.json",
                "--matrix_key",
                "relative_delta_stack",
                "--builtin_cost",
                resolved_grouping_builtin_cost,
                "--num_blocks",
                str(resolved_num_timestep_blocks),
                "--output",
                grouping,
                "--plot_output",
                layout.profile_root / "grouping" / "timestep_grouping.png",
                "--plot_matrix_source",
                "recomputed_similarity",
                "--cost_curve_output",
                grouping_cost_curve,
        ],
        environment,
        grouping,
        grouping_cost_curve,
        upstream_fingerprints=(profile_action.fingerprint,),
    )
    actions.append(grouping_action)
    allocation_command: list[str | Path] = [
        sys.executable,
        "scripts/dry_run_capacity_allocation.py",
        "--eval-output-dir",
        layout.profile_root,
        "--timestep-grouping",
        grouping,
        "--allocation-metric",
        resolved_allocation_metric,
        "--score-reduction",
        resolved_score_reduction,
        "--shuffle-seed",
        str(resolved_shuffle_seed),
    ]
    for variant in resolved_variants:
        allocation_command += ["--student-variant", variant]
    allocation_action = _action(
        "allocate:students",
        "allocate",
        allocation_command,
        environment,
        layout.profile_root / "allocation_results" / "summary.json",
        upstream_fingerprints=(grouping_action.fingerprint,),
    )
    actions.append(allocation_action)
    plan_command: list[str | Path] = [
        sys.executable,
        "scripts/prepare_edm_distillation.py",
        "--eval-output-dir",
        layout.profile_root,
        "--output-dir",
        layout.plans_root,
        "--shuffle-seed",
        str(resolved_shuffle_seed),
    ]
    for variant in resolved_variants:
        plan_command += ["--variant", variant]
    plans_action = _action(
        "plans:students",
        "plans",
        plan_command,
        environment,
        layout.plans_root / "architecture_summary.json",
        upstream_fingerprints=(allocation_action.fingerprint,),
    )
    actions.append(plans_action)

    train_actions: dict[str, PipelineAction] = {}
    for variant in resolved_variants:
        train_output = layout.training_root / variant / "seed0"
        train_command = _python_prefix(gpu_ids) + [
            "scripts/train_edm_distillation.py",
            "--architecture-plan",
            str(layout.plans_root / variant / "architecture_plan.json"),
            "--output-dir",
            str(train_output),
            "--dataset",
            preset.dataset,
            "--data-root",
            str(layout.dataset_root),
            "--dataset-manifest",
            str(layout.dataset_manifest),
            "--dataset-split",
            "train",
            "--model-cache-dir",
            str(layout.model_cache),
            "--dtype",
            preset.train_dtype,
            "--steps",
            str(preset.train_steps),
            "--snapshot-every",
            "5000",
            "--keep-last-snapshots",
            "0",
            "--checkpoint-selection",
            "best_val",
            "--batch-size",
            str(preset.train_batch_size),
            "--microbatch",
            str(preset.train_microbatch),
            "--num-workers",
            "16",
            "--val-every",
            "5000",
            "--val-max-images",
            "10000",
            "--val-batch-size",
            "64" if preset.resolution == 256 else "256",
            "--val-microbatch",
            "1" if preset.resolution == 256 else "16",
            "--val-seed",
            "12345",
            "--early-stop-patience",
            "4",
            "--early-stop-min-steps",
            "10000",
            "--fid-every",
            "0",
        ]
        if preset.dataset == "ffhq":
            train_command += ["--ffhq-protocol", "ffhq256_numeric_v1"]
        else:
            train_command += ["--lsun-monitor-size", "10000", "--lsun-monitor-seed", "12345"]
        train_action = _action(
            f"train:{variant}",
            "train",
            train_command,
            environment,
            train_output / "student-best-val.pt",
            resume_output_dir=train_output,
            upstream_fingerprints=(plans_action.fingerprint,),
        )
        actions.append(train_action)
        train_actions[variant] = train_action
    # Teacher and students share the exact benchmark orchestration.  FFHQ emits
    # OpenAI-truncated samples alongside the first canonical NVLabs run so the
    # custom ADM suite does not require another expensive model pass.
    benchmark_models: list[tuple[str, list[str], Path | None]] = [
        ("teacher", ["--network-preset", preset.teacher_preset], None),
    ]
    benchmark_models += [
        (
            variant,
            ["--checkpoint", str(layout.training_root / variant / "seed0" / "student-best-val.pt")],
            layout.training_root / variant / "seed0" / "student-best-val.pt",
        )
        for variant in resolved_variants
    ]
    for model_name, model_args, _checkpoint in benchmark_models:
        model_upstream_fingerprints = (
            (train_actions[model_name].fingerprint,)
            if model_name in train_actions
            else ()
        )
        model_root = layout.benchmark_root / model_name
        common = [
            "--model-cache-dir",
            str(layout.model_cache),
            "--dataset-manifest",
            str(layout.dataset_manifest),
            "--artifact-retention",
            "keep" if preset.dataset == "ffhq" else artifact_retention,
            "--preview-count",
            str(preview_count),
        ]
        model_precision = [
            "--teacher-dtype",
            "fp16" if preset.dataset == "lsun_bedroom" and model_name == "teacher" else "fp32",
        ]
        if preset.dataset == "ffhq":
            nvidia_results: list[Path] = []
            nvidia_actions: list[PipelineAction] = []
            shared_openai_samples = model_root / "shared" / "seed0_openai_truncate"
            for run_index, (start_seed, _end_seed) in enumerate(FFHQ64_NVIDIA_PROTOCOL.sampling.seed_ranges):
                run_output = model_root / FFHQ64_NVIDIA_PROTOCOL.identity / f"run{run_index}"
                nvidia_results.append(run_output / "evaluation_result.json")
                command = _python_prefix(gpu_ids) + [
                    "scripts/evaluate_edm_checkpoint.py",
                    *model_args,
                    "--output-dir",
                    str(run_output),
                    "--reference-stats",
                    str(layout.reference_root / "ffhq-64x64.npz"),
                    "--num-samples",
                    "50000",
                    "--seed",
                    str(start_seed),
                    "--num-steps",
                    "40",
                    "--batch-size-per-rank",
                    "32",
                    "--metric-backend",
                    "nvlabs_edm_fid",
                    "--protocol-id",
                    FFHQ64_NVIDIA_PROTOCOL.identity,
                    *model_precision,
                    *common,
                ]
                if nvlabs_edm_root is not None:
                    command += ["--nvlabs-edm-root", str(Path(nvlabs_edm_root).expanduser().resolve())]
                if run_index == 0:
                    command += [
                        "--secondary-samples-dir",
                        str(shared_openai_samples),
                        "--secondary-quantizer",
                        "openai_x_plus_1_x127_5",
                    ]
                benchmark_action = _action(
                    f"benchmark:{model_name}:{FFHQ64_NVIDIA_PROTOCOL.identity}:run{run_index}",
                    "benchmark",
                    command,
                    environment,
                    run_output / "evaluation_result.json",
                    run_output / "benchmark_manifest.json",
                    upstream_fingerprints=model_upstream_fingerprints,
                )
                actions.append(benchmark_action)
                nvidia_actions.append(benchmark_action)
            summary_path = model_root / FFHQ64_NVIDIA_PROTOCOL.identity / "evaluation_summary.json"
            summary_command: list[str | Path] = [
                sys.executable,
                "scripts/paper/summarize_edm_benchmark.py",
                "--protocol-id",
                FFHQ64_NVIDIA_PROTOCOL.identity,
                "--metric",
                "fid_nvlabs_legacy",
                "--aggregation",
                "minimum",
                "--output",
                summary_path,
            ]
            for result in nvidia_results:
                summary_command += ["--result", result]
            aggregate_action = _action(
                f"benchmark:{model_name}:{FFHQ64_NVIDIA_PROTOCOL.identity}:aggregate",
                "benchmark",
                summary_command,
                environment,
                summary_path,
                upstream_fingerprints=tuple(action.fingerprint for action in nvidia_actions),
            )
            actions.append(aggregate_action)
            adm_output = model_root / FFHQ64_ADM_CUSTOM_PROTOCOL.identity
            adm_reference = (
                Path(ffhq_adm_reference).expanduser().resolve()
                if ffhq_adm_reference is not None
                else layout.reference_root / "VIRTUAL_ffhq64_first50k_adm.npz"
            )
            adm_command = _python_prefix(gpu_ids) + [
                "scripts/evaluate_edm_checkpoint.py",
                *model_args,
                "--reuse-samples",
                "--reuse-provenance",
                str(nvidia_results[0]),
                "--samples-dir",
                str(shared_openai_samples),
                "--output-dir",
                str(adm_output),
                "--reference-stats",
                str(adm_reference),
                "--num-samples",
                "50000",
                "--seed",
                "0",
                "--num-steps",
                "40",
                "--batch-size-per-rank",
                "32",
                "--metric-backend",
                "openai_adm",
                "--protocol-id",
                FFHQ64_ADM_CUSTOM_PROTOCOL.identity,
                *model_precision,
                *common,
            ]
            if adm_evaluator is not None:
                adm_command += ["--adm-evaluator", str(Path(adm_evaluator).expanduser().resolve())]
            if adm_python is not None:
                adm_command += [
                    "--adm-python",
                    str(Path(os.path.abspath(Path(adm_python).expanduser()))),
                ]
            adm_command += ["--adm-detector", str(resolved_adm_detector)]
            adm_action = _action(
                f"benchmark:{model_name}:{FFHQ64_ADM_CUSTOM_PROTOCOL.identity}",
                "benchmark",
                adm_command,
                environment,
                adm_output / "evaluation_result.json",
                adm_output / "benchmark_manifest.json",
                upstream_fingerprints=(nvidia_actions[0].fingerprint,),
            )
            actions.append(adm_action)
            cleanup_output = model_root / "artifact_cleanup.json"
            cleanup_command: list[str | Path] = [
                sys.executable,
                "scripts/paper/finalize_edm_benchmark_artifacts.py",
                "--summary",
                summary_path,
                "--retention",
                artifact_retention,
                "--preview-dir",
                model_root / "preview",
                "--preview-count",
                str(preview_count),
                "--metric-artifact",
                adm_output / "adm_samples.npz",
                "--output",
                cleanup_output,
            ]
            for result_path in [*nvidia_results, adm_output / "evaluation_result.json"]:
                cleanup_command += ["--result", result_path]
            for run_index in range(len(FFHQ64_NVIDIA_PROTOCOL.sampling.seed_ranges)):
                cleanup_command += [
                    "--sample-dir",
                    model_root / FFHQ64_NVIDIA_PROTOCOL.identity / f"run{run_index}" / "samples",
                ]
            cleanup_command += ["--sample-dir", shared_openai_samples]
            actions.append(
                _action(
                    f"benchmark:{model_name}:finalize_artifacts",
                    "benchmark",
                    cleanup_command,
                    environment,
                    cleanup_output,
                    upstream_fingerprints=(
                        aggregate_action.fingerprint,
                        adm_action.fingerprint,
                        *(action.fingerprint for action in nvidia_actions),
                    ),
                )
            )
        else:
            protocol = LSUN_BEDROOM256_ADM_PROTOCOL
            run_output = model_root / protocol.identity
            command = _python_prefix(gpu_ids) + [
                "scripts/evaluate_edm_checkpoint.py",
                *model_args,
                "--output-dir",
                str(run_output),
                "--reference-stats",
                str(layout.reference_root / "VIRTUAL_lsun_bedroom256.npz"),
                "--num-samples",
                "50000",
                "--seed",
                "2100000",
                "--num-steps",
                "40",
                "--batch-size-per-rank",
                "8",
                "--metric-backend",
                "openai_adm",
                "--protocol-id",
                protocol.identity,
                *model_precision,
                *common,
            ]
            if adm_evaluator is not None:
                command += ["--adm-evaluator", str(Path(adm_evaluator).expanduser().resolve())]
            if adm_python is not None:
                command += [
                    "--adm-python",
                    str(Path(os.path.abspath(Path(adm_python).expanduser()))),
                ]
            command += ["--adm-detector", str(resolved_adm_detector)]
            actions.append(
                _action(
                    f"benchmark:{model_name}:{protocol.identity}",
                    "benchmark",
                    command,
                    environment,
                    run_output / "evaluation_result.json",
                    run_output / "benchmark_manifest.json",
                    upstream_fingerprints=model_upstream_fingerprints,
                )
            )
    return layout, actions


def run_preflight(
    preset: PipelinePreset,
    *,
    source_root: str | Path,
    shared_root: str | Path,
    gpu_ids: Sequence[int],
    min_free_memory_mib: int | None = None,
    min_free_storage_gib: int | None = None,
    scan_lsun: bool = True,
    benchmark_requested: bool = False,
    ffhq_adm_reference: str | Path | None = None,
    nvlabs_edm_root: str | Path | None = None,
    adm_evaluator: str | Path | None = None,
    adm_python: str | Path | None = None,
    adm_detector: str | Path | None = None,
    gpu_requested: bool = True,
    required_gpu_count: int | None = None,
) -> dict[str, Any]:
    checks = [
        check_storage(
            shared_root,
            min_free_storage_gib=(
                preset.min_free_storage_gib if min_free_storage_gib is None else min_free_storage_gib
            ),
        ),
        check_dataset_source(preset, source_root, scan_lsun=scan_lsun),
    ]
    if gpu_requested:
        checks.insert(
            1,
            check_gpus(
                gpu_ids,
                required_count=(
                    preset.required_gpu_count
                    if required_gpu_count is None
                    else required_gpu_count
                ),
                min_free_memory_mib=(
                    preset.min_free_memory_mib if min_free_memory_mib is None else min_free_memory_mib
                ),
            ),
        )
    if benchmark_requested:
        checks.append(
            check_benchmark_dependencies(
                preset,
                shared_root=shared_root,
                ffhq_adm_reference=ffhq_adm_reference,
                nvlabs_edm_root=nvlabs_edm_root,
                adm_evaluator=adm_evaluator,
                adm_python=adm_python,
                adm_detector=adm_detector,
            )
        )
    return {
        "preset": preset.name,
        "passed": all(check.passed for check in checks),
        "checks": [dataclasses.asdict(check) for check in checks],
    }


def _load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"state_format": PIPELINE_STATE_FORMAT, "actions": {}}
    try:
        state = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"could not read pipeline state {path}: {exc}") from exc
    if state.get("state_format") != PIPELINE_STATE_FORMAT or not isinstance(state.get("actions"), dict):
        raise PreflightError(f"unsupported pipeline state file: {path}")
    return state


def execute_pipeline_actions(
    actions: Iterable[PipelineAction],
    *,
    state_path: str | Path,
    repo_root: str | Path,
    selected_stages: set[str] | None = None,
    force_actions: set[str] | None = None,
) -> dict[str, Any]:
    """Execute sequentially, recording success only after each process exits 0."""

    state_file = Path(state_path)
    state = _load_state(state_file)
    force = force_actions or set()
    executed: list[str] = []
    skipped: list[str] = []
    for action in actions:
        if selected_stages is not None and action.stage not in selected_stages:
            continue
        previous = state["actions"].get(action.action_id, {})
        if (
            action.action_id not in force
            and previous.get("status") == "complete"
            and previous.get("fingerprint") == action.fingerprint
            and all(Path(path).exists() for path in action.output_paths)
        ):
            skipped.append(action.action_id)
            continue
        command = list(action.command)
        if action.resume_output_dir is not None:
            output = Path(action.resume_output_dir)
            if (output / "latest-checkpoint.json").is_file() and "--resume" not in command:
                previous_fingerprint = previous.get("fingerprint")
                if previous_fingerprint != action.fingerprint:
                    observed = previous_fingerprint or "<missing>"
                    raise PreflightError(
                        "refusing automatic resume for "
                        f"{action.action_id}: checkpoint state fingerprint {observed} "
                        f"does not match current action fingerprint {action.fingerprint}; "
                        "the existing checkpoint was left untouched"
                    )
                command += ["--resume", "auto"]
        environment = os.environ.copy()
        environment.update(action.environment)
        for name in (
            "PACE_SHARED_ROOT",
            "HF_HOME",
            "TORCH_HOME",
            "DNNLIB_CACHE_DIR",
            "XDG_CACHE_HOME",
            "PIP_CACHE_DIR",
            "MPLCONFIGDIR",
            "WANDB_DIR",
            "TMPDIR",
        ):
            cache_path = action.environment.get(name)
            if cache_path:
                Path(cache_path).mkdir(parents=True, exist_ok=True)
        started = time.time()
        state["actions"][action.action_id] = {
            "status": "running",
            "fingerprint": action.fingerprint,
            "started_unix": started,
            "command": command,
        }
        atomic_write_json(state_file, state)
        try:
            subprocess.run(command, cwd=repo_root, env=environment, check=True)
        except subprocess.CalledProcessError as exc:
            state["actions"][action.action_id].update(
                {"status": "failed", "ended_unix": time.time(), "returncode": exc.returncode}
            )
            atomic_write_json(state_file, state)
            raise
        state["actions"][action.action_id].update(
            {"status": "complete", "ended_unix": time.time(), "returncode": 0}
        )
        atomic_write_json(state_file, state)
        executed.append(action.action_id)
    return {"state_path": str(state_file.resolve()), "executed": executed, "skipped": skipped}


def pipeline_plan_payload(
    preset: PipelinePreset,
    layout: PipelineLayout,
    actions: Sequence[PipelineAction],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "dry_run": True,
        "preset": dataclasses.asdict(preset),
        "layout": {key: str(value) for key, value in dataclasses.asdict(layout).items()},
        "preflight": preflight,
        "estimated_metric_sample_bytes_per_variant": {
            FFHQ64_NVIDIA_PROTOCOL.identity: estimated_sample_bytes(FFHQ64_NVIDIA_PROTOCOL),
            FFHQ64_ADM_CUSTOM_PROTOCOL.identity: estimated_sample_bytes(FFHQ64_ADM_CUSTOM_PROTOCOL),
            LSUN_BEDROOM256_ADM_PROTOCOL.identity: estimated_sample_bytes(LSUN_BEDROOM256_ADM_PROTOCOL),
        },
        "actions": [action.to_dict() for action in actions],
    }
