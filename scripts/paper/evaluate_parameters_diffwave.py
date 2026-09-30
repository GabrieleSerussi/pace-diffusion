#!/usr/bin/env python3
"""Run the repository's parameter-usage analysis on the SC09 DiffWave teacher."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shlex
import subprocess
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from pace.audio_datasets import SC09Dataset
from pace.parameter_analysis import (
    AnalysisAxis,
    BinnedEvaluationBatch,
    BinStats,
    ExactLevelPFIBatchSampler,
    FingerprintedBinnedCache,
    PFILevelSampleIndex,
    atomic_torch_save,
    assert_distributed_canonical_invariant,
    assert_distributed_hash_invariant,
    build_ablation_protocol,
    canonical_json_sha256,
    choose_dtype,
    compute_signed_and_positive_deltas,
    compute_usage_metrics,
    cleanup_distributed_analysis,
    count_parameters,
    dataset_population_fingerprint,
    evaluate_binned_losses,
    infer_bin_axis_metadata,
    init_distributed_analysis,
    level_to_bin,
    mse_per_example,
    persist_exact_level_pfi_plan,
    plot_binned_series,
    plot_labeled_heatmap,
    run_rank_zero_analysis_operation,
    run_binned_ablation_profile,
    save_results_and_plots,
    set_seed,
    should_compute_group_correlation,
    tensor_or_none_to_list,
)
from pace.teacher_models import (
    TeacherSpec,
    load_teacher_network,
    resolve_teacher_spec,
    teacher_model_metadata,
    teacher_preset_config,
)
from pace.vendor.diffwave_legacy import (
    DIFFWAVE_SASHIMI_CHECKPOINTS_COMMIT,
    DIFFWAVE_SASHIMI_CHECKPOINT_SHA256,
    LegacyCompatibleDiffWave,
    diffwave_diffusion_hyperparameters,
)


ANALYSIS_FORMAT = "diffdist_diffwave_parameter_analysis_v1"
VERIFICATION_FORMAT = "diffdist_diffwave_teacher_reproduction_v1"
BASELINE_CACHE_FORMAT = "diffdist_diffwave_pfi_baseline_v1"
ABLATION_CACHE_FORMAT = "diffdist_diffwave_pfi_checkpoint_v1"
PER_FILTER_EXPECTED_CONV_MODULES = 111
PER_FILTER_EXPECTED_GROUPS = 37_377
PER_FILTER_EXPECTED_TRUE_PARAMETERS = 19_015_169
PER_FILTER_EXPECTED_EDM_PROXY_PARAMETERS = 18_977_793
PER_FILTER_EXPECTED_UNASSIGNED_PARAMETERS = 5_056_512
PER_FILTER_EXPECTED_RESIDUAL_PARAMETERS = 18_948_096


def format_invocation_command(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(argument)) for argument in argv)


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def atomic_save_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(dict(payload), indent=2) + "\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_teacher_verification_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path).expanduser().resolve()
    report = load_json(report_path)
    if report.get("format") != VERIFICATION_FORMAT:
        raise ValueError(
            f"Teacher verification report must use {VERIFICATION_FORMAT!r}, "
            f"got {report.get('format')!r}"
        )
    if report.get("passed") is not True:
        raise ValueError(f"Teacher verification did not pass: {report_path}")
    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Teacher verification report is missing checkpoint provenance")
    observed_hashes = {
        str(value).lower()
        for key, value in checkpoint.items()
        if "sha256" in str(key).lower() and isinstance(value, str)
    }
    if DIFFWAVE_SASHIMI_CHECKPOINT_SHA256 not in observed_hashes:
        raise ValueError(
            "Teacher verification report does not identify the required 1M checkpoint SHA-256"
        )
    upstream = report.get("upstream")
    if not isinstance(upstream, Mapping) or DIFFWAVE_SASHIMI_CHECKPOINTS_COMMIT not in {
        str(value) for value in upstream.values()
    }:
        raise ValueError("Teacher verification report does not identify the pinned upstream commit")
    checks = report.get("checks")
    required_checks = (
        "prediction_and_trajectory_parity",
        "autograd_safe",
        "valid_audio",
        "untouched_reference_generation",
        "generated_audio",
        "reference_environment_captured",
    )
    if not isinstance(checks, Mapping) or any(
        checks.get(key) is not True for key in required_checks
    ):
        raise ValueError("Teacher verification report is missing a required passing check")

    implementation = report.get("implementation")
    vendor_source = REPO_ROOT / "pace" / "vendor" / "diffwave_legacy.py"
    if (
        not isinstance(implementation, Mapping)
        or implementation.get("legacy_forward_semantics")
        != "timestep_projection_in_residual_identity_v1"
        or implementation.get("checkpoint_format")
        != "diffwave_sashimi_legacy_state_dict_v1"
        or implementation.get("strict_state_dict_load") is not True
        or implementation.get("checkpoint_container_key") != "model_state_dict"
        or implementation.get("source_sha256") != _file_sha256(vendor_source)
        or implementation.get("teacher_loader_sha256")
        != _file_sha256(REPO_ROOT / "pace" / "teacher_models.py")
    ):
        raise ValueError(
            "Teacher verification is not bound to the current safe legacy implementation"
        )

    configuration = report.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("Teacher verification report is missing its configuration")
    exact_configuration = {
        "batch_size": 1,
        "prediction_length": 16_000,
        "prediction_timesteps": [0, 1, 25, 50, 100, 150, 198, 199],
        "trajectory_steps": 10,
        "trajectory_length": 16_000,
    }
    if any(configuration.get(key) != value for key, value in exact_configuration.items()):
        raise ValueError(
            "Teacher verification did not use the required full-length prediction/trajectory probes"
        )
    for tolerance_key in ("atol", "rtol", "relative_l2_tolerance"):
        tolerance = configuration.get(tolerance_key)
        if (
            not isinstance(tolerance, (int, float))
            or not math.isfinite(float(tolerance))
            or float(tolerance) < 0
            or float(tolerance) > 1e-6
        ):
            raise ValueError(
                f"Teacher verification {tolerance_key} must be finite and at most 1e-6"
            )

    def require_close_metrics(value: Any, label: str) -> None:
        if not isinstance(value, Mapping):
            raise ValueError(f"Teacher verification is missing metrics for {label}")
        for key in ("max_abs", "mean_abs", "rmse", "relative_l2"):
            number = value.get(key)
            if not isinstance(number, (int, float)) or not math.isfinite(float(number)):
                raise ValueError(f"Teacher verification has invalid {label}.{key}")
        if value.get("elementwise_close") is not True or float(value["relative_l2"]) > 1e-6:
            raise ValueError(f"Teacher verification parity failed for {label}")

    predictions = report.get("prediction_parity")
    required_timesteps = exact_configuration["prediction_timesteps"]
    if not isinstance(predictions, list) or [item.get("timestep") for item in predictions] != required_timesteps:
        raise ValueError("Teacher verification prediction timesteps are incomplete or reordered")
    for item in predictions:
        require_close_metrics(item, f"prediction_t{item['timestep']}")

    trajectory = report.get("trajectory_parity")
    required_trajectory = list(range(199, 189, -1))
    if (
        not isinstance(trajectory, Mapping)
        or trajectory.get("timesteps") != required_trajectory
        or not isinstance(trajectory.get("per_step"), list)
        or len(trajectory["per_step"]) != 10
    ):
        raise ValueError("Teacher verification trajectory must cover exactly t=199 through t=190")
    for expected_timestep, step in zip(required_trajectory, trajectory["per_step"], strict=True):
        if not isinstance(step, Mapping) or step.get("timestep") != expected_timestep:
            raise ValueError("Teacher verification trajectory steps are incomplete or reordered")
        require_close_metrics(step.get("epsilon"), f"trajectory_t{expected_timestep}.epsilon")
        require_close_metrics(step.get("state"), f"trajectory_t{expected_timestep}.state")

    autograd = report.get("autograd")
    if (
        not isinstance(autograd, Mapping)
        or autograd.get("input_gradient_finite") is not True
        or autograd.get("input_unchanged_by_residual_blocks") is not True
        or not isinstance(autograd.get("input_gradient_max_abs"), (int, float))
        or not math.isfinite(float(autograd["input_gradient_max_abs"]))
        or float(autograd["input_gradient_max_abs"]) <= 0
    ):
        raise ValueError("Teacher verification does not prove a finite, nonzero safe backward pass")

    generated_audio = report.get("generated_audio")
    reference_audio = report.get("untouched_reference_audio")
    if (
        not isinstance(generated_audio, Mapping)
        or generated_audio.get("finite") is not True
        or generated_audio.get("sample_rate") != 16_000
        or generated_audio.get("num_samples") != 16_000
        or float(generated_audio.get("rms", 0.0)) <= 1e-6
        or not isinstance(reference_audio, Mapping)
        or reference_audio.get("finite") is not True
        or reference_audio.get("dtype") != "float32"
        or reference_audio.get("num_channels") != 1
        or reference_audio.get("sample_rate") != 16_000
        or reference_audio.get("num_samples") != 16_000
        or float(reference_audio.get("rms", 0.0)) <= 1e-6
        or float(reference_audio.get("standard_deviation", 0.0)) <= 1e-7
    ):
        raise ValueError("Teacher verification is missing valid generated/reference audio")

    reference_generation = report.get("untouched_reference_generation")
    if not isinstance(reference_generation, Mapping):
        raise ValueError(
            "Teacher verification is missing untouched reference generation provenance"
        )
    reference_environment = reference_generation.get("environment")
    required_environment_strings = (
        "interpreter",
        "resolved_interpreter",
        "python",
        "platform",
        "torch",
        "torchaudio",
    )
    if (
        not isinstance(reference_environment, Mapping)
        or any(
            not isinstance(reference_environment.get(key), str)
            or not reference_environment[key]
            for key in required_environment_strings
        )
        or not isinstance(reference_environment.get("cuda_available"), bool)
        or (
            reference_environment.get("cuda_runtime") is not None
            and not isinstance(reference_environment.get("cuda_runtime"), str)
        )
        or (
            reference_environment.get("cudnn") is not None
            and not isinstance(reference_environment.get("cudnn"), int)
        )
    ):
        raise ValueError("Teacher verification has an invalid reference environment")
    pip_freeze = reference_environment.get("pip_freeze")
    if (
        not isinstance(pip_freeze, list)
        or not pip_freeze
        or any(not isinstance(line, str) or not line for line in pip_freeze)
    ):
        raise ValueError("Teacher verification is missing the full reference pip freeze")
    freeze_text = "\n".join(pip_freeze) + "\n"
    if reference_environment.get("pip_freeze_sha256") != hashlib.sha256(
        freeze_text.encode("utf-8")
    ).hexdigest():
        raise ValueError("Teacher verification reference pip freeze hash is invalid")
    interpreter = reference_environment["interpreter"]
    expected_argv = [
        interpreter,
        "generate.py",
        "experiment=sc09",
        "model=wavenet",
        "generate.ckpt_iter=1000000",
        "generate.n_samples=1",
        "generate.batch_size=1",
    ]
    upstream_repository = upstream.get("repository")
    if not isinstance(upstream_repository, str) or not upstream_repository:
        raise ValueError("Teacher verification is missing the upstream repository path")
    expected_command = (
        f"cd {shlex.quote(upstream_repository)}\n"
        f"CUDA_VISIBLE_DEVICES=0 "
        f"{' '.join(shlex.quote(value) for value in expected_argv)}"
    )
    if (
        reference_generation.get("working_directory") != upstream_repository
        or reference_generation.get("cuda_visible_devices") != "0"
        or reference_generation.get("argv") != expected_argv
        or reference_generation.get("command") != expected_command
        or reference_environment.get("pip_freeze_command")
        != f"{shlex.quote(interpreter)} -m pip freeze"
    ):
        raise ValueError(
            "Teacher verification does not record the exact untouched generation command"
        )
    return {
        "path": str(report_path),
        "sha256": _file_sha256(report_path),
        "format": report["format"],
        "passed": True,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_artifact_sha256_manifest(output_dir: Path) -> dict[str, str]:
    """Hash every finalized, non-hidden file below an analysis output root."""

    manifest: dict[str, str] = {}
    for path in sorted(output_dir.rglob("*")):
        if (
            not path.is_file()
            or path.name == "_SUCCESS.json"
            or any(part.startswith(".") for part in path.relative_to(output_dir).parts)
        ):
            continue
        manifest[path.relative_to(output_dir).as_posix()] = _file_sha256(path)
    return manifest


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def configure_deterministic_runtime() -> dict[str, Any]:
    """Apply and record the exact FP32 CUDA execution policy for analysis."""

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return {
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "autocast": False,
        "torch_compile": False,
    }


def runtime_environment(device: torch.device, determinism: Mapping[str, Any]) -> dict[str, Any]:
    packages: dict[str, Optional[str]] = {}
    for name in ("torch", "torchaudio", "torchcodec", "numpy", "scipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "determinism": dict(determinism),
    }


def runtime_cache_identity(device: torch.device) -> dict[str, Any]:
    """Stable execution identity included in cache compatibility checks."""

    packages: dict[str, Optional[str]] = {}
    for name in ("torch", "torchaudio", "torchcodec"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        **packages,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_type": device.type,
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
    }


class DiffWaveCorruptionDataset(Dataset):
    """Build a bounded round-robin population at each selected native timestep."""

    def __init__(
        self,
        audio_dataset: Dataset,
        *,
        num_diffusion_steps: int,
        seed: int,
        num_timestep_levels: int,
        samples_per_timestep: int,
    ) -> None:
        if num_diffusion_steps <= 0:
            raise ValueError("num_diffusion_steps must be positive")
        if not 1 <= num_timestep_levels <= num_diffusion_steps:
            raise ValueError(
                f"num_timestep_levels must be in [1, {num_diffusion_steps}], "
                f"got {num_timestep_levels}"
            )
        if samples_per_timestep < 2:
            raise ValueError("samples_per_timestep must be at least 2 for exact-timestep PFI")
        if samples_per_timestep > len(audio_dataset):
            raise ValueError(
                f"samples_per_timestep={samples_per_timestep} exceeds the "
                f"{len(audio_dataset)} selected base examples"
            )
        self.audio_dataset = audio_dataset
        self.base_population_size = len(audio_dataset)
        # The PFI sampler sees the fixed number of replicas at each level, not
        # the size of the rotating base-audio pool.
        self.num_examples = int(samples_per_timestep)
        self.level_indices = (
            torch.linspace(
                num_diffusion_steps - 1,
                0,
                steps=num_timestep_levels,
                dtype=torch.float64,
            )
            .round()
            .to(torch.long)
            .tolist()
        )
        if len(set(self.level_indices)) != len(self.level_indices):
            raise RuntimeError("Representative native timesteps are not unique")
        self.noise_seeds: list[int] = []
        generator = random.Random(seed)
        for _ in range(self.base_population_size):
            self.noise_seeds.append(generator.randrange(0, 2**31 - 1))
        if len(set(self.noise_seeds)) != len(self.noise_seeds):
            raise RuntimeError(
                "Fixed-corruption noise seeds must uniquely identify the selected base examples"
            )

    def __len__(self) -> int:
        return self.num_examples * len(self.level_indices)

    def sample_index(self, example_index: int, level_position: int) -> int:
        if not 0 <= example_index < self.num_examples:
            raise IndexError(example_index)
        if not 0 <= level_position < len(self.level_indices):
            raise IndexError(level_position)
        return level_position * self.num_examples + example_index

    def base_example_index(self, replica_index: int, level_position: int) -> int:
        if not 0 <= replica_index < self.num_examples:
            raise IndexError(replica_index)
        if not 0 <= level_position < len(self.level_indices):
            raise IndexError(level_position)
        return (
            level_position * self.num_examples + replica_index
        ) % self.base_population_size

    def __getitem__(self, index: int | PFILevelSampleIndex):
        pfi_reference = index if isinstance(index, PFILevelSampleIndex) else None
        flat_index = pfi_reference.sample_index if pfi_reference is not None else int(index)
        level_position, replica_index = divmod(flat_index, self.num_examples)
        timestep = self.level_indices[level_position]
        if pfi_reference is not None:
            if (
                replica_index != pfi_reference.example_index
                or timestep != pfi_reference.level_index
            ):
                raise RuntimeError("PFI plan does not match the DiffWave corruption dataset")
        base_index = self.base_example_index(replica_index, level_position)
        item = self.audio_dataset[base_index]
        waveform = item[0] if isinstance(item, tuple) else item
        if not torch.is_tensor(waveform) or waveform.ndim != 2:
            raise ValueError(
                "SC09 dataset must return a waveform tensor [channels, length], "
                f"got {type(waveform).__name__}"
            )
        base = (
            waveform,
            int(timestep),
            int(level_position),
            int(self.noise_seeds[base_index]),
        )
        if pfi_reference is None:
            return base
        return (*base, int(pfi_reference.donor_position))


class PreloadedAudioDataset(Dataset):
    """Small immutable in-memory view of a selected audio population.

    Per-filter analysis revisits the same bounded SC09 population tens of
    thousands of times.  Loading each WAV once per rank removes repeated NFS
    and TorchAudio work without changing the decoded tensors or population
    fingerprint used by the PFI plan.
    """

    def __init__(self, source: Dataset) -> None:
        self.source = source
        self.metadata = dict(getattr(source, "metadata", {}))
        items: list[tuple[torch.Tensor, Any]] = []
        for index in range(len(source)):
            item = source[index]
            if not isinstance(item, tuple) or len(item) < 2:
                raise ValueError("Preloaded audio source must return (waveform, label)")
            waveform = item[0]
            if (
                not torch.is_tensor(waveform)
                or waveform.dtype != torch.float32
                or waveform.ndim != 2
                or not torch.isfinite(waveform).all()
            ):
                raise ValueError("Preloaded audio contains an invalid waveform")
            items.append((waveform.detach().cpu().clone(), item[1]))
        self.items = tuple(items)
        self.metadata["preloaded_in_memory"] = True
        self.metadata["preloaded_count"] = len(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Any]:
        return self.items[index]


def collate_diffwave_corruption(batch):
    waveforms, timesteps, level_positions, noise_seeds = zip(*batch)
    return (
        torch.stack(waveforms, dim=0),
        torch.tensor(timesteps, dtype=torch.long),
        torch.tensor(level_positions, dtype=torch.long),
        torch.tensor(noise_seeds, dtype=torch.long),
        None,
    )


def collate_diffwave_pfi(batch):
    waveforms, timesteps, level_positions, noise_seeds, donors = zip(*batch)
    donor_positions = [int(value) for value in donors]
    if sorted(donor_positions) != list(range(len(donor_positions))):
        raise ValueError("PFI donor positions must form a batch-local bijection")
    if any(index == donor for index, donor in enumerate(donor_positions)):
        raise ValueError("PFI donor positions must be fixed-point-free")
    return (
        torch.stack(waveforms, dim=0),
        torch.tensor(timesteps, dtype=torch.long),
        torch.tensor(level_positions, dtype=torch.long),
        torch.tensor(noise_seeds, dtype=torch.long),
        torch.tensor(donor_positions, dtype=torch.long),
    )


def make_timestep_axis_metadata(
    *,
    evaluated_timesteps: Sequence[int],
    num_diffusion_steps: int,
    num_bins: int,
    alpha_bar: torch.Tensor,
) -> tuple[list[str], AnalysisAxis]:
    if not 0 < num_bins <= len(evaluated_timesteps):
        raise ValueError(
            f"num_bins must be in [1, {len(evaluated_timesteps)}], got {num_bins}"
        )
    expected_order = sorted(evaluated_timesteps, reverse=True)
    if list(evaluated_timesteps) != expected_order:
        raise ValueError("DiffWave timesteps must be ordered high-to-low noise")

    labels: list[str] = []
    bin_records: list[dict[str, Any]] = []
    bin_assignments = level_to_bin(
        torch.arange(len(evaluated_timesteps), dtype=torch.long),
        num_levels=len(evaluated_timesteps),
        num_bins=num_bins,
    ).tolist()
    for bin_index in range(num_bins):
        members = [
            int(value)
            for position, value in enumerate(evaluated_timesteps)
            if bin_assignments[position] == bin_index
        ]
        if not members:
            raise ValueError(f"Timestep bin {bin_index} is empty")
        label = str(members[0]) if len(members) == 1 else f"{members[0]}-{members[-1]}"
        labels.append(label)
        normalized = [value / float(num_diffusion_steps - 1) for value in members]
        selected_alpha_bar = alpha_bar[torch.tensor(members, dtype=torch.long)].to(torch.float64)
        log_snr = torch.log(
            selected_alpha_bar.clamp(min=1e-30)
            / (1.0 - selected_alpha_bar).clamp(min=1e-30)
        )
        equivalent_sigma = torch.sqrt(
            (1.0 - selected_alpha_bar).clamp(min=0.0)
            / selected_alpha_bar.clamp(min=1e-30)
        )
        bin_records.append(
            {
                "bin_index": bin_index,
                "label": label,
                "native_timesteps": members,
                "normalized_timesteps": normalized,
                "native_timestep_bounds": [members[0], members[-1]],
                "normalized_timestep_bounds": [normalized[0], normalized[-1]],
                "log_snr_bounds": [float(log_snr[0]), float(log_snr[-1])],
                "equivalent_sigma_bounds": [
                    float(equivalent_sigma[0]),
                    float(equivalent_sigma[-1]),
                ],
            }
        )

    sampled_alpha_bar = alpha_bar[
        torch.tensor(evaluated_timesteps, dtype=torch.long)
    ].to(torch.float64)
    normalized_timesteps = tuple(
        value / float(num_diffusion_steps - 1) for value in evaluated_timesteps
    )
    return labels, AnalysisAxis(
        kind="timestep",
        ordering="high_noise_to_low_noise",
        native_coordinate="diffwave_timestep",
        normalized_coordinate="timestep_over_T_minus_1",
        native_values=tuple(int(value) for value in evaluated_timesteps),
        normalized_values=normalized_timesteps,
        bin_labels=tuple(labels),
        bin_members=tuple(
            tuple(int(value) for value in record["native_timesteps"])
            for record in bin_records
        ),
        metadata={
            "num_diffusion_steps": int(num_diffusion_steps),
            # Explicit compatibility aliases make raw DiffWave coordinates
            # easy to inspect without knowing the generic axis schema.
            "evaluated_timesteps": [int(value) for value in evaluated_timesteps],
            "normalized_timesteps": list(normalized_timesteps),
            "alpha_bar": sampled_alpha_bar.tolist(),
            "log_snr": torch.log(
                sampled_alpha_bar.clamp(min=1e-30)
                / (1.0 - sampled_alpha_bar).clamp(min=1e-30)
            ).tolist(),
            "equivalent_sigma": torch.sqrt(
                (1.0 - sampled_alpha_bar).clamp(min=0.0)
                / sampled_alpha_bar.clamp(min=1e-30)
            ).tolist(),
            "bins": bin_records,
        },
    )


def collect_diffwave_groups(
    model: LegacyCompatibleDiffWave,
) -> tuple["OrderedDict[str, torch.nn.Module]", dict[str, str]]:
    groups: "OrderedDict[str, torch.nn.Module]" = OrderedDict()
    module_paths: dict[str, str] = {}
    for index, (module_path, module) in enumerate(model.named_residual_blocks()):
        name = f"residual_block_{index:02d}"
        groups[name] = module
        module_paths[name] = module_path
    return groups, module_paths


@dataclass(frozen=True)
class DiffWaveGroupCatalog:
    grouping: str
    groups: "OrderedDict[str, Any]"
    module_paths: dict[str, str]
    structural_keys: dict[str, str]
    stage_keys: dict[str, str]
    allocation_structural_keys: dict[str, Optional[str]]
    allocatable: dict[str, bool]
    filter_indices: dict[str, Optional[int]]
    parameter_counts: dict[str, int]
    edm_proxy_parameter_counts: dict[str, int]

    def selected(self, names: Sequence[str]) -> "DiffWaveGroupCatalog":
        selected_names = list(names)
        return DiffWaveGroupCatalog(
            grouping=self.grouping,
            groups=OrderedDict((name, self.groups[name]) for name in selected_names),
            module_paths={name: self.module_paths[name] for name in selected_names},
            structural_keys={name: self.structural_keys[name] for name in selected_names},
            stage_keys={name: self.stage_keys[name] for name in selected_names},
            allocation_structural_keys={
                name: self.allocation_structural_keys[name] for name in selected_names
            },
            allocatable={name: self.allocatable[name] for name in selected_names},
            filter_indices={name: self.filter_indices[name] for name in selected_names},
            parameter_counts={name: self.parameter_counts[name] for name in selected_names},
            edm_proxy_parameter_counts={
                name: self.edm_proxy_parameter_counts[name] for name in selected_names
            },
        )


def _conv_filter_true_parameter_count(module: torch.nn.Conv1d, filter_index: int) -> int:
    """Count the disjoint trainable slice owned by one output channel."""

    count = 0
    for name, parameter in module.named_parameters(recurse=False):
        if parameter.ndim == 0 or parameter.shape[0] != module.out_channels:
            raise ValueError(
                f"Conv1d parameter {name!r} cannot be assigned by output channel: "
                f"shape={tuple(parameter.shape)}, out_channels={module.out_channels}"
            )
        count += int(parameter[filter_index].numel())
    return count


def _conv_filter_edm_proxy_parameter_count(
    module: torch.nn.Conv1d,
    filter_index: int,
) -> int:
    weight = getattr(module, "weight", None)
    if not torch.is_tensor(weight):
        raise ValueError("Conv1d filter proxy requires a weight tensor")
    return int(weight[filter_index].numel()) + (1 if module.bias is not None else 0)


def _diffwave_stage_for_conv(module_path: str) -> tuple[str, Optional[str], bool]:
    prefix = "residual_layer.residual_blocks."
    if module_path.startswith(prefix):
        suffix = module_path[len(prefix) :]
        block_text = suffix.split(".", 1)[0]
        if not block_text.isdigit():
            raise ValueError(f"Could not parse residual-block index from {module_path!r}")
        stage = f"residual_block_{int(block_text):02d}"
        return stage, stage, True
    if module_path.startswith("init_conv."):
        return "stem", None, False
    if module_path.startswith("final_conv."):
        return "head", None, False
    raise ValueError(f"Unexpected DiffWave Conv1d module path {module_path!r}")


def collect_diffwave_group_catalog(
    model: LegacyCompatibleDiffWave,
    *,
    grouping: str,
) -> DiffWaveGroupCatalog:
    if grouping == "residual_blocks":
        groups, module_paths = collect_diffwave_groups(model)
        names = list(groups)
        counts = {name: count_parameters(groups[name]) for name in names}
        return DiffWaveGroupCatalog(
            grouping=grouping,
            groups=OrderedDict(groups),
            module_paths=module_paths,
            structural_keys={name: name for name in names},
            stage_keys={name: name for name in names},
            allocation_structural_keys={name: name for name in names},
            allocatable={name: True for name in names},
            filter_indices={name: None for name in names},
            parameter_counts=counts,
            edm_proxy_parameter_counts=dict(counts),
        )
    if grouping != "per_filter":
        raise ValueError(f"Unsupported DiffWave grouping {grouping!r}")

    # Imported lazily so the legacy residual-block mode remains usable when a
    # downstream checkout has not yet adopted indexed targets.
    from pace.parameter_analysis import IndexedOutputTarget

    groups: "OrderedDict[str, Any]" = OrderedDict()
    module_paths: dict[str, str] = {}
    structural_keys: dict[str, str] = {}
    stage_keys: dict[str, str] = {}
    allocation_keys: dict[str, Optional[str]] = {}
    allocatable: dict[str, bool] = {}
    filter_indices: dict[str, Optional[int]] = {}
    parameter_counts: dict[str, int] = {}
    proxy_counts: dict[str, int] = {}
    conv_modules = 0
    for module_path, module in model.named_modules():
        if not module_path or not isinstance(module, torch.nn.Conv1d):
            continue
        conv_modules += 1
        stage_key, allocation_key, is_allocatable = _diffwave_stage_for_conv(module_path)
        for filter_index in range(int(module.out_channels)):
            name = f"{module_path}.filter_{filter_index}"
            groups[name] = IndexedOutputTarget(
                module=module,
                channel_index=filter_index,
                channel_dim=1,
            )
            module_paths[name] = module_path
            structural_keys[name] = module_path
            stage_keys[name] = stage_key
            allocation_keys[name] = allocation_key
            allocatable[name] = is_allocatable
            filter_indices[name] = filter_index
            parameter_counts[name] = _conv_filter_true_parameter_count(module, filter_index)
            proxy_counts[name] = _conv_filter_edm_proxy_parameter_count(module, filter_index)

    if len(model.residual_blocks) == 36 and count_parameters(model) == 24_071_681:
        observed = {
            "conv_modules": conv_modules,
            "groups": len(groups),
            "true_parameters": sum(parameter_counts.values()),
            "edm_proxy_parameters": sum(proxy_counts.values()),
        }
        expected = {
            "conv_modules": PER_FILTER_EXPECTED_CONV_MODULES,
            "groups": PER_FILTER_EXPECTED_GROUPS,
            "true_parameters": PER_FILTER_EXPECTED_TRUE_PARAMETERS,
            "edm_proxy_parameters": PER_FILTER_EXPECTED_EDM_PROXY_PARAMETERS,
        }
        if observed != expected:
            raise RuntimeError(
                f"DiffWave per-filter catalog differs from the pinned teacher: "
                f"observed={observed}, expected={expected}"
            )

    return DiffWaveGroupCatalog(
        grouping=grouping,
        groups=groups,
        module_paths=module_paths,
        structural_keys=structural_keys,
        stage_keys=stage_keys,
        allocation_structural_keys=allocation_keys,
        allocatable=allocatable,
        filter_indices=filter_indices,
        parameter_counts=parameter_counts,
        edm_proxy_parameter_counts=proxy_counts,
    )


def select_groups(
    groups: "OrderedDict[str, Any]",
    *,
    max_groups: Optional[int],
    mode: str,
    seed: int,
) -> "OrderedDict[str, Any]":
    if max_groups is None or max_groups >= len(groups):
        return groups
    if max_groups <= 0:
        raise ValueError("max_groups must be positive")
    names = list(groups)
    if mode == "first":
        selected = names[:max_groups]
    elif mode == "random":
        selected_set = set(random.Random(seed).sample(names, max_groups))
        selected = [name for name in names if name in selected_set]
    elif mode == "stratified":
        if max_groups == 1:
            selected = [names[len(names) // 2]]
        else:
            positions = [
                round(index * (len(names) - 1) / (max_groups - 1))
                for index in range(max_groups)
            ]
            selected = [names[position] for position in positions]
            if len(set(selected)) != len(selected):
                raise RuntimeError("Stratified group selection produced duplicate groups")
    else:
        raise ValueError(f"Unsupported max_groups_mode: {mode}")
    return OrderedDict((name, groups[name]) for name in selected)


def select_group_catalog(
    catalog: DiffWaveGroupCatalog,
    *,
    max_groups: Optional[int],
    mode: str,
    seed: int,
) -> DiffWaveGroupCatalog:
    if max_groups is None or max_groups >= len(catalog.groups):
        return catalog
    if mode != "stratified" or catalog.grouping != "per_filter":
        selected = select_groups(
            catalog.groups,
            max_groups=max_groups,
            mode=mode,
            seed=seed,
        )
        return catalog.selected(list(selected))

    if max_groups <= 0:
        raise ValueError("max_groups must be positive")
    names = list(catalog.groups)
    module_to_names: "OrderedDict[str, list[str]]" = OrderedDict()
    for name in names:
        module_to_names.setdefault(catalog.module_paths[name], []).append(name)
    chosen: set[str] = set()
    if max_groups >= len(module_to_names):
        for module_names in module_to_names.values():
            chosen.add(module_names[len(module_names) // 2])
    remaining = max_groups - len(chosen)
    candidates = [name for name in names if name not in chosen]
    if remaining > 0:
        if remaining == 1:
            chosen.add(candidates[len(candidates) // 2])
        else:
            positions = [
                round(index * (len(candidates) - 1) / (remaining - 1))
                for index in range(remaining)
            ]
            chosen.update(candidates[position] for position in positions)
    selected_names = [name for name in names if name in chosen]
    if len(selected_names) != max_groups:
        raise RuntimeError(
            f"Stratified catalog selection produced {len(selected_names)} groups, "
            f"expected {max_groups}"
        )
    return catalog.selected(selected_names)


def _safe_row_normalize(matrix: torch.Tensor) -> torch.Tensor:
    matrix = torch.nan_to_num(matrix.to(torch.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return torch.nan_to_num(
        matrix / (matrix.sum(dim=1, keepdim=True) + 1e-12),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def aggregate_filter_matrices(
    *,
    group_names: Sequence[str],
    keys: Mapping[str, str],
    signed_delta_stack: torch.Tensor,
    delta_stack: torch.Tensor,
    baseline_mean: torch.Tensor,
    parameter_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Aggregate individual-filter estimands by an explicit structural key."""

    key_to_indices: "OrderedDict[str, list[int]]" = OrderedDict()
    for index, name in enumerate(group_names):
        key_to_indices.setdefault(keys[name], []).append(index)
    names = list(key_to_indices)
    signed_source = signed_delta_stack.to(torch.float64)
    positive_source = delta_stack.to(torch.float64)
    signed = torch.stack(
        [signed_source[indices].sum(dim=0) for indices in key_to_indices.values()]
    )
    positive = torch.stack(
        [positive_source[indices].sum(dim=0) for indices in key_to_indices.values()]
    )
    relative = torch.nan_to_num(
        positive / (baseline_mean.to(torch.float64).unsqueeze(0) + 1e-12),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if len(names) == 1:
        correlation = torch.ones((1, 1), dtype=torch.float64)
    elif relative.shape[1] < 2:
        correlation = torch.zeros((len(names), len(names)), dtype=torch.float64)
    else:
        correlation = torch.nan_to_num(
            torch.corrcoef(relative), nan=0.0, posinf=0.0, neginf=0.0
        )
    return {
        "names": names,
        "member_counts": [len(indices) for indices in key_to_indices.values()],
        "parameter_counts": {
            key: sum(parameter_counts[group_names[index]] for index in indices)
            for key, indices in key_to_indices.items()
        },
        "signed_delta_stack": signed,
        "delta_stack": positive,
        "relative_delta_stack": relative,
        "row_normalized_relative_delta_stack": _safe_row_normalize(relative),
        "correlation": correlation,
    }


def top_filter_indices(relative_delta_stack: torch.Tensor, count: int) -> list[int]:
    count = min(int(count), int(relative_delta_stack.shape[0]))
    scores = relative_delta_stack.to(torch.float64).sum(dim=1)
    # Python's stable ordering gives the canonical group index as the tie-break.
    return sorted(
        range(int(scores.numel())),
        key=lambda index: (-float(scores[index]), index),
    )[:count]


def tensor_aggregate_to_json(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.tolist() if torch.is_tensor(value) else value
        for key, value in record.items()
    }


def save_per_filter_aggregate_plots(
    output_dir: Path,
    aggregates: Mapping[str, Mapping[str, Any]],
    *,
    timestep_labels: Sequence[str],
) -> None:
    for view_name, record in aggregates.items():
        names = list(record["names"])
        for matrix_key, suffix, title, zmin, zmax in [
            (
                "relative_delta_stack",
                "relative_delta_heatmap",
                "Summed individual-filter relative PFI deltas",
                None,
                None,
            ),
            (
                "row_normalized_relative_delta_stack",
                "row_normalized_relative_delta_heatmap",
                "Row-normalized summed individual-filter relative PFI deltas",
                0.0,
                1.0,
            ),
        ]:
            plot_labeled_heatmap(
                matrix=record[matrix_key],
                row_labels=names,
                col_labels=timestep_labels,
                out_path=output_dir / f"{view_name}_{suffix}.png",
                title=f"{view_name.title()} view: {title}",
                xaxis_title="Timestep bin",
                yaxis_title=view_name.title(),
                colorbar_title="Relative delta",
                row_hover_label=view_name,
                col_hover_label="timestep_bin",
                hover_value_label="relative_delta",
                zmin=zmin,
                zmax=zmax,
            )
        plot_labeled_heatmap(
            matrix=record["correlation"],
            row_labels=names,
            col_labels=names,
            out_path=output_dir / f"{view_name}_correlation_heatmap.png",
            title=(
                f"{view_name.title()} correlation of summed individual-filter PFI profiles"
            ),
            xaxis_title=view_name.title(),
            yaxis_title=view_name.title(),
            colorbar_title="Correlation",
            row_hover_label=f"{view_name}_y",
            col_hover_label=f"{view_name}_x",
            hover_value_label="correlation",
            zmin=-1.0,
            zmax=1.0,
        )


class DiffWaveUsageEvaluator:
    def __init__(
        self,
        model: LegacyCompatibleDiffWave,
        schedule: Mapping[str, Any],
        *,
        device: torch.device,
        dtype: torch.dtype,
        axis: AnalysisAxis,
        groups: Mapping[str, Any],
        cache_fixed_corruptions: bool = False,
    ) -> None:
        self.model = model
        self.schedule = schedule
        self.device = device
        self.dtype = dtype
        self.axis = axis
        self.groups = OrderedDict(groups)
        self.cache_fixed_corruptions = bool(cache_fixed_corruptions)
        self._fixed_corruption_cache: dict[
            tuple[int, int], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def named_analysis_groups(self) -> tuple[tuple[str, Any], ...]:
        return tuple(self.groups.items())

    def predict_noise(self, x_t: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
        return self.model.predict_noise(x_t, levels)

    def unpack_analysis_batch(self, batch: Any) -> BinnedEvaluationBatch:
        waveforms, timesteps, level_positions, noise_seeds, pfi_permutation = batch
        return BinnedEvaluationBatch(
            loss_inputs=(waveforms, timesteps, noise_seeds),
            level_positions=level_positions,
            pfi_permutation=pfi_permutation,
        )

    @torch.inference_mode()
    def forward_losses_from_fixed_corruption(
        self,
        waveforms: torch.Tensor,
        timesteps: torch.Tensor,
        noise_seeds: torch.Tensor,
    ) -> torch.Tensor:
        cache_keys = tuple(
            (int(timestep), int(noise_seed))
            for timestep, noise_seed in zip(
                timesteps.detach().cpu().tolist(),
                noise_seeds.detach().cpu().tolist(),
                strict=True,
            )
        )
        cached = (
            [self._fixed_corruption_cache.get(key) for key in cache_keys]
            if self.cache_fixed_corruptions
            else []
        )
        if cached and all(value is not None for value in cached):
            x_t = torch.stack([value[0] for value in cached if value is not None])
            noise = torch.stack([value[1] for value in cached if value is not None])
            timesteps = timesteps.to(self.device, dtype=torch.long, non_blocking=True)
        else:
            waveforms = waveforms.to(self.device, dtype=self.dtype, non_blocking=True)
            timesteps = timesteps.to(self.device, dtype=torch.long, non_blocking=True)
            noise = torch.empty_like(waveforms)
            for index, seed in enumerate(noise_seeds.tolist()):
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int(seed))
                sample = torch.randn(
                    tuple(waveforms[index].shape),
                    generator=generator,
                    device="cpu",
                    dtype=torch.float32,
                )
                noise[index] = sample.to(device=self.device, dtype=self.dtype)
            alpha_bar = self.schedule["Alpha_bar"].to(self.device, dtype=self.dtype)[timesteps]
            x_t = (
                torch.sqrt(alpha_bar).view(-1, 1, 1) * waveforms
                + torch.sqrt(1.0 - alpha_bar).view(-1, 1, 1) * noise
            )
            if self.cache_fixed_corruptions:
                for index, key in enumerate(cache_keys):
                    candidate = (x_t[index].detach().clone(), noise[index].detach().clone())
                    previous = self._fixed_corruption_cache.setdefault(key, candidate)
                    if not torch.equal(previous[0], candidate[0]) or not torch.equal(
                        previous[1], candidate[1]
                    ):
                        raise ValueError(
                            "A fixed-corruption cache key resolved to different tensors"
                        )
        predicted_noise = self.predict_noise(x_t, timesteps.to(torch.float32))
        if predicted_noise.shape != noise.shape:
            raise ValueError(
                f"DiffWave predicted shape {tuple(predicted_noise.shape)}, "
                f"expected {tuple(noise.shape)}"
            )
        if not torch.isfinite(predicted_noise).all():
            raise ValueError("DiffWave produced non-finite predicted noise")
        return mse_per_example(predicted_noise.float(), noise.float())

    def fixed_corruption_cache_record(self) -> dict[str, Any]:
        tensor_bytes = sum(
            first.numel() * first.element_size() + second.numel() * second.element_size()
            for first, second in self._fixed_corruption_cache.values()
        )
        key_digest = canonical_json_sha256(
            [[timestep, noise_seed] for timestep, noise_seed in sorted(self._fixed_corruption_cache)]
        )
        return {
            "enabled": self.cache_fixed_corruptions,
            "entries": len(self._fixed_corruption_cache),
            "tensor_bytes": int(tensor_bytes),
            "key_sha256": key_digest,
            "device": str(self.device),
            "protocol": "diffwave_fixed_xt_noise_cache_v1",
        }

    def evaluate(
        self,
        dataloader: DataLoader,
        *,
        num_bins: int,
        ablate_target: Optional[torch.nn.Module] = None,
        ablation_mode: str = "zero",
        ablation_random_seed: Optional[int] = None,
        progress_desc: Optional[str] = None,
    ) -> BinStats:
        return evaluate_binned_losses(
            self,
            dataloader,
            num_bins=num_bins,
            ablate_target=ablate_target,
            ablation_mode=ablation_mode,  # type: ignore[arg-type]
            ablation_random_seed=ablation_random_seed,
            progress_desc=progress_desc,
            progress_factory=lambda batches, description: tqdm(
                batches,
                desc=description,
                leave=False,
            ),
        )


def build_dataloader(
    corruption_dataset: DiffWaveCorruptionDataset,
    audio_dataset: SC09Dataset,
    *,
    ablation_mode: str,
    batch_size: int,
    num_workers: int,
    pfi_seed: int,
):
    loader_options: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        # The teacher is commonly moved to CUDA before the DataLoader is first
        # iterated.  Forking workers after CUDA initialization can deadlock;
        # spawn clean workers and keep them alive across the baseline and all
        # group-ablation passes instead.
        loader_options.update(
            multiprocessing_context="spawn",
            persistent_workers=True,
        )
    if ablation_mode == "pfi":
        sampler = ExactLevelPFIBatchSampler(
            corruption_dataset,
            batch_size=batch_size,
            pfi_seed=pfi_seed,
            population_fingerprint=dataset_population_fingerprint(audio_dataset),
            level_kind="timestep",
        )
        loader = DataLoader(
            corruption_dataset,
            batch_sampler=sampler,
            collate_fn=collate_diffwave_pfi,
            **loader_options,
        )
        return loader, sampler
    loader = DataLoader(
        corruption_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_diffwave_corruption,
        **loader_options,
    )
    return loader, None


def run_postprocessing(
    output_dir: Path,
    *,
    grouping: str,
    num_timestep_groups: int,
    score_reduction: str,
    allocation_variants: Sequence[str],
) -> list[list[str]]:
    if grouping == "per_filter":
        if num_timestep_groups != 3:
            raise ValueError(
                "corrected per-filter postprocessing requires three phases for "
                "the canonical fixed-Pearson design"
            )
        if score_reduction != "mean":
            raise ValueError(
                "corrected per-filter geometric allocation requires the "
                "duration-neutral score_reduction='mean'"
            )
        required_variants = {"combined_blockwise", "combined_layerwise"}
        if not required_variants.issubset(allocation_variants):
            raise ValueError(
                "corrected per-filter postprocessing requires allocation variants "
                "combined_blockwise and combined_layerwise"
            )

        residual_dir = output_dir / "residual_only"
        commands: list[list[str]] = []
        residual_command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "paper" / "postprocess_diffwave_residual.py"),
            "--results",
            str(output_dir),
            "--output-dir",
            str(residual_dir),
        ]
        subprocess.run(residual_command, cwd=REPO_ROOT, check=True)
        commands.append(residual_command)

        grouping_specs = (
            (
                "grouping_spearman_auto",
                "matrix_spearman_cross_penalty",
                ["--select_num_blocks", "--max_num_blocks", "5"],
            ),
            (
                "grouping_pearson_auto",
                "matrix_correlation_cross_penalty",
                ["--select_num_blocks", "--max_num_blocks", "5"],
            ),
            (
                "grouping_pearson_k3_min2",
                "matrix_correlation_cross_penalty",
                [
                    "--cross_block_lambda",
                    "0",
                    "--num_blocks",
                    str(num_timestep_groups),
                ],
            ),
            (
                "grouping_spearman_k3_min2",
                "matrix_spearman_cross_penalty",
                ["--num_blocks", str(num_timestep_groups)],
            ),
        )
        grouping_paths: dict[str, Path] = {}
        for directory_name, objective, selection_args in grouping_specs:
            grouping_dir = residual_dir / directory_name
            grouping_path = grouping_dir / "timestep_grouping.json"
            grouping_paths[directory_name] = grouping_path
            grouping_command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "optimize_timestep_grouping.py"),
                "--matrix",
                str(residual_dir / "results.json"),
                "--matrix_key",
                "relative_delta_stack",
                "--builtin_cost",
                objective,
                "--min-block-size",
                "2",
                *selection_args,
                "--output",
                str(grouping_path),
                "--plot_output",
                str(grouping_dir / "timestep_grouping_matrix.png"),
                "--plot_matrix_source",
                "recomputed_similarity",
                "--plot_labels_key",
                "timestep_bin_labels",
                "--plot_series_output",
                str(grouping_dir / "timestep_grouping_n_eff.png"),
                "--plot_series",
                str(residual_dir / "results.json"),
                "--plot_series_key",
                "n_eff",
                "--plot_series_ylabel",
                "N_eff",
                "--cost_curve_output",
                str(grouping_dir / "timestep_grouping_cost_curve.png"),
            ]
            subprocess.run(grouping_command, cwd=REPO_ROOT, check=True)
            commands.append(grouping_command)

        allocation_specs = (
            (
                "allocation_geomean_mean_spearman_auto",
                grouping_paths["grouping_spearman_auto"],
            ),
            (
                "allocation_geomean_mean_pearson_auto",
                grouping_paths["grouping_pearson_auto"],
            ),
            (
                "allocation_geomean_mean_pearson_k3_min2",
                grouping_paths["grouping_pearson_k3_min2"],
            ),
            (
                "allocation_geomean_mean_spearman_k3_min2",
                grouping_paths["grouping_spearman_k3_min2"],
            ),
        )
        for directory_name, grouping_path in allocation_specs:
            allocation_command = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "dry_run_capacity_allocation.py"),
                "--eval-output-dir",
                str(residual_dir),
                "--timestep-grouping",
                str(grouping_path),
                "--allocation-results-dir",
                str(residual_dir / directory_name),
                "--allocation-metric",
                "delta_p_eff_geomean",
                "--allocation-group-scope",
                "allocatable",
                "--layer-score-source",
                "delta_p_eff_geomean",
                "--score-reduction",
                "mean",
                "--allocation-alpha",
                "1",
            ]
            for variant in allocation_variants:
                allocation_command.extend(["--student-variant", variant])
            subprocess.run(allocation_command, cwd=REPO_ROOT, check=True)
            commands.append(allocation_command)

        report_command = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "paper" / "report_diffwave_per_filter_decision.py"),
            "--residual-dir",
            str(residual_dir),
        ]
        subprocess.run(report_command, cwd=REPO_ROOT, check=True)
        commands.append(report_command)
        return commands

    results_path = output_dir / "results.json"
    grouping_dir = output_dir / "grouping"
    grouping_path = grouping_dir / "timestep_grouping.json"
    grouping_command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "optimize_timestep_grouping.py"),
        "--matrix",
        str(results_path),
        "--matrix_key",
        "relative_delta_stack",
        "--builtin_cost",
        "matrix_correlation_cross_penalty",
        "--num_blocks",
        str(num_timestep_groups),
        "--output",
        str(grouping_path),
        "--plot_output",
        str(grouping_dir / "timestep_grouping_matrix.png"),
        "--plot_matrix_source",
        "recomputed_similarity",
        "--plot_labels_key",
        "timestep_bin_labels",
        "--plot_series_output",
        str(grouping_dir / "timestep_grouping_n_eff.png"),
        "--plot_series",
        str(results_path),
        "--plot_series_key",
        "n_eff",
        "--plot_series_ylabel",
        "N_eff",
        "--cost_curve_output",
        str(grouping_dir / "timestep_grouping_cost_curve.png"),
    ]
    subprocess.run(grouping_command, cwd=REPO_ROOT, check=True)

    allocation_command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "dry_run_capacity_allocation.py"),
        "--eval-output-dir",
        str(output_dir),
        "--allocation-metric",
        "n_eff",
        "--score-reduction",
        score_reduction,
    ]
    for variant in allocation_variants:
        allocation_command.extend(["--student-variant", variant])
    subprocess.run(allocation_command, cwd=REPO_ROOT, check=True)
    return [grouping_command, allocation_command]


def _parse_config_defaults(path: str | None) -> dict[str, Any]:
    return {} if path is None else load_json(path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)
    defaults = _parse_config_defaults(pre_args.config)

    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument("--data-root", "--data_root", dest="data_root", default=None)
    parser.add_argument("--dataset-split", "--dataset_split", dest="dataset_split", default="validation")
    parser.add_argument("--max-samples", "--max_samples", dest="max_samples", type=int, default=None)
    parser.add_argument("--subset-seed", "--subset_seed", dest="subset_seed", type=int, default=0)
    parser.add_argument("--strict-protocol", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataset-preflight", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--teacher-preset", "--teacher_preset", dest="teacher_preset", default="sc09_diffwave_legacy_1m")
    parser.add_argument("--teacher-source", "--teacher_source", dest="teacher_source", default=None)
    parser.add_argument("--model-cache-dir", "--model_cache_dir", dest="model_cache_dir", default=None)
    parser.add_argument("--teacher-verification-report", "--teacher_verification_report", dest="teacher_verification_report", default=None)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp32"], default="fp32")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=8)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=0)
    parser.add_argument("--num-bins", "--num_bins", dest="num_bins", type=int, default=20)
    parser.add_argument("--num-timestep-levels", "--num_timestep_levels", dest="num_timestep_levels", type=int, default=200)
    parser.add_argument("--samples-per-timestep", "--samples_per_timestep", dest="samples_per_timestep", type=int, default=2)
    parser.add_argument("--ablation-mode", "--ablation_mode", dest="ablation_mode", choices=["zero", "random_same_norm", "pfi"], default="pfi")
    parser.add_argument("--grouping", choices=["residual_blocks", "per_filter"], default="residual_blocks")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pfi-seed", "--pfi_seed", dest="pfi_seed", type=int, default=None)
    parser.add_argument("--max-groups", "--max_groups", dest="max_groups", type=int, default=None)
    parser.add_argument("--max-groups-mode", "--max_groups_mode", dest="max_groups_mode", choices=["first", "random", "stratified"], default="first")
    parser.add_argument("--preload-audio", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cache-fixed-corruptions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--top-filter-plot-count", type=int, default=256)
    parser.add_argument("--require-full-filter-catalog", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--distributed-timeout-seconds", type=int, default=259_200)
    parser.add_argument("--checkpoint-interval-groups", type=int, default=1)
    parser.add_argument("--group-correlation", "--group_correlation", dest="group_correlation", choices=["auto", "always", "never"], default="always")
    parser.add_argument("--max-group-correlation-groups", type=int, default=2048)
    parser.add_argument("--require-positive-delta-mass", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-postprocessing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-timestep-groups", type=int, default=3)
    parser.add_argument("--score-reduction", choices=["mean", "sum", "max", "q90"], default="q90")
    parser.add_argument("--allocation-variants", nargs="+", default=["blockwise_capacity", "layerwise_capacity"])
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.pfi_seed is None:
        args.pfi_seed = args.seed
    if args.data_root is None:
        parser.error("--data-root is required")
    if args.output_dir is None:
        parser.error("--output-dir is required")
    if args.model_cache_dir is None:
        parser.error("--model-cache-dir is required")
    if args.teacher_verification_report is None:
        parser.error("--teacher-verification-report is required for every DiffWave analysis")
    if args.grouping == "per_filter" and args.group_correlation != "never":
        parser.error("--grouping=per_filter requires --group-correlation=never")
    if (
        args.grouping == "per_filter"
        and args.run_postprocessing
        and (args.max_groups is not None or not args.require_full_filter_catalog)
    ):
        parser.error(
            "Per-filter grouping/allocation requires the complete filter catalog; "
            "use --no-run-postprocessing for smoke or stability subsets"
        )
    if args.grouping == "per_filter" and args.run_postprocessing:
        if args.num_timestep_groups != 3:
            parser.error(
                "Corrected per-filter postprocessing reserves K=3 for the "
                "fixed-phase sensitivity view; set --num-timestep-groups=3"
            )
        if args.score_reduction != "mean":
            parser.error(
                "Corrected per-filter geometric allocation requires "
                "--score-reduction=mean"
            )
        required_variants = {"combined_blockwise", "combined_layerwise"}
        if not required_variants.issubset(args.allocation_variants):
            parser.error(
                "Corrected per-filter postprocessing requires allocation variants "
                "combined_blockwise and combined_layerwise"
            )
    if args.top_filter_plot_count <= 0:
        parser.error("--top-filter-plot-count must be positive")
    if args.distributed_timeout_seconds <= 0:
        parser.error("--distributed-timeout-seconds must be positive")
    if args.checkpoint_interval_groups <= 0:
        parser.error("--checkpoint-interval-groups must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    started = time.perf_counter()
    args = parse_args(argv)
    distributed_context = init_distributed_analysis(
        args.device,
        timeout_seconds=args.distributed_timeout_seconds,
    )
    determinism = configure_deterministic_runtime()
    set_seed(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    run_rank_zero_analysis_operation(
        distributed_context,
        "analysis output directory creation",
        lambda: output_dir.mkdir(parents=True, exist_ok=True),
    )

    verification = validate_teacher_verification_report(args.teacher_verification_report)

    device = distributed_context.device
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = choose_dtype(args.dtype)
    preset = teacher_preset_config(args.teacher_preset)
    teacher_source = args.teacher_source or preset["source"]
    teacher_spec = resolve_teacher_spec(teacher_source, preset=args.teacher_preset)
    model = load_teacher_network(
        teacher_spec,
        device=device,
        dtype=dtype,
        cache_dir=args.model_cache_dir,
    )
    if not isinstance(model, LegacyCompatibleDiffWave):
        raise TypeError(
            "The SC09 analysis requires LegacyCompatibleDiffWave, got "
            f"{model.__class__.__name__}"
        )
    resolved_teacher_spec = TeacherSpec.from_dict(model.teacher_spec)
    model_metadata = teacher_model_metadata(model, resolved_teacher_spec)
    if model.num_diffusion_steps != 200 or len(model.residual_blocks) != 36:
        raise ValueError(
            "The required teacher must have T=200 and 36 residual blocks; "
            f"got T={model.num_diffusion_steps}, blocks={len(model.residual_blocks)}"
        )

    audio_dataset = SC09Dataset(
        args.data_root,
        split=args.dataset_split,
        max_samples=args.max_samples,
        subset_seed=args.subset_seed,
        strict_protocol=args.strict_protocol,
        preflight=args.dataset_preflight,
    )
    analysis_audio_dataset: Dataset = (
        PreloadedAudioDataset(audio_dataset) if args.preload_audio else audio_dataset
    )
    schedule = diffwave_diffusion_hyperparameters(
        num_diffusion_steps=model.num_diffusion_steps,
        beta_0=model.beta_0,
        beta_T=model.beta_T,
        device="cpu",
        dtype=torch.float32,
    )
    schedule_fingerprint = {
        "protocol": "diffwave_linear_beta_epsilon_corruption_v1",
        "num_diffusion_steps": model.num_diffusion_steps,
        "beta_0": model.beta_0,
        "beta_T": model.beta_T,
        "beta_sha256": _tensor_sha256(schedule["Beta"]),
        "alpha_sha256": _tensor_sha256(schedule["Alpha"]),
        "alpha_bar_sha256": _tensor_sha256(schedule["Alpha_bar"]),
        "corruption": "x_t=sqrt(alpha_bar_t)*x_0+sqrt(1-alpha_bar_t)*epsilon",
        "prediction_target": "epsilon",
        "loss": "per_example_mean_squared_error",
        "noise_rng": "independent_cpu_torch_generator_from_fixed_example_seed",
    }
    corruption_dataset = DiffWaveCorruptionDataset(
        analysis_audio_dataset,
        num_diffusion_steps=model.num_diffusion_steps,
        seed=args.seed,
        num_timestep_levels=args.num_timestep_levels,
        samples_per_timestep=args.samples_per_timestep,
    )
    dataloader, pfi_plan = build_dataloader(
        corruption_dataset,
        audio_dataset,
        ablation_mode=args.ablation_mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pfi_seed=args.pfi_seed,
    )
    pfi_plan_path = output_dir / "pfi_plan.pt"
    run_rank_zero_analysis_operation(
        distributed_context,
        "PFI plan publication",
        lambda: persist_exact_level_pfi_plan(pfi_plan_path, pfi_plan),
    )
    # Every rank reloads and validates the exact published plan before model
    # evaluation.  The rank-zero helper above propagates writer failures rather
    # than leaving peers blocked at a bare barrier.
    pfi_plan_record = persist_exact_level_pfi_plan(pfi_plan_path, pfi_plan)
    if pfi_plan is not None:
        assert_distributed_hash_invariant(
            distributed_context,
            "pfi_plan",
            pfi_plan.plan_sha256,
        )
    ablation_protocol = build_ablation_protocol(
        args.ablation_mode,
        level_kind="timestep",
        pfi_plan=pfi_plan,
        pfi_seed=args.pfi_seed,
    )
    if args.grouping == "per_filter":
        ablation_protocol = json.loads(json.dumps(ablation_protocol))
        ablation_protocol["replacement"] = (
            "batch_local_exact_timestep_single_conv1d_output_channel_activation_exchange"
        )
        ablation_protocol["group_granularity"] = "conv1d_output_filter"
        if "pfi" in ablation_protocol:
            ablation_protocol["pfi"]["whole_group_tensor"] = False
            ablation_protocol["pfi"]["channel_dimension"] = 1
            ablation_protocol["pfi"]["raw_gated_halves_are_separate"] = True

    full_catalog = collect_diffwave_group_catalog(model, grouping=args.grouping)
    selected_catalog = select_group_catalog(
        full_catalog,
        max_groups=args.max_groups,
        mode=args.max_groups_mode,
        seed=args.seed,
    )
    groups = selected_catalog.groups
    group_names = list(groups)
    group_param_counts = selected_catalog.parameter_counts
    group_edm_proxy_param_counts = selected_catalog.edm_proxy_parameter_counts
    full_group_param_counts = full_catalog.parameter_counts
    total_parameters = count_parameters(model)
    analyzed_parameters = sum(group_param_counts.values())
    if args.grouping == "per_filter":
        full_analyzed_parameters = sum(full_group_param_counts.values())
        all_residual_block_parameters = sum(
            count
            for name, count in full_group_param_counts.items()
            if full_catalog.allocatable[name]
        )
        shared_parameters = total_parameters - full_analyzed_parameters
        unanalyzed_residual_block_parameters = all_residual_block_parameters - sum(
            count
            for name, count in group_param_counts.items()
            if selected_catalog.allocatable[name]
        )
        if args.require_full_filter_catalog and len(groups) != PER_FILTER_EXPECTED_GROUPS:
            raise ValueError(
                "--require-full-filter-catalog requires all 37,377 per-filter groups"
            )
    else:
        full_analyzed_parameters = sum(full_group_param_counts.values())
        all_residual_block_parameters = full_analyzed_parameters
        shared_parameters = total_parameters - all_residual_block_parameters
        unanalyzed_residual_block_parameters = all_residual_block_parameters - analyzed_parameters

    labels, analysis_axis = make_timestep_axis_metadata(
        evaluated_timesteps=corruption_dataset.level_indices,
        num_diffusion_steps=model.num_diffusion_steps,
        num_bins=args.num_bins,
        alpha_bar=schedule["Alpha_bar"],
    )
    axis_metadata = infer_bin_axis_metadata({"timestep_bin_labels": labels}, args.num_bins)
    implementation_fingerprint = {
        "diffwave_legacy_sha256": _file_sha256(
            REPO_ROOT / "pace" / "vendor" / "diffwave_legacy.py"
        ),
        "teacher_models_sha256": _file_sha256(
            REPO_ROOT / "pace" / "teacher_models.py"
        ),
        "audio_datasets_sha256": _file_sha256(
            REPO_ROOT / "pace" / "audio_datasets.py"
        ),
        "parameter_analysis_sha256": _file_sha256(
            REPO_ROOT / "pace" / "parameter_analysis.py"
        ),
        "evaluator_sha256": _file_sha256(Path(__file__).resolve()),
    }
    profile_fingerprint = {
        "format": "diffdist_diffwave_profile_fingerprint_v1",
        "teacher": {
            "checkpoint_sha256": resolved_teacher_spec.checkpoint_sha256,
            "checkpoint_size_bytes": resolved_teacher_spec.checkpoint_size_bytes,
            "upstream_commit": DIFFWAVE_SASHIMI_CHECKPOINTS_COMMIT,
            "legacy_forward_semantics": model.legacy_forward_semantics,
            "verification_report_sha256": verification["sha256"] if verification else None,
        },
        "dataset": {
            "population_fingerprint": dataset_population_fingerprint(audio_dataset),
            "metadata": dict(audio_dataset.metadata),
        },
        "runtime": {
            "identity": runtime_cache_identity(device),
            "dtype": str(dtype),
            "batch_size": args.batch_size,
            "num_bins": args.num_bins,
            "num_timestep_levels": args.num_timestep_levels,
            "samples_per_timestep": args.samples_per_timestep,
            "seed": args.seed,
            "pfi_seed": args.pfi_seed,
            "determinism": determinism,
        },
        "implementation": implementation_fingerprint,
        "diffusion_corruption": schedule_fingerprint,
        "groups": {
            "grouping": args.grouping,
            "full_count": len(full_catalog.groups),
            "selected_count": len(group_names),
            "selected_catalog_sha256": canonical_json_sha256(group_names),
            "selected_parameter_count": sum(group_param_counts.values()),
            "selected_edm_proxy_parameter_count": sum(
                group_edm_proxy_param_counts.values()
            ),
            "full_parameter_count": sum(full_group_param_counts.values()),
            "full_edm_proxy_parameter_count": sum(
                full_catalog.edm_proxy_parameter_counts.values()
            ),
            "full_catalog_sha256": canonical_json_sha256(
                [
                    {
                        "name": name,
                        "module_path": full_catalog.module_paths[name],
                        "filter_index": full_catalog.filter_indices[name],
                        "parameter_count": full_catalog.parameter_counts[name],
                        "edm_proxy_parameter_count": full_catalog.edm_proxy_parameter_counts[name],
                        "stage_key": full_catalog.stage_keys[name],
                    }
                    for name in full_catalog.groups
                ]
            ),
        },
        "ablation_protocol": ablation_protocol,
    }
    profile_fingerprint_sha256 = canonical_json_sha256(profile_fingerprint)
    analysis_basis = json.loads(json.dumps(profile_fingerprint))
    for key in (
        "selected_count",
        "selected_catalog_sha256",
        "selected_parameter_count",
        "selected_edm_proxy_parameter_count",
    ):
        analysis_basis["groups"].pop(key, None)
    analysis_basis_sha256 = canonical_json_sha256(analysis_basis)
    assert_distributed_hash_invariant(
        distributed_context,
        "profile_fingerprint",
        profile_fingerprint_sha256,
    )
    assert_distributed_hash_invariant(
        distributed_context,
        "analysis_basis",
        analysis_basis_sha256,
    )
    assert_distributed_hash_invariant(
        distributed_context,
        "teacher_checkpoint",
        str(
            resolved_teacher_spec.checkpoint_sha256
            or resolved_teacher_spec.expected_sha256
        ),
    )
    assert_distributed_canonical_invariant(
        distributed_context,
        "dataset_population",
        profile_fingerprint["dataset"],
    )
    assert_distributed_canonical_invariant(
        distributed_context,
        "implementation",
        implementation_fingerprint,
    )
    evaluator = DiffWaveUsageEvaluator(
        model,
        schedule,
        device=device,
        dtype=dtype,
        axis=analysis_axis,
        groups=groups,
        cache_fixed_corruptions=args.cache_fixed_corruptions,
    )
    cache = (
        FingerprintedBinnedCache(
            baseline_path=output_dir / "baseline_pfi.pt",
            baseline_format=BASELINE_CACHE_FORMAT,
            groups_path=output_dir
            / f"checkpoint_pfi_rank{distributed_context.rank}.pt",
            groups_format=ABLATION_CACHE_FORMAT,
            profile_fingerprint=profile_fingerprint,
            profile_fingerprint_sha256=profile_fingerprint_sha256,
            group_extra_metadata={
                "analysis_basis_sha256": analysis_basis_sha256,
            },
            group_shard_pattern=(
                output_dir / "checkpoint_pfi_rank*.pt"
                if args.grouping == "per_filter"
                or distributed_context.is_distributed
                else None
            ),
            require_rank_metadata=(
                args.grouping == "per_filter"
                or distributed_context.is_distributed
            ),
        )
        if args.ablation_mode == "pfi"
        else None
    )
    profile = run_binned_ablation_profile(
        evaluator,
        dataloader,
        num_bins=args.num_bins,
        ablation_mode=args.ablation_mode,
        ablation_random_seed=args.seed,
        progress_prefix="DiffWave",
        progress_factory=lambda batches, description: tqdm(
            batches,
            desc=description,
            leave=False,
        ),
        cache=cache,
        distributed_context=distributed_context,
        checkpoint_interval_groups=args.checkpoint_interval_groups,
    )
    distributed_record = {
        "world_size": distributed_context.world_size,
        "rank": distributed_context.rank,
        "local_rank": distributed_context.local_rank,
        "backend": distributed_context.backend,
        "device": str(distributed_context.device),
        "sharding": "globally_missing_groups_round_robin_by_canonical_index",
        "teacher_wrapped_in_ddp": False,
        "examples_sharded": False,
    }
    is_main_process = distributed_context.is_main_process
    cleanup_distributed_analysis(distributed_context)
    if not is_main_process:
        completed = len(profile.ablated_means)
        del profile, evaluator, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "rank": distributed_record["rank"],
            "world_size": distributed_record["world_size"],
            "completed_group_count": completed,
            "output_dir": str(output_dir),
        }
    baseline_mean = profile.baseline_mean
    baseline_stderr = profile.baseline_stderr
    baseline_count = profile.baseline_count
    ablated_means = profile.ablated_means
    if not torch.isfinite(baseline_mean).all() or bool(torch.any(baseline_count <= 0)):
        raise ValueError("DiffWave baseline contains non-finite values or empty timestep bins")

    signed_delta_stack, delta_stack = compute_signed_and_positive_deltas(
        ablated_means,
        baseline_mean,
        group_names,
    )
    compute_group_correlation = should_compute_group_correlation(
        args.group_correlation,
        len(group_names),
        args.max_group_correlation_groups,
    )
    metrics = compute_usage_metrics(
        delta_stack,
        baseline_mean,
        group_param_counts,
        group_names,
        compute_group_correlation=compute_group_correlation,
        require_positive_delta_mass=args.require_positive_delta_mass,
    )
    edm_proxy_metrics = (
        compute_usage_metrics(
            delta_stack,
            baseline_mean,
            group_edm_proxy_param_counts,
            group_names,
            compute_group_correlation=False,
            require_positive_delta_mass=False,
        )
        if args.grouping == "per_filter"
        else metrics
    )
    relative_delta_stack = metrics["relative_delta_stack"]
    if not torch.is_tensor(relative_delta_stack):
        raise TypeError("relative_delta_stack must be a tensor")
    n_eff_fraction = metrics["n_eff"] / float(
        len(full_catalog.groups) if args.grouping == "per_filter" else len(group_names)
    )
    signed_delta_positive_fraction = (signed_delta_stack > 0).to(torch.float64).mean(dim=0)
    signed_delta_negative_fraction = (signed_delta_stack < 0).to(torch.float64).mean(dim=0)
    signed_delta_zero_fraction = (signed_delta_stack == 0).to(torch.float64).mean(dim=0)
    per_filter_aggregates: dict[str, dict[str, Any]] = {}
    selected_top_filter_indices: list[int] = []
    if args.grouping == "per_filter":
        per_filter_aggregates = {
            "module": aggregate_filter_matrices(
                group_names=group_names,
                keys=selected_catalog.structural_keys,
                signed_delta_stack=signed_delta_stack,
                delta_stack=delta_stack,
                baseline_mean=baseline_mean,
                parameter_counts=group_param_counts,
            ),
            "stage": aggregate_filter_matrices(
                group_names=group_names,
                keys=selected_catalog.stage_keys,
                signed_delta_stack=signed_delta_stack,
                delta_stack=delta_stack,
                baseline_mean=baseline_mean,
                parameter_counts=group_param_counts,
            ),
        }
        selected_top_filter_indices = top_filter_indices(
            relative_delta_stack,
            args.top_filter_plot_count,
        )
    tensors_to_check = [
        baseline_mean,
        baseline_stderr,
        signed_delta_stack,
        delta_stack,
        metrics["relative_delta_stack"],
        metrics["row_normalized_relative_delta_stack"],
        metrics["C_noise_levels"],
        metrics["n_eff"],
        metrics["p_eff"],
        n_eff_fraction,
        signed_delta_positive_fraction,
        signed_delta_negative_fraction,
        signed_delta_zero_fraction,
    ]
    if any(not torch.isfinite(value).all() for value in tensors_to_check if torch.is_tensor(value)):
        raise ValueError("DiffWave analysis produced non-finite derived statistics")

    invocation_argv = [sys.argv[0], *(argv if argv is not None else sys.argv[1:])]
    results: dict[str, Any] = {
        "format": ANALYSIS_FORMAT,
        "config": vars(args),
        "invocation": {
            "argv": invocation_argv,
            "command": format_invocation_command(invocation_argv),
        },
        "dataset_info": dict(audio_dataset.metadata),
        "teacher": resolved_teacher_spec.to_dict(),
        "teacher_verification": verification,
        "runtime_environment": runtime_environment(device, determinism),
        "distributed": distributed_record,
        "model_info": {
            **model_metadata,
            "total_parameter_count": total_parameters,
            "analyzed_group_parameter_count": analyzed_parameters,
            "all_residual_block_parameter_count": all_residual_block_parameters,
            "shared_parameter_count": shared_parameters,
            "unanalyzed_residual_block_parameter_count": unanalyzed_residual_block_parameters,
            "allocation_unassigned_parameter_count": total_parameters - analyzed_parameters,
            "grouping": args.grouping,
            "full_analyzed_group_parameter_count": full_analyzed_parameters,
            "full_edm_proxy_group_parameter_count": sum(
                full_catalog.edm_proxy_parameter_counts.values()
            ),
            "allocatable_residual_filter_parameter_count": all_residual_block_parameters,
            "fixed_unassigned_parameter_count": total_parameters - all_residual_block_parameters,
            # Compatibility alias used by early audio-profile artifacts.  It
            # includes unselected residual blocks as well as truly shared
            # parameters when max_groups limits the smoke profile.
            "shared_unassigned_parameter_count": total_parameters - analyzed_parameters,
        },
        "diffusion_axis": analysis_axis.to_dict(),
        "diffusion_corruption": schedule_fingerprint,
        "timestep_values": [int(value) for value in corruption_dataset.level_indices],
        "timestep_bin_members": [
            list(members) for members in analysis_axis.bin_members
        ],
        "ablation_info": {
            "mode": args.ablation_mode,
            "replacement": ablation_protocol["replacement"],
        },
        "ablation_protocol": ablation_protocol,
        "pfi_plan": pfi_plan_record,
        "profile_fingerprint": profile_fingerprint,
        "profile_fingerprint_sha256": profile_fingerprint_sha256,
        "analysis_basis": analysis_basis,
        "analysis_basis_sha256": analysis_basis_sha256,
        "group_names": group_names,
        "group_param_counts": group_param_counts,
        "group_edm_proxy_param_counts": group_edm_proxy_param_counts,
        "group_module_paths": selected_catalog.module_paths,
        "group_structural_keys": selected_catalog.structural_keys,
        "group_stage_keys": selected_catalog.stage_keys,
        "group_allocation_structural_keys": selected_catalog.allocation_structural_keys,
        "group_allocatable": selected_catalog.allocatable,
        "group_filter_indices": selected_catalog.filter_indices,
        "full_group_count": len(full_catalog.groups),
        "baseline_mean": baseline_mean.tolist(),
        "baseline_stderr": baseline_stderr.tolist(),
        "raw_baseline_mean": baseline_mean.tolist(),
        "raw_baseline_stderr": baseline_stderr.tolist(),
        "baseline_count": baseline_count.tolist(),
        "delta_stack": delta_stack.tolist(),
        "signed_delta_stack": signed_delta_stack.tolist(),
        "relative_delta_stack": metrics["relative_delta_stack"].tolist(),
        "row_normalized_relative_delta_stack": metrics["row_normalized_relative_delta_stack"].tolist(),
        "positive_delta_mass": metrics["positive_delta_mass"].tolist(),
        "C_groups": tensor_or_none_to_list(metrics["C_groups"]),
        "C_timesteps": metrics["C_noise_levels"].tolist(),
        "timestep_bin_labels": labels,
        "weights": metrics["weights"].tolist(),
        "n_eff": metrics["n_eff"].tolist(),
        "n_eff_fraction": n_eff_fraction.tolist(),
        "p_eff": metrics["p_eff"].tolist(),
        "p_eff_edm_proxy": edm_proxy_metrics["p_eff"].tolist(),
        "signed_delta_positive_fraction": signed_delta_positive_fraction.tolist(),
        "signed_delta_negative_fraction": signed_delta_negative_fraction.tolist(),
        "signed_delta_zero_fraction": signed_delta_zero_fraction.tolist(),
        "per_filter_aggregates": {
            key: tensor_aggregate_to_json(value)
            for key, value in per_filter_aggregates.items()
        },
        "top_filter_plot": {
            "selection": "sum_positive_relative_delta_descending_then_group_index",
            "count": len(selected_top_filter_indices),
            "indices": selected_top_filter_indices,
            "names": [group_names[index] for index in selected_top_filter_indices],
        },
        "fixed_corruption_cache": evaluator.fixed_corruption_cache_record(),
        "group_correlation": {
            "mode": args.group_correlation,
            "computed": compute_group_correlation,
            "max_groups": args.max_group_correlation_groups,
            "num_groups": len(group_names),
        },
    }
    if args.grouping == "per_filter":
        plot_indices = torch.tensor(selected_top_filter_indices, dtype=torch.long)
        plotted_names = [group_names[index] for index in selected_top_filter_indices]
        plotted_delta_stack = delta_stack.index_select(0, plot_indices)
        plotted_relative_delta_stack = relative_delta_stack.index_select(0, plot_indices)
        plotted_row_normalized = metrics["row_normalized_relative_delta_stack"].index_select(
            0, plot_indices
        )
    else:
        plotted_names = group_names
        plotted_delta_stack = delta_stack
        plotted_relative_delta_stack = relative_delta_stack
        plotted_row_normalized = metrics["row_normalized_relative_delta_stack"]

    metrics_path = output_dir / "metrics.pt"
    atomic_torch_save(
        {
            "format": "diffdist_diffwave_parameter_metrics_v1",
            "profile_fingerprint_sha256": profile_fingerprint_sha256,
            "group_names": group_names,
            "baseline_mean": baseline_mean,
            "baseline_stderr": baseline_stderr,
            "baseline_count": baseline_count,
            "signed_delta_stack": signed_delta_stack,
            "delta_stack": delta_stack,
            "relative_delta_stack": relative_delta_stack,
            "row_normalized_relative_delta_stack": metrics[
                "row_normalized_relative_delta_stack"
            ],
            "weights": metrics["weights"],
            "n_eff": metrics["n_eff"],
            "n_eff_fraction": n_eff_fraction,
            "p_eff": metrics["p_eff"],
            "positive_delta_mass": metrics["positive_delta_mass"],
            "signed_delta_positive_fraction": signed_delta_positive_fraction,
            "signed_delta_negative_fraction": signed_delta_negative_fraction,
            "signed_delta_zero_fraction": signed_delta_zero_fraction,
            "C_timesteps": metrics["C_noise_levels"],
            "per_filter_aggregates": per_filter_aggregates,
        },
        metrics_path,
    )

    save_results_and_plots(
        output_dir=output_dir,
        results=results,
        baseline_mean=baseline_mean,
        baseline_stderr=baseline_stderr,
        raw_baseline_mean=baseline_mean,
        raw_baseline_stderr=baseline_stderr,
        delta_stack=plotted_delta_stack,
        relative_delta_stack=plotted_relative_delta_stack,
        row_normalized_relative_delta_stack=plotted_row_normalized,
        C_groups=metrics["C_groups"],
        C_levels=metrics["C_noise_levels"],
        p_eff=metrics["p_eff"],
        n_eff=metrics["n_eff"],
        names=plotted_names,
        axis_metadata=axis_metadata,
    )
    if args.grouping == "per_filter":
        for values, filename, label, ylabel in (
            (
                n_eff_fraction,
                "effective_group_fraction.png",
                "Effective fraction of all DiffWave filters",
                "N_eff / 37,377",
            ),
            (
                metrics["positive_delta_mass"],
                "positive_delta_mass.png",
                "Total positive individual-filter PFI mass",
                "Positive delta mass",
            ),
            (
                signed_delta_positive_fraction,
                "positive_filter_fraction.png",
                "Fraction of filters with positive signed PFI delta",
                "Positive filter fraction",
            ),
            (
                signed_delta_negative_fraction,
                "negative_filter_fraction.png",
                "Fraction of filters with negative signed PFI delta",
                "Negative filter fraction",
            ),
        ):
            plot_binned_series(
                values,
                output_dir / filename,
                label=label,
                ylabel=ylabel,
                axis_title=str(axis_metadata["axis_title"]),
                bin_labels=labels,
            )
    if per_filter_aggregates:
        save_per_filter_aggregate_plots(
            output_dir,
            per_filter_aggregates,
            timestep_labels=labels,
        )

    if args.run_postprocessing:
        run_postprocessing(
            output_dir,
            grouping=args.grouping,
            num_timestep_groups=args.num_timestep_groups,
            score_reduction=args.score_reduction,
            allocation_variants=args.allocation_variants,
        )
    runtime_seconds = time.perf_counter() - started
    results_path = output_dir / "results.json"
    artifact_sha256 = output_artifact_sha256_manifest(output_dir)
    success = {
        "format": "diffdist_analysis_success_v1",
        "profile_fingerprint_sha256": profile_fingerprint_sha256,
        "analysis_basis_sha256": analysis_basis_sha256,
        "results_path": str(results_path),
        "results_sha256": _file_sha256(results_path),
        "metrics_path": str(metrics_path),
        "metrics_sha256": _file_sha256(metrics_path),
        "artifact_sha256": artifact_sha256,
        "artifact_count": len(artifact_sha256),
        "grouping": args.grouping,
        "completed_group_count": len(group_names),
        "full_group_count": len(full_catalog.groups),
        "num_bins": args.num_bins,
        "baseline_count": baseline_count.tolist(),
        "runtime_seconds": runtime_seconds,
        "world_size": distributed_record["world_size"],
    }
    atomic_save_json(output_dir / "_SUCCESS.json", success)
    print(json.dumps({
        "results": str(results_path),
        "num_groups": len(group_names),
        "num_bins": args.num_bins,
        "runtime_seconds": runtime_seconds,
    }, indent=2))
    return results


if __name__ == "__main__":
    main()
