#!/usr/bin/env python3
"""Verify the SC09 DiffWave teacher against the original checkpoint branch.

The script deliberately imports and executes the pinned upstream source rather
than re-implementing the unsafe reference.  It compares epsilon predictions,
a shared-noise reverse trajectory, and the autograd-safe vendored formulation.
Optionally it also runs the complete 200-step upstream sampler and writes a
one-second PCM WAV for manual validation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import types
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pace.teacher_models import (
    DIFFWAVE_SASHIMI_LEGACY_STATE_DICT,
    TeacherSpec,
    load_teacher_network,
    resolve_teacher_spec,
)
from pace.vendor.diffwave_legacy import (
    DIFFWAVE_SASHIMI_CHECKPOINT_PATH,
    DIFFWAVE_SASHIMI_CHECKPOINT_SHA256,
    DIFFWAVE_SASHIMI_CHECKPOINT_SIZE_BYTES,
    DIFFWAVE_SASHIMI_CHECKPOINT_SOURCE,
    DIFFWAVE_SASHIMI_CHECKPOINTS_COMMIT,
    DIFFWAVE_SASHIMI_LEGACY_SEMANTICS,
    DIFFWAVE_SASHIMI_MODEL_PATH,
    LegacyCompatibleDiffWave,
    diffwave_diffusion_hyperparameters,
)


RESULT_FORMAT = "diffdist_diffwave_teacher_reproduction_v1"
UPSTREAM_UTILS_PATH = "models/utils.py"
RELATIVE_L2_TOLERANCE = 1e-6
REFERENCE_GENERATION_ARGUMENTS = (
    "generate.py",
    "experiment=sc09",
    "model=wavenet",
    "generate.ckpt_iter=1000000",
    "generate.n_samples=1",
    "generate.batch_size=1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream-repo",
        type=Path,
        required=True,
        help="Clean albertfgu/diffwave-sashimi checkout pinned to the checkpoints commit.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Downloaded 1M checkpoint (not the Git LFS pointer).",
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON verification report.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prediction-length", type=int, default=16_000)
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=[0, 1, 25, 50, 100, 150, 198, 199],
    )
    parser.add_argument("--trajectory-steps", type=int, default=10)
    parser.add_argument("--trajectory-length", type=int, default=16_000)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument(
        "--audio-output",
        type=Path,
        required=True,
        help="Run all 200 upstream reverse steps and write a 16 kHz PCM16 WAV.",
    )
    parser.add_argument(
        "--reference-audio",
        type=Path,
        required=True,
        help="Float32 WAV produced by the untouched pinned upstream generate.py CLI.",
    )
    parser.add_argument(
        "--reference-python",
        type=Path,
        required=True,
        help=(
            "Python executable used for the untouched upstream generate.py run; "
            "its environment and complete pip freeze are recorded in the report."
        ),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_random_normal(
    shape: tuple[int, ...] | torch.Size,
    *,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Match upstream ``torch.normal(...).cuda()`` RNG placement."""

    return torch.normal(
        0.0,
        1.0,
        size=tuple(shape),
        generator=generator,
        device="cpu",
    ).to(device)


def _tensor_sequence_sha256(tensors: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _git_output(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *arguments],
        text=True,
    ).strip()


def _portable_pip_freeze(lines: list[str]) -> list[str]:
    """Drop editable and VCS requirement lines from ``pip freeze`` output.

    Those lines record local checkout paths and private remote URLs; the
    pinned package versions that the report exists to capture are kept.
    """

    return [
        line
        for line in lines
        if line.strip()
        and not line.lstrip().startswith(("-e ", "--editable"))
        and "git+" not in line
        and " @ file:" not in line
    ]


def _verify_upstream_checkout(repository: Path) -> dict[str, Any]:
    repository = repository.expanduser().resolve()
    if not (repository / ".git").exists():
        raise ValueError(f"Not a Git checkout: {repository}")
    commit = _git_output(repository, "rev-parse", "HEAD")
    if commit != DIFFWAVE_SASHIMI_CHECKPOINTS_COMMIT:
        raise ValueError(
            "Wrong diffwave-sashimi commit: expected "
            f"{DIFFWAVE_SASHIMI_CHECKPOINTS_COMMIT}, got {commit}"
        )
    for relative_path in (DIFFWAVE_SASHIMI_MODEL_PATH, UPSTREAM_UTILS_PATH):
        working_copy = (repository / relative_path).read_bytes()
        committed_copy = subprocess.check_output(
            ["git", "-C", str(repository), "show", f"{commit}:{relative_path}"]
        )
        if working_copy != committed_copy:
            raise ValueError(f"Upstream source differs from {commit}:{relative_path}")
    return {
        "repository": str(repository),
        "remote": _git_output(repository, "remote", "get-url", "origin"),
        "branch": _git_output(repository, "branch", "--show-current") or None,
        "expected_branch": "checkpoints",
        "commit": commit,
        "model_path": DIFFWAVE_SASHIMI_MODEL_PATH,
        "utils_path": UPSTREAM_UTILS_PATH,
    }


def _load_upstream_wavenet_class(repository: Path) -> type[torch.nn.Module]:
    """Import upstream files without importing its optional SaShiMi stack."""

    model_dir = repository / "models"
    saved_modules = {
        name: sys.modules.get(name)
        for name in ("models", "models.utils", "models.wavenet")
    }
    try:
        package = types.ModuleType("models")
        package.__path__ = [str(model_dir)]  # type: ignore[attr-defined]
        sys.modules["models"] = package
        for name in ("utils", "wavenet"):
            module_name = f"models.{name}"
            spec = importlib.util.spec_from_file_location(module_name, model_dir / f"{name}.py")
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not import upstream {module_name}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        upstream_class = sys.modules["models.wavenet"].WaveNet
    finally:
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return upstream_class


def _model_kwargs() -> dict[str, Any]:
    return {
        "in_channels": 1,
        "res_channels": 256,
        "skip_channels": 256,
        "out_channels": 1,
        "num_res_layers": 36,
        "dilation_cycle": 12,
        "diffusion_step_embed_dim_in": 128,
        "diffusion_step_embed_dim_mid": 512,
        "diffusion_step_embed_dim_out": 512,
        "unconditional": True,
    }


def _load_checkpoint_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("Checkpoint does not contain checkpoint['model_state_dict']")
    return payload


def _error_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    difference = (reference - candidate).to(torch.float64)
    absolute = difference.abs()
    denominator = reference.detach().to(torch.float64).abs().clamp_min(1e-12)
    reference_l2 = torch.linalg.vector_norm(reference.detach().to(torch.float64))
    relative_l2 = torch.linalg.vector_norm(difference) / reference_l2.clamp_min(1e-24)
    try:
        torch.testing.assert_close(candidate, reference, atol=atol, rtol=rtol)
        elementwise_close = True
    except AssertionError:
        elementwise_close = False
    return {
        "bitwise_equal": bool(torch.equal(reference, candidate)),
        "elementwise_close": elementwise_close,
        "max_abs": float(absolute.max().item()),
        "mean_abs": float(absolute.mean().item()),
        "rmse": float(difference.square().mean().sqrt().item()),
        "max_relative": float((absolute / denominator).max().item()),
        "relative_l2": float(relative_l2.item()),
        "reference_max_abs": float(reference.detach().abs().max().item()),
    }


def _prediction_parity(
    upstream: torch.nn.Module,
    safe: LegacyCompatibleDiffWave,
    *,
    device: torch.device,
    seed: int,
    batch_size: int,
    length: int,
    timesteps: list[int],
    atol: float,
    rtol: float,
) -> list[dict[str, Any]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x_t = _cpu_random_normal(
        (batch_size, 1, length),
        generator=generator,
        device=device,
    )
    results: list[dict[str, Any]] = []
    with torch.no_grad():
        for timestep in timesteps:
            t = torch.full((batch_size, 1), float(timestep), device=device)
            reference = upstream((x_t.clone(), t.clone()))
            candidate = safe(x_t.clone(), t.clone())
            results.append(
                {
                    "timestep": timestep,
                    **_error_metrics(reference, candidate, atol=atol, rtol=rtol),
                }
            )
    return results


def _run_trajectory(
    model: torch.nn.Module,
    initial_noise: torch.Tensor,
    step_noises: list[torch.Tensor],
    *,
    schedule: dict[str, torch.Tensor | int],
    timesteps: list[int],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    alpha = schedule["Alpha"]
    alpha_bar = schedule["Alpha_bar"]
    sigma = schedule["Sigma"]
    assert torch.is_tensor(alpha) and torch.is_tensor(alpha_bar) and torch.is_tensor(sigma)
    x = initial_noise.clone()
    epsilons: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    with torch.no_grad():
        for timestep, noise in zip(timesteps, step_noises, strict=True):
            t = torch.full((x.shape[0], 1), float(timestep), device=x.device)
            epsilon = model((x, t))
            epsilons.append(epsilon.clone())
            x = (
                x
                - (1 - alpha[timestep])
                / torch.sqrt(1 - alpha_bar[timestep])
                * epsilon
            ) / torch.sqrt(alpha[timestep])
            if timestep > 0:
                x = x + sigma[timestep] * noise
            states.append(x.clone())
    return epsilons, states


def _trajectory_parity(
    upstream: torch.nn.Module,
    safe: LegacyCompatibleDiffWave,
    *,
    device: torch.device,
    seed: int,
    length: int,
    num_steps: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if not 1 <= num_steps <= 200:
        raise ValueError("trajectory_steps must be in [1, 200]")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial_noise = _cpu_random_normal(
        (1, 1, length),
        generator=generator,
        device=device,
    )
    timesteps = list(range(199, 199 - num_steps, -1))
    step_noises = [
        _cpu_random_normal(initial_noise.shape, generator=generator, device=device)
        for _ in timesteps
    ]
    schedule = diffwave_diffusion_hyperparameters(device=device)
    reference_epsilons, reference_states = _run_trajectory(
        upstream,
        initial_noise,
        step_noises,
        schedule=schedule,
        timesteps=timesteps,
    )
    candidate_epsilons, candidate_states = _run_trajectory(
        safe,
        initial_noise,
        step_noises,
        schedule=schedule,
        timesteps=timesteps,
    )
    per_step = [
        {
            "timestep": timestep,
            "epsilon": _error_metrics(
                reference_epsilon,
                candidate_epsilon,
                atol=atol,
                rtol=rtol,
            ),
            "state": _error_metrics(
                reference_state,
                candidate_state,
                atol=atol,
                rtol=rtol,
            ),
        }
        for (
            timestep,
            reference_epsilon,
            candidate_epsilon,
            reference_state,
            candidate_state,
        ) in zip(
            timesteps,
            reference_epsilons,
            candidate_epsilons,
            reference_states,
            candidate_states,
            strict=True,
        )
    ]
    return {
        "initial_noise_sha256": hashlib.sha256(
            initial_noise.detach().cpu().numpy().tobytes()
        ).hexdigest(),
        "step_noises_sha256": _tensor_sequence_sha256(step_noises),
        "timesteps": timesteps,
        "per_step": per_step,
        "final_epsilon": {"timestep": timesteps[-1], **per_step[-1]["epsilon"]},
        "final_state": {"timestep": timesteps[-1], **per_step[-1]["state"]},
    }


def _autograd_check(
    model: LegacyCompatibleDiffWave,
    device: torch.device,
    seed: int,
    *,
    length: int = 257,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = _cpu_random_normal((1, 1, length), generator=generator, device=device)
    original = x.detach().clone()
    x.requires_grad_(True)
    t = torch.tensor([[100.0]], device=device)
    loss = model(x, t).square().mean()
    loss.backward()
    if x.grad is None:
        raise RuntimeError("Autograd did not populate an input gradient")
    return {
        "loss": float(loss.detach().item()),
        "input_gradient_finite": bool(torch.isfinite(x.grad).all().item()),
        "input_gradient_max_abs": float(x.grad.detach().abs().max().item()),
        "input_unchanged_by_residual_blocks": bool(torch.equal(x.detach(), original)),
        "model": "strict_loaded_1m_checkpoint",
        "input_shape": list(x.shape),
        "timestep": 100,
    }


def _write_pcm16(path: Path, audio: torch.Tensor, sample_rate: int = 16_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (
        audio.detach()
        .flatten()
        .clamp(-1, 1)
        .mul(32_767)
        .round()
        .to(torch.int16)
        .cpu()
        .numpy()
        .tobytes()
    )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)


def _validate_reference_audio(path: Path) -> dict[str, Any]:
    """Validate the artifact written by untouched upstream ``generate.py``."""

    from scipy.io.wavfile import read as wavread

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Untouched upstream generation WAV is missing: {resolved}")
    sample_rate, samples = wavread(resolved)
    if not isinstance(samples, np.ndarray):
        raise ValueError(f"Unexpected WAV decoder result for {resolved}")
    if samples.dtype != np.float32:
        raise ValueError(
            "Untouched upstream generate.py must produce float32 WAV samples; "
            f"got {samples.dtype}: {resolved}"
        )
    if samples.ndim != 1:
        raise ValueError(f"Reference generation must be mono, got shape {samples.shape}")
    if int(sample_rate) != 16_000 or int(samples.shape[0]) != 16_000:
        raise ValueError(
            "Reference generation must contain exactly 16,000 samples at 16 kHz; "
            f"got {samples.shape[0]} at {sample_rate} Hz"
        )
    finite = bool(np.isfinite(samples).all())
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    peak = float(np.max(np.abs(samples)))
    standard_deviation = float(np.std(samples, dtype=np.float64))
    if not finite or rms <= 1e-6 or standard_deviation <= 1e-7:
        raise ValueError(
            "Untouched upstream generation is non-finite, constant, or effectively silent: "
            f"finite={finite}, rms={rms}, std={standard_deviation}"
        )
    return {
        "path": str(resolved),
        "sample_rate": int(sample_rate),
        "num_samples": int(samples.shape[0]),
        "num_channels": 1,
        "dtype": str(samples.dtype),
        "finite": finite,
        "rms": rms,
        "standard_deviation": standard_deviation,
        "peak_abs": peak,
        "wav_size_bytes": resolved.stat().st_size,
        "wav_sha256": _sha256(resolved),
    }


def _capture_reference_environment(reference_python: Path) -> dict[str, Any]:
    """Inspect the exact interpreter used by the untouched upstream CLI."""

    invocation_interpreter = Path(
        os.path.abspath(os.path.expanduser(str(reference_python)))
    )
    if not invocation_interpreter.is_file() or not os.access(invocation_interpreter, os.X_OK):
        raise ValueError(
            "Reference Python must be an existing executable file: "
            f"{invocation_interpreter}"
        )
    probe = """
import json
import pathlib
import platform
import sys
import torch
import torchaudio

print(json.dumps({
    "interpreter": sys.executable,
    "resolved_interpreter": str(pathlib.Path(sys.executable).resolve()),
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torchaudio": torchaudio.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_runtime": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
}))
"""
    try:
        payload = json.loads(
            subprocess.check_output(
                [str(invocation_interpreter), "-c", probe],
                text=True,
            )
        )
        pip_freeze = _portable_pip_freeze(subprocess.check_output(
            [str(invocation_interpreter), "-m", "pip", "freeze"],
            text=True,
        ).splitlines())
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Could not inspect reference interpreter {invocation_interpreter}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("Reference interpreter probe did not return a JSON object")
    reported_interpreter = Path(str(payload.get("interpreter", "")))
    if reported_interpreter != invocation_interpreter:
        raise ValueError(
            "Reference interpreter reported an unexpected sys.executable: "
            f"invoked {invocation_interpreter}, reported {reported_interpreter}"
        )
    if not pip_freeze:
        raise ValueError("Reference interpreter returned an empty pip freeze")
    freeze_text = "\n".join(pip_freeze) + "\n"
    return {
        **payload,
        "pip_freeze_command": f"{shlex.quote(str(invocation_interpreter))} -m pip freeze",
        "pip_freeze": pip_freeze,
        "pip_freeze_sha256": hashlib.sha256(freeze_text.encode("utf-8")).hexdigest(),
    }


def _reference_generation_provenance(
    repository: Path,
    environment: dict[str, Any],
) -> dict[str, Any]:
    interpreter = str(environment["interpreter"])
    argv = [interpreter, *REFERENCE_GENERATION_ARGUMENTS]
    command = (
        f"cd {shlex.quote(str(repository))}\n"
        f"CUDA_VISIBLE_DEVICES=0 {' '.join(shlex.quote(value) for value in argv)}"
    )
    return {
        "working_directory": str(repository),
        "cuda_visible_devices": "0",
        "argv": argv,
        "command": command,
        "environment": environment,
    }


def _generate_upstream_audio(
    upstream: torch.nn.Module,
    *,
    device: torch.device,
    seed: int,
    output: Path,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = _cpu_random_normal((1, 1, 16_000), generator=generator, device=device)
    initial_noise_sha256 = hashlib.sha256(
        x.detach().cpu().numpy().tobytes()
    ).hexdigest()
    # The upstream sampler draws one CPU tensor at every nonzero timestep and
    # then copies it to CUDA.  Materialize the complete stochastic input before
    # either implementation runs so model execution cannot perturb its RNG.
    step_noises = [
        _cpu_random_normal(x.shape, generator=generator, device=device)
        for _ in range(199)
    ]
    schedule = diffwave_diffusion_hyperparameters(device=device)
    alpha = schedule["Alpha"]
    alpha_bar = schedule["Alpha_bar"]
    sigma = schedule["Sigma"]
    assert torch.is_tensor(alpha) and torch.is_tensor(alpha_bar) and torch.is_tensor(sigma)
    with torch.no_grad():
        for timestep in range(199, -1, -1):
            t = torch.full((1, 1), float(timestep), device=device)
            epsilon = upstream((x, t))
            x = (
                x
                - (1 - alpha[timestep])
                / torch.sqrt(1 - alpha_bar[timestep])
                * epsilon
            ) / torch.sqrt(alpha[timestep])
            if timestep > 0:
                x = x + sigma[timestep] * step_noises[199 - timestep]
    _write_pcm16(output, x[0, 0])
    finite = bool(torch.isfinite(x).all().item())
    rms = float(x.square().mean().sqrt().item())
    peak = float(x.abs().max().item())
    clipped_fraction = float((x.abs() > 1).to(torch.float32).mean().item())
    return {
        "path": str(output.resolve()),
        "sample_rate": 16_000,
        "num_samples": 16_000,
        "seed": seed,
        "initial_noise_sha256": initial_noise_sha256,
        "step_noises_sha256": _tensor_sequence_sha256(step_noises),
        "finite": finite,
        "rms": rms,
        "peak_abs": peak,
        "fraction_outside_pcm_range": clipped_fraction,
        "wav_size_bytes": output.stat().st_size,
        "wav_sha256": _sha256(output),
    }


def _local_repository_metadata() -> dict[str, Any]:
    """Record the commit of this checkout and whether it has local changes.

    The checkout path, the remote URL and the list of changed files are not
    recorded, so a report can be shared without local path information.
    """
    repository = Path(__file__).resolve().parents[2]
    try:
        commit = _git_output(repository, "rev-parse", "HEAD")
        status = _git_output(repository, "status", "--porcelain")
    except (OSError, subprocess.SubprocessError):
        return {"available": False}
    return {
        "available": True,
        "commit": commit,
        "dirty": bool(status),
    }


def _environment(device: torch.device) -> dict[str, Any]:
    driver = None
    if device.type == "cuda":
        try:
            driver = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                    "--id=0",
                ],
                text=True,
            ).splitlines()[0]
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        pip_freeze = _portable_pip_freeze(subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
        ).splitlines())
    except (OSError, subprocess.SubprocessError):
        pip_freeze = []
    try:
        environment_ffmpeg = Path(sys.executable).with_name("ffmpeg")
        ffmpeg_executable = str(environment_ffmpeg) if environment_ffmpeg.is_file() else "ffmpeg"
        ffmpeg_version = subprocess.check_output(
            [ffmpeg_executable, "-version"],
            text=True,
        ).splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        ffmpeg_version = None
    package_versions: dict[str, str | None] = {}
    for package in ("torch", "torchaudio", "torchcodec", "numpy", "scipy"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": importlib.metadata.version("numpy"),
        "packages": package_versions,
        "ffmpeg": ffmpeg_version,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "driver": driver,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "pip_freeze": pip_freeze,
    }


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.prediction_length <= 0 or args.trajectory_length <= 0:
        raise ValueError("Batch size and waveform lengths must be positive")
    if any(timestep < 0 or timestep >= 200 for timestep in args.timesteps):
        raise ValueError("All prediction timesteps must be in [0, 199]")
    required_timesteps = [0, 1, 25, 50, 100, 150, 198, 199]
    if (
        args.batch_size != 1
        or args.prediction_length != 16_000
        or list(args.timesteps) != required_timesteps
        or args.trajectory_steps != 10
        or args.trajectory_length != 16_000
    ):
        raise ValueError(
            "The reproduction gate requires batch=1, length=16000, prediction timesteps "
            f"{required_timesteps}, and the ten-step trajectory t=199..190"
        )
    if (
        not math.isfinite(args.atol)
        or not math.isfinite(args.rtol)
        or args.atol < 0
        or args.rtol < 0
        or args.atol > 1e-6
        or args.rtol > 1e-6
    ):
        raise ValueError("The reproduction gate requires finite atol and rtol no larger than 1e-6")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("The original checkpoints-branch reference requires a CUDA device")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    upstream_repo = args.upstream_repo.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    checkout_metadata = _verify_upstream_checkout(upstream_repo)
    reference_environment = _capture_reference_environment(args.reference_python)
    reference_generation = _reference_generation_provenance(
        upstream_repo,
        reference_environment,
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_metadata = {
        "path": str(checkpoint),
        "source": DIFFWAVE_SASHIMI_CHECKPOINT_SOURCE,
        "upstream_relative_path": DIFFWAVE_SASHIMI_CHECKPOINT_PATH,
        "size_bytes": checkpoint.stat().st_size,
        "sha256": _sha256(checkpoint),
        "expected_size_bytes": DIFFWAVE_SASHIMI_CHECKPOINT_SIZE_BYTES,
        "expected_sha256": DIFFWAVE_SASHIMI_CHECKPOINT_SHA256,
    }
    if checkpoint_metadata["size_bytes"] != DIFFWAVE_SASHIMI_CHECKPOINT_SIZE_BYTES:
        raise ValueError(f"Checkpoint size mismatch: {checkpoint_metadata}")
    if checkpoint_metadata["sha256"] != DIFFWAVE_SASHIMI_CHECKPOINT_SHA256:
        raise ValueError(f"Checkpoint SHA-256 mismatch: {checkpoint_metadata}")

    upstream_class = _load_upstream_wavenet_class(upstream_repo)
    payload = _load_checkpoint_payload(checkpoint)
    upstream = upstream_class(**_model_kwargs())
    upstream.load_state_dict(payload["model_state_dict"], strict=True)
    upstream = upstream.eval().requires_grad_(False).to(device=device, dtype=torch.float32)

    spec = resolve_teacher_spec(
        checkpoint,
        network_format=DIFFWAVE_SASHIMI_LEGACY_STATE_DICT,
        preset="sc09_diffwave_legacy_1m",
    )
    safe = load_teacher_network(spec, device=device, dtype=torch.float32)
    if not isinstance(safe, LegacyCompatibleDiffWave):
        raise TypeError(f"Unexpected safe model type {type(safe).__name__}")

    predictions = _prediction_parity(
        upstream,
        safe,
        device=device,
        seed=args.seed,
        batch_size=args.batch_size,
        length=args.prediction_length,
        timesteps=list(args.timesteps),
        atol=args.atol,
        rtol=args.rtol,
    )
    trajectory = _trajectory_parity(
        upstream,
        safe,
        device=device,
        seed=args.seed + 1,
        length=args.trajectory_length,
        num_steps=args.trajectory_steps,
        atol=args.atol,
        rtol=args.rtol,
    )
    autograd = _autograd_check(safe, device, args.seed + 2)
    reference_audio = _validate_reference_audio(args.reference_audio)
    audio = _generate_upstream_audio(
        upstream,
        device=device,
        seed=args.seed + 3,
        output=args.audio_output.expanduser().resolve(),
    )

    trajectory_errors = [
        metrics
        for step in trajectory["per_step"]
        for metrics in (step["epsilon"], step["state"])
    ]
    all_errors = predictions + trajectory_errors
    parity_passed = all(
        item["elementwise_close"]
        and item["relative_l2"] <= RELATIVE_L2_TOLERANCE
        for item in all_errors
    )
    autograd_passed = (
        autograd["input_gradient_finite"]
        and autograd["input_gradient_max_abs"] > 0
    )
    generated_audio_passed = (
        audio["finite"]
        and audio["rms"] > 1e-6
        and audio["num_samples"] == 16_000
        and audio["sample_rate"] == 16_000
        and audio["wav_size_bytes"] > 44
    )
    reference_audio_passed = (
        reference_audio["finite"]
        and reference_audio["dtype"] == "float32"
        and reference_audio["num_channels"] == 1
        and reference_audio["num_samples"] == 16_000
        and reference_audio["sample_rate"] == 16_000
        and reference_audio["rms"] > 1e-6
        and reference_audio["standard_deviation"] > 1e-7
    )
    audio_passed = generated_audio_passed and reference_audio_passed
    passed = parity_passed and autograd_passed and audio_passed

    report = {
        "format": RESULT_FORMAT,
        "passed": passed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(shlex.quote(value) for value in sys.argv),
        "working_directory": os.getcwd(),
        "upstream": checkout_metadata,
        "checkpoint": checkpoint_metadata,
        "implementation": {
            "class": f"{safe.__class__.__module__}.{safe.__class__.__name__}",
            "legacy_forward_semantics": DIFFWAVE_SASHIMI_LEGACY_SEMANTICS,
            "checkpoint_format": DIFFWAVE_SASHIMI_LEGACY_STATE_DICT,
            "strict_state_dict_load": True,
            "checkpoint_container_key": "model_state_dict",
            "source_path": "pace/vendor/diffwave_legacy.py",
            "source_sha256": _sha256(
                Path(__file__).resolve().parents[2]
                / "pace"
                / "vendor"
                / "diffwave_legacy.py"
            ),
            "teacher_loader_path": "pace/teacher_models.py",
            "teacher_loader_sha256": _sha256(
                Path(__file__).resolve().parents[2]
                / "pace"
                / "teacher_models.py"
            ),
        },
        "configuration": {
            "seed": args.seed,
            "seeds": {
                "prediction": args.seed,
                "trajectory": args.seed + 1,
                "autograd": args.seed + 2,
                "audio": args.seed + 3,
            },
            "atol": args.atol,
            "rtol": args.rtol,
            "relative_l2_tolerance": RELATIVE_L2_TOLERANCE,
            "batch_size": args.batch_size,
            "prediction_length": args.prediction_length,
            "prediction_timesteps": list(args.timesteps),
            "trajectory_steps": args.trajectory_steps,
            "trajectory_length": args.trajectory_length,
            "model": _model_kwargs(),
            "diffusion": {"T": 200, "beta_0": 0.0001, "beta_T": 0.02},
        },
        "environment": _environment(device),
        "untouched_reference_generation": reference_generation,
        "pace_repository": _local_repository_metadata(),
        "prediction_parity": predictions,
        "trajectory_parity": trajectory,
        "autograd": autograd,
        "generated_audio": audio,
        "untouched_reference_audio": reference_audio,
        "checks": {
            "prediction_and_trajectory_parity": parity_passed,
            "autograd_safe": autograd_passed,
            "valid_audio": audio_passed,
            "untouched_reference_generation": reference_audio_passed,
            "generated_audio": generated_audio_passed,
            "reference_environment_captured": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
