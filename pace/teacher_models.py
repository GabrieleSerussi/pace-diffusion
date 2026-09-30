"""Structured teacher checkpoint specifications and shared model loading."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import torch

from .vendor.openai_consistency_unet import UNetModel, create_unet
from .vendor.diffwave_legacy import (
    DIFFWAVE_SASHIMI_CHECKPOINT_PATH,
    DIFFWAVE_SASHIMI_CHECKPOINT_SHA256,
    DIFFWAVE_SASHIMI_CHECKPOINT_SIZE_BYTES,
    DIFFWAVE_SASHIMI_CHECKPOINT_SOURCE,
    DIFFWAVE_SASHIMI_CHECKPOINTS_COMMIT,
    DIFFWAVE_SASHIMI_LEGACY_SEMANTICS,
    DIFFWAVE_SASHIMI_MODEL_PATH,
    DIFFWAVE_SASHIMI_UPSTREAM_REPOSITORY,
    LegacyCompatibleDiffWave,
    create_legacy_compatible_diffwave,
)


NVLABS_EDM_PICKLE = "nvlabs_edm_pickle"
OPENAI_CONSISTENCY_STATE_DICT = "openai_consistency_edm_state_dict_v1"
DIFFWAVE_SASHIMI_LEGACY_STATE_DICT = "diffwave_sashimi_legacy_state_dict_v1"

LSUN_BEDROOM_256_SOURCE = (
    "https://openaipublic.blob.core.windows.net/consistency/"
    "edm_bedroom256_ema.pt"
)
FFHQ_64_VP_SOURCE = (
    "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/"
    "edm-ffhq-64x64-uncond-vp.pkl"
)


_PRESETS: dict[str, dict[str, Any]] = {
    "sc09_diffwave_legacy_1m": {
        "source": DIFFWAVE_SASHIMI_CHECKPOINT_SOURCE,
        "format": DIFFWAVE_SASHIMI_LEGACY_STATE_DICT,
        "expected_size_bytes": DIFFWAVE_SASHIMI_CHECKPOINT_SIZE_BYTES,
        "expected_sha256": DIFFWAVE_SASHIMI_CHECKPOINT_SHA256,
        "architecture_metadata": {
            "parameter_count": 24_071_681,
            "state_dict_key_count": 408,
        },
        "model_config": {
            "architecture": "diffwave_wavenet_legacy_safe",
            "architecture_lineage": "diffwave_sashimi_checkpoints_branch",
            "model_family": "diffwave",
            "input_representation": "raw_waveform",
            "unconditional": True,
            "in_channels": 1,
            "out_channels": 1,
            "res_channels": 256,
            "skip_channels": 256,
            "num_res_layers": 36,
            "dilation_cycle": 12,
            "diffusion_step_embed_dim_in": 128,
            "diffusion_step_embed_dim_mid": 512,
            "diffusion_step_embed_dim_out": 512,
            "num_diffusion_steps": 200,
            "beta_0": 0.0001,
            "beta_T": 0.02,
            "sample_rate": 16_000,
            "example_length": 16_000,
            "legacy_forward_semantics": DIFFWAVE_SASHIMI_LEGACY_SEMANTICS,
            "weight_norm_state": "legacy_weight_g_weight_v",
            "checkpoint_container_key": "model_state_dict",
            "upstream_repository": DIFFWAVE_SASHIMI_UPSTREAM_REPOSITORY,
            "upstream_commit": DIFFWAVE_SASHIMI_CHECKPOINTS_COMMIT,
            "upstream_model_path": DIFFWAVE_SASHIMI_MODEL_PATH,
            "upstream_checkpoint_path": DIFFWAVE_SASHIMI_CHECKPOINT_PATH,
        },
        "sampling": {
            "sampler": "diffwave_ancestral",
            "num_steps": 200,
            "beta_0": 0.0001,
            "beta_T": 0.02,
            "prediction_type": "epsilon",
        },
    },
    "lsun_bedroom_256": {
        "source": LSUN_BEDROOM_256_SOURCE,
        "format": OPENAI_CONSISTENCY_STATE_DICT,
        "expected_size_bytes": 2_105_395_349,
        "expected_sha256": "5947bb5ae7b664feef1796ebe0a531efcf689d76e81d03ba88159dd8019b674f",
        "architecture_metadata": {
            "parameter_count": 526_304_771,
            "state_dict_key_count": 566,
        },
        "model_config": {
            "architecture": "openai_consistency_unet",
            "model_family": "edm",
            "image_size": 256,
            "in_channels": 3,
            "out_channels": 3,
            "label_dim": 0,
            "model_channels": 256,
            "num_res_blocks": 2,
            "channel_mult": [1, 1, 2, 2, 4, 4],
            "attention_resolutions": [32, 16, 8],
            "num_heads": 4,
            "num_head_channels": 64,
            "num_heads_upsample": -1,
            "resblock_updown": True,
            "use_scale_shift_norm": False,
            "use_new_attention_order": False,
            "conv_resample": True,
            "dropout": 0.1,
            "sigma_data": 0.5,
            "sigma_min": 0.002,
            "sigma_max": 80.0,
            "rho": 7.0,
            "time_scale": 250.0,
            "student_topology": {
                "channel_mult": [1, 1, 2, 2, 4, 4],
                "num_blocks": 2,
                "attn_resolutions": [32, 16, 8],
                "dropout": 0.1,
            },
        },
        "sampling": {
            "sampler": "heun",
            "num_steps": 40,
            "sigma_min": 0.002,
            "sigma_max": 80.0,
            "rho": 7.0,
            "s_churn": 0.0,
            "clip_denoised": True,
            "rng": "openai_deterministic_individual",
            "global_seed": 42,
        },
    },
    "ffhq_64_vp": {
        "source": FFHQ_64_VP_SOURCE,
        "format": NVLABS_EDM_PICKLE,
        "expected_size_bytes": 247_513_128,
        "expected_sha256": "f6f8f24a2b46ae79807b0f919e2550c6ed37cc3f8b7a2100629092cad9b5d2f5",
        "model_config": {
            "architecture": "song_unet",
            "architecture_lineage": "vp_ddpmpp",
            # The filename's "vp" suffix describes the DDPM++/VP-derived
            # architecture. The serialized teacher itself is EDMPrecond.
            "model_family": "edm",
            "image_size": 64,
            "in_channels": 3,
            "out_channels": 3,
            "label_dim": 0,
            "model_channels": 128,
            "channel_mult": [1, 2, 2, 2],
            "num_blocks": 4,
            "attention_resolutions": [16],
            "dropout": 0.05,
            "augment_dim": 9,
            "sigma_data": 0.5,
            "sigma_min": 0.0,
            # Keep structured metadata strict-JSON compatible.  Consumers pass
            # preconditioning values through ``float()``, so the explicit
            # string retains the mathematical meaning without emitting the
            # non-standard JSON token ``Infinity``.
            "sigma_max": "inf",
            "student_topology": {
                "channel_mult": [1, 2, 2, 2],
                "num_blocks": 4,
                "attn_resolutions": [16],
                "dropout": 0.05,
                "augment_dim": 9,
            },
        },
        "sampling": {
            "sampler": "heun",
            "num_steps": 40,
            "rho": 7.0,
            "s_churn": 0.0,
            "clip_denoised": False,
            "rng": "stacked_per_image_seed",
        },
    },
}

_PRESET_ALIASES = {
    "diffwave_sc09_1m": "sc09_diffwave_legacy_1m",
    "diffwave-sc09-1m": "sc09_diffwave_legacy_1m",
    "wnet_h256_d36_t200_betat0.02_uncond": "sc09_diffwave_legacy_1m",
    "bedroom256": "lsun_bedroom_256",
    "lsun_bedroom256": "lsun_bedroom_256",
    "lsun-bedroom-256": "lsun_bedroom_256",
    "ffhq64": "ffhq_64_vp",
    "ffhq-64-vp": "ffhq_64_vp",
}


def canonical_teacher_preset(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    normalized = _PRESET_ALIASES.get(normalized, normalized)
    if normalized not in _PRESETS:
        raise ValueError(
            f"Unknown teacher preset {value!r}; expected one of {sorted(_PRESETS)}"
        )
    return normalized


def teacher_preset_config(preset: str) -> dict[str, Any]:
    canonical = canonical_teacher_preset(preset)
    assert canonical is not None
    # JSON round-trip provides a small dependency-free deep copy.
    return json.loads(json.dumps(_PRESETS[canonical]))


@dataclass(frozen=True)
class TeacherSpec:
    """Serializable description of a teacher checkpoint and architecture."""

    source: str
    format: str = NVLABS_EDM_PICKLE
    preset: str | None = None
    model_config: dict[str, Any] = field(default_factory=dict)
    sampling: dict[str, Any] = field(default_factory=dict)
    expected_sha256: str | None = None
    expected_size_bytes: int | None = None
    checkpoint_sha256: str | None = None
    checkpoint_size_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TeacherSpec":
        source = value.get("source") or value.get("network_pkl") or value.get("path")
        if not source:
            raise ValueError("Teacher specification requires a non-empty source")
        resolved = resolve_teacher_spec(
            str(source),
            network_format=value.get("format") or value.get("network_format"),
            preset=value.get("preset") or value.get("network_preset"),
            model_config=value.get("model_config") or {},
            sampling=value.get("sampling") or {},
        )
        return replace(
            resolved,
            expected_sha256=value.get("expected_sha256", resolved.expected_sha256),
            expected_size_bytes=value.get("expected_size_bytes", resolved.expected_size_bytes),
            checkpoint_sha256=value.get("checkpoint_sha256"),
            checkpoint_size_bytes=value.get("checkpoint_size_bytes"),
        )


def _inferred_preset(source: str) -> str | None:
    basename = Path(urllib.parse.urlparse(source).path).name.lower()
    if basename == "edm_bedroom256_ema.pt":
        return "lsun_bedroom_256"
    if basename == "edm-ffhq-64x64-uncond-vp.pkl":
        return "ffhq_64_vp"
    return None


def _inferred_format(source: str, preset: str | None) -> str:
    if preset is not None:
        return str(_PRESETS[preset]["format"])
    suffix = Path(urllib.parse.urlparse(source).path).suffix.lower()
    if suffix == ".pkl":
        return NVLABS_EDM_PICKLE
    if suffix in {".pt", ".pth"}:
        raise ValueError(
            "A generic PyTorch checkpoint is ambiguous; specify teacher format "
            f"{OPENAI_CONSISTENCY_STATE_DICT!r} and an explicit model_config/preset"
        )
    raise ValueError(f"Could not infer teacher checkpoint format from {source!r}")


def resolve_teacher_spec(
    value: str | os.PathLike[str] | Mapping[str, Any] | TeacherSpec,
    *,
    network_format: str | None = None,
    preset: str | None = None,
    model_config: Mapping[str, Any] | None = None,
    sampling: Mapping[str, Any] | None = None,
) -> TeacherSpec:
    """Resolve new structured inputs and legacy ``network_pkl`` strings."""

    if isinstance(value, TeacherSpec):
        if any(item is not None for item in (network_format, preset, model_config, sampling)):
            raise ValueError("TeacherSpec overrides must be applied before resolution")
        return value
    if isinstance(value, Mapping):
        if "teacher" in value and isinstance(value["teacher"], Mapping):
            value = value["teacher"]
        return TeacherSpec.from_dict(value)

    source = os.fspath(value)
    canonical_preset = canonical_teacher_preset(preset or _inferred_preset(source))
    preset_payload = teacher_preset_config(canonical_preset) if canonical_preset else {}
    resolved_format = network_format or preset_payload.get("format")
    if resolved_format is None:
        resolved_format = _inferred_format(source, canonical_preset)
    if resolved_format not in {
        NVLABS_EDM_PICKLE,
        OPENAI_CONSISTENCY_STATE_DICT,
        DIFFWAVE_SASHIMI_LEGACY_STATE_DICT,
    }:
        raise ValueError(f"Unsupported teacher checkpoint format {resolved_format!r}")
    if canonical_preset and resolved_format != preset_payload["format"]:
        raise ValueError(
            f"Preset {canonical_preset!r} requires format {preset_payload['format']!r}, "
            f"got {resolved_format!r}"
        )
    merged_model_config = dict(preset_payload.get("model_config", {}))
    merged_model_config.update(dict(model_config or {}))
    merged_sampling = dict(preset_payload.get("sampling", {}))
    merged_sampling.update(dict(sampling or {}))
    if resolved_format == OPENAI_CONSISTENCY_STATE_DICT:
        required = {"image_size", "model_channels", "num_res_blocks", "channel_mult"}
        missing = sorted(required - merged_model_config.keys())
        if missing:
            raise ValueError(
                "OpenAI consistency state dicts require an explicit architecture; "
                f"missing model_config keys {missing}"
            )
    if resolved_format == DIFFWAVE_SASHIMI_LEGACY_STATE_DICT:
        required = {
            "in_channels",
            "out_channels",
            "res_channels",
            "skip_channels",
            "num_res_layers",
            "dilation_cycle",
            "diffusion_step_embed_dim_in",
            "diffusion_step_embed_dim_mid",
            "diffusion_step_embed_dim_out",
            "unconditional",
            "num_diffusion_steps",
            "beta_0",
            "beta_T",
            "sample_rate",
            "example_length",
            "legacy_forward_semantics",
        }
        missing = sorted(required - merged_model_config.keys())
        if missing:
            raise ValueError(
                "DiffWave legacy state dicts require an explicit architecture; "
                f"missing model_config keys {missing}"
            )
    return TeacherSpec(
        source=source,
        format=resolved_format,
        preset=canonical_preset,
        model_config=merged_model_config,
        sampling=merged_sampling,
        expected_sha256=preset_payload.get("expected_sha256"),
        expected_size_bytes=preset_payload.get("expected_size_bytes"),
    )


def teacher_spec_from_results(results: Mapping[str, Any]) -> TeacherSpec:
    for candidate in (
        results.get("teacher"),
        results.get("model_info", {}).get("teacher"),
        results.get("config", {}).get("teacher"),
    ):
        if isinstance(candidate, Mapping):
            return resolve_teacher_spec(candidate)
    config = results.get("config", {})
    source = config.get("network_pkl") or results.get("network_pkl")
    if not source:
        raise ValueError("Results do not contain teacher metadata or legacy network_pkl")
    model_info = results.get("model_info", {})
    model_config = dict(model_info.get("model_config") or {})
    for source_key, target_key in (
        ("img_resolution", "image_size"),
        ("img_channels", "in_channels"),
        ("label_dim", "label_dim"),
        ("model_family", "model_family"),
    ):
        if source_key in model_info:
            model_config.setdefault(target_key, model_info[source_key])
    return resolve_teacher_spec(
        str(source),
        network_format=config.get("network_format"),
        preset=config.get("network_preset"),
        model_config=model_config,
    )


def teacher_spec_from_plan(plan: Mapping[str, Any]) -> TeacherSpec:
    if isinstance(plan.get("teacher"), Mapping):
        return resolve_teacher_spec(plan["teacher"])
    source = plan.get("network_pkl")
    if not source:
        raise ValueError("Architecture plan has neither teacher metadata nor network_pkl")
    return resolve_teacher_spec(
        str(source),
        network_format=plan.get("network_format"),
        preset=plan.get("network_preset"),
    )


def _default_cache_dir() -> Path:
    configured = os.environ.get("PACE_MODEL_CACHE")
    if not configured:
        raise ValueError(
            "Remote teacher checkpoints require --model-cache-dir or PACE_MODEL_CACHE; "
            "an implicit home-directory cache is intentionally disabled for multi-GB checkpoints"
        )
    return Path(configured).expanduser()


def materialize_teacher_source(
    spec_or_source: TeacherSpec | str,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    """Return a local checkpoint path, atomically caching URLs once per host."""

    source = spec_or_source.source if isinstance(spec_or_source, TeacherSpec) else str(spec_or_source)
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme not in {"http", "https"}:
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Teacher checkpoint does not exist: {path}")
        return path

    destination_dir = Path(cache_dir).expanduser() if cache_dir is not None else _default_cache_dir()
    destination_dir.mkdir(parents=True, exist_ok=True)
    basename = Path(parsed.path).name or "teacher-checkpoint"
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    destination = destination_dir / f"{digest}-{basename}"
    canonical_destination = destination_dir / basename
    if canonical_destination.is_file() and canonical_destination.stat().st_size > 0:
        if (
            isinstance(spec_or_source, TeacherSpec)
            and spec_or_source.expected_size_bytes is not None
            and canonical_destination.stat().st_size != int(spec_or_source.expected_size_bytes)
        ):
            raise ValueError(
                f"Cached teacher checkpoint size mismatch for {canonical_destination}: expected "
                f"{spec_or_source.expected_size_bytes}, got {canonical_destination.stat().st_size}"
            )
        return canonical_destination.resolve()
    lock_path = destination.with_suffix(destination.suffix + ".lock")
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+b") as lock_handle:
        try:
            import fcntl

            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - fcntl is available on Linux and macOS.
            pass
        if destination.is_file() and destination.stat().st_size > 0:
            return destination.resolve()
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        try:
            request = urllib.request.Request(source, headers={"User-Agent": "PACE/1.0"})
            with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            if temporary.stat().st_size <= 0:
                raise OSError(f"Downloaded checkpoint is empty: {source}")
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
    return destination.resolve()


def checkpoint_file_metadata(path: Path, spec: TeacherSpec) -> TeacherSpec:
    """Validate expected size/checksum and attach resolved checkpoint identity."""

    size_bytes = int(path.stat().st_size)
    if spec.expected_size_bytes is not None and size_bytes != int(spec.expected_size_bytes):
        raise ValueError(
            f"Teacher checkpoint size mismatch for {path}: expected "
            f"{spec.expected_size_bytes}, got {size_bytes}"
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    sha256 = digest.hexdigest()
    if spec.expected_sha256 is not None and sha256.lower() != spec.expected_sha256.lower():
        raise ValueError(
            f"Teacher checkpoint SHA-256 mismatch for {path}: expected "
            f"{spec.expected_sha256}, got {sha256}"
        )
    return replace(
        spec,
        checkpoint_sha256=sha256,
        checkpoint_size_bytes=size_bytes,
    )


_EDM_PICKLE_MODULES = frozenset({"torch_utils", "dnnlib", "training"})


def _ensure_nvlabs_edm_importable() -> None:
    """Make ``torch_utils``/``dnnlib`` from NVlabs/edm importable for EDM pickles.

    A missing checkout is not an error here: pickles that do not reference
    EDM classes still load, and :func:`_load_nvlabs_pickle` reports a missing
    checkout clearly when the unpickler needs it.
    """

    from .external_repos import ExternalRepositoryError, ensure_edm_importable

    try:
        ensure_edm_importable()
    except ExternalRepositoryError:
        pass


def _load_nvlabs_pickle(path: Path) -> Any:
    """Unpickle an NVLabs EDM checkpoint with a clear error when EDM is missing."""

    from .external_repos import ensure_edm_importable

    _ensure_nvlabs_edm_importable()
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] not in _EDM_PICKLE_MODULES:
            raise
        # Raises ExternalRepositoryError with setup instructions.
        ensure_edm_importable()
        raise


class OpenAIConsistencyEDMPrecond(torch.nn.Module):
    """EDM denoiser wrapper for OpenAI's raw guided-diffusion-style UNet."""

    def __init__(self, model: UNetModel, config: Mapping[str, Any], *, teacher_spec: TeacherSpec):
        super().__init__()
        self.model = model
        self.teacher_spec = teacher_spec.to_dict()
        self.model_config = dict(config)
        self.img_resolution = int(config["image_size"])
        self.img_channels = int(config.get("in_channels", 3))
        self.label_dim = int(config.get("label_dim", 0))
        self.sigma_data = float(config.get("sigma_data", 0.5))
        self.sigma_min = float(config.get("sigma_min", 0.002))
        self.sigma_max = float(config.get("sigma_max", 80.0))
        self.time_scale = float(config.get("time_scale", 250.0))
        self.model_family = "edm"
        self.checkpoint_format = OPENAI_CONSISTENCY_STATE_DICT
        self._structural_roots = _openai_structural_roots(config)

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        class_labels: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        x = x.to(torch.float32)
        sigma = sigma.to(device=x.device, dtype=torch.float32).reshape(-1, 1, 1, 1)
        if self.label_dim == 0:
            if class_labels is not None and class_labels.numel() > 0:
                raise ValueError("The LSUN Bedroom teacher is unconditional")
            y = None
        elif class_labels is None:
            raise ValueError("A conditional OpenAI teacher requires class labels")
        elif class_labels.ndim == 1:
            y = class_labels.to(device=x.device, dtype=torch.long)
        else:
            if class_labels.shape[-1] != self.label_dim:
                raise ValueError(
                    f"Expected one-hot labels of width {self.label_dim}, got {class_labels.shape}"
                )
            y = class_labels.argmax(dim=-1).to(device=x.device, dtype=torch.long)

        c_skip = self.sigma_data**2 / (sigma.square() + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma.square() + self.sigma_data**2).sqrt()
        c_in = 1 / (sigma.square() + self.sigma_data**2).sqrt()
        timesteps = self.time_scale * torch.log(sigma.flatten() + 1e-44)
        raw = self.model(c_in * x, timesteps, y=y)
        return c_skip * x + c_out * raw.to(torch.float32)

    def round_sigma(self, sigma: torch.Tensor | float) -> torch.Tensor:
        return torch.as_tensor(sigma)

    def structural_key_for_module(self, module_name: str) -> str:
        normalized = module_name.removeprefix("model.")
        matches = [root for root in self._structural_roots if normalized == root or normalized.startswith(root + ".")]
        if not matches:
            return normalized
        return self._structural_roots[max(matches, key=len)]


def _openai_structural_roots(config: Mapping[str, Any]) -> dict[str, str]:
    image_size = int(config["image_size"])
    channel_mult = list(config["channel_mult"])
    num_res_blocks = int(config["num_res_blocks"])
    attention_resolutions = {int(value) for value in config.get("attention_resolutions", [])}
    roots: dict[str, str] = {"input_blocks.0": f"enc.{image_size}x{image_size}_conv"}
    input_index = 1
    for level in range(len(channel_mult)):
        resolution = image_size >> level
        for block_index in range(num_res_blocks):
            roots[f"input_blocks.{input_index}"] = f"enc.{resolution}x{resolution}_block{block_index}"
            input_index += 1
        if level != len(channel_mult) - 1:
            next_resolution = image_size >> (level + 1)
            roots[f"input_blocks.{input_index}"] = f"enc.{next_resolution}x{next_resolution}_down"
            input_index += 1

    lowest_resolution = image_size >> (len(channel_mult) - 1)
    roots["middle_block.0"] = f"dec.{lowest_resolution}x{lowest_resolution}_in0"
    roots["middle_block.1"] = f"dec.{lowest_resolution}x{lowest_resolution}_in0"
    roots["middle_block.2"] = f"dec.{lowest_resolution}x{lowest_resolution}_in1"

    output_index = 0
    for level in reversed(range(len(channel_mult))):
        resolution = image_size >> level
        for block_index in range(num_res_blocks + 1):
            root = f"output_blocks.{output_index}"
            roots[root] = f"dec.{resolution}x{resolution}_block{block_index}"
            if level and block_index == num_res_blocks:
                up_layer_index = 2 if resolution in attention_resolutions else 1
                next_resolution = image_size >> (level - 1)
                roots[f"{root}.{up_layer_index}"] = f"dec.{next_resolution}x{next_resolution}_up"
            output_index += 1
    roots["out"] = "out_conv"
    return roots


def _load_openai_state_dict(path: Path) -> Mapping[str, torch.Tensor]:
    load_kwargs: dict[str, Any] = {"map_location": "cpu", "weights_only": True}
    try:
        payload = torch.load(path, mmap=True, **load_kwargs)
    except (TypeError, RuntimeError):
        payload = torch.load(path, **load_kwargs)
    if isinstance(payload, Mapping) and "state_dict" in payload and isinstance(payload["state_dict"], Mapping):
        payload = payload["state_dict"]
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("OpenAI consistency checkpoint must contain a non-empty state dict")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in payload.items()):
        raise ValueError("OpenAI consistency checkpoint contains non-tensor state entries")
    return payload  # type: ignore[return-value]


def _load_diffwave_state_dict(path: Path) -> Mapping[str, torch.Tensor]:
    """Load only the published checkpoint's explicit model-state envelope."""

    load_kwargs: dict[str, Any] = {"map_location": "cpu", "weights_only": True}
    try:
        payload = torch.load(path, mmap=True, **load_kwargs)
    except (TypeError, RuntimeError):
        payload = torch.load(path, **load_kwargs)
    if not isinstance(payload, Mapping) or "model_state_dict" not in payload:
        keys = sorted(payload.keys()) if isinstance(payload, Mapping) else []
        raise ValueError(
            "DiffWave checkpoint must contain checkpoint['model_state_dict']; "
            f"top-level keys={keys}"
        )
    state_dict = payload["model_state_dict"]
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("DiffWave checkpoint model_state_dict must be a non-empty mapping")
    if not all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in state_dict.items()
    ):
        raise ValueError("DiffWave checkpoint model_state_dict contains non-tensor entries")
    return state_dict  # type: ignore[return-value]


def load_teacher_network(
    spec_or_source: TeacherSpec | Mapping[str, Any] | str | os.PathLike[str],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    cache_dir: str | Path | None = None,
    network_format: str | None = None,
    preset: str | None = None,
    trust_local_pickle: bool = False,
) -> torch.nn.Module:
    """Load a supported, provenance-checked teacher checkpoint."""

    spec = resolve_teacher_spec(
        spec_or_source,
        network_format=network_format,
        preset=preset,
    ) if not isinstance(spec_or_source, TeacherSpec) else spec_or_source
    path = materialize_teacher_source(spec, cache_dir=cache_dir)
    spec = checkpoint_file_metadata(path, spec)
    if spec.format == DIFFWAVE_SASHIMI_LEGACY_STATE_DICT:
        if dtype != torch.float32:
            raise ValueError(
                "Exact DiffWave checkpoint reproduction requires torch.float32; "
                f"got {dtype}"
            )
        network = create_legacy_compatible_diffwave(spec.model_config)
        state_dict = _load_diffwave_state_dict(path)
        network.load_state_dict(state_dict, strict=True)
        network.teacher_spec = spec.to_dict()  # type: ignore[attr-defined]
        network.checkpoint_format = spec.format  # type: ignore[attr-defined]
        return network.eval().requires_grad_(False).to(device=device, dtype=dtype)

    if spec.format == NVLABS_EDM_PICKLE:
        trusted_builtin_source = spec.source == FFHQ_64_VP_SOURCE
        if not trusted_builtin_source and not trust_local_pickle:
            raise ValueError(
                "Refusing to unpickle a local/third-party teacher checkpoint without explicit trust. "
                "Use the built-in FFHQ source URL or pass --trust-local-pickle only for a checkpoint "
                "whose origin you have verified."
            )
        payload = _load_nvlabs_pickle(path)
        if isinstance(payload, Mapping):
            network = payload.get("ema")
            if network is None:
                network = payload.get("model")
        elif isinstance(payload, torch.nn.Module):
            network = payload
        else:
            network = None
        if not isinstance(network, torch.nn.Module):
            keys = sorted(payload.keys()) if isinstance(payload, Mapping) else []
            raise ValueError(f"Unsupported NVLabs checkpoint payload; dictionary keys={keys}")
        network.teacher_spec = spec.to_dict()  # type: ignore[attr-defined]
        network.checkpoint_format = spec.format  # type: ignore[attr-defined]
        return network.eval().requires_grad_(False).to(device=device, dtype=dtype)

    if dtype == torch.bfloat16:
        raise ValueError("OpenAI consistency teachers currently support float32 or float16, not bfloat16")
    use_fp16 = dtype == torch.float16 and device.type == "cuda"
    if dtype not in {torch.float32, torch.float16}:
        raise ValueError(f"Unsupported OpenAI teacher dtype {dtype}")
    raw_model = create_unet(spec.model_config, use_fp16=use_fp16)
    state_dict = _load_openai_state_dict(path)
    raw_model.load_state_dict(state_dict, strict=True)
    raw_model.to(device=device)
    if use_fp16:
        raw_model.convert_to_fp16()
    wrapper = OpenAIConsistencyEDMPrecond(raw_model, spec.model_config, teacher_spec=spec)
    return wrapper.eval().requires_grad_(False)


def infer_network_model_family(network: torch.nn.Module) -> str:
    explicit = getattr(network, "model_family", None)
    if explicit:
        return str(explicit).lower()
    name = network.__class__.__name__.lower()
    if "vpprecond" in name:
        return "vp"
    if "veprecond" in name:
        return "ve"
    return "edm"


def teacher_model_metadata(network: torch.nn.Module, spec: TeacherSpec) -> dict[str, Any]:
    """Return explicit, serializable model facts for results and plans."""

    model_family = infer_network_model_family(network)
    if model_family == "diffwave":
        model_config = dict(spec.model_config)
        model_config.update(
            {
                "model_family": "diffwave",
                "input_representation": getattr(
                    network,
                    "input_representation",
                    spec.model_config.get("input_representation", "raw_waveform"),
                ),
                "in_channels": int(
                    spec.model_config.get("in_channels", 1)
                ),
                "out_channels": int(
                    spec.model_config.get("out_channels", 1)
                ),
                "res_channels": int(
                    getattr(network, "res_channels", spec.model_config["res_channels"])
                ),
                "skip_channels": int(
                    getattr(network, "skip_channels", spec.model_config["skip_channels"])
                ),
                "num_res_layers": int(
                    getattr(network, "num_res_layers", spec.model_config["num_res_layers"])
                ),
                "dilation_cycle": int(
                    getattr(network, "dilation_cycle", spec.model_config["dilation_cycle"])
                ),
                "num_diffusion_steps": int(
                    getattr(
                        network,
                        "num_diffusion_steps",
                        spec.model_config["num_diffusion_steps"],
                    )
                ),
                "beta_0": float(
                    getattr(network, "beta_0", spec.model_config["beta_0"])
                ),
                "beta_T": float(
                    getattr(network, "beta_T", spec.model_config["beta_T"])
                ),
                "sample_rate": int(
                    getattr(network, "sample_rate", spec.model_config["sample_rate"])
                ),
                "example_length": int(
                    getattr(network, "example_length", spec.model_config["example_length"])
                ),
                "legacy_forward_semantics": getattr(
                    network,
                    "legacy_forward_semantics",
                    spec.model_config["legacy_forward_semantics"],
                ),
            }
        )
        return {
            "teacher": spec.to_dict(),
            "checkpoint_format": spec.format,
            "checkpoint_preset": spec.preset,
            "checkpoint_sha256": spec.checkpoint_sha256,
            "checkpoint_size_bytes": spec.checkpoint_size_bytes,
            "expected_sha256": spec.expected_sha256,
            "expected_size_bytes": spec.expected_size_bytes,
            "model_family": "diffwave",
            "net_class_name": network.__class__.__name__,
            "input_representation": model_config["input_representation"],
            "channels": model_config["in_channels"],
            "sample_rate": model_config["sample_rate"],
            "example_length": model_config["example_length"],
            "num_diffusion_steps": model_config["num_diffusion_steps"],
            "beta_0": model_config["beta_0"],
            "beta_T": model_config["beta_T"],
            "prediction_type": "epsilon",
            "model_config": model_config,
        }

    image_size = int(getattr(network, "img_resolution", spec.model_config.get("image_size", 0)))
    img_channels = int(getattr(network, "img_channels", spec.model_config.get("in_channels", 3)))
    label_dim = int(getattr(network, "label_dim", spec.model_config.get("label_dim", 0)))
    model_config = dict(spec.model_config)
    init_kwargs = getattr(network, "_init_kwargs", None)
    if isinstance(init_kwargs, Mapping):
        for key in (
            "model_channels",
            "channel_mult",
            "dropout",
            "augment_dim",
            "embedding_type",
            "encoder_type",
            "decoder_type",
            "channel_mult_noise",
            "resample_filter",
        ):
            if key in init_kwargs:
                value = init_kwargs[key]
                model_config[key] = list(value) if isinstance(value, tuple) else value
    model_config.update(
        {
            "image_size": image_size,
            "in_channels": img_channels,
            "label_dim": label_dim,
            "model_family": model_family,
        }
    )
    sigma_max = float(getattr(network, "sigma_max", spec.model_config.get("sigma_max", float("inf"))))
    if math.isnan(sigma_max):
        raise ValueError("Teacher sigma_max must not be NaN")
    serialized_sigma_max: float | str = (
        "inf" if sigma_max == float("inf") else "-inf" if sigma_max == float("-inf") else sigma_max
    )
    return {
        "teacher": spec.to_dict(),
        "checkpoint_format": spec.format,
        "checkpoint_preset": spec.preset,
        "checkpoint_sha256": spec.checkpoint_sha256,
        "checkpoint_size_bytes": spec.checkpoint_size_bytes,
        "expected_sha256": spec.expected_sha256,
        "expected_size_bytes": spec.expected_size_bytes,
        "model_family": model_family,
        "net_class_name": network.__class__.__name__,
        "img_resolution": image_size,
        "img_channels": img_channels,
        "label_dim": label_dim,
        "sigma_min": float(getattr(network, "sigma_min", spec.model_config.get("sigma_min", 0.0))),
        "sigma_max": serialized_sigma_max,
        "sigma_data": float(getattr(network, "sigma_data", spec.model_config.get("sigma_data", 0.5))),
        "model_config": model_config,
    }
