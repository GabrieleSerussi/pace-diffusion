"""Autograd-safe WaveNet with the legacy DiffWave checkpoint semantics.

This is a compact adaptation of ``models/wavenet.py`` and ``models/utils.py``
from ``albertfgu/diffwave-sashimi`` at the pinned ``checkpoints`` commit below.
The published one-million-step SC09 checkpoint was trained with an aliasing
side effect in every residual block::

    h = x
    h += part_t

The augmented assignment mutates ``x`` as well as ``h``.  Consequently, the
timestep projection participates in both the dilated-convolution path and the
residual identity path.  Replacing it with ``h = h + part_t`` changes the
network represented by the checkpoint.  ``LegacyCompatibleResidualBlock``
preserves the learned function without mutating an autograd-tracked tensor by
materializing ``residual_base = x + part_t`` and using that value in both
paths.  See the adjacent license and the provenance constants in this file.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn


DIFFWAVE_SASHIMI_UPSTREAM_REPOSITORY = "https://github.com/albertfgu/diffwave-sashimi"
DIFFWAVE_SASHIMI_CHECKPOINTS_COMMIT = "9bd78f8c894cad0952a5692450f2145e24466b29"
DIFFWAVE_SASHIMI_MODEL_PATH = "models/wavenet.py"
DIFFWAVE_SASHIMI_CHECKPOINT_PATH = (
    "exp/wnet_h256_d36_T200_betaT0.02_uncond/checkpoint/1000000.pkl"
)
DIFFWAVE_SASHIMI_CHECKPOINT_SHA256 = (
    "f34b9bcca4970572775fff15dbccda09f7ec8d56befe113620e1ad25eee416e1"
)
DIFFWAVE_SASHIMI_CHECKPOINT_SIZE_BYTES = 96_451_975
DIFFWAVE_SASHIMI_CHECKPOINT_SOURCE = (
    "https://media.githubusercontent.com/media/albertfgu/diffwave-sashimi/"
    f"{DIFFWAVE_SASHIMI_CHECKPOINTS_COMMIT}/{DIFFWAVE_SASHIMI_CHECKPOINT_PATH}"
)
DIFFWAVE_SASHIMI_LEGACY_SEMANTICS = "timestep_projection_in_residual_identity_v1"


def calc_diffusion_step_embedding(
    diffusion_steps: torch.Tensor,
    diffusion_step_embed_dim_in: int,
) -> torch.Tensor:
    """Match the checkpoint branch's sinusoidal timestep embedding.

    The upstream implementation constructs its frequency tensor on the CPU
    and then calls ``.cuda()``.  CPU and CUDA ``exp`` differ slightly for this
    calculation, enough to break bitwise checkpoint parity at later
    timesteps.  We retain CPU construction and then move the result to the
    input device, which is device-safe while matching the original CUDA path.
    """

    if diffusion_step_embed_dim_in % 2:
        raise ValueError("diffusion_step_embed_dim_in must be even")
    if diffusion_steps.ndim != 2 or diffusion_steps.shape[1] != 1:
        raise ValueError(
            "diffusion_steps must have shape [batch, 1], got "
            f"{tuple(diffusion_steps.shape)}"
        )

    half_dim = diffusion_step_embed_dim_in // 2
    if half_dim <= 1:
        raise ValueError("diffusion_step_embed_dim_in must be at least 4")
    scale = np.log(10_000) / (half_dim - 1)
    frequencies = torch.exp(torch.arange(half_dim) * -scale).to(
        device=diffusion_steps.device
    )
    arguments = diffusion_steps * frequencies
    return torch.cat((torch.sin(arguments), torch.cos(arguments)), dim=1)


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class Conv(nn.Module):
    """Weight-normalized 1-D convolution with upstream-compatible keys."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        convolution = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=padding,
        )
        # The legacy hook-based API is intentional: the published state dict
        # contains ``weight_g``/``weight_v`` keys from this parametrization.
        self.conv = nn.utils.weight_norm(convolution)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ZeroConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class LegacyCompatibleResidualBlock(nn.Module):
    """One residual block with the checkpoint's effective legacy function."""

    def __init__(
        self,
        res_channels: int,
        skip_channels: int,
        *,
        dilation: int = 1,
        diffusion_step_embed_dim_out: int = 512,
    ) -> None:
        super().__init__()
        self.res_channels = int(res_channels)
        self.fc_t = nn.Linear(diffusion_step_embed_dim_out, self.res_channels)
        self.dilated_conv_layer = Conv(
            self.res_channels,
            2 * self.res_channels,
            kernel_size=3,
            dilation=dilation,
        )
        self.res_conv = nn.utils.weight_norm(
            nn.Conv1d(self.res_channels, self.res_channels, kernel_size=1)
        )
        nn.init.kaiming_normal_(self.res_conv.weight)
        self.skip_conv = nn.utils.weight_norm(
            nn.Conv1d(self.res_channels, skip_channels, kernel_size=1)
        )
        nn.init.kaiming_normal_(self.skip_conv.weight)

    def forward(
        self,
        input_data: tuple[torch.Tensor, torch.Tensor],
        mel_spec: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mel_spec is not None:
            raise ValueError("The reproduced SC09 DiffWave teacher is unconditional")
        x, diffusion_step_embed = input_data
        if x.ndim != 3 or x.shape[1] != self.res_channels:
            raise ValueError(
                f"Expected residual input [batch, {self.res_channels}, length], "
                f"got {tuple(x.shape)}"
            )

        batch_size = x.shape[0]
        part_t = self.fc_t(diffusion_step_embed).view(
            batch_size, self.res_channels, 1
        )

        # This is the essential compatibility behavior.  Upstream's ``h = x;
        # h += part_t`` changed x in place, so the augmented value entered the
        # identity path too.  Keeping it under a new name preserves that math
        # without modifying x or invalidating autograd version counters.
        residual_base = x + part_t
        h = self.dilated_conv_layer(residual_base)
        out = torch.tanh(h[:, : self.res_channels, :]) * torch.sigmoid(
            h[:, self.res_channels :, :]
        )
        residual = self.res_conv(out)
        skip = self.skip_conv(out)
        return (residual_base + residual) * math.sqrt(0.5), skip


class LegacyCompatibleResidualGroup(nn.Module):
    def __init__(
        self,
        res_channels: int,
        skip_channels: int,
        *,
        num_res_layers: int = 30,
        dilation_cycle: int = 10,
        diffusion_step_embed_dim_in: int = 128,
        diffusion_step_embed_dim_mid: int = 512,
        diffusion_step_embed_dim_out: int = 512,
    ) -> None:
        super().__init__()
        self.num_res_layers = int(num_res_layers)
        self.diffusion_step_embed_dim_in = int(diffusion_step_embed_dim_in)
        self.fc_t1 = nn.Linear(
            self.diffusion_step_embed_dim_in,
            diffusion_step_embed_dim_mid,
        )
        self.fc_t2 = nn.Linear(
            diffusion_step_embed_dim_mid,
            diffusion_step_embed_dim_out,
        )
        self.residual_blocks = nn.ModuleList(
            [
                LegacyCompatibleResidualBlock(
                    res_channels,
                    skip_channels,
                    dilation=2 ** (index % dilation_cycle),
                    diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                )
                for index in range(self.num_res_layers)
            ]
        )

    def forward(
        self,
        input_data: tuple[torch.Tensor, torch.Tensor],
        mel_spec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x, diffusion_steps = input_data
        diffusion_step_embed = calc_diffusion_step_embedding(
            diffusion_steps,
            self.diffusion_step_embed_dim_in,
        )
        diffusion_step_embed = swish(self.fc_t1(diffusion_step_embed))
        diffusion_step_embed = swish(self.fc_t2(diffusion_step_embed))

        h = x
        skip: torch.Tensor | int = 0
        for block in self.residual_blocks:
            h, skip_n = block((h, diffusion_step_embed), mel_spec=mel_spec)
            # Preserve upstream's left-to-right accumulation order while
            # avoiding the second unnecessary in-place mutation.
            skip = skip + skip_n
        if not torch.is_tensor(skip):  # pragma: no cover - construction rejects zero blocks.
            raise RuntimeError("Residual group produced no skip activations")
        return skip * math.sqrt(1.0 / self.num_res_layers)


class LegacyCompatibleDiffWave(nn.Module):
    """Unconditional DiffWave teacher compatible with the published SC09 weights."""

    def __init__(
        self,
        in_channels: int = 1,
        res_channels: int = 256,
        skip_channels: int = 256,
        out_channels: int = 1,
        num_res_layers: int = 36,
        dilation_cycle: int = 12,
        diffusion_step_embed_dim_in: int = 128,
        diffusion_step_embed_dim_mid: int = 512,
        diffusion_step_embed_dim_out: int = 512,
        unconditional: bool = True,
        num_diffusion_steps: int = 200,
        beta_0: float = 0.0001,
        beta_T: float = 0.02,
        sample_rate: int = 16_000,
        example_length: int = 16_000,
    ) -> None:
        super().__init__()
        if not unconditional:
            raise ValueError("Only the unconditional SC09 DiffWave teacher is reproduced")
        if num_res_layers <= 0:
            raise ValueError("num_res_layers must be positive")
        if dilation_cycle <= 0:
            raise ValueError("dilation_cycle must be positive")

        self.res_channels = int(res_channels)
        self.skip_channels = int(skip_channels)
        self.num_res_layers = int(num_res_layers)
        self.dilation_cycle = int(dilation_cycle)
        self.unconditional = True
        self.num_diffusion_steps = int(num_diffusion_steps)
        self.beta_0 = float(beta_0)
        self.beta_T = float(beta_T)
        self.sample_rate = int(sample_rate)
        self.example_length = int(example_length)
        self.input_representation = "raw_waveform"
        self.model_family = "diffwave"
        self.legacy_forward_semantics = DIFFWAVE_SASHIMI_LEGACY_SEMANTICS

        self.init_conv = nn.Sequential(
            Conv(in_channels, self.res_channels, kernel_size=1),
            nn.ReLU(),
        )
        self.residual_layer = LegacyCompatibleResidualGroup(
            res_channels=self.res_channels,
            skip_channels=self.skip_channels,
            num_res_layers=self.num_res_layers,
            dilation_cycle=self.dilation_cycle,
            diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
            diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
            diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
        )
        self.final_conv = nn.Sequential(
            Conv(self.skip_channels, self.skip_channels, kernel_size=1),
            nn.ReLU(),
            ZeroConv1d(self.skip_channels, out_channels),
        )

    @property
    def residual_blocks(self) -> nn.ModuleList:
        return self.residual_layer.residual_blocks

    def named_residual_blocks(
        self,
    ) -> tuple[tuple[str, LegacyCompatibleResidualBlock], ...]:
        return tuple(
            (f"residual_layer.residual_blocks.{index}", block)
            for index, block in enumerate(self.residual_blocks)
        )

    def _normalize_inputs(
        self,
        input_data: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        diffusion_steps: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(input_data, tuple):
            if diffusion_steps is not None:
                raise ValueError("Pass timesteps either in the input tuple or as the second argument")
            if len(input_data) != 2:
                raise ValueError("DiffWave input tuple must be (audio, diffusion_steps)")
            audio, diffusion_steps = input_data
        else:
            audio = input_data
        if diffusion_steps is None:
            raise ValueError("DiffWave forward requires diffusion_steps")
        if audio.ndim != 3:
            raise ValueError(f"Expected audio [batch, channels, length], got {tuple(audio.shape)}")
        batch_size = audio.shape[0]
        if diffusion_steps.ndim == 0:
            diffusion_steps = diffusion_steps.expand(batch_size).reshape(batch_size, 1)
        elif diffusion_steps.ndim == 1:
            if diffusion_steps.shape[0] != batch_size:
                raise ValueError("Timestep batch size does not match audio batch size")
            diffusion_steps = diffusion_steps.reshape(batch_size, 1)
        elif diffusion_steps.ndim != 2 or diffusion_steps.shape != (batch_size, 1):
            raise ValueError(
                f"Expected timesteps [batch] or [batch, 1], got {tuple(diffusion_steps.shape)}"
            )
        return audio, diffusion_steps.to(device=audio.device)

    def forward(
        self,
        input_data: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        diffusion_steps: torch.Tensor | None = None,
        *,
        mel_spec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mel_spec is not None:
            raise ValueError("The reproduced SC09 DiffWave teacher is unconditional")
        audio, diffusion_steps = self._normalize_inputs(input_data, diffusion_steps)
        x = self.init_conv(audio)
        x = self.residual_layer((x, diffusion_steps))
        return self.final_conv(x)

    def predict_noise(
        self,
        x_t: torch.Tensor,
        diffusion_steps: torch.Tensor,
    ) -> torch.Tensor:
        return self(x_t, diffusion_steps)

    def __repr__(self) -> str:
        return (
            f"legacy_compatible_wavenet_h{self.res_channels}_d{self.num_res_layers}_uncond"
        )


_CONSTRUCTOR_KEYS = {
    "in_channels",
    "res_channels",
    "skip_channels",
    "out_channels",
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
}


def create_legacy_compatible_diffwave(
    config: Mapping[str, Any],
) -> LegacyCompatibleDiffWave:
    """Construct the model while ignoring provenance-only config fields."""

    architecture = config.get("architecture", "diffwave_wavenet_legacy_safe")
    if architecture != "diffwave_wavenet_legacy_safe":
        raise ValueError(
            "The SC09 checkpoint loader only supports the autograd-safe legacy "
            f"architecture, got {architecture!r}"
        )
    semantics = config.get(
        "legacy_forward_semantics",
        DIFFWAVE_SASHIMI_LEGACY_SEMANTICS,
    )
    if semantics != DIFFWAVE_SASHIMI_LEGACY_SEMANTICS:
        raise ValueError(
            "The SC09 checkpoint requires legacy residual semantics "
            f"{DIFFWAVE_SASHIMI_LEGACY_SEMANTICS!r}, got {semantics!r}"
        )
    kwargs = {key: config[key] for key in _CONSTRUCTOR_KEYS if key in config}
    return LegacyCompatibleDiffWave(**kwargs)


def diffwave_diffusion_hyperparameters(
    *,
    num_diffusion_steps: int = 200,
    beta_0: float = 0.0001,
    beta_T: float = 0.02,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor | int]:
    """Return the native DiffWave schedule with upstream operation order."""

    if num_diffusion_steps <= 0:
        raise ValueError("num_diffusion_steps must be positive")
    beta = torch.linspace(beta_0, beta_T, num_diffusion_steps, dtype=dtype)
    alpha = 1 - beta
    alpha_bar = alpha + 0
    beta_tilde = beta + 0
    for index in range(1, num_diffusion_steps):
        alpha_bar[index] *= alpha_bar[index - 1]
        beta_tilde[index] *= (1 - alpha_bar[index - 1]) / (1 - alpha_bar[index])
    sigma = torch.sqrt(beta_tilde)
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    return {
        "T": int(num_diffusion_steps),
        "Beta": beta.to(target_device),
        "Alpha": alpha.to(target_device),
        "Alpha_bar": alpha_bar.to(target_device),
        "Sigma": sigma.to(target_device),
    }


__all__ = [
    "DIFFWAVE_SASHIMI_CHECKPOINTS_COMMIT",
    "DIFFWAVE_SASHIMI_CHECKPOINT_PATH",
    "DIFFWAVE_SASHIMI_CHECKPOINT_SHA256",
    "DIFFWAVE_SASHIMI_CHECKPOINT_SIZE_BYTES",
    "DIFFWAVE_SASHIMI_CHECKPOINT_SOURCE",
    "DIFFWAVE_SASHIMI_LEGACY_SEMANTICS",
    "DIFFWAVE_SASHIMI_MODEL_PATH",
    "DIFFWAVE_SASHIMI_UPSTREAM_REPOSITORY",
    "LegacyCompatibleDiffWave",
    "LegacyCompatibleResidualBlock",
    "LegacyCompatibleResidualGroup",
    "calc_diffusion_step_embedding",
    "create_legacy_compatible_diffwave",
    "diffwave_diffusion_hyperparameters",
]
