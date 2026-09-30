"""Portable UNet compatible with OpenAI consistency-model checkpoints.

Derived from ``cm/unet.py`` and ``cm/nn.py`` in OpenAI consistency_models,
commit e32b69ee436d518377db86fb2127a3972d0d8716. See the adjacent license.
Only the 2-D diffusion UNet used by the published checkpoints is retained.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class GroupNorm32(nn.GroupNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).to(x.dtype)


def normalization(channels: int) -> nn.Module:
    return GroupNorm32(32, channels)


def zero_module(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10_000) -> torch.Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * frequencies[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        for layer in self:
            x = layer(x, emb) if isinstance(layer, TimestepBlock) else layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool, out_channels: int | None = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, self.out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[1]}")
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x) if self.use_conv else x


class Downsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool, out_channels: int | None = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        if use_conv:
            self.op = nn.Conv2d(channels, self.out_channels, 3, stride=2, padding=1)
        else:
            if channels != self.out_channels:
                raise ValueError("Average-pool downsampling cannot change channel count")
            self.op = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {x.shape[1]}")
        return self.op(x)


class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        *,
        out_channels: int | None = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        use_checkpoint: bool = False,
        up: bool = False,
        down: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )
        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # The compatibility model is inference-only, so the old custom
        # checkpoint autograd function is intentionally unnecessary here.
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = self.h_upd(in_rest(x))
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).to(h.dtype)
        while emb_out.ndim < h.ndim:
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_rest(out_norm(h) * (1 + scale) + shift)
        else:
            h = self.out_layers(h + emb_out)
        return self.skip_connection(x) + h


class QKVFlashAttention(nn.Module):
    """Portable explicit attention preserving the parameterless state contract."""

    def __init__(self, embed_dim: int, num_heads: int, **_: object):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

    def forward(self, qkv: torch.Tensor, *_: object, **__: object) -> torch.Tensor:
        batch, width, tokens = qkv.shape
        if width != 3 * self.embed_dim:
            raise ValueError(f"Expected QKV width {3 * self.embed_dim}, got {width}")
        qkv = qkv.reshape(batch, 3, self.num_heads, self.head_dim, tokens)
        q, k, v = qkv.unbind(dim=1)
        scale = 1 / math.sqrt(self.head_dim)
        weights = torch.einsum("bhct,bhcs->bhts", q, k) * scale
        # Match upstream's numerically stable FP32 softmax while keeping the
        # surrounding mixed-FP16 tensor dtype and exact state-dict keys.
        weights = torch.softmax(weights.float(), dim=-1).to(q.dtype)
        attended = torch.einsum("bhts,bhcs->bhct", weights, v)
        return attended.reshape(batch, self.embed_dim, tokens)


class QKVAttentionLegacy(nn.Module):
    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        batch, width, tokens = qkv.shape
        if width % (3 * self.n_heads):
            raise ValueError("QKV width is not divisible by the head count")
        head_dim = width // (3 * self.n_heads)
        qkv = qkv.reshape(batch, 3, self.n_heads, head_dim, tokens)
        q, k, v = qkv.unbind(dim=1)
        scale = 1 / math.sqrt(math.sqrt(head_dim))
        weights = torch.einsum("bhct,bhcs->bhts", q * scale, k * scale)
        weights = torch.softmax(weights.float(), dim=-1).to(weights.dtype)
        attended = torch.einsum("bhts,bhcs->bhct", weights, v)
        return attended.reshape(batch, -1, tokens)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        num_heads: int = 1,
        num_head_channels: int = -1,
        use_checkpoint: bool = False,
        attention_type: str = "flash",
        use_new_attention_order: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads if num_head_channels == -1 else channels // num_head_channels
        if channels % self.num_heads:
            raise ValueError("attention channels must be divisible by num_heads")
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        # OpenAI consistency_models deliberately used Conv2d here. Keeping the
        # 4-D weights is what makes the published state dict load strictly.
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.attention_type = attention_type
        self.attention = (
            QKVFlashAttention(channels, self.num_heads)
            if attention_type == "flash"
            else QKVAttentionLegacy(self.num_heads)
        )
        self.proj_out = zero_module(nn.Conv2d(channels, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, *spatial = x.shape
        qkv = self.qkv(self.norm(x)).view(batch, -1, math.prod(spatial))
        h = self.attention(qkv).view(batch, -1, *spatial)
        return x + self.proj_out(h)


def _convert_conv(module: nn.Module, dtype: torch.dtype) -> None:
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        module.weight.data = module.weight.data.to(dtype)
        if module.bias is not None:
            module.bias.data = module.bias.data.to(dtype)


class UNetModel(nn.Module):
    """State-dict-compatible OpenAI consistency-model diffusion UNet."""

    def __init__(
        self,
        *,
        image_size: int,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: Sequence[int],
        dropout: float = 0,
        channel_mult: Sequence[int] = (1, 2, 4, 8),
        conv_resample: bool = True,
        num_classes: int | None = None,
        use_checkpoint: bool = False,
        use_fp16: bool = False,
        num_heads: int = 1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        use_new_attention_order: bool = False,
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = tuple(attention_resolutions)
        self.dropout = dropout
        self.channel_mult = tuple(channel_mult)
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(nn.Conv2d(in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_channels = [ch]
        downsample = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers: list[nn.Module] = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if downsample in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_channels.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                layer = (
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=out_ch,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        down=True,
                    )
                    if resblock_updown
                    else Downsample(ch, conv_resample, out_channels=out_ch)
                )
                self.input_blocks.append(TimestepEmbedSequential(layer))
                ch = out_ch
                input_block_channels.append(ch)
                downsample *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            for index in range(num_res_blocks + 1):
                skip_channels = input_block_channels.pop()
                layers = [
                    ResBlock(
                        ch + skip_channels,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if downsample in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and index == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, out_channels=out_ch)
                    )
                    downsample //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(input_ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self) -> None:
        self.dtype = torch.float16
        self.input_blocks.apply(lambda module: _convert_conv(module, torch.float16))
        self.middle_block.apply(lambda module: _convert_conv(module, torch.float16))
        self.output_blocks.apply(lambda module: _convert_conv(module, torch.float16))

    def convert_to_fp32(self) -> None:
        self.dtype = torch.float32
        self.input_blocks.apply(lambda module: _convert_conv(module, torch.float32))
        self.middle_block.apply(lambda module: _convert_conv(module, torch.float32))
        self.output_blocks.apply(lambda module: _convert_conv(module, torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (y is not None) != (self.num_classes is not None):
            raise ValueError("y must be specified if and only if the model is class-conditional")
        hidden_states: list[torch.Tensor] = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            if y is None or y.shape != (x.shape[0],):
                raise ValueError("class labels must have shape [batch]")
            emb = emb + self.label_emb(y)
        h = x.to(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hidden_states.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = module(torch.cat([h, hidden_states.pop()], dim=1), emb)
        return self.out(h.to(x.dtype))


def create_unet(config: dict[str, object], *, use_fp16: bool = False) -> UNetModel:
    """Construct a UNet from an explicit serialized checkpoint config."""

    image_size = int(config["image_size"])
    attention_resolutions = tuple(
        image_size // int(resolution)
        for resolution in config.get("attention_resolutions", [16])  # type: ignore[arg-type]
    )
    label_dim = int(config.get("label_dim", 0))
    return UNetModel(
        image_size=image_size,
        in_channels=int(config.get("in_channels", 3)),
        model_channels=int(config["model_channels"]),
        out_channels=int(config.get("out_channels", 3)),
        num_res_blocks=int(config["num_res_blocks"]),
        attention_resolutions=attention_resolutions,
        dropout=float(config.get("dropout", 0.0)),
        channel_mult=tuple(int(value) for value in config["channel_mult"]),  # type: ignore[index]
        conv_resample=bool(config.get("conv_resample", True)),
        num_classes=label_dim or None,
        use_checkpoint=bool(config.get("use_checkpoint", False)),
        use_fp16=use_fp16,
        num_heads=int(config.get("num_heads", 1)),
        num_head_channels=int(config.get("num_head_channels", -1)),
        num_heads_upsample=int(config.get("num_heads_upsample", -1)),
        use_scale_shift_norm=bool(config.get("use_scale_shift_norm", False)),
        resblock_updown=bool(config.get("resblock_updown", False)),
        use_new_attention_order=bool(config.get("use_new_attention_order", False)),
    )
