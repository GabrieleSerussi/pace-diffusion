"""DiT width-allocation distillation port: NarrowDiT + budget/matcher/plan.

Width allocation for DiT students, first developed on the CIFAR-10 DiT-Micro
teacher. Every block keeps the shared residual dim ``D`` but shrinks its own
attention inner dim ``A`` (bottleneck attention) and MLP hidden dim ``M``.

Two layers:

1. ``NarrowDiT`` (torch): an I/O-compatible clone of the DiTMicro teacher
   (same ``forward(x, t, y)`` signature, same output shape, no learn_sigma), but
   with per-block ``(num_heads, attn_inner, mlp_hidden)``. It reuses the stock
   DiT repo's ``TimestepEmbedder``/``LabelEmbedder``/``FinalLayer``/``PatchEmbed``/
   ``get_2d_sincos_pos_embed``/``modulate`` verbatim; only the transformer block is
   new. When every block is (A==D, M==int(D*mlp_ratio), H==num_heads) and depth/D
   match the teacher, NarrowDiT has the SAME param count as the stock DiTMicro.

2. Pure-numpy allocation functions (no torch): turn a permutation-importance
   results dict + timestep phases into a per-phase architecture plan that spends
   the teacher's parameter budget across blocks/layers by importance.
"""
from __future__ import annotations

import json
import math
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- The stock DiT components (reused verbatim, imported lazily) --------------
# The upstream facebookresearch/DiT checkout is located through ``--dit_repo`` or
# ``$DIT_REPO`` (see ``pace.external_repos``).  It is imported on first use, when
# a ``NarrowDiT`` is constructed, so the pure-numpy planner below and ``import
# pace.dit_arch_alloc`` work without DiT or timm installed.
_DIT_SYMBOL_NAMES: Tuple[str, ...] = (
    "FinalLayer",
    "LabelEmbedder",
    "Mlp",
    "PatchEmbed",
    "TimestepEmbedder",
    "get_2d_sincos_pos_embed",
    "modulate",
)


def load_dit_symbols() -> Dict[str, object]:
    """Import DiT's ``models`` module and bind the reused components here.

    The names are bound as module globals, so the classes below reference them
    exactly as they did when they were imported at module import time.
    """
    missing = [name for name in _DIT_SYMBOL_NAMES if name not in globals()]
    if missing:
        from pace.external_repos import import_dit_module

        dit_models = import_dit_module("models")
        globals().update({name: getattr(dit_models, name) for name in _DIT_SYMBOL_NAMES})
    return {name: globals()[name] for name in _DIT_SYMBOL_NAMES}


def __getattr__(name: str):
    # ``from pace.dit_arch_alloc import Mlp`` keeps working after the lazy import.
    if name in _DIT_SYMBOL_NAMES:
        return load_dit_symbols()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# The DiTMicro teacher config (from scripts/evaluate_parameters_dit_micro.py).
MICRO_TEACHER_CFG: Dict[str, object] = {
    "hidden_size": 192,
    "depth": 8,
    "num_heads": 3,
    "mlp_ratio": 4.0,
    "patch_size": 2,
    "in_channels": 3,
    "input_size": 32,
    "num_classes": 10,
    "learn_sigma": False,
}

# The CIFAR-10 DiT-S/2-style teacher (D=384, depth 12; the "global_s" plan of
# the teacher-training study, the CIFAR-10 teacher this config's SHAPE was
# hand-authored for). Uniform per-block (num_heads=6,
# attn_inner=384, mlp_hidden=1536=384*4) at this hidden_size/depth analytically
# accounts for exactly 32,475,660 params -- the same "teacher_params" value
# recorded in that plan file and matching the teacher_v4_smicro run's reported
# 32,479,116 realized (= 32,475,660 + 9*384 for the augment_dim=9 embedder,
# which is a training-recipe addition threaded separately via
# --augment_prob/--net_dropout, not part of this budget-accounting config).
SMICRO_TEACHER_CFG: Dict[str, object] = {
    "hidden_size": 384,
    "depth": 12,
    "num_heads": 6,
    "mlp_ratio": 4.0,
    "patch_size": 2,
    "in_channels": 3,
    "input_size": 32,
    "num_classes": 10,
    "learn_sigma": False,
}

_BLOCK_RE = re.compile(r"blocks\.(\d+)\.")


# =============================================================================
# NarrowDiT (torch)
# =============================================================================

class NarrowAttention(nn.Module):
    """Bottleneck self-attention: project D -> A, attend with H heads of dim A//H,
    project A -> D. Mirrors timm ``Attention`` math but with an inner dim ``A``
    that can be smaller than the residual dim ``D``.

    With A == D and qkv_bias == True this is bit-for-bit the same module layout as
    the stock DiT's timm ``Attention`` (qkv Linear(D, 3D), proj Linear(D, D)), so
    it contributes the identical parameter count.

    ``attn_impl`` (default ``"manual"``, BYTE-IDENTICAL to every module built before
    this option existed -- a hard repo rule) selects the attention execution path:

      * ``"manual"`` (default): the original explicit ``(q @ kᵀ) * scale -> softmax
        -> @ v`` math, unchanged.
      * ``"sdpa"``: reshapes to the same ``(B, H, N, head_dim)`` q/k/v this class
        already computes, then calls ``F.scaled_dot_product_attention`` (a fused/
        flash kernel) instead of doing the matmuls by hand. This is how ``timm``'s
        ``Attention`` (``fused_attn=True``) executes attention, and was measured at
        about 41-44%% faster at D=1024 -- purely an execution-path
        change, SAME weights and SAME math (up to kernel-level float non-associativity),
        since ``qkv``/``proj``/``proj_drop`` are untouched and no new parameters are
        added. Two things to get right so it does not silently change the result:
        (1) ``scale=self.scale`` is passed explicitly; SDPA's ``scale`` kwarg REPLACES
        its default ``1/sqrt(head_dim)`` (same value for an unpadded module), and it
        is what keeps ``pad_attention_heads`` exact; (2) this module has no
        attention-probability dropout (only ``proj_drop`` on the post-projection
        output, see the class docstring above and ``NarrowDiTBlock``'s), so
        ``dropout_p=0.0`` is passed unconditionally -- there is nothing to map from
        a nonexistent ``attn_drop``.
    """

    def __init__(self, hidden_size: int, num_heads: int, attn_inner: int, dropout: float = 0.0,
                 attn_impl: str = "manual"):
        super().__init__()
        if attn_inner % num_heads != 0:
            raise ValueError(
                f"attn_inner ({attn_inner}) must be divisible by num_heads ({num_heads})"
            )
        if attn_impl not in ("manual", "sdpa"):
            raise ValueError(f"attn_impl must be 'manual' or 'sdpa', got {attn_impl!r}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attn_inner = attn_inner
        self.head_dim = attn_inner // num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_impl = attn_impl
        self.qkv = nn.Linear(hidden_size, 3 * attn_inner, bias=True)
        self.proj = nn.Linear(attn_inner, hidden_size)
        # Output (post-projection) dropout, mirroring timm ``Attention.proj_drop``.
        # nn.Dropout carries no parameters/buffers, so the state_dict is unchanged;
        # with p == 0.0 (the default) it is an exact identity that consumes no RNG
        # even in train() mode, so augment/dropout-off runs stay byte-identical.
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        A, H, hd = self.attn_inner, self.num_heads, self.head_dim
        qkv = self.qkv(x).reshape(B, N, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, N, hd)
        if self.attn_impl == "sdpa":
            # Fused kernel. scale= REPLACES SDPA's default 1/sqrt(hd) (it does not
            # stack on it), so passing self.scale is value-identical for an unpadded
            # module and is what keeps ``pad_attention_heads`` exact (padded hd, original
            # scale). No attn-prob dropout exists in this module -> dropout_p 0.0.
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=self.scale)
            x = x.transpose(1, 2).reshape(B, N, A)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(B, N, A)
        return self.proj_drop(self.proj(x))


@torch.no_grad()
def pad_attention_heads(model: nn.Module, multiple: int = 8) -> int:
    """Inference-only, mathematically EXACT: zero-pad every ``NarrowAttention`` head_dim
    up to a multiple of ``multiple`` so ``F.scaled_dot_product_attention`` can pick the
    flash/efficient kernels (they require head_dim % 8 == 0; the width-allocation
    students have head dims like 34/38/51 and otherwise fall back to the math kernel,
    ~20% slower end-to-end).

    Exact because padded q/k slots are 0 (add 0 to every logit), padded v slots are 0
    and their ``proj`` input columns are 0 (contribute 0 to the output), and the
    softmax scale stays the ORIGINAL ``hd ** -0.5`` (``forward`` passes ``self.scale``
    explicitly). The padded module has more parameters, all zero -- param/FLOP
    accounting must use the unpadded model. Returns the number of modules padded.
    """
    n = 0
    for m in model.modules():
        if not isinstance(m, NarrowAttention) or m.head_dim % multiple == 0:
            continue
        H, hd = m.num_heads, m.head_dim
        hdp = -(-hd // multiple) * multiple
        Ap = H * hdp
        dev, dt = m.qkv.weight.device, m.qkv.weight.dtype
        # qkv rows are laid out (3, H, hd): scatter each (which, head) slice into (which, head, :hd)
        qkv = nn.Linear(m.hidden_size, 3 * Ap, bias=True).to(dev, dt)
        qkv.weight.zero_(); qkv.bias.zero_()
        qkv.weight.view(3, H, hdp, -1)[:, :, :hd] = m.qkv.weight.view(3, H, hd, -1)
        qkv.bias.view(3, H, hdp)[:, :, :hd] = m.qkv.bias.view(3, H, hd)
        proj = nn.Linear(Ap, m.hidden_size, bias=m.proj.bias is not None).to(dev, dt)
        proj.weight.zero_()
        proj.weight.view(-1, H, hdp)[:, :, :hd] = m.proj.weight.view(-1, H, hd)
        if proj.bias is not None:
            proj.bias.copy_(m.proj.bias)
        m.qkv, m.proj = qkv, proj
        m.attn_inner, m.head_dim = Ap, hdp  # self.scale deliberately left at hd ** -0.5
        n += 1
    return n


class NarrowDiTBlock(nn.Module):
    """DiT block whose attention inner dim and MLP hidden dim can differ from the
    residual dim. adaLN-Zero conditioning identical to the stock ``DiTBlock``.

    ``dropout`` (default 0.0 = off) is applied on both residual branches' outputs:
    the attention output projection (``NarrowAttention.proj_drop``) and inside the
    MLP (timm ``Mlp(drop=...)``, i.e. after fc1+act and after fc2). That mirrors
    where EDM's reference nets put dropout -- in the residual branch, on the block's
    contribution to the residual stream (``UNetBlock``: norm -> silu -> dropout ->
    conv) -- expressed in the DiT/timm idiom. No dropout is applied to the attention
    probabilities (timm's ``attn_drop``), matching EDM, which has no analogue.
    """

    def __init__(self, hidden_size: int, num_heads: int, attn_inner: int, mlp_hidden: int,
                 dropout: float = 0.0, attn_impl: str = "manual"):
        super().__init__()
        load_dit_symbols()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = NarrowAttention(hidden_size, num_heads, attn_inner, dropout=dropout,
                                    attn_impl=attn_impl)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden, act_layer=approx_gelu,
                       drop=dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class NarrowDiT(nn.Module):
    """DiT with a shared residual dim ``D`` and per-block attention/MLP widths.

    I/O-compatible with the DiTMicro teacher: ``forward(x, t, y)`` where ``t`` is a
    per-example scalar (EDM ``c_noise``) and ``y`` is class indices; returns
    ``(N, out_channels, H, W)`` with ``out_channels == in_channels`` when
    ``learn_sigma=False`` (the teacher's setting).

    Two OPT-IN Karras/EDM regularizers, both defaulting to OFF so that every model
    built before they existed is reproduced bit-for-bit (same parameter count, same
    ``state_dict`` keys, same forward output -- and existing checkpoints therefore
    still load with ``strict=True``):

      * ``augment_dim`` (default 0): width of the non-leaky augmentation-parameter
        vector (``pace.edm_augment``; 9 for EDM's CIFAR-10 config). When 0 the
        embedding module is NOT created at all. When > 0 a bias-free
        ``Linear(augment_dim, hidden_size)`` is added whose output is summed into the
        timestep+class conditioning embedding, mirroring EDM's ``map_augment``
        (``emb = emb + self.map_augment(augment_labels)``). It is initialized
        ``normal_(std=0.02)``: the same init DiT gives its sibling conditioning
        embedders (``t_embedder``, ``y_embedder``), and non-zero like ``map_augment``
        in EDM's ``SongUNet``/ddpmpp -- the architecture EDM actually uses on CIFAR-10.
        (Zero-init would be a poor choice here: DiT is adaLN-ZERO, so ``dL/dc`` is
        already exactly 0 at step 0 for EVERY conditioning embedder, and a zero
        augment embedding would stack a second dead layer on top of that.)
      * ``dropout`` (default 0.0): per-block residual-branch dropout, see
        ``NarrowDiTBlock``.

    ``attn_impl`` (default ``"manual"``, BYTE-IDENTICAL to every model built before
    this option existed) is a THIRD opt-in, model-wide execution-path switch
    (``"manual"`` or ``"sdpa"``) forwarded verbatim to every ``NarrowDiTBlock`` /
    ``NarrowAttention`` -- same weights, same math, no new parameters, no
    ``state_dict`` change. See ``NarrowAttention``'s docstring for the numerics
    (scale/dropout) that make this a pure execution-path change.
    """

    def __init__(
        self,
        hidden_size: int,
        depth: int,
        patch_size: int,
        in_channels: int,
        num_classes: int,
        input_size: int,
        per_block: Sequence[Dict[str, int]],
        learn_sigma: bool = False,
        class_dropout_prob: float = 0.1,
        label_drop_train: bool = False,
        augment_dim: int = 0,
        dropout: float = 0.0,
        attn_impl: str = "manual",
    ):
        super().__init__()
        load_dit_symbols()
        if len(per_block) != depth:
            raise ValueError(f"per_block has {len(per_block)} entries but depth={depth}")
        self.learn_sigma = learn_sigma
        self.augment_dim = int(augment_dim or 0)
        self.dropout = float(dropout or 0.0)
        self.attn_impl = str(attn_impl)
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = per_block[0]["num_heads"]

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        # class_dropout_prob > 0 -> LabelEmbedder table has num_classes + 1 rows,
        # matching the DiTMicro teacher's null-class row (needed by forward_with_cfg's
        # unconditional branch, used by both Micro cfg=2.0 and XL cfg=1.5).
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        # KD-fidelity: this is a distillation student whose KD target is the teacher on
        # the TRUE labels, so nulling ~10% of labels during training would corrupt that
        # fraction of the signal. Keep the null-class row (table = num_classes+1) but
        # never null a label during training: setting dropout_prob=0.0 AFTER construction
        # makes token_drop's `drop_ids = torch.rand(...) < 0.0` all-False (no drop),
        # while leaving the already-allocated null row intact. class_dropout_prob=0 would
        # instead have dropped the null row (use_cfg_embedding = dropout_prob > 0).
        if not label_drop_train:
            self.y_embedder.dropout_prob = 0.0
        # Non-leaky-augmentation conditioning (EDM ``map_augment``). augment_dim == 0
        # (the default) -> NOT created, so parameters and state_dict keys are exactly
        # as they were before this option existed.
        self.aug_embedder = (
            nn.Linear(self.augment_dim, hidden_size, bias=False) if self.augment_dim > 0 else None
        )
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            NarrowDiTBlock(
                hidden_size,
                num_heads=pb["num_heads"],
                attn_inner=pb["attn_inner"],
                mlp_hidden=pb["mlp_hidden"],
                dropout=self.dropout,
                attn_impl=self.attn_impl,
            )
            for pb in per_block
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Identical init to the stock DiT.
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Augment conditioning is a sibling of the t/y conditioning embedders, so it
        # gets their init (see class docstring). Only present when augment_dim > 0,
        # so the default init sequence is untouched.
        if self.aug_embedder is not None:
            nn.init.normal_(self.aug_embedder.weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor,
                augment_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (N, C, H, W); t: (N,) scalar noise; y: (N,) class labels.
        Returns (N, out_channels, H, W). Matches DiTMicro.forward exactly.

        ``augment_labels`` (N, augment_dim) is the non-leaky augmentation-parameter
        vector from ``pace.edm_augment.AugmentPipe``. It is only consumed when
        the model was built with ``augment_dim > 0``; callers that pass nothing (every
        pre-existing call site, and all eval/sampling paths) get exactly the previous
        behaviour."""
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y
        if self.aug_embedder is not None and augment_labels is not None:
            c = c + self.aug_embedder(augment_labels.to(c.dtype))
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x

    def forward_with_cfg(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor,
                         cfg_scale: float) -> torch.Tensor:
        """Classifier-free-guidance forward, copied verbatim from the stock DiT (incl. the
        reference `[:3]`-channel CFG quirk) so composite sampling matches the stock DiT
        teacher. Needed by the DDPM (XL) sampler."""
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


# --- NarrowDiT construction helpers ------------------------------------------

def _uniform_per_block(hidden_size: int, depth: int, num_heads: int, mlp_ratio: float) -> List[Dict[str, int]]:
    """Per-block spec where every block matches a stock uniform DiT of this width."""
    return [
        {"num_heads": num_heads, "attn_inner": hidden_size, "mlp_hidden": int(hidden_size * mlp_ratio)}
        for _ in range(depth)
    ]


def _narrow_dit_kwargs(teacher_cfg: Dict[str, object], hidden_size: int,
                       per_block: List[Dict[str, int]]) -> Dict[str, object]:
    """Full kwargs dict to construct a NarrowDiT, derived from a teacher cfg."""
    return {
        "hidden_size": hidden_size,
        "depth": int(teacher_cfg["depth"]),
        "patch_size": int(teacher_cfg["patch_size"]),
        "in_channels": int(teacher_cfg["in_channels"]),
        "num_classes": int(teacher_cfg["num_classes"]),
        "input_size": int(teacher_cfg["input_size"]),
        "learn_sigma": bool(teacher_cfg["learn_sigma"]),
        "per_block": per_block,
    }


def build_narrow_dit(cfg: Dict[str, object], attn_impl: Optional[str] = None) -> NarrowDiT:
    """Instantiate a NarrowDiT from a cfg dict.

    Accepts either an exact NarrowDiT kwargs dict (as produced by ``build_plan``) or a
    richer cfg dict (e.g. a teacher cfg like ``MICRO_TEACHER_CFG`` carrying extra
    num_heads/mlp_ratio keys): only keys that are NarrowDiT constructor parameters are
    forwarded. ``label_drop_train`` is passed through if present, defaulting to False
    (no train-time label dropout -> full-fidelity KD; the null-class row is still kept).

    ``augment_dim`` / ``dropout`` are likewise forwarded when present in ``cfg`` --
    that is how the trainer's ``--augment_prob`` / ``--net_dropout`` reach the model,
    and how ``arch_cfg.json`` lets eval rebuild the exact same architecture.

    ``attn_impl`` (default ``None``, BYTE-IDENTICAL to every call site before this
    parameter existed) is an explicit CLI-facing override of the attention execution
    path ("manual"/"sdpa"; see ``NarrowDiT``/``NarrowAttention``). ``None`` leaves
    ``cfg`` untouched -- so if ``cfg`` has no ``"attn_impl"`` key (true of every plan
    file, since ``build_plan`` never writes one), ``NarrowDiT`` falls back to its own
    default ("manual"), exactly as before this parameter existed. Passing a string
    overrides (a COPY of) ``cfg`` with that value regardless of what ``cfg`` carries --
    this is how ``--attn_impl`` flags in ``bench_throughput.py`` / ``evaluate_students.py``
    reach every NarrowDiT they build, without mutating the caller's ``cfg`` dict."""
    import inspect
    if attn_impl is not None:
        cfg = dict(cfg)
        cfg["attn_impl"] = attn_impl
    valid = set(inspect.signature(NarrowDiT.__init__).parameters) - {"self"}
    kwargs = {k: v for k, v in cfg.items() if k in valid}
    kwargs.setdefault("label_drop_train", False)
    return NarrowDiT(**kwargs)  # type: ignore[arg-type]


# =============================================================================
# Allocation functions (pure numpy; no torch)
# =============================================================================

def count_dit_params(model: nn.Module) -> int:
    """Total trainable parameters."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def analytic_narrow_dit_params(cfg: Dict[str, object]) -> int:
    """Trainable-parameter count of a NarrowDiT computed from layer shapes ONLY --
    no torch module is instantiated. Used to make the matcher search tractable at
    XL scale (D up to 1152, depth 28): the grid + binary searches evaluate dozens
    of candidate cfgs, and building a real ~100M-700M-param NarrowDiT per candidate
    on CPU is prohibitively slow/memory-heavy.

    Counts exactly what ``count_dit_params`` counts (``requires_grad`` params):

      * x_embedder  PatchEmbed = Conv2d(in_channels, D, k=p, s=p): D*in_channels*p^2 + D
      * t_embedder  Linear(256, D) + Linear(D, D): (256*D + D) + (D*D + D)
      * y_embedder  Embedding(num_classes + 1, D)  [class_dropout_prob>0 -> +1 null row]
      * pos_embed   FROZEN (requires_grad=False) -> EXCLUDED
      * per block   adaLN Linear(D, 6D) = 6D*D + 6D
                    attn  qkv Linear(D, 3A) = 3A*D + 3A ; proj Linear(A, D) = A*D + D
                    mlp   fc1 Linear(D, M) = M*D + M ; fc2 Linear(M, D) = M*D + D
                    norm1/norm2 LayerNorm(affine=False) -> 0
      * final_layer norm_final(affine=False)=0 ; linear Linear(D, p^2*out_ch)=D*p^2*out_ch + p^2*out_ch
                    adaLN Linear(D, 2D) = 2D*D + 2D

    ``per_block`` must be a list of {num_heads, attn_inner (A), mlp_hidden (M)}.
    The frequency-embedding size (256) is the stock DiT ``TimestepEmbedder`` default,
    which NarrowDiT reuses verbatim.
    """
    D = int(cfg["hidden_size"])
    depth = int(cfg["depth"])
    p = int(cfg["patch_size"])
    in_channels = int(cfg["in_channels"])
    num_classes = int(cfg["num_classes"])
    learn_sigma = bool(cfg["learn_sigma"])
    per_block = cfg["per_block"]  # type: ignore[assignment]
    if len(per_block) != depth:  # type: ignore[arg-type]
        raise ValueError(f"per_block has {len(per_block)} entries but depth={depth}")  # type: ignore[arg-type]

    out_channels = in_channels * 2 if learn_sigma else in_channels
    freq = 256  # TimestepEmbedder.frequency_embedding_size default

    # x_embedder (PatchEmbed Conv2d with bias)
    total = D * in_channels * p * p + D
    # t_embedder (two Linears with bias)
    total += (freq * D + D) + (D * D + D)
    # y_embedder (Embedding; class_dropout_prob>0 in NarrowDiT -> +1 null-class row)
    total += (num_classes + 1) * D
    # aug_embedder Linear(augment_dim, D, bias=False) -- ONLY when augment_dim > 0
    # (absent by default, so this term is 0 for every pre-existing plan/cfg).
    total += int(cfg.get("augment_dim", 0) or 0) * D

    for pb in per_block:  # type: ignore[assignment]
        A = int(pb["attn_inner"])
        M = int(pb["mlp_hidden"])
        adaln = 6 * D * D + 6 * D
        attn = (3 * A * D + 3 * A) + (A * D + D)
        mlp = (D * M + M) + (M * D + D)
        total += adaln + attn + mlp

    # final_layer: linear + adaLN Linear(D, 2D) (norm has no affine params)
    total += (D * p * p * out_channels + p * p * out_channels) + (2 * D * D + 2 * D)
    return int(total)


def q90_block_scores(n_eff: Sequence[float], phases: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Per-phase 0.9-quantile of ``n_eff`` over each phase's bin range."""
    n_eff = np.asarray(n_eff, dtype=float)
    return np.array([np.quantile(n_eff[s:e], 0.9) for (s, e) in phases])


# Valid choices for ``block_budgets``'s / ``build_plan``'s phase-budget aggregator.
PHASE_BUDGET_AGGS = ("q90", "geomean", "mean")


def _phase_budget_scores(
    n_eff: Sequence[float], phases: Sequence[Tuple[int, int]], agg: str
) -> np.ndarray:
    """Per-phase aggregate of ``n_eff`` over each phase's bin range.

    ``agg`` selects the aggregator:

      * ``"q90"`` (the default everywhere): delegates to the pre-existing
        ``q90_block_scores`` VERBATIM (same ``np.quantile`` call), so the default
        path is bit-identical to every plan built before ``agg`` existed.
      * ``"geomean"``: ``exp(mean(log(max(x, 1e-9))))`` per phase (geomean and
        mean phase shares were observed to agree within about 1.3pp). The ``1e-9``
        floor guards ``log(0)`` for bins with an exactly-zero measurement.
      * ``"mean"``: arithmetic mean per phase.
    """
    if agg == "q90":
        return q90_block_scores(n_eff, phases)
    arr = np.asarray(n_eff, dtype=float)
    if agg == "geomean":
        return np.array(
            [float(np.exp(np.mean(np.log(np.maximum(arr[s:e], 1e-9))))) for (s, e) in phases]
        )
    if agg == "mean":
        return np.array([float(np.mean(arr[s:e])) for (s, e) in phases])
    raise ValueError(f"unknown phase-budget agg: {agg!r} (choose from {PHASE_BUDGET_AGGS})")


def block_budgets(
    n_eff: Sequence[float],
    phases: Sequence[Tuple[int, int]],
    total_budget: float,
    variant: str,
    alpha: float = 1.0,
    agg: str = "q90",
) -> List[float]:
    """Per-phase target parameter budgets summing to ~total_budget.

    - global               -> [total_budget]  (single phase over the full range)
    - uniform_blockwise    -> [total_budget/N]*N
    - blockwise_capacity /
      layerwise_capacity   -> proportional to agg(n_eff over phase bins)^alpha per phase.

    ``agg`` (default ``"q90"``, UNCHANGED from every plan built before this
    parameter existed -- the q90 path still calls ``q90_block_scores`` /
    ``np.quantile`` verbatim, so default output is bit-identical) selects the
    per-phase aggregator of ``n_eff``; see ``_phase_budget_scores``. ``global``
    and ``uniform_blockwise`` never consult ``n_eff``, so ``agg`` cannot affect
    them (it is validated for typo-safety regardless of variant).
    """
    if agg not in PHASE_BUDGET_AGGS:
        raise ValueError(f"unknown phase-budget agg: {agg!r} (choose from {PHASE_BUDGET_AGGS})")
    if variant == "global":
        return [float(total_budget)]
    n = len(phases)
    if variant == "uniform_blockwise":
        return [float(total_budget) / n] * n
    if variant in ("blockwise_capacity", "layerwise_capacity"):
        score = _phase_budget_scores(n_eff, phases, agg)
        t = np.power(score, alpha)
        s = float(t.sum())
        if s <= 0:
            return [float(total_budget) / n] * n
        return [float(total_budget) * float(tb) / s for tb in t]
    raise ValueError(f"unknown variant: {variant!r}")


def layer_scores(
    relative_delta_stack: Sequence[Sequence[float]],
    group_names: Sequence[str],
    num_layers: int,
    phases: Sequence[Tuple[int, int]],
    eps: float = 0.0,
) -> np.ndarray:
    """Aggregate per-head importance to per-transformer-block scores.

    ``group_names`` look like ``blocks.B.attn.head_H``; we map each row to its
    block B. For each phase and block ``l`` the score is the 0.9-quantile over the
    phase's bins of the summed relative-importance of that block's heads.

    ``eps`` (default 0.0, i.e. UNCHANGED from every plan built before this parameter
    existed -> byte-identical output when re-run with no override) is an opt-in
    per-phase floor: after computing every layer's raw q90 score for a phase, add
    ``eps * mean_over_layers(raw scores in that phase)`` to every layer's score in
    that phase, i.e. ``score'_l = score_l + eps * phase_mean(score)``.

    Why: the permutation-importance measurement clips negative (noise-dominated)
    deltas to exactly 0.0 at ablation time (``evaluate_parameters_*.py``,
    ``clamp(ablated_mean - baseline_mean, min=0.0)``). In a phase where the true
    signal is small relative to measurement noise for EVERY layer (observed on
    DiT-Micro's phase 0, [0,4) bins: 86.5%% of the raw (head, bin)
    entries feeding that phase are exactly clipped, vs. 0-52%% in every other
    phase), some layers draw a positive-noise sample (small nonzero score) and
    others draw a negative-noise sample (clipped to hard 0.0) -- an artifact of
    where each layer's noise happened to land, not a real difference in
    importance. ``match_layerwise_cfg``'s ``sqrt(score_l / mean_score)``
    redistribution treats a hard 0.0 as an absolute, ``g_max``-proof floor (0 * g
    == 0 for every finite g), permanently starving that layer regardless of how
    wide the g-search is. A small ``eps`` (e.g. 0.05) makes the redistribution
    degrade gracefully toward UNIFORM intra-phase allocation when a phase's true
    signal is genuinely at the noise floor, while leaving phases with real
    dynamic range (e.g. Micro phases 2-3, 0%% clipped) essentially untouched,
    since ``eps * phase_mean(score)`` is tiny relative to those layers' own
    scores there.

    Returns array of shape [num_phases, num_layers].
    """
    rds = np.asarray(relative_delta_stack, dtype=float)  # (n_heads, n_bins)
    block_of = []
    for nm in group_names:
        m = _BLOCK_RE.search(nm)
        if not m:
            raise ValueError(f"group name has no block index: {nm!r}")
        block_of.append(int(m.group(1)))
    block_of = np.asarray(block_of)

    # Per-block summed importance across its heads -> (num_layers, n_bins).
    per_block = np.zeros((num_layers, rds.shape[1]), dtype=float)
    for l in range(num_layers):
        rows = rds[block_of == l]
        if rows.size:
            per_block[l] = rows.sum(axis=0)

    out = np.zeros((len(phases), num_layers), dtype=float)
    for p, (s, e) in enumerate(phases):
        for l in range(num_layers):
            out[p, l] = np.quantile(per_block[l, s:e], 0.9)
        if eps:
            out[p] = out[p] + eps * float(out[p].mean())
    return out


def _round_to(value: float, mult: int, min_val: int, max_val: int) -> int:
    """Round ``value`` to the nearest multiple of ``mult``, clamped to [min,max]."""
    r = int(round(value / mult)) * mult
    r = max(min_val, min(max_val, r))
    # Ensure the clamped result is still a multiple of mult (min/max are chosen to be).
    return r


def _lcm(a: int, b: int) -> int:
    return a * b // math.gcd(a, b)


def _uniform_D_grid(teacher_cfg: Dict[str, object], d_mult: Optional[int] = None) -> List[int]:
    """Candidate hidden sizes: multiples of ``lcm(num_heads, 8)`` up to teacher D
    (``d_mult`` overrides the step, e.g. 64 for tensor-core-friendly widths)."""
    D_teacher = int(teacher_cfg["hidden_size"])
    step = int(d_mult) if d_mult else _lcm(int(teacher_cfg["num_heads"]), 8)
    grid = list(range(step, D_teacher + 1, step))
    if D_teacher not in grid:
        grid.append(D_teacher)
    return grid


def _uniform_params_by_D(teacher_cfg: Dict[str, object], d_mult: Optional[int] = None,
                         cost_fn: Optional[Callable[[int, List[Dict[str, int]]], float]] = None,
                         head_dim: Optional[int] = None) -> Dict[int, float]:
    """Uniform NarrowDiT param count for every candidate hidden size D.

    Uses the ANALYTIC count (no torch instantiation) so the D-grid sweep is cheap
    even at XL scale; it is exact vs ``count_dit_params`` (verified in tests and by
    the end-of-plan assertion in ``build_plan``)."""
    depth = int(teacher_cfg["depth"])
    num_heads = int(teacher_cfg["num_heads"])
    mlp_ratio = float(teacher_cfg["mlp_ratio"])
    out: Dict[int, float] = {}
    for D in _uniform_D_grid(teacher_cfg, d_mult):
        pb = _uniform_per_block(D, depth, (D // head_dim) if head_dim else num_heads, mlp_ratio)
        out[D] = cost_fn(D, pb) if cost_fn else analytic_narrow_dit_params(_narrow_dit_kwargs(teacher_cfg, D, pb))
    return out


def _closest_uniform_D(target_params: float, teacher_cfg: Dict[str, object], **kw) -> int:
    """Base hidden size whose uniform NarrowDiT param count is closest to target
    (within 5% if any grid point qualifies, else the strictly closest)."""
    params = _uniform_params_by_D(teacher_cfg, **kw)
    return min(params, key=lambda D: abs(params[D] - target_params))


def _base_D_for_layerwise(phase_budget: float, teacher_cfg: Dict[str, object], **kw) -> int:
    """Base hidden size for layerwise redistribution: the SMALLEST D whose uniform
    param count is >= ``phase_budget`` (so there is downward headroom to trim
    low-importance layers below D while keeping high-importance ones wide -- the
    UNet-style "base big enough to hold the budget, then trim by importance").
    Falls back to the largest D (teacher) when even that undershoots the budget."""
    params = _uniform_params_by_D(teacher_cfg, **kw)
    feasible = [D for D, n in params.items() if n >= phase_budget]
    if feasible:
        return min(feasible)
    return max(params)  # even teacher width undershoots -> use the largest available


def match_uniform_cfg(target_params: float, teacher_cfg: Dict[str, object], *,
                      d_mult: Optional[int] = None, head_dim: Optional[int] = None,
                      cost_fn: Optional[Callable[[int, List[Dict[str, int]]], float]] = None) -> Dict[str, object]:
    """Find the uniform-width NarrowDiT whose param count is closest to
    ``target_params`` (accept a hit within 5%, else the closest).

    Search ``hidden_size`` over multiples of ``lcm(num_heads, 8)`` from that step up
    to the teacher's D, keeping depth/num_heads/mlp_ratio fixed. Returns a full
    NarrowDiT kwargs dict.
    """
    depth = int(teacher_cfg["depth"])
    num_heads = int(teacher_cfg["num_heads"])
    mlp_ratio = float(teacher_cfg["mlp_ratio"])
    best_D = _closest_uniform_D(target_params, teacher_cfg, d_mult=d_mult, cost_fn=cost_fn, head_dim=head_dim)
    pb = _uniform_per_block(best_D, depth, (best_D // head_dim) if head_dim else num_heads, mlp_ratio)
    return _narrow_dit_kwargs(teacher_cfg, best_D, pb)


def _layer_movable_params_uniform(D: int, num_heads: int, mlp_ratio: float) -> int:
    """Parameter count of one teacher block's movable width (attn + mlp only, i.e.
    everything that scales when we shrink attn_inner and mlp_hidden together).

    attn: qkv Linear(D, 3D)=3D*D+3D ; proj Linear(D, D)=D*D+D
    mlp : fc1 Linear(D, 4D)=4D*D+4D ; fc2 Linear(4D, D)=4D*D+D
    (D-fixed pieces -- LayerNorm affine=False, adaLN -- are excluded; they don't move.)
    """
    A = D
    M = int(D * mlp_ratio)
    attn = (3 * A * D + 3 * A) + (A * D + D)
    mlp = (D * M + M) + (M * D + D)
    return attn + mlp


def _realized_layer_movable_params(D: int, A: int, M: int) -> int:
    attn = (3 * A * D + 3 * A) + (A * D + D)
    mlp = (D * M + M) + (M * D + D)
    return attn + mlp


def match_layerwise_cfg(
    phase_budget: float,
    teacher_cfg: Dict[str, object],
    layer_score_vec: Sequence[float],
    g_max: float = 1.75,
    *,
    head_dim: Optional[int] = None,
    d_mult: Optional[int] = None,
    m_mult: int = 8,
    cost_fn: Optional[Callable[[int, List[Dict[str, int]]], float]] = None,
) -> Dict[str, object]:
    """Hit ``phase_budget`` with the SAME overall size as blockwise, but redistribute
    the movable (attn+mlp) capacity *across layers* by importance.

    Two stages, mirroring the UNet width-allocation method (a base ``model_channels``
    picked for the budget, plus a per-module ``width_profile``):

    1. Pick a shared base hidden size ``D`` big enough to hold ``phase_budget`` --
       the smallest grid D whose uniform NarrowDiT param count is >= the budget
       (same D-grid as ``match_uniform_cfg``). This sets the overall size *including*
       the D-fixed adaLN/embedder overhead (so small budgets shrink D rather than
       flooring on that overhead) AND leaves downward headroom so step 2 can trim
       low-importance layers. Base per-layer widths at this D are attn_inner=D,
       mlp_hidden=int(D*mlp_ratio).
    2. Redistribute across the ``depth`` layers by ``sqrt(score_l / mean_score)``
       (higher-importance layer -> wider, lower -> narrower), keeping the residual
       dim ``D`` shared. A global scale ``g`` is binary-searched (<=12 steps, range
       ``[0.05, g_max]``) so the total realized params re-hit ``phase_budget``.

    Depth stays = teacher depth; D = the base D from step 1 (shared across layers so
    the residual stream is consistent).

    ``g_max`` (default 1.75, UNCHANGED from every plan built before this parameter
    existed -- so default-arg callers, e.g. ``dit_arch_to_plans.py`` with no
    ``--layerwise_g_max`` flag, regenerate byte-identical plans) is the upper bound
    of the ``g`` binary search.

    Why this needs to be tunable:
    the per-layer redistribution weight ``sqrt(score_l / mean_score)`` is concave, so
    by Jensen's inequality its layer-sum is systematically BELOW the uniform-D sum
    whenever scores are dispersed across layers (the common case for real importance
    measurements) -- ``g`` must rise above 1.0 to compensate. But ``A``/``M`` are
    clamped at ``D``/``max_M`` (a layer can never exceed its uniform-D width), so once
    enough high-importance layers saturate that ceiling, only the *unsaturated*
    minority can keep absorbing a rising ``g``, and the search can need g far above 1.
    With ``g_max=1.75`` (tuned for small/CIFAR-10-scale score dispersion) this silently
    saturates before reaching ``phase_budget`` for skewed distributions at other
    scales, producing an UNDERSHOOT of up to ~24% (DiT-Micro) / ~20% (latent
    DiT-L/2 scale) with NO error raised -- the only assertion in ``build_plan`` checks the
    analytic formula against a real instantiation, not realized-vs-target budget.
    The U-Net compiler (``pace.edm_distillation.compile_layerwise_student``)
    widens this same bound to 4096 whenever the problem is "large"
    (``label_dim >= 1000 or img_resolution >= 64``); this DiT port does not have
    that scale-adaptivity. Passing a larger ``g_max`` (e.g. 50-100) lets the
    search reach the same ceiling-saturated ceiling as blockwise/uniform, closing
    the gap to ~1-4%; it is NOT the default because that changes ``layerwise_capacity``
    plan output and this repo's hard rule is that new/changed behavior is opt-in.
    """
    depth = int(teacher_cfg["depth"])
    num_heads = int(teacher_cfg["num_heads"])
    mlp_ratio = float(teacher_cfg["mlp_ratio"])

    scores = np.asarray(layer_score_vec, dtype=float)
    if len(scores) != depth:
        raise ValueError(f"layer_score_vec length {len(scores)} != depth {depth}")
    if float(scores.sum()) <= 0:
        scores = np.ones(depth, dtype=float)

    # Stage 1: base D big enough to hold the budget (sets overall size incl. fixed
    # overhead, and leaves downward headroom for importance-weighted trimming).
    # Opt-in knobs (all defaults reproduce the historical plans byte-for-byte):
    #   head_dim : cut attention as WHOLE heads of this dim (num_heads = attn_inner // head_dim) instead
    #              of shrinking the per-head dim at a fixed head count (hd 14-46 -> flash/softmax inefficiency);
    #   d_mult   : hidden-size grid step (64 = tensor-core tiles);  m_mult : mlp_hidden rounding (64);
    #   cost_fn  : cost of a whole model (D, per_block) -> float in the budget's unit -- when given, the
    #              budget is that unit (e.g. measured ms/step at batch 256 from a latency table) instead of params.
    D = _base_D_for_layerwise(phase_budget, teacher_cfg, d_mult=d_mult, cost_fn=cost_fn, head_dim=head_dim)

    # Fixed (D-fixed) params at this D: everything but the movable attn+mlp.
    # Analytic count (no torch instantiation) -- exact vs count_dit_params.
    uniform_pb = _uniform_per_block(D, depth, num_heads, mlp_ratio)
    teacher_total = analytic_narrow_dit_params(_narrow_dit_kwargs(teacher_cfg, D, uniform_pb))
    orig_l = _layer_movable_params_uniform(D, num_heads, mlp_ratio)
    fixed = teacher_total - orig_l * depth

    # Stage 2: per-layer width multiplier from importance (mean-normalized so the
    # average layer keeps ~D width; g then re-centers to hit the budget exactly).
    mean_score = float(scores.mean())
    per_layer_scale = np.sqrt(np.clip(scores / max(mean_score, 1e-12), 0.0, None))  # (depth,)

    max_M = int(D * mlp_ratio)
    if m_mult != 8:
        max_M = max(m_mult, (max_M // m_mult) * m_mult)
    A_step = int(head_dim) if head_dim else _lcm(num_heads, 8)  # whole heads, or a multiple of both num_heads and 8
    A_max = D if not head_dim else max(A_step, (D // A_step) * A_step)

    def realize(g: float) -> Tuple[List[Dict[str, int]], float]:
        pb: List[Dict[str, int]] = []
        realized = fixed
        for l in range(depth):
            s = per_layer_scale[l] * g
            # min = A_step keeps A a valid multiple of both num_heads and 8
            # (the smallest such positive width; still >= num_heads as required).
            A = _round_to(D * s, mult=A_step, min_val=A_step, max_val=A_max)
            M = _round_to(max_M * s, mult=m_mult, min_val=m_mult, max_val=max_M)
            pb.append({"num_heads": (int(A) // int(head_dim)) if head_dim else num_heads,
                       "attn_inner": int(A), "mlp_hidden": int(M)})
            realized += _realized_layer_movable_params(D, int(A), int(M))
        if cost_fn is not None:
            realized = cost_fn(D, pb)
        return pb, realized

    # Binary-search g in [0.05, g_max] so total realized re-hits phase_budget.
    lo, hi = 0.05, float(g_max)
    best_pb, best_err = None, None
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        pb, realized = realize(mid)
        err = abs(realized - phase_budget)
        if best_err is None or err < best_err:
            best_err, best_pb = err, pb
        if realized < phase_budget:
            lo = mid
        else:
            hi = mid
    for g in (lo, hi):
        pb, realized = realize(g)
        err = abs(realized - phase_budget)
        if err < best_err:
            best_err, best_pb = err, pb

    return _narrow_dit_kwargs(teacher_cfg, D, best_pb)


def match_blockwise_budget_cfg(
    phase_budget: float,
    teacher_cfg: Dict[str, object],
    g_max: float = 50.0,
    **knobs,
) -> Dict[str, object]:
    """Budget-aware variant of ``match_uniform_cfg`` for ``blockwise_capacity``.

    ``match_uniform_cfg`` picks a single grid ``D`` (multiples of
    ``lcm(num_heads, 8)``) by closest ABSOLUTE distance to the phase target. At
    DiT-Micro scale that grid has only 8 candidate points and the gap between
    consecutive points grows monotonically with ``D`` (param count is roughly
    quadratic in ``D``), so a phase target that falls inside a wide gap commits to
    whichever end is closest even when that is tens of percent off (observed:
    Micro's blockwise_capacity realizes +21.9%/+17.8%% in its two worst phases).
    There is no per-layer trimming in that path to correct it.

    This function keeps blockwise's defining property -- ONE width shared by every
    layer in the phase, not an importance-driven per-layer split (that is what
    distinguishes it from ``layerwise_capacity``) -- but picks that shared width at
    a MUCH finer grain than the 8-point D-grid: it picks a headroom base ``D`` (the
    same "smallest grid D whose uniform count >= budget" rule
    ``match_layerwise_cfg`` already uses) and then trims ``attn_inner``/
    ``mlp_hidden`` UNIFORMLY across every layer via the same ``g``-search
    ``match_layerwise_cfg`` uses for ``layerwise_capacity`` -- just fed a FLAT
    (all-equal) per-layer score vector, so every layer gets the identical scale and
    therefore the identical (attn_inner, mlp_hidden), same as a true uniform-width
    plan. This is a thin, deliberately minimal reuse of already-tested machinery
    (no new search logic): ``match_layerwise_cfg`` with ``scores = ones(depth)``
    reduces exactly to "pick base D, then scale every layer's width by the same g
    to re-hit budget," since ``per_layer_scale = sqrt(score_l / mean_score) ==
    sqrt(1/1) == 1`` for every layer when all scores are equal.
    """
    depth = int(teacher_cfg["depth"])
    flat_scores = np.ones(depth, dtype=float)
    return match_layerwise_cfg(phase_budget, teacher_cfg, flat_scores, g_max=g_max, **knobs)


def build_plan(
    results_dict: Dict,
    phases: Sequence[Tuple[int, int]],
    teacher_cfg: Dict[str, object],
    variant: str,
    alpha: float = 1.0,
    layerwise_g_max: float = 1.75,
    layer_score_eps: float = 0.0,
    blockwise_budget_match: bool = False,
    blockwise_budget_g_max: float = 50.0,
    uniform_budget_match: bool = False,
    phase_budget_agg: str = "q90",
    head_dim: Optional[int] = None,
    d_mult: Optional[int] = None,
    m_mult: int = 8,
    cost_fn: Optional[Callable[[int, List[Dict[str, int]]], float]] = None,
    cost_target: Optional[float] = None,
    phase_weights: Optional[Sequence[float]] = None,
) -> Dict:
    """Build a per-phase architecture plan for one allocation variant.

    Opt-in knobs (defaults byte-identical): ``head_dim``/``d_mult``/``m_mult`` are forwarded to
    the matchers (whole-head attention cuts, 64-multiple hidden/mlp). ``cost_fn(D, per_block)`` switches the
    budget unit from parameters to that cost (e.g. measured ms/step from a latency table); then
    ``cost_target`` is the target COMPOSITE cost per sampling step (sum_p w_p * cost_p, ``phase_weights`` =
    fraction of sampler steps each phase serves; default bin-width fractions) and the per-phase budgets keep
    the variant's proportional split. Realized cost is recorded per phase as ``realized_cost``.

    Returns::

        { "teacher_params": int,
          "phases": [ {"phase": i, "bins": [s, e], "target_params": float,
                       "realized_params": int, "cfg": <NarrowDiT kwargs>} ... ] }

    ``layerwise_g_max`` (default 1.75, matching every plan built before this
    parameter existed) is forwarded to ``match_layerwise_cfg`` for the
    ``layerwise_capacity`` variant only; see that function's docstring for why a
    larger value (e.g. 50-100) may be needed to keep ``layerwise_capacity``
    parameter-matched to ``blockwise_capacity``/``uniform_blockwise`` at scales
    where the per-layer importance signal is skewed.

    ``layer_score_eps`` (default 0.0, byte-identical to every plan built before
    this parameter existed) is forwarded to ``layer_scores()`` for the
    ``layerwise_capacity`` variant only -- see that function's docstring
    (an opt-in per-phase floor so
    a phase whose true importance signal is at the measurement noise floor
    degrades toward uniform intra-phase allocation instead of permanently
    starving whichever layers happened to draw a negative-noise (clipped-to-zero)
    sample).

    ``blockwise_budget_match`` (default False, byte-identical) and
    ``blockwise_budget_g_max`` (default 50.0, only used when the former is True)
    switch the ``blockwise_capacity`` variant from ``match_uniform_cfg`` (a single
    grid D chosen by closest absolute distance -- can overshoot a phase target by
    double-digit percentages when the target falls in a wide grid gap, see
    ``match_blockwise_budget_cfg``'s docstring) to
    ``match_blockwise_budget_cfg`` (same "one width per phase" semantics, but able
    to hit the budget far more precisely by trimming that shared width below the
    grid ceiling). ``uniform_blockwise`` is NEVER affected by ``blockwise_budget_match``
    (it keeps calling ``match_uniform_cfg`` unconditionally when
    ``uniform_budget_match`` is not separately set) -- only ``blockwise_capacity``.

    ``uniform_budget_match`` (default False, byte-identical) is the SAME fix
    applied to ``uniform_blockwise`` instead, as its own independent opt-in flag
    (never bundled with ``blockwise_budget_match``, so the existing invariant
    "uniform_blockwise is never affected by blockwise_budget_match" stays exactly
    true). Needed because the D-grid gap ``match_blockwise_budget_cfg`` fixes for
    ``blockwise_capacity`` is a property of the teacher's (hidden_size, num_heads)
    grid, not of which variant hits it -- at some (teacher, grouping) combinations
    ``uniform_blockwise``'s own target can land in just as wide a gap (e.g. the
    grid step of the CIFAR-10 DiT teacher, D=384 with 6 heads, is 24 up to D=384: a
    2-phase 50/50 split's target sits between D=264 (94.8%) and D=288 (112.7%), an
    18-point gap wider than a +/-3%% budget tolerance allows) -- reusing
    ``match_blockwise_budget_cfg`` here preserves ``uniform_blockwise``'s defining
    property (one identical width for every layer in the phase; still NOT
    importance-weighted per-layer) while removing the coarse-grid artifact,
    exactly mirroring why the same fix was built for ``blockwise_capacity``.

    ``phase_budget_agg`` (default ``"q90"``, bit-identical to every plan built
    before this parameter existed -- the default path still routes through
    ``q90_block_scores``/``np.quantile`` verbatim) selects how ``block_budgets``
    aggregates ``n_eff`` over each phase's bins before the ``^alpha`` split:
    ``"q90"``, ``"geomean"`` (``exp(mean(log(max(x, 1e-9))))``; geomean was
    observed to agree with mean within about 1.3pp phase share) or
    ``"mean"``. Only ``blockwise_capacity``/``layerwise_capacity`` consult
    ``n_eff``, so ``global``/``uniform_blockwise`` budgets are never affected.

    Provenance: when (and ONLY when) ``phase_budget_agg`` is non-default, the
    returned plan dict carries a top-level ``"phase_budget_agg": "<value>"`` key.
    Keying it only on non-default keeps default rebuilds of existing plan files
    byte-identical in JSON structure (no new key appears anywhere unless the
    caller opted in), which is this repo's hard opt-in rule.
    """
    n_eff = results_dict["n_eff"]
    group_names = results_dict["group_names"]
    relative_delta_stack = results_dict["relative_delta_stack"]
    depth = int(teacher_cfg["depth"])
    D = int(teacher_cfg["hidden_size"])
    num_heads = int(teacher_cfg["num_heads"])
    mlp_ratio = float(teacher_cfg["mlp_ratio"])

    uniform_pb = _uniform_per_block(D, depth, num_heads, mlp_ratio)
    teacher_kwargs = _narrow_dit_kwargs(teacher_cfg, D, uniform_pb)
    # Analytic teacher param count for the budget (exact vs count_dit_params; the
    # real-instantiation cross-check is done once at the end of this function).
    teacher_params = analytic_narrow_dit_params(teacher_kwargs)

    total_budget = float(teacher_params)

    n_bins = len(n_eff)
    if variant == "global":
        phase_ranges: List[Tuple[int, int]] = [(0, n_bins)]
    else:
        phase_ranges = [tuple(p) for p in phases]

    knobs = {}
    if head_dim:
        knobs["head_dim"] = int(head_dim)
    if d_mult:
        knobs["d_mult"] = int(d_mult)
    if m_mult != 8:
        knobs["m_mult"] = int(m_mult)
    if cost_fn is not None:
        knobs["cost_fn"] = cost_fn
        if cost_target is None:
            raise ValueError("cost_fn requires cost_target (composite cost per sampling step)")
        shares = block_budgets(n_eff, phase_ranges, 1.0, variant, alpha=alpha, agg=phase_budget_agg)
        if phase_weights is None:
            phase_weights = [(e - s0) / n_bins for (s0, e) in phase_ranges]
        comp = sum(w * sh for w, sh in zip(phase_weights, shares))
        # floor-aware: cost_p = c0 + k*share_p with c0 = smallest cost the grid can realize (a tiny phase
        # cannot go below the fixed embedder/launch overhead), k chosen so sum_p w_p cost_p == cost_target.
        c0 = float(getattr(cost_fn, "floor", 0.0))
        if cost_target <= c0:
            raise ValueError(f"cost_target {cost_target} <= grid floor {c0}")
        k = (float(cost_target) - c0) / comp
        budgets = [c0 + k * sh for sh in shares]
    else:
        budgets = block_budgets(n_eff, phase_ranges, total_budget, variant, alpha=alpha,
                                agg=phase_budget_agg)
    # match_uniform_cfg has no m_mult (a uniform width never rounds mlp_hidden separately).
    uniform_knobs = {k: v for k, v in knobs.items() if k in ("d_mult", "head_dim", "cost_fn")}

    layer_score_mat = None
    if variant == "layerwise_capacity":
        layer_score_mat = layer_scores(relative_delta_stack, group_names, depth, phase_ranges,
                                        eps=layer_score_eps)

    out_phases = []
    for i, (rng, budget) in enumerate(zip(phase_ranges, budgets)):
        if variant == "global":
            cfg = dict(teacher_kwargs)
        elif variant in ("uniform_blockwise",):
            if uniform_budget_match:
                cfg = match_blockwise_budget_cfg(budget, teacher_cfg, g_max=blockwise_budget_g_max, **knobs)
            else:
                cfg = match_uniform_cfg(budget, teacher_cfg, **uniform_knobs)
        elif variant == "blockwise_capacity":
            if blockwise_budget_match:
                cfg = match_blockwise_budget_cfg(budget, teacher_cfg, g_max=blockwise_budget_g_max, **knobs)
            else:
                cfg = match_uniform_cfg(budget, teacher_cfg, **uniform_knobs)
        elif variant == "layerwise_capacity":
            cfg = match_layerwise_cfg(budget, teacher_cfg, layer_score_mat[i], g_max=layerwise_g_max, **knobs)
        else:
            raise ValueError(f"unknown variant: {variant!r}")

        realized = analytic_narrow_dit_params(cfg)
        rec = {
            "phase": i,
            "bins": [int(rng[0]), int(rng[1])],
            "target_params": float(budget),
            "realized_params": int(realized),
            "cfg": cfg,
        }
        if cost_fn is not None:   # budget was a cost (e.g. ms/step), not params: record both
            rec["target_cost"] = float(budget)
            rec["realized_cost"] = float(cost_fn(int(cfg["hidden_size"]), cfg["per_block"]))
            rec["target_params"] = None
        out_phases.append(rec)

    # Verification: instantiate the real NarrowDiT ONCE per chosen cfg and confirm
    # its count_dit_params matches the analytic estimate used throughout the search.
    # (The whole search above is analytic -- cheap at XL scale; this is the only
    # place a real model is built, and it must agree exactly with the analytic count.)
    for ph in out_phases:
        real = count_dit_params(NarrowDiT(**ph["cfg"]))  # type: ignore[arg-type]
        ana = int(ph["realized_params"])
        if abs(real - ana) > max(1, int(0.001 * real)):
            raise AssertionError(
                f"analytic param count {ana:,} disagrees with count_dit_params "
                f"{real:,} for phase {ph['phase']} cfg (variant); "
                f"diff={abs(real - ana):,}"
            )
        # Record the exact real count (identical to analytic; keeps prior behavior
        # where realized_params == count_dit_params of the built model).
        ph["realized_params"] = int(real)

    out = {"teacher_params": int(teacher_params), "phases": out_phases}
    # Provenance annotation, added ONLY when the aggregator is non-default: a
    # default rebuild must stay byte-identical to every existing plans.json --
    # including its JSON structure -- so the key must not exist at all on the
    # default path (an always-present "phase_budget_agg": "q90" would change
    # every default rebuild's bytes, violating the repo's opt-in hard rule).
    if phase_budget_agg != "q90":
        out["phase_budget_agg"] = str(phase_budget_agg)
    return out


def latency_cost_fn_from_tables(glob_pattern: str, key: str = "lat_ms_compiled_b256",
                                depth: int = 12) -> Callable[[int, List[Dict[str, int]]], float]:
    """Build ``cost_fn(D, per_block) -> ms/step`` from measured latency grids: JSON files of records
    with ``hidden``/``heads``/``mlp`` fields plus latency columns (``key``), measured on ``depth``-block
    uniform NarrowDiTs at a fixed batch. Model latency ~= sum over layers of table(D, heads_l, mlp_l)/depth:
    exact for uniform widths, first-order for layerwise plans (embedder/final-layer overhead folded in
    proportionally). Exact hidden match required (use --d_mult on the grid step); linear interpolation
    (and extrapolation) in heads and in mlp_hidden."""
    import glob as _glob
    table: Dict[int, Dict[int, Dict[int, float]]] = {}
    for f in _glob.glob(glob_pattern):
        recs = json.load(open(f))
        recs = recs.values() if isinstance(recs, dict) else recs
        for rec in recs:
            if not isinstance(rec, dict) or key not in rec:
                continue
            by_heads = table.setdefault(int(rec["hidden"]), {}).setdefault(int(rec["heads"]), {})
            by_heads[int(rec["mlp"])] = float(rec[key])
    if not table:
        raise FileNotFoundError(f"no latency records with key {key!r} match {glob_pattern!r}")

    def _interp(pts: Dict[int, float], x: int) -> float:
        if x in pts:
            return pts[x]
        ks = sorted(pts)
        if len(ks) == 1:
            return pts[ks[0]]
        lo = max([k for k in ks if k < x], default=None)
        hi = min([k for k in ks if k > x], default=None)
        a, b = (ks[0], ks[1]) if lo is None else (ks[-2], ks[-1]) if hi is None else (lo, hi)
        return pts[a] + (pts[b] - pts[a]) * (x - a) / (b - a)

    def layer_ms(D: int, heads: int, mlp: int) -> float:
        # linear in hidden too (extrapolated from the two nearest grid points outside the table range),
        # so a D-grid step below/between the measured hiddens still gets a cost instead of a KeyError
        by_hidden = {H: _interp({h: _interp(m, mlp) for h, m in hs.items()}, heads) for H, hs in table.items()}
        return max(_interp(by_hidden, D), 1e-3) / depth

    def cost_fn(D: int, per_block: List[Dict[str, int]]) -> float:
        return float(sum(layer_ms(int(D), int(b["num_heads"]), int(b["mlp_hidden"])) for b in per_block))
    cost_fn.floor = min(v for hs in table.values() for m in hs.values() for v in m.values())  # smallest measured model
    return cost_fn
