"""Karras/EDM regularizers: non-leaky AugmentPipe + augment conditioning + dropout.

Two invariants dominate this file, because both new knobs are opt-in and the repo
rule is that every pre-existing study must stay byte-identical:

  1. DEFAULTS ARE INERT: a default-constructed NarrowDiT has the same parameter
     count, the same state_dict keys, and the same forward output as before these
     options existed, so checkpoints trained before them still load with
     strict=True.
  2. --augment_prob 0 consumes ZERO RNG (no pipe is even built), so the data /
     noise / label-drop RNG streams are untouched.
"""
import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from pace.dit_arch_alloc import (  # noqa: E402
    MICRO_TEACHER_CFG,
    NarrowDiT,
    _narrow_dit_kwargs,
    _uniform_per_block,
    analytic_narrow_dit_params,
    build_narrow_dit,
    count_dit_params,
)
from pace.edm_augment import (  # noqa: E402
    CIFAR10_AUGMENT_DIM,
    AugmentPipe,
    build_augment_pipe,
    cifar10_augment_pipe,
)

# The DiT-Micro "global" plan cfg (D=192/depth8) -- the teacher_v3/v4 architecture.
D = int(MICRO_TEACHER_CFG["hidden_size"])
DEPTH = int(MICRO_TEACHER_CFG["depth"])
MICRO_CFG = _narrow_dit_kwargs(
    MICRO_TEACHER_CFG, D,
    _uniform_per_block(D, DEPTH, int(MICRO_TEACHER_CFG["num_heads"]),
                       float(MICRO_TEACHER_CFG["mlp_ratio"])),
)
# Hard external anchor: the realized_params of the DiT-Micro `global` plan and of
# the teacher checkpoints trained with it.
MICRO_PARAMS = 5_498_892



def _build(seed=0, **kw):
    torch.manual_seed(seed)
    return NarrowDiT(**{**MICRO_CFG, **kw})


def _inputs(n=4, seed=1234):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 32, 32, generator=g)
    t = torch.rand(n, generator=g) * 2 - 1
    y = torch.randint(0, 10, (n,), generator=g)
    return x, t, y


def _untrain_zeros(m, seed=5):
    """Emulate a model that has taken at least one step.

    DiT is adaLN-ZERO: at init ``final_layer.linear`` and every ``adaLN_modulation``
    projection are exactly zero, so the network outputs identical ZEROS for any input
    and no conditioning effect (nor dropout, since dropout of zeros is zeros) is
    observable. Filling those with small random values makes the network a real
    function of its conditioning, which is what the behavioural tests below need.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        mods = [b.adaLN_modulation[-1] for b in m.blocks]
        mods += [m.final_layer.adaLN_modulation[-1], m.final_layer.linear]
        for lin in mods:
            lin.weight.normal_(0.0, 0.02, generator=g)
            lin.bias.normal_(0.0, 0.02, generator=g)
    return m


# ---------------------------------------------------------------------------
# 1. Defaults are byte-identical (params / state_dict / forward).
# ---------------------------------------------------------------------------

@pytest.mark.external_dit
def test_default_params_and_state_dict_unchanged():
    m = _build()
    sd = m.state_dict()
    # Exact param count still equals the committed plan's realized_params: no new
    # parameter can have crept into the default architecture.
    assert count_dit_params(m) == MICRO_PARAMS
    assert analytic_narrow_dit_params(MICRO_CFG) == MICRO_PARAMS
    # No augment / dropout state anywhere (nn.Dropout is param- and buffer-free).
    assert not [k for k in sd if "aug" in k or "drop" in k]
    # Explicitly passing the new options at their defaults is the same model.
    m2 = _build(augment_dim=0, dropout=0.0)
    assert list(sd.keys()) == list(m2.state_dict().keys())
    for k in sd:
        assert torch.equal(sd[k], m2.state_dict()[k]), k
    assert m.aug_embedder is None and m2.aug_embedder is None


@pytest.mark.external_dit
def test_default_forward_identical_and_ignores_augment_labels():
    m = _build().eval()
    x, t, y = _inputs()
    with torch.no_grad():
        ref = m(x, t, y)
        # Passing the new kwarg to a model built WITHOUT augment_dim is a no-op,
        # not an error -- existing call sites and eval paths are unaffected.
        also = m(x, t, y, augment_labels=torch.randn(x.shape[0], CIFAR10_AUGMENT_DIM))
        explicit_defaults = _build(augment_dim=0, dropout=0.0).eval()(x, t, y)
    assert torch.equal(ref, also)
    assert torch.equal(ref, explicit_defaults)


# ---------------------------------------------------------------------------
# 2. Augment conditioning (augment_dim > 0).
# ---------------------------------------------------------------------------

@pytest.mark.external_dit
def test_augment_dim_adds_exactly_one_linear():
    base = _build()
    aug = _build(augment_dim=CIFAR10_AUGMENT_DIM)
    added = set(aug.state_dict()) - set(base.state_dict())
    assert added == {"aug_embedder.weight"}
    assert not set(base.state_dict()) - set(aug.state_dict())
    # Exactly augment_dim * D new parameters (bias-free Linear), and the analytic
    # counter agrees with the real module.
    assert count_dit_params(aug) - count_dit_params(base) == CIFAR10_AUGMENT_DIM * D
    cfg_aug = {**MICRO_CFG, "augment_dim": CIFAR10_AUGMENT_DIM}
    assert analytic_narrow_dit_params(cfg_aug) == count_dit_params(aug)
    # Initialized like DiT's sibling conditioning embedders (normal, std=0.02),
    # NOT zero -- see NarrowDiT's docstring.
    w = aug.aug_embedder.weight
    assert w.shape == (D, CIFAR10_AUGMENT_DIM)
    assert torch.count_nonzero(w) == w.numel()
    assert 0.005 < float(w.std()) < 0.05


@pytest.mark.external_dit
def test_augment_conditioning_changes_the_prediction():
    m = _untrain_zeros(_build(augment_dim=CIFAR10_AUGMENT_DIM)).eval()
    x, t, y = _inputs()
    al = torch.randn(x.shape[0], CIFAR10_AUGMENT_DIM)
    with torch.no_grad():
        assert not torch.allclose(m(x, t, y), m(x, t, y, augment_labels=al))
        # Different augmentation parameters -> different prediction. That IS the
        # non-leakiness mechanism: the net can tell the augmentations apart.
        assert not torch.allclose(m(x, t, y, augment_labels=al),
                                  m(x, t, y, augment_labels=-al))


@pytest.mark.external_dit
def test_augment_embedding_is_in_the_graph_and_trains():
    m = _build(augment_dim=CIFAR10_AUGMENT_DIM)
    x, t, y = _inputs()
    al = torch.randn(x.shape[0], CIFAR10_AUGMENT_DIM)
    target = torch.randn_like(x)
    opt = torch.optim.SGD(m.parameters(), lr=0.5)

    before = m.aug_embedder.weight.detach().clone()
    nnz = []
    for _ in range(3):
        torch.nn.functional.mse_loss(m(x, t, y, augment_labels=al), target).backward()
        # The gradient must EXIST from the very first step: that is what DDP requires
        # (a parameter that never receives a gradient is reported as unused and
        # crashes DDP), and it is why the augment path must be fed every batch.
        assert m.aug_embedder.weight.grad is not None
        nnz.append(int(torch.count_nonzero(m.aug_embedder.weight.grad)))
        opt.step()
        opt.zero_grad()

    # adaLN-ZERO puts TWO zero-init projections in series between c and the output
    # (block/final adaLN, then final_layer.linear), so dL/dc -- and hence the
    # gradient of EVERY conditioning embedder, t_embedder and y_embedder included --
    # is numerically zero for the first two steps and becomes non-zero on the third.
    assert nnz[0] == 0 and nnz[-1] > 0, nnz
    assert not torch.equal(before, m.aug_embedder.weight.detach())


# ---------------------------------------------------------------------------
# 3. The AugmentPipe itself.
# ---------------------------------------------------------------------------

def test_pipe_off_is_identity_and_consumes_no_rng():
    # build_augment_pipe is what the trainer calls: p<=0 -> no pipe at all, so the
    # batch loop never touches the RNG (existing runs byte-identical).
    assert build_augment_pipe(0.0) is None
    assert build_augment_pipe(None) is None
    assert build_augment_pipe(0.12) is not None
    # And an all-multipliers-zero pipe is a literal identity that draws nothing.
    x = torch.randn(4, 3, 32, 32)
    state = torch.get_rng_state()
    out, labels = AugmentPipe()(x)
    assert torch.equal(torch.get_rng_state(), state)
    assert torch.equal(out, x)
    assert labels.shape == (4, 0)
    assert AugmentPipe().label_dim == 0


def test_cifar10_pipe_label_dim_is_nine():
    pipe = cifar10_augment_pipe(0.12)
    # EDM sets network_kwargs.augment_dim = 9 for this exact configuration.
    assert pipe.label_dim == CIFAR10_AUGMENT_DIM == 9
    x = torch.randn(8, 3, 32, 32).clamp(-1, 1)
    out, labels = pipe(x)
    assert out.shape == x.shape
    assert labels.shape == (8, 9) and labels.dtype == torch.float32
    assert torch.isfinite(out).all() and torch.isfinite(labels).all()


def test_pipe_is_deterministic_under_a_fixed_seed():
    pipe = cifar10_augment_pipe(0.5)
    x = torch.randn(8, 3, 32, 32).clamp(-1, 1)
    torch.manual_seed(99)
    o1, l1 = pipe(x)
    torch.manual_seed(99)
    o2, l2 = pipe(x)
    assert torch.equal(o1, o2) and torch.equal(l1, l2)
    # Different seed -> different draw (the pipe really is stochastic).
    torch.manual_seed(100)
    o3, l3 = pipe(x)
    assert not torch.equal(l1, l3)


def test_label_vector_matches_the_applied_transform():
    x = torch.randn(16, 3, 32, 32)
    # x-flip alone, fired with probability 1: label 1 <=> the image WAS flipped.
    torch.manual_seed(7)
    out, labels = AugmentPipe(p=1.0, xflip=1)(x)
    assert labels.shape == (16, 1)
    w = labels[:, 0]
    assert (w == 1).any() and (w == 0).any()
    for i in range(x.shape[0]):
        expect = x[i].flip(-1) if w[i] == 1 else x[i]
        assert torch.equal(out[i], expect), i
    # 90-degree (integer) rotation: the two labels encode the flip pair applied.
    torch.manual_seed(3)
    out_r, labels_r = AugmentPipe(p=1.0, rotate_int=1)(x)
    assert labels_r.shape == (16, 2)
    for i in range(x.shape[0]):
        a, b = float(labels_r[i, 0]), float(labels_r[i, 1])
        expect = x[i]
        if a:
            expect = expect.flip(-1)
        if b:
            expect = expect.flip(-2)
        if bool(a) != bool(b):          # w in {1, 3} -> also transposed
            expect = expect.transpose(-2, -1)
        assert torch.equal(out_r[i], expect), i


def test_geometric_branch_with_zero_parameters_reconstructs_the_image():
    """The geometric resampler (reflect-pad -> sym6 upsample -> grid_sample ->
    sym6 downsample -> crop) executes whenever any geometric augmentation is
    enabled, even for samples whose drawn parameters are all zero. With p ~ 0 the
    transform matrix is the identity, so the round-trip must return the input --
    this is the end-to-end check on all the coordinate bookkeeping."""
    x = torch.randn(4, 3, 32, 32).clamp(-1, 1)
    pipe = AugmentPipe(p=1e-30, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
    out, labels = pipe(x)
    assert bool((labels == 0).all())
    assert out.shape == x.shape
    assert (out - x).abs().max() < 1e-4


# ---------------------------------------------------------------------------
# 4. Dropout.
# ---------------------------------------------------------------------------

@pytest.mark.external_dit
def test_dropout_changes_nothing_structural():
    base = _build()
    drop = _build(dropout=0.13)
    assert list(base.state_dict().keys()) == list(drop.state_dict().keys())
    assert count_dit_params(base) == count_dit_params(drop) == MICRO_PARAMS
    assert drop.blocks[0].attn.proj_drop.p == 0.13
    assert base.blocks[0].attn.proj_drop.p == 0.0


@pytest.mark.external_dit
def test_dropout_active_in_train_and_inert_in_eval():
    m = _untrain_zeros(_build(dropout=0.5))
    x, t, y = _inputs()
    m.train()
    torch.manual_seed(0)
    with torch.no_grad():
        a = m(x, t, y)
        b = m(x, t, y)
    assert not torch.equal(a, b), "dropout must be stochastic in train()"

    m.eval()
    with torch.no_grad():
        c = m(x, t, y)
        d = m(x, t, y)
    assert torch.equal(c, d), "dropout must be inert in eval()"

    # In eval() the dropout model is EXACTLY the same function as a p=0 model
    # holding the same weights -> sampling/FID is unaffected by the flag.
    plain = _build(dropout=0.0)
    plain.load_state_dict(m.state_dict(), strict=True)
    plain.eval()
    with torch.no_grad():
        assert torch.equal(c, plain(x, t, y))


@pytest.mark.external_dit
def test_zero_dropout_consumes_no_rng_in_train_mode():
    """nn.Dropout(0.0) is an exact identity that draws no mask, which is why the
    default can keep existing runs byte-identical without special-casing."""
    m = _untrain_zeros(_build(dropout=0.0)).train()
    x, t, y = _inputs()
    with torch.no_grad():
        state = torch.get_rng_state()
        out = m(x, t, y)
        assert torch.equal(torch.get_rng_state(), state)
        m.eval()
        assert torch.equal(out, m(x, t, y))


# ---------------------------------------------------------------------------
# 5. Trainer-level flag enforcement.
# ---------------------------------------------------------------------------

_BASE_ARGV = [
    "train_phase_students.py",
    "--model_type", "dit_micro",
    "--teacher_checkpoint", "/nonexistent.pt",
    "--output_dir", "/tmp/_unused_augment_test",
    "--dataset", "cifar10",
    "--diffusion", "edm",
    "--arch_plan", "/nonexistent_plan.json",
    "--variant", "global",
    "--kd_weight", "0",
]


def _main_with(extra):
    import train_phase_students
    argv = sys.argv
    sys.argv = _BASE_ARGV + extra
    try:
        with pytest.raises(SystemExit) as ei:
            train_phase_students.main()
        return str(ei.value)
    finally:
        sys.argv = argv


def test_augment_prob_rejects_hflip_and_kd():
    # The pipe's x-flip already fires with probability 1 and is recorded in
    # augment_labels; a second dataset-level flip would be unrecorded (leaky).
    assert "mutually exclusive with --hflip" in _main_with(
        ["--augment_prob", "0.12", "--hflip"])
    # A stock-DiT teacher cannot be told which augmentation was applied.
    assert "requires --kd_weight 0" in _main_with(
        ["--augment_prob", "0.12", "--kd_weight", "1.0"])
    assert "only wired for --model_type dit_micro" in _main_with(
        ["--augment_prob", "0.12", "--diffusion", "ddpm"])
    assert "must be in [0, 1]" in _main_with(["--augment_prob", "1.5"])
    assert "must be in [0, 1)" in _main_with(["--net_dropout", "1.0"])
