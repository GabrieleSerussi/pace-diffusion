"""Tests for --grad_accum (train_phase_students.py), CPU-only.

Covers the four contract points of the flag:

  1. DEFAULT-OFF REGRESSION: at grad_accum == 1 (explicit, or the attribute
     entirely ABSENT from args, as in every pre-flag caller) train_phase's
     parameter trajectory is bit-identical (torch.equal) to a manual reference
     loop that re-implements the documented PRE-CHANGE algorithm
     (zero_grad -> backward(unscaled loss) -> clip -> opt.step -> sched.step
     per batch). We cannot run the literal old code, so this is the honest
     equivalent: the reference is written to the historical spec and consumes
     the identical global-RNG stream, so any extra/reordered FP op or RNG draw
     in the accum-1 path would break exact equality.
  2. EQUIVALENCE: 1 optimizer step at batch 32 vs grad_accum=4 x batch 8 on
     the SAME effective data gives allclose parameter deltas (fp32
     tolerances). The two runs are fed identical effective batches by
     patching the sigma/t draw and torch.randn_like to serve consecutive
     slices of one pre-generated pool (the natural global-RNG streams would
     otherwise interleave differently between the two runs). Exercised for
     the dit_micro/EDM family.
  3. STEP SEMANTICS: with grad_accum=4 an epoch of M microbatches yields
     floor(M/4) optimizer steps (tail dropped, mirroring drop_last=True);
     the `step` stored in full checkpoints counts optimizer steps; ckpt_every
     fires per optimizer step (not per microbatch); the WSD scheduler's LR
     history matches the batch-equivalent (batch_size*4, grad_accum=1) run
     exactly.
  4. EMA: ema_update is called once per OPTIMIZER step, not per microbatch.

The --grad_accum arg itself is asserted present (default 1) in parse_args's
namespace, which is what main() dumps verbatim into run_config.json
(save_json(..., {"args": vars(args), ...})).
"""

import copy
import math
import os
import sys
import types

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import train_phase_students as tps  # noqa: E402
from train_phase_students import train_phase  # noqa: E402


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class TinyNet(nn.Module):
    """Few-thousand-param stand-in with the trainer's forward(x, t, y) calling
    convention (t and y ignored; the EDM branch calls model(x, t, y))."""

    def __init__(self, ch=3, hidden=16):
        super().__init__()
        self.c1 = nn.Conv2d(ch, hidden, 3, padding=1)
        self.c2 = nn.Conv2d(hidden, ch, 3, padding=1)

    def forward(self, x, t, y):
        return self.c2(torch.tanh(self.c1(x)))


def _make_args(*, num_epochs=1, diffusion="edm", model_type="dit_micro", **overrides):
    a = types.SimpleNamespace(
        model_type=model_type,
        diffusion=diffusion,
        edm_loss_space="f",
        num_epochs=num_epochs,
        lr=1e-4,
        weight_decay=0.01,
        lr_min_ratio=0.01,
        grad_clip=1.0,
        gt_weight=1.0,
        kd_weight=0.0,
        sigma_min=0.002,
        sigma_max=80.0,
        sigma_data=0.5,
        rho=7.0,
        num_timesteps=1000,
        log_every=50,
        device="cpu",
        ckpt_every=0,
        wandb_project=None,
        wandb_run_name=None,
    )
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


def _loader(images, labels, batch_size):
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size, shuffle=False)


_PHASE = {"index": 0, "start": 0, "end": 20}
_NUM_BINS = 20
_ALPHA_BAR = torch.linspace(0.99, 0.01, 1000)


def _run_train_phase(model, loader, args, tmp_path, sub):
    d = os.path.join(str(tmp_path), sub)
    os.makedirs(d, exist_ok=True)
    return train_phase(
        args=args, teacher=None, student_ddp=model, raw_student=model,
        vae=None, dataloader=loader, sampler=None, phase=_PHASE,
        num_bins=_NUM_BINS, alpha_bar=_ALPHA_BAR, in_channels=3,
        compute_dtype=torch.float32, phase_dir=d, use_wandb=False,
        resume_ckpt=None,
    )


class _SlicePool:
    """Serves consecutive slices of pre-generated (sigma-or-t, noise) pools so a
    batch-32 run and a 4x-batch-8 run see the SAME effective global batch."""

    def __init__(self, timesteps, noise):
        self.timesteps, self.noise = timesteps, noise
        self.reset()

    def reset(self):
        self._tc = 0
        self._nc = 0

    def next_timesteps(self, batch_size):
        s = self.timesteps[self._tc:self._tc + batch_size]
        assert s.shape[0] == batch_size, "timestep pool exhausted"
        self._tc += batch_size
        return s

    def next_noise(self, like):
        n = self.noise[self._nc:self._nc + like.shape[0]]
        assert n.shape == like.shape, "noise pool exhausted / shape mismatch"
        self._nc += like.shape[0]
        return n.clone()


def _param_deltas(model, init_sd):
    return {k: p.detach() - init_sd[k] for k, p in model.state_dict().items()}


# ---------------------------------------------------------------------------
# 1. Default-off regression: grad_accum == 1 (or absent) == historical loop
# ---------------------------------------------------------------------------

def _reference_pre_change_loop(model, loader, args):
    """The PRE---grad_accum training algorithm, re-implemented verbatim:
    per batch: sigma draw -> noise draw -> forward -> mse -> weighted sum ->
    zero_grad -> backward (UNSCALED) -> clip -> opt.step -> sched.step.
    The tqdm wrapper is reproduced too: constructing tqdm(loader) creates a
    dataloader iterator (one global-RNG base-seed draw) and the for loop
    creates a second, so the trainer has always consumed TWO base-seed draws
    per epoch -- the reference must match to stay on the same RNG stream."""
    from tqdm.auto import tqdm
    device = torch.device("cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.num_epochs * len(loader), eta_min=args.lr * args.lr_min_ratio)
    model.train()
    for _epoch in range(args.num_epochs):
        for images, labels in tqdm(loader, disable=False, dynamic_ncols=True, leave=False):
            clean = images.to(torch.float32)
            timesteps = tps.draw_phase_sigmas(
                args, batch_size=clean.shape[0], phase_start=_PHASE["start"],
                phase_end=_PHASE["end"], num_bins=_NUM_BINS, device=device)
            noise = torch.randn_like(clean)
            pred, target = tps.model_forward_for_loss(
                model=model, args=args, clean=clean, timesteps=timesteps,
                labels=labels, alpha_bar=_ALPHA_BAR, in_channels=3,
                compute_dtype=torch.float32, noise=noise)
            loss_gt = torch.nn.functional.mse_loss(pred, target)
            loss_kd = torch.zeros((), device=device)
            loss = args.gt_weight * loss_gt + args.kd_weight * loss_kd
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            opt.step()
            sched.step()
    return model


@pytest.mark.parametrize("accum_attr", ["absent", "one"])
def test_grad_accum_default_off_bit_identical(tmp_path, accum_attr):
    torch.manual_seed(7)
    base = TinyNet()
    images = torch.randn(24, 3, 8, 8)
    labels = torch.zeros(24, dtype=torch.long)

    args = _make_args(num_epochs=2)
    if accum_attr == "one":
        args.grad_accum = 1  # explicit 1 must equal the absent-attribute path
    # else: attribute ABSENT entirely -> getattr(args, "grad_accum", 1) path,
    # exactly like every pre-flag caller of train_phase.

    m_real = copy.deepcopy(base)
    torch.manual_seed(123)
    _run_train_phase(m_real, _loader(images, labels, 8), args, tmp_path, f"real_{accum_attr}")

    m_ref = copy.deepcopy(base)
    torch.manual_seed(123)
    _reference_pre_change_loop(m_ref, _loader(images, labels, 8), args)

    for (k, pa), (k2, pb) in zip(m_real.state_dict().items(), m_ref.state_dict().items()):
        assert k == k2
        # bit-identical: same RNG stream + same FP-op sequence, no tolerance
        assert torch.equal(pa, pb), f"param {k} differs from pre-change reference"


# ---------------------------------------------------------------------------
# 2. Equivalence: batch 32 x 1 step  ==  batch 8 x grad_accum 4
# ---------------------------------------------------------------------------

def _equivalence_run(tmp_path, monkeypatch, *, diffusion, model_type, compute_dtype,
                     channels, timestep_pool):
    n_total, micro, accum = 64, 8, 4  # 2 optimizer steps either way
    g = torch.Generator().manual_seed(42)
    images = torch.randn(n_total, channels, 8, 8, generator=g)
    labels = torch.zeros(n_total, dtype=torch.long)
    noise_pool = torch.randn(n_total * 1, channels, 8, 8, generator=g)
    pool = _SlicePool(timestep_pool, noise_pool)

    assert diffusion == "edm"
    monkeypatch.setattr(tps, "draw_phase_sigmas",
                        lambda a, batch_size, **kw: pool.next_timesteps(batch_size))
    monkeypatch.setattr(torch, "randn_like", lambda t, **kw: pool.next_noise(t))

    torch.manual_seed(3)
    base = TinyNet(ch=channels)
    init_sd = {k: v.clone() for k, v in base.state_dict().items()}

    def run(batch_size, grad_accum, sub):
        pool.reset()
        m = copy.deepcopy(base)
        args = _make_args(num_epochs=1, diffusion=diffusion, model_type=model_type,
                          grad_accum=grad_accum, batch_size=batch_size)
        d = os.path.join(str(tmp_path), sub)
        os.makedirs(d, exist_ok=True)
        train_phase(args=args, teacher=None, student_ddp=m, raw_student=m, vae=None,
                    dataloader=_loader(images, labels, batch_size), sampler=None,
                    phase=_PHASE, num_bins=_NUM_BINS, alpha_bar=_ALPHA_BAR,
                    in_channels=channels, compute_dtype=compute_dtype,
                    phase_dir=d, use_wandb=False, resume_ckpt=None)
        return _param_deltas(m, init_sd)

    d_big = run(micro * accum, 1, "big")        # batch 32, 2 plain steps
    d_acc = run(micro, accum, "acc")            # batch 8 x accum 4, 2 accum steps

    for k in d_big:
        assert torch.allclose(d_big[k], d_acc[k], rtol=1e-5, atol=1e-7), (
            f"[{diffusion}] param-delta mismatch for {k}: "
            f"max abs diff {(d_big[k] - d_acc[k]).abs().max():.3e}, "
            f"delta scale {d_big[k].abs().max():.3e}"
        )
        # and the runs actually trained (deltas are not trivially zero)
        assert d_big[k].abs().max() > 0


def test_grad_accum_equivalence_edm(tmp_path, monkeypatch):
    g = torch.Generator().manual_seed(99)
    # sigmas anywhere in [sigma_min, sigma_max]; log-uniform-ish spread
    sig = torch.exp(torch.empty(64).uniform_(math.log(0.05), math.log(5.0), generator=g))
    _equivalence_run(tmp_path, monkeypatch, diffusion="edm", model_type="dit_micro",
                     compute_dtype=torch.float32, channels=3, timestep_pool=sig)



# ---------------------------------------------------------------------------
# 3. Step semantics: floor(M/N) steps/epoch, ckpt on optimizer steps, LR match
# ---------------------------------------------------------------------------

def test_grad_accum_step_counter_and_ckpt_every(tmp_path):
    # M = 10 microbatches/epoch, N = 4 -> 2 optimizer steps/epoch (tail of 2
    # microbatches dropped, drop_last convention); 2 epochs -> 4 steps total.
    torch.manual_seed(11)
    images = torch.randn(40, 3, 8, 8)
    labels = torch.zeros(40, dtype=torch.long)
    m = TinyNet()
    args = _make_args(num_epochs=2, grad_accum=4, batch_size=4,
                      ckpt_every=1, full_ckpt=True)
    _run_train_phase(m, _loader(images, labels, 4), args, tmp_path, "steps")
    d = os.path.join(str(tmp_path), "steps")

    # `step` stored in the final full checkpoint counts OPTIMIZER steps:
    # 2 epochs * floor(10/4) = 4, not the 20 microbatches consumed.
    final = torch.load(os.path.join(d, "student.pt"), map_location="cpu", weights_only=False)
    assert final["step"] == 4

    # ckpt_every=1 fired once per OPTIMIZER step (pre-increment step ids 0..3),
    # under curve/ in full-ckpt mode.
    steps = sorted(int(f[len("step_"):-len(".pt")])
                   for f in os.listdir(os.path.join(d, "curve")) if f.startswith("step_"))
    assert steps == [0, 1, 2, 3]


def test_grad_accum_wsd_lr_matches_batch_equivalent_run(tmp_path):
    # Same WSD schedule knobs; run A = batch 16 x grad_accum 1, run B =
    # batch 4 x grad_accum 4. Same optimizer-step count -> IDENTICAL lr_history.
    torch.manual_seed(13)
    images = torch.randn(32, 3, 8, 8)
    labels = torch.zeros(32, dtype=torch.long)
    wsd = dict(num_epochs=2, lr_schedule="wsd", warmup_steps=2, cooldown_frac=0.5)

    m_a = TinyNet()
    meta_a = _run_train_phase(
        m_a, _loader(images, labels, 16),
        _make_args(grad_accum=1, batch_size=16, **wsd), tmp_path, "wsd_a")
    m_b = TinyNet()
    meta_b = _run_train_phase(
        m_b, _loader(images, labels, 4),
        _make_args(grad_accum=4, batch_size=4, **wsd), tmp_path, "wsd_b")

    assert meta_a["lr_history"] == meta_b["lr_history"]  # exact float equality
    assert len(meta_a["lr_history"]) == 2  # one entry per epoch, both runs


def test_grad_accum_larger_than_epoch_raises(tmp_path):
    images = torch.randn(8, 3, 8, 8)
    labels = torch.zeros(8, dtype=torch.long)
    args = _make_args(grad_accum=4, batch_size=4)  # 2 microbatches < accum 4
    with pytest.raises(ValueError, match="grad_accum"):
        _run_train_phase(TinyNet(), _loader(images, labels, 4), args, tmp_path, "err")


# ---------------------------------------------------------------------------
# 4. EMA updated once per optimizer step, never per microbatch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("grad_accum,expected", [(1, 8), (4, 2)])
def test_grad_accum_ema_updates_once_per_optimizer_step(tmp_path, monkeypatch,
                                                        grad_accum, expected):
    calls = {"n": 0}
    real = tps.ema_update

    def counting(ema, model, beta):
        calls["n"] += 1
        return real(ema, model, beta)

    monkeypatch.setattr(tps, "ema_update", counting)
    torch.manual_seed(17)
    images = torch.randn(32, 3, 8, 8)  # 8 microbatches at batch 4
    labels = torch.zeros(32, dtype=torch.long)
    args = _make_args(grad_accum=grad_accum, batch_size=4, ema_beta=0.99)
    _run_train_phase(TinyNet(), _loader(images, labels, 4), args, tmp_path,
                     f"ema_{grad_accum}")
    assert calls["n"] == expected


# ---------------------------------------------------------------------------
# Arg surface: default 1, present in vars(args) (what run_config.json dumps)
# ---------------------------------------------------------------------------

def test_parse_args_grad_accum_default_and_dumpable(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "train_phase_students.py", "--model_type", "dit_micro",
        "--teacher_checkpoint", "t.pt", "--output_dir", "o",
        "--dataset", "cifar10", "--grouping_json", "g.json",
    ])
    args = tps.parse_args()
    assert args.grad_accum == 1                # opt-in: default preserves history
    assert "grad_accum" in vars(args)          # run_config.json dumps vars(args)
