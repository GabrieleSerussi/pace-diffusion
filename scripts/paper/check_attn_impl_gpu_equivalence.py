#!/usr/bin/env python3
"""GPU correctness gate for NarrowAttention's opt-in ``attn_impl`` switch (manual vs sdpa).

The CPU equivalence tests (``tests/test_narrow_attention_sdpa.py``) cover random-init and
synthetic-plan cfgs in fp32 on CPU. This script is the complementary GPU check the perf-fix
task calls for: load ONE real trained checkpoint (a ``--student_dir`` staged the same way
``evaluate_students.py --mode composite`` reads it -- ``phase_<i>/{arch_cfg.json,student.pt}``),
build it TWICE (``attn_impl="manual"`` and ``attn_impl="sdpa"``) with the IDENTICAL loaded
weights, run both forward under the same autocast dtype used for real inference/eval
(default bf16, matching the latent DiT family's actual eval dtype), and report the max
absolute / relative difference between the two outputs.

Uses the same combined criterion as ``torch.testing.assert_close``/``torch.allclose``
(``|actual - expected| <= atol + rtol * |expected|``), NOT a raw per-element relative
difference -- a pure relative metric blows up at outputs near zero (division by ~0),
which is a known artifact of that metric, not a sign of a real bug. Exit code is 0
(pass) iff the fraction of elements violating that combined bound is below
--max_violation_frac (default 0.1%%). Defaults (--atol 2e-2 --rtol 1e-2) are looser than
the CPU fp32 gate's ~1e-5 -- bf16 has ~3 decimal digits of mantissa, and a fused SDPA
kernel's internal reduction order differs from the manual matmul path's, so bf16-level
rounding noise, not a correctness bug, is expected here.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/ (for evaluate_students)

from pace.dit_arch_alloc import build_narrow_dit, count_dit_params
from pace.external_repos import configure_dit_repo
from evaluate_students import _student_state_from_payload


def load_phase_state_and_cfg(phase_dir: str):
    with open(os.path.join(phase_dir, "arch_cfg.json")) as f:
        arch_cfg = json.load(f)
    payload = torch.load(os.path.join(phase_dir, "student.pt"), map_location="cpu", weights_only=False)
    state = _student_state_from_payload(payload)
    return arch_cfg["cfg"], state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_dir", required=True,
                    help="Staged student dir (e.g. $STUDENTS/bedroom_global) containing "
                         "phase_<i>/{arch_cfg.json,student.pt}.")
    ap.add_argument("--phase", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--atol", type=float, default=2e-2)
    ap.add_argument("--rtol", type=float, default=1e-2)
    ap.add_argument("--max_violation_frac", type=float, default=1e-3,
                    help="max fraction of elements allowed to violate the combined "
                         "atol+rtol*|expected| bound before this is reported as FAIL.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dit_repo", default=None,
                    help="facebookresearch/DiT checkout (default: $DIT_REPO).")
    args = ap.parse_args()
    configure_dit_repo(args.dit_repo)

    device = torch.device("cuda")
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    use_autocast = dtype != torch.float32

    phase_dir = os.path.join(args.student_dir, f"phase_{args.phase}")
    cfg, state = load_phase_state_and_cfg(phase_dir)
    print(f"loaded {phase_dir}: hidden_size={cfg['hidden_size']} depth={cfg['depth']} "
          f"in_channels={cfg['in_channels']} input_size={cfg['input_size']}")

    manual = build_narrow_dit(cfg, attn_impl="manual").to(device)
    sdpa = build_narrow_dit(cfg, attn_impl="sdpa").to(device)
    missing_m, unexpected_m = manual.load_state_dict(state, strict=False)
    missing_s, unexpected_s = sdpa.load_state_dict(state, strict=False)
    extra_missing = [k for k in missing_m if not k.endswith("pos_embed")]
    extra_unexpected = [k for k in unexpected_m if not k.endswith("pos_embed")]
    if extra_missing or extra_unexpected:
        raise RuntimeError(f"state-dict mismatch loading {phase_dir}: "
                            f"missing={extra_missing[:3]} unexpected={extra_unexpected[:3]}")
    assert missing_m == missing_s and unexpected_m == unexpected_s, (
        "manual and sdpa builds disagree on which keys loaded -- same cfg should give "
        "identical state_dict keys (see tests/test_narrow_attention_sdpa.py test (c))")
    manual.eval()
    sdpa.eval()
    n_manual = count_dit_params(manual)
    n_sdpa = count_dit_params(sdpa)
    print(f"params: manual={n_manual:,} sdpa={n_sdpa:,} (equal={n_manual == n_sdpa})")

    gen = torch.Generator(device=device).manual_seed(args.seed)
    B, C, S = args.batch, int(cfg["in_channels"]), int(cfg["input_size"])
    x = torch.randn(B, C, S, S, device=device, generator=gen)
    t = torch.rand(B, device=device, generator=gen)
    y = torch.randint(0, int(cfg["num_classes"]), (B,), device=device, generator=gen)

    with torch.no_grad():
        with torch.autocast("cuda", dtype=dtype, enabled=use_autocast):
            out_manual = manual(x, t, y)
            out_sdpa = sdpa(x, t, y)
    out_manual = out_manual.float()
    out_sdpa = out_sdpa.float()

    abs_diff = (out_manual - out_sdpa).abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()
    # Combined criterion (same formula as torch.allclose/assert_close): a bound that
    # scales with the reference magnitude but has an absolute floor, so near-zero
    # elements don't blow up a naive relative-diff metric.
    bound = args.atol + args.rtol * out_manual.abs()
    violation = (abs_diff > bound)
    violation_frac = violation.float().mean().item()
    worst_excess = (abs_diff - bound).max().item()  # how far the worst element overshoots

    print(f"dtype={args.dtype} batch={B}: max_abs_diff={max_abs:.6g} mean_abs_diff={mean_abs:.6g} "
          f"violation_frac={violation_frac:.6g} (atol={args.atol}, rtol={args.rtol}, "
          f"max_violation_frac={args.max_violation_frac}) worst_excess={worst_excess:.6g}")

    ok = violation_frac <= args.max_violation_frac
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
