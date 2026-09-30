#!/usr/bin/env python3
"""Measure composite inference throughput (denoising steps/sec) per allocation variant.

Throughput is an architecture property (independent of training), so we build each
variant's per-phase NarrowDiT students from the arch plan (random weights are fine for
timing) and time a forward pass. The composite routes each denoising step to one phase
student, so its steps/sec = 1 / sum_phase(w_phase * latency_phase), where w_phase is the
fraction of the schedule's timesteps that fall in that phase (= bin_width / num_bins).
global (single phase over the whole range) => steps/sec = 1 / latency.

Emits JSON: {variant: {steps_per_sec, params, per_phase:[{bins, params, latency_ms}]}}.

Optional ``--model_spec`` (opt-in, default off; MUTUALLY EXCLUSIVE with
``--arch_plan``/``--variants``) benches an arbitrary list of models --
each possibly drawn from a DIFFERENT arch_plan file/variant key
-- back-to-back IN ONE PROCESS ON ONE GPU, in INTERLEAVED round-robin order (A, B, C, A,
B, C, ... for ``--rounds`` rounds), reporting each model's MEDIAN steps/sec + within-job
min/max spread across rounds. This exists to kill cross-job node-to-node throughput
variance (measured up to 59% for an identical model benched in two different cluster
allocations) when comparing several variants' speed to each other: build every model once, keep them
all resident, and round-robin between them so every number comes from the same GPU at
roughly the same point in the job's thermal/clock history. See ``--model_spec``'s own
help for the JSON schema.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from pace.dit_arch_alloc import build_narrow_dit, count_dit_params
from pace.external_repos import configure_dit_repo


def time_forward(model, x, t, y, dtype, iters, warmup):
    dev = x.device
    use_ac = dtype in (torch.float16, torch.bfloat16)
    with torch.no_grad():
        for _ in range(warmup):
            with torch.autocast("cuda", dtype=dtype, enabled=use_ac):
                model(x, t, y)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for _ in range(iters):
            with torch.autocast("cuda", dtype=dtype, enabled=use_ac):
                model(x, t, y)
        torch.cuda.synchronize(dev)
    return (time.perf_counter() - t0) / iters  # seconds/forward


def compile_model(model, dtype, inputs, mode="default", tol=None):
    """torch.compile ``model`` for inference at fixed shapes and check it against eager.

    ``inputs`` = list of (x, t, y) covering every shape the timing loop will use; each is
    pre-warmed 5x so compilation and CUDA-graph capture (mode='reduce-overhead') happen here,
    not inside a timed region. The first input gives the equivalence record: rel-L2 of the
    compiled vs eager output under the same autocast as ``time_forward`` (tol 2e-2 bf16 /
    1e-3 fp32 -- inductor reorders reductions; this is a sanity gate, not bit-exactness).
    Returns ``(compiled_model, record)``."""
    use_ac = dtype in (torch.float16, torch.bfloat16)
    if tol is None:
        tol = 2e-2 if use_ac else 1e-3
    # Every NarrowDiT instance shares ONE forward code object; dynamo recompiles it per distinct
    # architecture x batch shape and silently falls back to eager past the recompile limit (8).
    # A bench compiles many distinct archs, so raise the limits; otherwise the models past the
    # limit are silently timed in eager mode while the rest get the compiled speed-up.
    from torch import _dynamo as tdyn   # NOT `import torch._dynamo`: that rebinds `torch` as a local
    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(tdyn.config, name):
            setattr(tdyn.config, name, 256)
    if hasattr(tdyn.config, "accumulated_recompile_limit"):
        tdyn.config.accumulated_recompile_limit = 4096
    x, t, y = inputs[0]
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=use_ac):
        ref = model(x, t, y).float().clone()
    comp = torch.compile(model, mode=mode, dynamic=False)
    with torch.no_grad():
        for xi, ti, yi in inputs:
            for _ in range(5):
                with torch.autocast("cuda", dtype=dtype, enabled=use_ac):
                    comp(xi, ti, yi)
        with torch.autocast("cuda", dtype=dtype, enabled=use_ac):
            got = comp(x, t, y).float().clone()   # clone: cudagraph output memory is reused
    torch.cuda.synchronize()
    rel = float((got - ref).norm() / ref.norm().clamp_min(1e-12))
    return comp, {"mode": mode, "rel_l2_vs_eager": rel, "tol": tol, "pass": rel <= tol}


def time_train_step(model, x, t, y, dtype, iters, warmup):
    """Time one optimization step: fwd -> proxy MSE loss -> backward -> Adam step.
    The proxy loss (mean-square of the model output) has negligible cost vs fwd+bwd,
    so this reflects the true training-step throughput. Adam(lr=2e-4, wd=0) mirrors runs."""
    dev = x.device
    use_ac = dtype in (torch.float16, torch.bfloat16)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=0.0)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype, enabled=use_ac):
            out = model(x, t, y)
        loss = out.float().pow(2).mean()
        loss.backward()
        opt.step()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize(dev)
    return (time.perf_counter() - t0) / iters  # seconds/train-step


# ---------------------------------------------------------------------------
# --model_spec: interleaved multi-model within-job round-robin bench
# ---------------------------------------------------------------------------

_NVIDIA_SMI_FIELD_SETS = (
    # Preferred: includes throttle-reason bitmask (newer nvidia-smi/driver). Some driver
    # versions renamed this field (clocks_throttle_reasons.active -> clocks_event_reasons.
    # active); we try both, then fall back to a throttle-agnostic set so a single unknown
    # field never sacrifices the clocks/temp/power diagnostics that DO matter here.
    "clocks.sm,clocks.mem,temperature.gpu,power.draw,pstate,clocks_throttle_reasons.active",
    "clocks.sm,clocks.mem,temperature.gpu,power.draw,pstate,clocks_event_reasons.active",
    "clocks.sm,clocks.mem,temperature.gpu,power.draw,pstate",
)


def nvidia_smi_snapshot():
    """Best-effort ``nvidia-smi`` query (GPU 0): clocks/temp/power/pstate/throttle state.

    Returns ``None`` (never raises) if ``nvidia-smi`` is unavailable or every field-set
    fails -- this is a diagnostic nicety for the within-job rebench, not a correctness
    requirement, so a missing/older nvidia-smi must not crash the bench itself."""
    for fields in _NVIDIA_SMI_FIELD_SETS:
        try:
            result = subprocess.run(
                ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None  # nvidia-smi itself not found/runnable -> no point trying other sets
        if result.returncode != 0:
            continue  # this field-set (likely the throttle-reason field name) isn't supported
        line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        if not line:
            continue
        values = [v.strip() for v in line.split(",")]
        keys = fields.split(",")
        if len(values) != len(keys):
            continue
        return dict(zip(keys, values))
    return None


def load_model_spec_entries(path):
    """Parse a ``--model_spec`` JSON file: a list of entries, each

        {"label": str, "arch_plan": path, "variant": key}

    Order is preserved -- it IS the round-robin order."""
    entries = json.load(open(path))
    if not isinstance(entries, list) or not entries:
        raise SystemExit(f"--model_spec {path} must be a non-empty JSON list of entries")
    for e in entries:
        if "label" not in e:
            raise SystemExit(f"--model_spec entry missing 'label': {e}")
        if "arch_plan" not in e or "variant" not in e:
            raise SystemExit(f"--model_spec entry {e['label']!r} needs both 'arch_plan'+'variant'")
    return entries


def build_spec_entry(entry, attn_impl, batch, num_bins, dev):
    """Build one ``--model_spec`` entry's phase model(s) + fixed input tensors (no timing).

    CPU-safe (``dev`` may be ``torch.device("cpu")``) for unit tests -- only ``time_forward``
    itself requires CUDA (``torch.cuda.synchronize``). Returns
    ``{"label", "total_params", "phases": [{"bins", "weight", "params", "model", "x", "t", "y"}]}``,
    the same per-phase shape (``bins``/``weight``/``params``) as the ``--arch_plan`` path
    above, so the two are directly comparable."""
    label = entry["label"]
    plan = json.load(open(entry["arch_plan"]))
    variant = entry["variant"]
    phase_defs = plan[variant]["phases"]
    phases = []
    total_params = 0
    for ph in phase_defs:
        cfg = ph["cfg"]
        s, e = ph["bins"]
        w = (e - s) / num_bins
        model = build_narrow_dit(cfg, attn_impl=attn_impl).to(dev)
        model.eval()
        params = count_dit_params(model)
        total_params += params
        B, C, S = batch, int(cfg["in_channels"]), int(cfg["input_size"])
        x = torch.randn(B, C, S, S, device=dev)
        t = torch.rand(B, device=dev)
        y = torch.randint(0, int(cfg["num_classes"]), (B,), device=dev)
        phases.append({"bins": [s, e], "weight": round(w, 4), "params": params,
                       "model": model, "x": x, "t": t, "y": y})
    return {"label": label, "total_params": total_params, "phases": phases}


def composite_steps_per_sec(weights, latencies_sec):
    """``1 / sum(w * latency)`` -- the same composite formula as the ``--arch_plan``
    loop, factored out so both the single-round and the round-robin path (and tests)
    share one implementation."""
    weighted_latency = sum(w * lat for w, lat in zip(weights, latencies_sec))
    return 1.0 / weighted_latency if weighted_latency > 0 else 0.0


def summarize_rounds(values):
    """median/min/max/spread-as-%-of-median over a list of per-round scalars. Pure
    Python (no torch) -- used for both steps/sec-per-round and (if ever needed)
    per-phase latency-per-round summaries."""
    med = statistics.median(values)
    lo, hi = min(values), max(values)
    spread_pct = round(100.0 * (hi - lo) / med, 2) if med > 0 else None
    return {"median": round(med, 4), "min": round(lo, 4), "max": round(hi, 4),
            "spread_pct_of_median": spread_pct}


def run_model_spec(args, dtype, dev):
    """Build every ``--model_spec`` entry's model(s) once, then round-robin ``--rounds``
    full (warmup+timed) passes over them in a FIXED order (entry order = round-robin
    order), so every entry's timing samples are interleaved across the same GPU's
    thermal/clock history rather than each entry hogging a contiguous time block."""
    entries = load_model_spec_entries(args.model_spec)
    built = [build_spec_entry(e, args.attn_impl, args.batch, args.num_bins, dev)
             for e in entries]
    for b in built:
        for ph in b["phases"]:
            ph["latencies_sec"] = []
            if getattr(args, "compile", None):
                ph["model"], ph["compile_equiv"] = compile_model(
                    ph["model"], dtype, [(ph["x"], ph["t"], ph["y"])], args.compile)
                print(f"  [compile:{args.compile}] {b['label']} bins {ph['bins']}: rel-L2 "
                      f"{ph['compile_equiv']['rel_l2_vs_eager']:.2e} pass={ph['compile_equiv']['pass']}",
                      flush=True)

    gpu_diagnostics = []
    for round_idx in range(args.rounds):
        snap = nvidia_smi_snapshot()
        if snap is not None:
            gpu_diagnostics.append({"round": round_idx, **snap})
        for b in built:
            for ph in b["phases"]:
                lat = time_forward(ph["model"], ph["x"], ph["t"], ph["y"], dtype,
                                    args.iters, args.warmup)
                ph["latencies_sec"].append(lat)
        print(f"  round {round_idx + 1}/{args.rounds} done", flush=True)

    out = {}
    for b in built:
        weights = [ph["weight"] for ph in b["phases"]]
        round_composites = [
            composite_steps_per_sec(weights, [ph["latencies_sec"][r] for ph in b["phases"]])
            for r in range(args.rounds)
        ]
        sps_summary = summarize_rounds(round_composites)
        per_phase_out = []
        for ph in b["phases"]:
            lats_ms = [round(l * 1000, 4) for l in ph["latencies_sec"]]
            per_phase_out.append({
                "bins": ph["bins"], "weight": ph["weight"], "params": ph["params"],
                "latency_ms_per_round": lats_ms,
                "latency_ms_median": round(statistics.median(lats_ms), 4),
                "latency_ms_min": round(min(lats_ms), 4),
                "latency_ms_max": round(max(lats_ms), 4),
                "compile_equiv": ph.get("compile_equiv"),
            })
        out[b["label"]] = {
            "steps_per_sec_per_round": [round(v, 3) for v in round_composites],
            "steps_per_sec_median": round(sps_summary["median"], 3),
            "steps_per_sec_min": round(sps_summary["min"], 3),
            "steps_per_sec_max": round(sps_summary["max"], 3),
            "spread_pct_of_median": sps_summary["spread_pct_of_median"],
            "rounds": args.rounds, "batch": args.batch, "mode": args.mode,
            "dtype": args.dtype, "total_params": b["total_params"],
            "per_phase": per_phase_out, "attn_impl": args.attn_impl,
            "compile": getattr(args, "compile", None),
        }
        print(f"  {b['label']}: median {sps_summary['median']:.2f} steps/s "
              f"[{sps_summary['min']:.2f}, {sps_summary['max']:.2f}] "
              f"(spread {sps_summary['spread_pct_of_median']}% of median, batch {args.batch}), "
              f"{b['total_params'] / 1e6:.1f}M params total", flush=True)

    for b in built:
        for ph in b["phases"]:
            del ph["model"], ph["x"]
    torch.cuda.empty_cache()
    return out, gpu_diagnostics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch_plan", default=None,
                    help="Required unless --model_spec is given.")
    ap.add_argument("--variants", default="global,uniform_blockwise,blockwise_capacity,layerwise_capacity")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--mode", choices=["infer", "train"], default="infer",
                    help="infer: no_grad forward. train: fwd+backward+Adam step (training throughput).")
    ap.add_argument("--num_bins", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dit_repo", default=None,
                    help="facebookresearch/DiT checkout (default: $DIT_REPO).")
    ap.add_argument("--attn_impl", choices=["manual", "sdpa"], default="manual",
                    help="NarrowAttention execution path (see pace.dit_arch_alloc). "
                         "'manual' (default) is BYTE-IDENTICAL to every bench run before "
                         "this flag existed (explicit QK^T/softmax/@V). 'sdpa' switches to "
                         "F.scaled_dot_product_attention (a fused/flash kernel) -- same "
                         "weights/math, execution-path-only change, expected to be "
                         "materially faster at large widths (D=1024).")
    ap.add_argument("--model_spec", default=None,
                    help="opt-in, MUTUALLY EXCLUSIVE with --arch_plan/--variants: "
                         "path to a JSON list of {label, arch_plan, variant} "
                         "entries, benched back-to-back in "
                         "ONE process on ONE GPU, interleaved round-robin over --rounds rounds, "
                         "reporting each label's median/min/max steps/sec across rounds -- see "
                         "the module docstring. Default None "
                         "leaves the --arch_plan path (and every number it has ever produced) "
                         "byte-identical.")
    ap.add_argument("--compile", default=None, choices=["default", "reduce-overhead", "max-autotune"],
                    help="--model_spec only: torch.compile every phase model (fixed shapes) before timing; "
                         "eager-vs-compiled rel-L2 recorded per phase (compile_equiv)")
    ap.add_argument("--rounds", type=int, default=3,
                    help="--model_spec only: number of full interleaved round-robin passes. "
                         "Each round re-does its own --warmup+--iters timing for every label "
                         "(so a label's reported latency is never averaged across rounds, only "
                         "medianed) -- this is what lets --rounds>1 catch within-job clock/thermal "
                         "drift as an outlier round rather than silently smoothing it away.")
    args = ap.parse_args()
    configure_dit_repo(args.dit_repo)
    if args.model_spec is not None:
        if args.mode != "infer":
            raise SystemExit("--model_spec only supports --mode infer (every entry is an "
                              "eval-only architecture timing, no trainable-param path)")
        if args.rounds < 1:
            raise SystemExit(f"--rounds must be >= 1, got {args.rounds}")
        dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
        dev = torch.device("cuda")
        out, gpu_diagnostics = run_model_spec(args, dtype, dev)
        if gpu_diagnostics:
            out["_gpu_diagnostics"] = gpu_diagnostics
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"wrote {args.out}", flush=True)
        return
    if args.arch_plan is None:
        raise SystemExit("--arch_plan is required unless --model_spec is given")

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    dev = torch.device("cuda")
    plan = json.load(open(args.arch_plan))
    out = {}

    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        phases = plan[variant]["phases"]
        per_phase = []
        weighted_latency = 0.0  # sum_phase w_phase * latency (seconds)
        total_params = 0
        for ph in phases:
            cfg = ph["cfg"]
            s, e = ph["bins"]
            w = (e - s) / args.num_bins
            model = build_narrow_dit(cfg, attn_impl=args.attn_impl).to(dev)
            model.eval() if args.mode == "infer" else model.train()
            params = count_dit_params(model)
            total_params += params
            B, C, S = args.batch, int(cfg["in_channels"]), int(cfg["input_size"])
            x = torch.randn(B, C, S, S, device=dev)
            t = torch.rand(B, device=dev)
            y = torch.randint(0, int(cfg["num_classes"]), (B,), device=dev)
            timer = time_forward if args.mode == "infer" else time_train_step
            lat = timer(model, x, t, y, dtype, args.iters, args.warmup)
            weighted_latency += w * lat
            per_phase.append({"bins": [s, e], "weight": round(w, 4),
                              "params": params, "latency_ms": round(lat * 1000, 3)})
            del model, x
            torch.cuda.empty_cache()
        sps = 1.0 / weighted_latency if weighted_latency > 0 else 0.0
        out[variant] = {"steps_per_sec": round(sps, 3), "batch": args.batch, "mode": args.mode,
                        "dtype": args.dtype, "total_params": total_params, "per_phase": per_phase,
                        "attn_impl": args.attn_impl}
        print(f"  {variant}: {sps:.2f} steps/s (batch {args.batch}), {total_params/1e6:.1f}M params total", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
