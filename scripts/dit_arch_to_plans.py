#!/usr/bin/env python3
"""Build per-variant DiT architecture plans for EDM width-allocation distillation.

Reads the teacher importance ``results.json`` + a timestep ``grouping.json`` and
emits ``plans.json``: {variant: {"teacher_params", "phases":[{phase,bins,
target_params,realized_params,cfg}]}}. ``cfg`` is a full NarrowDiT kwargs dict.

This is the DiT planner of the released code.  It is not the Section 3.4 rule
of the paper: phase budgets come from a per-phase aggregate (``--phase_budget_agg``,
default q90) of ``n_eff``, and layerwise widths from a q90 of the relative
deltas with a square-root rule (see ``pace.dit_arch_alloc``).  The U-Net
allocator (``scripts/dry_run_capacity_allocation.py``) implements Section 3.4.
Instantiating the plan's NarrowDiT students to verify parameter counts needs the
facebookresearch/DiT checkout (``--dit_repo`` or ``$DIT_REPO``).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from pace.dit_arch_alloc import (MICRO_TEACHER_CFG, PHASE_BUDGET_AGGS,
                                      SMICRO_TEACHER_CFG, build_plan)
from pace.external_repos import configure_dit_repo
from pace.jsonio import load_json

# DiT-XL/2 teacher (ImageNet-256 latents).
XL_TEACHER_CFG = {
    "hidden_size": 1152, "depth": 28, "num_heads": 16, "mlp_ratio": 4.0,
    "patch_size": 2, "in_channels": 4, "input_size": 32,
    "num_classes": 1000, "learn_sigma": True,
}
# DiT-B/2 dims (facebookresearch/DiT's "DiT-B/2" config), for the from-scratch
# unconditional DDPM teachers trained on LSUN Bedroom 256 / FFHQ 256 VAE latents
# via --model_type dit_xl --diffusion ddpm --arch_plan --variant global
# --kd_weight 0 --unconditional --dataset image_folder. num_classes=1 (single
# dummy class, y=0 convention) and learn_sigma=True
# (matches the DDPM/dit_xl branch's eps-in-first-half-of-2*channels
# convention). `global` variant realizes 129,548,576 params (~130M, matching
# the DiT-B/2 paper spec).
DITB_TEACHER_CFG = {
    "hidden_size": 768, "depth": 12, "num_heads": 12, "mlp_ratio": 4.0,
    "patch_size": 2, "in_channels": 4, "input_size": 32,
    "num_classes": 1, "learn_sigma": True,
}
TEACHER_CFGS = {"dit_micro": MICRO_TEACHER_CFG, "dit_xl": XL_TEACHER_CFG,
                "dit_smicro": SMICRO_TEACHER_CFG, "dit_b": DITB_TEACHER_CFG}


def load_phases(grouping_json):
    b = load_json(grouping_json)["boundaries"]
    return [(int(b[i]), int(b[i + 1])) for i in range(len(b) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_json", required=True)
    ap.add_argument("--grouping_json", required=True)
    ap.add_argument("--model", required=True, choices=list(TEACHER_CFGS))
    ap.add_argument("--dit_repo", default=None,
                    help="facebookresearch/DiT checkout (default: $DIT_REPO).")
    ap.add_argument("--variants",
                    default="global,uniform_blockwise,blockwise_capacity,layerwise_capacity")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--layerwise_g_max", type=float, default=1.75,
                    help="Upper bound of match_layerwise_cfg's per-layer width-scale "
                         "binary search (default 1.75 = UNCHANGED from every plan "
                         "built before this flag existed -> byte-identical plans.json "
                         "when re-run with no override). The default saturates "
                         "this search before it re-hits the phase budget whenever "
                         "per-layer importance is skewed, silently under-provisioning "
                         "layerwise_capacity students relative to blockwise_capacity/ "
                         "uniform_blockwise by up to ~20-24%%. Pass e.g. 50-100 to "
                         "close that gap for a NEW plan (this does not change any "
                         "already-committed plans.json).")
    ap.add_argument("--layer_score_eps", type=float, default=0.0,
                    help="Per-phase epsilon floor added to layer_scores() before "
                         "match_layerwise_cfg's sqrt(score/mean) weighting: "
                         "score'_l = score_l + eps * phase_mean(score). Default 0.0 "
                         "= UNCHANGED from every plan built before this flag existed "
                         "-> byte-identical plans.json when re-run with no override. "
                         "On the DiT-Micro profile, phase 0 has "
                         "4/8 layers with an EXACTLY-ZERO measured importance score "
                         "(a measurement-noise artifact of the permutation-ablation "
                         "clip, not a real per-layer difference: 86.5%% of that "
                         "phase's raw measurements are clipped to 0, vs 0-52%% in "
                         "every other phase) -- a hard 0.0 score permanently floors "
                         "that layer at the minimum grid width regardless of "
                         "--layerwise_g_max (0 * g == 0 for every finite g). A small "
                         "eps (e.g. 0.05) makes the redistribution degrade toward "
                         "UNIFORM intra-phase allocation when a phase's true signal "
                         "is at the noise floor, while leaving phases with real "
                         "dynamic range (0%% clipped) essentially unchanged.")
    ap.add_argument("--blockwise_budget_match", action="store_true", default=False,
                    help="Opt-in (default off = byte-identical) budget-aware "
                         "rounding for blockwise_capacity ONLY (never "
                         "uniform_blockwise). match_uniform_cfg picks a single grid "
                         "D by closest ABSOLUTE distance to the phase target; at "
                         "DiT-Micro scale that grid has only 8 points and the gaps "
                         "between them grow with D, so a target inside a wide gap "
                         "can overshoot by double-digit percentages (observed: "
                         "+21.9%%/+17.8%% on Micro's 2 worst phases). When set, "
                         "blockwise_capacity instead calls "
                         "match_blockwise_budget_cfg, which keeps the same "
                         "one-width-per-phase semantics but trims that shared width "
                         "below the grid ceiling (reusing match_layerwise_cfg's "
                         "g-search with a FLAT per-layer score vector) to land "
                         "within a few percent of budget in every phase.")
    ap.add_argument("--blockwise_budget_g_max", type=float, default=50.0,
                    help="g_max forwarded to match_blockwise_budget_cfg when "
                         "--blockwise_budget_match is set (ignored otherwise).")
    ap.add_argument("--uniform_budget_match", action="store_true", default=False,
                    help="Opt-in (default off = byte-identical), INDEPENDENT of "
                         "--blockwise_budget_match: applies the SAME "
                         "match_blockwise_budget_cfg fix to uniform_blockwise "
                         "instead of blockwise_capacity. The D-grid gap is a "
                         "property of the teacher's (hidden_size, num_heads) grid, "
                         "not of which variant's target happens to hit it -- "
                         "uniform_blockwise's own equal-per-phase target can land "
                         "in just as wide a gap as blockwise_capacity's at a "
                         "different teacher scale/grouping. Preserves "
                         "uniform_blockwise's defining property (one identical "
                         "width per phase, never importance-weighted per-layer). "
                         "Kept as a separate flag (never bundled with "
                         "--blockwise_budget_match) so the existing invariant "
                         "'uniform_blockwise is never affected by "
                         "--blockwise_budget_match' stays exactly true.")
    ap.add_argument("--phase_budget_agg", default="q90", choices=list(PHASE_BUDGET_AGGS),
                    help="How block_budgets aggregates n_eff over each phase's bins "
                         "before the ^alpha proportional split (blockwise_capacity/"
                         "layerwise_capacity only; global/uniform_blockwise never "
                         "consult n_eff). Default 'q90' = UNCHANGED from every plan "
                         "built before this flag existed (the default path still "
                         "calls q90_block_scores/np.quantile verbatim) -> "
                         "byte-identical plans.json when re-run with no override. "
                         "'geomean' = exp(mean(log(max(x,1e-9)))) per phase "
                         "(observed to agree with mean within about 1.3pp phase "
                         "share); 'mean' = the "
                         "arithmetic mean itself. When non-default, each variant "
                         "dict in the output JSON gains a top-level "
                         "'phase_budget_agg' provenance key (absent on default "
                         "rebuilds so existing files' structure stays "
                         "byte-identical).")
    ap.add_argument("--head_dim", type=int, default=None,
                    help="Opt-in: cut attention as WHOLE heads of this dim (e.g. 64) instead of "
                         "shrinking the per-head dim at the teacher's head count (small head "
                         "dims fall off the fast SDPA kernels). Default None = legacy behaviour.")
    ap.add_argument("--d_mult", type=int, default=None,
                    help="Opt-in: hidden-size grid step (e.g. 64 = tensor-core tiles). "
                         "Default None = lcm(num_heads, 8), as before.")
    ap.add_argument("--m_mult", type=int, default=8,
                    help="Opt-in: mlp_hidden rounding multiple (default 8 = legacy).")
    ap.add_argument("--latency_table", default=None,
                    help="Opt-in: glob of latency-grid JSON files (records with hidden/heads/mlp "
                         "fields plus latency columns). The budget unit becomes measured ms/step "
                         "and --latency_target_ms is the composite (bin-weighted) target.")
    ap.add_argument("--latency_target_ms", type=float, default=None,
                    help="Composite (bin-weighted) ms/step target; required with --latency_table.")
    ap.add_argument("--latency_key", default="lat_ms_compiled_b256",
                    help="Latency column to read from the table "
                         "(e.g. lat_ms_compiled_b256 | lat_ms_eager_b256).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    configure_dit_repo(args.dit_repo)

    results = load_json(args.results_json)
    phases = load_phases(args.grouping_json)
    teacher_cfg = TEACHER_CFGS[args.model]

    cost_fn = None
    if args.latency_table:
        if args.latency_target_ms is None:
            raise SystemExit("--latency_table requires --latency_target_ms")
        from pace.dit_arch_alloc import latency_cost_fn_from_tables
        cost_fn = latency_cost_fn_from_tables(args.latency_table, key=args.latency_key, depth=int(teacher_cfg["depth"]))

    out = {}
    for v in [s.strip() for s in args.variants.split(",") if s.strip()]:
        out[v] = build_plan(results, phases, teacher_cfg, v, alpha=args.alpha,
                             head_dim=args.head_dim, d_mult=args.d_mult, m_mult=args.m_mult,
                             cost_fn=cost_fn, cost_target=args.latency_target_ms,
                             layerwise_g_max=args.layerwise_g_max,
                             layer_score_eps=args.layer_score_eps,
                             blockwise_budget_match=args.blockwise_budget_match,
                             blockwise_budget_g_max=args.blockwise_budget_g_max,
                             uniform_budget_match=args.uniform_budget_match,
                             phase_budget_agg=args.phase_budget_agg)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    for v, plan in out.items():
        tp = plan["teacher_params"]
        rows = []
        for p in plan["phases"]:
            head = f"p{p['phase']}{p['bins']}={p['realized_params']:,}"
            if p.get("target_params") is not None:
                rows.append(f"{head}(t{p['target_params']:,.0f})")
            else:
                rows.append(f"{head} cost {p['realized_cost']:.2f}(t{p['target_cost']:.2f})")
        print(f"  {v}: teacher={tp:,} | " + "  ".join(rows))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
