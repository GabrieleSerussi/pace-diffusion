import os
import sys

import numpy as np
import pytest
import torch

# Mirror the repo convention: DiT repo components come from $DIT_REPO, DiTMicro
# from scripts/ (see tests/test_alloc_pruning.py).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from evaluate_parameters_dit_micro import DiTMicro  # noqa: E402

from pace.dit_arch_alloc import (  # noqa: E402
    MICRO_TEACHER_CFG,
    SMICRO_TEACHER_CFG,
    NarrowDiT,
    analytic_narrow_dit_params,
    block_budgets,
    build_plan,
    count_dit_params,
    layer_scores,
    match_blockwise_budget_cfg,
    match_layerwise_cfg,
    match_uniform_cfg,
    q90_block_scores,
)

# The archived DiT-Micro permutation profile released for Figure 6 (Appendix C.2).
_MICRO_RESULTS = os.path.join(
    os.path.dirname(__file__), "..", "artifacts", "dit_micro", "dit_micro_perm_results.json"
)


def _teacher_uniform_per_block():
    D = MICRO_TEACHER_CFG["hidden_size"]
    H = MICRO_TEACHER_CFG["num_heads"]
    M = int(D * MICRO_TEACHER_CFG["mlp_ratio"])
    return [{"num_heads": H, "attn_inner": D, "mlp_hidden": M}
            for _ in range(MICRO_TEACHER_CFG["depth"])]


def _teacher_narrow():
    return NarrowDiT(
        hidden_size=MICRO_TEACHER_CFG["hidden_size"],
        depth=MICRO_TEACHER_CFG["depth"],
        patch_size=MICRO_TEACHER_CFG["patch_size"],
        in_channels=MICRO_TEACHER_CFG["in_channels"],
        num_classes=MICRO_TEACHER_CFG["num_classes"],
        input_size=MICRO_TEACHER_CFG["input_size"],
        per_block=_teacher_uniform_per_block(),
        learn_sigma=MICRO_TEACHER_CFG["learn_sigma"],
    )


def _synthetic_results():
    """24 Micro head names (8 blocks x 3 heads), n_eff len 20, rds 24x20."""
    names = [f"blocks.{b}.attn.head_{h}" for b in range(8) for h in range(3)]
    rng = np.random.default_rng(0)
    rds = rng.random((24, 20))
    n_eff = np.linspace(2.0, 15.0, 20).tolist()
    return {"group_names": names, "relative_delta_stack": rds.tolist(), "n_eff": n_eff}


# --- Test 1: NarrowDiT uniform-teacher == stock DiTMicro ----------------------

@pytest.mark.external_dit
def test_narrow_uniform_matches_stock_micro_param_count_and_shape():
    narrow = _teacher_narrow()
    stock = DiTMicro()
    assert count_dit_params(narrow) == count_dit_params(stock), (
        count_dit_params(narrow), count_dit_params(stock))

    narrow.eval()
    stock.eval()
    x = torch.randn(2, 3, 32, 32)
    t = torch.rand(2)
    y = torch.randint(0, 10, (2,))
    with torch.no_grad():
        out_n = narrow(x, t, y)
        out_s = stock(x, t, y)
    assert out_n.shape == out_s.shape == (2, 3, 32, 32)


# --- Test 2: count decreases when hidden_size halves --------------------------

@pytest.mark.external_dit
def test_count_decreases_when_hidden_halves():
    full = _teacher_narrow()
    D = MICRO_TEACHER_CFG["hidden_size"] // 2  # 96, divisible by num_heads=3 and 8
    pb = [{"num_heads": 3, "attn_inner": D, "mlp_hidden": int(D * 4)} for _ in range(8)]
    half = NarrowDiT(
        hidden_size=D, depth=8, patch_size=2, in_channels=3, num_classes=10,
        input_size=32, per_block=pb, learn_sigma=False,
    )
    assert count_dit_params(half) < count_dit_params(full)


# --- Test 3: block_budgets ----------------------------------------------------

def test_block_budgets_variants():
    res = _synthetic_results()
    n_eff = res["n_eff"]
    phases = [(0, 7), (7, 14), (14, 20)]
    total = 1_000_000.0

    g = block_budgets(n_eff, [(0, 20)], total, "global")
    assert g == [total]

    u = block_budgets(n_eff, phases, total, "uniform_blockwise")
    assert len(u) == 3
    assert abs(sum(u) - total) < 1e-6
    assert max(u) - min(u) < 1e-6  # equal

    b = block_budgets(n_eff, phases, total, "blockwise_capacity")
    assert abs(sum(b) - total) < 1.0
    # Ordered by q90(n_eff) per phase.
    q = q90_block_scores(n_eff, phases)
    assert list(np.argsort(b)) == list(np.argsort(q))


# --- Test 4: match_uniform_cfg ------------------------------------------------

@pytest.mark.external_dit
def test_match_uniform_cfg_half_budget():
    teacher_params = count_dit_params(_teacher_narrow())
    target = 0.5 * teacher_params
    cfg = match_uniform_cfg(target, MICRO_TEACHER_CFG)
    model = NarrowDiT(**cfg)
    realized = count_dit_params(model)
    D = cfg["hidden_size"]
    # Valid uniform cfg: D divisible by num_heads, every block A == D.
    assert D % MICRO_TEACHER_CFG["num_heads"] == 0
    for pb in cfg["per_block"]:
        assert pb["attn_inner"] == D
    # Spec: accept a hit within 5%, else the CLOSEST grid point. At Micro scale the
    # quadratic param grid is coarse, so verify we returned the strictly closest D
    # over the whole search grid (rather than hard-coding a tolerance).
    H = MICRO_TEACHER_CFG["num_heads"]
    step = H * 8 // np.gcd(H, 8)  # lcm(num_heads, 8)
    D_teacher = MICRO_TEACHER_CFG["hidden_size"]
    grid = list(range(step, D_teacher + 1, step))
    if D_teacher not in grid:
        grid.append(D_teacher)
    dist = {}
    for d in grid:
        pb = [{"num_heads": H, "attn_inner": d, "mlp_hidden": int(d * MICRO_TEACHER_CFG["mlp_ratio"])}
              for _ in range(MICRO_TEACHER_CFG["depth"])]
        m = NarrowDiT(hidden_size=d, depth=MICRO_TEACHER_CFG["depth"], patch_size=2,
                      in_channels=3, num_classes=10, input_size=32, per_block=pb, learn_sigma=False)
        dist[d] = abs(count_dit_params(m) - target)
    closest = min(dist, key=dist.get)
    rel = abs(realized - target) / target
    assert rel <= 0.05 or D == closest, (D, closest, rel)


# --- Test 5: match_layerwise_cfg ----------------------------------------------

@pytest.mark.external_dit
def test_match_layerwise_cfg_budget_and_validity_and_determinism():
    teacher_params = count_dit_params(_teacher_narrow())
    # SMALL budget: the fix must shrink the base hidden_size to reach it, not floor
    # on the ~2.1M D-fixed adaLN/embedder overhead of the teacher-width model.
    phase_budget = 0.1 * teacher_params
    scores = np.array([1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0])
    cfg1 = match_layerwise_cfg(phase_budget, MICRO_TEACHER_CFG, scores)
    realized = count_dit_params(NarrowDiT(**cfg1))
    rel = abs(realized - phase_budget) / phase_budget
    assert rel <= 0.10, (realized, phase_budget, rel)
    H = MICRO_TEACHER_CFG["num_heads"]
    for pb in cfg1["per_block"]:
        assert pb["attn_inner"] >= H
        assert pb["attn_inner"] % H == 0
        assert pb["mlp_hidden"] >= 8
    # Redistribution actually happened: per-layer attn_inner is NOT all equal when
    # layer scores differ.
    attn_inners = [pb["attn_inner"] for pb in cfg1["per_block"]]
    assert len(set(attn_inners)) > 1, attn_inners
    # Determinism.
    cfg2 = match_layerwise_cfg(phase_budget, MICRO_TEACHER_CFG, scores)
    assert cfg1["per_block"] == cfg2["per_block"]


# --- Test 6: layer_scores shape and aggregation -------------------------------

def test_layer_scores_shape_and_aggregation():
    names = [f"blocks.{b}.attn.head_{h}" for b in range(8) for h in range(3)]
    # Deterministic rds: every head of block b has constant value (b+1) across bins.
    rds = np.zeros((24, 20))
    for i, nm in enumerate(names):
        b = int(nm.split(".")[1])
        rds[i, :] = b + 1
    phases = [(0, 10), (10, 20)]
    scores = layer_scores(rds, names, num_layers=8, phases=phases)
    assert scores.shape == (2, 8)
    # Block l has 3 heads each = (l+1); summed -> 3*(l+1); q90 of a constant = itself.
    for p in range(2):
        for l in range(8):
            assert abs(scores[p, l] - 3 * (l + 1)) < 1e-9


# --- Test 7: build_plan global -> phase 0 realized == teacher_params ----------

@pytest.mark.external_dit
def test_build_plan_global_realized_equals_teacher():
    res = _synthetic_results()
    phases = [(0, 7), (7, 14), (14, 20)]
    plan = build_plan(res, phases, MICRO_TEACHER_CFG, "global")
    assert len(plan["phases"]) == 1
    assert plan["phases"][0]["realized_params"] == plan["teacher_params"]


# --- Test 8: real Micro results -> all 4 plans build --------------------------

@pytest.mark.external_dit
def test_build_all_four_plans_from_real_micro_results(capsys):
    if not os.path.exists(_MICRO_RESULTS):
        pytest.skip("real Micro perm results not present")
    import json
    res = json.load(open(_MICRO_RESULTS))
    n_bins = len(res["n_eff"])
    # Three timestep phases over the 20 bins.
    phases = [(0, 7), (7, 14), (14, n_bins)]
    variants = ["global", "uniform_blockwise", "blockwise_capacity", "layerwise_capacity"]
    for v in variants:
        plan = build_plan(res, phases, MICRO_TEACHER_CFG, v)
        assert plan["teacher_params"] > 0
        realized = [p["realized_params"] for p in plan["phases"]]
        assert all(r > 0 for r in realized)
        with capsys.disabled():
            print(f"{v}: teacher={plan['teacher_params']} "
                  f"per-phase realized={realized}")


# --- Test 9: analytic param count == count_dit_params (Micro regression) -------

def _uniform_cfg(D, teacher_cfg):
    H = teacher_cfg["num_heads"]
    M = int(D * teacher_cfg["mlp_ratio"])
    pb = [{"num_heads": H, "attn_inner": D, "mlp_hidden": M}
          for _ in range(teacher_cfg["depth"])]
    return {
        "hidden_size": D, "depth": teacher_cfg["depth"],
        "patch_size": teacher_cfg["patch_size"], "in_channels": teacher_cfg["in_channels"],
        "num_classes": teacher_cfg["num_classes"], "input_size": teacher_cfg["input_size"],
        "learn_sigma": teacher_cfg["learn_sigma"], "per_block": pb,
    }


@pytest.mark.external_dit
def test_analytic_param_count_matches_count_dit_params_micro():
    # Micro teacher: analytic == count_dit_params == the pinned 5,498,892.
    teacher_cfg_full = _uniform_cfg(MICRO_TEACHER_CFG["hidden_size"], MICRO_TEACHER_CFG)
    real = count_dit_params(NarrowDiT(**teacher_cfg_full))
    ana = analytic_narrow_dit_params(teacher_cfg_full)
    assert real == ana == 5_498_892, (real, ana)

    # A couple of narrow Micro cfgs: uniform D=96 and a per-block-mixed cfg.
    cfg_96 = _uniform_cfg(96, MICRO_TEACHER_CFG)
    assert count_dit_params(NarrowDiT(**cfg_96)) == analytic_narrow_dit_params(cfg_96)

    pb_mixed = []
    for l in range(MICRO_TEACHER_CFG["depth"]):
        A = 24 if l % 2 == 0 else 96  # multiples of lcm(num_heads=3, 8) = 24
        M = 8 * (l + 1)
        pb_mixed.append({"num_heads": 3, "attn_inner": A, "mlp_hidden": M})
    cfg_mixed = dict(_uniform_cfg(96, MICRO_TEACHER_CFG), per_block=pb_mixed)
    assert count_dit_params(NarrowDiT(**cfg_mixed)) == analytic_narrow_dit_params(cfg_mixed)


@pytest.mark.external_dit
def test_analytic_param_count_matches_count_dit_params_smicro():
    # teacher_v4_smicro (D=384/depth12): analytic == count_dit_params == the
    # pinned 32,475,660 (== teacher_v2_plans.json's "global_s" teacher_params,
    # == the teacher_v4_smicro run's reported 32,479,116 realized minus 9*384 for
    # augment_dim=9, which is threaded separately and not part of this cfg).
    teacher_cfg_full = _uniform_cfg(SMICRO_TEACHER_CFG["hidden_size"], SMICRO_TEACHER_CFG)
    real = count_dit_params(NarrowDiT(**teacher_cfg_full))
    ana = analytic_narrow_dit_params(teacher_cfg_full)
    assert real == ana == 32_475_660, (real, ana)


@pytest.mark.external_dit
def test_narrowdit_no_train_label_dropout_keeps_null_token():
    import torch
    from pace.dit_arch_alloc import build_narrow_dit, MICRO_TEACHER_CFG, _uniform_per_block
    cfg = dict(MICRO_TEACHER_CFG)
    cfg["per_block"] = _uniform_per_block(cfg["hidden_size"], cfg["depth"], cfg["num_heads"], cfg["mlp_ratio"])
    m = build_narrow_dit(cfg).train()  # training mode
    assert m.y_embedder.embedding_table.weight.shape[0] == cfg["num_classes"] + 1  # null token present
    y = torch.arange(cfg["num_classes"]).long()
    torch.manual_seed(0); e_train = m.y_embedder(y, train=True)
    e_eval = m.y_embedder(y, train=False)
    assert torch.equal(e_train, e_eval), "training-time label dropout still active"
    x = torch.randn(4, cfg["in_channels"], 32, 32); t = torch.rand(4)
    out = m.forward_with_cfg(x, t, torch.arange(4).long(), 1.5)   # CFG path needs null token
    # MICRO_TEACHER_CFG has no "out_channels" key; learn_sigma=False -> out == in_channels.
    assert out.shape == (4, m.out_channels, 32, 32)


# --- Test 10: match_layerwise_cfg g_max (regression) ----------------------------
#
# match_layerwise_cfg's per-layer redistribution
# weight sqrt(score/mean_score) is concave, so by Jensen's inequality its layer-sum
# is systematically BELOW the uniform-D sum whenever scores are dispersed across
# layers -- the g-binary-search must push g well above 1.0 to compensate, but
# high-importance layers are clamped at the uniform-D ceiling (a layer's width can
# never exceed what it would have in a uniform model at that D), so once enough
# layers saturate that ceiling only the unsaturated minority can keep absorbing a
# rising g. With the (default, historical) g_max=1.75 -- tuned for CIFAR-10-scale
# score dispersion -- this silently saturates before re-hitting phase_budget for
# more skewed distributions, producing a large UNDISCLOSED parameter deficit for
# layerwise_capacity vs blockwise_capacity/uniform_blockwise (up to ~24% on real
# DiT-Micro data). Raising g_max (opt-in; the default is unchanged so existing
# plans.json stay byte-identical) closes the gap.

@pytest.mark.external_dit
def test_match_layerwise_cfg_default_g_max_unchanged():
    """g_max defaults to 1.75 -- calling without it must be byte-identical."""
    teacher_params = count_dit_params(_teacher_narrow())
    scores = np.array([0.001, 0.001, 0.001, 0.001, 5.0, 5.0, 5.0, 5.0])
    budget = 0.5 * teacher_params
    cfg_implicit = match_layerwise_cfg(budget, MICRO_TEACHER_CFG, scores)
    cfg_explicit = match_layerwise_cfg(budget, MICRO_TEACHER_CFG, scores, g_max=1.75)
    assert cfg_implicit == cfg_explicit


@pytest.mark.external_dit
def test_match_layerwise_cfg_skewed_scores_undershoot_and_g_max_fix():
    teacher_params = count_dit_params(_teacher_narrow())
    # Skewed per-layer importance (4 near-zero, 4 dominant), the pattern observed in
    # real DiT-Micro permutation-importance results (a large contiguous fraction of
    # layers pinned at the width ceiling).
    scores = np.array([0.001, 0.001, 0.001, 0.001, 5.0, 5.0, 5.0, 5.0])
    budget = 0.5 * teacher_params

    # Default g_max=1.75: the search saturates and silently undershoots by >15%,
    # with NO error raised (build_plan only checks analytic-vs-instantiated counts,
    # never realized-vs-target budget).
    cfg_default = match_layerwise_cfg(budget, MICRO_TEACHER_CFG, scores)
    realized_default = count_dit_params(NarrowDiT(**cfg_default))
    rel_err_default = (realized_default - budget) / budget
    assert rel_err_default < -0.15, (realized_default, budget, rel_err_default)

    # Opt-in wider bracket: same scores/budget, closes the gap to <1%.
    cfg_wide = match_layerwise_cfg(budget, MICRO_TEACHER_CFG, scores, g_max=50.0)
    realized_wide = count_dit_params(NarrowDiT(**cfg_wide))
    rel_err_wide = abs(realized_wide - budget) / budget
    assert rel_err_wide < 0.01, (realized_wide, budget, rel_err_wide)


@pytest.mark.external_dit
def test_build_plan_layerwise_g_max_threads_through():
    res = _synthetic_results()
    phases = [(0, 7), (7, 14), (14, 20)]
    plan_default = build_plan(res, phases, MICRO_TEACHER_CFG, "layerwise_capacity")
    plan_explicit = build_plan(res, phases, MICRO_TEACHER_CFG, "layerwise_capacity",
                                layerwise_g_max=1.75)
    assert plan_default == plan_explicit
    # A wider bracket is accepted and produces a (potentially different) valid plan.
    plan_wide = build_plan(res, phases, MICRO_TEACHER_CFG, "layerwise_capacity",
                            layerwise_g_max=50.0)
    assert len(plan_wide["phases"]) == 3


# --- Test 11: layer_scores eps floor (regression) --------------------------------
#
# DiT-Micro's phase 0 ([0,4) bins,
# the 4 HIGHEST-noise timesteps) has 4/8 transformer blocks with an EXACTLY-ZERO
# measured importance score, traced to the permutation-ablation clip
# (clamp(ablated_mean - baseline_mean, min=0.0)) flooring noise-dominated deltas
# (86.5%% of that phase's raw measurements are clipped, vs 0-52%% elsewhere) --
# a hard 0.0 score permanently floors match_layerwise_cfg's per-layer width at the
# minimum grid step regardless of g_max (0 * g == 0 for every finite g). The
# coordinator-approved fix: an opt-in per-phase epsilon floor,
# score'_l = score_l + eps * phase_mean(score), so the redistribution degrades
# toward uniform allocation instead of a permanent floor when a phase's true
# signal is at the noise floor.

def test_layer_scores_eps_default_unchanged():
    """eps defaults to 0.0 -- calling without it must be byte-identical."""
    names = [f"blocks.{b}.attn.head_{h}" for b in range(8) for h in range(3)]
    rng = np.random.default_rng(1)
    rds = rng.random((24, 20))
    phases = [(0, 7), (7, 14), (14, 20)]
    implicit = layer_scores(rds, names, num_layers=8, phases=phases)
    explicit = layer_scores(rds, names, num_layers=8, phases=phases, eps=0.0)
    assert np.array_equal(implicit, explicit)


@pytest.mark.external_dit
def test_layer_scores_eps_floors_hard_zero_and_recovers_budget():
    # Mirrors the real Micro phase-0 pattern: 4 of 8 blocks exactly zero, 4
    # nonzero (values on the same scale observed in the real data).
    names = [f"blocks.{b}.attn.head_{h}" for b in range(8) for h in range(3)]
    n_bins = 4
    rds = np.zeros((24, n_bins))
    nonzero_blocks = {0: 0.0058, 4: 0.0048, 5: 0.0018, 6: 0.0177}
    for b, val in nonzero_blocks.items():
        # split evenly across the block's 3 heads so layer_scores' per-block SUM
        # reproduces `val` after aggregation.
        for h in range(3):
            rds[b * 3 + h, :] = val / 3.0
    phases = [(0, n_bins)]

    scores_eps0 = layer_scores(rds, names, num_layers=8, phases=phases, eps=0.0)[0]
    zero_layers = [l for l in range(8) if l not in nonzero_blocks]
    assert all(scores_eps0[l] == 0.0 for l in zero_layers), scores_eps0

    teacher_params = count_dit_params(_teacher_narrow())
    budget = 0.5 * teacher_params

    # eps=0: the zero layers stay floored at the grid minimum for EVERY g_max --
    # widening g_max does not help (regression-pins the bug's existence).
    cfg_small_gmax = match_layerwise_cfg(budget, MICRO_TEACHER_CFG, scores_eps0, g_max=2.5)
    cfg_big_gmax = match_layerwise_cfg(budget, MICRO_TEACHER_CFG, scores_eps0, g_max=200.0)
    for cfg in (cfg_small_gmax, cfg_big_gmax):
        for l in zero_layers:
            assert cfg["per_block"][l]["attn_inner"] == 24  # A_step = lcm(3, 8), the floor

    # eps>0 (same g_max as the small-g_max case above): the floored layers gain
    # real width, and the realized total moves meaningfully closer to budget --
    # regression-pins the fix's efficacy without requiring exact numeric parity
    # with the full real-data sweep (see the real-data test below for that).
    scores_eps = layer_scores(rds, names, num_layers=8, phases=phases, eps=0.2)[0]
    cfg_eps = match_layerwise_cfg(budget, MICRO_TEACHER_CFG, scores_eps, g_max=2.5)
    assert any(cfg_eps["per_block"][l]["attn_inner"] > 24 for l in zero_layers), cfg_eps
    realized_noeps = count_dit_params(NarrowDiT(**cfg_small_gmax))
    realized_eps = count_dit_params(NarrowDiT(**cfg_eps))
    assert abs(realized_eps - budget) < abs(realized_noeps - budget)


def test_layer_scores_eps_barely_moves_real_dynamic_range_phases():
    """Where a phase already has real, nonzero dynamic range (no clipped
    layers), a small eps must change scores by only a tiny relative amount."""
    names = [f"blocks.{b}.attn.head_{h}" for b in range(8) for h in range(3)]
    rng = np.random.default_rng(2)
    rds = rng.uniform(0.01, 1.0, size=(24, 20))  # no zeros anywhere
    phases = [(0, 20)]
    s0 = layer_scores(rds, names, num_layers=8, phases=phases, eps=0.0)[0]
    s_eps = layer_scores(rds, names, num_layers=8, phases=phases, eps=0.05)[0]
    rel_change = np.abs(s_eps - s0) / s0
    assert np.all(rel_change < 0.10), rel_change  # eps=0.05 -> well under 10%% shift


@pytest.mark.external_dit
def test_build_plan_layer_score_eps_threads_through():
    res = _synthetic_results()
    phases = [(0, 7), (7, 14), (14, 20)]
    plan_default = build_plan(res, phases, MICRO_TEACHER_CFG, "layerwise_capacity")
    plan_explicit = build_plan(res, phases, MICRO_TEACHER_CFG, "layerwise_capacity",
                                layer_score_eps=0.0)
    assert plan_default == plan_explicit
    plan_eps = build_plan(res, phases, MICRO_TEACHER_CFG, "layerwise_capacity",
                           layer_score_eps=0.05)
    assert len(plan_eps["phases"]) == 3


@pytest.mark.external_dit
def test_layer_score_eps_real_micro_phase0_reaches_budget_at_chosen_params():
    """Real-data regression: eps=0.05 + g_max=4.0 (the values chosen for the
    DiT-Micro plans) get phase 0 within 3%% of budget, vs. the structurally
    stuck -23.9%% of eps=0/g_max=2.5."""
    if not os.path.exists(_MICRO_RESULTS):
        pytest.skip("real Micro perm results not present")
    import json
    res = json.load(open(_MICRO_RESULTS))
    phases = [(0, 4), (4, 8), (8, 16), (16, 20)]
    teacher_params = count_dit_params(_teacher_narrow())
    budgets = block_budgets(res["n_eff"], phases, teacher_params, "layerwise_capacity")

    scores_old = layer_scores(res["relative_delta_stack"], res["group_names"], 8, phases, eps=0.0)
    cfg_old = match_layerwise_cfg(budgets[0], MICRO_TEACHER_CFG, scores_old[0], g_max=2.5)
    realized_old = count_dit_params(NarrowDiT(**cfg_old))
    assert (realized_old - budgets[0]) / budgets[0] < -0.15  # pins the old bug

    scores_new = layer_scores(res["relative_delta_stack"], res["group_names"], 8, phases, eps=0.05)
    cfg_new = match_layerwise_cfg(budgets[0], MICRO_TEACHER_CFG, scores_new[0], g_max=4.0)
    realized_new = count_dit_params(NarrowDiT(**cfg_new))
    rel_err_new = abs(realized_new - budgets[0]) / budgets[0]
    assert rel_err_new <= 0.03, (realized_new, budgets[0], rel_err_new)

    # Phases 1-3 (already-real dynamic range, 0%% clipped) barely move.
    for p in range(1, 4):
        cfg_p_old = match_layerwise_cfg(budgets[p], MICRO_TEACHER_CFG, scores_old[p], g_max=2.5)
        cfg_p_new = match_layerwise_cfg(budgets[p], MICRO_TEACHER_CFG, scores_new[p], g_max=4.0)
        r_old = count_dit_params(NarrowDiT(**cfg_p_old))
        r_new = count_dit_params(NarrowDiT(**cfg_p_new))
        assert abs(r_new - r_old) / budgets[p] < 0.01, (p, r_old, r_new)


# --- Test 12: match_blockwise_budget_cfg (regression) ----------------------------
#
# match_uniform_cfg picks a single grid D by closest ABSOLUTE distance to the
# phase target; Micro's D-grid has only 8 points and the gaps between them grow
# with D, so blockwise_capacity realizes +21.9%%/+17.8%% in its two worst phases.
# match_blockwise_budget_cfg keeps one-width-per-phase semantics but trims that
# shared width below the grid ceiling (reusing match_layerwise_cfg's g-search
# with a FLAT per-layer score vector) to land within a few percent of budget.

@pytest.mark.external_dit
def test_match_blockwise_budget_cfg_uniform_across_layers():
    teacher_params = count_dit_params(_teacher_narrow())
    budget = 0.35 * teacher_params
    cfg = match_blockwise_budget_cfg(budget, MICRO_TEACHER_CFG)
    attn_inners = {pb["attn_inner"] for pb in cfg["per_block"]}
    mlp_hiddens = {pb["mlp_hidden"] for pb in cfg["per_block"]}
    # Defining property preserved: ONE shared width across every layer (unlike
    # layerwise_capacity, which varies per layer by importance).
    assert len(attn_inners) == 1 and len(mlp_hiddens) == 1, cfg["per_block"]


@pytest.mark.external_dit
def test_match_blockwise_budget_cfg_beats_match_uniform_cfg_in_a_wide_grid_gap():
    """Real Micro phase-1 target (1,144,021) sits in match_uniform_cfg's worst
    grid gap (+21.9%% overshoot);
    match_blockwise_budget_cfg must land within 3%% of the same target."""
    target = 1_144_020.6951895305
    cfg_naive = match_uniform_cfg(target, MICRO_TEACHER_CFG)
    realized_naive = count_dit_params(NarrowDiT(**cfg_naive))
    rel_err_naive = (realized_naive - target) / target
    assert rel_err_naive > 0.15, (realized_naive, target, rel_err_naive)  # pins the bug

    cfg_fixed = match_blockwise_budget_cfg(target, MICRO_TEACHER_CFG)
    realized_fixed = count_dit_params(NarrowDiT(**cfg_fixed))
    rel_err_fixed = abs(realized_fixed - target) / target
    assert rel_err_fixed <= 0.03, (realized_fixed, target, rel_err_fixed)


@pytest.mark.external_dit
def test_build_plan_blockwise_budget_match_threads_through():
    res = _synthetic_results()
    phases = [(0, 7), (7, 14), (14, 20)]

    plan_default = build_plan(res, phases, MICRO_TEACHER_CFG, "blockwise_capacity")
    plan_explicit_off = build_plan(res, phases, MICRO_TEACHER_CFG, "blockwise_capacity",
                                    blockwise_budget_match=False)
    assert plan_default == plan_explicit_off

    plan_matched = build_plan(res, phases, MICRO_TEACHER_CFG, "blockwise_capacity",
                               blockwise_budget_match=True)
    assert len(plan_matched["phases"]) == 3
    for ph_default, ph_matched in zip(plan_default["phases"], plan_matched["phases"]):
        err_default = abs(ph_default["realized_params"] - ph_default["target_params"]) / ph_default["target_params"]
        err_matched = abs(ph_matched["realized_params"] - ph_matched["target_params"]) / ph_matched["target_params"]
        assert err_matched <= err_default + 1e-9

    # uniform_blockwise is NEVER affected by the flag (only blockwise_capacity is).
    plan_u_off = build_plan(res, phases, MICRO_TEACHER_CFG, "uniform_blockwise",
                             blockwise_budget_match=False)
    plan_u_on = build_plan(res, phases, MICRO_TEACHER_CFG, "uniform_blockwise",
                            blockwise_budget_match=True)
    assert plan_u_off == plan_u_on


@pytest.mark.external_dit
def test_blockwise_budget_match_real_micro_all_phases_within_3pct():
    if not os.path.exists(_MICRO_RESULTS):
        pytest.skip("real Micro perm results not present")
    import json
    res = json.load(open(_MICRO_RESULTS))
    phases = [(0, 4), (4, 8), (8, 16), (16, 20)]
    plan = build_plan(res, phases, MICRO_TEACHER_CFG, "blockwise_capacity",
                       blockwise_budget_match=True, blockwise_budget_g_max=50.0)
    for ph in plan["phases"]:
        rel_err = abs(ph["realized_params"] - ph["target_params"]) / ph["target_params"]
        assert rel_err <= 0.03, (ph["phase"], ph["realized_params"], ph["target_params"], rel_err)


# --- Test 13: uniform_budget_match (independent opt-in fix for uniform_blockwise) --
#
# The SAME D-grid gap match_blockwise_budget_cfg fixes for blockwise_capacity is a
# property of the teacher's (hidden_size, num_heads) grid, not of which variant's
# target lands in it -- at some (teacher, grouping) combinations uniform_blockwise's
# own equal-per-phase target can land in just as wide a gap (the CIFAR-10 DiT
# teacher, D=384/heads=6, 2-phase 50/50 split -> 94.8%% realized, outside a
# +/-3%% budget tolerance). uniform_budget_match applies the identical fix,
# as an INDEPENDENT flag from blockwise_budget_match, so the existing invariant
# ("uniform_blockwise is never affected by blockwise_budget_match", tested above)
# stays exactly true.

@pytest.mark.external_dit
def test_uniform_budget_match_default_off_is_byte_identical():
    res = _synthetic_results()
    phases = [(0, 7), (7, 14), (14, 20)]
    plan_default = build_plan(res, phases, MICRO_TEACHER_CFG, "uniform_blockwise")
    plan_explicit_off = build_plan(res, phases, MICRO_TEACHER_CFG, "uniform_blockwise",
                                    uniform_budget_match=False)
    assert plan_default == plan_explicit_off


@pytest.mark.external_dit
def test_uniform_budget_match_independent_of_blockwise_budget_match():
    """Setting blockwise_budget_match alone must still leave uniform_blockwise
    unaffected (re-asserts the existing invariant survives this change); setting
    uniform_budget_match alone must leave blockwise_capacity unaffected (the
    mirror-image guarantee)."""
    res = _synthetic_results()
    phases = [(0, 7), (7, 14), (14, 20)]

    plan_u_default = build_plan(res, phases, MICRO_TEACHER_CFG, "uniform_blockwise")
    plan_u_with_bbm = build_plan(res, phases, MICRO_TEACHER_CFG, "uniform_blockwise",
                                  blockwise_budget_match=True)
    assert plan_u_default == plan_u_with_bbm

    plan_b_default = build_plan(res, phases, MICRO_TEACHER_CFG, "blockwise_capacity")
    plan_b_with_ubm = build_plan(res, phases, MICRO_TEACHER_CFG, "blockwise_capacity",
                                  uniform_budget_match=True)
    assert plan_b_default == plan_b_with_ubm


@pytest.mark.external_dit
def test_uniform_budget_match_beats_match_uniform_cfg_in_a_wide_grid_gap():
    """The CIFAR-10 DiT teacher case that motivated this flag: D=384/heads=6,
    a 2-phase 50/50 split's target sits between D=264 (94.8%%) and D=288 (112.7%%)
    -- match_uniform_cfg cannot clear +/-3%%; match_blockwise_budget_cfg (via
    uniform_budget_match) must."""
    res = _synthetic_results()  # 8-block Micro-shaped synthetic; reuse SMICRO shape below
    phases = [(0, 10), (10, 20)]
    plan_naive = build_plan(res, phases, SMICRO_TEACHER_CFG, "uniform_blockwise")
    plan_fixed = build_plan(res, phases, SMICRO_TEACHER_CFG, "uniform_blockwise",
                             uniform_budget_match=True)
    for ph in plan_naive["phases"]:
        rel_err = abs(ph["realized_params"] - ph["target_params"]) / ph["target_params"]
        assert rel_err > 0.03, (ph["phase"], ph["realized_params"], ph["target_params"], rel_err)
    for ph in plan_fixed["phases"]:
        rel_err = abs(ph["realized_params"] - ph["target_params"]) / ph["target_params"]
        assert rel_err <= 0.03, (ph["phase"], ph["realized_params"], ph["target_params"], rel_err)


@pytest.mark.external_dit
def test_uniform_budget_match_preserves_uniform_width_property():
    res = _synthetic_results()
    phases = [(0, 10), (10, 20)]
    plan = build_plan(res, phases, SMICRO_TEACHER_CFG, "uniform_blockwise",
                       uniform_budget_match=True)
    for ph in plan["phases"]:
        attn_inners = {pb["attn_inner"] for pb in ph["cfg"]["per_block"]}
        mlp_hiddens = {pb["mlp_hidden"] for pb in ph["cfg"]["per_block"]}
        assert len(attn_inners) == 1 and len(mlp_hiddens) == 1, ph["cfg"]["per_block"]


# --- Test 14: phase_budget_agg (opt-in phase-budget aggregator) ----------------
#
# block_budgets historically split the teacher budget across phases proportional
# to q90(n_eff over phase bins)^alpha (blockwise_capacity/layerwise_capacity
# only). The opt-in ``agg`` parameter ("q90" default / "geomean" / "mean") adds
# a geometric-mean and an arithmetic-mean option. Hard rule: the default path must stay
# bit-identical (it still routes through q90_block_scores/np.quantile verbatim),
# and the JSON provenance key appears ONLY when the flag is non-default.

def test_block_budgets_phase_budget_agg_default_unchanged():
    """agg defaults to "q90" -- calling with and without it must be bit-identical,
    and must equal the historical q90_block_scores-based split exactly."""
    res = _synthetic_results()
    n_eff = res["n_eff"]
    phases = [(0, 7), (7, 14), (14, 20)]
    total = 1_000_000.0
    for variant in ("global", "uniform_blockwise", "blockwise_capacity", "layerwise_capacity"):
        implicit = block_budgets(n_eff, phases if variant != "global" else [(0, 20)],
                                 total, variant)
        explicit = block_budgets(n_eff, phases if variant != "global" else [(0, 20)],
                                 total, variant, agg="q90")
        assert implicit == explicit, variant  # bit-identical, not just close
    # And the capacity split still IS the q90 split (same np.quantile path).
    b = block_budgets(n_eff, phases, total, "blockwise_capacity", agg="q90")
    q = q90_block_scores(n_eff, phases)
    expected = [total * float(qi) / float(q.sum()) for qi in q]
    assert b == expected

    with pytest.raises(ValueError):
        block_budgets(n_eff, phases, total, "blockwise_capacity", agg="median")


def test_block_budgets_geomean_and_mean_hand_computed():
    """Hand-computable n_eff: [2,8 | 4,4] -> geomean 4 and 4 (equal split), mean
    5 and 4 (5:4 split), q90 7.4 and 4 (different from both)."""
    n_eff = [2.0, 8.0, 4.0, 4.0]
    phases = [(0, 2), (2, 4)]
    total = 900_000.0

    gm = block_budgets(n_eff, phases, total, "blockwise_capacity", agg="geomean")
    # geomean([2,8]) = sqrt(16) = 4 == geomean([4,4]) -> exactly equal budgets.
    assert gm[0] == pytest.approx(total / 2, rel=1e-12)
    assert gm[1] == pytest.approx(total / 2, rel=1e-12)

    am = block_budgets(n_eff, phases, total, "blockwise_capacity", agg="mean")
    assert am[0] == pytest.approx(total * 5.0 / 9.0, rel=1e-12)
    assert am[1] == pytest.approx(total * 4.0 / 9.0, rel=1e-12)

    q = block_budgets(n_eff, phases, total, "blockwise_capacity", agg="q90")
    # q90([2,8]) = 2 + 0.9*6 = 7.4 ; q90([4,4]) = 4.
    assert q[0] == pytest.approx(total * 7.4 / 11.4, rel=1e-12)

    # Zero-valued bin: geomean must clamp via max(x, 1e-9), not blow up on log(0).
    n_eff_zero = [0.0, 4.0, 1.0, 1.0]
    gz = block_budgets(n_eff_zero, phases, total, "blockwise_capacity", agg="geomean")
    g0 = float(np.exp(0.5 * (np.log(1e-9) + np.log(4.0))))  # = sqrt(4e-9)
    assert gz[0] == pytest.approx(total * g0 / (g0 + 1.0), rel=1e-9)


@pytest.mark.external_dit
def test_build_plan_phase_budget_agg_threads_and_annotates():
    """Default build_plan output has NO phase_budget_agg key (byte-identical
    structure); non-default agg changes capacity budgets and adds the key."""
    res = _synthetic_results()
    phases = [(0, 7), (7, 14), (14, 20)]

    plan_default = build_plan(res, phases, MICRO_TEACHER_CFG, "blockwise_capacity")
    plan_explicit = build_plan(res, phases, MICRO_TEACHER_CFG, "blockwise_capacity",
                                phase_budget_agg="q90")
    assert plan_default == plan_explicit
    assert "phase_budget_agg" not in plan_default
    assert "phase_budget_agg" not in plan_explicit

    plan_gm = build_plan(res, phases, MICRO_TEACHER_CFG, "blockwise_capacity",
                          phase_budget_agg="geomean")
    assert plan_gm["phase_budget_agg"] == "geomean"
    # Targets actually moved off the q90 split for this (dispersed) synthetic n_eff.
    t_q90 = [p["target_params"] for p in plan_default["phases"]]
    t_gm = [p["target_params"] for p in plan_gm["phases"]]
    assert t_q90 != t_gm
    assert sum(t_gm) == pytest.approx(plan_gm["teacher_params"], rel=1e-9)

    # global/uniform_blockwise budgets never consult n_eff: only the provenance
    # key may differ under a non-default agg.
    for v in ("global", "uniform_blockwise"):
        p_def = build_plan(res, phases, MICRO_TEACHER_CFG, v)
        p_gm = build_plan(res, phases, MICRO_TEACHER_CFG, v, phase_budget_agg="geomean")
        assert p_gm.pop("phase_budget_agg") == "geomean"
        assert p_def == p_gm


@pytest.mark.external_dit
def test_dit_arch_to_plans_phase_budget_agg_flag(tmp_path, monkeypatch):
    """The --phase_budget_agg flag parses, threads through, and annotates the
    output JSON ONLY when non-default."""
    import json
    import dit_arch_to_plans as script  # scripts/ is on sys.path (see header)

    res = _synthetic_results()
    results_json = tmp_path / "results.json"
    results_json.write_text(json.dumps(res))
    grouping_json = tmp_path / "grouping.json"
    grouping_json.write_text(json.dumps({"boundaries": [0, 7, 14, 20]}))

    def run(out_name, *extra):
        out = tmp_path / out_name
        argv = ["dit_arch_to_plans.py",
                "--results_json", str(results_json),
                "--grouping_json", str(grouping_json),
                "--model", "dit_micro",
                "--variants", "global,blockwise_capacity",
                "--out", str(out), *extra]
        monkeypatch.setattr(sys, "argv", argv)
        script.main()
        return json.load(open(out))

    out_default = run("plans_default.json")
    out_q90 = run("plans_q90.json", "--phase_budget_agg", "q90")
    out_gm = run("plans_gmean.json", "--phase_budget_agg", "geomean")

    # Default == explicit q90, and neither carries the provenance key anywhere.
    assert out_default == out_q90
    for v, plan in out_default.items():
        assert "phase_budget_agg" not in plan, v

    # Non-default: every variant dict is annotated; capacity targets moved.
    for v, plan in out_gm.items():
        assert plan["phase_budget_agg"] == "geomean", v
    assert ([p["target_params"] for p in out_gm["blockwise_capacity"]["phases"]]
            != [p["target_params"] for p in out_default["blockwise_capacity"]["phases"]])
    # global is agg-independent apart from the annotation.
    gm_global = dict(out_gm["global"])
    gm_global.pop("phase_budget_agg")
    assert gm_global == out_default["global"]


@pytest.mark.external_dit
def test_analytic_param_count_matches_count_dit_params_xl_small():
    # XL uses learn_sigma=True + in_channels=4 + num_classes=1000; verify the analytic
    # formula matches a REAL build for a SMALL-depth XL-flavored cfg (depth kept tiny so
    # the CPU build is cheap; the formula is what matters, not the size).
    xl_like = {
        "hidden_size": 128, "depth": 2, "patch_size": 2, "in_channels": 4,
        "num_classes": 1000, "input_size": 32, "learn_sigma": True,
        "per_block": [{"num_heads": 16, "attn_inner": 128, "mlp_hidden": 512},
                      {"num_heads": 16, "attn_inner": 64, "mlp_hidden": 256}],
    }
    assert count_dit_params(NarrowDiT(**xl_like)) == analytic_narrow_dit_params(xl_like)


# ---------------------------------------------------------------------------
# DITB_TEACHER_CFG (objective-ablation pretrain: a from-scratch, unconditional
# DiT-B/2-dims NarrowDiT for --model_type dit_xl --diffusion ddpm --arch_plan
# --variant global).
# ---------------------------------------------------------------------------

@pytest.mark.external_dit
def test_ditb_teacher_cfg_global_plan_realizes_dit_b_param_count():
    from dit_arch_to_plans import DITB_TEACHER_CFG  # scripts/ is on sys.path

    assert DITB_TEACHER_CFG["hidden_size"] == 768
    assert DITB_TEACHER_CFG["depth"] == 12
    assert DITB_TEACHER_CFG["num_heads"] == 12
    assert DITB_TEACHER_CFG["num_classes"] == 1        # unconditional, y=0 convention
    assert DITB_TEACHER_CFG["learn_sigma"] is True      # matches the ddpm/dit_xl branch

    res = _synthetic_results()  # n_eff length is irrelevant to the "global" variant
    plan = build_plan(res, [(0, 20)], DITB_TEACHER_CFG, "global")
    assert len(plan["phases"]) == 1
    phase = plan["phases"][0]
    assert phase["realized_params"] == 129_548_576
    assert plan["teacher_params"] == phase["realized_params"]

    # The realized cfg actually builds (real NarrowDiT, not just the analytic count).
    model = NarrowDiT(**phase["cfg"])
    assert count_dit_params(model) == 129_548_576
    x = torch.randn(2, 4, 32, 32)  # input_size=32 (32x32 latents, patch_size=2)
    t = torch.rand(2)
    y = torch.zeros(2, dtype=torch.long)  # forced-unconditional convention
    out = model(x, t, y)
    assert out.shape == (2, 8, 32, 32)  # learn_sigma=True -> 2*in_channels


# --- head_dim / d_mult / m_mult / cost_fn knobs -------------------------------------

@pytest.mark.external_dit
def test_build_plan_knobs_default_off_is_byte_identical_and_thread_through():
    """Omitting head_dim/d_mult/m_mult (or passing their defaults) yields the exact
    legacy plan dict; when set, they shape every per-block cfg as documented."""
    res = _synthetic_results()
    phases = [(0, 7), (7, 14), (14, 20)]
    for v in ("uniform_blockwise", "blockwise_capacity", "layerwise_capacity"):
        p_def = build_plan(res, phases, MICRO_TEACHER_CFG, v)
        p_exp = build_plan(res, phases, MICRO_TEACHER_CFG, v, head_dim=None, d_mult=None, m_mult=8)
        assert p_def == p_exp
        for ph in p_def["phases"]:
            assert ph["target_params"] is not None
            assert "realized_cost" not in ph and "target_cost" not in ph

    hd = 32
    teacher_D = int(MICRO_TEACHER_CFG["hidden_size"])
    p_knobs = build_plan(res, phases, MICRO_TEACHER_CFG, "layerwise_capacity",
                         head_dim=hd, d_mult=64, m_mult=64)
    for ph in p_knobs["phases"]:
        cfg = ph["cfg"]
        assert cfg["hidden_size"] % 64 == 0 or cfg["hidden_size"] == teacher_D
        for pb in cfg["per_block"]:
            assert pb["attn_inner"] == pb["num_heads"] * hd  # whole heads of dim hd
            assert pb["mlp_hidden"] % 64 == 0


@pytest.mark.external_dit
def test_build_plan_cost_fn_mode_records_costs_and_hits_composite_target():
    """With cost_fn the budget unit is that cost: phases carry target_cost/realized_cost
    (target_params None), cost_target is mandatory, and the bin-weighted composite
    realized cost lands on cost_target."""
    res = _synthetic_results()
    phases = [(0, 7), (7, 14), (14, 20)]

    def cost_fn(D, per_block):  # smooth, monotone proxy "ms/step" with a fixed floor
        return 1.0 + sum(D * (pb["attn_inner"] + pb["mlp_hidden"]) for pb in per_block) / 1e6

    cost_fn.floor = 1.0
    with pytest.raises(ValueError):
        build_plan(res, phases, MICRO_TEACHER_CFG, "blockwise_capacity", cost_fn=cost_fn)
    with pytest.raises(ValueError):
        build_plan(res, phases, MICRO_TEACHER_CFG, "blockwise_capacity", cost_fn=cost_fn,
                   cost_target=0.5)  # below the grid floor

    target = 2.0
    plan = build_plan(res, phases, MICRO_TEACHER_CFG, "blockwise_capacity", cost_fn=cost_fn,
                      cost_target=target, blockwise_budget_match=True)
    for ph in plan["phases"]:
        assert ph["target_params"] is None
        assert ph["target_cost"] > cost_fn.floor
        assert ph["realized_cost"] == pytest.approx(
            cost_fn(ph["cfg"]["hidden_size"], ph["cfg"]["per_block"]))
    weights = [(e - s) / 20 for (s, e) in phases]
    composite = sum(w * ph["realized_cost"] for w, ph in zip(weights, plan["phases"]))
    assert composite == pytest.approx(target, rel=0.1)
