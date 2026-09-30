import hashlib
import json
import os
import subprocess
import sys
import tarfile
import types
import urllib.error
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import pace.edm_distillation as distillation
from pace.jsonio import find_json_file
from pace.edm_distillation import (
    BlockwiseEDMStudent,
    NarrowEDMPrecond,
    StudentArchitectureReport,
    _compile_extra_arch_candidate_student,
    _uniform_architecture_specs,
    compile_layerwise_student,
    compile_uniform_student,
    construct_student_from_kwargs,
    count_grouped_parameters,
    default_model_kwargs,
    hybrid_distillation_loss,
    infer_img_resolution,
    infer_label_dim,
    infer_model_family,
    is_edm_group_norm_safe,
    load_json,
    group_original_structural_budgets,
    prepare_distillation_architectures,
    build_uniform_candidate_table,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO_ROOT / "artifacts"
CIFAR_PROFILE = ARTIFACTS / "profiles" / "cifar10_ddpmpp_random_same_norm.json.gz"
IMAGENET_PROFILE = ARTIFACTS / "profiles" / "imagenet64_adm_random_same_norm.json.gz"
CIFAR_ALLOCATIONS = ARTIFACTS / "allocations" / "cifar10"
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import train_edm_distillation as train_module
from scripts.data import convert_imagenet_parquet_to_webdataset as convert_wds
from train_edm_distillation import (
    BestMetricTracker,
    FidReferenceConfig,
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    TensorBoardLogger,
    ValidationEarlyStopper,
    append_jsonl,
    build_checkpoint_selection,
    build_arg_parser,
    create_tensorboard_logger,
    evaluate_validation_loss,
    load_jsonl_rows,
    prune_periodic_snapshots,
    repair_resume_log,
    require_divisible_global_batch,
    resolve_resume_snapshot,
    restore_model_snapshot,
    restore_training_state,
    run_fid_evaluation,
    save_snapshot,
    save_training_state,
    should_run_interval,
    validate_resume_log,
)


def _canonical_name_digest(names):
    payload = json.dumps(sorted(names), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stratified_sampling(group_names, expansion_weights):
    module_counts = {}
    population_names = []
    probabilities = {}
    expansions = {}
    for index, (name, raw_weight) in enumerate(zip(group_names, expansion_weights)):
        population_count = int(raw_weight)
        module_name = f"fixture_module_{index}"
        module_counts[module_name] = {
            "population_filter_count": population_count,
            "selected_filter_count": 1,
            "inclusion_probability": 1.0 / population_count,
            "expansion_weight": float(population_count),
        }
        probabilities[name] = 1.0 / population_count
        expansions[name] = float(population_count)
        population_names.extend(
            f"{module_name}.filter_{filter_index}" for filter_index in range(population_count)
        )
    return {
        "format": "diffdist_edm_filter_sampling_protocol_v1",
        "protocol_id": "per_module_hash_stratified_filter_sampling_v1",
        "mode": "stratified_module",
        "seed": 0,
        "filters_per_module": 1,
        "population_group_count": len(population_names),
        "selected_group_count": len(group_names),
        "population_module_count": len(module_counts),
        "selected_module_count": len(module_counts),
        "module_counts": module_counts,
        "population_sha256": _canonical_name_digest(population_names),
        "selection_sha256": _canonical_name_digest(group_names),
        "selected_group_names": list(group_names),
        "selected_group_inclusion_probabilities": probabilities,
        "selected_group_expansion_weights": expansions,
    }


def test_released_cifar10_allocation_budgets_and_seed3_shuffle():
    """The released CIFAR-10 allocations (Section 3.4 budgets) and the seed-3 shuffle."""
    blockwise = load_json(CIFAR_ALLOCATIONS / "combined_blockwise.json")
    budgets = np.asarray(blockwise["target_budget_plan"]["block_budgets"], dtype=np.float64)
    assert budgets.tolist() == pytest.approx([5406655.841691938, 45619011.15830806])
    assert blockwise["timestep_blocks"] == [[0, 8], [8, 20]]
    assert blockwise["score_sources"]["block_capacity_scores"]["metric"] == "delta_p_eff_geomean"
    assert blockwise["score_sources"]["block_capacity_scores"]["reduction"] == "sum"

    shuffled = np.random.default_rng(3).permutation(budgets)
    assert shuffled.tolist() == pytest.approx([45619011.15830806, 5406655.841691938])

    layerwise = load_json(CIFAR_ALLOCATIONS / "combined_layerwise.json.gz")
    assert layerwise["target_budget_plan"]["block_budgets"] == pytest.approx(budgets.tolist())
    assert np.asarray(layerwise["target_budget_plan"]["layer_budgets"]).sum(axis=1) == pytest.approx(
        budgets.tolist()
    )
    uniform = load_json(CIFAR_ALLOCATIONS / "uniform_blockwise.json")
    assert uniform["target_budget_plan"]["block_budgets"] == pytest.approx([25512833.5, 25512833.5])
    glob = load_json(CIFAR_ALLOCATIONS / "global.json")
    assert glob["target_budget_plan"]["block_budgets"] == pytest.approx([51025667.0])


@pytest.mark.external_edm
def test_uniform_compiler_matches_real_small_block_budget():
    target = 2738708.486731164
    report = compile_uniform_student(
        student_index=0,
        timestep_block=[8, 20],
        target_grouped_budget=target,
        label_dim=10,
        img_resolution=32,
        candidate_table=[(28, 2448267, 2683047)],
    )

    assert report.relative_mismatch <= 0.05
    assert report.rounding["mode"] == "uniform_architecture_candidate"
    assert report.realized_grouped_budget == pytest.approx(target, rel=0.05)


def test_uniform_schedule_search_preserves_ffhq_topology_and_teacher_widths():
    topology = {
        "model_channels": 128,
        "channel_mult": [1, 2, 2, 2],
        "num_blocks": 4,
        "attn_resolutions": [16],
    }

    specs = _uniform_architecture_specs(topology)

    assert (32, (1, 2, 3, 4)) in specs
    assert (56, (1, 1, 2, 4)) in specs
    assert (96, (1, 1, 1, 2)) in specs
    assert specs == _uniform_architecture_specs(dict(reversed(list(topology.items()))))
    teacher_widths = [128, 256, 256, 256]
    for model_channels, channel_mult in specs:
        assert len(channel_mult) == 4
        assert channel_mult[0] == 1
        assert list(channel_mult) == sorted(channel_mult)
        assert all(
            model_channels * multiplier <= maximum
            for multiplier, maximum in zip(channel_mult, teacher_widths)
        )


@pytest.mark.external_edm
def test_generalized_uniform_candidate_keeps_non_width_topology():
    topology = {
        "model_channels": 16,
        "channel_mult": [1, 2],
        "num_blocks": 1,
        "attn_resolutions": [],
        "dropout": 0.0,
        "model_family": "edm",
        "preconditioning": {},
        "augment_dim": 0,
    }

    report = _compile_extra_arch_candidate_student(
        student_index=2,
        timestep_block=[3, 7],
        target_grouped_budget=79027,
        label_dim=0,
        img_resolution=8,
        img_channels=3,
        budget_tolerance=0.0,
        topology=topology,
    )

    assert report is not None
    assert report.realized_grouped_budget == 79027
    assert report.model_kwargs["model_channels"] == 8
    assert report.model_kwargs["channel_mult"] == [1, 3]
    assert report.model_kwargs["num_blocks"] == 1
    assert report.model_kwargs["attn_resolutions"] == []
    assert report.rounding["teacher_channel_mult"] == [1, 2]


@pytest.mark.external_edm
def test_structural_uniform_search_refines_a_scalar_rounding_plateau():
    target = 826319.0
    report = compile_uniform_student(
        student_index=0,
        timestep_block=[0, 1],
        target_grouped_budget=target,
        label_dim=0,
        img_resolution=8,
        candidate_table=[(8, 7995, 9999)],
        topology={
            "channel_mult": [1],
            "num_blocks": 1,
            "attn_resolutions": [],
            "dropout": 0.0,
        },
    )

    assert report.relative_mismatch <= 0.05
    assert report.realized_grouped_budget == pytest.approx(target, rel=0.05)
    assert report.rounding["mode"] == "uniform_structural_widths"
    assert report.rounding["profile_refinement"] == "deterministic_bracket_path"
    assert report.rounding["profile_refinement_steps"] > 0
    assert len(set(report.width_profile.values())) > 1
    assert all(8 <= width <= 128 for width in report.width_profile.values())
    assert all(is_edm_group_norm_safe(width) for width in report.width_profile.values())


def test_infer_imagenet_plan_metadata_from_results():
    results = load_json(IMAGENET_PROFILE)

    assert infer_label_dim(results) == 1000
    assert infer_img_resolution(results) == 64
    assert infer_model_family(results) == "edm"


@pytest.mark.external_edm
@pytest.mark.slow
def test_prepare_architectures_reads_combined_allocations_from_custom_dir(tmp_path):
    eval_dir = tmp_path / "out_eval"
    allocation_dir = tmp_path / "allocation_results_delta_peff"
    output_dir = tmp_path / "plans"
    eval_dir.mkdir()
    allocation_dir.mkdir()

    results = load_json(CIFAR_PROFILE)
    # Treat the expanded release copy as a full profile in this fixture.
    results.pop("pace_slim_profile", None)
    intended_protocols = [{"identity": "fixture_protocol_v1", "backend": "fixture"}]
    ablation_protocol = {
        "protocol_id": "batch_local_exact_sigma_pfi_v1",
        "ablation_mode": "pfi",
    }
    results["benchmark_protocol_ids"] = ["fixture_protocol_v1"]
    results["benchmark_protocols"] = intended_protocols
    results["ablation_protocol"] = ablation_protocol
    group_names = list(results["group_names"])
    filter_sampling = _stratified_sampling(group_names, [1] * len(group_names))
    results["filter_sampling"] = filter_sampling
    results["group_sampling_weights"] = {name: 1.0 for name in group_names}
    results["profile_fingerprint"] = {
        "pfi_seed": 0,
        "fixture": True,
        "grouping": {"filter_sampling": filter_sampling},
    }
    results["profile_fingerprint_sha256"] = hashlib.sha256(
        json.dumps(results["profile_fingerprint"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (eval_dir / "results.json").write_text(json.dumps(results))
    for variant in ("combined_blockwise", "combined_layerwise"):
        payload = load_json(find_json_file(CIFAR_ALLOCATIONS, variant))
        payload["allocation_results_dir"] = str(allocation_dir)
        payload["allocation_results_path"] = str(allocation_dir / f"{variant}.json")
        payload["filter_sampling"] = filter_sampling
        (allocation_dir / f"{variant}.json").write_text(json.dumps(payload))

    summary = prepare_distillation_architectures(
        eval_output_dir=eval_dir,
        allocation_results_dir=allocation_dir,
        output_dir=output_dir,
        variants=["combined_blockwise", "combined_layerwise"],
        layerwise_search_steps=4,
    )

    assert summary["allocation_results_dir"] == str(allocation_dir)
    assert summary["variants"] == ["combined_blockwise", "combined_layerwise"]
    assert summary["plans"]["combined_blockwise"]["benchmark_protocol_ids"] == ["fixture_protocol_v1"]
    assert summary["plans"]["combined_blockwise"]["benchmark_protocols"] == intended_protocols
    assert summary["ablation_protocol"] == ablation_protocol
    assert summary["plans"]["combined_blockwise"]["ablation_protocol"] == ablation_protocol
    assert summary["plans"]["combined_blockwise"]["source_profile"] == summary["source_profile"]
    assert summary["filter_sampling"] == filter_sampling
    assert summary["plans"]["combined_blockwise"]["filter_sampling"] == filter_sampling
    assert summary["source_profile"]["profile_fingerprint_sha256"] == results["profile_fingerprint_sha256"]
    assert (output_dir / "combined_blockwise" / "architecture_plan.json").is_file()
    assert (output_dir / "combined_layerwise" / "architecture_plan.json").is_file()


def test_shuffled_only_plan_uses_its_artifact_budgets_and_embeds_provenance(monkeypatch, tmp_path):
    eval_dir = tmp_path / "profile"
    allocation_dir = tmp_path / "allocations"
    grouping_dir = eval_dir / "grouping"
    grouping_dir.mkdir(parents=True)
    allocation_dir.mkdir()
    results = {
        "config": {"dataset": "ffhq", "image_size": 8, "model_family": "vp"},
        "model_info": {"label_dim": 0, "img_resolution": 8, "img_channels": 3},
        "sigma_bin_labels": [f"bin-{index}" for index in range(20)],
        "n_eff": [1.0] * 20,
    }
    (eval_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    grouping_path = grouping_dir / "timestep_grouping.json"
    grouping_path.write_text(
        json.dumps(
            {
                "num_blocks": 2,
                "boundaries": [0, 7, 20],
                "builtin_cost": "matrix_correlation",
                "pairwise_normalization": "size",
            },
            indent=2,
        )
        + "\n"
    )
    allocation = {
        "student_variant": "shuffled_capacity",
        "shuffle_seed": 3,
        "allocation_rule": {
            "student_variant": "shuffled_capacity",
            "allocation_metric": "n_eff",
            "score_reduction": "q90",
            "layer_score_source": "relative_delta_stack",
            "allocation_alpha": 1.0,
            "shuffle_seed": 3,
        },
        "timestep_grouping_path": str(grouping_path),
        "timestep_blocks": [[0, 7], [7, 20]],
        "score_sources": {
            "block_capacity_scores": {"metric": "n_eff", "reduction": "q90"}
        },
        # The planner must preserve this order, not reshuffle another artifact.
        "target_budget_plan": {"block_budgets": [11.0, 29.0]},
    }
    allocation_path = allocation_dir / "shuffled_capacity.json"
    allocation_path.write_text(json.dumps(allocation, indent=2) + "\n")

    observed_budgets = []

    def fake_compile_uniform_student(**kwargs):
        observed_budgets.append(kwargs["target_grouped_budget"])
        return StudentArchitectureReport(
            student_index=kwargs["student_index"],
            timestep_block=list(kwargs["timestep_block"]),
            target_grouped_budget=kwargs["target_grouped_budget"],
            realized_grouped_budget=int(kwargs["target_grouped_budget"]),
            relative_mismatch=0.0,
            full_parameter_count=int(kwargs["target_grouped_budget"]),
            model_kwargs={},
        )

    monkeypatch.setattr(distillation, "build_uniform_candidate_table", lambda **_kwargs: [])
    monkeypatch.setattr(distillation, "compile_uniform_student", fake_compile_uniform_student)
    summary = prepare_distillation_architectures(
        eval_output_dir=eval_dir,
        allocation_results_dir=allocation_dir,
        output_dir=tmp_path / "plans",
        variants=["shuffled_capacity"],
        shuffle_seed=3,
    )

    plan = summary["plans"]["shuffled_capacity"]
    assert observed_budgets == [11.0, 29.0]
    assert plan["source_allocation_path"] == str(allocation_path)
    assert plan["source_allocation_sha256"] == hashlib.sha256(
        allocation_path.read_bytes()
    ).hexdigest()
    assert plan["allocation_rule"] == allocation["allocation_rule"]
    assert plan["grouping_identity"]["sha256"] == hashlib.sha256(
        grouping_path.read_bytes()
    ).hexdigest()
    assert plan["grouping_identity"]["boundaries"] == [0, 7, 20]
    assert not (allocation_dir / "blockwise_capacity.json").exists()


def test_shuffled_plan_rejects_allocation_seed_mismatch_before_compilation(monkeypatch, tmp_path):
    eval_dir = tmp_path / "profile"
    allocation_dir = tmp_path / "allocations"
    eval_dir.mkdir()
    allocation_dir.mkdir()
    (eval_dir / "results.json").write_text(
        json.dumps(
            {
                "config": {"dataset": "ffhq", "image_size": 8},
                "model_info": {"label_dim": 0, "img_resolution": 8},
                "sigma_bin_labels": ["low", "high"],
            }
        )
    )
    (allocation_dir / "shuffled_capacity.json").write_text(
        json.dumps(
            {
                "student_variant": "shuffled_capacity",
                "shuffle_seed": 3,
                "timestep_blocks": [[0, 1], [1, 2]],
                "target_budget_plan": {"block_budgets": [1.0, 2.0]},
            }
        )
    )
    monkeypatch.setattr(distillation, "build_uniform_candidate_table", lambda **_kwargs: [])
    monkeypatch.setattr(
        distillation,
        "compile_uniform_student",
        lambda **_kwargs: pytest.fail("seed mismatch must fail before architecture compilation"),
    )

    with pytest.raises(ValueError, match="artifact shuffle_seed=3, planner shuffle_seed=4"):
        prepare_distillation_architectures(
            eval_output_dir=eval_dir,
            allocation_results_dir=allocation_dir,
            output_dir=tmp_path / "plans",
            variants=["shuffled_capacity"],
            shuffle_seed=4,
        )


def test_ddp_global_batch_must_split_evenly():
    assert require_divisible_global_batch("batch_size", 128, 8) == 16
    with pytest.raises(ValueError, match="divisible by distributed world size"):
        require_divisible_global_batch("batch_size", 130, 8)


@pytest.mark.external_edm
def test_layerwise_compiler_preserves_real_block_budget_sum():
    results = load_json(CIFAR_PROFILE)
    layerwise = load_json(CIFAR_ALLOCATIONS / "combined_layerwise.json.gz")
    block_budget = layerwise["target_budget_plan"]["block_budgets"][1]
    layer_budgets = layerwise["target_budget_plan"]["layer_budgets"][1]

    report = compile_layerwise_student(
        student_index=1,
        timestep_block=[8, 20],
        target_grouped_budget=block_budget,
        layerwise_budgets=layer_budgets,
        results=results,
        label_dim=10,
        img_resolution=32,
        search_steps=8,
    )

    assert report.relative_mismatch <= 0.05
    assert report.layerwise_target_sum == pytest.approx(block_budget)
    assert report.width_profile
    assert all(width >= 8 and width % 8 == 0 for width in report.width_profile.values())
    assert all(is_edm_group_norm_safe(width) for width in report.width_profile.values())

    model = construct_student_from_kwargs(report.model_kwargs).eval()
    x = torch.randn(1, 3, 32, 32)
    sigmas = torch.tensor([0.01])
    labels = torch.zeros(1, 10)
    labels[:, 0] = 1
    with torch.no_grad():
        y = model(x, sigmas, labels)
    assert y.shape == x.shape


@pytest.mark.external_edm
def test_narrow_student_and_blockwise_dispatch_forward():
    first = NarrowEDMPrecond(
        img_resolution=8,
        img_channels=3,
        label_dim=10,
        model_channels=8,
        channel_mult=[1],
        dropout=0.0,
    )
    second = NarrowEDMPrecond(
        img_resolution=8,
        img_channels=3,
        label_dim=10,
        model_channels=8,
        channel_mult=[1],
        dropout=0.0,
    )
    student = BlockwiseEDMStudent(
        [first, second],
        [[0, 1], [1, 2]],
        sigma_values=[10.0, 1.0, 0.1, 0.01],
        num_sigma_bins=2,
    )

    x = torch.randn(4, 3, 8, 8)
    sigmas = torch.tensor([10.0, 1.0, 0.1, 0.01])
    labels = torch.zeros(4, 10)
    labels[:, 0] = 1
    y = student(x, sigmas, labels)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert student.block_ids_for_sigma(sigmas).tolist() == [0, 0, 1, 1]


@pytest.mark.external_edm
def test_grouped_parameter_counter_uses_filter_proxy():
    kwargs = default_model_kwargs(model_channels=8, label_dim=10, img_resolution=8)
    kwargs["channel_mult"] = [1]
    model = construct_student_from_kwargs(kwargs)

    assert count_grouped_parameters(model) > 0
    assert count_grouped_parameters(model) <= sum(param.numel() for param in model.parameters())


@pytest.mark.external_edm
def test_topology_candidate_counting_uses_storage_free_meta_models(monkeypatch):
    import pace.edm_distillation as module

    original = module.construct_student_from_kwargs
    observed_devices = []

    def recording_constructor(kwargs):
        model = original(kwargs)
        observed_devices.append(next(model.parameters()).device.type)
        return model

    monkeypatch.setattr(module, "construct_student_from_kwargs", recording_constructor)
    table = build_uniform_candidate_table(
        label_dim=0,
        img_resolution=256,
        candidates=[8],
        topology={
            "channel_mult": [1, 1, 2, 2, 4, 4],
            "num_blocks": 2,
            "attn_resolutions": [32, 16, 8],
            "dropout": 0.1,
            "model_family": "edm",
        },
    )

    assert table[0][0] == 8
    assert observed_devices == ["meta"]


@pytest.mark.external_edm
def test_snapshot_uses_state_dicts_not_pickled_modules(tmp_path):
    kwargs = default_model_kwargs(model_channels=8, label_dim=10, img_resolution=8)
    kwargs["channel_mult"] = [1]
    plan = {
        "variant": "global",
        "students": [{"model_kwargs": kwargs}],
        "timestep_blocks": [[0, 1]],
        "sigma_values": [1.0],
        "num_sigma_bins": 1,
        "benchmark_protocol_ids": ["fixture_protocol_v1"],
        "benchmark_protocols": [{"identity": "fixture_protocol_v1"}],
        "source_profile": {
            "format": "diffdist_edm_source_profile_v1",
            "results_json_path": "/profiles/pfi/results.json",
            "results_sha256": "a" * 64,
            "profile_fingerprint": {"pfi_seed": 0},
            "profile_fingerprint_sha256": "b" * 64,
        },
        "ablation_protocol": {
            "protocol_id": "batch_local_exact_sigma_pfi_v1",
            "ablation_mode": "pfi",
        },
        "filter_sampling": _stratified_sampling(["g0"], [4]),
    }
    student = construct_student_from_kwargs(kwargs)
    ema = construct_student_from_kwargs(kwargs)
    path = tmp_path / "snapshot.pt"

    save_snapshot(path, step=3, student=student, ema=ema, plan=plan, stats=[{"loss": 1.0}])
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert payload["snapshot_format"] == "diffdist_edm_state_dict_v2"
    assert "student" not in payload
    assert "ema" not in payload
    assert "student_state_dict" in payload
    assert "ema_state_dict" in payload
    assert payload["step"] == 3
    assert payload["benchmark_protocol_ids"] == ["fixture_protocol_v1"]
    assert payload["benchmark_protocols"] == [{"identity": "fixture_protocol_v1"}]
    assert payload["source_profile"] == plan["source_profile"]
    assert payload["ablation_protocol"] == plan["ablation_protocol"]
    assert payload["filter_sampling"] == plan["filter_sampling"]


def test_sampled_structural_parameter_budgets_use_expansion_weights():
    group_names = ["model.enc.block.filter_0", "model.dec.block.filter_0"]
    results = {
        "group_names": group_names,
        "group_param_counts": {
            "model.enc.block.filter_0": 4,
            "model.dec.block.filter_0": 7,
        },
        "group_structural_keys": {
            "model.enc.block.filter_0": "enc.block",
            "model.dec.block.filter_0": "dec.block",
        },
        "filter_sampling": _stratified_sampling(group_names, [3, 2]),
    }

    assert group_original_structural_budgets(results) == {
        "enc.block": 12.0,
        "dec.block": 14.0,
    }


@pytest.mark.external_edm
def test_resume_restores_model_ema_and_optimizer_state(tmp_path):
    kwargs = default_model_kwargs(model_channels=8, label_dim=10, img_resolution=8)
    kwargs["channel_mult"] = [1]
    plan = {
        "variant": "global",
        "students": [{"model_kwargs": kwargs}],
        "timestep_blocks": [[0, 1]],
        "sigma_values": [1.0],
        "num_sigma_bins": 1,
    }
    student = construct_student_from_kwargs(kwargs)
    ema = construct_student_from_kwargs(kwargs)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    loss = sum(parameter.square().mean() for parameter in student.parameters())
    loss.backward()
    optimizer.step()
    ema.load_state_dict(student.state_dict())

    snapshot_path = tmp_path / "student-final.pt"
    save_snapshot(snapshot_path, step=7, student=student, ema=ema, plan=plan, stats=[{"step": 7}])
    training_state_path = tmp_path / "training-state-latest.pt"
    save_training_state(training_state_path, step=7, optimizer=optimizer, student=student)

    resolved_path, payload = resolve_resume_snapshot(tmp_path, "auto", expected_step=7)
    resumed_student = construct_student_from_kwargs(kwargs)
    resumed_ema = construct_student_from_kwargs(kwargs)
    resumed_optimizer = torch.optim.Adam(resumed_student.parameters(), lr=1e-3)

    assert resolved_path == snapshot_path
    assert restore_model_snapshot(payload, student=resumed_student, ema=resumed_ema) == 7
    assert restore_training_state(
        training_state_path,
        expected_step=7,
        optimizer=resumed_optimizer,
        device=torch.device("cpu"),
        restore_rng=False,
    )
    for expected, actual in zip(student.parameters(), resumed_student.parameters()):
        assert torch.equal(expected, actual)
    for expected, actual in zip(ema.parameters(), resumed_ema.parameters()):
        assert torch.equal(expected, actual)
    assert resumed_optimizer.state


def test_resume_logs_require_strict_steps_and_model_alignment(tmp_path):
    path = tmp_path / "distillation_stats.jsonl"
    append_jsonl(path, {"step": 1, "loss": 2.0})
    append_jsonl(path, {"step": 2, "loss": 1.0})
    rows = load_jsonl_rows(path, required=True)

    validate_resume_log(rows, path, resume_step=2, require_last=True)
    with pytest.raises(ValueError, match="does not match model step"):
        validate_resume_log(rows, path, resume_step=3, require_last=True)

    append_jsonl(path, {"step": 2, "loss": 0.5})
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_resume_log(load_jsonl_rows(path), path, resume_step=2, require_last=True)


@pytest.mark.external_edm
def test_auto_resume_selects_newest_valid_checkpoint_independent_of_logs(tmp_path):
    kwargs = default_model_kwargs(model_channels=8, label_dim=10, img_resolution=8)
    kwargs["channel_mult"] = [1]
    plan = {"students": [{"model_kwargs": kwargs}]}
    student = construct_student_from_kwargs(kwargs)
    ema = construct_student_from_kwargs(kwargs)
    save_snapshot(tmp_path / "student-final.pt", step=10, student=student, ema=ema, plan=plan, stats=[])
    save_snapshot(tmp_path / "student-best-val.pt", step=12, student=student, ema=ema, plan=plan, stats=[])
    (tmp_path / "student-best-fid.pt").write_text("corrupt")
    (tmp_path / "latest-checkpoint.json").write_text('{"path": "missing.pt", "step": 99}\n')
    append_jsonl(tmp_path / "distillation_stats.jsonl", {"step": 15, "loss": 1.0})

    path, payload = resolve_resume_snapshot(tmp_path, "auto", expected_step=15)

    assert path == tmp_path / "student-best-val.pt"
    assert payload["step"] == 12
    assert payload["_resume_selection_reason"] == "directory_scan"


@pytest.mark.external_edm
def test_explicit_resume_overrides_newer_automatic_candidate(tmp_path):
    kwargs = default_model_kwargs(model_channels=8, label_dim=10, img_resolution=8)
    kwargs["channel_mult"] = [1]
    plan = {"students": [{"model_kwargs": kwargs}]}
    student = construct_student_from_kwargs(kwargs)
    ema = construct_student_from_kwargs(kwargs)
    explicit = tmp_path / "student-snapshot-step000005.pt"
    save_snapshot(explicit, step=5, student=student, ema=ema, plan=plan, stats=[])
    save_snapshot(tmp_path / "student-final.pt", step=10, student=student, ema=ema, plan=plan, stats=[])

    path, payload = resolve_resume_snapshot(tmp_path, str(explicit))

    assert path == explicit
    assert payload["step"] == 5
    assert payload["_resume_selection_reason"] == "explicit_path"


def test_periodic_snapshot_retention_preserves_latest_during_run_then_removes_it(tmp_path):
    for step in (5, 10, 15):
        (tmp_path / f"student-snapshot-step{step:06d}.pt").write_bytes(b"snapshot")
    (tmp_path / "student-best-val.pt").write_bytes(b"best")
    (tmp_path / "student-final.pt").write_bytes(b"final")

    removed = prune_periodic_snapshots(tmp_path, keep=0, training_active=True)

    assert [path.name for path in removed] == [
        "student-snapshot-step000005.pt",
        "student-snapshot-step000010.pt",
    ]
    assert (tmp_path / "student-snapshot-step000015.pt").is_file()
    assert (tmp_path / "student-best-val.pt").is_file()
    assert (tmp_path / "student-final.pt").is_file()

    prune_periodic_snapshots(tmp_path, keep=0, training_active=False)
    assert not list(tmp_path.glob("student-snapshot-step*.pt"))
    assert (tmp_path / "student-best-val.pt").is_file()
    assert (tmp_path / "student-final.pt").is_file()


def test_best_val_checkpoint_selection_ignores_available_fid(tmp_path):
    best_val = BestMetricTracker(metric_name="val_loss", snapshot_name="student-best-val.pt")
    best_fid = BestMetricTracker(metric_name="fid", snapshot_name="student-best-fid.pt")
    best_val.update({"step": 10, "val_loss": 1.25}, output_dir=tmp_path)
    best_fid.update({"step": 20, "fid": 3.0}, output_dir=tmp_path)

    selected = build_checkpoint_selection(
        stop_reason="max_steps",
        final_step=30,
        final_snapshot=tmp_path / "student-final.pt",
        best_val=best_val,
        best_fid=best_fid,
        selection_policy="best_val",
    )

    assert selected["selected"]["kind"] == "best_val"
    assert selected["selected"]["step"] == 10
    assert selected["validation_split"]["is_heldout"] is False


def test_repair_resume_log_truncates_newer_rows_and_partial_tail(tmp_path):
    path = tmp_path / "distillation_stats.jsonl"
    path.write_text(
        '{"step": 1, "loss": 3.0}\n'
        '{"step": 2, "loss": 2.0}\n'
        '{"step": 3, "loss": 1.0}\n'
        '{"step":'
    )

    rows, report = repair_resume_log(path, resume_step=2)

    assert [row["step"] for row in rows] == [1, 2]
    assert load_jsonl_rows(path) == rows
    assert report == {
        "original_rows": 3,
        "kept_rows": 2,
        "discarded_rows": 1,
        "discarded_malformed": 1,
    }


def test_repair_resume_log_rejects_middle_corruption_and_retained_duplicates(tmp_path):
    middle = tmp_path / "middle.jsonl"
    middle.write_text('{"step": 1}\nnot-json\n{"step": 2}\n')
    with pytest.raises(ValueError, match="middle"):
        repair_resume_log(middle, resume_step=2)

    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text('{"step": 1}\n{"step": 1}\n')
    with pytest.raises(ValueError, match="strictly increasing"):
        repair_resume_log(duplicate, resume_step=2)


def test_snapshot_bundles_optimizer_rng_and_writes_manifest(tmp_path):
    student = ScaleDenoiser(0.1)
    ema = ScaleDenoiser(0.1)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    student.scale.square().backward()
    optimizer.step()
    path = tmp_path / "student-final.pt"

    save_snapshot(
        path,
        step=4,
        student=student,
        ema=ema,
        plan={},
        stats=[],
        optimizer=optimizer,
        update_manifest=True,
    )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    manifest = json.loads((tmp_path / "latest-checkpoint.json").read_text())
    assert payload["optimizer_state_dict"]["state"]
    assert {"python", "numpy", "torch"} <= payload["rng_state"].keys()
    assert manifest["path"] == path.name
    assert manifest["step"] == 4


class ScaleDenoiser(torch.nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(float(scale)))

    def forward(self, x, sigma, labels=None):
        return x * self.scale


def test_main_resume_appends_logs_and_continues_optimizer_steps(monkeypatch, tmp_path):
    plan = {
        "variant": "global",
        "network_pkl": "unused.pkl",
        "students": [{"model_kwargs": {"img_resolution": 8, "label_dim": 0}}],
        "timestep_blocks": [[0, 1]],
        "sigma_values": [1.0],
        "num_sigma_bins": 1,
    }
    plan_path = tmp_path / "architecture_plan.json"
    plan_path.write_text(json.dumps(plan))
    output_dir = tmp_path / "run"
    images = torch.ones(4, 3, 8, 8)
    labels = torch.zeros(4, dtype=torch.long)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(images, labels),
        batch_size=2,
        shuffle=False,
    )

    def fake_loss(*, student, images, sigmas, labels, **kwargs):
        loss = (student(images, sigmas, labels) - images).square().mean()
        value = float(loss.detach())
        return loss, {"loss": value, "kd_loss": value, "data_loss": value}

    monkeypatch.setattr(train_module, "construct_student_from_plan", lambda unused_plan: ScaleDenoiser(0.1))
    monkeypatch.setattr(train_module, "load_edm_network", lambda *args, **kwargs: ScaleDenoiser(0.5))
    monkeypatch.setattr(train_module, "create_ema", lambda student: ScaleDenoiser(float(student.scale.detach())))
    monkeypatch.setattr(train_module, "make_image_loader", lambda **kwargs: loader)
    monkeypatch.setattr(train_module, "hybrid_distillation_loss", fake_loss)

    common_args = [
        "train_edm_distillation.py",
        "--architecture-plan",
        str(plan_path),
        "--output-dir",
        str(output_dir),
        "--batch-size",
        "2",
        "--microbatch",
        "1",
        "--num-workers",
        "0",
        "--snapshot-every",
        "0",
        "--val-every",
        "0",
        "--fid-every",
        "0",
        "--no-tensorboard",
        "--device",
        "cpu",
    ]
    monkeypatch.setattr(sys, "argv", [*common_args, "--steps", "2"])
    train_module.main()
    (output_dir / "training-state-latest.pt").unlink()
    monkeypatch.setattr(sys, "argv", [*common_args, "--resume", "--steps", "4"])
    train_module.main()

    rows = load_jsonl_rows(output_dir / "distillation_stats.jsonl")
    assert [row["step"] for row in rows] == [1, 2, 3, 4]
    final_payload = torch.load(output_dir / "student-final.pt", map_location="cpu", weights_only=False)
    assert final_payload["step"] == 4
    training_payload = torch.load(output_dir / "training-state-latest.pt", map_location="cpu", weights_only=False)
    assert training_payload["step"] == 4
    optimizer_steps = [state["step"].item() for state in training_payload["optimizer_state_dict"]["state"].values()]
    assert optimizer_steps == [4]


def make_validation_loader(num_examples=4):
    images = torch.linspace(-1.0, 1.0, num_examples * 3 * 8 * 8, dtype=torch.float32).reshape(num_examples, 3, 8, 8)
    labels = torch.arange(num_examples, dtype=torch.long) % 10
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images, labels), batch_size=2, shuffle=False)


def test_validation_loss_is_deterministic_no_grad_and_restores_modes():
    student = ScaleDenoiser(0.1)
    teacher = ScaleDenoiser(0.5)
    student.train()
    teacher.train()
    loader = make_validation_loader()
    kwargs = dict(
        student=student,
        teacher=teacher,
        loader=loader,
        sigma_values=torch.tensor([1.0, 0.5]),
        label_dim=10,
        device=torch.device("cpu"),
        microbatch=1,
        seed=123,
        step=7,
        plan={"timestep_blocks": [[0, 2]], "num_sigma_bins": 2},
        model_family="vp",
        sigma_data=0.5,
        kd_weight=1.0,
        data_weight=0.25,
    )

    first = evaluate_validation_loss(**kwargs)
    second = evaluate_validation_loss(**kwargs)

    assert first == second
    assert first["val_loss"] >= 0
    assert first["val_kd_loss"] >= 0
    assert first["val_data_loss"] >= 0
    assert first["val_count"] == 4
    assert student.training
    assert teacher.training
    assert all(param.grad is None for param in student.parameters())


def test_early_stopper_respects_delta_patience_min_steps_and_disabled():
    stopper = ValidationEarlyStopper(patience=2, min_steps=5, min_delta=0.1)

    assert stopper.update(step=1, val_loss=1.0).improved
    assert not stopper.update(step=2, val_loss=0.95).should_stop
    assert stopper.update(step=3, val_loss=0.8).improved
    assert not stopper.update(step=4, val_loss=0.75).should_stop
    result = stopper.update(step=5, val_loss=0.74)

    assert not result.improved
    assert result.bad_checks == 2
    assert result.should_stop

    disabled = ValidationEarlyStopper(patience=1, min_steps=0, min_delta=0.0, disabled=True)
    assert disabled.update(step=1, val_loss=1.0).improved
    assert not disabled.update(step=2, val_loss=1.1).should_stop


class FirstBlockOnlyDenoiser(ScaleDenoiser):
    def block_ids_for_sigma(self, sigma):
        return torch.zeros(sigma.shape[0], device=sigma.device, dtype=torch.long)


def test_validation_logs_blockwise_metrics_and_allows_missing_blocks():
    student = FirstBlockOnlyDenoiser(0.1)
    teacher = ScaleDenoiser(0.5)

    row = evaluate_validation_loss(
        student=student,
        teacher=teacher,
        loader=make_validation_loader(num_examples=2),
        sigma_values=torch.tensor([1.0]),
        label_dim=10,
        device=torch.device("cpu"),
        microbatch=2,
        seed=123,
        step=1,
        plan={"timestep_blocks": [[0, 1], [1, 2]], "num_sigma_bins": 2},
        model_family="vp",
        sigma_data=0.5,
        kd_weight=1.0,
        data_weight=0.25,
    )

    assert row["val_block_0_count"] == row["val_count"]
    assert row["val_block_0_loss"] is not None
    assert row["val_block_1_count"] == 0
    assert row["val_block_1_loss"] is None


class FakeTensorBoardWriter:
    def __init__(self):
        self.scalars = []
        self.text = []
        self.flushed = False
        self.closed = False

    def add_scalar(self, tag, value, global_step):
        self.scalars.append((tag, value, global_step))

    def add_text(self, tag, text, global_step=0):
        self.text.append((tag, text, global_step))

    def flush(self):
        self.flushed = True

    def close(self):
        self.closed = True


def test_tensorboard_logger_tags_metrics_with_direction_arrows():
    writer = FakeTensorBoardWriter()
    logger = TensorBoardLogger(writer)

    logger.log_row(
        "validation",
        {
            "step": 5,
            "val_loss": 1.0,
            "val_kd_loss": 0.5,
            "val_count": 8,
            "val_block_0_loss": 0.75,
            "val_block_1_loss": None,
            "sample_dir": "/tmp/samples",
        },
    )
    logger.log_row("train", {"step": 5, "grad_norm": 0.9, "examples_per_second": 12.0})
    logger.log_metric("checkpoint/best_val_updated", 1.0, step=5, direction=HIGHER_IS_BETTER)
    logger.flush()
    logger.close()

    tags = {item[0] for item in writer.scalars}
    assert f"validation/loss {LOWER_IS_BETTER}" in tags
    assert f"validation/kd_loss {LOWER_IS_BETTER}" in tags
    assert f"validation/count {HIGHER_IS_BETTER}" in tags
    assert f"validation/block_0/loss {LOWER_IS_BETTER}" in tags
    assert f"train/grad_norm {LOWER_IS_BETTER}" in tags
    assert f"train/examples_per_second {HIGHER_IS_BETTER}" in tags
    assert f"checkpoint/best_val_updated {HIGHER_IS_BETTER}" in tags
    assert not any("sample_dir" in tag for tag in tags)
    assert writer.flushed
    assert writer.closed


def test_tensorboard_logger_reports_missing_pkg_resources(monkeypatch, capsys, tmp_path):
    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch.utils.tensorboard":
            raise ModuleNotFoundError("No module named 'pkg_resources'", name="pkg_resources")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    logger = create_tensorboard_logger(output_dir=tmp_path, tensorboard_dir=None, disabled=False)

    assert not logger.enabled
    assert "setuptools<81" in capsys.readouterr().out


def test_tensorboard_resume_reuses_log_dir_and_purges_future_steps(monkeypatch, tmp_path):
    writers = []

    class FakeSummaryWriter(FakeTensorBoardWriter):
        def __init__(self, *, log_dir, purge_step):
            super().__init__()
            self.log_dir = log_dir
            self.purge_step = purge_step
            writers.append(self)

    fake_module = types.ModuleType("torch.utils.tensorboard")
    fake_module.SummaryWriter = FakeSummaryWriter
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", fake_module)

    logger = create_tensorboard_logger(
        output_dir=tmp_path,
        tensorboard_dir=None,
        disabled=False,
        purge_step=51,
    )

    assert logger.enabled
    assert writers[0].log_dir == str(tmp_path / "tensorboard")
    assert writers[0].purge_step == 51
    logger.close()


def test_train_cli_exposes_fid_seed_and_resume():
    parser = build_arg_parser()

    default_args = parser.parse_args(["--architecture-plan", "plan.json", "--fid-every", "50"])
    custom_args = parser.parse_args(["--architecture-plan", "plan.json", "--fid-seed", "123"])
    cleanup_args = parser.parse_args(["--architecture-plan", "plan.json", "--discard-fid-samples"])
    auto_resume_args = parser.parse_args(["--architecture-plan", "plan.json", "--resume"])
    explicit_resume_args = parser.parse_args(
        ["--architecture-plan", "plan.json", "--resume", "student-snapshot-step000100.pt"]
    )

    assert default_args.fid_seed == 0
    assert default_args.resume is None
    assert not default_args.discard_fid_samples
    assert custom_args.fid_seed == 123
    assert cleanup_args.discard_fid_samples
    assert auto_resume_args.resume == "auto"
    assert explicit_resume_args.resume == "student-snapshot-step000100.pt"


def _tiny_jpeg_bytes(color):
    import io

    from PIL import Image

    image = Image.new("RGB", (8, 8), color)
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG")
    return encoded.getvalue()


def _write_tiny_imagenet_parquet(root: Path, split: str, labels: list[int], *, nested_image: bool = False) -> None:
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    root.mkdir(parents=True, exist_ok=True)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 0)]
    images = [_tiny_jpeg_bytes(colors[index % len(colors)]) for index in range(len(labels))]
    if nested_image:
        table = pa.table({"image": [{"bytes": image, "path": None} for image in images], "label": labels})
    else:
        table = pa.table({"bytes": images, "label": labels})
    pq.write_table(table, root / f"{split}-00000-of-00001.parquet")


def test_convert_imagenet_parquet_to_webdataset_writes_shards_and_metadata(tmp_path):
    input_root = tmp_path / "parquet"
    output_root = tmp_path / "webdataset"
    _write_tiny_imagenet_parquet(input_root, "train", [3, 4, 5], nested_image=True)
    _write_tiny_imagenet_parquet(input_root, "validation", [7, 8], nested_image=True)

    metadata = convert_wds.convert_imagenet_parquet_to_webdataset(
        input_root=input_root,
        output_root=output_root,
        splits=["train", "validation"],
        image_column="bytes",
        label_column="label",
        samples_per_shard=2,
        parquet_batch_size=2,
        jpeg_quality=90,
    )

    assert metadata["splits"]["train"]["num_samples"] == 3
    assert metadata["splits"]["validation"]["num_samples"] == 2
    assert metadata["splits"]["train"]["resolved_image_column"] == "image"
    assert (output_root / "metadata.json").is_file()
    assert sorted(path.name for path in output_root.glob("train-*.tar")) == ["train-000000.tar", "train-000001.tar"]
    with tarfile.open(output_root / "train-000000.tar") as tar:
        names = set(tar.getnames())
        assert "train-000000000.jpg" in names
        assert "train-000000000.cls" in names
        image_file = tar.extractfile("train-000000000.jpg")
        assert image_file is not None
        assert image_file.read() == _tiny_jpeg_bytes((255, 0, 0))
        cls_file = tar.extractfile("train-000000000.cls")
        assert cls_file is not None
        assert cls_file.read().decode("utf-8").strip() == "3"


def test_imagenet_parquet_loader_accepts_bytes_leaf_column(tmp_path):
    input_root = tmp_path / "parquet"
    _write_tiny_imagenet_parquet(input_root, "train", [9], nested_image=True)

    dataset = train_module.ImageNet1KParquetDataset(
        root=str(input_root),
        image_size=4,
        split="train",
        max_images=None,
        image_column="bytes",
        label_column="label",
    )
    image, label = dataset[0]

    assert dataset.image_column == "image"
    assert image.shape == (3, 4, 4)
    assert image.min().item() >= -1.0
    assert image.max().item() <= 1.0
    assert label == 9


def _make_webdataset_loader(tmp_path, *, train: bool, shuffle: bool):
    return train_module.make_image_loader(
        dataset="imagenet1k_webdataset",
        data_root=str(tmp_path / "webdataset"),
        image_size=4,
        batch_size=2,
        num_workers=0,
        download=False,
        max_images=None,
        train=train,
        shuffle=shuffle,
        drop_last=False,
        imagenet_split="train",
        val_imagenet_split="val",
        parquet_split="train",
        val_parquet_split="validation",
        parquet_image_column=None,
        parquet_label_column=None,
        webdataset_shuffle_buffer=2,
    )


def test_imagenet_webdataset_loader_shapes_normalization_and_validation_order(tmp_path):
    input_root = tmp_path / "parquet"
    output_root = tmp_path / "webdataset"
    _write_tiny_imagenet_parquet(input_root, "train", [0, 1, 2])
    _write_tiny_imagenet_parquet(input_root, "validation", [10, 11])
    convert_wds.convert_imagenet_parquet_to_webdataset(
        input_root=input_root,
        output_root=output_root,
        splits=["train", "validation"],
        image_column="bytes",
        label_column="label",
        samples_per_shard=2,
    )

    train_loader = _make_webdataset_loader(tmp_path, train=True, shuffle=True)
    train_images, train_labels = next(iter(train_loader))

    assert isinstance(train_loader.dataset, train_module.ImageNet1KWebDataset)
    assert train_loader.dataset.shuffle_samples
    assert train_images.shape == (2, 3, 4, 4)
    assert train_images.min().item() >= -1.0
    assert train_images.max().item() <= 1.0
    assert train_labels.dtype == torch.long

    first_val_loader = _make_webdataset_loader(tmp_path, train=False, shuffle=False)
    second_val_loader = _make_webdataset_loader(tmp_path, train=False, shuffle=False)
    first_images, first_labels = next(iter(first_val_loader))
    second_images, second_labels = next(iter(second_val_loader))

    assert not first_val_loader.dataset.shuffle_samples
    assert first_images.shape == (2, 3, 4, 4)
    assert first_labels.tolist() == [10, 11]
    assert second_labels.tolist() == first_labels.tolist()
    assert torch.equal(first_images, second_images)


class TinySamplerNet(torch.nn.Module):
    img_channels = 3
    img_resolution = 4
    label_dim = 0
    sigma_min = 0.002
    sigma_max = 80.0

    def forward(self, x, sigma, class_labels=None):
        return torch.zeros_like(x)

    def round_sigma(self, sigma):
        return sigma


def test_fid_sample_generation_shards_seeds_by_rank(tmp_path):
    net = TinySamplerNet()

    for rank in (0, 1):
        train_module.generate_fid_samples(
            net=net,
            output_dir=tmp_path,
            step=8,
            seed=10,
            num_samples=5,
            batch_size=2,
            num_steps=2,
            device=torch.device("cpu"),
            rank=rank,
            world_size=2,
        )

    sample_dir = tmp_path / "fid_samples" / "step000008_n5_seed10"
    assert sorted(path.name for path in sample_dir.glob("seed*.png")) == [
        "seed000010.png",
        "seed000011.png",
        "seed000012.png",
        "seed000013.png",
        "seed000014.png",
    ]

    rank0_manifest = json.loads((sample_dir / "sample_manifest_rank0.json").read_text())
    rank1_manifest = json.loads((sample_dir / "sample_manifest_rank1.json").read_text())
    assert [Path(path).name for path in rank0_manifest["images"]] == [
        "seed000010.png",
        "seed000012.png",
        "seed000014.png",
    ]
    assert [Path(path).name for path in rank1_manifest["images"]] == [
        "seed000011.png",
        "seed000013.png",
    ]


def test_distributed_worker_fid_returns_after_sample_shard_without_barrier(monkeypatch, tmp_path):
    def unexpected_compute_fid(*args, **kwargs):
        raise AssertionError("worker ranks should not compute FID")

    calls = _install_fake_cleanfid(monkeypatch, compute_fid=unexpected_compute_fid)

    def unexpected_barrier():
        raise AssertionError("FID sync should not use a distributed barrier")

    monkeypatch.setattr(train_module.dist, "barrier", unexpected_barrier)
    row = run_fid_evaluation(
        net=TinySamplerNet(),
        output_dir=tmp_path,
        step=9,
        seed=20,
        num_samples=4,
        batch_size=2,
        num_steps=2,
        ref_split="validation",
        ref_dataset_name="imagenet",
        ref_dataset_res=64,
        fid_mode="clean",
        device=torch.device("cpu"),
        distributed=True,
        rank=1,
        world_size=2,
    )

    assert row["fid"] is None
    assert row["rank"] == 1
    assert calls["compute_fid"] == 0
    sample_dir = tmp_path / "fid_samples" / "step000009_n4_seed20"
    assert sorted(path.name for path in sample_dir.glob("seed*.png")) == ["seed000021.png", "seed000023.png"]


def test_distributed_main_fid_waits_for_manifests_without_barrier(monkeypatch, tmp_path):
    def fake_compute_fid(fdir, **kwargs):
        return 7.0

    calls = _install_fake_cleanfid(monkeypatch, compute_fid=fake_compute_fid)

    def unexpected_barrier():
        raise AssertionError("FID sync should not use a distributed barrier")

    monkeypatch.setattr(train_module.dist, "barrier", unexpected_barrier)

    train_module.generate_fid_samples(
        net=TinySamplerNet(),
        output_dir=tmp_path,
        step=10,
        seed=30,
        num_samples=2,
        batch_size=1,
        num_steps=2,
        device=torch.device("cpu"),
        rank=1,
        world_size=2,
        run_id="test-run",
    )

    row = run_fid_evaluation(
        net=TinySamplerNet(),
        output_dir=tmp_path,
        step=10,
        seed=30,
        num_samples=2,
        batch_size=1,
        num_steps=2,
        ref_split="validation",
        ref_dataset_name="imagenet",
        ref_dataset_res=64,
        fid_mode="clean",
        device=torch.device("cpu"),
        distributed=True,
        rank=0,
        world_size=2,
        run_id="test-run",
    )

    assert row["fid"] == pytest.approx(7.0)
    assert row["rank"] == 0
    assert calls["compute_fid"] == 1


def test_cleanfid_monitoring_writes_row_and_best_checkpoint(monkeypatch, tmp_path):
    calls = {}

    def fake_compute_fid(fdir, **kwargs):
        calls["fdir"] = fdir
        calls["kwargs"] = kwargs
        return 12.5

    cleanfid_module = types.ModuleType("cleanfid")
    cleanfid_module.fid = types.SimpleNamespace(compute_fid=fake_compute_fid)
    monkeypatch.setitem(sys.modules, "cleanfid", cleanfid_module)

    net = TinySamplerNet()
    row = run_fid_evaluation(
        net=net,
        output_dir=tmp_path,
        step=3,
        seed=10,
        num_samples=2,
        batch_size=1,
        num_steps=2,
        ref_split="train",
        ref_dataset_name="cifar10",
        ref_dataset_res=32,
        fid_mode="clean",
        device=torch.device("cpu"),
    )
    append_jsonl(tmp_path / "fid_stats.jsonl", row)
    tracker = BestMetricTracker(metric_name="fid", snapshot_name="student-best-fid.pt")
    if tracker.update(row, output_dir=tmp_path):
        save_snapshot(tmp_path / tracker.snapshot_name, step=3, student=net, ema=net, plan={}, stats=[])

    assert row["fid"] == pytest.approx(12.5)
    assert Path(row["sample_dir"]).is_dir()
    assert len(list(Path(row["sample_dir"]).glob("seed*.png"))) == 2
    assert (tmp_path / "fid_stats.jsonl").read_text().strip()
    assert (tmp_path / "student-best-fid.pt").is_file()
    assert calls["kwargs"]["dataset_name"] == "cifar10"
    assert calls["kwargs"]["dataset_split"] == "train"
    assert not should_run_interval(3, 0)


def test_cleanfid_discards_samples_after_success(monkeypatch, tmp_path):
    def fake_compute_fid(fdir, **kwargs):
        assert Path(fdir).is_dir()
        assert len(list(Path(fdir).glob("seed*.png"))) == 2
        return 8.75

    cleanfid_module = types.ModuleType("cleanfid")
    cleanfid_module.fid = types.SimpleNamespace(compute_fid=fake_compute_fid)
    monkeypatch.setitem(sys.modules, "cleanfid", cleanfid_module)

    row = run_fid_evaluation(
        net=TinySamplerNet(),
        output_dir=tmp_path,
        step=4,
        seed=20,
        num_samples=2,
        batch_size=1,
        num_steps=2,
        ref_split="train",
        ref_dataset_name="cifar10",
        ref_dataset_res=32,
        fid_mode="clean",
        device=torch.device("cpu"),
        discard_samples=True,
    )

    assert row["fid"] == pytest.approx(8.75)
    assert "sample_dir" in row
    assert not Path(row["sample_dir"]).exists()


def test_cleanfid_keeps_samples_after_failure(monkeypatch, tmp_path):
    def fake_compute_fid(fdir, **kwargs):
        raise RuntimeError("fid failed")

    cleanfid_module = types.ModuleType("cleanfid")
    cleanfid_module.fid = types.SimpleNamespace(compute_fid=fake_compute_fid)
    monkeypatch.setitem(sys.modules, "cleanfid", cleanfid_module)

    with pytest.raises(RuntimeError, match="fid failed"):
        run_fid_evaluation(
            net=TinySamplerNet(),
            output_dir=tmp_path,
            step=5,
            seed=30,
            num_samples=2,
            batch_size=1,
            num_steps=2,
            ref_split="train",
            ref_dataset_name="cifar10",
            ref_dataset_res=32,
            fid_mode="clean",
            device=torch.device("cpu"),
            discard_samples=True,
        )

    sample_dir = train_module.fid_sample_dir(tmp_path, step=5, num_samples=2, seed=30)
    assert sample_dir.is_dir()
    assert len(list(sample_dir.glob("seed*.png"))) == 2


def _tiny_fid_reference_config(tmp_path):
    return FidReferenceConfig(
        dataset="imagenet1k_parquet",
        data_root=str(tmp_path / "imagenet_parquet"),
        image_size=4,
        num_workers=0,
        download=False,
        imagenet_split="train",
        val_imagenet_split="val",
        parquet_split="train",
        val_parquet_split="validation",
        parquet_image_column="image",
        parquet_label_column="label",
    )


def _install_fake_cleanfid(monkeypatch, *, compute_fid):
    calls = {"compute_fid": 0, "get_folder_features": 0, "frechet_distance": 0}

    def fake_build_feature_extractor(mode, device):
        return object()

    def fake_build_resizer(mode):
        return lambda image: image.astype(np.float32, copy=False)

    def fake_get_batch_features(batch, model, device):
        flat = batch.detach().cpu().numpy().reshape(batch.shape[0], -1)
        return np.stack([flat.mean(axis=1), flat.std(axis=1) + 1.0], axis=1)

    def fake_get_folder_features(*args, **kwargs):
        calls["get_folder_features"] += 1
        return np.asarray([[5.0, 1.0], [6.0, 1.5]], dtype=np.float32)

    def fake_frechet_distance(mu, sigma, ref_mu, ref_sigma):
        calls["frechet_distance"] += 1
        calls["ref_mu"] = np.asarray(ref_mu)
        calls["ref_sigma"] = np.asarray(ref_sigma)
        return 42.25

    def counting_compute_fid(*args, **kwargs):
        calls["compute_fid"] += 1
        return compute_fid(*args, **kwargs)

    cleanfid_module = types.ModuleType("cleanfid")
    cleanfid_module.fid = types.SimpleNamespace(
        compute_fid=counting_compute_fid,
        build_feature_extractor=fake_build_feature_extractor,
        build_resizer=fake_build_resizer,
        get_batch_features=fake_get_batch_features,
        get_folder_features=fake_get_folder_features,
        frechet_distance=fake_frechet_distance,
    )
    monkeypatch.setitem(sys.modules, "cleanfid", cleanfid_module)
    return calls


def _patch_tiny_reference_loader(monkeypatch):
    def fake_make_image_loader(**kwargs):
        images = torch.stack(
            [
                torch.full((3, 4, 4), -0.5, dtype=torch.float32),
                torch.full((3, 4, 4), 0.5, dtype=torch.float32),
            ]
        )
        labels = torch.tensor([0, 1])
        dataset = torch.utils.data.TensorDataset(images, labels)
        return torch.utils.data.DataLoader(dataset, batch_size=kwargs["batch_size"], shuffle=False)

    monkeypatch.setattr(train_module, "make_image_loader", fake_make_image_loader)


def test_cleanfid_fallback_computes_and_caches_reference_stats(monkeypatch, tmp_path):
    def missing_remote_stats(*args, **kwargs):
        raise urllib.error.HTTPError(
            url="https://www.cs.cmu.edu/~clean-fid/stats/imagenet_clean_validation_64.npz",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )

    calls = _install_fake_cleanfid(monkeypatch, compute_fid=missing_remote_stats)
    _patch_tiny_reference_loader(monkeypatch)
    ref_config = _tiny_fid_reference_config(tmp_path)
    cache_dir = tmp_path / "fid_reference_stats"

    row = run_fid_evaluation(
        net=TinySamplerNet(),
        output_dir=tmp_path,
        step=5,
        seed=1,
        num_samples=1,
        batch_size=1,
        num_steps=2,
        ref_split="validation",
        ref_dataset_name="imagenet",
        ref_dataset_res=64,
        fid_mode="clean",
        device=torch.device("cpu"),
        ref_stats_cache_dir=cache_dir,
        ref_config=ref_config,
        feature_batch_size=2,
        num_workers=0,
    )

    cache_files = list(cache_dir.glob("*.npz"))
    assert len(cache_files) == 1
    with np.load(cache_files[0]) as cached:
        assert cached["mu"].shape == (2,)
        assert cached["sigma"].shape == (2, 2)
        assert json.loads(str(cached["metadata"]))["num_reference_images"] == 2
    assert row["fid"] == pytest.approx(42.25)
    assert calls["compute_fid"] == 1
    assert calls["get_folder_features"] == 1
    assert calls["frechet_distance"] == 1


def test_cleanfid_uses_existing_reference_stats_cache_without_remote_call(monkeypatch, tmp_path):
    def unexpected_remote_call(*args, **kwargs):
        raise AssertionError("remote CleanFID should not be called when cache exists")

    calls = _install_fake_cleanfid(monkeypatch, compute_fid=unexpected_remote_call)
    ref_config = _tiny_fid_reference_config(tmp_path)
    cache_dir = tmp_path / "fid_reference_stats"
    cache_path = train_module.fid_reference_stats_path(
        cache_dir=cache_dir,
        ref_config=ref_config,
        dataset_name="imagenet",
        dataset_res=64,
        dataset_split="validation",
        mode="clean",
    )
    cache_path.parent.mkdir(parents=True)
    np.savez_compressed(
        cache_path,
        mu=np.asarray([1.0, 2.0]),
        sigma=np.eye(2),
        metadata=json.dumps({"num_reference_images": 2}),
    )

    row = run_fid_evaluation(
        net=TinySamplerNet(),
        output_dir=tmp_path,
        step=6,
        seed=2,
        num_samples=1,
        batch_size=1,
        num_steps=2,
        ref_split="validation",
        ref_dataset_name="imagenet",
        ref_dataset_res=64,
        fid_mode="clean",
        device=torch.device("cpu"),
        ref_stats_cache_dir=cache_dir,
        ref_config=ref_config,
        feature_batch_size=2,
        num_workers=0,
    )

    assert row["fid"] == pytest.approx(42.25)
    assert calls["compute_fid"] == 0
    assert calls["get_folder_features"] == 1


def test_cleanfid_non_download_error_is_not_handled_by_cache_fallback(monkeypatch, tmp_path):
    def broken_cleanfid(*args, **kwargs):
        raise RuntimeError("feature extraction failed")

    _install_fake_cleanfid(monkeypatch, compute_fid=broken_cleanfid)
    _patch_tiny_reference_loader(monkeypatch)

    with pytest.raises(RuntimeError, match="feature extraction failed"):
        train_module.compute_clean_fid_for_dataset(
            tmp_path,
            dataset_name="imagenet",
            dataset_res=64,
            dataset_split="validation",
            mode="clean",
            ref_stats_cache_dir=tmp_path / "fid_reference_stats",
            ref_config=_tiny_fid_reference_config(tmp_path),
            feature_batch_size=2,
            num_workers=0,
            device=torch.device("cpu"),
        )


def test_eval_completion_sync_uses_filesystem_marker(tmp_path):
    main_ctx = train_module.DistributedContext(
        enabled=True,
        rank=0,
        local_rank=0,
        world_size=2,
        is_main=True,
        device=torch.device("cpu"),
    )
    worker_ctx = train_module.DistributedContext(
        enabled=True,
        rank=1,
        local_rank=1,
        world_size=2,
        is_main=False,
        device=torch.device("cpu"),
    )

    assert train_module.sync_eval_completion(
        stop_after_eval=True,
        ctx=main_ctx,
        output_dir=tmp_path,
        step=7,
        run_id="test-run",
        poll_seconds=0.0,
    )
    assert train_module.sync_eval_completion(
        stop_after_eval=False,
        ctx=worker_ctx,
        output_dir=tmp_path,
        step=7,
        run_id="test-run",
        poll_seconds=0.0,
    )

    payload = json.loads((tmp_path / ".dist_sync" / "test-run" / "step000007.json").read_text())
    assert payload == {"step": 7, "stop_after_eval": True}


def test_hybrid_distillation_loss_has_finite_gradients():
    student = ScaleDenoiser(0.1)
    teacher = ScaleDenoiser(0.5).requires_grad_(False)
    images = torch.randn(3, 3, 8, 8)
    sigmas = torch.tensor([1.0, 0.5, 0.25])
    noise = torch.randn_like(images)

    loss, metrics = hybrid_distillation_loss(
        student=student,
        teacher=teacher,
        images=images,
        labels=None,
        sigmas=sigmas,
        noise=noise,
        model_family="vp",
        kd_weight=1.0,
        data_weight=0.25,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics["kd_loss"] >= 0
    assert torch.isfinite(student.scale.grad)
