import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from pace.benchmark_pipeline import (
    PFI_OUTPUT_NAMESPACE,
    PFI_PROFILE_PROTOCOL_ID,
    PIPELINE_PRESETS,
    SUPPORTED_PIPELINE_VARIANTS,
    PipelineAction,
    PreflightError,
    build_pipeline_actions,
    execute_pipeline_actions,
    inspect_lsun_flat_directory,
    parse_gpu_ids,
)
from pace.evaluation_protocols import FFHQ64_NVIDIA_PROTOCOL


REPO_ROOT = Path(__file__).resolve().parents[1]


def _script_and_flags(command):
    script_index = next(index for index, token in enumerate(command) if token.startswith("scripts/") and token.endswith(".py"))
    script = command[script_index]
    flags = {token.split("=", 1)[0] for token in command[script_index + 1 :] if token.startswith("--")}
    return script, flags


def _flag_values(command, flag):
    return [command[index + 1] for index, token in enumerate(command[:-1]) if token == flag]


def test_parse_gpu_ids_requires_explicit_unique_nonnegative_ids():
    assert parse_gpu_ids("0,3,4,5") == (0, 3, 4, 5)
    with pytest.raises(Exception, match="unique"):
        parse_gpu_ids("0,0")


def test_ffhq_pipeline_contains_approved_operations_and_protocols(tmp_path):
    assert PIPELINE_PRESETS["ffhq64"].default_gpu_ids == tuple(range(8))
    assert PIPELINE_PRESETS["ffhq64"].required_gpu_count == 8
    layout, actions = build_pipeline_actions(
        PIPELINE_PRESETS["ffhq64"],
        source_root=tmp_path / "ffhq256",
        shared_root=tmp_path / "shared",
        gpu_ids=(0, 3, 4, 5),
        variants=("global",),
        project_root=tmp_path / "repo",
        adm_evaluator=tmp_path / "guided" / "evaluations" / "evaluator.py",
        adm_python=tmp_path / "adm-venv" / "bin" / "python",
        nvlabs_edm_root=tmp_path / "edm",
    )
    assert layout.dataset_root.name == "ffhq256_to64_lanczos_v1"
    assert layout.run_root == tmp_path / "shared" / "benchmarks" / "ffhq64" / PFI_OUTPUT_NAMESPACE
    assert layout.analysis_root == tmp_path / "repo" / "out_eval_edm_ffhq64" / PFI_OUTPUT_NAMESPACE
    assert layout.profile_root == layout.analysis_root
    assert layout.plans_root == layout.analysis_root / "plans"
    assert layout.training_root == layout.run_root / "training"
    assert layout.benchmark_root == layout.run_root / "benchmark"
    assert layout.profile_protocol_id == PFI_PROFILE_PROTOCOL_ID
    assert layout.dataset_manifest == tmp_path / "shared" / "benchmarks" / "ffhq64" / "dataset" / "manifest.json"
    by_id = {action.action_id: action for action in actions}
    prepare = by_id["prepare:ffhq64"]
    assert prepare.output_paths == (str(layout.dataset_root / "dataset_manifest.json"),)
    manifest = by_id["prepare:ffhq_manifest"]
    assert "--dataset-manifest" in manifest.command
    assert "--output-manifest" not in manifest.command
    assert manifest.command[manifest.command.index("--image-size") + 1] == "64"
    assert sum("--dataset-preflight" in action.command for action in actions) == 0
    assert sum(any(token.endswith("preflight_edm_dataset.py") for token in action.command) for action in actions) == 1

    profile = by_id["profile:teacher"]
    assert "--confirm-full-profile" in profile.command
    assert "--max_groups" not in profile.command and "--max-groups" not in profile.command
    assert profile.command[profile.command.index("--ablation_mode") + 1] == "pfi"
    assert profile.command[profile.command.index("--pfi_seed") + 1] == "0"
    assert profile.command[profile.command.index("--distributed-timeout-seconds") + 1] == "86400"
    grouping = by_id["group:timesteps"]
    assert grouping.command[grouping.command.index("--num_blocks") + 1] == "3"
    assert grouping.command[grouping.command.index("--builtin_cost") + 1] == "matrix_correlation"
    cost_curve = layout.profile_root / "grouping" / "timestep_grouping_cost_curve.png"
    assert grouping.command[grouping.command.index("--cost_curve_output") + 1] == str(cost_curve)
    assert str(cost_curve) in grouping.output_paths
    allocation = by_id["allocate:students"]
    assert allocation.command[allocation.command.index("--allocation-metric") + 1] == (
        "delta_p_eff_geomean"
    )
    assert allocation.command[allocation.command.index("--score-reduction") + 1] == "sum"
    assert _flag_values(allocation.command, "--student-variant") == ["global"]
    assert allocation.command[allocation.command.index("--shuffle-seed") + 1] == "3"
    plans = by_id["plans:students"]
    assert _flag_values(plans.command, "--variant") == ["global"]
    assert plans.command[plans.command.index("--shuffle-seed") + 1] == "3"
    train = by_id["train:global"]
    for flag, value in {
        "--batch-size": "512",
        "--microbatch": "16",
        "--val-every": "5000",
        "--val-max-images": "10000",
        "--val-seed": "12345",
        "--early-stop-patience": "4",
        "--early-stop-min-steps": "10000",
        "--fid-every": "0",
        "--snapshot-every": "5000",
        "--keep-last-snapshots": "0",
        "--checkpoint-selection": "best_val",
    }.items():
        assert train.command[train.command.index(flag) + 1] == value
    assert train.output_paths == (str(layout.training_root / "global" / "seed0" / "student-best-val.pt"),)

    benchmark = [action for action in actions if action.stage == "benchmark"]
    assert len(benchmark) == 12  # Teacher + student: three FIDs, aggregate, ADM suite, final cleanup.
    assert all("--dataset-manifest" in action.command for action in benchmark if "evaluate_edm_checkpoint.py" in action.command)
    assert any("--network-preset" in action.command for action in benchmark)
    assert any(
        "--checkpoint" in action.command and any(token.endswith("student-best-val.pt") for token in action.command)
        for action in benchmark
    )
    summaries = [action for action in benchmark if "summarize_edm_benchmark.py" in action.command]
    assert all("fid_nvlabs_legacy" in action.command for action in summaries)
    assert grouping.upstream_fingerprints == (profile.fingerprint,)
    assert allocation.upstream_fingerprints == (grouping.fingerprint,)
    assert plans.upstream_fingerprints == (allocation.fingerprint,)
    assert train.upstream_fingerprints == (plans.fingerprint,)
    student_run = by_id[
        f"benchmark:global:{FFHQ64_NVIDIA_PROTOCOL.identity}:run0"
    ]
    assert student_run.upstream_fingerprints == (train.fingerprint,)
    student_aggregate = by_id[
        f"benchmark:global:{FFHQ64_NVIDIA_PROTOCOL.identity}:aggregate"
    ]
    assert student_run.fingerprint in student_aggregate.upstream_fingerprints
    serialized = student_aggregate.to_dict()
    assert serialized["upstream_fingerprints"] == list(student_aggregate.upstream_fingerprints)
    assert serialized["fingerprint"] == student_aggregate.fingerprint


def test_bedroom_pipeline_uses_approved_batch_precision_and_crop_size(tmp_path):
    assert PIPELINE_PRESETS["lsun_bedroom256"].default_variants == (
        "global",
        "uniform_blockwise",
        "combined_blockwise",
        "combined_layerwise",
    )
    layout, actions = build_pipeline_actions(
        PIPELINE_PRESETS["lsun_bedroom256"],
        source_root=tmp_path / "lsun",
        shared_root=tmp_path / "shared",
        gpu_ids=tuple(range(8)),
        variants=("global",),
        project_root=tmp_path / "repo",
    )
    by_id = {action.action_id: action for action in actions}
    assert layout.analysis_root == (
        tmp_path / "repo" / "out_eval_edm_lsun_bedroom256" / PFI_OUTPUT_NAMESPACE
    )
    assert layout.profile_root == layout.analysis_root
    assert layout.training_root == layout.run_root / "training"
    manifest = by_id["prepare:lsun_bedroom_manifest"]
    assert manifest.command[manifest.command.index("--image-size") + 1] == "256"
    grouping = by_id["group:timesteps"]
    assert grouping.command[grouping.command.index("--builtin_cost") + 1] == (
        "matrix_correlation_cross_penalty"
    )
    allocation = by_id["allocate:students"]
    assert allocation.command[allocation.command.index("--allocation-metric") + 1] == "delta_p_eff_geomean"
    assert allocation.command[allocation.command.index("--score-reduction") + 1] == "sum"
    assert _flag_values(allocation.command, "--student-variant") == ["global"]
    # Full decode is deliberately centralized in the prepare action; running it
    # under torchrun would decode one million JPEGs independently on every rank.
    assert sum("--dataset-preflight" in action.command for action in actions) == 0
    assert sum(any(token.endswith("preflight_edm_dataset.py") for token in action.command) for action in actions) == 1
    train = by_id["train:global"]
    assert train.command[train.command.index("--batch-size") + 1] == "64"
    assert train.command[train.command.index("--microbatch") + 1] == "1"
    teacher_benchmark = by_id["benchmark:teacher:lsun_bedroom256_openai_adm_v1"]
    assert teacher_benchmark.command[teacher_benchmark.command.index("--teacher-dtype") + 1] == "fp16"
    student_benchmark = by_id["benchmark:global:lsun_bedroom256_openai_adm_v1"]
    assert student_benchmark.command[student_benchmark.command.index("--teacher-dtype") + 1] == "fp32"


def test_pipeline_explicitly_overrides_timestep_blocks_and_shared_shuffle_seed(tmp_path):
    _, actions = build_pipeline_actions(
        PIPELINE_PRESETS["ffhq64"],
        source_root=tmp_path / "ffhq",
        shared_root=tmp_path / "shared",
        gpu_ids=(0, 3, 4, 5),
        variants=("shuffled_capacity",),
        num_timestep_blocks=5,
        grouping_builtin_cost="matrix_cosine",
        allocation_metric="n_eff",
        score_reduction="q90",
        shuffle_seed=17,
    )
    by_id = {action.action_id: action for action in actions}

    grouping = by_id["group:timesteps"].command
    allocation = by_id["allocate:students"].command
    plans = by_id["plans:students"].command
    assert grouping[grouping.index("--num_blocks") + 1] == "5"
    assert grouping[grouping.index("--builtin_cost") + 1] == "matrix_cosine"
    assert allocation[allocation.index("--allocation-metric") + 1] == "n_eff"
    assert allocation[allocation.index("--score-reduction") + 1] == "q90"
    assert _flag_values(allocation, "--student-variant") == ["shuffled_capacity"]
    assert _flag_values(plans, "--variant") == ["shuffled_capacity"]
    assert allocation[allocation.index("--shuffle-seed") + 1] == "17"
    assert plans[plans.index("--shuffle-seed") + 1] == "17"


def test_ffhq_preset_defaults_to_combined_delta_peff_experiment(tmp_path):
    preset = PIPELINE_PRESETS["ffhq64"]
    layout, actions = build_pipeline_actions(
        preset,
        source_root=tmp_path / "ffhq",
        shared_root=tmp_path / "shared",
        gpu_ids=(0, 3, 4, 5),
    )
    expected_variants = [
        "global",
        "uniform_blockwise",
        "combined_blockwise",
        "combined_layerwise",
    ]
    by_id = {action.action_id: action for action in actions}
    allocation = by_id["allocate:students"].command
    plans = by_id["plans:students"].command

    assert preset.default_variants == tuple(expected_variants)
    assert preset.allocation_metric == "delta_p_eff_geomean"
    assert preset.score_reduction == "sum"
    assert {"combined_blockwise", "combined_layerwise"} <= set(SUPPORTED_PIPELINE_VARIANTS)
    assert allocation[allocation.index("--allocation-metric") + 1] == "delta_p_eff_geomean"
    assert allocation[allocation.index("--score-reduction") + 1] == "sum"
    assert _flag_values(allocation, "--student-variant") == expected_variants
    assert _flag_values(plans, "--variant") == expected_variants
    assert {action.action_id for action in actions if action.stage == "train"} == {
        f"train:{variant}" for variant in expected_variants
    }
    assert layout.plans_root == layout.analysis_root / "plans"


def test_combined_pipeline_variants_require_geomean_sum_but_allow_controls(tmp_path):
    with pytest.raises(PreflightError, match="combined variants.*require"):
        build_pipeline_actions(
            PIPELINE_PRESETS["ffhq64"],
            source_root=tmp_path / "ffhq",
            shared_root=tmp_path / "shared",
            gpu_ids=(0, 3, 4, 5),
            variants=("global", "uniform_blockwise", "combined_blockwise"),
            allocation_metric="n_eff",
            score_reduction="q90",
        )

    _, actions = build_pipeline_actions(
        PIPELINE_PRESETS["ffhq64"],
        source_root=tmp_path / "ffhq",
        shared_root=tmp_path / "shared",
        gpu_ids=(0, 3, 4, 5),
        variants=("global", "uniform_blockwise", "combined_blockwise"),
        allocation_metric="delta_p_eff_geomean",
        score_reduction="sum",
    )
    assert {action.action_id for action in actions if action.stage == "train"} == {
        "train:global",
        "train:uniform_blockwise",
        "train:combined_blockwise",
    }


def test_upstream_fingerprints_invalidate_plans_and_training_when_allocation_changes(tmp_path):
    common = {
        "source_root": tmp_path / "ffhq",
        "shared_root": tmp_path / "shared",
        "gpu_ids": (0, 3, 4, 5),
        "variants": ("global", "uniform_blockwise"),
        "project_root": tmp_path / "repo",
    }
    _, q90_actions = build_pipeline_actions(
        PIPELINE_PRESETS["ffhq64"],
        **common,
        num_timestep_blocks=4,
        grouping_builtin_cost="matrix_correlation_cross_penalty",
        allocation_metric="n_eff",
        score_reduction="q90",
    )
    _, geomean_actions = build_pipeline_actions(
        PIPELINE_PRESETS["ffhq64"],
        **common,
        num_timestep_blocks=3,
        grouping_builtin_cost="matrix_correlation",
        allocation_metric="delta_p_eff_geomean",
        score_reduction="sum",
    )
    q90 = {action.action_id: action for action in q90_actions}
    geomean = {action.action_id: action for action in geomean_actions}

    # These commands are intentionally identical; only the serialized upstream
    # chain makes their identities reflect the changed grouping/allocation rule.
    assert q90["plans:students"].command == geomean["plans:students"].command
    assert q90["train:global"].command == geomean["train:global"].command
    assert q90["plans:students"].fingerprint != geomean["plans:students"].fingerprint
    assert q90["train:global"].fingerprint != geomean["train:global"].fingerprint
    assert (
        q90[f"benchmark:global:{FFHQ64_NVIDIA_PROTOCOL.identity}:run0"].fingerprint
        != geomean[f"benchmark:global:{FFHQ64_NVIDIA_PROTOCOL.identity}:run0"].fingerprint
    )


@pytest.mark.parametrize("ablation_mode", ["zero", "random_same_norm"])
def test_explicit_legacy_profile_modes_keep_historical_root_layout(tmp_path, ablation_mode):
    layout, actions = build_pipeline_actions(
        PIPELINE_PRESETS["ffhq64"],
        source_root=tmp_path / "ffhq",
        shared_root=tmp_path / "shared",
        gpu_ids=(0, 3, 4, 5),
        variants=("global",),
        profile_ablation_mode=ablation_mode,
        project_root=tmp_path / "repo",
    )

    assert layout.run_root == tmp_path / "shared" / "benchmarks" / "ffhq64"
    assert layout.analysis_root == layout.run_root
    assert layout.profile_root == layout.run_root / "profile"
    assert layout.profile_protocol_id == "legacy_unspecified"
    profile = next(action for action in actions if action.action_id == "profile:teacher")
    assert profile.command[profile.command.index("--ablation_mode") + 1] == ablation_mode
    assert "--pfi_seed" not in profile.command


def test_every_generated_script_flag_appears_in_its_help(tmp_path):
    _, actions = build_pipeline_actions(
        PIPELINE_PRESETS["ffhq64"],
        source_root=tmp_path / "ffhq",
        shared_root=tmp_path / "shared",
        gpu_ids=(0, 3, 4, 5),
        variants=("global",),
    )
    expected_by_script = {}
    for action in actions:
        script, flags = _script_and_flags(action.command)
        expected_by_script.setdefault(script, set()).update(flags)
    for script, flags in expected_by_script.items():
        completed = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
            env={**os.environ, "MKL_THREADING_LAYER": "GNU"},
        )
        advertised = set(re.findall(r"--[A-Za-z0-9_-]+", completed.stdout))
        assert flags <= advertised, f"{script} does not advertise generated flags {sorted(flags - advertised)}"


def test_lsun_flat_inspection_detects_names_subdirs_and_permissions(tmp_path):
    (tmp_path / "0000000.jpg").write_bytes(b"one")
    unreadable = tmp_path / "0000001.jpg"
    unreadable.write_bytes(b"two")
    unreadable.chmod(0)
    (tmp_path / "nested").mkdir()
    details = inspect_lsun_flat_directory(tmp_path, expected_count=2)
    assert details["image_count"] == 2
    assert details["unreadable_count"] == 1
    assert details["subdirectory_examples"] == ["nested"]


def test_executor_only_resumes_when_all_declared_outputs_still_exist(monkeypatch, tmp_path):
    output = tmp_path / "artifact.json"
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        output.write_text("{}")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    action = PipelineAction(
        action_id="test",
        stage="prepare",
        command=(sys.executable, "-c", "pass"),
        environment={},
        output_paths=(str(output),),
    )
    state = tmp_path / "state.json"
    execute_pipeline_actions([action], state_path=state, repo_root=REPO_ROOT)
    second = execute_pipeline_actions([action], state_path=state, repo_root=REPO_ROOT)
    assert second["skipped"] == ["test"]
    output.unlink()
    third = execute_pipeline_actions([action], state_path=state, repo_root=REPO_ROOT)
    assert third["executed"] == ["test"]
    assert len(calls) == 2


def test_executor_refuses_mismatched_automatic_resume_without_touching_state(monkeypatch, tmp_path):
    resume_dir = tmp_path / "training"
    resume_dir.mkdir()
    checkpoint_marker = resume_dir / "latest-checkpoint.json"
    checkpoint_marker.write_text('{"path":"snapshot.pt"}\n')
    action = PipelineAction(
        action_id="train:fixture",
        stage="train",
        command=(sys.executable, "-c", "pass"),
        environment={},
        resume_output_dir=str(resume_dir),
    )
    state_path = tmp_path / "state.json"
    original_state = {
        "state_format": "diffdist_edm_pipeline_state_v1",
        "actions": {
            action.action_id: {
                "status": "failed",
                "fingerprint": "0" * 64,
            }
        },
    }
    state_path.write_text(json.dumps(original_state, indent=2) + "\n")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("mismatched resume must not launch a process"),
    )

    with pytest.raises(PreflightError, match="refusing automatic resume.*left untouched"):
        execute_pipeline_actions([action], state_path=state_path, repo_root=REPO_ROOT)

    assert json.loads(state_path.read_text()) == original_state
    assert checkpoint_marker.read_text() == '{"path":"snapshot.pt"}\n'


def test_executor_automatically_resumes_only_matching_action_state(monkeypatch, tmp_path):
    resume_dir = tmp_path / "training"
    resume_dir.mkdir()
    (resume_dir / "latest-checkpoint.json").write_text("{}\n")
    action = PipelineAction(
        action_id="train:fixture",
        stage="train",
        command=(sys.executable, "-c", "pass"),
        environment={},
        resume_output_dir=str(resume_dir),
    )
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "state_format": "diffdist_edm_pipeline_state_v1",
                "actions": {
                    action.action_id: {
                        "status": "failed",
                        "fingerprint": action.fingerprint,
                    }
                },
            }
        )
    )
    calls = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = execute_pipeline_actions([action], state_path=state_path, repo_root=REPO_ROOT)

    assert result["executed"] == [action.action_id]
    assert calls == [[*action.command, "--resume", "auto"]]


def test_driver_dry_run_emits_exact_resume_state_and_log_paths(tmp_path):
    source = tmp_path / "ffhq256"
    source.mkdir()
    shared = tmp_path / "shared"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/paper/run_edm_benchmark_pipeline.py",
            "--preset",
            "ffhq64",
            "--source-root",
            str(source),
            "--shared-root",
            str(shared),
            "--gpu-ids",
            "0,3,4,5",
            "--min-free-memory-mib",
            "0",
            "--min-free-storage-gib",
            "0",
            "--stage",
            "prepare",
            "--variant",
            "global",
            "--variant",
            "uniform_blockwise",
            "--num-timestep-blocks",
            "5",
            "--shuffle-seed",
            "17",
            "--grouping-builtin-cost",
            "matrix_cosine",
            "--allocation-metric",
            "n_eff",
            "--score-reduction",
            "q90",
        ],
        check=True,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env={**os.environ, "MKL_THREADING_LAYER": "GNU"},
    )
    payload = json.loads(completed.stdout)

    assert payload["preflight"]["passed"] is True
    namespace_root = shared / "benchmarks" / "ffhq64" / PFI_OUTPUT_NAMESPACE
    assert payload["driver"]["state_path"] == str(namespace_root / "pipeline_state.json")
    assert payload["driver"]["recommended_log_path"] == str(
        namespace_root / "pipeline_driver.log"
    )
    analysis_root = REPO_ROOT / "out_eval_edm_ffhq64" / PFI_OUTPUT_NAMESPACE
    assert payload["layout"]["analysis_root"] == str(analysis_root)
    assert payload["layout"]["profile_root"] == str(analysis_root)
    assert payload["layout"]["plans_root"] == str(analysis_root / "plans")
    assert payload["layout"]["training_root"] == str(namespace_root / "training")
    assert payload["layout"]["benchmark_root"] == str(namespace_root / "benchmark")
    resume = payload["driver"]["resume_command"]
    assert resume[-1] == "--execute"
    assert resume[resume.index("--stage") + 1] == "prepare"
    assert _flag_values(resume, "--variant") == ["global", "uniform_blockwise"]
    assert resume[resume.index("--profile-ablation-mode") + 1] == "pfi"
    assert resume[resume.index("--num-timestep-blocks") + 1] == "5"
    assert resume[resume.index("--shuffle-seed") + 1] == "17"
    assert resume[resume.index("--grouping-builtin-cost") + 1] == "matrix_cosine"
    assert resume[resume.index("--allocation-metric") + 1] == "n_eff"
    assert resume[resume.index("--score-reduction") + 1] == "q90"
    assert payload["driver"]["num_timestep_blocks"] == 5
    assert payload["driver"]["grouping_builtin_cost"] == "matrix_cosine"
    assert payload["driver"]["allocation_metric"] == "n_eff"
    assert payload["driver"]["score_reduction"] == "q90"
    assert payload["driver"]["variants"] == ["global", "uniform_blockwise"]
    assert payload["driver"]["shuffle_seed"] == 17


def _ffhq_finalizer_fixture(tmp_path):
    results = []
    nv_values = (9.5, 8.25, 10.0)
    for index, (seed, value) in enumerate(zip((0, 50_000, 100_000), nv_values)):
        result_path = tmp_path / f"nv{index}" / "evaluation_result.json"
        manifest_path = result_path.parent / "benchmark_manifest.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_sha = f"manifest-nv-{index}"
        manifest_path.write_text(json.dumps({
            "manifest_sha256": manifest_sha,
            "protocol": {"identity": "ffhq64_nvlabs_edm_fid_v1"},
        }))
        result_path.write_text(json.dumps({
            "status": "complete",
            "benchmark_protocol_id": "ffhq64_nvlabs_edm_fid_v1",
            "sampling": {"seed": seed, "num_samples": 50_000},
            "metrics": {"fid_nvlabs_legacy": value},
            "benchmark_manifest": {"path": str(manifest_path), "manifest_sha256": manifest_sha},
        }))
        results.append(result_path)

    adm_result = tmp_path / "adm" / "evaluation_result.json"
    adm_manifest = adm_result.parent / "benchmark_manifest.json"
    adm_manifest.parent.mkdir(parents=True)
    adm_manifest.write_text(json.dumps({
        "manifest_sha256": "manifest-adm",
        "protocol": {"identity": "ffhq64_openai_adm_custom_first50k_v1"},
    }))
    adm_result.write_text(json.dumps({
        "status": "complete",
        "benchmark_protocol_id": "ffhq64_openai_adm_custom_first50k_v1",
        "sampling": {"seed": 0, "num_samples": 50_000},
        "metrics": {
            "fid_adm": 1.0,
            "sfid_adm": 2.0,
            "precision_adm": 0.7,
            "recall_adm": 0.6,
            "inception_score_adm": 3.0,
        },
        "benchmark_manifest": {"path": str(adm_manifest), "manifest_sha256": "manifest-adm"},
    }))
    results.append(adm_result)

    summary = tmp_path / "evaluation_summary.json"
    summary.write_text(json.dumps({
        "protocol_id": "ffhq64_nvlabs_edm_fid_v1",
        "metric": "fid_nvlabs_legacy",
        "aggregation": "minimum",
        "aggregate": min(nv_values),
        "selected_run_index": nv_values.index(min(nv_values)),
        "runs": [
            {"path": str(path.resolve()), "value": value}
            for path, value in zip(results[:3], nv_values)
        ],
    }))
    sample_dirs = []
    for index in range(4):
        sample_dir = tmp_path / f"samples{index}"
        sample_dir.mkdir()
        for seed in range(2):
            (sample_dir / f"seed{seed:06d}.png").write_bytes(f"{index}-{seed}".encode())
        sample_dirs.append(sample_dir)
    metric_artifact = tmp_path / "adm_samples.npz"
    metric_artifact.write_bytes(b"fixture")
    return results, summary, sample_dirs, metric_artifact


def _run_ffhq_finalizer(tmp_path, results, summary, sample_dirs, metric_artifact):
    command = [
        sys.executable,
        "scripts/paper/finalize_edm_benchmark_artifacts.py",
        "--summary",
        str(summary),
        "--metric-artifact",
        str(metric_artifact),
        "--retention",
        "keep_preview",
        "--preview-dir",
        str(tmp_path / "preview"),
        "--preview-count",
        "2",
        "--expected-samples-per-dir",
        "2",
        "--output",
        str(tmp_path / "artifact_cleanup.json"),
    ]
    for result in results:
        command += ["--result", str(result)]
    for sample_dir in sample_dirs:
        command += ["--sample-dir", str(sample_dir)]
    return subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)


def test_ffhq_finalizer_deletes_only_after_complete_correlated_metric_suite(tmp_path):
    results, summary, sample_dirs, metric_artifact = _ffhq_finalizer_fixture(tmp_path)

    completed = _run_ffhq_finalizer(tmp_path, results, summary, sample_dirs, metric_artifact)

    assert completed.returncode == 0, completed.stderr
    cleanup = json.loads((tmp_path / "artifact_cleanup.json").read_text())
    assert cleanup["status"] == "complete"
    assert len(list((tmp_path / "preview").glob("*.png"))) == 2
    assert not any(list(sample_dir.glob("seed*.png")) for sample_dir in sample_dirs)
    assert not metric_artifact.exists()
    assert all(path.is_file() for path in results)


def test_ffhq_finalizer_failure_is_non_destructive(tmp_path):
    results, summary, sample_dirs, metric_artifact = _ffhq_finalizer_fixture(tmp_path)
    payload = json.loads(summary.read_text())
    payload["runs"][0]["value"] = 999.0
    summary.write_text(json.dumps(payload))

    completed = _run_ffhq_finalizer(tmp_path, results, summary, sample_dirs, metric_artifact)

    assert completed.returncode != 0
    assert "summary value does not match" in completed.stderr
    assert all(len(list(sample_dir.glob("seed*.png"))) == 2 for sample_dir in sample_dirs)
    assert metric_artifact.is_file()
    assert not (tmp_path / "artifact_cleanup.json").exists()
