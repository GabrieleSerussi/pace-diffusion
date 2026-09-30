import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import evaluate_parameters_diffwave as diffwave_cli
from pace.parameter_analysis import DiffusionAnalysisAdapter
from pace.teacher_models import resolve_teacher_spec
from pace.vendor.diffwave_legacy import (
    DIFFWAVE_SASHIMI_CHECKPOINTS_COMMIT,
    DIFFWAVE_SASHIMI_CHECKPOINT_SHA256,
    LegacyCompatibleDiffWave,
    diffwave_diffusion_hyperparameters,
)


class _TinyAudioDataset(torch.utils.data.Dataset):
    def __init__(self, size=2, length=16):
        self.items = [
            (torch.linspace(-0.5, 0.5, length).reshape(1, length) + index * 0.01, index)
            for index in range(size)
        ]
        self.metadata = {
            "dataset_id": "tiny_sc09",
            "protocol": "fixture",
            "split": "validation",
            "is_heldout": True,
            "selected_count": size,
            "selected_entries_sha256": "a" * 64,
            "source_listing_sha256": "b" * 64,
            "source_records_sha256": "c" * 64,
            "sample_rate": 16000,
            "sample_length": length,
            "channels": 1,
            "input_representation": "raw_waveform",
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def _tiny_model(num_blocks=3):
    return LegacyCompatibleDiffWave(
        res_channels=4,
        skip_channels=4,
        num_res_layers=num_blocks,
        dilation_cycle=max(1, num_blocks),
        diffusion_step_embed_dim_in=4,
        diffusion_step_embed_dim_mid=8,
        diffusion_step_embed_dim_out=8,
        num_diffusion_steps=200,
    ).eval()


def test_corruption_dataset_and_pfi_loader_keep_high_to_low_timesteps():
    audio = _TinyAudioDataset()
    corruption = diffwave_cli.DiffWaveCorruptionDataset(
        audio,
        num_diffusion_steps=200,
        seed=0,
        num_timestep_levels=2,
        samples_per_timestep=2,
    )
    loader, plan = diffwave_cli.build_dataloader(
        corruption,
        audio,
        ablation_mode="pfi",
        batch_size=2,
        num_workers=0,
        pfi_seed=0,
    )

    assert corruption.level_indices == [199, 0]
    assert plan is not None
    batches = list(loader)
    assert [int(batch[1][0]) for batch in batches] == [199, 0]
    assert all(batch[4].tolist() == [1, 0] for batch in batches)


def test_worker_loader_uses_spawn_and_persistent_workers_after_cuda_init():
    audio = _TinyAudioDataset()
    corruption = diffwave_cli.DiffWaveCorruptionDataset(
        audio,
        num_diffusion_steps=200,
        seed=0,
        num_timestep_levels=2,
        samples_per_timestep=2,
    )

    loader, plan = diffwave_cli.build_dataloader(
        corruption,
        audio,
        ablation_mode="pfi",
        batch_size=2,
        num_workers=2,
        pfi_seed=0,
    )

    assert plan is not None
    assert loader.num_workers == 2
    assert loader.persistent_workers is True
    assert loader.multiprocessing_context is not None
    assert loader.multiprocessing_context.get_start_method() == "spawn"


def test_research_pairing_uses_two_round_robin_examples_at_every_native_timestep():
    audio = _TinyAudioDataset(size=20)
    corruption = diffwave_cli.DiffWaveCorruptionDataset(
        audio,
        num_diffusion_steps=200,
        seed=0,
        num_timestep_levels=200,
        samples_per_timestep=2,
    )

    assert len(corruption) == 400
    assert corruption.num_examples == 2
    assert corruption.level_indices == list(range(199, -1, -1))
    for level_position in range(200):
        selected = [
            corruption.base_example_index(replica, level_position)
            for replica in range(2)
        ]
        assert selected[0] != selected[1]
    assert [
        corruption.base_example_index(replica, level)
        for level in range(10)
        for replica in range(2)
    ] == list(range(20))


def test_axis_metadata_uses_raw_timestep_labels_and_noise_coordinates():
    schedule = diffwave_diffusion_hyperparameters(num_diffusion_steps=200)
    labels, axis = diffwave_cli.make_timestep_axis_metadata(
        evaluated_timesteps=list(range(199, -1, -1)),
        num_diffusion_steps=200,
        num_bins=20,
        alpha_bar=schedule["Alpha_bar"],
    )

    assert labels[0] == "199-190"
    assert labels[-1] == "9-0"
    assert axis.ordering == "high_noise_to_low_noise"
    assert axis.normalized_values[0] == 1.0
    assert axis.normalized_values[-1] == 0.0
    assert len(axis.metadata["log_snr"]) == 200


def test_non_divisible_axis_memberships_match_runtime_level_binning():
    timesteps = [199, 150, 100, 50, 0]
    _, axis = diffwave_cli.make_timestep_axis_metadata(
        evaluated_timesteps=timesteps,
        num_diffusion_steps=200,
        num_bins=2,
        alpha_bar=diffwave_diffusion_hyperparameters(num_diffusion_steps=200)["Alpha_bar"],
    )
    assignments = diffwave_cli.level_to_bin(
        torch.arange(len(timesteps)),
        num_levels=len(timesteps),
        num_bins=2,
    ).tolist()
    expected = tuple(
        tuple(timestep for timestep, assigned in zip(timesteps, assignments) if assigned == bin_index)
        for bin_index in range(2)
    )

    assert axis.bin_members == expected == ((199, 150, 100), (50, 0))


def test_groups_are_dot_free_ordered_blocks_with_parameter_counts():
    model = _tiny_model(num_blocks=3)
    groups, module_paths = diffwave_cli.collect_diffwave_groups(model)
    assert list(groups) == ["residual_block_00", "residual_block_01", "residual_block_02"]
    assert module_paths["residual_block_02"] == "residual_layer.residual_blocks.2"
    assert all("." not in name for name in groups)
    assert all(diffwave_cli.count_parameters(module) > 0 for module in groups.values())


def test_all_36_blocks_have_disjoint_parameters_and_exact_shared_accounting():
    model = _tiny_model(num_blocks=36)
    groups, _ = diffwave_cli.collect_diffwave_groups(model)
    assert len(groups) == 36

    block_parameter_ids = [
        {id(parameter) for parameter in module.parameters()}
        for module in groups.values()
    ]
    seen: set[int] = set()
    for parameter_ids in block_parameter_ids:
        assert seen.isdisjoint(parameter_ids)
        seen.update(parameter_ids)

    model_parameters = list(model.parameters())
    shared_parameters = [parameter for parameter in model_parameters if id(parameter) not in seen]
    block_parameter_count = sum(
        parameter.numel()
        for module in groups.values()
        for parameter in module.parameters()
    )
    shared_parameter_count = sum(parameter.numel() for parameter in shared_parameters)
    assert block_parameter_count + shared_parameter_count == sum(
        parameter.numel() for parameter in model_parameters
    )


def test_adapter_forward_and_pfi_hook_produce_finite_binned_losses():
    audio = _TinyAudioDataset()
    corruption = diffwave_cli.DiffWaveCorruptionDataset(
        audio,
        num_diffusion_steps=200,
        seed=3,
        num_timestep_levels=2,
        samples_per_timestep=2,
    )
    loader, _ = diffwave_cli.build_dataloader(
        corruption,
        audio,
        ablation_mode="pfi",
        batch_size=2,
        num_workers=0,
        pfi_seed=5,
    )
    model = _tiny_model(num_blocks=3)
    groups, _ = diffwave_cli.collect_diffwave_groups(model)
    _, axis = diffwave_cli.make_timestep_axis_metadata(
        evaluated_timesteps=corruption.level_indices,
        num_diffusion_steps=200,
        num_bins=2,
        alpha_bar=diffwave_diffusion_hyperparameters(num_diffusion_steps=200)["Alpha_bar"],
    )
    evaluator = diffwave_cli.DiffWaveUsageEvaluator(
        model,
        diffwave_diffusion_hyperparameters(num_diffusion_steps=200),
        device=torch.device("cpu"),
        dtype=torch.float32,
        axis=axis,
        groups=groups,
    )
    assert isinstance(evaluator, DiffusionAnalysisAdapter)

    baseline = evaluator.evaluate(
        loader,
        num_bins=2,
        ablation_mode="pfi",
    )
    ablated = evaluator.evaluate(
        loader,
        num_bins=2,
        ablate_target=model.residual_blocks[0],
        ablation_mode="pfi",
    )
    assert baseline.count.tolist() == [2, 2]
    assert ablated.count.tolist() == [2, 2]
    assert torch.isfinite(baseline.mean()).all()
    assert torch.isfinite(ablated.mean()).all()


def test_verification_report_gate_checks_format_hash_commit_and_checks(tmp_path):
    close_metrics = {
        "elementwise_close": True,
        "max_abs": 0.0,
        "mean_abs": 0.0,
        "rmse": 0.0,
        "relative_l2": 0.0,
    }
    prediction_timesteps = [0, 1, 25, 50, 100, 150, 198, 199]
    trajectory_timesteps = list(range(199, 189, -1))
    vendor_source = Path(diffwave_cli.__file__).resolve().parents[2] / "pace" / "vendor" / "diffwave_legacy.py"
    teacher_loader_source = (
        Path(diffwave_cli.__file__).resolve().parents[2] / "pace" / "teacher_models.py"
    )
    report = {
        "format": diffwave_cli.VERIFICATION_FORMAT,
        "passed": True,
        "upstream": {
            "commit": DIFFWAVE_SASHIMI_CHECKPOINTS_COMMIT,
            "repository": "/external/diffwave-sashimi-checkpoints",
        },
        "checkpoint": {"actual_sha256": DIFFWAVE_SASHIMI_CHECKPOINT_SHA256},
        "checks": {
            "prediction_and_trajectory_parity": True,
            "autograd_safe": True,
            "valid_audio": True,
            "untouched_reference_generation": True,
            "generated_audio": True,
            "reference_environment_captured": True,
        },
        "implementation": {
            "legacy_forward_semantics": "timestep_projection_in_residual_identity_v1",
            "checkpoint_format": "diffwave_sashimi_legacy_state_dict_v1",
            "strict_state_dict_load": True,
            "checkpoint_container_key": "model_state_dict",
            "source_sha256": diffwave_cli._file_sha256(vendor_source),
            "teacher_loader_sha256": diffwave_cli._file_sha256(teacher_loader_source),
        },
        "configuration": {
            "batch_size": 1,
            "prediction_length": 16_000,
            "prediction_timesteps": prediction_timesteps,
            "trajectory_steps": 10,
            "trajectory_length": 16_000,
            "atol": 1e-6,
            "rtol": 1e-6,
            "relative_l2_tolerance": 1e-6,
        },
        "prediction_parity": [
            {"timestep": timestep, **close_metrics}
            for timestep in prediction_timesteps
        ],
        "trajectory_parity": {
            "timesteps": trajectory_timesteps,
            "per_step": [
                {
                    "timestep": timestep,
                    "epsilon": dict(close_metrics),
                    "state": dict(close_metrics),
                }
                for timestep in trajectory_timesteps
            ],
        },
        "autograd": {
            "input_gradient_finite": True,
            "input_unchanged_by_residual_blocks": True,
            "input_gradient_max_abs": 1.0,
        },
        "generated_audio": {
            "finite": True,
            "sample_rate": 16_000,
            "num_samples": 16_000,
            "rms": 0.1,
        },
        "untouched_reference_audio": {
            "finite": True,
            "dtype": "float32",
            "num_channels": 1,
            "sample_rate": 16_000,
            "num_samples": 16_000,
            "rms": 0.1,
            "standard_deviation": 0.1,
        },
        "untouched_reference_generation": {
            "working_directory": "/external/diffwave-sashimi-checkpoints",
            "cuda_visible_devices": "0",
            "argv": [
                "/reference/bin/python",
                "generate.py",
                "experiment=sc09",
                "model=wavenet",
                "generate.ckpt_iter=1000000",
                "generate.n_samples=1",
                "generate.batch_size=1",
            ],
            "command": (
                "cd /external/diffwave-sashimi-checkpoints\n"
                "CUDA_VISIBLE_DEVICES=0 /reference/bin/python generate.py "
                "experiment=sc09 model=wavenet generate.ckpt_iter=1000000 "
                "generate.n_samples=1 generate.batch_size=1"
            ),
            "environment": {
                "interpreter": "/reference/bin/python",
                "resolved_interpreter": "/base/bin/python3.11",
                "python": "3.11.13",
                "platform": "Linux-test",
                "torch": "2.10.0+cu128",
                "torchaudio": "2.10.0+cu128",
                "cuda_available": True,
                "cuda_runtime": "12.8",
                "cudnn": 91002,
                "pip_freeze_command": "/reference/bin/python -m pip freeze",
                "pip_freeze": ["torch==2.10.0", "torchaudio==2.10.0"],
                "pip_freeze_sha256": hashlib.sha256(
                    b"torch==2.10.0\ntorchaudio==2.10.0\n"
                ).hexdigest(),
            },
        },
    }
    path = tmp_path / "report.json"

    path.write_text(json.dumps({
        "format": diffwave_cli.VERIFICATION_FORMAT,
        "passed": True,
        "upstream": report["upstream"],
        "checkpoint": report["checkpoint"],
        "checks": report["checks"],
    }))
    with pytest.raises(ValueError, match="safe legacy implementation"):
        diffwave_cli.validate_teacher_verification_report(path)

    path.write_text(json.dumps(report))
    validated = diffwave_cli.validate_teacher_verification_report(path)
    assert validated["passed"] is True
    assert len(validated["sha256"]) == 64

    valid_reference_generation = report["untouched_reference_generation"]
    del report["untouched_reference_generation"]
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="reference generation provenance"):
        diffwave_cli.validate_teacher_verification_report(path)
    report["untouched_reference_generation"] = {
        "working_directory": "/external/diffwave-sashimi-checkpoints"
    }
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="invalid reference environment"):
        diffwave_cli.validate_teacher_verification_report(path)

    report["untouched_reference_generation"] = dict(valid_reference_generation)
    del report["untouched_reference_generation"]["command"]
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="exact untouched generation command"):
        diffwave_cli.validate_teacher_verification_report(path)

    report["checks"]["valid_audio"] = False
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="required passing check"):
        diffwave_cli.validate_teacher_verification_report(path)


def test_config_presets_pin_sc09_root_and_native_schedule():
    root = Path(__file__).resolve().parents[1]
    for name, expected_groups, expected_bins in [
        ("diffwave_sc09_analysis_smoke.json", 2, 4),
        ("diffwave_sc09_analysis_research.json", None, 20),
    ]:
        config = json.loads((root / "reproduce" / "configs" / name).read_text())
        assert config["data_root"] == "data/sc09_v0.02"
        assert config["teacher_preset"] == "sc09_diffwave_legacy_1m"
        assert config["num_timestep_levels"] == (4 if expected_groups == 2 else 200)
        assert config["samples_per_timestep"] == 2
        assert config["max_groups"] == expected_groups
        assert config["num_bins"] == expected_bins
        assert config["score_reduction"] == "q90"
        parsed = diffwave_cli.parse_args(["--config", str(root / "reproduce" / "configs" / name)])
        assert parsed.data_root == config["data_root"]
        assert parsed.num_timestep_levels == config["num_timestep_levels"]
        assert parsed.samples_per_timestep == 2


def test_per_filter_presets_share_the_exact_eight_gpu_scientific_basis():
    root = Path(__file__).resolve().parents[1]
    expected = {
        "diffwave_sc09_analysis_per_filter_smoke.json": (16, False, False, 1),
        "diffwave_sc09_analysis_per_filter_stability.json": (256, True, False, 4),
        "diffwave_sc09_analysis_per_filter_research.json": (None, True, True, 8),
    }
    for name, (max_groups, positive_gate, postprocess, interval) in expected.items():
        path = root / "reproduce" / "configs" / name
        config = json.loads(path.read_text())
        parsed = diffwave_cli.parse_args(["--config", str(path)])
        assert config["data_root"] == "data/sc09_v0.02"
        assert config["max_samples"] == 100
        assert config["subset_seed"] == 0
        assert config["num_timestep_levels"] == 200
        assert config["samples_per_timestep"] == 4
        assert config["batch_size"] == 2
        assert config["num_bins"] == 20
        assert config["grouping"] == "per_filter"
        assert config["max_groups"] == max_groups
        assert config["max_groups_mode"] == "stratified"
        assert config["group_correlation"] == "never"
        assert config["preload_audio"] is True
        assert config["cache_fixed_corruptions"] is True
        assert config["require_positive_delta_mass"] is positive_gate
        assert config["run_postprocessing"] is postprocess
        assert config["checkpoint_interval_groups"] == interval
        assert parsed.max_samples == 100
        assert parsed.samples_per_timestep == 4
        assert parsed.grouping == "per_filter"

    research = json.loads(
        (root / "reproduce" / "configs" / "diffwave_sc09_analysis_per_filter_research.json").read_text()
    )
    assert research["score_reduction"] == "mean"
    assert research["allocation_variants"] == [
        "combined_blockwise",
        "combined_layerwise",
    ]


def test_per_filter_postprocessing_runs_scoped_robust_geometric_workflow(
    tmp_path, monkeypatch
):
    commands = []

    def fake_run(command, *, cwd, check):
        assert cwd == diffwave_cli.REPO_ROOT
        assert check is True
        commands.append(command)

    monkeypatch.setattr(diffwave_cli.subprocess, "run", fake_run)
    returned = diffwave_cli.run_postprocessing(
        tmp_path / "profile",
        grouping="per_filter",
        num_timestep_groups=3,
        score_reduction="mean",
        allocation_variants=["combined_blockwise", "combined_layerwise"],
    )

    assert returned == commands
    assert len(commands) == 10
    assert commands[0][1].endswith("postprocess_diffwave_residual.py")
    assert commands[-1][1].endswith("report_diffwave_per_filter_decision.py")
    grouping_commands = commands[1:5]
    assert "matrix_spearman_cross_penalty" in grouping_commands[0]
    assert "--select_num_blocks" in grouping_commands[0]
    assert "matrix_correlation_cross_penalty" in grouping_commands[1]
    assert "--select_num_blocks" in grouping_commands[1]
    assert "matrix_correlation_cross_penalty" in grouping_commands[2]
    assert grouping_commands[2][grouping_commands[2].index("--cross_block_lambda") + 1] == "0"
    assert grouping_commands[2][grouping_commands[2].index("--num_blocks") + 1] == "3"
    assert grouping_commands[3][grouping_commands[3].index("--num_blocks") + 1] == "3"
    for command in commands[5:9]:
        assert command[command.index("--allocation-metric") + 1] == "delta_p_eff_geomean"
        assert command[command.index("--allocation-group-scope") + 1] == "allocatable"
        assert command[command.index("--score-reduction") + 1] == "mean"
        assert command.count("--student-variant") == 2


def test_parser_requires_teacher_verification_report(tmp_path):
    with pytest.raises(SystemExit):
        diffwave_cli.parse_args(
            [
                "--data-root",
                str(tmp_path / "sc09"),
                "--model-cache-dir",
                str(tmp_path / "models"),
                "--output-dir",
                str(tmp_path / "analysis"),
            ]
        )


def test_parser_rejects_partial_per_filter_allocation(tmp_path):
    common = [
        "--data-root",
        str(tmp_path / "sc09"),
        "--model-cache-dir",
        str(tmp_path / "models"),
        "--output-dir",
        str(tmp_path / "output"),
        "--teacher-verification-report",
        str(tmp_path / "verification.json"),
        "--grouping",
        "per_filter",
        "--group-correlation",
        "never",
        "--max-groups",
        "16",
    ]
    with pytest.raises(SystemExit):
        diffwave_cli.parse_args(common)
    parsed = diffwave_cli.parse_args([*common, "--no-run-postprocessing"])
    assert parsed.max_groups == 16


def test_parser_rejects_legacy_per_filter_postprocessing_settings(tmp_path):
    common = [
        "--data-root",
        str(tmp_path / "sc09"),
        "--model-cache-dir",
        str(tmp_path / "models"),
        "--output-dir",
        str(tmp_path / "output"),
        "--teacher-verification-report",
        str(tmp_path / "verification.json"),
        "--grouping",
        "per_filter",
        "--group-correlation",
        "never",
        "--require-full-filter-catalog",
    ]
    with pytest.raises(SystemExit):
        diffwave_cli.parse_args([*common, "--score-reduction", "q90"])
    with pytest.raises(SystemExit):
        diffwave_cli.parse_args(
            [
                *common,
                "--score-reduction",
                "mean",
                "--allocation-variants",
                "blockwise_capacity",
                "layerwise_capacity",
            ]
        )


def test_runtime_cache_identity_records_stable_execution_fields():
    identity = diffwave_cli.runtime_cache_identity(torch.device("cpu"))
    assert set(identity) == {
        "python",
        "torch",
        "torchaudio",
        "torchcodec",
        "cuda_runtime",
        "cudnn",
        "device_type",
        "device_name",
    }
    assert identity["device_type"] == "cpu"
    assert identity["device_name"] == "cpu"


def test_tiny_main_writes_comparable_results_and_plots(tmp_path, monkeypatch):
    model = LegacyCompatibleDiffWave(
        res_channels=2,
        skip_channels=2,
        num_res_layers=36,
        dilation_cycle=12,
        diffusion_step_embed_dim_in=4,
        diffusion_step_embed_dim_mid=8,
        diffusion_step_embed_dim_out=8,
        num_diffusion_steps=200,
    ).eval()
    spec = resolve_teacher_spec(
        "https://example.invalid/teacher.pkl",
        preset="sc09_diffwave_legacy_1m",
    )
    model.teacher_spec = spec.to_dict()
    monkeypatch.setattr(diffwave_cli, "load_teacher_network", lambda *args, **kwargs: model)
    monkeypatch.setattr(
        diffwave_cli,
        "SC09Dataset",
        lambda *args, **kwargs: _TinyAudioDataset(size=2, length=8),
    )
    monkeypatch.setattr(
        diffwave_cli,
        "teacher_model_metadata",
        lambda network, teacher_spec: {"model_family": "diffwave", "model_config": teacher_spec.model_config},
    )
    monkeypatch.setattr(
        diffwave_cli,
        "validate_teacher_verification_report",
        lambda path: {
            "path": str(path),
            "sha256": "d" * 64,
            "format": diffwave_cli.VERIFICATION_FORMAT,
            "passed": True,
        },
    )
    output = tmp_path / "analysis"

    results = diffwave_cli.main(
        [
            "--data-root",
            str(tmp_path / "sc09"),
            "--model-cache-dir",
            str(tmp_path / "models"),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
            "--teacher-verification-report",
            str(tmp_path / "verification.json"),
            "--no-run-postprocessing",
            "--no-require-positive-delta-mass",
            "--num-timestep-levels",
            "2",
            "--num-bins",
            "2",
            "--max-groups",
            "2",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
        ]
    )

    assert results["group_names"] == ["residual_block_00", "residual_block_01"]
    assert results["timestep_bin_labels"] == ["199", "0"]
    assert results["profile_fingerprint"]["runtime"]["identity"]["device_type"] == "cpu"
    assert len(results["n_eff"]) == 2
    assert (output / "results.json").is_file()
    assert (output / "baseline_pfi.pt").is_file()
    assert (output / "checkpoint_pfi_rank0.pt").is_file()
    success = json.loads((output / "_SUCCESS.json").read_text())
    assert success["artifact_count"] == len(success["artifact_sha256"])
    assert {
        "results.json",
        "metrics.pt",
        "pfi_plan.pt",
        "baseline_pfi.pt",
        "checkpoint_pfi_rank0.pt",
    }.issubset(success["artifact_sha256"])
    for artifact in [
        "baseline_error.png",
        "delta_heatmap.png",
        "relative_delta_heatmap.png",
        "effective_group_count.png",
        "timestep_correlation_heatmap.png",
    ]:
        assert (output / artifact).is_file()
