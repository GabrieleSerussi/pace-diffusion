import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import evaluate_parameters_diffwave as cli
from pace.parameter_analysis import IndexedOutputTarget
from pace.teacher_models import teacher_preset_config
from pace.vendor.diffwave_legacy import (
    LegacyCompatibleDiffWave,
    create_legacy_compatible_diffwave,
    diffwave_diffusion_hyperparameters,
)


class CountingAudio(torch.utils.data.Dataset):
    def __init__(self, size: int = 4, length: int = 16):
        self.calls = 0
        self.items = [
            (torch.full((1, length), index / 10.0, dtype=torch.float32), index)
            for index in range(size)
        ]
        self.metadata = {"selected_count": size}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        self.calls += 1
        return self.items[index]


def tiny_model() -> LegacyCompatibleDiffWave:
    return LegacyCompatibleDiffWave(
        res_channels=4,
        skip_channels=4,
        num_res_layers=2,
        dilation_cycle=2,
        diffusion_step_embed_dim_in=4,
        diffusion_step_embed_dim_mid=8,
        diffusion_step_embed_dim_out=8,
        num_diffusion_steps=200,
    ).eval()


def test_preloaded_audio_reads_each_selected_waveform_once():
    source = CountingAudio()
    cached = cli.PreloadedAudioDataset(source)
    assert source.calls == len(source)
    first = cached[0][0]
    second = cached[0][0]
    assert source.calls == len(source)
    assert torch.equal(first, second)
    assert cached.metadata["preloaded_count"] == len(source)


def test_pinned_teacher_per_filter_catalog_has_exact_counts_and_order():
    model = create_legacy_compatible_diffwave(
        teacher_preset_config("sc09_diffwave_legacy_1m")["model_config"]
    )
    catalog = cli.collect_diffwave_group_catalog(model, grouping="per_filter")

    assert len(catalog.groups) == cli.PER_FILTER_EXPECTED_GROUPS
    assert len(set(catalog.module_paths.values())) == cli.PER_FILTER_EXPECTED_CONV_MODULES
    assert sum(catalog.parameter_counts.values()) == cli.PER_FILTER_EXPECTED_TRUE_PARAMETERS
    assert (
        sum(catalog.edm_proxy_parameter_counts.values())
        == cli.PER_FILTER_EXPECTED_EDM_PROXY_PARAMETERS
    )
    assert next(iter(catalog.groups)) == "init_conv.0.conv.filter_0"
    assert next(reversed(catalog.groups)) == "final_conv.2.conv.filter_0"
    assert all(isinstance(target, IndexedOutputTarget) for target in catalog.groups.values())
    residual_parameters = sum(
        catalog.parameter_counts[name]
        for name in catalog.groups
        if catalog.allocatable[name]
    )
    assert residual_parameters == cli.PER_FILTER_EXPECTED_RESIDUAL_PARAMETERS
    assert sum(parameter.numel() for parameter in model.parameters()) - sum(
        catalog.parameter_counts.values()
    ) == cli.PER_FILTER_EXPECTED_UNASSIGNED_PARAMETERS


def test_stratified_256_filter_selection_covers_every_conv_module():
    model = create_legacy_compatible_diffwave(
        teacher_preset_config("sc09_diffwave_legacy_1m")["model_config"]
    )
    full = cli.collect_diffwave_group_catalog(model, grouping="per_filter")
    selected = cli.select_group_catalog(
        full,
        max_groups=256,
        mode="stratified",
        seed=0,
    )
    assert len(selected.groups) == 256
    assert set(selected.module_paths.values()) == set(full.module_paths.values())


def test_fixed_corruption_cache_matches_dynamic_path_exactly():
    model = tiny_model()
    schedule = diffwave_diffusion_hyperparameters(num_diffusion_steps=200)
    _, axis = cli.make_timestep_axis_metadata(
        evaluated_timesteps=[199, 0],
        num_diffusion_steps=200,
        num_bins=2,
        alpha_bar=schedule["Alpha_bar"],
    )
    groups = cli.collect_diffwave_group_catalog(model, grouping="residual_blocks").groups
    dynamic = cli.DiffWaveUsageEvaluator(
        model,
        schedule,
        device=torch.device("cpu"),
        dtype=torch.float32,
        axis=axis,
        groups=groups,
        cache_fixed_corruptions=False,
    )
    cached = cli.DiffWaveUsageEvaluator(
        model,
        schedule,
        device=torch.device("cpu"),
        dtype=torch.float32,
        axis=axis,
        groups=groups,
        cache_fixed_corruptions=True,
    )
    waveforms = torch.stack(
        [torch.linspace(-0.2, 0.2, 16), torch.linspace(0.2, -0.2, 16)]
    ).unsqueeze(1)
    timesteps = torch.tensor([199, 199])
    noise_seeds = torch.tensor([13, 17])

    expected = dynamic.forward_losses_from_fixed_corruption(
        waveforms, timesteps, noise_seeds
    )
    first = cached.forward_losses_from_fixed_corruption(waveforms, timesteps, noise_seeds)
    second = cached.forward_losses_from_fixed_corruption(waveforms, timesteps, noise_seeds)
    assert torch.equal(first, expected)
    assert torch.equal(second, expected)
    record = cached.fixed_corruption_cache_record()
    assert record["entries"] == 2
    assert record["tensor_bytes"] == 2 * 2 * 16 * 4


def test_filter_aggregation_sums_individual_estimands_by_stage():
    names = ["a.filter_0", "a.filter_1", "b.filter_0"]
    signed = torch.tensor([[1.0, -1.0], [2.0, 3.0], [-4.0, 5.0]])
    positive = signed.clamp_min(0)
    aggregate = cli.aggregate_filter_matrices(
        group_names=names,
        keys={names[0]: "stage_a", names[1]: "stage_a", names[2]: "stage_b"},
        signed_delta_stack=signed,
        delta_stack=positive,
        baseline_mean=torch.tensor([2.0, 4.0]),
        parameter_counts={names[0]: 2, names[1]: 3, names[2]: 5},
    )
    assert aggregate["names"] == ["stage_a", "stage_b"]
    assert torch.equal(aggregate["signed_delta_stack"], torch.tensor([[3.0, 2.0], [-4.0, 5.0]], dtype=torch.float64))
    assert torch.equal(aggregate["delta_stack"], torch.tensor([[3.0, 3.0], [0.0, 5.0]], dtype=torch.float64))
    assert aggregate["parameter_counts"] == {"stage_a": 5, "stage_b": 5}
