import math
import types

import pytest

torch = pytest.importorskip("torch")

from pace.teacher_models import (
    DIFFWAVE_SASHIMI_LEGACY_STATE_DICT,
    TeacherSpec,
    load_teacher_network,
    resolve_teacher_spec,
    teacher_model_metadata,
    teacher_preset_config,
)
from pace.vendor.diffwave_legacy import (
    DIFFWAVE_SASHIMI_CHECKPOINT_SHA256,
    DIFFWAVE_SASHIMI_CHECKPOINT_SIZE_BYTES,
    DIFFWAVE_SASHIMI_CHECKPOINTS_COMMIT,
    DIFFWAVE_SASHIMI_LEGACY_SEMANTICS,
    LegacyCompatibleDiffWave,
    LegacyCompatibleResidualBlock,
    calc_diffusion_step_embedding,
    diffwave_diffusion_hyperparameters,
)


def tiny_config() -> dict:
    return {
        "architecture": "diffwave_wavenet_legacy_safe",
        "model_family": "diffwave",
        "input_representation": "raw_waveform",
        "in_channels": 1,
        "out_channels": 1,
        "res_channels": 4,
        "skip_channels": 4,
        "num_res_layers": 2,
        "dilation_cycle": 2,
        "diffusion_step_embed_dim_in": 8,
        "diffusion_step_embed_dim_mid": 16,
        "diffusion_step_embed_dim_out": 16,
        "unconditional": True,
        "num_diffusion_steps": 8,
        "beta_0": 0.0001,
        "beta_T": 0.02,
        "sample_rate": 16_000,
        "example_length": 17,
        "legacy_forward_semantics": DIFFWAVE_SASHIMI_LEGACY_SEMANTICS,
    }


def test_sc09_diffwave_preset_pins_checkpoint_architecture_and_provenance():
    preset = teacher_preset_config("sc09_diffwave_legacy_1m")
    assert preset["format"] == DIFFWAVE_SASHIMI_LEGACY_STATE_DICT
    assert preset["expected_sha256"] == DIFFWAVE_SASHIMI_CHECKPOINT_SHA256
    assert preset["expected_size_bytes"] == DIFFWAVE_SASHIMI_CHECKPOINT_SIZE_BYTES
    assert preset["architecture_metadata"] == {
        "parameter_count": 24_071_681,
        "state_dict_key_count": 408,
    }
    config = preset["model_config"]
    assert config["upstream_commit"] == DIFFWAVE_SASHIMI_CHECKPOINTS_COMMIT
    assert config["legacy_forward_semantics"] == DIFFWAVE_SASHIMI_LEGACY_SEMANTICS
    assert config["weight_norm_state"] == "legacy_weight_g_weight_v"
    assert config["checkpoint_container_key"] == "model_state_dict"
    assert config["res_channels"] == config["skip_channels"] == 256
    assert config["num_res_layers"] == 36
    assert config["dilation_cycle"] == 12
    assert preset["sampling"] == {
        "sampler": "diffwave_ancestral",
        "num_steps": 200,
        "beta_0": 0.0001,
        "beta_T": 0.02,
        "prediction_type": "epsilon",
    }

    resolved = resolve_teacher_spec(
        "checkpoint.pkl",
        preset="diffwave_sc09_1m",
    )
    assert resolved.preset == "sc09_diffwave_legacy_1m"
    assert resolved.format == DIFFWAVE_SASHIMI_LEGACY_STATE_DICT


def test_full_model_has_published_parameter_shapes_and_legacy_weight_norm_keys():
    with torch.device("meta"):
        model = LegacyCompatibleDiffWave()
    state = model.state_dict()
    assert sum(parameter.numel() for parameter in model.parameters()) == 24_071_681
    assert len(state) == 408
    assert state["init_conv.0.conv.weight_g"].shape == (256, 1, 1)
    assert state["init_conv.0.conv.weight_v"].shape == (256, 1, 1)
    assert state[
        "residual_layer.residual_blocks.35.dilated_conv_layer.conv.weight_v"
    ].shape == (512, 256, 3)
    assert state[
        "residual_layer.residual_blocks.35.res_conv.weight_g"
    ].shape == (256, 1, 1)
    assert state["final_conv.2.conv.weight"].shape == (1, 256, 1)
    assert [name for name, _ in model.named_residual_blocks()] == [
        f"residual_layer.residual_blocks.{index}" for index in range(36)
    ]


def test_residual_block_preserves_legacy_math_without_mutating_input_and_backpropagates():
    torch.manual_seed(11)
    block = LegacyCompatibleResidualBlock(
        4,
        4,
        dilation=2,
        diffusion_step_embed_dim_out=16,
    )
    x = torch.randn(2, 4, 17, requires_grad=True)
    embedding = torch.randn(2, 16)
    original = x.detach().clone()

    part_t = block.fc_t(embedding).view(2, 4, 1)
    residual_base = x + part_t
    h = block.dilated_conv_layer(residual_base)
    gated = torch.tanh(h[:, :4]) * torch.sigmoid(h[:, 4:])
    expected_residual = (residual_base + block.res_conv(gated)) * math.sqrt(0.5)
    expected_skip = block.skip_conv(gated)

    actual_residual, actual_skip = block((x, embedding))
    assert torch.equal(actual_residual, expected_residual)
    assert torch.equal(actual_skip, expected_skip)
    assert torch.equal(x.detach(), original)

    (actual_residual.square().mean() + actual_skip.square().mean()).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().max() > 0


def _legacy_inplace_block_forward(self, input_data, mel_spec=None):
    assert mel_spec is None
    x, diffusion_step_embed = input_data
    h = x
    part_t = self.fc_t(diffusion_step_embed).view(x.shape[0], self.res_channels, 1)
    h += part_t
    h = self.dilated_conv_layer(h)
    gated = torch.tanh(h[:, : self.res_channels]) * torch.sigmoid(
        h[:, self.res_channels :]
    )
    residual = self.res_conv(gated)
    skip = self.skip_conv(gated)
    return (x + residual) * math.sqrt(0.5), skip


def _corrected_master_block_forward(self, input_data, mel_spec=None):
    assert mel_spec is None
    x, diffusion_step_embed = input_data
    part_t = self.fc_t(diffusion_step_embed).view(x.shape[0], self.res_channels, 1)
    h = self.dilated_conv_layer(x + part_t)
    gated = torch.tanh(h[:, : self.res_channels]) * torch.sigmoid(
        h[:, self.res_channels :]
    )
    residual = self.res_conv(gated)
    skip = self.skip_conv(gated)
    return (x + residual) * math.sqrt(0.5), skip


def _tiny_model_with_nonzero_output() -> LegacyCompatibleDiffWave:
    model = LegacyCompatibleDiffWave(**{
        key: value
        for key, value in tiny_config().items()
        if key
        not in {
            "architecture",
            "model_family",
            "input_representation",
            "legacy_forward_semantics",
        }
    })
    with torch.no_grad():
        model.final_conv[2].conv.weight.normal_(mean=0.0, std=0.1)
    return model


def test_safe_model_matches_inplace_legacy_and_excludes_corrected_master_semantics():
    torch.manual_seed(19)
    safe = _tiny_model_with_nonzero_output().eval()
    legacy = _tiny_model_with_nonzero_output().eval()
    corrected = _tiny_model_with_nonzero_output().eval()
    legacy.load_state_dict(safe.state_dict(), strict=True)
    corrected.load_state_dict(safe.state_dict(), strict=True)
    for block in legacy.residual_blocks:
        block.forward = types.MethodType(_legacy_inplace_block_forward, block)
    for block in corrected.residual_blocks:
        block.forward = types.MethodType(_corrected_master_block_forward, block)

    x = torch.randn(2, 1, 31)
    corrected_differs = False
    with torch.no_grad():
        for timestep in (0, 1, 4, 7):
            t = torch.full((2, 1), float(timestep))
            expected = legacy((x.clone(), t.clone()))
            actual = safe(x.clone(), t.clone())
            corrected_output = corrected(x.clone(), t.clone())
            assert torch.equal(actual, expected)
            corrected_differs |= not torch.equal(corrected_output, expected)
    assert corrected_differs


def test_timestep_embedding_and_schedule_preserve_upstream_operation_order():
    timesteps = torch.tensor([[0.0], [1.0], [50.0], [199.0]])
    half_dim = 64
    scale = torch.as_tensor(float(torch.log(torch.tensor(10_000.0)))) / (half_dim - 1)
    frequencies = torch.exp(torch.arange(half_dim) * -scale)
    arguments = timesteps * frequencies
    expected = torch.cat((torch.sin(arguments), torch.cos(arguments)), dim=1)
    actual = calc_diffusion_step_embedding(timesteps, 128)
    assert torch.equal(actual, expected)

    schedule = diffwave_diffusion_hyperparameters(
        num_diffusion_steps=8,
        beta_0=0.0001,
        beta_T=0.02,
    )
    assert schedule["T"] == 8
    assert torch.is_tensor(schedule["Beta"])
    assert schedule["Beta"][0] == pytest.approx(0.0001)
    assert schedule["Beta"][-1] == pytest.approx(0.02)
    expected_alpha_bar = schedule["Alpha"] + 0
    for index in range(1, 8):
        expected_alpha_bar[index] *= expected_alpha_bar[index - 1]
    assert torch.equal(schedule["Alpha_bar"], expected_alpha_bar)


def test_diffwave_checkpoint_loader_requires_model_state_dict_and_loads_strictly(tmp_path):
    torch.manual_seed(23)
    raw = _tiny_model_with_nonzero_output()
    checkpoint = tmp_path / "tiny.pkl"
    torch.save({"model_state_dict": raw.state_dict()}, checkpoint)
    spec = TeacherSpec(
        source=str(checkpoint),
        format=DIFFWAVE_SASHIMI_LEGACY_STATE_DICT,
        model_config=tiny_config(),
    )
    loaded = load_teacher_network(spec, device=torch.device("cpu"))
    assert isinstance(loaded, LegacyCompatibleDiffWave)
    assert loaded.checkpoint_format == DIFFWAVE_SASHIMI_LEGACY_STATE_DICT
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    for key, expected in raw.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], expected), key

    resolved = TeacherSpec.from_dict(loaded.teacher_spec)
    metadata = teacher_model_metadata(loaded, resolved)
    assert metadata["checkpoint_sha256"] == resolved.checkpoint_sha256
    assert metadata["checkpoint_size_bytes"] == checkpoint.stat().st_size
    assert metadata["model_family"] == "diffwave"
    assert metadata["sample_rate"] == 16_000
    assert metadata["example_length"] == 17
    assert metadata["prediction_type"] == "epsilon"

    ambiguous = tmp_path / "ambiguous.pkl"
    torch.save({"state_dict": raw.state_dict()}, ambiguous)
    with pytest.raises(ValueError, match=r"checkpoint\['model_state_dict'\]"):
        load_teacher_network(
            TeacherSpec(
                source=str(ambiguous),
                format=DIFFWAVE_SASHIMI_LEGACY_STATE_DICT,
                model_config=tiny_config(),
            ),
            device=torch.device("cpu"),
        )

    incomplete = dict(raw.state_dict())
    incomplete.pop(next(iter(incomplete)))
    bad = tmp_path / "bad.pkl"
    torch.save({"model_state_dict": incomplete}, bad)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_teacher_network(
            TeacherSpec(
                source=str(bad),
                format=DIFFWAVE_SASHIMI_LEGACY_STATE_DICT,
                model_config=tiny_config(),
            ),
            device=torch.device("cpu"),
        )


def test_diffwave_loader_rejects_non_fp32_and_corrected_model_identity(tmp_path):
    raw = _tiny_model_with_nonzero_output()
    checkpoint = tmp_path / "tiny.pkl"
    torch.save({"model_state_dict": raw.state_dict()}, checkpoint)
    config = tiny_config()
    spec = TeacherSpec(
        source=str(checkpoint),
        format=DIFFWAVE_SASHIMI_LEGACY_STATE_DICT,
        model_config=config,
    )
    with pytest.raises(ValueError, match="requires torch.float32"):
        load_teacher_network(spec, device=torch.device("cpu"), dtype=torch.float16)

    corrected_config = dict(config)
    corrected_config["architecture"] = "diffwave_wavenet_corrected_master"
    with pytest.raises(ValueError, match="only supports the autograd-safe legacy"):
        load_teacher_network(
            TeacherSpec(
                source=str(checkpoint),
                format=DIFFWAVE_SASHIMI_LEGACY_STATE_DICT,
                model_config=corrected_config,
            ),
            device=torch.device("cpu"),
        )
