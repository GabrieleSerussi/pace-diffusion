import json
import pickle
import threading
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pace.edm_distillation import (
    NarrowEDMPrecond,
    NarrowVPPrecond,
    construct_student_from_kwargs,
    default_model_kwargs,
    infer_img_resolution,
    infer_label_dim,
    infer_student_topology,
)
from pace.teacher_models import (
    NVLABS_EDM_PICKLE,
    OPENAI_CONSISTENCY_STATE_DICT,
    OpenAIConsistencyEDMPrecond,
    TeacherSpec,
    load_teacher_network,
    materialize_teacher_source,
    resolve_teacher_spec,
    teacher_model_metadata,
    teacher_preset_config,
    FFHQ_64_VP_SOURCE,
)
from pace.vendor.openai_consistency_unet import QKVFlashAttention, create_unet


def tiny_openai_config() -> dict:
    return {
        "architecture": "openai_consistency_unet",
        "model_family": "edm",
        "image_size": 8,
        "in_channels": 3,
        "out_channels": 3,
        "label_dim": 0,
        "model_channels": 32,
        "num_res_blocks": 1,
        "channel_mult": [1],
        "attention_resolutions": [8],
        "num_heads": 2,
        "num_head_channels": 16,
        "resblock_updown": True,
        "use_scale_shift_norm": False,
        "dropout": 0.0,
        "sigma_data": 0.5,
        "sigma_min": 0.002,
        "sigma_max": 80.0,
        "time_scale": 250.0,
    }


def test_teacher_presets_are_explicit_and_ffhq_follows_real_wrapper_contract():
    bedroom = teacher_preset_config("lsun_bedroom_256")
    assert bedroom["format"] == OPENAI_CONSISTENCY_STATE_DICT
    assert bedroom["expected_size_bytes"] == 2_105_395_349
    assert bedroom["expected_sha256"] == "5947bb5ae7b664feef1796ebe0a531efcf689d76e81d03ba88159dd8019b674f"
    assert bedroom["architecture_metadata"] == {
        "parameter_count": 526_304_771,
        "state_dict_key_count": 566,
    }
    assert bedroom["model_config"]["channel_mult"] == [1, 1, 2, 2, 4, 4]
    assert bedroom["sampling"]["global_seed"] == 42

    ffhq = teacher_preset_config("ffhq_64_vp")
    assert ffhq["format"] == NVLABS_EDM_PICKLE
    assert ffhq["expected_sha256"] == "f6f8f24a2b46ae79807b0f919e2550c6ed37cc3f8b7a2100629092cad9b5d2f5"
    assert ffhq["model_config"]["model_family"] == "edm"
    assert ffhq["model_config"]["architecture_lineage"] == "vp_ddpmpp"
    assert ffhq["model_config"]["channel_mult"] == [1, 2, 2, 2]
    assert ffhq["model_config"]["augment_dim"] == 9


def test_full_bedroom_preset_has_exact_published_architecture_on_meta_device():
    config = teacher_preset_config("lsun_bedroom_256")["model_config"]
    with torch.device("meta"):
        model = create_unet(config)
    state = model.state_dict()
    assert sum(parameter.numel() for parameter in model.parameters()) == 526_304_771
    assert len(state) == 566
    assert state["input_blocks.0.0.weight"].shape == (256, 3, 3, 3)
    assert state["input_blocks.1.0.in_layers.2.weight"].shape == (256, 256, 3, 3)
    assert state["middle_block.1.qkv.weight"].shape == (3072, 1024, 1, 1)
    assert state["output_blocks.17.0.out_layers.3.weight"].shape == (256, 256, 3, 3)
    assert state["out.2.weight"].shape == (3, 256, 3, 3)


def test_generic_pt_requires_explicit_format_and_architecture(tmp_path):
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.touch()
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_teacher_spec(checkpoint)
    with pytest.raises(ValueError, match="missing model_config"):
        resolve_teacher_spec(checkpoint, network_format=OPENAI_CONSISTENCY_STATE_DICT)


def test_materializer_reuses_canonical_cached_basename_without_download(tmp_path, monkeypatch):
    cached = tmp_path / "edm_bedroom256_ema.pt"
    cached.write_bytes(b"cached")
    spec = TeacherSpec(
        source="https://example.invalid/edm_bedroom256_ema.pt",
        format=OPENAI_CONSISTENCY_STATE_DICT,
        model_config=tiny_openai_config(),
        expected_size_bytes=len(b"cached"),
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *_args, **_kwargs: pytest.fail("canonical cache hit attempted a download"),
    )
    assert materialize_teacher_source(spec, cache_dir=tmp_path) == cached.resolve()


def test_materializer_concurrent_callers_publish_one_complete_download(tmp_path, monkeypatch):
    payload = b"checkpoint-bytes" * 1024
    spec = TeacherSpec(
        source="https://example.invalid/concurrent.pt",
        format=OPENAI_CONSISTENCY_STATE_DICT,
        model_config=tiny_openai_config(),
        expected_size_bytes=len(payload),
    )
    calls = 0
    calls_lock = threading.Lock()

    class Response:
        def __init__(self):
            self.offset = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, size):
            nonlocal calls
            with calls_lock:
                calls += 1
            chunk = payload[self.offset : self.offset + size]
            self.offset += len(chunk)
            return chunk

    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: Response())
    results: list[Path] = []
    failures: list[BaseException] = []

    def run():
        try:
            results.append(materialize_teacher_source(spec, cache_dir=tmp_path))
        except BaseException as exc:  # pragma: no cover - assertion reports thread failures
            failures.append(exc)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not failures
    assert len(set(results)) == 1
    assert results[0].read_bytes() == payload
    # One content read plus one EOF read proves the URL was opened by only one
    # caller; all other callers observed the atomically published file.
    assert calls == 2
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_local_pickle_requires_explicit_trust(tmp_path):
    checkpoint = tmp_path / "teacher.pkl"
    with checkpoint.open("wb") as handle:
        pickle.dump({"ema": torch.nn.Linear(2, 2)}, handle)
    spec = TeacherSpec(source=str(checkpoint), format=NVLABS_EDM_PICKLE)

    with pytest.raises(ValueError, match="Refusing to unpickle"):
        load_teacher_network(spec, device=torch.device("cpu"))

    loaded = load_teacher_network(
        spec,
        device=torch.device("cpu"),
        trust_local_pickle=True,
    )
    assert isinstance(loaded, torch.nn.Linear)


def test_builtin_ffhq_url_is_trusted_when_reused_from_cache(tmp_path):
    checkpoint = tmp_path / "edm-ffhq-64x64-uncond-vp.pkl"
    with checkpoint.open("wb") as handle:
        pickle.dump({"ema": torch.nn.Linear(2, 2)}, handle)
    spec = TeacherSpec(source=FFHQ_64_VP_SOURCE, format=NVLABS_EDM_PICKLE)

    loaded = load_teacher_network(
        spec,
        device=torch.device("cpu"),
        cache_dir=tmp_path,
    )
    assert isinstance(loaded, torch.nn.Linear)


def test_portable_attention_matches_explicit_reference():
    module = QKVFlashAttention(embed_dim=8, num_heads=2)
    qkv = torch.randn(2, 24, 5)
    actual = module(qkv)
    reshaped = qkv.reshape(2, 3, 2, 4, 5)
    query, key, value = reshaped.unbind(dim=1)
    scale = 1 / (4**0.5)
    weights = torch.softmax(torch.einsum("bhct,bhcs->bhts", query, key) * scale, dim=-1)
    expected = torch.einsum("bhts,bhcs->bhct", weights, value).reshape(2, 8, 5)
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_tiny_openai_state_dict_loads_strictly_and_preconditions(tmp_path):
    config = tiny_openai_config()
    raw = create_unet(config)
    state = raw.state_dict()
    assert state["input_blocks.0.0.weight"].shape == (32, 3, 3, 3)
    assert state["middle_block.1.qkv.weight"].shape == (96, 32, 1, 1)
    assert state["middle_block.1.proj_out.weight"].shape == (32, 32, 1, 1)
    assert state["out.2.weight"].shape == (3, 32, 3, 3)
    checkpoint = tmp_path / "tiny.pt"
    torch.save(state, checkpoint)
    spec = TeacherSpec(
        source=str(checkpoint),
        format=OPENAI_CONSISTENCY_STATE_DICT,
        model_config=config,
    )
    loaded = load_teacher_network(spec, device=torch.device("cpu"), dtype=torch.float32)
    assert isinstance(loaded, OpenAIConsistencyEDMPrecond)
    resolved = TeacherSpec.from_dict(loaded.teacher_spec)
    assert resolved.checkpoint_size_bytes == checkpoint.stat().st_size
    assert len(resolved.checkpoint_sha256 or "") == 64

    x = torch.randn(2, 3, 8, 8)
    sigma = torch.tensor([0.5, 1.0])
    with torch.no_grad():
        actual = loaded(x, sigma)
        sigma_4d = sigma.reshape(-1, 1, 1, 1)
        c_skip = 0.25 / (sigma_4d.square() + 0.25)
        c_out = sigma_4d * 0.5 / (sigma_4d.square() + 0.25).sqrt()
        c_in = 1 / (sigma_4d.square() + 0.25).sqrt()
        expected = c_skip * x + c_out * raw(
            c_in * x,
            250 * torch.log(sigma),
        )
    assert torch.allclose(actual, expected)
    assert loaded.structural_key_for_module("model.input_blocks.0.0") == "enc.8x8_conv"


def test_openai_structural_mapping_covers_every_parameterized_module():
    config = tiny_openai_config()
    spec = TeacherSpec(
        source="tiny.pt",
        format=OPENAI_CONSISTENCY_STATE_DICT,
        model_config=config,
    )
    wrapper = OpenAIConsistencyEDMPrecond(create_unet(config), config, teacher_spec=spec)
    valid_prefixes = ("enc.", "dec.", "out_conv")
    for name, module in wrapper.model.named_modules():
        if not name or not any(parameter.numel() for parameter in module.parameters(recurse=False)):
            continue
        canonical = wrapper.structural_key_for_module(f"model.{name}")
        # Time-embedding layers are global conditioning rather than spatial
        # structural widths; all spatial modules must map to Narrow keys.
        if name.startswith("time_embed"):
            continue
        assert canonical.startswith(valid_prefixes), (name, canonical)


def test_openai_loader_rejects_non_strict_state_dict(tmp_path):
    config = tiny_openai_config()
    raw = create_unet(config)
    state = raw.state_dict()
    state.pop(next(iter(state)))
    checkpoint = tmp_path / "bad.pt"
    torch.save(state, checkpoint)
    spec = TeacherSpec(
        source=str(checkpoint),
        format=OPENAI_CONSISTENCY_STATE_DICT,
        model_config=config,
    )
    with pytest.raises(RuntimeError, match="Missing key"):
        load_teacher_network(spec, device=torch.device("cpu"))


@pytest.mark.external_edm
def test_narrow_vp_preconditioning_and_topology_round_trip():
    kwargs = default_model_kwargs(
        model_channels=8,
        label_dim=0,
        img_resolution=8,
        channel_mult=[1],
        num_blocks=1,
        attn_resolutions=[8],
        dropout=0.0,
        model_family="vp",
        preconditioning={"beta_d": 19.9, "beta_min": 0.1, "M": 1000, "epsilon_t": 1e-5},
    )
    model = construct_student_from_kwargs(kwargs)
    assert isinstance(model, NarrowVPPrecond)
    x = torch.randn(1, 3, 8, 8)
    sigma = torch.tensor([0.5])
    with torch.no_grad():
        prediction = model.model(
            x / (sigma.reshape(-1, 1, 1, 1).square() + 1).sqrt(),
            ((model.M - 1) * model.sigma_inv(sigma)).flatten(),
            class_labels=None,
        )
        expected = x - sigma.reshape(-1, 1, 1, 1) * prediction
        actual = model(x, sigma)
    assert torch.allclose(actual, expected)

    results = {
        "model_info": {
            "label_dim": 0,
            "img_resolution": 256,
            "model_family": "edm",
            "model_config": teacher_preset_config("lsun_bedroom_256")["model_config"],
        }
    }
    assert infer_label_dim(results) == 0
    assert infer_img_resolution(results) == 256
    assert infer_student_topology(results)["channel_mult"] == [1, 1, 2, 2, 4, 4]


@pytest.mark.external_edm
def test_actual_wrapper_class_overrides_filename_lineage_metadata():
    model = NarrowEDMPrecond(
        img_resolution=8,
        img_channels=3,
        label_dim=0,
        model_channels=8,
        channel_mult=[1],
        dropout=0.0,
    )
    preset = teacher_preset_config("ffhq_64_vp")
    spec = resolve_teacher_spec(
        "edm-ffhq-64x64-uncond-vp.pkl",
        preset="ffhq_64_vp",
    )
    metadata = teacher_model_metadata(model, spec)
    assert metadata["model_family"] == "edm"
    assert metadata["model_config"]["architecture_lineage"] == "vp_ddpmpp"
    assert preset["model_config"]["student_topology"]["augment_dim"] == 9
    assert metadata["sigma_max"] == "inf"
    json.dumps(metadata, allow_nan=False)
