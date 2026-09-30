import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
wavfile = pytest.importorskip("scipy.io.wavfile")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import verify_diffwave_teacher as verifier
from pace.vendor.diffwave_legacy import LegacyCompatibleDiffWave


@pytest.mark.parametrize(
    "required_flag",
    ["--audio-output", "--reference-audio", "--reference-python"],
)
def test_cli_requires_both_generated_and_untouched_reference_audio(
    monkeypatch,
    capsys,
    required_flag,
):
    arguments = [
        "verify_diffwave_teacher.py",
        "--upstream-repo",
        "/upstream",
        "--checkpoint",
        "/checkpoint.pkl",
        "--output",
        "/report.json",
        "--audio-output",
        "/generated.wav",
        "--reference-audio",
        "/untouched.wav",
        "--reference-python",
        "/reference/bin/python",
    ]
    flag_index = arguments.index(required_flag)
    del arguments[flag_index : flag_index + 2]
    monkeypatch.setattr(sys, "argv", arguments)

    with pytest.raises(SystemExit) as error:
        verifier.parse_args()

    assert error.value.code == 2
    assert required_flag in capsys.readouterr().err


def _write_reference_wav(path, samples, *, sample_rate=16_000):
    wavfile.write(path, sample_rate, samples)
    return path


def test_reference_audio_accepts_nonconstant_float32_mono_16khz_16000_samples(
    tmp_path,
):
    phase = np.linspace(0.0, 8.0 * np.pi, 16_000, endpoint=False)
    samples = (0.1 * np.sin(phase)).astype(np.float32)
    path = _write_reference_wav(tmp_path / "valid.wav", samples)

    metadata = verifier._validate_reference_audio(path)

    assert metadata["path"] == str(path.resolve())
    assert metadata["dtype"] == "float32"
    assert metadata["num_channels"] == 1
    assert metadata["sample_rate"] == 16_000
    assert metadata["num_samples"] == 16_000
    assert metadata["finite"] is True
    assert metadata["rms"] > 1e-6
    assert metadata["standard_deviation"] > 1e-7
    assert metadata["wav_size_bytes"] == path.stat().st_size
    assert len(metadata["wav_sha256"]) == 64


@pytest.mark.parametrize(
    ("samples", "sample_rate", "message"),
    [
        (np.ones(16_000, dtype=np.int16), 16_000, "float32 WAV samples"),
        (np.ones((16_000, 2), dtype=np.float32), 16_000, "must be mono"),
        (np.ones(15_999, dtype=np.float32), 16_000, "exactly 16,000 samples"),
        (np.ones(16_000, dtype=np.float32), 8_000, "at 16 kHz"),
        (np.zeros(16_000, dtype=np.float32), 16_000, "effectively silent"),
    ],
    ids=["pcm16", "stereo", "wrong-length", "wrong-rate", "silent"],
)
def test_reference_audio_rejects_incompatible_or_invalid_wavs(
    tmp_path,
    samples,
    sample_rate,
    message,
):
    path = _write_reference_wav(tmp_path / "invalid.wav", samples, sample_rate=sample_rate)

    with pytest.raises(ValueError, match=message):
        verifier._validate_reference_audio(path)


def _valid_gate_args(**overrides):
    values = {
        "upstream_repo": Path("/upstream"),
        "checkpoint": Path("/checkpoint.pkl"),
        "output": Path("/report.json"),
        "device": "cuda:0",
        "seed": 20260819,
        "batch_size": 1,
        "prediction_length": 16_000,
        "timesteps": [0, 1, 25, 50, 100, 150, 198, 199],
        "trajectory_steps": 10,
        "trajectory_length": 16_000,
        "atol": 1e-6,
        "rtol": 1e-6,
        "audio_output": Path("/generated.wav"),
        "reference_audio": Path("/untouched.wav"),
        "reference_python": Path("/reference/bin/python"),
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_reference_environment_records_interpreter_versions_and_full_freeze(
    monkeypatch,
):
    interpreter = Path(sys.executable).absolute()
    probe_payload = {
        "interpreter": str(interpreter),
        "resolved_interpreter": str(interpreter.resolve()),
        "python": "3.11.13",
        "platform": "Linux-test",
        "torch": "2.10.0+cu128",
        "torchaudio": "2.10.0+cu128",
        "cuda_available": True,
        "cuda_runtime": "12.8",
        "cudnn": 91002,
    }

    def fake_check_output(arguments, *, text):
        assert text is True
        if arguments[1:3] == ["-m", "pip"]:
            assert arguments[3] == "freeze"
            return "torch==2.10.0\ntorchaudio==2.10.0\n"
        assert arguments[1] == "-c"
        return __import__("json").dumps(probe_payload)

    monkeypatch.setattr(verifier.subprocess, "check_output", fake_check_output)
    environment = verifier._capture_reference_environment(interpreter)

    assert environment["interpreter"] == str(interpreter)
    assert environment["resolved_interpreter"] == str(interpreter.resolve())
    assert environment["torch"] == "2.10.0+cu128"
    assert environment["torchaudio"] == "2.10.0+cu128"
    assert environment["cuda_runtime"] == "12.8"
    assert environment["cudnn"] == 91002
    assert environment["pip_freeze"] == ["torch==2.10.0", "torchaudio==2.10.0"]
    assert len(environment["pip_freeze_sha256"]) == 64


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"batch_size": 2}, "reproduction gate requires batch=1"),
        ({"prediction_length": 15_999}, "reproduction gate requires batch=1"),
        (
            {"timesteps": [0, 1, 25, 50, 100, 150, 199, 198]},
            "reproduction gate requires batch=1",
        ),
        ({"trajectory_steps": 9}, "reproduction gate requires batch=1"),
        ({"trajectory_length": 15_999}, "reproduction gate requires batch=1"),
        ({"atol": 1.000001e-6}, "finite atol and rtol no larger than 1e-6"),
        ({"rtol": -1e-12}, "finite atol and rtol no larger than 1e-6"),
        ({"atol": float("nan")}, "finite atol and rtol no larger than 1e-6"),
    ],
    ids=[
        "batch-size",
        "prediction-length",
        "prediction-timesteps",
        "trajectory-steps",
        "trajectory-length",
        "loose-atol",
        "negative-rtol",
        "nonfinite-atol",
    ],
)
def test_exact_probe_and_tolerance_guards_fail_before_cuda_or_model_work(
    monkeypatch,
    override,
    message,
):
    reached = {"cuda": False, "model": False}

    def unexpected_cuda_query():
        reached["cuda"] = True
        raise AssertionError("CUDA must not be queried before validating the gate")

    def unexpected_model_load(*args, **kwargs):
        reached["model"] = True
        raise AssertionError("Models must not be loaded before validating the gate")

    monkeypatch.setattr(verifier, "parse_args", lambda: _valid_gate_args(**override))
    monkeypatch.setattr(torch.cuda, "is_available", unexpected_cuda_query)
    monkeypatch.setattr(verifier, "_load_upstream_wavenet_class", unexpected_model_load)
    monkeypatch.setattr(verifier, "load_teacher_network", unexpected_model_load)

    with pytest.raises(ValueError, match=message):
        verifier.main()

    assert reached == {"cuda": False, "model": False}


def test_autograd_check_uses_real_safe_model_without_mutation_and_has_gradient():
    torch.manual_seed(71)
    model = LegacyCompatibleDiffWave(
        res_channels=4,
        skip_channels=4,
        num_res_layers=2,
        dilation_cycle=2,
        diffusion_step_embed_dim_in=8,
        diffusion_step_embed_dim_mid=16,
        diffusion_step_embed_dim_out=16,
        example_length=33,
    ).eval()
    # The upstream architecture zero-initializes its output projection.  Give
    # this tiny untrained fixture a nonzero projection so it can exercise the
    # complete input-gradient path checked for the trained teacher.
    with torch.no_grad():
        model.final_conv[2].conv.weight.fill_(0.25)

    result = verifier._autograd_check(
        model,
        torch.device("cpu"),
        seed=73,
        length=33,
    )

    assert result["input_shape"] == [1, 1, 33]
    assert result["timestep"] == 100
    assert result["input_unchanged_by_residual_blocks"] is True
    assert result["input_gradient_finite"] is True
    assert result["input_gradient_max_abs"] > 0
    assert np.isfinite(result["loss"])
