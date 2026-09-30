"""Offline tests for the canonical SC09 raw-waveform protocol."""

from __future__ import annotations

from array import array
import hashlib
from pathlib import Path
import stat
import wave
import zipfile

import pytest
import torch

import pace.audio_datasets as audio_datasets
from pace.audio_datasets import (
    SC09_ARCHIVE_SHA256,
    SC09_ARCHIVE_SIZE,
    SC09_ARCHIVE_URL,
    SC09_LABELS,
    SC09_PROTOCOL,
    SC09_SAMPLE_LENGTH,
    SC09_SOURCE_RECORDS_SHA256,
    SC09_SPLIT_COUNTS,
    SC09_SPLIT_ENTRIES_SHA256,
    SC09Dataset,
    canonical_sc09_split,
    load_sc09_waveform,
    partition_sc09_paths,
    preflight_sc09_dataset,
)
from tests._external import requires_torchcodec
from scripts.data.prepare_sc09_dataset import (
    _acquisition_record,
    build_arg_parser,
    default_sc09_archive_path,
    download_file_atomic,
    extract_zip_safely,
    verify_sc09_archive,
)


def _write_pcm_wav(
    path: Path,
    samples: list[int],
    *,
    sample_rate: int = 16_000,
    channels: int = 1,
    sample_width: int = 2,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if sample_width == 2:
        payload = array("h", samples).tobytes()
    elif sample_width == 1:
        payload = bytes(samples)
    else:
        raise AssertionError("fixture helper only supports 8-bit and 16-bit PCM")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(sample_rate)
        handle.writeframes(payload)


def _fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "sc09"
    _write_pcm_wav(root / "zero" / "speaker_a_nohash_0.wav", [1, 2, 3])
    _write_pcm_wav(root / "one" / "speaker_b_nohash_0.wav", [4, 5, 6])
    _write_pcm_wav(root / "two" / "speaker_c_nohash_1.wav", [7, 8, 9])
    (root / "validation_list.txt").write_text("one/speaker_b_nohash_0.wav\ncat/ignored_nohash_0.wav\n")
    (root / "testing_list.txt").write_text("two/speaker_c_nohash_1.wav\n")
    return root


def test_pinned_sc09_protocol_constants():
    assert SC09_PROTOCOL == "sc09_speech_commands_v002_official_lists_raw16k_v1"
    assert SC09_ARCHIVE_URL.endswith("fe62f33d2af5db6f01e504ec1f360da7df9692e8/sc09.zip")
    assert SC09_ARCHIVE_SIZE == 893_855_183
    assert SC09_ARCHIVE_SHA256 == "ca0cff7168708fe3e1d2d6fd8ac7b0c26e4c574143ea708951e19ed6a58f7792"
    assert SC09_SPLIT_COUNTS == {"all": 38_908, "train": 31_158, "validation": 3_643, "test": 4_107}
    assert SC09_SOURCE_RECORDS_SHA256 == "9e846a24d5fdbd576b5e3a3859ef1a05cca20eac4e1d9ca7f9a1b99849930d54"
    assert set(SC09_SPLIT_ENTRIES_SHA256) == set(SC09_SPLIT_COUNTS)
    assert SC09_LABELS == ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def test_default_archive_path_recognizes_pinned_cache_name(tmp_path):
    assert default_sc09_archive_path(tmp_path / "datasets" / "sc09") == (
        tmp_path / "datasets" / "_archives" / "sc09-fe62f33.zip"
    )


def test_acquisition_record_persists_verified_local_archive(tmp_path):
    archive = tmp_path / "sc09.zip"
    report = {
        "path": str(archive.resolve()),
        "size": SC09_ARCHIVE_SIZE,
        "sha256": SC09_ARCHIVE_SHA256,
        "valid": True,
    }
    record = _acquisition_record(report)
    assert record["archive_path"] == str(archive.resolve())
    assert record["archive_size"] == SC09_ARCHIVE_SIZE
    assert record["archive_sha256"] == SC09_ARCHIVE_SHA256
    assert record["archive_valid"] is True
    assert record["dataset_commit"] == "fe62f33d2af5db6f01e504ec1f360da7df9692e8"


def test_approved_prepare_cli_aliases(tmp_path):
    args = build_arg_parser().parse_args(
        [
            "--dataset-root",
            str(tmp_path / "sc09_v0.02"),
            "--archive-dir",
            str(tmp_path / "_archives"),
            "--no-download",
        ]
    )
    assert args.output_dir == str(tmp_path / "sc09_v0.02")
    assert args.archive_dir == str(tmp_path / "_archives")
    assert args.no_download is True


def test_official_list_partition_is_disjoint_complete_and_counted():
    paths = ["zero/a_nohash_0.wav", "one/b_nohash_0.wav", "two/c_nohash_0.wav"]
    splits = partition_sc09_paths(
        paths,
        validation_paths={"one/b_nohash_0.wav"},
        testing_paths={"two/c_nohash_0.wav"},
        expected_counts={"all": 3, "train": 1, "validation": 1, "test": 1},
    )
    assert splits["train"] == ["zero/a_nohash_0.wav"]
    assert set(splits["train"]) | set(splits["validation"]) | set(splits["test"]) == set(paths)
    assert not (set(splits["train"]) & set(splits["validation"]))
    with pytest.raises(ValueError, match="overlap"):
        partition_sc09_paths(paths, validation_paths={paths[0]}, testing_paths={paths[0]})
    with pytest.raises(ValueError, match="missing digit WAVs"):
        partition_sc09_paths(paths, validation_paths={"nine/missing_nohash_0.wav"}, testing_paths=set())


@requires_torchcodec
def test_dataset_splits_labels_metadata_and_aliases(tmp_path):
    root = _fixture_root(tmp_path)
    train = SC09Dataset(root, split="train", strict_protocol=False)
    validation = SC09Dataset(root, split="monitor", strict_protocol=False)
    test = SC09Dataset(root, split="test", strict_protocol=False)
    assert train.paths == ["zero/speaker_a_nohash_0.wav"]
    assert validation.paths == ["one/speaker_b_nohash_0.wav"]
    assert test.paths == ["two/speaker_c_nohash_1.wav"]
    waveform, label = validation[0]
    assert waveform.shape == (1, 16_000)
    assert waveform.dtype == torch.float32
    assert label == 1
    assert validation.metadata["is_heldout"] is True
    assert validation.metadata["input_representation"] == "raw_waveform"
    assert validation.metadata["normalization"] == "signed_pcm16_div_32768"
    assert validation.metadata["selected_entries_sha256"]
    assert canonical_sc09_split("val") == "validation"
    assert canonical_sc09_split("monitor") == "validation"


def test_deterministic_hash_subset_repeats_and_changes_with_seed(tmp_path):
    root = tmp_path / "sc09"
    for index in range(12):
        _write_pcm_wav(root / "zero" / f"speaker_{index:02d}_nohash_0.wav", [index])
    (root / "validation_list.txt").write_text("")
    (root / "testing_list.txt").write_text("")
    first = SC09Dataset(root, split="train", max_samples=5, subset_seed=123, strict_protocol=False)
    again = SC09Dataset(root, split="train", max_samples=5, subset_seed=123, strict_protocol=False)
    other = SC09Dataset(root, split="train", max_samples=5, subset_seed=124, strict_protocol=False)
    assert first.paths == again.paths
    assert first.metadata["selected_entries_sha256"] == again.metadata["selected_entries_sha256"]
    assert first.paths != other.paths


def test_selected_content_digest_detects_same_size_audio_mutation(tmp_path):
    root = _fixture_root(tmp_path)
    before = SC09Dataset(root, split="train", strict_protocol=False)
    path = root / before.paths[0]
    size = path.stat().st_size
    _write_pcm_wav(path, [3, 2, 1])
    assert path.stat().st_size == size

    after = SC09Dataset(root, split="train", strict_protocol=False)
    assert before.metadata["selected_entries_sha256"] == after.metadata["selected_entries_sha256"]
    assert before.metadata["source_records_sha256"] == after.metadata["source_records_sha256"]
    assert before.metadata["selected_content_sha256"] != after.metadata["selected_content_sha256"]


@requires_torchcodec
def test_pcm16_normalization_leading_crop_and_right_pad(tmp_path):
    short = tmp_path / "short.wav"
    values = [-32768, -1, 0, 1, 32767]
    _write_pcm_wav(short, values)
    waveform = load_sc09_waveform(short)
    expected = torch.tensor(values, dtype=torch.float32) / 32768.0
    torch.testing.assert_close(waveform[0, : len(values)], expected, rtol=0, atol=0)
    assert torch.count_nonzero(waveform[0, len(values) :]) == 0

    long = tmp_path / "long.wav"
    long_values = list(range(SC09_SAMPLE_LENGTH)) + [30_000, 30_001]
    _write_pcm_wav(long, long_values)
    cropped = load_sc09_waveform(long)
    assert cropped.shape == (1, SC09_SAMPLE_LENGTH)
    assert cropped[0, -1].item() == pytest.approx((SC09_SAMPLE_LENGTH - 1) / 32768.0)


def test_waveform_loader_uses_torchaudio_and_checks_decoded_shape_and_rate(
    tmp_path, monkeypatch
):
    path = tmp_path / "valid.wav"
    _write_pcm_wav(path, [1, 2, 3])
    calls = []

    def fake_load(filename, *, normalize, channels_first):
        calls.append((filename, normalize, channels_first))
        return torch.tensor([[0.25, -0.5, 0.75]], dtype=torch.float32), 16_000

    monkeypatch.setattr(audio_datasets.torchaudio, "load", fake_load)
    waveform = load_sc09_waveform(path)
    assert calls == [(str(path), True, True)]
    torch.testing.assert_close(
        waveform[:, :3],
        torch.tensor([[0.25, -0.5, 0.75]], dtype=torch.float32),
    )

    monkeypatch.setattr(
        audio_datasets.torchaudio,
        "load",
        lambda *args, **kwargs: (torch.zeros(1, 3, dtype=torch.float32), 8_000),
    )
    with pytest.raises(ValueError, match="16000 Hz"):
        load_sc09_waveform(path)

    monkeypatch.setattr(
        audio_datasets.torchaudio,
        "load",
        lambda *args, **kwargs: (torch.zeros(2, 3, dtype=torch.float32), 16_000),
    )
    with pytest.raises(ValueError, match="normalized mono float32"):
        load_sc09_waveform(path)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"channels": 2}, "mono"),
        ({"sample_rate": 8_000}, "16000 Hz"),
        ({"sample_width": 1}, "PCM16"),
    ],
)
def test_waveform_loader_rejects_incompatible_media(tmp_path, kwargs, message):
    path = tmp_path / "bad.wav"
    samples = [0, 0] if kwargs.get("channels") == 2 else [0]
    _write_pcm_wav(path, samples, **kwargs)
    with pytest.raises(ValueError, match=message):
        load_sc09_waveform(path)


@requires_torchcodec
def test_preflight_reports_padding_exact_and_crop(tmp_path):
    root = tmp_path / "sc09"
    _write_pcm_wav(root / "zero" / "short_nohash_0.wav", [0] * 5)
    _write_pcm_wav(root / "one" / "exact_nohash_0.wav", [0] * 16_000)
    _write_pcm_wav(root / "two" / "long_nohash_0.wav", [0] * 16_001)
    (root / "validation_list.txt").write_text("")
    (root / "testing_list.txt").write_text("")
    dataset = SC09Dataset(root, split="all", strict_protocol=False)
    report = preflight_sc09_dataset(dataset, full_decode=True)
    assert report["valid"] is True
    assert report["checked"] == 3
    assert report["right_padded"] == 1
    assert report["exact_length"] == 1
    assert report["leading_cropped"] == 1
    assert len(report["content_sha256"]) == 64


def test_discovery_rejects_malformed_or_nested_digit_wavs(tmp_path):
    root = tmp_path / "sc09"
    _write_pcm_wav(root / "zero" / "bad.wav", [0])
    (root / "validation_list.txt").write_text("")
    (root / "testing_list.txt").write_text("")
    with pytest.raises(ValueError, match="Malformed"):
        SC09Dataset(root, split="all", strict_protocol=False)

    (root / "zero" / "bad.wav").unlink()
    _write_pcm_wav(root / "zero" / "nested" / "good_nohash_0.wav", [0])
    with pytest.raises(ValueError, match="directly inside"):
        SC09Dataset(root, split="all", strict_protocol=False)


def test_safe_zip_extraction_and_traversal_rejection(tmp_path):
    archive_path = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("sc09/validation_list.txt", "")
        archive.writestr("sc09/testing_list.txt", "")
    extracted = extract_zip_safely(archive_path, tmp_path / "extract")
    assert extracted == tmp_path / "extract" / "sc09"
    assert (extracted / "validation_list.txt").is_file()

    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as archive:
        archive.writestr("sc09/../../escape.txt", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        extract_zip_safely(unsafe, tmp_path / "unsafe-extract")
    assert not (tmp_path / "escape.txt").exists()


def test_safe_zip_rejects_symlinks(tmp_path):
    archive_path = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("sc09/link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(info, "target")
    with pytest.raises(ValueError, match="symlinks"):
        extract_zip_safely(archive_path, tmp_path / "extract")


def test_atomic_file_download_and_verification_with_local_url(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"pinned bytes")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = tmp_path / "download.bin"
    report = download_file_atomic(
        source.as_uri(),
        destination,
        expected_size=source.stat().st_size,
        expected_sha256=digest,
    )
    assert report["valid"] is True
    assert destination.read_bytes() == source.read_bytes()
    assert not list(tmp_path.glob("*.part-*"))
    assert verify_sc09_archive(
        destination,
        expected_size=source.stat().st_size,
        expected_sha256=digest,
    )["valid"] is True
