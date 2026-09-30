"""Offline tests for deterministic FFHQ/LSUN image dataset protocols."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest
import torch
from PIL import Image

from pace.dataset_specs import (
    FFHQ_PROTOCOL,
    canonical_split,
    ffhq_split_paths,
    hash_ranked_paths,
    normalize_relative_path,
    ordered_path_digest,
)
from pace.image_datasets import (
    ImageFolderFlat,
    SharedImageDataset,
    build_dataset_manifest,
    discover_image_records,
    discover_numeric_image_records,
    preflight_dataset,
    write_dataset_manifest,
)


def _write_rgb(path: Path, size: tuple[int, int] = (16, 16), color=(30, 90, 180)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path)


def test_recursive_directory_and_zip_discovery_are_stable(tmp_path):
    root = tmp_path / "images"
    _write_rgb(root / "nested" / "b.png")
    _write_rgb(root / "a.jpg")
    (root / "ignore.txt").write_text("not an image")
    kind, records = discover_image_records(root)
    assert kind == "directory"
    assert [record.path for record in records] == ["a.jpg", "nested/b.png"]

    archive_path = tmp_path / "images.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(root / "nested" / "b.png", "nested/b.png")
        archive.write(root / "a.jpg", "a.jpg")
        archive.writestr("ignore.txt", "not an image")
    zip_kind, zip_records = discover_image_records(archive_path)
    assert zip_kind == "zip"
    assert [record.path for record in zip_records] == ["a.jpg", "nested/b.png"]


def test_numeric_zip_accepts_flat_prefix_and_rejects_duplicate_ids(tmp_path):
    archive_path = tmp_path / "canonical.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for image_id in range(3):
            archive.writestr(f"{image_id:05d}.png", bytes([image_id]))
    kind, records = discover_numeric_image_records(
        archive_path,
        count=3,
        digits=5,
        suffix=".png",
        display_name="fixture",
    )
    assert kind == "zip"
    assert [record.path for record in records] == ["00000.png", "00001.png", "00002.png"]

    duplicate_path = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(duplicate_path, "w") as archive:
        archive.writestr("one/00000.png", b"a")
        archive.writestr("two/00000.png", b"b")
        archive.writestr("one/00002.png", b"c")
    with pytest.raises(ValueError, match="invalid/duplicate"):
        discover_numeric_image_records(
            duplicate_path,
            count=3,
            digits=5,
            suffix=".png",
            display_name="fixture",
        )


@pytest.mark.parametrize("unsafe", ["../x.png", "/x.png", "a/../x.png", "./x.png"])
def test_normalized_paths_reject_unsafe_entries(unsafe):
    with pytest.raises(ValueError, match="safe relative path"):
        normalize_relative_path(unsafe)


def test_legacy_image_folder_is_recursive_zip_capable_and_keeps_prefix_cap(tmp_path):
    root = tmp_path / "images"
    for name, color in [("c.png", (3, 0, 0)), ("nested/a.png", (1, 0, 0)), ("b.png", (2, 0, 0))]:
        _write_rgb(root / name, color=color)
    dataset = ImageFolderFlat(str(root), image_size=8, max_images=2)
    assert dataset.paths == ["b.png", "c.png"]
    image, label = dataset[0]
    assert image.shape == (3, 8, 8)
    assert image.dtype == torch.float32
    assert image.min() >= -1 and image.max() <= 1
    assert label == -1

    archive_path = tmp_path / "images.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in root.rglob("*.png"):
            archive.write(path, path.relative_to(root).as_posix())
    zip_dataset = ImageFolderFlat(str(archive_path), image_size=8)
    assert len(zip_dataset) == 3
    assert zip_dataset[0][0].shape == (3, 8, 8)


def test_ffhq_numeric_protocol_train_monitor_and_fid_populations():
    paths = [f"prefix/{image_id:05d}.png" for image_id in range(70_000)]
    train = ffhq_split_paths(paths, "train")
    monitor = ffhq_split_paths(paths, "validation")
    fid = ffhq_split_paths(paths, "fid")
    assert len(train) == 70_000
    assert train[0].endswith("00000.png") and train[-1].endswith("69999.png")
    assert len(monitor) == 10_000
    assert monitor[0].endswith("60000.png") and monitor[-1].endswith("69999.png")
    assert len(fid) == 50_000
    assert fid[0].endswith("00000.png") and fid[-1].endswith("49999.png")
    assert canonical_split("ffhq", "val") == "monitor"
    assert ordered_path_digest(path.removeprefix("prefix/") for path in train) == (
        "ce394e07711f007391f21603ee34c9827708edd179553caacbb22d547fd77f9c"
    )


def test_ffhq_protocol_rejects_gap_or_duplicate():
    paths = [f"{image_id:05d}.png" for image_id in range(70_000)]
    paths[-1] = paths[-2]
    with pytest.raises(ValueError, match="missing numeric IDs"):
        ffhq_split_paths(paths, "train")


def test_lsun_monitor_hash_ranking_encodes_seed_and_is_ordered_by_digest():
    paths = ["0000002.jpg", "0000000.jpg", "0000001.jpg", "nested/0000003.jpg"]
    selected = hash_ranked_paths(paths, size=3, seed=12345)
    expected = sorted(
        paths,
        key=lambda path: hashlib.sha256(b"12345\0" + path.encode("utf-8")).digest(),
    )[:3]
    assert selected == expected
    assert hash_ranked_paths(paths, size=3, seed=12345) == selected
    assert hash_ranked_paths(paths, size=3, seed=1) != selected


def test_manifest_persists_lsun_monitor_entries_and_detects_source_drift(tmp_path):
    root = tmp_path / "bedroom"
    for image_id in range(6):
        _write_rgb(root / f"{image_id:07d}.jpg", size=(24, 16), color=(image_id, 1, 2))
    payload = build_dataset_manifest(
        dataset_id="lsun_bedroom",
        root=root,
        lsun_monitor_size=3,
        lsun_monitor_seed=12345,
        strict_protocol=False,
    )
    monitor = payload["splits"]["monitor"]
    assert monitor["seed"] == 12345
    assert monitor["is_heldout"] is False
    assert len(monitor["entries"]) == 3
    assert monitor["entries_sha256"] == ordered_path_digest(monitor["entries"])

    manifest_path = tmp_path / "bedroom_manifest.json"
    write_dataset_manifest(manifest_path, payload)
    dataset = SharedImageDataset(
        dataset_id="lsun_bedroom",
        root=root,
        image_size=16,
        split="monitor",
        manifest=manifest_path,
        strict_protocol=False,
    )
    assert dataset.paths == monitor["entries"]
    _write_rgb(root / "0000006.jpg", size=(24, 16))
    with pytest.raises(ValueError, match="fingerprint differs"):
        SharedImageDataset(
            dataset_id="lsun_bedroom",
            root=root,
            image_size=16,
            split="monitor",
            manifest=manifest_path,
            strict_protocol=False,
        )


def test_lsun_variable_width_is_center_cropped_and_preflights(tmp_path):
    root = tmp_path / "bedroom"
    _write_rgb(root / "0000000.jpg", size=(24, 16))
    dataset = SharedImageDataset(
        dataset_id="lsun_bedroom",
        root=root,
        image_size=16,
        split="train",
        strict_protocol=False,
    )
    report = preflight_dataset(dataset)
    assert report["valid"] is True
    assert report["sizes"] == {"24x16": 1}
    assert dataset[0][0].shape == (3, 16, 16)


def test_preflight_reports_unreadable_population_before_decode(monkeypatch, tmp_path):
    root = tmp_path / "images"
    _write_rgb(root / "a.png")
    _write_rgb(root / "b.png")
    dataset = SharedImageDataset(dataset_id="image_folder", root=root, image_size=16, split="all")
    real_access = __import__("os").access

    def fake_access(path, mode):
        return False if Path(path).name == "b.png" else real_access(path, mode)

    monkeypatch.setattr("pace.image_datasets.os.access", fake_access)
    with pytest.raises(ValueError, match=r"1/2 image files are unreadable"):
        preflight_dataset(dataset)


def test_manifest_rejects_undefined_split(tmp_path):
    root = tmp_path / "images"
    _write_rgb(root / "a.png")
    manifest = build_dataset_manifest(dataset_id="image_folder", root=root)
    manifest_path = tmp_path / "manifest.json"
    write_dataset_manifest(manifest_path, manifest)
    with pytest.raises(ValueError, match="does not define requested split"):
        SharedImageDataset(
            dataset_id="image_folder",
            root=root,
            image_size=16,
            split="monitor",
            manifest=manifest_path,
        )


def test_ffhq_non_strict_mode_does_not_guess_monitor_ranges(tmp_path):
    root = tmp_path / "ffhq"
    _write_rgb(root / "00000.png", size=(256, 256))
    with pytest.raises(ValueError, match="requires the complete validated"):
        SharedImageDataset(
            dataset_id="ffhq",
            root=root,
            image_size=256,
            split="monitor",
            strict_protocol=False,
        )
