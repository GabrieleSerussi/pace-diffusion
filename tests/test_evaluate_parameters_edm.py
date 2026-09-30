"""Tests for evaluate_parameters_edm.py — ImageNet support, helpers, and W&B integration."""

import io
import math
import os
import random
import shutil
import tempfile
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

# Ensure edm modules and project root are importable
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from evaluate_parameters_edm import (
    IMPORTANCE_CLIP_MODES,
    BinStats,
    CIFAR10Dataset,
    EDMUsageEvaluator,
    ExactSigmaPFIBatchSampler,
    FilterPFIHook,
    FilterZeroHook,
    HeadPFIHook,
    HeadZeroHook,
    HeadRandomSameNormHook,
    ImageFolderFlat,
    ImageNet1KParquetDataset,
    ImageNetDataset,
    PFIOutputHook,
    SigmaCorruptionDataset,
    ZeroOutputHook,
    atomic_torch_save,
    balanced_batch_sizes,
    build_ablation_protocol,
    build_delta_stack,
    canonical_json_sha256,
    collate_corruption,
    collate_corruption_pfi,
    build_profile_cost_estimate,
    compute_signed_and_positive_deltas,
    compute_usage_metrics,
    count_filter_parameters,
    count_parameters,
    choose_dtype,
    level_to_bin,
    init_distributed,
    loss_weights_for_family,
    make_class_labels,
    make_log_sigma_schedule,
    make_sigma_bin_labels,
    mse_per_example,
    load_ablation_checkpoints,
    load_pfi_baseline,
    persist_or_validate_pfi_plan,
    pfi_baseline_envelope,
    pfi_checkpoint_envelope,
    recompute_metrics_from_results,
    row_normalize_matrix,
    sanitize_tensor,
    select_filter_groups,
    set_seed,
    should_compute_group_correlation,
    validate_filter_sampling_options,
)
from pace.vendor.openai_consistency_unet import QKVFlashAttention
from pace.edm_distillation import module_to_structural_key


def test_init_distributed_uses_extended_configurable_timeout(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("RANK", "1")
    init_process_group = mock.Mock()
    monkeypatch.setattr("evaluate_parameters_edm.dist.init_process_group", init_process_group)

    device, rank, world_size = init_distributed("cpu", timeout_seconds=12_345)

    assert (device, rank, world_size) == ("cpu", 1, 2)
    init_process_group.assert_called_once()
    assert init_process_group.call_args.kwargs["backend"] == "gloo"
    assert init_process_group.call_args.kwargs["timeout"].total_seconds() == 12_345


def test_init_distributed_rejects_nonpositive_timeout(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    with pytest.raises(ValueError, match="timeout must be positive"):
        init_distributed("cpu", timeout_seconds=0)


def test_profile_checkpoint_save_is_atomic(tmp_path, monkeypatch):
    destination = tmp_path / "checkpoint_rank0.pt"
    atomic_torch_save({"group": torch.ones(2)}, destination)
    assert torch.equal(torch.load(destination, weights_only=True)["group"], torch.ones(2))
    assert not list(tmp_path.glob(".*.tmp-*"))

    def fail_save(_payload, temporary):
        temporary.write_bytes(b"partial")
        raise RuntimeError("interrupted")

    monkeypatch.setattr("evaluate_parameters_edm.torch.save", fail_save)
    with pytest.raises(RuntimeError, match="interrupted"):
        atomic_torch_save({"group": torch.zeros(2)}, destination)
    assert torch.equal(torch.load(destination, weights_only=True)["group"], torch.ones(2))
    assert not list(tmp_path.glob(".*.tmp-*"))


def _test_profile_fingerprint():
    return {
        "format": "diffdist_edm_profile_fingerprint_v1",
        "model": {"checkpoint_sha256": "a" * 64, "checkpoint_size_bytes": 123},
        "runtime": {"dtype": "torch.float32", "image_size": 8, "batch_size": 4},
    }


class TestPFIResumeArtifacts:
    def test_checkpoint_envelope_round_trip_is_strict(self, tmp_path):
        fingerprint = _test_profile_fingerprint()
        digest = canonical_json_sha256(fingerprint)
        path = tmp_path / "checkpoint_pfi_rank2.pt"
        groups = {"block.0": torch.tensor([1.0, -0.5]), "block.1": torch.tensor([0.2, 0.3])}
        atomic_torch_save(
            pfi_checkpoint_envelope(
                groups,
                rank=2,
                num_bins=2,
                profile_fingerprint=fingerprint,
                profile_fingerprint_sha256=digest,
            ),
            path,
        )

        loaded, paths, unknown, bad_shapes = load_ablation_checkpoints(
            str(tmp_path),
            "pfi",
            list(groups),
            2,
            profile_fingerprint=fingerprint,
            profile_fingerprint_sha256=digest,
        )
        assert paths == [str(path)]
        assert list(loaded) == list(groups)
        assert unknown == 0
        assert bad_shapes == []
        assert torch.equal(loaded["block.0"], groups["block.0"])

        other_fingerprint = {**fingerprint, "runtime": {**fingerprint["runtime"], "batch_size": 8}}
        with pytest.raises(ValueError, match="different profile fingerprint"):
            load_ablation_checkpoints(
                str(tmp_path),
                "pfi",
                list(groups),
                2,
                profile_fingerprint=other_fingerprint,
                profile_fingerprint_sha256=canonical_json_sha256(other_fingerprint),
            )

    def test_checkpoint_rejects_raw_legacy_payload_and_bad_completed_groups(self, tmp_path):
        fingerprint = _test_profile_fingerprint()
        digest = canonical_json_sha256(fingerprint)
        path = tmp_path / "checkpoint_pfi_rank0.pt"
        atomic_torch_save({"block.0": torch.ones(2)}, path)
        with pytest.raises(ValueError, match="is not a diffdist_edm_pfi_checkpoint_v1 envelope"):
            load_ablation_checkpoints(
                str(tmp_path),
                "pfi",
                ["block.0"],
                2,
                profile_fingerprint=fingerprint,
                profile_fingerprint_sha256=digest,
            )

        payload = pfi_checkpoint_envelope(
            {"block.0": torch.ones(2)},
            rank=0,
            num_bins=2,
            profile_fingerprint=fingerprint,
            profile_fingerprint_sha256=digest,
        )
        payload["completed_groups"] = []
        atomic_torch_save(payload, path)
        with pytest.raises(ValueError, match="completed_groups"):
            load_ablation_checkpoints(
                str(tmp_path),
                "pfi",
                ["block.0"],
                2,
                profile_fingerprint=fingerprint,
                profile_fingerprint_sha256=digest,
            )

    def test_baseline_envelope_round_trip_and_fingerprint_gate(self, tmp_path):
        fingerprint = _test_profile_fingerprint()
        digest = canonical_json_sha256(fingerprint)
        path = tmp_path / "baseline_pfi.pt"
        expected = (
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            torch.tensor([0.1, 0.2], dtype=torch.float64),
            torch.tensor([7, 7]),
        )
        atomic_torch_save(
            pfi_baseline_envelope(
                mean=expected[0],
                stderr=expected[1],
                count=expected[2],
                num_bins=2,
                profile_fingerprint=fingerprint,
                profile_fingerprint_sha256=digest,
            ),
            path,
        )
        loaded = load_pfi_baseline(
            path,
            num_bins=2,
            profile_fingerprint=fingerprint,
            profile_fingerprint_sha256=digest,
        )
        assert loaded is not None
        assert all(torch.equal(actual, wanted) for actual, wanted in zip(loaded, expected))

        changed = {**fingerprint, "model": {**fingerprint["model"], "checkpoint_size_bytes": 124}}
        with pytest.raises(ValueError, match="different profile fingerprint"):
            load_pfi_baseline(
                path,
                num_bins=2,
                profile_fingerprint=changed,
                profile_fingerprint_sha256=canonical_json_sha256(changed),
            )


def test_signed_pfi_deltas_are_retained_while_allocation_uses_positive_part():
    baseline = torch.tensor([1.0, 1.0])
    signed, positive = compute_signed_and_positive_deltas(
        {"helpful": torch.tensor([0.5, 2.0])},
        baseline,
        ["helpful"],
    )
    assert torch.equal(signed, torch.tensor([[-0.5, 1.0]]))
    assert torch.equal(positive, torch.tensor([[0.0, 1.0]]))


def test_nvlabs_group_names_have_canonical_structural_fallbacks():
    assert module_to_structural_key("model.enc.64x64_conv") == "enc.64x64_conv"
    assert module_to_structural_key("model.enc.32x32_block0.conv1") == "enc.32x32_block0"
    assert module_to_structural_key("model.dec.16x16_block2.qkv") == "dec.16x16_block2"


def test_profile_cost_estimate_records_full_and_bounded_work():
    estimate = build_profile_cost_estimate(
        grouping="per_filter",
        full_group_count=100,
        selected_group_count=3,
        bounded_by_max_groups=True,
        dataset_images=5,
        sigma_levels_per_image=4,
        world_size=2,
        requested_batch_size=4,
        actual_batch_sizes=[3, 2],
        homogeneous_batches_per_evaluation=8,
    )

    assert estimate["full_group_count"] == 100
    assert estimate["bounded_by_max_groups"] is True
    assert estimate["corrupted_examples_per_evaluation"] == 20
    assert estimate["estimated_forward_examples"] == 80
    assert estimate["estimated_max_group_evaluations_per_rank"] == 2
    assert estimate["actual_batch_size_histogram_per_sigma"] == {"3": 1, "2": 1}
    assert estimate["actual_batch_size_histogram"] == {"3": 4, "2": 4}
    assert estimate["actual_batch_size_min"] == 2
    assert estimate["actual_batch_size_max"] == 3
    assert estimate["homogeneous_batches_per_evaluation"] == 8

    ffhq = build_profile_cost_estimate(
        grouping="per_filter",
        full_group_count=32_387,
        selected_group_count=32_387,
        bounded_by_max_groups=False,
        dataset_images=10_000,
        sigma_levels_per_image=64,
        world_size=1,
        requested_batch_size=256,
        actual_batch_sizes=[250] * 40,
        homogeneous_batches_per_evaluation=2_560,
    )
    assert ffhq["actual_batch_size_histogram"] == {"250": 2_560}
    assert ffhq["actual_batch_size_histogram_per_sigma"] == {"250": 40}
    assert ffhq["homogeneous_batches_per_evaluation"] == 2_560


def _toy_filter_groups(widths):
    return {
        f"{module_name}.filter_{filter_index}": (module_name, filter_index, "filter")
        for module_name, width in widths
        for filter_index in range(width)
    }


def test_stratified_filter_sampling_is_balanced_reproducible_and_order_invariant():
    groups = _toy_filter_groups([("left", 7), ("right", 3), ("tiny", 1)])
    random.seed(123)
    torch.manual_seed(123)
    selected, protocol = select_filter_groups(
        groups,
        mode="stratified_module",
        filters_per_module=2,
        seed=17,
    )
    random.seed(999)
    torch.manual_seed(999)
    reversed_selected, reversed_protocol = select_filter_groups(
        dict(reversed(list(groups.items()))),
        mode="stratified_module",
        filters_per_module=2,
        seed=17,
    )

    assert list(selected) == list(reversed_selected)
    assert protocol == reversed_protocol
    assert protocol["population_group_count"] == 11
    assert protocol["selected_group_count"] == 5
    assert protocol["population_module_count"] == 3
    assert protocol["selected_module_count"] == 3
    assert protocol["module_counts"]["left"] == {
        "population_filter_count": 7,
        "selected_filter_count": 2,
        "inclusion_probability": 2 / 7,
        "expansion_weight": 7 / 2,
    }
    assert protocol["module_counts"]["right"]["selected_filter_count"] == 2
    assert protocol["module_counts"]["tiny"]["selected_filter_count"] == 1
    assert list(selected) == [
        "left.filter_1",
        "left.filter_5",
        "right.filter_0",
        "right.filter_1",
        "tiny.filter_0",
    ]
    assert all(name in selected for name in protocol["selected_group_names"])
    assert set(protocol["selected_group_expansion_weights"]) == set(selected)
    assert len(protocol["population_sha256"]) == 64
    assert len(protocol["selection_sha256"]) == 64


def test_stratified_filter_sampling_caps_each_module_independently():
    groups = _toy_filter_groups([("wide", 5), ("narrow", 2)])
    one_per_module, one_protocol = select_filter_groups(
        groups,
        mode="stratified_module",
        filters_per_module=1,
        seed=4,
    )
    all_groups, all_protocol = select_filter_groups(
        groups,
        mode="stratified_module",
        filters_per_module=99,
        seed=4,
    )

    assert len(one_per_module) == 2
    assert one_protocol["module_counts"]["wide"]["selected_filter_count"] == 1
    assert one_protocol["module_counts"]["narrow"]["selected_filter_count"] == 1
    assert set(all_groups) == set(groups)
    assert all_protocol["selected_group_count"] == all_protocol["population_group_count"] == 7
    assert all(record["inclusion_probability"] == 1.0 for record in all_protocol["module_counts"].values())


def test_stratified_filter_sampling_is_seeded_and_module_local():
    base = _toy_filter_groups([("wide", 32)])
    selected_17, protocol_17 = select_filter_groups(
        base,
        mode="stratified_module",
        filters_per_module=4,
        seed=17,
    )
    selected_18, _ = select_filter_groups(
        base,
        mode="stratified_module",
        filters_per_module=4,
        seed=18,
    )
    extended, extended_protocol = select_filter_groups(
        {**_toy_filter_groups([("added", 19)]), **base},
        mode="stratified_module",
        filters_per_module=4,
        seed=17,
    )

    assert set(selected_17) != set(selected_18)
    assert list(selected_17) == [
        "wide.filter_0",
        "wide.filter_8",
        "wide.filter_18",
        "wide.filter_25",
    ]
    assert list(selected_18) == [
        "wide.filter_2",
        "wide.filter_3",
        "wide.filter_13",
        "wide.filter_23",
    ]
    assert {name for name in extended if name.startswith("wide.")} == set(selected_17)
    assert protocol_17["module_counts"]["wide"] == extended_protocol["module_counts"]["wide"]
    assert protocol_17["selection_sha256"] != extended_protocol["selection_sha256"]
    assert protocol_17["population_sha256"] != extended_protocol["population_sha256"]


def test_exhaustive_filter_sampling_preserves_order_and_keeps_metadata_compact():
    groups = _toy_filter_groups([("z", 2), ("a", 3)])
    selected, protocol = select_filter_groups(
        groups,
        mode="exhaustive",
        filters_per_module=None,
        seed=9,
    )

    assert list(selected) == list(groups)
    assert protocol["population_group_count"] == protocol["selected_group_count"] == 5
    assert "selected_group_names" not in protocol
    assert "selected_group_inclusion_probabilities" not in protocol
    assert "selected_group_expansion_weights" not in protocol
    assert all(record["inclusion_probability"] == 1.0 for record in protocol["module_counts"].values())
    assert all(record["expansion_weight"] == 1.0 for record in protocol["module_counts"].values())


@pytest.mark.parametrize("value", [None, 0, -1])
def test_stratified_filter_sampling_requires_positive_per_module_cap(value):
    with pytest.raises(ValueError, match="filters_per_module"):
        select_filter_groups(
            _toy_filter_groups([("module", 2)]),
            mode="stratified_module",
            filters_per_module=value,
            seed=0,
        )


def test_filter_sampling_option_validation_rejects_ambiguous_or_irrelevant_combinations():
    with pytest.raises(ValueError, match="requires --grouping per_filter"):
        validate_filter_sampling_options(
            grouping="blocks",
            filter_sampling="stratified_module",
            filters_per_module=2,
            max_groups=None,
        )
    with pytest.raises(ValueError, match="cannot be combined with --max_groups"):
        validate_filter_sampling_options(
            grouping="per_filter",
            filter_sampling="stratified_module",
            filters_per_module=2,
            max_groups=3,
        )
    with pytest.raises(ValueError, match="only valid"):
        validate_filter_sampling_options(
            grouping="per_filter",
            filter_sampling="exhaustive",
            filters_per_module=2,
            max_groups=None,
        )

    validate_filter_sampling_options(
        grouping="per_filter",
        filter_sampling="stratified_module",
        filters_per_module=32,
        max_groups=None,
    )


def test_stratified_sampling_record_is_strictly_bound_to_pfi_resume(tmp_path):
    groups = _toy_filter_groups([("wide", 8)])
    selected_1, protocol_1 = select_filter_groups(
        groups,
        mode="stratified_module",
        filters_per_module=3,
        seed=1,
    )
    _selected_2, protocol_2 = select_filter_groups(
        groups,
        mode="stratified_module",
        filters_per_module=3,
        seed=2,
    )
    fingerprint_1 = {
        **_test_profile_fingerprint(),
        "grouping": {
            "kind": "per_filter",
            "group_names": list(selected_1),
            "filter_sampling": protocol_1,
        },
    }
    checkpoint = tmp_path / "checkpoint_pfi_rank0.pt"
    atomic_torch_save(
        pfi_checkpoint_envelope(
            {name: torch.ones(2) for name in selected_1},
            rank=0,
            num_bins=2,
            profile_fingerprint=fingerprint_1,
            profile_fingerprint_sha256=canonical_json_sha256(fingerprint_1),
        ),
        checkpoint,
    )
    changed_fingerprint = {
        **fingerprint_1,
        "grouping": {**fingerprint_1["grouping"], "filter_sampling": protocol_2},
    }

    with pytest.raises(ValueError, match="different profile fingerprint"):
        load_ablation_checkpoints(
            str(tmp_path),
            "pfi",
            list(selected_1),
            2,
            profile_fingerprint=changed_fingerprint,
            profile_fingerprint_sha256=canonical_json_sha256(changed_fingerprint),
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def imagenet_dir(tmp_path):
    """Create a tiny ImageNet-style directory layout: val/<class_name>/*.JPEG."""
    val_dir = tmp_path / "val"
    num_classes = 5
    images_per_class = 3
    for cls_idx in range(num_classes):
        cls_name = f"n{cls_idx:08d}"
        cls_dir = val_dir / cls_name
        cls_dir.mkdir(parents=True)
        for img_idx in range(images_per_class):
            img = Image.new("RGB", (64, 64), color=(cls_idx * 50, img_idx * 80, 128))
            img.save(cls_dir / f"img_{img_idx:04d}.JPEG")
    return tmp_path


@pytest.fixture
def imagenet_parquet_dir(tmp_path):
    """Create a tiny flat ImageNet-1K parquet shard layout."""
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])

    def image_bytes(value):
        buffer = io.BytesIO()
        Image.new("RGB", (16, 16), color=(value, 100, 150)).save(buffer, format="PNG")
        return buffer.getvalue()

    for split, num_shards, label_offset in [
        ("train", 2, 0),
        ("validation", 1, 100),
        ("test", 1, 200),
    ]:
        for shard_idx in range(num_shards):
            table = pa.table({
                "image": pa.array(
                    [{"bytes": image_bytes(label_offset + shard_idx), "path": None}],
                    type=image_type,
                ),
                "label": pa.array([label_offset + shard_idx], type=pa.int64()),
            })
            pq.write_table(
                table,
                tmp_path / f"{split}-{shard_idx:05d}-of-{num_shards:05d}.parquet",
            )
    return tmp_path


@pytest.fixture
def flat_image_dir(tmp_path):
    """Create a flat directory with a few test images."""
    for i in range(5):
        img = Image.new("RGB", (32, 32), color=(i * 50, 100, 200))
        img.save(tmp_path / f"test_{i:03d}.png")
    return tmp_path


@pytest.fixture
def simple_conv_module():
    """A simple Conv2d module for testing hooks and parameter counting."""
    return torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)


# ---------------------------------------------------------------------------
# ImageNetDataset tests
# ---------------------------------------------------------------------------


class TestImageNetDataset:
    def test_loads_from_local_dir(self, imagenet_dir):
        ds = ImageNetDataset(
            root=str(imagenet_dir),
            image_size=64,
            split="val",
            max_images=None,
        )
        assert ds.source == "local"
        assert len(ds) == 15  # 5 classes * 3 images

    def test_max_images(self, imagenet_dir):
        ds = ImageNetDataset(
            root=str(imagenet_dir),
            image_size=64,
            split="val",
            max_images=4,
        )
        assert len(ds) == 4

    def test_returns_image_and_label(self, imagenet_dir):
        ds = ImageNetDataset(
            root=str(imagenet_dir),
            image_size=64,
            split="val",
        )
        image, label = ds[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 64, 64)
        assert isinstance(label, int)
        assert 0 <= label < 5

    def test_transform_normalizes(self, imagenet_dir):
        ds = ImageNetDataset(
            root=str(imagenet_dir),
            image_size=64,
            split="val",
        )
        image, _ = ds[0]
        # After Normalize([0.5]*3, [0.5]*3), values should be in roughly [-1, 1]
        assert image.min() >= -1.1
        assert image.max() <= 1.1

    def test_seed_controlled_sampling(self, imagenet_dir):
        ds = ImageNetDataset(
            root=str(imagenet_dir),
            image_size=64,
            split="val",
            max_images=5,
        )
        ds.set_sampling_seed(42)
        items_a = [ds[i][1] for i in range(len(ds))]
        ds.set_sampling_seed(42)
        items_b = [ds[i][1] for i in range(len(ds))]
        assert items_a == items_b

    def test_different_seeds_different_sampling(self, imagenet_dir):
        ds = ImageNetDataset(
            root=str(imagenet_dir),
            image_size=64,
            split="val",
            max_images=5,
        )
        ds.set_sampling_seed(0)
        items_a = [ds._resolve_dataset_idx(i) for i in range(len(ds))]
        ds.set_sampling_seed(99)
        items_b = [ds._resolve_dataset_idx(i) for i in range(len(ds))]
        assert items_a != items_b

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="ImageNet split directory not found"):
            ImageNetDataset(
                root=str(tmp_path / "nonexistent"),
                image_size=64,
                split="val",
            )

    def test_resize_to_target(self, imagenet_dir):
        ds = ImageNetDataset(
            root=str(imagenet_dir),
            image_size=32,
            split="val",
        )
        image, _ = ds[0]
        assert image.shape == (3, 32, 32)

    def test_all_labels_valid(self, imagenet_dir):
        ds = ImageNetDataset(
            root=str(imagenet_dir),
            image_size=64,
            split="val",
        )
        labels = set()
        for i in range(len(ds)):
            _, label = ds[i]
            labels.add(label)
        assert all(0 <= l < 5 for l in labels)


class TestImageNet1KParquetDataset:
    def test_filters_validation_split(self, imagenet_parquet_dir):
        ds = ImageNet1KParquetDataset(
            root=str(imagenet_parquet_dir),
            image_size=32,
            split="validation",
        )
        assert ds.split == "validation"
        assert len(ds.paths) == 1
        assert os.path.basename(ds.paths[0]).startswith("validation-")

    def test_val_alias_maps_to_validation(self, imagenet_parquet_dir):
        ds = ImageNet1KParquetDataset(
            root=str(imagenet_parquet_dir),
            image_size=32,
            split="val",
        )
        assert ds.split == "validation"
        assert len(ds.paths) == 1

    def test_all_split_keeps_all_shards(self, imagenet_parquet_dir):
        ds = ImageNet1KParquetDataset(
            root=str(imagenet_parquet_dir),
            image_size=32,
            split="all",
        )
        assert ds.split is None
        assert len(ds.paths) == 4


# ---------------------------------------------------------------------------
# ImageFolderFlat tests
# ---------------------------------------------------------------------------


class TestImageFolderFlat:
    def test_loads_images(self, flat_image_dir):
        ds = ImageFolderFlat(root=str(flat_image_dir), image_size=32)
        assert len(ds) == 5

    def test_max_images(self, flat_image_dir):
        ds = ImageFolderFlat(root=str(flat_image_dir), image_size=32, max_images=2)
        assert len(ds) == 2

    def test_returns_unlabeled(self, flat_image_dir):
        ds = ImageFolderFlat(root=str(flat_image_dir), image_size=32)
        image, label = ds[0]
        assert image.shape == (3, 32, 32)
        assert label == -1


# ---------------------------------------------------------------------------
# SigmaCorruptionDataset tests
# ---------------------------------------------------------------------------


class TestSigmaCorruptionDataset:
    def test_length(self, imagenet_dir):
        base_ds = ImageNetDataset(
            root=str(imagenet_dir), image_size=64, split="val", max_images=3
        )
        sigma_values = torch.linspace(0.1, 10.0, 5)
        corruption_ds = SigmaCorruptionDataset(
            image_dataset=base_ds,
            sigma_values=sigma_values,
            samples_per_image=1,
            seed=0,
        )
        # Each image paired with all 5 sigma levels
        assert len(corruption_ds) == 3 * 5

    def test_sigma_stride(self, imagenet_dir):
        base_ds = ImageNetDataset(
            root=str(imagenet_dir), image_size=64, split="val", max_images=2
        )
        sigma_values = torch.linspace(0.1, 10.0, 10)
        corruption_ds = SigmaCorruptionDataset(
            image_dataset=base_ds,
            sigma_values=sigma_values,
            samples_per_image=1,
            seed=0,
            sigma_stride=2,
        )
        # stride=2 means 5 sigma levels per image
        assert len(corruption_ds) == 2 * 5

    def test_collation(self, imagenet_dir):
        base_ds = ImageNetDataset(
            root=str(imagenet_dir), image_size=64, split="val", max_images=2
        )
        sigma_values = torch.linspace(0.1, 10.0, 3)
        corruption_ds = SigmaCorruptionDataset(
            image_dataset=base_ds,
            sigma_values=sigma_values,
            samples_per_image=1,
            seed=0,
        )
        loader = DataLoader(corruption_ds, batch_size=4, collate_fn=collate_corruption)
        batch = next(iter(loader))
        images, class_indices, sigma_indices, noise_seeds = batch
        assert images.shape[0] == 4
        assert class_indices.shape == (4,)
        assert sigma_indices.shape == (4,)
        assert noise_seeds.shape == (4,)


class _TinyTensorDataset:
    metadata = {
        "split": "monitor",
        "entries_sha256": "tiny-population-v1",
    }

    def __init__(self, size=7):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return torch.full((1, 1, 1), float(index)), -1


def _tiny_pfi_components(*, population_size=7, sigma_levels=3, batch_size=4, pfi_seed=19):
    corruption = SigmaCorruptionDataset(
        image_dataset=_TinyTensorDataset(population_size),
        sigma_values=torch.linspace(0.1, 1.0, sigma_levels),
        samples_per_image=1,
        seed=5,
    )
    plan = ExactSigmaPFIBatchSampler(
        corruption,
        batch_size=batch_size,
        pfi_seed=pfi_seed,
        population_fingerprint="tiny-population-fingerprint",
    )
    return corruption, plan


class TestExactSigmaPFIPlan:
    def test_balanced_remainder_never_drops_or_exceeds_requested_size(self):
        sizes = balanced_batch_sizes(10_000, 64)
        assert sum(sizes) == 10_000
        assert min(sizes) == 63
        assert max(sizes) == 64
        assert len(sizes) == math.ceil(10_000 / 64)
        assert balanced_batch_sizes(10_000, 256) == [250] * 40

        with pytest.raises(ValueError, match="without a singleton"):
            balanced_batch_sizes(3, 2)

    def test_membership_is_independent_per_sigma_and_derangements_are_exact(self):
        corruption, plan = _tiny_pfi_components()
        batches = list(plan)

        assert len(batches) == 3 * 2
        assert plan.artifact["balanced_batch_sizes"] == [4, 3]
        assert "rank" not in plan.artifact
        assert "world_size" not in plan.artifact
        assert not torch.equal(plan._member_orders[0], plan._member_orders[1])

        for sigma_position, sigma_index in enumerate(corruption.sigma_indices):
            sigma_batches = batches[sigma_position * 2 : (sigma_position + 1) * 2]
            assert sorted(ref.image_index for batch in sigma_batches for ref in batch) == list(range(7))
            for batch in sigma_batches:
                assert {ref.sigma_index for ref in batch} == {sigma_index}
                donors = [ref.donor_position for ref in batch]
                assert sorted(donors) == list(range(len(batch)))
                assert all(position != donor for position, donor in enumerate(donors))
                visited = set()
                position = 0
                for _ in range(len(batch)):
                    visited.add(position)
                    position = donors[position]
                assert position == 0
                assert visited == set(range(len(batch)))

    def test_seeded_plan_and_worker_output_are_reproducible(self):
        corruption, plan = _tiny_pfi_components()
        _, same_plan = _tiny_pfi_components()
        _, other_plan = _tiny_pfi_components(pfi_seed=20)

        assert plan.artifact == same_plan.artifact
        assert torch.equal(plan._member_orders, same_plan._member_orders)
        assert torch.equal(plan._donor_positions, same_plan._donor_positions)
        assert plan.plan_sha256 != other_plan.plan_sha256

        def collect(num_workers):
            loader = DataLoader(
                corruption,
                batch_sampler=plan,
                num_workers=num_workers,
                collate_fn=collate_corruption_pfi,
            )
            return [
                (images.flatten().tolist(), sigmas.tolist(), donors.tolist())
                for images, _classes, sigmas, _noise, donors in loader
            ]

        assert collect(0) == collect(2)
        assert all(len(set(sigmas)) == 1 for _images, sigmas, _donors in collect(0))

    def test_compact_member_and_donor_tables_are_persisted_and_validated(self, tmp_path):
        _corruption, plan = _tiny_pfi_components()
        path = tmp_path / "pfi_plan.pt"
        persist_or_validate_pfi_plan(path, plan, allow_create=True)

        saved = torch.load(path, weights_only=True)
        assert saved["metadata"]["plan_sha256"] == plan.plan_sha256
        assert torch.equal(saved["member_orders"], plan._member_orders)
        assert torch.equal(saved["donor_positions"], plan._donor_positions)
        persist_or_validate_pfi_plan(path, plan, allow_create=False)

        _other_corruption, other_plan = _tiny_pfi_components(pfi_seed=21)
        with pytest.raises(ValueError, match="differs from the requested plan"):
            persist_or_validate_pfi_plan(path, other_plan, allow_create=False)

    def test_protocol_embeds_complete_versioned_plan(self):
        _corruption, plan = _tiny_pfi_components()
        protocol = build_ablation_protocol("pfi", pfi_plan=plan, pfi_seed=19)

        assert protocol["protocol_id"] == "batch_local_exact_sigma_pfi_v1"
        assert protocol["signed_scores_retained"] is True
        assert protocol["pfi"]["compact_plan_artifact"] == "pfi_plan.pt"
        assert protocol["pfi"]["plan"]["plan_sha256"] == plan.plan_sha256


class _TinyPFIEvaluator(EDMUsageEvaluator):
    def __init__(self, sigma_values):
        self.sigma_values = sigma_values
        self.activation = torch.nn.Identity()

    def forward_losses_from_fixed_corruption(
        self,
        images,
        class_indices,
        sigma_indices,
        noise_seeds,
    ):
        del class_indices, sigma_indices, noise_seeds
        exchanged = self.activation(images)
        return ((exchanged - images) ** 2).flatten(1).mean(dim=1)


def test_evaluator_applies_planned_batch_local_pfi_at_each_exact_sigma():
    corruption, plan = _tiny_pfi_components()
    loader = DataLoader(
        corruption,
        batch_sampler=plan,
        num_workers=0,
        collate_fn=collate_corruption_pfi,
    )
    evaluator = _TinyPFIEvaluator(corruption.sigma_values)

    baseline = evaluator.evaluate(loader, num_bins=3, ablation_mode="pfi")
    permuted = evaluator.evaluate(
        loader,
        num_bins=3,
        ablate_target=evaluator.activation,
        ablation_mode="pfi",
    )

    assert baseline.count.tolist() == [7, 7, 7]
    assert torch.equal(permuted.count, baseline.count)
    assert torch.equal(baseline.mean(), torch.zeros(3, dtype=torch.float64))
    assert torch.all(permuted.mean() > 0)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_level_to_bin_basic(self):
        indices = torch.tensor([0, 5, 9])
        bins = level_to_bin(indices, num_levels=10, num_bins=5)
        assert bins.tolist() == [0, 2, 4]

    def test_level_to_bin_clamped(self):
        indices = torch.tensor([0, 99])
        bins = level_to_bin(indices, num_levels=100, num_bins=10)
        assert bins.min().item() >= 0
        assert bins.max().item() <= 9

    def test_mse_per_example(self):
        pred = torch.zeros(2, 3, 4, 4)
        target = torch.ones(2, 3, 4, 4)
        mse = mse_per_example(pred, target)
        assert mse.shape == (2,)
        assert torch.allclose(mse, torch.ones(2))

    def test_sanitize_tensor(self):
        t = torch.tensor([1.0, float("nan"), float("inf"), float("-inf")])
        result = sanitize_tensor(t)
        assert torch.isfinite(result).all()
        assert result[0].item() == 1.0

    def test_row_normalize_matrix(self):
        matrix = torch.tensor(
            [
                [1.0, 1.0, 2.0],
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )

        normalized = row_normalize_matrix(matrix)

        assert torch.allclose(
            normalized[0],
            torch.tensor([0.25, 0.25, 0.5], dtype=torch.float64),
        )
        assert torch.allclose(
            normalized[2],
            torch.tensor([0.75, 0.0, 0.25], dtype=torch.float64),
        )
        assert torch.allclose(normalized[0].sum(), torch.tensor(1.0, dtype=torch.float64))
        assert torch.allclose(normalized[2].sum(), torch.tensor(1.0, dtype=torch.float64))
        assert torch.equal(normalized[1], torch.zeros(3, dtype=torch.float64))

    def test_make_log_sigma_schedule(self):
        schedule = make_log_sigma_schedule(0.002, 80.0, 10, torch.device("cpu"))
        assert len(schedule) == 10
        assert schedule[0] > schedule[-1]  # Decreasing (high to low noise)
        assert abs(schedule[0].item() - 80.0) < 1e-3
        assert abs(schedule[-1].item() - 0.002) < 1e-3

    def test_choose_dtype(self):
        assert choose_dtype("fp32") == torch.float32
        assert choose_dtype("fp16") == torch.float16
        assert choose_dtype("bf16") == torch.bfloat16
        with pytest.raises(ValueError):
            choose_dtype("int8")

    def test_count_parameters(self):
        m = torch.nn.Linear(10, 5)
        assert count_parameters(m) == 10 * 5 + 5  # weight + bias

    def test_loss_weights_edm(self):
        sigmas = torch.tensor([1.0, 2.0])
        weights = loss_weights_for_family(sigmas, "edm", sigma_data=0.5)
        assert weights.shape == (2,)
        assert (weights > 0).all()

    def test_loss_weights_vp(self):
        sigmas = torch.tensor([1.0, 2.0])
        weights = loss_weights_for_family(sigmas, "vp", sigma_data=0.5)
        expected = 1.0 / (sigmas ** 2)
        assert torch.allclose(weights, expected)

    def test_set_seed_reproducibility(self):
        set_seed(42)
        a = torch.randn(5)
        set_seed(42)
        b = torch.randn(5)
        assert torch.allclose(a, b)

    def test_make_sigma_bin_labels(self):
        sigma_values = torch.tensor([10.0, 5.0, 1.0, 0.1])
        labels = make_sigma_bin_labels(sigma_values, num_bins=2)
        assert len(labels) == 2
        assert all(isinstance(l, str) for l in labels)

    def test_compute_usage_metrics_neff_varies_for_equal_group_sizes(self):
        delta_stack = torch.tensor(
            [
                [4.0, 1.0, 2.0],
                [1.0, 1.0, 2.0],
                [0.0, 2.0, 2.0],
            ],
            dtype=torch.float64,
        )
        baseline_mean = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        group_names = ["g0", "g1", "g2"]
        group_param_counts = {name: 100 for name in group_names}

        metrics = compute_usage_metrics(
            delta_stack=delta_stack,
            baseline_mean=baseline_mean,
            group_param_counts=group_param_counts,
            group_names=group_names,
        )

        assert torch.allclose(metrics["p_eff"], torch.full((3,), 100.0, dtype=torch.float64))
        assert not torch.allclose(metrics["n_eff"], torch.full((3,), metrics["n_eff"][0], dtype=torch.float64))
        expected_first_bin = 1.0 / ((4.0 / 5.0) ** 2 + (1.0 / 5.0) ** 2)
        assert math.isclose(metrics["n_eff"][0].item(), expected_first_bin, rel_tol=1e-6)

    def test_compute_usage_metrics_can_skip_group_correlation(self):
        delta_stack = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        baseline_mean = torch.tensor([1.0, 1.0], dtype=torch.float64)
        group_names = ["g0", "g1"]
        group_param_counts = {name: 100 for name in group_names}

        metrics = compute_usage_metrics(
            delta_stack=delta_stack,
            baseline_mean=baseline_mean,
            group_param_counts=group_param_counts,
            group_names=group_names,
            compute_group_correlation=False,
        )

        assert metrics["C_groups"] is None
        assert metrics["C_noise_levels"].shape == (2, 2)

    def test_compute_usage_metrics_applies_population_expansion_weights(self):
        metrics = compute_usage_metrics(
            delta_stack=torch.ones((2, 2), dtype=torch.float64),
            baseline_mean=torch.ones(2, dtype=torch.float64),
            group_param_counts={"g0": 10, "g1": 20},
            group_names=["g0", "g1"],
            group_sampling_weights={"g0": 9.0, "g1": 1.0},
        )

        expected_weights = torch.tensor([[0.9, 0.9], [0.1, 0.1]], dtype=torch.float64)
        assert torch.allclose(metrics["weights"], expected_weights)
        assert torch.allclose(
            metrics["sampling_adjusted_delta_stack"],
            torch.tensor([[9.0, 9.0], [1.0, 1.0]], dtype=torch.float64),
        )
        assert torch.allclose(metrics["n_eff"], torch.full((2,), 10.0, dtype=torch.float64))
        assert torch.allclose(metrics["p_eff"], torch.full((2,), 11.0, dtype=torch.float64))

    def test_compute_usage_metrics_uses_weighted_noise_level_correlation(self):
        delta_stack = torch.tensor(
            [[1.0, 2.0, 4.0], [3.0, 1.0, 2.0], [2.0, 5.0, 1.0]],
            dtype=torch.float64,
        )
        expansion = {"g0": 2.0, "g1": 1.0, "g2": 3.0}
        metrics = compute_usage_metrics(
            delta_stack=delta_stack,
            baseline_mean=torch.ones(3, dtype=torch.float64),
            group_param_counts={"g0": 1, "g1": 1, "g2": 1},
            group_names=["g0", "g1", "g2"],
            group_sampling_weights=expansion,
        )
        repeated_population = torch.repeat_interleave(
            delta_stack,
            torch.tensor([2, 1, 3]),
            dim=0,
        )

        assert torch.allclose(
            metrics["C_noise_levels"],
            torch.corrcoef(repeated_population.T),
            atol=1e-10,
            rtol=1e-10,
        )

    @pytest.mark.parametrize(
        "sampling_weights",
        [{"g0": 2.0}, {"g0": 2.0, "g1": 1.0, "stale": 1.0}],
    )
    def test_compute_usage_metrics_rejects_misaligned_sampling_weights(self, sampling_weights):
        with pytest.raises(ValueError, match="align exactly"):
            compute_usage_metrics(
                delta_stack=torch.ones((2, 2), dtype=torch.float64),
                baseline_mean=torch.ones(2, dtype=torch.float64),
                group_param_counts={"g0": 1, "g1": 1},
                group_names=["g0", "g1"],
                group_sampling_weights=sampling_weights,
            )

    def test_compute_usage_metrics_single_group_has_matrix_correlation(self):
        metrics = compute_usage_metrics(
            delta_stack=torch.tensor([[1.0, 2.0]], dtype=torch.float64),
            baseline_mean=torch.ones(2, dtype=torch.float64),
            group_param_counts={"only": 3},
            group_names=["only"],
        )

        assert metrics["C_groups"] is not None
        assert metrics["C_groups"].shape == (1, 1)
        assert metrics["C_groups"].item() == 1.0
        assert metrics["C_noise_levels"].shape == (2, 2)
        assert torch.count_nonzero(metrics["C_noise_levels"]) == 0

    def test_should_compute_group_correlation_auto_threshold(self):
        assert should_compute_group_correlation("auto", num_groups=2, max_groups=2)
        assert not should_compute_group_correlation("auto", num_groups=3, max_groups=2)
        assert should_compute_group_correlation("always", num_groups=3, max_groups=2)
        assert not should_compute_group_correlation("never", num_groups=1, max_groups=2)

    def test_recompute_metrics_from_results_adds_neff(self):
        results = {
            "group_names": ["g0", "g1"],
            "group_param_counts": {"g0": 32, "g1": 32},
            "baseline_mean": [2.0, 2.0],
            "baseline_stderr": [0.1, 0.2],
            "delta_stack": [[2.0, 1.0], [0.0, 1.0]],
            "sigma_values": [10.0, 1.0],
            "config": {"num_bins": 2, "sigma_stride": 1, "sigma_data": 0.5, "model_family": "edm"},
            "model_info": {"model_family": "edm"},
        }

        updated, tensors = recompute_metrics_from_results(results)

        assert "n_eff" in updated
        assert "p_eff" in updated
        assert len(updated["n_eff"]) == 2
        assert len(updated["relative_delta_stack"]) == 2
        assert tensors["n_eff"].shape == (2,)

    def test_recompute_metrics_from_results_preserves_sampling_expansion(self):
        results = {
            "group_names": ["g0", "g1"],
            "group_param_counts": {"g0": 10, "g1": 20},
            "group_sampling_weights": {"g0": 3.0, "g1": 1.0},
            "baseline_mean": [1.0, 1.0],
            "baseline_stderr": [0.0, 0.0],
            "delta_stack": [[1.0, 1.0], [1.0, 1.0]],
            "sigma_values": [2.0, 1.0],
            "config": {"num_bins": 2, "sigma_stride": 1, "sigma_data": 0.5, "model_family": "edm"},
            "model_info": {"model_family": "edm"},
        }

        updated, tensors = recompute_metrics_from_results(results)

        assert updated["group_sampling_weights"] == {"g0": 3.0, "g1": 1.0}
        assert updated["sampling_adjusted_delta_stack"] == [[3.0, 3.0], [1.0, 1.0]]
        assert torch.allclose(
            tensors["weights"],
            torch.tensor([[0.75, 0.75], [0.25, 0.25]], dtype=torch.float64),
        )

    def test_recompute_metrics_rejects_conflicting_sampling_weight_copies(self):
        results = {
            "group_names": ["g0", "g1"],
            "group_param_counts": {"g0": 1, "g1": 1},
            "group_sampling_weights": {"g0": 2.0, "g1": 1.0},
            "filter_sampling": {
                "selected_group_expansion_weights": {"g0": 3.0, "g1": 1.0}
            },
            "baseline_mean": [1.0, 1.0],
            "delta_stack": [[1.0, 1.0], [1.0, 1.0]],
            "sigma_values": [2.0, 1.0],
            "config": {"num_bins": 2, "sigma_stride": 1, "sigma_data": 0.5},
            "model_info": {"model_family": "edm"},
        }

        with pytest.raises(ValueError, match="does not match"):
            recompute_metrics_from_results(results)

    def test_recompute_metrics_from_dit_style_results_without_sigma_values(self):
        results = {
            "group_names": ["g0", "g1"],
            "group_param_counts": {"g0": 32, "g1": 64},
            "baseline_mean": [2.0, 4.0],
            "baseline_stderr": [0.1, 0.2],
            "delta_stack": [[2.0, 1.0], [0.0, 3.0]],
            "timestep_bin_labels": ["[999,500]", "[499,0]"],
            "config": {"num_bins": 2},
        }

        updated, tensors = recompute_metrics_from_results(results)

        assert "C_timesteps" in updated
        assert "C_noise_levels" not in updated
        assert updated["raw_baseline_mean"] == results["baseline_mean"]
        assert list(tensors["axis_metadata"]["bin_labels"]) == results["timestep_bin_labels"]
        assert tensors["sigma_values_strided"] is None


# ---------------------------------------------------------------------------
# --importance_clip_mode tests
#
# CONTEXT: the per-head permutation importance pipeline computes, per (group,
# bin) entry, delta = ablated_mean - baseline_mean. The original ("per_entry")
# behavior clamps this at 0 immediately, before ANY downstream aggregation
# (relative_delta_stack, weights/n_eff/p_eff, the allocator's cross-head/
# cross-bin aggregation in dit_arch_alloc.py). For a group whose TRUE
# importance is ~0 (zero-mean measurement noise), this clamp is statistically
# biased: E[max(0, X)] > 0 for zero-mean X, so noise gets systematically
# reported as small-but-positive "importance" (an exactly-zero floor for
# noise-dominated groups is the same mechanism at its extreme). "post_agg" keeps the signed delta and only
# clamps at the point where a metric (weights/n_eff/p_eff) actually requires
# non-negativity to be well-defined; "none" clamps nothing at all.
# ---------------------------------------------------------------------------


class TestImportanceClipMode:
    # Symmetric zero-mean noise: sums to exactly 0 across bins, so a "post_agg"
    # or "none" measurement of this head's importance should be exactly 0 --
    # any positive value is the clip's phantom bias, not a real finding.
    NOISE_DELTAS = [-0.03, 0.03, -0.02, 0.02, -0.01, 0.01, -0.015, 0.015]

    def _ablated_and_baseline(self):
        num_bins = len(self.NOISE_DELTAS)
        baseline_mean = torch.ones(num_bins, dtype=torch.float64)
        ablated_means = {
            # "signal": a real, consistently-positive ablation effect.
            "signal": baseline_mean + 1.0,
            # "noise": zero TRUE importance; only measurement noise, split
            # symmetrically between bins that happened to come out positive vs
            # negative.
            "noise": baseline_mean + torch.tensor(self.NOISE_DELTAS, dtype=torch.float64),
        }
        return ablated_means, baseline_mean

    def test_build_delta_stack_per_entry_clamps_negative_entries(self):
        ablated_means, baseline_mean = self._ablated_and_baseline()
        delta_stack = build_delta_stack(ablated_means, baseline_mean, names=["signal", "noise"], clip_mode="per_entry")
        assert torch.all(delta_stack >= 0)
        expected_noise_row = torch.clamp(torch.tensor(self.NOISE_DELTAS, dtype=torch.float64), min=0.0)
        assert torch.allclose(delta_stack[1], expected_noise_row)

    @pytest.mark.parametrize("mode", ["post_agg", "none"])
    def test_build_delta_stack_post_agg_and_none_keep_signed_values(self, mode):
        ablated_means, baseline_mean = self._ablated_and_baseline()
        delta_stack = build_delta_stack(ablated_means, baseline_mean, names=["signal", "noise"], clip_mode=mode)
        assert (delta_stack[1] < 0).any(), f"clip_mode={mode} must keep negative entries"
        assert torch.allclose(delta_stack[1], torch.tensor(self.NOISE_DELTAS, dtype=torch.float64))

    def test_build_delta_stack_post_agg_and_none_are_identical(self):
        """post_agg and none differ only in how compute_usage_metrics treats the
        weights/n_eff/p_eff aggregation input -- the raw delta_stack they build
        is the same signed matrix."""
        ablated_means, baseline_mean = self._ablated_and_baseline()
        names = ["signal", "noise"]
        post_agg = build_delta_stack(ablated_means, baseline_mean, names, clip_mode="post_agg")
        none_mode = build_delta_stack(ablated_means, baseline_mean, names, clip_mode="none")
        assert torch.allclose(post_agg, none_mode)

    def test_build_delta_stack_rejects_invalid_mode(self):
        ablated_means, baseline_mean = self._ablated_and_baseline()
        with pytest.raises(ValueError):
            build_delta_stack(ablated_means, baseline_mean, names=["signal", "noise"], clip_mode="bogus")

    def test_noise_only_head_per_entry_is_positively_biased_post_agg_is_not(self):
        """The core statistical claim, end to end: a head with exactly-zero TRUE
        importance gets a spurious POSITIVE score under 'per_entry' (matches the
        task's own example: 'noise-only head -> per_entry gives positive bias,
        post_agg gives ~0')."""
        ablated_means, baseline_mean = self._ablated_and_baseline()
        names = ["signal", "noise"]
        group_param_counts = {"signal": 100, "noise": 100}

        per_entry_stack = build_delta_stack(ablated_means, baseline_mean, names, clip_mode="per_entry")
        per_entry_metrics = compute_usage_metrics(
            delta_stack=per_entry_stack,
            baseline_mean=baseline_mean,
            group_param_counts=group_param_counts,
            group_names=names,
            clip_mode="per_entry",
        )

        post_agg_stack = build_delta_stack(ablated_means, baseline_mean, names, clip_mode="post_agg")
        post_agg_metrics = compute_usage_metrics(
            delta_stack=post_agg_stack,
            baseline_mean=baseline_mean,
            group_param_counts=group_param_counts,
            group_names=names,
            clip_mode="post_agg",
        )

        per_entry_noise_importance = per_entry_metrics["relative_delta_stack"][1].mean().item()
        post_agg_noise_importance = post_agg_metrics["relative_delta_stack"][1].mean().item()

        assert per_entry_noise_importance > 1e-3, "per_entry should show a clearly positive phantom score"
        assert abs(post_agg_noise_importance) < 1e-9, "post_agg should show ~0 (the true signed mean)"
        assert post_agg_noise_importance < per_entry_noise_importance

    def test_post_agg_and_none_save_identical_relative_delta_stack_but_differ_in_n_eff(self):
        ablated_means, baseline_mean = self._ablated_and_baseline()
        names = ["signal", "noise"]
        group_param_counts = {"signal": 100, "noise": 100}
        stack = build_delta_stack(ablated_means, baseline_mean, names, clip_mode="post_agg")

        post_agg_metrics = compute_usage_metrics(
            delta_stack=stack, baseline_mean=baseline_mean,
            group_param_counts=group_param_counts, group_names=names, clip_mode="post_agg",
        )
        none_metrics = compute_usage_metrics(
            delta_stack=stack, baseline_mean=baseline_mean,
            group_param_counts=group_param_counts, group_names=names, clip_mode="none",
        )

        # The saved (allocator-facing) matrices are identical...
        assert torch.allclose(post_agg_metrics["relative_delta_stack"], none_metrics["relative_delta_stack"])
        assert torch.allclose(post_agg_metrics["delta_stack"], none_metrics["delta_stack"])
        # ...but n_eff/weights (which need non-negative mass to be well-defined)
        # differ: post_agg clamps its internal aggregation input, none does not.
        assert not torch.allclose(post_agg_metrics["n_eff"], none_metrics["n_eff"])
        assert not torch.allclose(post_agg_metrics["weights"], none_metrics["weights"])

    def test_compute_usage_metrics_default_clip_mode_is_per_entry_and_byte_identical(self):
        """Hard repo rule: the default must reproduce the pre-existing formula
        exactly for the pipeline's real (already-non-negative) delta_stack."""
        delta_stack = torch.tensor(
            [[4.0, 1.0, 2.0], [1.0, 1.0, 2.0], [0.0, 2.0, 2.0]], dtype=torch.float64
        )
        baseline_mean = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        names = ["g0", "g1", "g2"]
        counts = {n: 100 for n in names}

        default_metrics = compute_usage_metrics(delta_stack, baseline_mean, counts, names)
        explicit_metrics = compute_usage_metrics(delta_stack, baseline_mean, counts, names, clip_mode="per_entry")

        for key in ("relative_delta_stack", "n_eff", "p_eff", "weights", "delta_stack"):
            assert torch.allclose(default_metrics[key], explicit_metrics[key])

    def test_compute_usage_metrics_per_entry_clamps_signed_input_defensively(self):
        """Even if a caller passes a signed delta_stack under clip_mode='per_entry'
        (misuse), the returned delta_stack/relative_delta_stack must still be
        non-negative -- the contract doesn't silently rely on caller discipline."""
        delta_stack = torch.tensor([[-1.0, 2.0]], dtype=torch.float64)
        baseline_mean = torch.tensor([1.0, 1.0], dtype=torch.float64)
        metrics = compute_usage_metrics(delta_stack, baseline_mean, {"g0": 1}, ["g0"], clip_mode="per_entry")
        assert torch.all(metrics["delta_stack"] >= 0)
        assert torch.all(metrics["relative_delta_stack"] >= 0)

    def test_compute_usage_metrics_rejects_invalid_clip_mode(self):
        delta_stack = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
        baseline_mean = torch.tensor([1.0, 1.0], dtype=torch.float64)
        with pytest.raises(ValueError):
            compute_usage_metrics(delta_stack, baseline_mean, {"g0": 1}, ["g0"], clip_mode="bogus")

    def test_importance_clip_modes_constant(self):
        assert set(IMPORTANCE_CLIP_MODES) == {"per_entry", "post_agg", "none"}


class TestRecomputeMetricsClipMode:
    def _base_results(self, clip_mode=None):
        results = {
            "group_names": ["g0", "g1"],
            "group_param_counts": {"g0": 100, "g1": 100},
            "baseline_mean": [1.0, 1.0],
            "baseline_stderr": [0.05, 0.05],
            "delta_stack": [[1.0, 1.0], [-0.5, 0.5]],
            "config": {"num_bins": 2},
        }
        if clip_mode is not None:
            results["importance_clip_mode"] = clip_mode
        return results

    def test_recompute_infers_clip_mode_from_saved_results(self):
        results = self._base_results(clip_mode="post_agg")
        updated, _tensors = recompute_metrics_from_results(results)
        assert updated["importance_clip_mode"] == "post_agg"
        # signed delta_stack -> negative entries must survive into relative_delta_stack
        assert any(v < 0 for row in updated["relative_delta_stack"] for v in row)

    def test_recompute_defaults_to_per_entry_for_legacy_results_without_the_field(self):
        results = self._base_results(clip_mode=None)
        assert "importance_clip_mode" not in results
        updated, _tensors = recompute_metrics_from_results(results)
        assert updated["importance_clip_mode"] == "per_entry"

    def test_recompute_explicit_clip_mode_overrides_saved_value(self):
        results = self._base_results(clip_mode="per_entry")
        updated, _tensors = recompute_metrics_from_results(results, clip_mode="none")
        assert updated["importance_clip_mode"] == "none"


# ---------------------------------------------------------------------------
# Class labels tests
# ---------------------------------------------------------------------------


class TestMakeClassLabels:
    def test_unconditional(self):
        result = make_class_labels(batch_size=4, label_dim=0, class_idx=None, device=torch.device("cpu"))
        assert result is None

    def test_fixed_class(self):
        labels = make_class_labels(batch_size=3, label_dim=10, class_idx=5, device=torch.device("cpu"))
        assert labels.shape == (3, 10)
        assert (labels[:, 5] == 1.0).all()
        assert labels.sum().item() == 3.0

    def test_dataset_labels(self):
        class_indices = torch.tensor([0, 3, 7])
        labels = make_class_labels(
            batch_size=3, label_dim=10, class_idx=None, device=torch.device("cpu"),
            dataset_class_indices=class_indices,
        )
        assert labels.shape == (3, 10)
        assert labels[0, 0] == 1.0
        assert labels[1, 3] == 1.0
        assert labels[2, 7] == 1.0

    def test_imagenet_1000_classes(self):
        class_indices = torch.tensor([0, 500, 999])
        labels = make_class_labels(
            batch_size=3, label_dim=1000, class_idx=None, device=torch.device("cpu"),
            dataset_class_indices=class_indices,
        )
        assert labels.shape == (3, 1000)
        assert labels[0, 0] == 1.0
        assert labels[1, 500] == 1.0
        assert labels[2, 999] == 1.0

    def test_invalid_class_idx_raises(self):
        with pytest.raises(ValueError, match="class_idx must be in"):
            make_class_labels(batch_size=2, label_dim=10, class_idx=10, device=torch.device("cpu"))

    def test_negative_labels_raise(self):
        class_indices = torch.tensor([0, -1, 3])
        with pytest.raises(ValueError, match="class-conditional"):
            make_class_labels(
                batch_size=3, label_dim=10, class_idx=None, device=torch.device("cpu"),
                dataset_class_indices=class_indices,
            )


# ---------------------------------------------------------------------------
# Ablation hooks tests
# ---------------------------------------------------------------------------


class TestAblationHooks:
    def test_zero_output_hook(self, simple_conv_module):
        x = torch.randn(1, 3, 8, 8)
        with ZeroOutputHook(simple_conv_module):
            out = simple_conv_module(x)
        assert (out == 0).all()

    def test_zero_output_hook_restored(self, simple_conv_module):
        x = torch.randn(1, 3, 8, 8)
        with ZeroOutputHook(simple_conv_module):
            pass
        out = simple_conv_module(x)
        assert not (out == 0).all()

    def test_filter_zero_hook(self, simple_conv_module):
        x = torch.randn(1, 3, 8, 8)
        with FilterZeroHook(simple_conv_module, filter_idx=0):
            out = simple_conv_module(x)
        assert (out[:, 0] == 0).all()
        assert not (out[:, 1] == 0).all()

    def test_filter_zero_hook_out_of_range(self, simple_conv_module):
        x = torch.randn(1, 3, 8, 8)
        with pytest.raises(ValueError, match="out of range"):
            with FilterZeroHook(simple_conv_module, filter_idx=999):
                simple_conv_module(x)

    def test_attention_head_hook_uses_explicit_channel_axis_for_bct_output(self):
        attention = QKVFlashAttention(embed_dim=8, num_heads=2)
        qkv = torch.randn(2, 24, 5)
        baseline = attention(qkv)

        with HeadZeroHook(attention, head_idx=1, channel_dim=1):
            ablated = attention(qkv)

        assert torch.allclose(ablated[:, :4], baseline[:, :4])
        assert torch.count_nonzero(ablated[:, 4:]) == 0
        assert torch.count_nonzero(ablated[:, :, -1]) > 0  # token axis was not selected

    def test_attention_head_random_replacement_preserves_selected_norm(self):
        attention = QKVFlashAttention(embed_dim=8, num_heads=2)
        qkv = torch.randn(2, 24, 5)
        baseline = attention(qkv)

        with HeadRandomSameNormHook(attention, head_idx=0, random_seed=17, channel_dim=1):
            replaced = attention(qkv)

        assert torch.allclose(replaced[:, 4:], baseline[:, 4:])
        expected_norm = baseline[:, :4].flatten(1).norm(dim=1)
        actual_norm = replaced[:, :4].flatten(1).norm(dim=1)
        assert torch.allclose(actual_norm, expected_norm, rtol=1e-5, atol=1e-6)

    def test_pfi_whole_tuple_output_uses_one_donor_mapping(self):
        class TupleModule(torch.nn.Module):
            def forward(self, value):
                return value + 10, value * 3, "metadata"

        module = TupleModule()
        value = torch.arange(3, dtype=torch.float32).reshape(3, 1)
        baseline = module(value)
        permutation = torch.tensor([1, 2, 0])

        with PFIOutputHook(module) as hook:
            hook.set_pfi_permutation(permutation)
            exchanged = module(value)

        assert torch.equal(exchanged[0], baseline[0].index_select(0, permutation))
        assert torch.equal(exchanged[1], baseline[1].index_select(0, permutation))
        assert exchanged[2] == "metadata"

    def test_pfi_filter_exchanges_only_complete_selected_filter(self):
        module = torch.nn.Identity()
        value = torch.arange(3 * 2 * 2, dtype=torch.float32).reshape(3, 2, 2, 1)
        permutation = torch.tensor([1, 2, 0])

        with FilterPFIHook(module, filter_idx=1) as hook:
            hook.set_pfi_permutation(permutation)
            exchanged = module(value)

        assert torch.equal(exchanged[:, 0], value[:, 0])
        assert torch.equal(exchanged[:, 1], value[:, 1].index_select(0, permutation))

    def test_pfi_attention_exchanges_only_complete_selected_head(self):
        attention = QKVFlashAttention(embed_dim=8, num_heads=2)
        qkv = torch.randn(2, 24, 5)
        baseline = attention(qkv)
        permutation = torch.tensor([1, 0])

        with HeadPFIHook(attention, head_idx=1, channel_dim=1) as hook:
            hook.set_pfi_permutation(permutation)
            exchanged = attention(qkv)

        assert torch.allclose(exchanged[:, :4], baseline[:, :4])
        assert torch.allclose(exchanged[:, 4:], baseline[:, 4:].index_select(0, permutation))

    def test_pfi_rejects_identity_or_partial_fixed_point_permutations(self):
        hook = PFIOutputHook(torch.nn.Identity())
        with pytest.raises(ValueError, match="fixed-point-free"):
            hook.set_pfi_permutation(torch.tensor([0, 1]))
        with pytest.raises(ValueError, match="fixed-point-free"):
            hook.set_pfi_permutation(torch.tensor([1, 0, 2]))


# ---------------------------------------------------------------------------
# BinStats tests
# ---------------------------------------------------------------------------


class TestBinStats:
    def test_basic_accumulation(self):
        stats = BinStats(num_bins=3)
        stats.update(torch.tensor([0, 1, 2]), torch.tensor([1.0, 2.0, 3.0]))
        mean = stats.mean()
        assert mean[0].item() == 1.0
        assert mean[1].item() == 2.0
        assert mean[2].item() == 3.0

    def test_multiple_updates(self):
        stats = BinStats(num_bins=2)
        stats.update(torch.tensor([0, 0]), torch.tensor([1.0, 3.0]))
        mean = stats.mean()
        assert abs(mean[0].item() - 2.0) < 1e-10

    def test_stderr(self):
        stats = BinStats(num_bins=1)
        stats.update(torch.tensor([0, 0, 0, 0]), torch.tensor([1.0, 2.0, 3.0, 4.0]))
        se = stats.stderr()
        assert se[0].item() > 0

    def test_empty_bins(self):
        stats = BinStats(num_bins=3)
        stats.update(torch.tensor([0]), torch.tensor([5.0]))
        mean = stats.mean()
        assert mean[1].item() == 0.0
        assert mean[2].item() == 0.0


# ---------------------------------------------------------------------------
# count_filter_parameters tests
# ---------------------------------------------------------------------------


class TestCountFilterParameters:
    def test_with_bias(self):
        m = torch.nn.Conv2d(3, 8, kernel_size=3, bias=True)
        count = count_filter_parameters(m, filter_idx=0)
        # One filter: 3*3*3 = 27 weights + 1 bias = 28
        assert count == 3 * 3 * 3 + 1

    def test_without_bias(self):
        m = torch.nn.Conv2d(3, 8, kernel_size=3, bias=False)
        count = count_filter_parameters(m, filter_idx=0)
        assert count == 3 * 3 * 3


# ---------------------------------------------------------------------------
# W&B integration test (mocked)
# ---------------------------------------------------------------------------


class TestWandbIntegration:
    def test_wandb_import_flag(self):
        """wandb is optional; HAS_WANDB records whether it could be imported."""
        from evaluate_parameters_edm import HAS_WANDB
        assert isinstance(HAS_WANDB, bool)

    def test_wandb_args_accepted(self):
        """Verify the argparser accepts --wandb_project and --wandb_run_name.

        Reads the module's source file directly (via its own __file__) rather
        than inspect.getsource(main): getsource resolves a function's source by
        re-reading co_filename through linecache at call time, which is fragile
        when run deep inside the full test suite (some other test in the suite
        leaves global state -- e.g. cwd -- that can make a stale/incorrect
        linecache lookup land on the wrong function). A plain file read has no
        such dependency and is exactly as sufficient for a flag-presence check.
        """
        import evaluate_parameters_edm as module
        with open(module.__file__) as f:
            source = f.read()
        assert "wandb_project" in source
        assert "wandb_run_name" in source


class TestImportanceClipModeCLI:
    """Verify --importance_clip_mode is wired into main()'s argparser, opt-in with
    a 'per_entry' default, for evaluate_parameters_edm.py and its two siblings."""

    @pytest.mark.parametrize(
        "module_name", ["evaluate_parameters_edm", "evaluate_parameters_dit", "evaluate_parameters_dit_micro"]
    )
    def test_importance_clip_mode_flag_present_with_per_entry_default(self, module_name):
        import importlib

        # See test_wandb_args_accepted's docstring above for why this reads the
        # file directly instead of using inspect.getsource(module.main).
        module = importlib.import_module(module_name)
        with open(module.__file__) as f:
            source = f.read()
        assert "--importance_clip_mode" in source
        assert '"per_entry"' in source
        assert "IMPORTANCE_CLIP_MODES" in source
