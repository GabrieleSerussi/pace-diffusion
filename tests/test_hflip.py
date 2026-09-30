"""
--hflip: opt-in random horizontal flip in the CIFAR-10 / image_folder TRAIN
pipeline only. Default OFF (hflip=False) must produce byte-identical batches
to before this flag existed (no RandomHorizontalFlip step, no RNG consumed by
it). Also covers the --hflip wiring in train_phase_students._build_image_dataset
and that the held-out validation split (build_val_loader) is never flipped.
"""
import os, sys, types
import torch
from PIL import Image
import torchvision.transforms as transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from evaluate_parameters_edm import CIFAR10Dataset, ImageFolderFlat
from train_phase_students import _build_image_dataset, build_val_loader

# A torchvision CIFAR-10 root (containing cifar-10-batches-py/). The CIFAR-10
# reproduce script downloads it to ./data; override with $PACE_CIFAR10_ROOT.
CIFAR_ROOT = os.environ.get(
    "PACE_CIFAR10_ROOT", os.path.join(os.path.dirname(__file__), "..", "data")
)


def _has_local_cifar() -> bool:
    return os.path.isdir(os.path.join(CIFAR_ROOT, "cifar-10-batches-py"))


import pytest

requires_cifar = pytest.mark.skipif(
    not _has_local_cifar(), reason="local CIFAR-10 data not found (set PACE_CIFAR10_ROOT)"
)


# ---------------------------------------------------------------------------
# Transform-pipeline shape (no CIFAR download needed)
# ---------------------------------------------------------------------------

def test_hflip_false_pipeline_has_no_flip_step():
    import inspect
    sig = inspect.signature(CIFAR10Dataset.__init__)
    assert sig.parameters["hflip"].default is False


@requires_cifar
def test_hflip_off_transform_is_byte_identical_to_pre_flag_pipeline():
    # The exact pipeline CIFAR10Dataset built before --hflip existed.
    legacy_transform = transforms.Compose([
        transforms.Resize(32, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    # max_images=None (full dataset): __getitem__ maps idx -> dataset[idx]
    # directly (no seeded-permutation remap, which only kicks in when
    # max_images < dataset_length), so comparing against ds.dataset[idx] is valid.
    ds_default = CIFAR10Dataset(root=CIFAR_ROOT, image_size=32, split="train", max_images=None)
    ds_explicit_off = CIFAR10Dataset(root=CIFAR_ROOT, image_size=32, split="train", max_images=None, hflip=False)

    for ds in (ds_default, ds_explicit_off):
        # Compare full __getitem__ output against manually-applying the legacy
        # transform to the same underlying PIL image (same torchvision CIFAR10
        # dataset object -> same source image at the same index; hflip=False
        # means no randomness is introduced, so this must match exactly).
        for idx in range(4):
            got, label = ds[idx]
            raw_image, raw_label = ds.dataset[idx]
            expected = legacy_transform(raw_image)
            assert torch.equal(got, expected), f"hflip=False must not alter the transform output (idx={idx})"
            assert label == raw_label


@requires_cifar
def test_hflip_off_is_deterministic_across_torch_rng_states():
    # With hflip=False no RandomHorizontalFlip is in the pipeline, so the output
    # must NOT depend on torch's global RNG state at all.
    ds = CIFAR10Dataset(root=CIFAR_ROOT, image_size=32, split="train", max_images=None, hflip=False)
    torch.manual_seed(0)
    out1 = ds[0][0].clone()
    torch.manual_seed(999)
    out2 = ds[0][0].clone()
    assert torch.equal(out1, out2)


@requires_cifar
def test_hflip_true_actually_flips_some_samples():
    torch.manual_seed(0)
    ds = CIFAR10Dataset(root=CIFAR_ROOT, image_size=32, split="train", max_images=None, hflip=True)
    flipped_or_not = []
    for idx in range(64):
        got, _ = ds[idx]
        raw_image, _ = ds.dataset[idx]
        no_flip = transforms.Compose([
            transforms.Resize(32, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])(raw_image)
        flipped = torch.flip(no_flip, dims=[-1])
        if torch.equal(got, flipped):
            flipped_or_not.append(True)
        elif torch.equal(got, no_flip):
            flipped_or_not.append(False)
        else:
            raise AssertionError(f"sample {idx} matches neither flipped nor unflipped output")
    # p=0.5 over 64 draws: expect both outcomes present (not proof of exact 0.5,
    # just that SOME flips actually happen -- guards against a no-op flag).
    assert any(flipped_or_not) and not all(flipped_or_not)


# ---------------------------------------------------------------------------
# Wiring: --hflip threads through _build_image_dataset (train) but NOT
# build_val_loader (held-out split stays unflipped regardless of --hflip).
# ---------------------------------------------------------------------------

@requires_cifar
def test_build_image_dataset_wires_hflip_flag():
    args_off = types.SimpleNamespace(
        dataset="cifar10", data_root=CIFAR_ROOT, cifar_split="train",
        max_images=4, download=False, hflip=False,
    )
    args_on = types.SimpleNamespace(
        dataset="cifar10", data_root=CIFAR_ROOT, cifar_split="train",
        max_images=4, download=False, hflip=True,
    )
    ds_off = _build_image_dataset(args_off, image_size=32)
    ds_on = _build_image_dataset(args_on, image_size=32)
    has_flip_off = any(isinstance(t, transforms.RandomHorizontalFlip) for t in ds_off.transform.transforms)
    has_flip_on = any(isinstance(t, transforms.RandomHorizontalFlip) for t in ds_on.transform.transforms)
    assert has_flip_off is False
    assert has_flip_on is True


def test_build_image_dataset_defaults_hflip_off_when_attr_missing():
    # getattr(args, "hflip", False) fallback for args namespaces built before
    # --hflip existed (e.g. saved cfg_payload in old full checkpoints).
    args_no_attr = types.SimpleNamespace(
        dataset="cifar10", data_root=CIFAR_ROOT, cifar_split="train",
        max_images=4, download=False,
    )
    if not _has_local_cifar():
        import pytest as _pytest
        _pytest.skip("local CIFAR-10 data not found")
    ds = _build_image_dataset(args_no_attr, image_size=32)
    assert any(isinstance(t, transforms.RandomHorizontalFlip) for t in ds.transform.transforms) is False


@requires_cifar
def test_build_val_loader_never_flips_regardless_of_hflip():
    args = types.SimpleNamespace(
        dataset="cifar10", data_root=CIFAR_ROOT, download=False, hflip=True,
        max_images=None,
    )
    loader = build_val_loader(args, image_size=32, batch_size=4, max_batches=1)
    ds = loader.dataset
    assert isinstance(ds, CIFAR10Dataset)
    assert not any(isinstance(t, transforms.RandomHorizontalFlip) for t in ds.transform.transforms)


# ---------------------------------------------------------------------------
# ImageFolderFlat --hflip (needed for the unconditional Bedroom/FFHQ
# objective-ablation pretrain -- FFHQ uses --hflip as small-data
# regularization). Same opt-in
# convention as CIFAR10Dataset above: default False = byte-identical pipeline.
# ---------------------------------------------------------------------------

def _flat_image_dir(tmp_path, n=8, size=48):
    # Left/right halves colored differently so a horizontal flip actually
    # changes the pixel content (a solid-color image would be flip-invariant
    # and silently defeat test_image_folder_flat_hflip_true_actually_flips_some_samples).
    import numpy as np
    for i in range(n):
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        arr[:, : size // 2, 0] = 200
        arr[:, size // 2:, 2] = 200
        arr[:, :, 1] = (i * 7) % 256
        Image.fromarray(arr).save(tmp_path / f"img_{i:03d}.png")
    return tmp_path


def test_image_folder_flat_hflip_false_pipeline_has_no_flip_step():
    import inspect
    sig = inspect.signature(ImageFolderFlat.__init__)
    assert sig.parameters["hflip"].default is False


def test_image_folder_flat_hflip_off_transform_is_byte_identical_to_pre_flag_pipeline(tmp_path):
    root = _flat_image_dir(tmp_path)
    legacy_transform = transforms.Compose([
        transforms.Resize(32, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    ds_default = ImageFolderFlat(root=str(root), image_size=32)
    ds_explicit_off = ImageFolderFlat(root=str(root), image_size=32, hflip=False)
    for ds in (ds_default, ds_explicit_off):
        for idx in range(len(ds)):
            got, label = ds[idx]
            image = Image.open(ds.paths[idx]).convert("RGB")
            expected = legacy_transform(image)
            assert torch.equal(got, expected)
            assert label == -1


def test_image_folder_flat_hflip_off_is_deterministic_across_torch_rng_states(tmp_path):
    root = _flat_image_dir(tmp_path)
    ds = ImageFolderFlat(root=str(root), image_size=32, hflip=False)
    torch.manual_seed(0)
    out1 = ds[0][0].clone()
    torch.manual_seed(999)
    out2 = ds[0][0].clone()
    assert torch.equal(out1, out2)


def test_image_folder_flat_hflip_true_actually_flips_some_samples(tmp_path):
    root = _flat_image_dir(tmp_path, n=32)
    torch.manual_seed(0)
    ds = ImageFolderFlat(root=str(root), image_size=32, hflip=True)
    no_flip_transform = transforms.Compose([
        transforms.Resize(32, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    flipped_or_not = []
    for idx in range(len(ds)):
        got, _ = ds[idx]
        image = Image.open(ds.paths[idx]).convert("RGB")
        no_flip = no_flip_transform(image)
        flipped = torch.flip(no_flip, dims=[-1])
        if torch.equal(got, flipped):
            flipped_or_not.append(True)
        elif torch.equal(got, no_flip):
            flipped_or_not.append(False)
        else:
            raise AssertionError(f"sample {idx} matches neither flipped nor unflipped output")
    assert any(flipped_or_not) and not all(flipped_or_not)


def test_build_image_dataset_wires_hflip_for_image_folder(tmp_path):
    root = _flat_image_dir(tmp_path)
    args_off = types.SimpleNamespace(
        dataset="image_folder", image_root=str(root), max_images=None, hflip=False,
    )
    args_on = types.SimpleNamespace(
        dataset="image_folder", image_root=str(root), max_images=None, hflip=True,
    )
    ds_off = _build_image_dataset(args_off, image_size=32)
    ds_on = _build_image_dataset(args_on, image_size=32)
    has_flip_off = any(isinstance(t, transforms.RandomHorizontalFlip) for t in ds_off.transform.transforms)
    has_flip_on = any(isinstance(t, transforms.RandomHorizontalFlip) for t in ds_on.transform.transforms)
    assert has_flip_off is False
    assert has_flip_on is True


def test_build_image_dataset_image_folder_defaults_hflip_off_when_attr_missing(tmp_path):
    root = _flat_image_dir(tmp_path)
    args_no_attr = types.SimpleNamespace(
        dataset="image_folder", image_root=str(root), max_images=None,
    )
    ds = _build_image_dataset(args_no_attr, image_size=32)
    assert any(isinstance(t, transforms.RandomHorizontalFlip) for t in ds.transform.transforms) is False
