import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import compare_diffwave_filter_stability as stability


def write_result(root: Path, seed: int, *, perturb: float = 0.0) -> Path:
    output = root / f"seed{seed}"
    output.mkdir()
    groups = [f"layer.filter_{index}" for index in range(256)]
    signed = np.arange(1, 256 * 2 + 1, dtype=np.float64).reshape(256, 2) + perturb
    relative = np.maximum(signed, 0.0) + 1.0
    payload = {
        "config": {"grouping": "per_filter", "seed": seed, "pfi_seed": seed},
        "group_names": groups,
        "group_module_paths": {name: "layer" for name in groups},
        "group_filter_indices": {name: index for index, name in enumerate(groups)},
        "group_stage_keys": {name: "residual_block_00" for name in groups},
        "timestep_bin_labels": ["high", "low"],
        "full_group_count": 37377,
        "signed_delta_stack": signed.tolist(),
        "relative_delta_stack": relative.tolist(),
        "signed_delta_positive_fraction": [1.0, 1.0],
        "signed_delta_negative_fraction": [0.0, 0.0],
        "per_filter_aggregates": {
            "module": {"names": ["layer"], "signed_delta_stack": signed.sum(axis=0, keepdims=True).tolist()},
            "stage": {"names": ["residual_block_00"], "signed_delta_stack": signed.sum(axis=0, keepdims=True).tolist()},
        },
        "profile_fingerprint": {
            "teacher": {"checkpoint_sha256": "a" * 64},
            "dataset": {"population_fingerprint": "b" * 64},
            "groups": {"full_catalog_sha256": "c" * 64},
        },
    }
    path = output / "results.json"
    path.write_text(json.dumps(payload))
    (output / "_SUCCESS.json").write_text("{}")
    return path


def test_compare_three_seed_results(tmp_path):
    paths = [write_result(tmp_path, seed, perturb=seed * 0.1) for seed in range(3)]
    report = stability.compare_results(paths)
    assert report["passed"] is True
    assert report["group_count"] == 256
    assert len(report["pairs"]) == 3
    assert all(pair["signed_delta_sign_agreement"] == 1.0 for pair in report["pairs"])


def test_rejects_incomplete_or_wrong_group_count(tmp_path):
    paths = [write_result(tmp_path, seed) for seed in range(3)]
    (paths[0].parent / "_SUCCESS.json").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        stability.compare_results(paths)
