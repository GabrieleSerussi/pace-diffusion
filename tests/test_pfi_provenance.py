import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np

from pace.filter_sampling import (
    aligned_group_expansion_weights,
    group_expansion_weight_map,
    normalize_filter_sampling,
)
from pace.profile_provenance import (
    LEGACY_ABLATION_PROTOCOL_ID,
    provenance_from_artifact,
    provenance_from_results_path,
    validate_matching_source_profile,
)
from scripts.optimize_timestep_grouping import _compute_pairwise_similarity_matrix


REPO_ROOT = Path(__file__).resolve().parents[1]
PFI_PROTOCOL = {
    "protocol_id": "batch_local_exact_sigma_pfi_v1",
    "ablation_mode": "pfi",
    "donor_scope": "batch_local",
    "sigma_pairing": "exact",
}
PFI_FINGERPRINT = {
    "dataset_manifest_sha256": "d" * 64,
    "network_sha256": "n" * 64,
    "pfi_seed": 0,
}
PFI_FINGERPRINT_SHA256 = hashlib.sha256(
    json.dumps(PFI_FINGERPRINT, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


def _canonical_name_digest(names):
    payload = json.dumps(sorted(names), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


SAMPLED_GROUP_NAMES = ["module0.filter_0", "module1.filter_0", "module2.filter_0"]
SAMPLED_POPULATION_NAMES = [
    "module0.filter_0",
    "module1.filter_0",
    "module1.filter_1",
    "module2.filter_0",
    "module2.filter_1",
    "module2.filter_2",
]

FILTER_SAMPLING = {
    "format": "diffdist_edm_filter_sampling_protocol_v1",
    "protocol_id": "per_module_hash_stratified_filter_sampling_v1",
    "mode": "stratified_module",
    "seed": 0,
    "filters_per_module": 1,
    "population_group_count": 6,
    "selected_group_count": 3,
    "population_module_count": 3,
    "selected_module_count": 3,
    "module_counts": {
        "module0": {
            "population_filter_count": 1,
            "selected_filter_count": 1,
            "inclusion_probability": 1.0,
            "expansion_weight": 1.0,
        },
        "module1": {
            "population_filter_count": 2,
            "selected_filter_count": 1,
            "inclusion_probability": 0.5,
            "expansion_weight": 2.0,
        },
        "module2": {
            "population_filter_count": 3,
            "selected_filter_count": 1,
            "inclusion_probability": 1.0 / 3.0,
            "expansion_weight": 3.0,
        },
    },
    "selected_group_names": SAMPLED_GROUP_NAMES,
    "selected_group_inclusion_probabilities": {
        SAMPLED_GROUP_NAMES[0]: 1.0,
        SAMPLED_GROUP_NAMES[1]: 0.5,
        SAMPLED_GROUP_NAMES[2]: 1.0 / 3.0,
    },
    "selected_group_expansion_weights": {
        SAMPLED_GROUP_NAMES[0]: 1.0,
        SAMPLED_GROUP_NAMES[1]: 2.0,
        SAMPLED_GROUP_NAMES[2]: 3.0,
    },
    "selection_sha256": _canonical_name_digest(SAMPLED_GROUP_NAMES),
    "population_sha256": _canonical_name_digest(SAMPLED_POPULATION_NAMES),
}


def test_filter_sampling_schema_rejects_malformed_or_unversioned_records():
    assert normalize_filter_sampling({"filter_sampling": FILTER_SAMPLING}) == FILTER_SAMPLING

    missing_format = dict(FILTER_SAMPLING)
    missing_format.pop("format")
    with pytest.raises(ValueError, match="unsupported filter_sampling format"):
        normalize_filter_sampling({"filter_sampling": missing_format})

    wrong_mode = dict(FILTER_SAMPLING, mode="per_module_hash_stratified")
    with pytest.raises(ValueError, match="filter_sampling.mode"):
        normalize_filter_sampling({"filter_sampling": wrong_mode})

    wrong_digest = dict(FILTER_SAMPLING, selection_sha256="0" * 64)
    with pytest.raises(ValueError, match="does not match selected_group_names"):
        normalize_filter_sampling({"filter_sampling": wrong_digest})

    with pytest.raises(ValueError, match="requires a versioned filter_sampling"):
        group_expansion_weight_map({"group_sampling_weights": {"g0": 2.0}})


def test_compact_exhaustive_sampling_record_aligns_unit_weights():
    names = ["m.filter_1", "m.filter_0"]
    digest = _canonical_name_digest(names)
    record = {
        "format": "diffdist_edm_filter_sampling_protocol_v1",
        "protocol_id": "per_filter_exhaustive_v1",
        "mode": "exhaustive",
        "seed": 0,
        "filters_per_module": None,
        "population_group_count": 2,
        "selected_group_count": 2,
        "population_module_count": 1,
        "selected_module_count": 1,
        "module_counts": {
            "m": {
                "population_filter_count": 2,
                "selected_filter_count": 2,
                "inclusion_probability": 1.0,
                "expansion_weight": 1.0,
            }
        },
        "selection_sha256": digest,
        "population_sha256": digest,
    }
    payload = {"filter_sampling": record, "group_names": names}
    assert aligned_group_expansion_weights(payload) == [1.0, 1.0]

    with pytest.raises(ValueError, match="selection_sha256"):
        aligned_group_expansion_weights(payload, ["m.filter_0", "m.filter_2"])


def _profile_payload() -> dict:
    return {
        "num_parameters": 8,
        "group_names": ["g0", "g1"],
        "group_param_counts": {"g0": 3, "g1": 5},
        "n_eff": [1.0, 2.0, 3.0, 4.0],
        "relative_delta_stack": [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
        ],
        "ablation_protocol": PFI_PROTOCOL,
        "profile_fingerprint": PFI_FINGERPRINT,
        "profile_fingerprint_sha256": PFI_FINGERPRINT_SHA256,
    }


def test_profile_provenance_preserves_versioned_records_and_marks_legacy(tmp_path):
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(_profile_payload()))

    provenance = provenance_from_results_path(results_path)

    assert provenance["ablation_protocol"] == PFI_PROTOCOL
    assert provenance["source_profile"]["format"] == "diffdist_edm_source_profile_v1"
    assert provenance["source_profile"]["results_json_path"] == str(results_path.resolve())
    assert provenance["source_profile"]["results_sha256"] == hashlib.sha256(
        results_path.read_bytes()
    ).hexdigest()
    assert provenance["source_profile"]["profile_fingerprint"] == PFI_FINGERPRINT
    assert provenance["source_profile"]["profile_fingerprint_sha256"] == PFI_FINGERPRINT_SHA256

    legacy_path = tmp_path / "legacy-results.json"
    legacy_path.write_text(json.dumps({"ablation_info": {"mode": "random_same_norm"}}))
    legacy = provenance_from_results_path(legacy_path)
    assert legacy["ablation_protocol"] == {"protocol_id": LEGACY_ABLATION_PROTOCOL_ID}
    assert provenance_from_artifact({"ablation_info": {"mode": "zero"}})["ablation_protocol"] == {
        "protocol_id": LEGACY_ABLATION_PROTOCOL_ID
    }


def test_grouping_and_allocation_propagate_exact_pfi_provenance(tmp_path):
    profile_dir = tmp_path / "profile"
    grouping_path = profile_dir / "grouping" / "timestep_grouping.json"
    profile_dir.mkdir()
    results_path = profile_dir / "results.json"
    results_path.write_text(json.dumps(_profile_payload()))

    subprocess.run(
        [
            sys.executable,
            "scripts/optimize_timestep_grouping.py",
            "--matrix",
            str(results_path),
            "--matrix_key",
            "relative_delta_stack",
            "--builtin_cost",
            "matrix_correlation_cross_penalty",
            "--num_blocks",
            "2",
            "--output",
            str(grouping_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    grouping = json.loads(grouping_path.read_text())
    expected = provenance_from_results_path(results_path)
    assert grouping["source_profile"] == expected["source_profile"]
    assert grouping["ablation_protocol"] == PFI_PROTOCOL

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/dry_run_capacity_allocation.py",
            "--eval-output-dir",
            str(profile_dir),
            "--student-variant",
            "blockwise_capacity",
            # Legacy n_eff scores: the fixture carries no delta_stack.
            "--allocation-metric",
            "auto",
            "--score-reduction",
            "mean",
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    allocation = json.loads(completed.stdout)
    summary = json.loads((profile_dir / "allocation_results" / "summary.json").read_text())
    assert allocation["source_profile"] == expected["source_profile"]
    assert allocation["ablation_protocol"] == PFI_PROTOCOL
    assert summary["source_profile"] == expected["source_profile"]
    assert summary["ablation_protocol"] == PFI_PROTOCOL


def test_versioned_source_profile_mismatch_is_rejected():
    expected = {
        "results_sha256": "a" * 64,
        "profile_fingerprint_sha256": "b" * 64,
    }
    observed = {
        "source_profile": {
            "results_sha256": "c" * 64,
            "profile_fingerprint_sha256": "b" * 64,
        }
    }
    with pytest.raises(ValueError, match="source profile SHA-256 mismatch"):
        validate_matching_source_profile(expected, observed, context="fixture")


def test_weighted_timestep_correlation_matches_expanded_population():
    selected = np.asarray(
        [
            [1.0, 4.0, 2.0, 8.0],
            [3.0, 1.0, 5.0, 2.0],
            [7.0, 2.0, 6.0, 3.0],
        ],
        dtype=np.float64,
    )
    expansion = np.asarray([1.0, 2.0, 3.0])
    population = np.repeat(selected, expansion.astype(int), axis=0)

    weighted = _compute_pairwise_similarity_matrix(
        selected,
        metric="correlation",
        feature_weights=expansion,
    )
    explicit_population = _compute_pairwise_similarity_matrix(population, metric="correlation")
    legacy = _compute_pairwise_similarity_matrix(selected, metric="correlation")
    unit_weighted = _compute_pairwise_similarity_matrix(
        selected,
        metric="correlation",
        feature_weights=np.ones(3),
    )

    assert np.allclose(weighted, explicit_population)
    assert np.array_equal(unit_weighted, legacy)
    assert not np.allclose(weighted, legacy)


def test_sampled_grouping_and_allocation_preserve_sampling_and_apply_weights(tmp_path):
    profile_dir = tmp_path / "sampled-profile"
    grouping_path = profile_dir / "grouping" / "timestep_grouping.json"
    profile_dir.mkdir()
    payload = {
        "group_names": SAMPLED_GROUP_NAMES,
        "group_param_counts": dict(zip(SAMPLED_GROUP_NAMES, [2, 3, 5])),
        "n_eff": [2.0, 3.0, 4.0, 5.0],
        "relative_delta_stack": [
            [1.0, 4.0, 2.0, 8.0],
            [3.0, 1.0, 5.0, 2.0],
            [7.0, 2.0, 6.0, 3.0],
        ],
        "ablation_protocol": PFI_PROTOCOL,
        "profile_fingerprint": PFI_FINGERPRINT,
        "profile_fingerprint_sha256": PFI_FINGERPRINT_SHA256,
        "filter_sampling": FILTER_SAMPLING,
        "group_sampling_weights": dict(zip(SAMPLED_GROUP_NAMES, [1.0, 2.0, 3.0])),
    }
    results_path = profile_dir / "results.json"
    results_path.write_text(json.dumps(payload))

    subprocess.run(
        [
            sys.executable,
            "scripts/optimize_timestep_grouping.py",
            "--matrix",
            str(results_path),
            "--matrix_key",
            "relative_delta_stack",
            "--builtin_cost",
            "matrix_correlation_cross_penalty",
            "--num_blocks",
            "2",
            "--output",
            str(grouping_path),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    grouping = json.loads(grouping_path.read_text())
    assert grouping["filter_sampling"] == FILTER_SAMPLING
    assert grouping["matrix_row_weighting"]["application"] == (
        "weighted_group_rows_for_timestep_similarity"
    )
    assert grouping["matrix_row_weighting"]["population_group_count_estimate"] == 6.0

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/dry_run_capacity_allocation.py",
            "--eval-output-dir",
            str(profile_dir),
            "--student-variant",
            "layerwise_capacity",
            # Legacy n_eff and relative-delta scores: the fixture carries no delta_stack.
            "--allocation-metric",
            "auto",
            "--score-reduction",
            "mean",
            "--layer-score-source",
            "relative_delta_stack",
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    allocation = json.loads(completed.stdout)
    assert allocation["filter_sampling"] == FILTER_SAMPLING
    assert allocation["original_model_budget"]["parameters"] == 23
    assert allocation["score_sources"]["layer_capacity_scores"]["filter_sampling_weighting"][
        "application"
    ] == "selected_group_population_expansion"
