"""Validation and alignment helpers for sampled per-filter EDM profiles.

The evaluator stores raw selected-group rows so plots and diagnostics remain
about observations that were actually measured.  Downstream population
quantities use the Horvitz--Thompson expansion weights recorded alongside
those rows.  Profiles without a versioned sampling record are treated as
historical exhaustive profiles and retain unit weights.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence


FILTER_SAMPLING_FORMAT = "diffdist_edm_filter_sampling_protocol_v1"
STRATIFIED_FILTER_SAMPLING_PROTOCOL_ID = "per_module_hash_stratified_filter_sampling_v1"
EXHAUSTIVE_FILTER_SAMPLING_PROTOCOL_ID = "per_filter_exhaustive_v1"
_PROTOCOL_MODES = {
    STRATIFIED_FILTER_SAMPLING_PROTOCOL_ID: "stratified_module",
    EXHAUSTIVE_FILTER_SAMPLING_PROTOCOL_ID: "exhaustive",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _positive_int(record: Mapping[str, Any], key: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"filter_sampling.{key} must be a positive integer")
    return value


def _canonical_name_digest(names: Sequence[str]) -> str:
    encoded = json.dumps(
        sorted(str(name) for name in names),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _positive_finite(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field} must be positive and finite")
    return number


def normalize_filter_sampling(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Copy a versioned filter-sampling record without inventing legacy data."""

    if not isinstance(payload, Mapping) or "filter_sampling" not in payload:
        return None
    raw = payload.get("filter_sampling")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("filter_sampling must be a mapping or null")
    record = copy.deepcopy(dict(raw))
    protocol_id = record.get("protocol_id")
    if protocol_id not in _PROTOCOL_MODES:
        raise ValueError(
            f"unsupported filter_sampling.protocol_id {protocol_id!r}; "
            f"expected one of {tuple(_PROTOCOL_MODES)}"
        )
    record_format = record.get("format")
    if record_format != FILTER_SAMPLING_FORMAT:
        raise ValueError(
            f"unsupported filter_sampling format {record_format!r}; expected {FILTER_SAMPLING_FORMAT!r}"
        )
    expected_mode = _PROTOCOL_MODES[protocol_id]
    if record.get("mode") != expected_mode:
        raise ValueError(
            f"filter_sampling.mode must be {expected_mode!r} for protocol {protocol_id!r}"
        )

    population_count = _positive_int(record, "population_group_count")
    selected_count = _positive_int(record, "selected_group_count")
    population_modules = _positive_int(record, "population_module_count")
    selected_modules = _positive_int(record, "selected_module_count")
    if selected_count > population_count:
        raise ValueError("filter_sampling.selected_group_count exceeds its population")
    if selected_modules > population_modules:
        raise ValueError("filter_sampling.selected_module_count exceeds its population")
    for digest_key in ("selection_sha256", "population_sha256"):
        digest = record.get(digest_key)
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"filter_sampling.{digest_key} must be a lowercase SHA-256 digest")

    raw_module_counts = record.get("module_counts")
    if not isinstance(raw_module_counts, Mapping) or len(raw_module_counts) != population_modules:
        raise ValueError(
            "filter_sampling.module_counts must contain exactly population_module_count entries"
        )
    total_population = 0
    total_selected = 0
    observed_selected_modules = 0
    for raw_module_name, raw_counts in raw_module_counts.items():
        module_name = str(raw_module_name)
        if not module_name or not isinstance(raw_counts, Mapping):
            raise ValueError("filter_sampling.module_counts entries must be named mappings")
        module_population = _positive_int(raw_counts, "population_filter_count")
        module_selected = _positive_int(raw_counts, "selected_filter_count")
        if module_selected > module_population:
            raise ValueError(
                f"filter_sampling.module_counts[{module_name!r}] selects more filters than exist"
            )
        probability = _positive_finite(
            raw_counts.get("inclusion_probability"),
            field=f"filter_sampling.module_counts[{module_name!r}].inclusion_probability",
        )
        expansion = _positive_finite(
            raw_counts.get("expansion_weight"),
            field=f"filter_sampling.module_counts[{module_name!r}].expansion_weight",
        )
        expected_probability = module_selected / module_population
        if not math.isclose(probability, expected_probability, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                f"filter_sampling.module_counts[{module_name!r}] inclusion probability is inconsistent"
            )
        if not math.isclose(expansion, 1.0 / probability, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                f"filter_sampling.module_counts[{module_name!r}] expansion weight is not 1/pi"
            )
        total_population += module_population
        total_selected += module_selected
        observed_selected_modules += 1
    if total_population != population_count or total_selected != selected_count:
        raise ValueError("filter_sampling module counts do not sum to the recorded group counts")
    if observed_selected_modules != selected_modules:
        raise ValueError("filter_sampling selected module count is inconsistent with module_counts")

    seed = record.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("filter_sampling.seed must be an integer")
    if protocol_id == STRATIFIED_FILTER_SAMPLING_PROTOCOL_ID:
        filters_per_module = record.get("filters_per_module")
        if isinstance(filters_per_module, bool) or not isinstance(filters_per_module, int) or filters_per_module <= 0:
            raise ValueError("stratified filter_sampling.filters_per_module must be a positive integer")
        raw_names = record.get("selected_group_names")
        if not isinstance(raw_names, Sequence) or isinstance(raw_names, (str, bytes)):
            raise ValueError("stratified filter_sampling.selected_group_names must be a sequence")
        names = [str(name) for name in raw_names]
        if len(names) != selected_count or len(set(names)) != selected_count:
            raise ValueError("stratified filter_sampling selected group names are not unique/count-aligned")
        if _canonical_name_digest(names) != record["selection_sha256"]:
            raise ValueError("filter_sampling.selection_sha256 does not match selected_group_names")
        probabilities = _coerce_weight_map(
            record.get("selected_group_inclusion_probabilities"),
            field="filter_sampling.selected_group_inclusion_probabilities",
        )
        expansions = _coerce_weight_map(
            record.get("selected_group_expansion_weights"),
            field="filter_sampling.selected_group_expansion_weights",
        )
        if probabilities is None or expansions is None or set(probabilities) != set(names) or set(expansions) != set(names):
            raise ValueError("stratified filter_sampling probability/weight maps must exactly cover selected groups")
        for name in names:
            probability = probabilities[name]
            if probability > 1.0:
                raise ValueError(f"filter_sampling inclusion probability exceeds one for {name!r}")
            if not math.isclose(expansions[name], 1.0 / probability, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"filter_sampling expansion weight is not 1/pi for {name!r}")
        if not math.isclose(sum(expansions.values()), population_count, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError("filter_sampling expansion weights do not recover population_group_count")
    else:
        if record.get("filters_per_module") is not None:
            raise ValueError("exhaustive filter_sampling.filters_per_module must be null")
        if selected_count != population_count or selected_modules != population_modules:
            raise ValueError("exhaustive filter_sampling must select its complete population")
        if record["selection_sha256"] != record["population_sha256"]:
            raise ValueError("exhaustive filter_sampling selection and population digests must match")
    return record


def _coerce_weight_map(value: Any, *, field: str) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping keyed by selected group name")
    weights: dict[str, float] = {}
    for raw_name, raw_weight in value.items():
        name = str(raw_name)
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field}[{name!r}] must be numeric") from exc
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"{field}[{name!r}] must be positive and finite, got {raw_weight!r}")
        weights[name] = weight
    return weights


def group_expansion_weight_map(payload: Mapping[str, Any] | None) -> dict[str, float] | None:
    """Return the canonical selected-group ``1 / pi`` map, when recorded.

    New results duplicate the map at top level as ``group_sampling_weights``
    for metric recomputation.  Requiring the duplicate and protocol record to
    agree prevents a downstream tool from silently using stale weights.
    """

    if not isinstance(payload, Mapping):
        return None
    direct = _coerce_weight_map(
        payload.get("group_sampling_weights"),
        field="group_sampling_weights",
    )
    sampling = normalize_filter_sampling(payload)
    if direct is not None and sampling is None:
        raise ValueError(
            "group_sampling_weights requires a versioned filter_sampling record"
        )
    nested = _coerce_weight_map(
        None if sampling is None else sampling.get("selected_group_expansion_weights"),
        field="filter_sampling.selected_group_expansion_weights",
    )
    if direct is not None and nested is not None and direct != nested:
        raise ValueError(
            "group_sampling_weights does not match "
            "filter_sampling.selected_group_expansion_weights"
        )
    return direct if direct is not None else nested


def aligned_group_expansion_weights(
    payload: Mapping[str, Any] | None,
    group_names: Sequence[str] | None = None,
) -> list[float]:
    """Return expansion weights aligned with ``group_names``.

    Missing sampling metadata means legacy/exhaustive unit weights.  A
    versioned stratified record must be complete: silently substituting ones
    would bias every downstream population-total estimate.
    """

    if group_names is None:
        if not isinstance(payload, Mapping) or not isinstance(payload.get("group_names"), Sequence):
            raise ValueError("group_names are required to align filter sampling weights")
        raw_names = payload["group_names"]
        if isinstance(raw_names, (str, bytes)):
            raise ValueError("group_names must be a sequence, not a string")
        names = [str(name) for name in raw_names]
    else:
        names = [str(name) for name in group_names]
    if len(set(names)) != len(names):
        raise ValueError("group_names must be unique when aligning filter sampling weights")

    sampling = normalize_filter_sampling(payload)
    if sampling is not None:
        selected_count = int(sampling["selected_group_count"])
        if len(names) != selected_count:
            raise ValueError(
                "group_names count does not match filter_sampling.selected_group_count"
            )
        if sampling["protocol_id"] == EXHAUSTIVE_FILTER_SAMPLING_PROTOCOL_ID:
            names_digest = _canonical_name_digest(names)
            if names_digest != sampling["selection_sha256"]:
                raise ValueError(
                    "group_names do not match exhaustive filter_sampling selection_sha256"
                )
    weight_map = group_expansion_weight_map(payload)
    protocol_id = None if sampling is None else sampling.get("protocol_id")
    requires_weights = protocol_id == STRATIFIED_FILTER_SAMPLING_PROTOCOL_ID
    if weight_map is None:
        if requires_weights:
            raise ValueError(
                "sampled filter profile is missing selected-group expansion weights"
            )
        return [1.0] * len(names)

    missing = [name for name in names if name not in weight_map]
    extra = sorted(set(weight_map) - set(names))
    if missing or extra:
        raise ValueError(
            "filter sampling weights do not align with group_names; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    selected_names = None if sampling is None else sampling.get("selected_group_names")
    if selected_names is not None:
        if not isinstance(selected_names, Sequence) or isinstance(selected_names, (str, bytes)):
            raise ValueError("filter_sampling.selected_group_names must be a sequence")
        if [str(name) for name in selected_names] != names:
            raise ValueError("filter_sampling.selected_group_names does not match group_names order")

    return [weight_map[name] for name in names]


def expansion_weighting_summary(
    payload: Mapping[str, Any] | None,
    group_names: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    """Describe the population expansion applied by a downstream artifact."""

    sampling = normalize_filter_sampling(payload)
    weight_map = group_expansion_weight_map(payload)
    if sampling is None and weight_map is None:
        return None
    weights = aligned_group_expansion_weights(payload, group_names)
    return {
        "estimator": "horvitz_thompson_expansion_v1",
        "protocol_id": None if sampling is None else sampling.get("protocol_id"),
        "selected_group_count": len(weights),
        "population_group_count_estimate": float(sum(weights)),
        "non_unit_weight_count": sum(weight != 1.0 for weight in weights),
        "weight_min": None if not weights else float(min(weights)),
        "weight_max": None if not weights else float(max(weights)),
        "weight_source": (
            "group_sampling_weights/filter_sampling.selected_group_expansion_weights"
        ),
    }
