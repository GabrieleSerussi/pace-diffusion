"""Stable provenance records for parameter-profile-derived artifacts.

The parameter evaluator owns the definition of an ablation protocol and its
profile fingerprint.  Downstream tools should copy those records, not infer a
new protocol from legacy ``ablation_info`` fields.  Profiles created before the
versioned record was introduced are therefore identified explicitly as
``legacy_unspecified``.

Released profiles are *slim* copies of the full ``results.json`` files: they keep
every array the grouping and allocation stages read and drop bulky listings such
as the embedded profile fingerprint.  A slim profile carries a
``pace_slim_profile`` record with the SHA-256 of the full profile it was cut
from, and :func:`source_profile_from_results` reports that digest, so the
released groupings and allocations (made from the full profiles) still validate
against it.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from pace.filter_sampling import normalize_filter_sampling
from pace.jsonio import load_json
from pace.slim_profile import SLIM_PROFILE_FORMAT, slim_profile_record as _slim_record


SOURCE_PROFILE_FORMAT = "diffdist_edm_source_profile_v1"
LEGACY_ABLATION_PROTOCOL_ID = "legacy_unspecified"


def slim_profile_record(results: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return the slim-profile record of a released profile, if present."""

    record = _slim_record(results)
    if record is None:
        return None
    source_sha256 = record.get("source_results_sha256")
    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        raise ValueError("slim profile record is missing source_results_sha256")
    return record


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_ablation_protocol(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the versioned protocol record, or an explicit legacy identity.

    In particular, an old ``ablation_info.mode == 'random_same_norm'`` record
    is not promoted to the new, precisely specified protocol.  The historical
    record did not pin donor pairing, sigma batching, or RNG semantics.
    """

    raw = payload.get("ablation_protocol") if isinstance(payload, Mapping) else None
    if not isinstance(raw, Mapping):
        return {"protocol_id": LEGACY_ABLATION_PROTOCOL_ID}
    protocol = copy.deepcopy(dict(raw))
    protocol_id = protocol.get("protocol_id") or protocol.get("identity")
    if not isinstance(protocol_id, str) or not protocol_id:
        protocol["protocol_id"] = LEGACY_ABLATION_PROTOCOL_ID
    elif "protocol_id" not in protocol:
        protocol["protocol_id"] = protocol_id
    return protocol


def source_profile_from_results(
    results: Mapping[str, Any] | None,
    results_json_path: str | Path | None,
) -> dict[str, Any]:
    """Build the immutable pointer copied by every downstream artifact."""

    source_path: Path | None = None
    source_sha256: str | None = None
    file_sha256: str | None = None
    if results_json_path is not None:
        source_path = Path(results_json_path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"profile results do not exist: {source_path}")
        file_sha256 = _file_sha256(source_path)
        source_sha256 = file_sha256
    slim = slim_profile_record(results)
    if slim is not None:
        # The slim file stands in for the full profile it was cut from.
        source_sha256 = str(slim["source_results_sha256"])

    profile_fingerprint = None
    fingerprint_digest = None
    if isinstance(results, Mapping):
        if "profile_fingerprint" in results:
            profile_fingerprint = copy.deepcopy(results.get("profile_fingerprint"))
        fingerprint_digest = (
            results.get("profile_fingerprint_sha256")
            or results.get("profile_fingerprint_digest")
        )
    if profile_fingerprint is not None:
        computed_fingerprint_digest = _canonical_digest(profile_fingerprint)
        if fingerprint_digest and fingerprint_digest != computed_fingerprint_digest:
            raise ValueError(
                "profile_fingerprint_sha256 does not match the canonical profile_fingerprint record"
            )
        fingerprint_digest = computed_fingerprint_digest

    record = {
        "format": SOURCE_PROFILE_FORMAT,
        "results_json_path": None if source_path is None else str(source_path),
        "results_sha256": source_sha256,
        "profile_fingerprint": profile_fingerprint,
        "profile_fingerprint_sha256": fingerprint_digest,
    }
    if slim is not None:
        record["slim_profile"] = {
            "format": SLIM_PROFILE_FORMAT,
            "slim_results_sha256": file_sha256,
        }
    return record


def provenance_from_results(
    results: Mapping[str, Any] | None,
    results_json_path: str | Path | None,
) -> dict[str, dict[str, Any]]:
    """Return canonical top-level provenance fields.

    Sampling provenance is omitted only when the source predates the versioned
    filter-sampling schema.  This keeps legacy artifacts byte-for-byte stable
    apart from their existing source/ablation provenance while ensuring new
    sampled and explicitly exhaustive profiles remain distinguishable.
    """

    provenance = {
        "source_profile": source_profile_from_results(results, results_json_path),
        "ablation_protocol": normalize_ablation_protocol(results),
    }
    filter_sampling = normalize_filter_sampling(results)
    if filter_sampling is not None:
        provenance["filter_sampling"] = filter_sampling
    return provenance


def provenance_from_results_path(path: str | Path) -> dict[str, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    try:
        payload = load_json(source)
    except (OSError, EOFError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read profile results {source}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"profile results must contain a JSON object: {source}")
    return provenance_from_results(payload, source)


def provenance_from_artifact(
    payload: Mapping[str, Any] | None,
    *,
    fallback: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Copy provenance from a downstream artifact with legacy fallback."""

    primary = payload if isinstance(payload, Mapping) else {}
    secondary = fallback if isinstance(fallback, Mapping) else {}
    raw_source = primary.get("source_profile")
    if not isinstance(raw_source, Mapping):
        raw_source = secondary.get("source_profile")
    if isinstance(raw_source, Mapping):
        source_profile = copy.deepcopy(dict(raw_source))
    else:
        results_path = primary.get("results_json_path") or secondary.get("results_json_path")
        # Do not retroactively hash a mutable source when reading a legacy
        # artifact; the missing digest is itself meaningful provenance.
        source_profile = {
            "format": SOURCE_PROFILE_FORMAT,
            "results_json_path": str(results_path) if results_path is not None else None,
            "results_sha256": None,
            "profile_fingerprint": None,
            "profile_fingerprint_sha256": None,
        }

    protocol_source: Mapping[str, Any] = primary
    if not isinstance(primary.get("ablation_protocol"), Mapping):
        protocol_source = secondary
    provenance = {
        "source_profile": source_profile,
        "ablation_protocol": normalize_ablation_protocol(protocol_source),
    }
    raw_sampling = primary.get("filter_sampling")
    if raw_sampling is None:
        raw_sampling = secondary.get("filter_sampling")
    if raw_sampling is not None:
        filter_sampling = normalize_filter_sampling({"filter_sampling": raw_sampling})
        if filter_sampling is not None:
            provenance["filter_sampling"] = filter_sampling
    return provenance


def validate_matching_source_profile(
    expected: Mapping[str, Any],
    observed_artifact: Mapping[str, Any],
    *,
    context: str,
    expected_ablation_protocol: Mapping[str, Any] | None = None,
    expected_filter_sampling: Mapping[str, Any] | None = None,
) -> None:
    """Reject a versioned downstream artifact tied to a different profile.

    Artifacts without source provenance remain accepted for backward
    compatibility.  Once both sides carry immutable digests, mismatches are an
    error rather than silently combining unrelated analysis runs.
    """

    observed = observed_artifact.get("source_profile")
    if not isinstance(observed, Mapping):
        return
    expected_sha = expected.get("results_sha256")
    observed_sha = observed.get("results_sha256")
    if expected_sha and observed_sha and expected_sha != observed_sha:
        raise ValueError(
            f"{context} source profile SHA-256 mismatch: expected {expected_sha}, got {observed_sha}"
        )
    expected_fingerprint = (
        expected.get("profile_fingerprint_sha256") or expected.get("profile_fingerprint_digest")
    )
    observed_fingerprint = (
        observed.get("profile_fingerprint_sha256") or observed.get("profile_fingerprint_digest")
    )
    if expected_fingerprint and observed_fingerprint and expected_fingerprint != observed_fingerprint:
        raise ValueError(
            f"{context} profile fingerprint mismatch: expected {expected_fingerprint}, got {observed_fingerprint}"
        )
    observed_protocol = observed_artifact.get("ablation_protocol")
    if expected_ablation_protocol is not None and isinstance(observed_protocol, Mapping):
        if dict(expected_ablation_protocol) != dict(observed_protocol):
            raise ValueError(f"{context} ablation protocol does not match its source profile")
    observed_sampling = observed_artifact.get("filter_sampling")
    if expected_filter_sampling is not None:
        if not isinstance(observed_sampling, Mapping):
            raise ValueError(f"{context} is missing filter sampling provenance")
        if dict(expected_filter_sampling) != dict(observed_sampling):
            raise ValueError(f"{context} filter sampling protocol does not match its source profile")
