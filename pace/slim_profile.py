"""Slim, lossless copies of teacher profiles for release.

A full ``results.json`` written by ``scripts/evaluate_parameters_edm.py`` holds
several group-by-bin matrices, an embedded profile fingerprint and, for LSUN, a
listing of the one million dataset files.  The phase-discovery and allocation
stages read only a part of it.  A *slim* profile keeps every key of the full
profile except:

* the matrices that no grouping, allocation or architecture-compilation stage
  reads (``weights``, ``row_normalized_relative_delta_stack``,
  ``signed_delta_stack``, ``sampling_adjusted_delta_stack``);
* ``relative_delta_stack``, which is stored implicitly: it is recomputed on
  load as ``delta_stack / (baseline_mean + 1e-12)``, the formula of the
  profiler, and the recomputation is exact (bit for bit) for every released
  profile;
* the embedded ``profile_fingerprint`` (its canonical SHA-256 digest is kept in
  ``profile_fingerprint_sha256``, which is what downstream validators compare);
* the per-file listings in ``dataset_info.manifest_metadata`` (their digests are
  kept).

``delta_stack`` is stored losslessly as a packed float64 block (see
:func:`pack_array`).  The ``pace_slim_profile`` record carries the SHA-256 of
the full profile, so artifacts derived from the full profile validate against
the slim copy (see :mod:`pace.profile_provenance`).

:func:`pace.jsonio.load_json` unpacks the arrays and restores the derived
matrices transparently, so every loader in this repository accepts a slim
profile wherever it accepts a full one.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

SLIM_PROFILE_KEY = "pace_slim_profile"
SLIM_PROFILE_FORMAT = "pace_slim_profile_v1"
PACKED_ARRAY_KEY = "__pace_ndarray__"
PACKED_ARRAY_FORMAT = "pace_ndarray_v1"

# Matrices that no grouping, allocation or compilation stage reads.
UNUSED_MATRIX_KEYS: tuple[str, ...] = (
    "weights",
    "row_normalized_relative_delta_stack",
    "signed_delta_stack",
    "sampling_adjusted_delta_stack",
)

# Matrices recomputed exactly on load: key -> (formula, epsilon).
DERIVED_MATRIX_KEYS: Mapping[str, str] = {
    "relative_delta_stack": "delta_stack / (baseline_mean + 1e-12)",
}
RELATIVE_DELTA_EPSILON = 1e-12

# Large matrices stored as packed float64 blocks.
PACKED_MATRIX_KEYS: tuple[str, ...] = ("delta_stack",)

# Bulky per-file listings of a dataset manifest (their digests are kept).
MANIFEST_LISTING_KEYS: tuple[str, ...] = ("entries", "splits")


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# ---------------------------------------------------------------------------
# Packed arrays
# ---------------------------------------------------------------------------


def pack_array(values: Any, *, codec: str = "xz") -> dict[str, Any]:
    """Encode a numeric array losslessly as little-endian float64 bytes.

    ``codec`` is ``"xz"`` (LZMA-compressed bytes, the default) or ``"none"``
    (raw bytes; the surrounding ``.json.gz`` compresses them).  The payload is
    base64 text, so the result stays valid JSON.
    """

    array = np.asarray(values, dtype="<f8")
    raw = array.tobytes()
    if codec == "xz":
        import lzma

        data = lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)
    elif codec == "none":
        data = raw
    else:
        raise ValueError(f"unsupported packed-array codec {codec!r}")
    return {
        PACKED_ARRAY_KEY: PACKED_ARRAY_FORMAT,
        "dtype": "<f8",
        "shape": [int(size) for size in array.shape],
        "codec": codec,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data": base64.b64encode(data).decode("ascii"),
    }


def is_packed_array(value: Any) -> bool:
    return isinstance(value, Mapping) and value.get(PACKED_ARRAY_KEY) == PACKED_ARRAY_FORMAT


def unpack_array(value: Mapping[str, Any]) -> list[Any]:
    """Decode :func:`pack_array` output into nested Python lists of floats.

    Nested lists are what ``json.load`` returns for the original matrix, so
    code that reads the full profile sees exactly the same values.
    """

    if value.get("dtype") != "<f8":
        raise ValueError(f"unsupported packed-array dtype {value.get('dtype')!r}")
    data = base64.b64decode(value["data"])
    codec = value.get("codec", "none")
    if codec == "xz":
        try:
            import lzma
        except ImportError as exc:  # pragma: no cover - lzma ships with CPython.
            raise ImportError("reading a packed array needs Python's lzma module") from exc
        raw = lzma.decompress(data)
    elif codec == "none":
        raw = data
    else:
        raise ValueError(f"unsupported packed-array codec {codec!r}")
    expected = value.get("sha256")
    if expected is not None and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("packed array payload does not match its SHA-256")
    shape = tuple(int(size) for size in value["shape"])
    return np.frombuffer(raw, dtype="<f8").reshape(shape).tolist()


def decode_packed_arrays(obj: dict[str, Any]) -> Any:
    """``json.loads`` object hook that unpacks packed arrays."""

    if obj.get(PACKED_ARRAY_KEY) == PACKED_ARRAY_FORMAT:
        return unpack_array(obj)
    return obj


# ---------------------------------------------------------------------------
# Slim profiles
# ---------------------------------------------------------------------------


def slim_profile_record(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return the ``pace_slim_profile`` record of a payload, if any."""

    if not isinstance(payload, Mapping):
        return None
    record = payload.get(SLIM_PROFILE_KEY)
    if isinstance(record, Mapping) and record.get("format") == SLIM_PROFILE_FORMAT:
        return dict(record)
    return None


def relative_delta_stack(delta_stack: Any, baseline_mean: Any) -> list[Any]:
    """The profiler's ``delta_stack / (baseline_mean + 1e-12)``, in float64."""

    delta = np.asarray(delta_stack, dtype=np.float64)
    baseline = np.asarray(baseline_mean, dtype=np.float64)
    return (delta / (baseline + RELATIVE_DELTA_EPSILON)).tolist()


def expand_slim_profile(payload: Any) -> Any:
    """Restore the derived matrices of a slim profile in place.

    Payloads without a slim-profile record are returned unchanged.
    """

    record = slim_profile_record(payload)
    if record is None:
        return payload
    derived = record.get("derived_keys") or {}
    if "relative_delta_stack" in derived and "relative_delta_stack" not in payload:
        payload["relative_delta_stack"] = relative_delta_stack(
            payload["delta_stack"], payload["baseline_mean"]
        )
    return payload


def _slim_dataset_info(dataset_info: Any) -> tuple[Any, list[str]]:
    if not isinstance(dataset_info, Mapping):
        return dataset_info, []
    slimmed = copy.deepcopy(dict(dataset_info))
    dropped: list[str] = []
    manifest = slimmed.get("manifest_metadata")
    if isinstance(manifest, dict):
        for key in MANIFEST_LISTING_KEYS:
            if key in manifest:
                manifest.pop(key)
                dropped.append(f"dataset_info.manifest_metadata.{key}")
    return slimmed, dropped


def make_slim_profile(
    results: Mapping[str, Any],
    *,
    source_sha256: str,
    source_size_bytes: int | None = None,
    source_path: str | None = None,
    codec: str = "xz",
) -> dict[str, Any]:
    """Return the slim copy of a full profile (see the module docstring).

    Raises ``ValueError`` when a derived matrix cannot be recomputed exactly
    or when the embedded fingerprint disagrees with its recorded digest.
    """

    if slim_profile_record(results) is not None:
        raise ValueError("profile is already slim")
    slim: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in results.items():
        if key in UNUSED_MATRIX_KEYS:
            dropped.append(key)
            continue
        if key in DERIVED_MATRIX_KEYS:
            continue
        if key == "profile_fingerprint":
            digest = _canonical_digest(value)
            recorded = results.get("profile_fingerprint_sha256") or results.get("profile_fingerprint_digest")
            if recorded and recorded != digest:
                raise ValueError("profile_fingerprint_sha256 does not match the embedded profile_fingerprint")
            slim["profile_fingerprint_sha256"] = digest
            dropped.append("profile_fingerprint")
            continue
        if key == "dataset_info":
            value, listing_keys = _slim_dataset_info(value)
            dropped.extend(listing_keys)
        slim[key] = copy.deepcopy(value)

    derived: dict[str, str] = {}
    if "relative_delta_stack" in results:
        if "delta_stack" not in results or "baseline_mean" not in results:
            raise ValueError("relative_delta_stack needs delta_stack and baseline_mean to be derived")
        recomputed = np.asarray(
            relative_delta_stack(results["delta_stack"], results["baseline_mean"]), dtype=np.float64
        )
        stored = np.asarray(results["relative_delta_stack"], dtype=np.float64)
        if recomputed.shape != stored.shape or not np.array_equal(recomputed, stored):
            raise ValueError("relative_delta_stack is not exactly delta_stack / (baseline_mean + 1e-12)")
        derived["relative_delta_stack"] = DERIVED_MATRIX_KEYS["relative_delta_stack"]

    packed: list[str] = []
    for key in PACKED_MATRIX_KEYS:
        if key in slim:
            slim[key] = pack_array(slim[key], codec=codec)
            packed.append(key)

    slim[SLIM_PROFILE_KEY] = {
        "format": SLIM_PROFILE_FORMAT,
        "source_results_sha256": str(source_sha256),
        "source_results_size_bytes": None if source_size_bytes is None else int(source_size_bytes),
        "source_results_path": source_path,
        "dropped_keys": dropped,
        "derived_keys": derived,
        "packed_keys": packed,
    }
    return slim


def compare_expanded(full: Mapping[str, Any], expanded: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    """Return the keys whose values differ between a full and an expanded profile."""

    mismatched: list[str] = []
    for key in keys:
        if key not in full:
            continue
        if key not in expanded:
            mismatched.append(key)
            continue
        left, right = full[key], expanded[key]
        if isinstance(left, list) and left and isinstance(left[0], list):
            left_array = np.asarray(left, dtype=np.float64)
            right_array = np.asarray(right, dtype=np.float64)
            if left_array.shape != right_array.shape or not np.array_equal(
                left_array, right_array, equal_nan=True
            ):
                mismatched.append(key)
        elif json.dumps(left, sort_keys=True) != json.dumps(right, sort_keys=True):
            mismatched.append(key)
    return mismatched
