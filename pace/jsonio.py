"""JSON helpers that also read gzip-compressed artifacts.

The released phase and budget artifacts under ``artifacts/`` are stored as
``*.json`` or, when they are large, as ``*.json.gz``.  Every loader that reads
a profile, a timestep grouping, an allocation or an architecture plan goes
through :func:`load_json`, so a compressed file can be passed wherever a plain
JSON file is accepted.

Compression is detected from the gzip magic number, not only from the file
name, so a compressed payload is read correctly even under a ``.json`` name.

:func:`load_json` also restores released *slim* profiles
(:mod:`pace.slim_profile`): packed float64 arrays are decoded into nested
lists and the derived ``relative_delta_stack`` is recomputed, so the result is
what ``json.load`` returns for the full profile, restricted to the kept keys.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from .slim_profile import decode_packed_arrays, expand_slim_profile

GZIP_MAGIC = b"\x1f\x8b"
JSON_SUFFIXES: tuple[str, ...] = (".json", ".json.gz")


def is_gzip_file(path: str | Path) -> bool:
    """Return whether ``path`` starts with the gzip magic number."""

    try:
        with Path(path).open("rb") as handle:
            return handle.read(2) == GZIP_MAGIC
    except OSError:
        return False


def is_json_path(path: str | Path) -> bool:
    """Return whether ``path`` names a ``.json`` or ``.json.gz`` file."""

    name = Path(path).name.lower()
    return any(name.endswith(suffix) for suffix in JSON_SUFFIXES)


def json_stem(path: str | Path) -> str:
    """Return the file name without its ``.json`` or ``.json.gz`` suffix."""

    name = Path(path).name
    lowered = name.lower()
    for suffix in (".json.gz", ".json"):
        if lowered.endswith(suffix):
            return name[: -len(suffix)]
    return Path(path).stem


def read_json_text(path: str | Path) -> str:
    """Return the decoded text of a plain or gzip-compressed JSON file."""

    source = Path(path)
    if is_gzip_file(source):
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            return handle.read()
    return source.read_text()


def load_json(path: str | Path) -> Any:
    """Load a plain or gzip-compressed JSON file (slim profiles are expanded)."""

    return loads_json(read_json_text(path))


def loads_json(text: str) -> Any:
    """``json.loads`` with packed-array decoding and slim-profile expansion."""

    payload = json.loads(text, object_hook=decode_packed_arrays)
    return expand_slim_profile(payload)


def find_json_file(directory: str | Path, stem: str) -> Path | None:
    """Return ``<directory>/<stem>.json`` or ``<stem>.json.gz`` when present.

    The uncompressed file wins when both exist, which keeps the historical
    layout of ``evaluate_parameters*`` output directories unchanged.
    """

    root = Path(directory)
    for suffix in JSON_SUFFIXES:
        candidate = root / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def dump_json(payload: Any, path: str | Path, *, indent: int | None = 2, compress: bool | None = None) -> Path:
    """Write ``payload`` as JSON, gzip-compressed when the name ends in ``.gz``."""

    target = Path(path)
    if compress is None:
        compress = target.name.lower().endswith(".gz")
    text = json.dumps(payload, indent=indent) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        # An empty stored file name and mtime=0 keep the compressed bytes
        # reproducible and free of local path information.
        with target.open("wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=9, mtime=0) as handle:
                handle.write(text.encode("utf-8"))
    else:
        target.write_text(text)
    return target
