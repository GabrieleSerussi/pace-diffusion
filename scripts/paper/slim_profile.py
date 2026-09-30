#!/usr/bin/env python3
"""Write a slim, lossless copy of a teacher profile (``results.json``).

The slim copy keeps what the grouping, allocation and architecture-compilation
stages read (see :mod:`pace.slim_profile` for the exact rule) and is written as
gzip-compressed JSON.  After writing, the script reloads the slim file through
:func:`pace.jsonio.load_json` and checks that every kept key and every derived
matrix equals the full profile exactly::

    python scripts/paper/slim_profile.py --results out_eval_edm_cifar10/results.json \\
        --output artifacts/profiles/my_profile.json.gz

This is how the profiles under ``artifacts/profiles/`` were produced (their
recorded source paths were additionally made repository-relative).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pace.jsonio import dump_json, load_json, read_json_text
from pace.slim_profile import (
    DERIVED_MATRIX_KEYS,
    PACKED_MATRIX_KEYS,
    SLIM_PROFILE_KEY,
    compare_expanded,
    make_slim_profile,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_round_trip(full: dict, slim_path: Path) -> list[str]:
    """Return the keys that do not survive the slim round trip."""

    expanded = load_json(slim_path)
    record = expanded[SLIM_PROFILE_KEY]
    dropped = set(record["dropped_keys"])
    kept = [
        key
        for key in full
        if key not in dropped and key not in {"profile_fingerprint", "dataset_info"}
    ]
    keys = kept + [key for key in DERIVED_MATRIX_KEYS if key in full] + list(PACKED_MATRIX_KEYS)
    return compare_expanded(full, expanded, sorted(set(keys)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True, help="Full profile results.json.")
    parser.add_argument("--output", type=Path, required=True, help="Slim profile path (.json.gz).")
    parser.add_argument(
        "--source-path",
        default=None,
        help="Path of the full profile to record in the slim copy (default: as given to --results).",
    )
    parser.add_argument("--codec", choices=("xz", "none"), default="xz", help="Packed-array codec.")
    args = parser.parse_args()

    full = json.loads(read_json_text(args.results))
    if not isinstance(full, dict):
        parser.error(f"{args.results} does not contain a JSON object")
    slim = make_slim_profile(
        full,
        source_sha256=file_sha256(args.results),
        source_size_bytes=args.results.stat().st_size,
        source_path=args.source_path if args.source_path is not None else str(args.results),
        codec=args.codec,
    )
    dump_json(slim, args.output, indent=None)
    mismatched = verify_round_trip(full, args.output)
    if mismatched:
        raise SystemExit(f"slim round trip changed these keys: {mismatched}")
    print(
        f"wrote {args.output} ({args.output.stat().st_size / 1e6:.2f} MB, "
        f"from {args.results.stat().st_size / 1e6:.2f} MB); round trip exact"
    )


if __name__ == "__main__":
    main()
