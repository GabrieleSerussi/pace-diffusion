#!/usr/bin/env python3
"""Download, safely extract, and validate canonical Speech Commands v0.02 SC09."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
from typing import Any
import urllib.request
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pace.audio_datasets import (
    SC09_ARCHIVE_SHA256,
    SC09_ARCHIVE_FILENAME,
    SC09_ARCHIVE_SIZE,
    SC09_ARCHIVE_URL,
    SC09_DATASET_COMMIT,
    SC09_PROTOCOL,
    SC09Dataset,
    build_sc09_manifest,
    preflight_sc09_dataset,
    write_sc09_manifest,
)


DOWNLOAD_CHUNK_SIZE = 4 * 1024 * 1024
MAX_EXTRACTED_BYTES = 3_000_000_000
MAX_ARCHIVE_MEMBERS = 50_000


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(DOWNLOAD_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sc09_archive(
    path: str | Path,
    *,
    expected_size: int = SC09_ARCHIVE_SIZE,
    expected_sha256: str = SC09_ARCHIVE_SHA256,
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"SC09 archive does not exist: {path}")
    size = path.stat().st_size
    if size != expected_size:
        raise ValueError(f"SC09 archive size mismatch: {size} != {expected_size}: {path}")
    sha256 = file_sha256(path)
    if sha256 != expected_sha256:
        raise ValueError(f"SC09 archive SHA-256 mismatch: {sha256} != {expected_sha256}: {path}")
    return {"path": os.fspath(path.resolve()), "size": size, "sha256": sha256, "valid": True}


def default_sc09_archive_path(output_dir: str | Path) -> Path:
    """Return the pinned archive-cache path adjacent to the prepared dataset."""
    output_dir = Path(output_dir).expanduser().resolve()
    return output_dir.parent / "_archives" / SC09_ARCHIVE_FILENAME


def download_file_atomic(
    url: str,
    destination: str | Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> dict[str, Any]:
    """Stream a file to a sibling temporary path and publish only after verification."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.part-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    digest = hashlib.sha256()
    size = 0
    request = urllib.request.Request(url, headers={"User-Agent": "PACE-SC09/1"})
    try:
        with urllib.request.urlopen(request) as response, temporary.open("xb") as output:
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        observed_sha256 = digest.hexdigest()
        if size != expected_size:
            raise ValueError(f"Downloaded SC09 archive size mismatch: {size} != {expected_size}")
        if observed_sha256 != expected_sha256:
            raise ValueError(
                f"Downloaded SC09 archive SHA-256 mismatch: {observed_sha256} != {expected_sha256}"
            )
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return {
        "path": os.fspath(destination.resolve()),
        "url": url,
        "size": size,
        "sha256": digest.hexdigest(),
        "valid": True,
    }


def _safe_archive_name(name: str) -> PurePosixPath:
    if "\\" in name:
        raise ValueError(f"ZIP member uses a non-portable backslash path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe ZIP member path: {name!r}")
    if path.parts[0] != "sc09":
        raise ValueError(f"SC09 archive member must be rooted under `sc09/`: {name!r}")
    return path


def extract_zip_safely(archive_path: str | Path, destination: str | Path) -> Path:
    """Extract regular files only, rejecting traversal, links, encryption, and zip bombs."""
    archive_path = Path(archive_path)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Safe extraction destination must not exist: {destination}")
    destination.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise ValueError(f"SC09 archive contains too many members: {len(members):,}")
            if sum(info.file_size for info in members) > MAX_EXTRACTED_BYTES:
                raise ValueError("SC09 archive exceeds the safe uncompressed-size limit")
            seen: set[str] = set()
            validated: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
            for info in members:
                relative = _safe_archive_name(info.filename)
                normalized = relative.as_posix().rstrip("/")
                if normalized in seen:
                    raise ValueError(f"SC09 archive contains a duplicate member: {normalized}")
                seen.add(normalized)
                if info.flag_bits & 0x1:
                    raise ValueError(f"Encrypted ZIP members are not supported: {info.filename}")
                unix_mode = info.external_attr >> 16
                if unix_mode and stat.S_ISLNK(unix_mode):
                    raise ValueError(f"SC09 archive must not contain symlinks: {info.filename}")
                file_type = stat.S_IFMT(unix_mode)
                if file_type and not (stat.S_ISREG(unix_mode) or stat.S_ISDIR(unix_mode)):
                    raise ValueError(f"SC09 archive must contain only regular files/directories: {info.filename}")
                validated.append((info, relative))
            for info, relative in validated:
                target = destination.joinpath(*relative.parts)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info, "r") as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=DOWNLOAD_CHUNK_SIZE)
        extracted_root = destination / "sc09"
        if not extracted_root.is_dir():
            raise ValueError("SC09 archive did not produce the expected `sc09/` root")
        return extracted_root
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def _validate_existing_dataset(output_dir: Path, *, full_decode: bool) -> tuple[SC09Dataset, dict[str, Any]]:
    dataset = SC09Dataset(output_dir, split="all", strict_protocol=True)
    report = preflight_sc09_dataset(dataset, full_decode=full_decode)
    return dataset, report


def _acquisition_record(archive_report: dict[str, Any]) -> dict[str, Any]:
    """Persist the exact local archive verification used for this dataset."""

    return {
        "archive_url": SC09_ARCHIVE_URL,
        "archive_path": archive_report["path"],
        "archive_size": archive_report["size"],
        "archive_sha256": archive_report["sha256"],
        "archive_valid": bool(archive_report["valid"]),
        "dataset_commit": SC09_DATASET_COMMIT,
    }


def prepare_sc09_dataset(
    *,
    output_dir: str | Path,
    archive_path: str | Path | None = None,
    download: bool = True,
    reuse: bool = True,
    full_decode: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    archive_path = (
        Path(archive_path).expanduser().resolve()
        if archive_path is not None
        else default_sc09_archive_path(output_dir)
    )
    if output_dir.exists():
        if not reuse:
            raise FileExistsError(f"SC09 output already exists and reuse is disabled: {output_dir}")
        if archive_path.exists():
            archive_report = verify_sc09_archive(archive_path)
        elif download:
            archive_report = download_file_atomic(
                SC09_ARCHIVE_URL,
                archive_path,
                expected_size=SC09_ARCHIVE_SIZE,
                expected_sha256=SC09_ARCHIVE_SHA256,
            )
        else:
            raise FileNotFoundError(
                f"SC09 archive is absent and download is disabled: {archive_path}"
            )
        dataset, preflight = _validate_existing_dataset(output_dir, full_decode=full_decode)
        manifest_path = output_dir / "dataset_manifest.json"
        manifest = build_sc09_manifest(dataset, preflight=preflight)
        manifest["acquisition"] = _acquisition_record(archive_report)
        write_sc09_manifest(manifest_path, manifest)
        return {
            "status": "reused",
            "output_dir": os.fspath(output_dir),
            "archive": archive_report,
            "manifest": os.fspath(manifest_path),
            "protocol": SC09_PROTOCOL,
            "preflight": preflight,
        }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        archive_report = verify_sc09_archive(archive_path)
    elif download:
        archive_report = download_file_atomic(
            SC09_ARCHIVE_URL,
            archive_path,
            expected_size=SC09_ARCHIVE_SIZE,
            expected_sha256=SC09_ARCHIVE_SHA256,
        )
    else:
        raise FileNotFoundError(f"SC09 archive is absent and download is disabled: {archive_path}")

    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.extract-", dir=output_dir.parent))
    shutil.rmtree(staging)
    try:
        extracted_root = extract_zip_safely(archive_path, staging)
        dataset = SC09Dataset(extracted_root, split="all", strict_protocol=True)
        preflight = preflight_sc09_dataset(dataset, full_decode=full_decode)
        manifest = build_sc09_manifest(dataset, preflight=preflight)
        manifest["dataset"]["root"] = os.fspath(output_dir)
        manifest["acquisition"] = _acquisition_record(archive_report)
        write_sc09_manifest(extracted_root / "dataset_manifest.json", manifest)
        os.replace(extracted_root, output_dir)
        shutil.rmtree(staging)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    final_dataset, final_preflight = _validate_existing_dataset(output_dir, full_decode=False)
    if final_dataset.metadata["source_listing_sha256"] != dataset.metadata["source_listing_sha256"]:
        raise RuntimeError("SC09 source listing changed during atomic installation")
    return {
        "status": "prepared",
        "output_dir": os.fspath(output_dir),
        "archive": archive_report,
        "manifest": os.fspath(output_dir / "dataset_manifest.json"),
        "protocol": SC09_PROTOCOL,
        "preflight": preflight,
        "post_install_preflight": final_preflight,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--output-dir",
        "--dataset-root",
        dest="output_dir",
        default="data/sc09_v0.02",
        help="Canonical extracted SC09 directory (`--dataset-root` is an alias).",
    )
    parser.add_argument(
        "--archive",
        default=None,
        help=(
            "Pinned SC09 archive path; defaults to the reusable "
            f"<output-parent>/_archives/{SC09_ARCHIVE_FILENAME} cache."
        ),
    )
    parser.add_argument(
        "--archive-dir",
        default=None,
        help=f"Directory containing {SC09_ARCHIVE_FILENAME}; mutually exclusive with --archive.",
    )
    parser.add_argument("--no-download", action="store_true", help="Require an existing verified archive.")
    parser.add_argument("--no-reuse", action="store_true", help="Fail instead of validating an existing output directory.")
    parser.add_argument(
        "--full-decode",
        action="store_true",
        help="Decode and hash all normalized waveforms in addition to validating every WAV header.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.archive is not None and args.archive_dir is not None:
        raise SystemExit("--archive and --archive-dir are mutually exclusive")
    archive_path = (
        Path(args.archive_dir).expanduser() / SC09_ARCHIVE_FILENAME
        if args.archive_dir is not None
        else args.archive
    )
    result = prepare_sc09_dataset(
        output_dir=args.output_dir,
        archive_path=archive_path,
        download=not args.no_download,
        reuse=not args.no_reuse,
        full_decode=args.full_decode,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
