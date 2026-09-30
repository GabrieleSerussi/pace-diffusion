#!/usr/bin/env python3
"""Create a reproducible FFHQ/LSUN manifest and preflight every selected image."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pace.dataset_specs import (
    DEFAULT_LSUN_MONITOR_SEED,
    DEFAULT_LSUN_MONITOR_SIZE,
    FFHQ_PROTOCOL,
)
from pace.image_datasets import (
    SharedImageDataset,
    build_dataset_manifest,
    preflight_dataset,
    write_dataset_manifest,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", choices=["ffhq", "lsun_bedroom"], required=True)
    parser.add_argument("--data-root", required=True, help="Recursive image directory or ZIP archive.")
    parser.add_argument("--dataset-manifest", required=True, help="Output path for the versioned dataset manifest.")
    parser.add_argument(
        "--dataset-split",
        choices=["all", "train", "monitor", "fid"],
        default="train",
        help="Split whose files are decoded during preflight.",
    )
    parser.add_argument("--dataset-preflight", action="store_true", help="Accepted for command symmetry; preflight is always performed by this command.")
    parser.add_argument("--image-size", type=int, default=256, help="Expected FFHQ size or minimum LSUN short side.")
    parser.add_argument("--ffhq-protocol", default=FFHQ_PROTOCOL)
    parser.add_argument("--lsun-monitor-size", type=int, default=DEFAULT_LSUN_MONITOR_SIZE)
    parser.add_argument("--lsun-monitor-seed", type=int, default=DEFAULT_LSUN_MONITOR_SEED)
    parser.add_argument("--report", default=None, help="Optional JSON path for the successful preflight report.")
    return parser


def create_manifest_and_preflight(args: argparse.Namespace) -> dict:
    manifest = build_dataset_manifest(
        dataset_id=args.dataset,
        root=args.data_root,
        protocol=args.ffhq_protocol if args.dataset == "ffhq" else None,
        lsun_monitor_size=args.lsun_monitor_size,
        lsun_monitor_seed=args.lsun_monitor_seed,
        strict_protocol=True,
    )
    # Preflight against the in-memory split manifest.  The declared manifest
    # output is written only after every selected image passes, so a failed run
    # cannot leave behind an apparently usable production manifest.
    temporary_manifest = Path(args.dataset_manifest).expanduser().with_name(
        f".{Path(args.dataset_manifest).name}.preflight-{os.getpid()}.json"
    )
    write_dataset_manifest(temporary_manifest, manifest)
    dataset = SharedImageDataset(
        dataset_id=args.dataset,
        root=args.data_root,
        image_size=args.image_size,
        split=args.dataset_split,
        manifest=temporary_manifest,
        ffhq_protocol=args.ffhq_protocol,
        lsun_monitor_size=args.lsun_monitor_size,
        lsun_monitor_seed=args.lsun_monitor_seed,
    )
    try:
        report = preflight_dataset(dataset, full_decode=True)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    report["dataset_manifest"] = str(Path(args.dataset_manifest).expanduser().resolve())
    manifest["preflight"] = {
        "valid": True,
        "checked": report["checked"],
        "content_sha256": report["content_sha256"],
        "modes": report["modes"],
        "sizes": report["sizes"],
    }
    write_dataset_manifest(args.dataset_manifest, manifest)
    if args.report is not None:
        report_path = Path(args.report).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    args = build_arg_parser().parse_args()
    report = create_manifest_and_preflight(args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
