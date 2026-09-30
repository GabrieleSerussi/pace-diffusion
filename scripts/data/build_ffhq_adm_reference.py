#!/usr/bin/env python3
"""Build the custom FFHQ-64 ADM reference from prepared IDs 00000--49999."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pace.evaluation_protocols import (
    FFHQ64_ADM_CUSTOM_PROTOCOL,
    atomic_write_json,
    build_ffhq64_adm_reference,
    file_digest,
)


PINNED_GUIDED_DIFFUSION_COMMIT = "22e0df8183507e13a7813f8d38d51b072ca1e67c"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--prepared-root", type=Path, required=True, help="Prepared 00000.png...69999.png FFHQ-64 root.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool-features", type=Path, default=None, help="Optional pinned OpenAI pool_3 activations for IDs 0--49999.")
    parser.add_argument("--spatial-features", type=Path, default=None, help="Optional pinned OpenAI mixed_6/conv activations for IDs 0--49999.")
    parser.add_argument("--adm-evaluator", type=Path, default=None, help="Pinned guided-diffusion evaluations/evaluator.py for direct extraction.")
    parser.add_argument("--adm-python", type=Path, default=None, help="Python from the isolated ADM environment for direct extraction.")
    parser.add_argument("--batch-size", type=int, default=64)
    detector_group = parser.add_mutually_exclusive_group()
    detector_group.add_argument(
        "--adm-detector",
        type=Path,
        default=None,
        help="Existing canonical classify_image_graph_def.pb; extraction runs in its parent directory.",
    )
    detector_group.add_argument(
        "--detector-cache-dir",
        type=Path,
        default=None,
        help="Legacy cache directory where upstream may download classify_image_graph_def.pb.",
    )
    parser.add_argument("--expected-evaluator-commit", default=PINNED_GUIDED_DIFFUSION_COMMIT)
    parser.add_argument("--_extract-direct", action="store_true", help=argparse.SUPPRESS)
    return parser


def _checkout_commit(evaluator_path: Path) -> str:
    checkout = evaluator_path.resolve().parent.parent
    try:
        result = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"ADM evaluator must live in a pinned Git checkout: {checkout}: {exc}") from exc
    return result.stdout.strip()


def _import_evaluator(path: Path):
    spec = importlib.util.spec_from_file_location("diffdist_ffhq_adm_reference_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import ADM evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ffhq_batches(root: Path, *, batch_size: int) -> Iterator[np.ndarray]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    paths = [root / f"{image_id:05d}.png" for image_id in range(50_000)]
    missing = [str(path) for path in paths if not path.is_file()][:5]
    if missing:
        raise FileNotFoundError(f"prepared FFHQ IDs 00000--49999 are incomplete; first missing: {missing}")
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        pixels = np.empty((len(batch_paths), 64, 64, 3), dtype=np.uint8)
        for index, path in enumerate(batch_paths):
            with Image.open(path) as image:
                if image.mode != "RGB" or image.size != (64, 64):
                    raise ValueError(f"prepared FFHQ image must be exact 64x64 RGB: {path}")
                pixels[index] = np.asarray(image, dtype=np.uint8)
        yield pixels


def _extract_direct(args: argparse.Namespace) -> dict:
    evaluator_path = args.adm_evaluator.expanduser().resolve()
    prepared_root = args.prepared_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    actual_commit = _checkout_commit(evaluator_path)
    if actual_commit != args.expected_evaluator_commit:
        raise RuntimeError(
            f"guided-diffusion commit mismatch: expected {args.expected_evaluator_commit}, got {actual_commit}"
        )
    detector = args.adm_detector.expanduser().resolve() if args.adm_detector is not None else None
    if detector is not None:
        expected_detector = FFHQ64_ADM_CUSTOM_PROTOCOL.detector
        if detector.name != "classify_image_graph_def.pb" or not detector.is_file():
            raise FileNotFoundError(
                "--adm-detector must be an existing file named classify_image_graph_def.pb: "
                f"{detector}"
            )
        if detector.stat().st_size != expected_detector.size_bytes:
            raise RuntimeError(
                f"OpenAI detector size mismatch: expected {expected_detector.size_bytes}, "
                f"got {detector.stat().st_size}"
            )
        detector_md5 = file_digest(detector, "md5")
        if detector_md5 != expected_detector.md5:
            raise RuntimeError(f"OpenAI detector MD5 mismatch: expected {expected_detector.md5}, got {detector_md5}")
        cache_dir = detector.parent
    else:
        cache_dir = (
            args.detector_cache_dir.expanduser().resolve()
            if args.detector_cache_dir is not None
            else output.parent / "openai_adm_detector"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    previous_cwd = Path.cwd()
    try:
        os.chdir(cache_dir)
        upstream = _import_evaluator(evaluator_path)
        config = upstream.tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with upstream.tf.Session(config=config) as session:
            evaluator = upstream.Evaluator(session, batch_size=args.batch_size)
            evaluator.warmup()
            pool_features, spatial_features = evaluator.compute_activations(
                _ffhq_batches(prepared_root, batch_size=args.batch_size)
            )
    finally:
        os.chdir(previous_cwd)

    detector = cache_dir / "classify_image_graph_def.pb"
    expected_detector = FFHQ64_ADM_CUSTOM_PROTOCOL.detector
    if not detector.is_file() or detector.stat().st_size != expected_detector.size_bytes:
        raise RuntimeError(f"OpenAI detector size mismatch or missing: {detector}")
    detector_md5 = file_digest(detector, "md5")
    if detector_md5 != expected_detector.md5:
        raise RuntimeError(f"OpenAI detector MD5 mismatch: expected {expected_detector.md5}, got {detector_md5}")

    result = build_ffhq64_adm_reference(
        output,
        prepared_image_root=prepared_root,
        pool_features=pool_features,
        spatial_features=spatial_features,
    )
    result["feature_extraction"] = {
        "backend": "openai_guided_diffusion_evaluator",
        "evaluator": str(evaluator_path),
        "evaluator_sha256": file_digest(evaluator_path),
        "guided_diffusion_commit": actual_commit,
        "detector": str(detector),
        "detector_md5": detector_md5,
        "batch_size": args.batch_size,
        "image_ids": [0, 49_999],
    }
    atomic_write_json(output.with_suffix(output.suffix + ".manifest.json"), result)
    return result


def _run_isolated(args: argparse.Namespace) -> None:
    if args.adm_evaluator is None or args.adm_python is None:
        raise ValueError(
            "direct feature extraction requires --adm-evaluator and --adm-python; alternatively provide both "
            "--pool-features and --spatial-features"
        )
    # Preserve the venv entry-point path. Resolving ``bin/python`` to its base
    # interpreter drops the virtual environment and its TensorFlow install.
    python_path = Path(os.path.abspath(args.adm_python.expanduser()))
    if not python_path.is_file() or not os.access(python_path, os.X_OK):
        raise PermissionError(f"isolated ADM Python is missing or not executable: {python_path}")
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python_path),
        str(Path(__file__).resolve()),
        "--prepared-root",
        str(args.prepared_root),
        "--output",
        str(output),
        "--adm-evaluator",
        str(args.adm_evaluator),
        "--batch-size",
        str(args.batch_size),
        "--expected-evaluator-commit",
        args.expected_evaluator_commit,
        "--_extract-direct",
    ]
    if args.detector_cache_dir is not None:
        command += ["--detector-cache-dir", str(args.detector_cache_dir)]
    if args.adm_detector is not None:
        command += ["--adm-detector", str(args.adm_detector)]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), environment.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    subprocess.run(command, cwd=output.parent, env=environment, check=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args._extract_direct:
        if args.adm_evaluator is None:
            parser.error("--_extract-direct requires --adm-evaluator")
        result = _extract_direct(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    feature_pair = (args.pool_features is not None, args.spatial_features is not None)
    if feature_pair == (True, True):
        result = build_ffhq64_adm_reference(
            args.output,
            prepared_image_root=args.prepared_root,
            pool_features=args.pool_features,
            spatial_features=args.spatial_features,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if any(feature_pair):
        parser.error("--pool-features and --spatial-features must be provided together")
    _run_isolated(args)


if __name__ == "__main__":
    main()
