#!/usr/bin/env python3
"""Smoke-test an isolated OpenAI ADM evaluator environment without downloads."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator", type=Path, required=True, help="Pinned guided-diffusion evaluations/evaluator.py.")
    args = parser.parse_args()
    evaluator_path = args.evaluator.expanduser().resolve()
    if not evaluator_path.is_file():
        parser.error(f"evaluator does not exist: {evaluator_path}")

    spec = importlib.util.spec_from_file_location("diffdist_openai_adm_evaluator_smoke", evaluator_path)
    if spec is None or spec.loader is None:
        parser.error(f"could not construct an import specification for {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    stats = module.FIDStatistics(np.zeros(2), np.eye(2))
    fid = float(stats.frechet_distance(stats))
    if abs(fid) > 1e-9:
        raise RuntimeError(f"ADM FIDStatistics identical-fixture check returned {fid}, expected 0")
    import scipy
    import tensorflow as tensorflow_module

    payload = {
        "status": "pass",
        "evaluator": str(evaluator_path),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "tensorflow_version": tensorflow_module.__version__,
        "inception_url": module.INCEPTION_V3_URL,
        "identical_fixture_fid": fid,
        "network_or_reference_downloaded": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
