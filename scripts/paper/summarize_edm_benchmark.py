#!/usr/bin/env python3
"""Aggregate repeated benchmark result files without recomputing metrics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def nested_metric(payload: dict, metric: str) -> float:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or metric not in metrics:
        raise ValueError(f"result does not contain metrics[{metric!r}]")
    return float(metrics[metric])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--aggregation", choices=("minimum", "mean"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = []
    for path in args.result:
        payload = json.loads(path.read_text())
        declared = payload.get("benchmark_protocol_id") or payload.get("declared_protocol_id")
        if declared != args.protocol_id:
            raise ValueError(f"{path} declares protocol {declared!r}, expected {args.protocol_id!r}")
        runs.append({"path": str(path.resolve()), "value": nested_metric(payload, args.metric)})
    values = [run["value"] for run in runs]
    aggregate = min(values) if args.aggregation == "minimum" else sum(values) / len(values)
    selected_index = values.index(min(values)) if args.aggregation == "minimum" else None
    output = {
        "summary_format": "diffdist_edm_benchmark_summary_v1",
        "protocol_id": args.protocol_id,
        "metric": args.metric,
        "aggregation": args.aggregation,
        "aggregate": aggregate,
        "selected_run_index": selected_index,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, args.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
