#!/usr/bin/env python3
"""Summarize completed headless car traces; never accept partial or invalid stage data."""
import argparse
import json
import math
from pathlib import Path


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "p50": None, "p95": None, "p99": None}
    return {"count": len(values), **{
        f"p{p}": values[max(0, math.ceil(len(values) * p / 100) - 1)] for p in (50, 95, 99)
    }}


def summarize(path, wall_only=False):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if not records or records[-1].get("type") != "summary":
        raise ValueError(f"{path}: incomplete run")
    rows = [row for row in records if row.get("type") == "tick"]
    if len(rows) != records[-1]["samples"] or not rows:
        raise ValueError(f"{path}: missing ticks")
    if any(b["tick"] != a["tick"] + 1 for a, b in zip(rows, rows[1:])):
        raise ValueError(f"{path}: nonconsecutive ticks")
    invalid_nested = sum(row["recovery_projection_ms"] > row["recovery_ms"] + 1e-6 for row in rows)
    if invalid_nested and not wall_only:
        raise ValueError(f"{path}: invalid nested GPU timestamps")
    invalid_total = sum(row["gpu_ms"] > row["wall_ms"] + 1 for row in rows)
    if invalid_total and not wall_only:
        raise ValueError(f"{path}: GPU timestamp total exceeds wall latency")
    fields = ["wall_ms", "preparation_ms", "publication_ms"]
    if not wall_only:
        fields += ["gpu_ms", "terrain_ms", "rotation_ms", "recovery_ms", "recovery_projection_ms", "solver_ms"]
    publications = [row for row in rows if row["publication_ms"] > 0]
    uploads = [row["uploaded_bytes"] for row in rows if "uploaded_bytes" in row]
    return {
        "file": str(path), "plane": rows[0]["plane"], "samples": len(rows),
        "completed_tps": records[-1]["completed_tps"],
        "simulated_seconds": len(rows) / 60,
        "wall_seconds": len(rows) / records[-1]["completed_tps"],
        "latency_ms": {key: distribution([row[key] for row in rows]) for key in fields},
        "gpu_timing_evaluated": not wall_only,
        "invalid_nested_gpu_samples": invalid_nested,
        "gpu_total_over_wall_samples": invalid_total,
        "publication_events": len(publications),
        "publication_event_ms": {key: distribution([row[key] for row in publications])
                                 for key in ("preparation_ms", "publication_ms")},
        "uploaded_bytes": sum(uploads) if uploads else None,
        "failure_flags": sorted({row["failure_flags"] for row in rows}),
        "minimum_up_y": min(row["up_y"] for row in rows),
        "serialized_readback": True,
        "world_acceptance_evaluated": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--wall-only", action="store_true", help="Omit all GPU distributions when instrumentation is known unreliable")
    args = parser.parse_args()
    print(json.dumps([summarize(path, args.wall_only) for path in args.traces], indent=2))
