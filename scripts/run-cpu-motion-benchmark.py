#!/usr/bin/env python3
"""Run three sequential release-binary pairs for the generated CPU motion cases.

Build the two binaries in separate target directories first. Keep profiling,
compilation, and application captures out of this timing run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(baseline, candidate, output, warmup, ticks):
    binaries = {"baseline": baseline.resolve(), "candidate": candidate.resolve()}
    hashes = {name: digest(path) for name, path in binaries.items()}
    if hashes["baseline"] == hashes["candidate"]:
        raise ValueError("baseline and candidate are the same executable")
    if warmup < 0 or ticks <= 0:
        raise ValueError("warmup must be nonnegative and ticks positive")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"binaries": {name: str(path) for name, path in binaries.items()},
                "sha256": hashes, "warmup": warmup, "ticks": ticks, "pairs": 3,
                "order": [["baseline", "candidate"], ["candidate", "baseline"],
                          ["baseline", "candidate"]]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    cases = {}
    for pair, order in enumerate(manifest["order"], 1):
        for name in order:
            result = subprocess.run(
                [str(binaries[name]), "--scenario", "fast-motion", "--warmup", str(warmup),
                 "--ticks", str(ticks)], check=True, capture_output=True, text=True)
            (output / f"{name}-{pair}.jsonl").write_text(result.stdout)
            for line in result.stdout.splitlines():
                record = json.loads(line)
                key = (record["scenario"], record["unrelated"], record["speed"])
                cases.setdefault(key, {"baseline": [], "candidate": []})[name].append(record)
            print(f"completed {name} pair {pair}", flush=True)
    summary = []
    for (scenario, unrelated, speed), records in cases.items():
        before, after = records["baseline"], records["candidate"]
        if len(before) != 3 or len(after) != 3:
            raise ValueError("case matrix changed between binaries or runs")
        summary.append({
            "scenario": scenario, "unrelated": unrelated, "speed": speed,
            "baseline_p50_ms": statistics.median(r["p50_ms"] for r in before),
            "candidate_p50_ms": statistics.median(r["p50_ms"] for r in after),
            "baseline_p95_ms": statistics.median(r["p95_ms"] for r in before),
            "candidate_p95_ms": statistics.median(r["p95_ms"] for r in after),
            "p50_ratios": [b["p50_ms"] / a["p50_ms"] for a, b in zip(before, after)],
            "p95_ratios": [b["p95_ms"] / a["p95_ms"] for a, b in zip(before, after)],
            "matching_states": all(a["state_hash"] == b["state_hash"]
                                   for a, b in zip(before, after)),
            "baseline_degraded_ticks": sum(r["degraded_ticks"] for r in before),
            "candidate_degraded_ticks": sum(r["degraded_ticks"] for r in after),
        })
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--warmup", type=int, default=120)
    parser.add_argument("--ticks", type=int, default=600)
    args = parser.parse_args()
    run(args.baseline, args.candidate, args.output, args.warmup, args.ticks)
