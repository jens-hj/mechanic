#!/usr/bin/env python3
"""Check paired foreground captures before interpreting their performance.

Compares complete submitted workloads; a shared prefix is diagnostic only.
Use --repeat for identical builds/backends and require completed-state hashes.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("capture_summary", Path(__file__).with_name("summarize-perf-capture.py"))
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def read(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    inputs, states, drives = {}, {}, {}
    for event in records[1:-1]:
        data = event["data"]
        if event["kind"] == "driving_input":
            tick = data["script_tick"]
            if tick in inputs:
                raise ValueError("duplicate scripted input tick")
            inputs[tick] = (data["tick"], data["held"])
        elif event["kind"] == "physics_drive_rows":
            if data["tick"] in drives:
                raise ValueError("duplicate drive rows")
            drives[data["tick"]] = data["hash"]
        elif event["kind"] == "physics_publication":
            if data["tick"] in states:
                raise ValueError("duplicate published tick")
            states[data["tick"]] = data.get("state_hash")
    return summary.summarize(path), inputs, states, drives


def compare(left, right, repeat=False):
    a, ai, ast, ad = read(left)
    b, bi, bst, bd = read(right)
    reasons = []
    for key in ("initial_state_hash", "initial_camera_matrix", "initial_terrain_fingerprint", "initial_terrain_layout_fingerprint", "adapter"):
        if a["metadata"].get(key) is None or a["metadata"].get(key) != b["metadata"].get(key):
            reasons.append(f"missing or different {key}")
    for capture in (a, b):
        if not capture["metadata"].get("foreground_requested") or capture["metadata"].get("automated_background"):
            reasons.append("foreground protocol was not requested")
        if capture["focused_frame_counts"]["true"] != capture["frame_ms"]["count"] or not capture["frame_ms"]["count"]:
            reasons.append("capture includes unfocused or missing frames")
        if capture["dropped_ticks"] or not capture["driving_input"]["matches_submitted_ticks"]:
            reasons.append("dropped ticks or incomplete scripted input")
        if capture["physics_error_flags"] != [0]:
            reasons.append("physics failed or produced no diagnostics")
    if a["settings_seen"] != b["settings_seen"] or any(len(values) != 1 for values in a["settings_seen"].values()):
        reasons.append("different or changing render settings")
    # Submission ordinals may differ at the boundary; compare physical script time.
    if not ai or {tick: value[1] for tick, value in ai.items()} != {tick: value[1] for tick, value in bi.items()}:
        reasons.append("different simulated durations or input scripts")
    shared = sorted(ai.keys() & bi.keys())
    drive_mismatch = next((tick for tick in shared if ad.get(ai[tick][0]) != bd.get(bi[tick][0])), None)
    if any(ai[tick][0] not in ad or bi[tick][0] not in bd for tick in shared):
        reasons.append("missing effective drive rows")
    if repeat and drive_mismatch is not None:
        reasons.append("effective drive rows differ")
    mismatch = next((tick for tick in shared
                     if ast.get(ai[tick][0]) is not None and bst.get(bi[tick][0]) is not None
                     and ast[ai[tick][0]] != bst[bi[tick][0]]), None)
    missing = [tick for tick in shared if ast.get(ai[tick][0]) is None or bst.get(bi[tick][0]) is None]
    if missing:
        reasons.append("submitted states missing at capture boundary or publication")
    if repeat and mismatch is not None:
        reasons.append("completed-state hashes differ")
    # World and build identities are written beside the raw capture by the runner.
    manifests = [json.loads(path.with_name("run.json").read_text()) for path in (left, right)]
    for key in ("source_world_files_sha256", "assets_files_sha256"):
        if not manifests[0].get(key) or manifests[0].get(key) != manifests[1].get(key):
            reasons.append(f"missing or different {key}")
    if repeat and (not manifests[0].get("binary_sha256") or manifests[0].get("binary_sha256") != manifests[1].get("binary_sha256")):
        reasons.append("repeatability requires identical builds")
    return {"comparable": not reasons, "rejections": sorted(set(reasons)),
            "repeat_requested": repeat, "shared_script_ticks": len(shared),
            "first_state_hash_mismatch_script_tick": mismatch,
            "first_drive_hash_mismatch_script_tick": drive_mismatch,
            "missing_published_states": len(missing),
            "physics_acceptance_passed": False,
            "note": "Full kernel coverage and physical-bound acceptance require independent evidence."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--repeat", action="store_true")
    args = parser.parse_args()
    result = compare(args.left, args.right, args.repeat)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["comparable"] else 1)
