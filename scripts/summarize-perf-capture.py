#!/usr/bin/env python3
"""Summarize raw capture events without averaging smoothed values or overlapping GPU spans."""
import argparse
import json
import math
from pathlib import Path


def percentile(values, percent):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * percent / 100) - 1)] if values else None


def summarize(path):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if not records or records[-1].get("kind") != "result" or not records[-1]["valid"]:
        raise ValueError(f"{path}: missing valid completion footer")
    metadata = records[0]
    events = records[1:-1]
    groups = {}
    for event in events:
        groups.setdefault(event["kind"], []).append(event)
    def values(kind, field):
        return [e["data"][field] for e in groups.get(kind, []) if e["data"].get(field) is not None]
    def stats(kind, field):
        samples = values(kind, field)
        return {"count": len(samples), "p50": percentile(samples, 50), "p95": percentile(samples, 95), "p99": percentile(samples, 99)}
    for kind in ("physics_readback", "physics_submit"):
        ticks = values(kind, "tick")
        if any(b != a + 1 for a, b in zip(ticks, ticks[1:])):
            raise ValueError(f"{path}: nonconsecutive {kind} ticks (scene reset or missing events)")
    frames = values("frame", "frame_ms")
    gpu = groups.get("render_gpu", [])
    sample_ids = [e["data"]["sample_id"] for e in gpu]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{path}: duplicated asynchronous GPU sample")
    frame_data = [e["data"] for e in groups.get("frame", [])]
    settings = ["f3", "present_mode", "window_pixels", "target_pixels", "viewport_pixels", "msaa"]
    return {
        "file": str(path), "metadata": metadata["metadata"],
        "fps": 1000 * len(frames) / sum(frames) if frames else None,
        "completed_tps": len(groups.get("physics_readback", [])) / metadata["duration_seconds"],
        "submitted_tps": len(groups.get("physics_submit", [])) / metadata["duration_seconds"],
        "frame_ms": stats("frame", "frame_ms"),
        "queue_submission_ms": stats("physics_submit", "submission_ms"),
        "readback_latency_ms": stats("physics_readback", "latency_ms"),
        "submission_to_callbacks_ms": stats("physics_readback", "submission_to_callbacks_ms"),
        "callback_to_publication_ms": stats("physics_publication", "callback_to_publication_ms"),
        "callbacks_during_poll_counts": {str(value).lower(): values("physics_readback", "callbacks_during_poll").count(value) for value in (True, False)},
        "physics_poll_ms": stats("physics_poll", "duration_ms"),
        "physics_poll_outcomes": {value: values("physics_poll", "outcome").count(value) for value in ("completed", "empty", "error")},
        "acquire_ms": stats("render_acquire", "duration_ms"),
        "render_cpu_ms": stats("render_cpu", "duration_ms"),
        "physics_gpu_ms": stats("physics_readback", "gpu_tick_ms"),
        "physics_cpu_stages_ms": {field: stats("physics_submit", field) for field in
                                  ("encoding_ms", "finalization_ms", "submission_ms", "readback_setup_ms")},
        # Independent distributions: these spans may overlap and must not be added.
        "render_gpu_ms": {field: stats("render_gpu", field) for field in
                          ("tracked_span_ms", "prepass_ms", "opaque_ms", "terrain_ms", "opaque_other_ms", "transparent_ms", "xray_ms", "other_ms")},
        "terrain_passes_per_sample": stats("render_gpu", "terrain_passes"),
        "render_gpu_sample_age_ms": stats("render_gpu", "sample_age_ms"),
        "in_flight_slots": stats("frame", "in_flight"),
        "focused_frame_counts": {str(value).lower(): sum(f.get("focused") == value for f in frame_data)
                                 for value in (True, False)},
        "backlog_start_end": [frame_data[0]["backlog"], frame_data[-1]["backlog"]] if frame_data else None,
        "settings_seen": {key: sorted({json.dumps(f[key]) for f in frame_data}) for key in settings},
        "terrain_busy_frames": sum(f["terrain_backlog"] != 0 for f in frame_data),
        "physics_error_flags": sorted(set(values("physics_readback", "error_flags"))),
        "gpu_samples": len(gpu),
        "gpu_status_counts": {status:sum(e["data"]["status"] == status for e in gpu) for status in sorted({e["data"]["status"] for e in gpu})},
        "gpu_samples_originating_before_capture": sum(e["data"]["sample_age_ms"] > e["elapsed_ms"] for e in gpu),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([summarize(path) for path in args.captures], indent=2))
