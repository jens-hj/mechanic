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
    if metadata.get("schema") != 3:
        raise ValueError(f"{path}: capture schema 3 required; use archived tools for historical captures")
    duration = metadata["duration_seconds"]
    submission_duration = metadata["submission_duration_seconds"]
    if not isinstance(submission_duration, (int, float)) or not 0 < submission_duration <= duration:
        raise ValueError(f"{path}: invalid capture duration or missing submission boundary")
    groups = {}
    drain_started = False
    for event in events:
        phase = event.get("phase")
        if phase not in ("measurement", "drain") or (drain_started and phase != "drain"):
            raise ValueError(f"{path}: invalid capture phase ordering")
        drain_started |= phase == "drain"
        if phase == "drain" and event["kind"] in ("physics_submit", "physics_drop", "driving_input", "physics_drive_rows"):
            raise ValueError(f"{path}: new simulation work during drain")
        if phase == "measurement" or event["kind"] in ("physics_readback", "physics_publication", "physics_poll"):
            groups.setdefault(event["kind"], []).append(event)
    def values(kind, field):
        return [e["data"][field] for e in groups.get(kind, []) if e["data"].get(field) is not None]
    def stats(kind, field):
        samples = values(kind, field)
        return {"count": len(samples), "p50": percentile(samples, 50), "p95": percentile(samples, 95), "p99": percentile(samples, 99)}
    # Ordinals follow the actual submissions, including those made before capture.
    # A completion's scheduler tick gap belongs to its submission interval, not
    # the frame interval in which its delayed readback happens to arrive.
    reported_drops = sum(values("physics_drop", "count"))
    drop_ranges = [(event["data"]["first_tick"], event["data"]["first_tick"] + event["data"]["count"])
                   for event in groups.get("physics_drop", [])]
    if any(end <= start for start, end in drop_ranges) or any(b[0] < a[1] for a, b in zip(drop_ranges, drop_ranges[1:])):
        raise ValueError(f"{path}: invalid or overlapping scheduler drop ranges")
    streams = {}
    for kind in ("physics_readback", "physics_submit"):
        ticks = values(kind, "tick")
        sequences = values(kind, "sequence")
        if len(sequences) != len(ticks):
            raise ValueError(f"{path}: {kind} missing submission sequence")
        if any(b <= a for a, b in zip(ticks, ticks[1:])):
            raise ValueError(f"{path}: {kind} ticks repeat or move backwards (scene reset)")
        if any(b != a + 1 for a, b in zip(sequences, sequences[1:])):
            raise ValueError(f"{path}: {kind} skips or repeats submission sequences (missing events or reset)")
        streams[kind] = dict(zip(sequences, ticks))
        if kind == "physics_submit":
            for a, b in zip(ticks, ticks[1:]):
                explained = sum(max(0, min(b, end) - max(a + 1, start)) for start, end in drop_ranges)
                if explained != b - a - 1:
                    raise ValueError(f"{path}: submission tick gap has no exact scheduler drop range")
            if any(start <= tick < end for tick in ticks for start, end in drop_ranges):
                raise ValueError(f"{path}: a submitted tick was also reported dropped")
    submitted, completed = streams["physics_submit"], streams["physics_readback"]
    if any(submitted[seq] != completed[seq] for seq in submitted.keys() & completed.keys()):
        raise ValueError(f"{path}: submission/completion tick identities disagree")
    frames = values("frame", "frame_ms")
    gpu = groups.get("render_gpu", [])
    sample_ids = [e["data"]["sample_id"] for e in gpu]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{path}: duplicated asynchronous GPU sample")
    frame_data = [e["data"] for e in groups.get("frame", [])]
    settings = ["f3", "present_mode", "window_pixels", "target_pixels", "viewport_pixels", "msaa"]
    input_ticks = values("driving_input", "tick")
    script_ticks = values("driving_input", "script_tick")
    submitted_ticks = values("physics_submit", "tick")
    driving_matches = bool(input_ticks) and input_ticks == submitted_ticks and len(script_ticks) == len(input_ticks)
    if driving_matches:
        driving_matches = all(script == tick - input_ticks[0] for tick, script in zip(input_ticks, script_ticks))
    publications = values("physics_publication", "tick")
    if len(publications) != len(set(publications)):
        raise ValueError(f"{path}: duplicated publication")
    missing_publications = set(submitted_ticks) - set(publications)
    pending = submitted.keys() - completed.keys()
    replay_ticks = metadata["metadata"].get("replay_ticks")
    if replay_ticks is not None and (len(submitted) != replay_ticks or reported_drops or pending or missing_publications):
        raise ValueError(f"{path}: incomplete fixed-duration replay")
    return {
        "file": str(path), "metadata": metadata["metadata"],
        "duration_seconds": duration, "submission_duration_seconds": submission_duration,
        "drain_duration_seconds": duration - submission_duration,
        "drain_complete": not pending and not missing_publications,
        "drained_readbacks": sum(e["phase"] == "drain" for e in groups.get("physics_readback", [])),
        "fps": 1000 * len(frames) / sum(frames) if frames else None,
        "completed_tps": len(groups.get("physics_readback", [])) / metadata["duration_seconds"],
        "submitted_tps": len(groups.get("physics_submit", [])) / submission_duration,
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
        "physics_terrain_stages_ms": {field: stats("physics_readback", field) for field in
                                     ("terrain_ms", "rotational_sweep_ms", "terrain_recovery_ms", "recovery_projection_ms", "contact_solver_ms")},
        "terrain_publication": {field: stats("terrain_publication", field) for field in
                                ("preparation_ms", "publication_ms", "latency_ms", "uploaded_bytes", "reused_chunks", "uploaded_chunks")},
        "terrain_uploaded_bytes": sum(values("terrain_publication", "uploaded_bytes")),
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
        "dropped_ticks": reported_drops,
        "completed_interval_skipped_ticks": sum(b - a - 1 for a, b in zip(values("physics_readback", "tick"), values("physics_readback", "tick")[1:])),
        "boundary_readbacks": sum(seq not in submitted for seq in completed),
        "pending_at_capture_end": sum(seq not in completed for seq in submitted), "backlog_start_end": [frame_data[0]["backlog"], frame_data[-1]["backlog"]] if frame_data else None,
        "settings_seen": {key: sorted({json.dumps(f[key]) for f in frame_data}) for key in settings},
        "terrain_busy_frames": sum(f["terrain_backlog"] != 0 for f in frame_data),
        "terrain_readiness_holds": len(groups.get("physics_terrain_hold", [])),
        "driving_input": {"count": len(input_ticks), "matches_submitted_ticks": driving_matches,
                          "first_script_tick": script_ticks[0] if script_ticks else None,
                          "last_script_tick": script_ticks[-1] if script_ticks else None},
        "physics_error_flags": sorted(set(values("physics_readback", "error_flags"))),
        "observed_execution_masks": sorted(set(values("physics_readback", "executed_stage_mask"))),
        "physics_execution": {field: stats("physics_readback", field) for field in
                              ("integrated_bodies", "published_bodies", "validated_bearings",
                               "planned_solver_sweeps", "executed_solver_sweeps", "anchor_residual_m", "axis_residual_deg")},
        "gpu_samples": len(gpu),
        "gpu_status_counts": {status:sum(e["data"]["status"] == status for e in gpu) for status in sorted({e["data"]["status"] for e in gpu})},
        "gpu_samples_originating_before_capture": sum(e["data"]["sample_age_ms"] > e["elapsed_ms"] for e in gpu),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([summarize(path) for path in args.captures], indent=2))
