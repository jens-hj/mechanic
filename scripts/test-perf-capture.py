#!/usr/bin/env python3
"""Regression checks for capture summaries (no GPU needed)."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import sys

sys.dont_write_bytecode = True

spec = importlib.util.spec_from_file_location("summary", Path(__file__).with_name("summarize-perf-capture.py"))
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class CaptureSummaryTests(unittest.TestCase):
    def test_freeze_summary_keeps_active_hitches_out_of_idle_percentiles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "freeze.jsonl"
            self.write_capture(path, [
                {"kind": "freeze_stage", "data": {"stage": "total", "ms": 0.01}},
                {"kind": "freeze_stage", "data": {"stage": "internal_path", "ms": 19}},
                {"kind": "freeze_stage", "data": {"stage": "planning", "ms": 20}},
                {"kind": "freeze_stage", "data": {"stage": "total", "ms": 21}},
            ])
            result = summary.summarize(path)
            self.assertEqual(result["freeze_stages_ms"]["active_total"],
                             {"count": 1, "p95": 21, "max": 21})
            self.assertEqual(result["freeze_stages_ms"]["internal_path"]["max"], 19)

    def test_freeze_input_cadence_uses_update_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "freeze.jsonl"
            self.write_capture(path, [
                {"kind": "freeze_input", "elapsed_ms": 1, "data": {"toggle": True}},
                {"kind": "freeze_stage", "elapsed_ms": 31, "data": {"stage": "total", "ms": 30}},
                {"kind": "freeze_stage", "elapsed_ms": 42, "data": {"stage": "total", "ms": 1}},
            ])
            result = summary.summarize(path)
            self.assertEqual(result["freeze_input_processing_ms"]["max"], 30)
            self.assertEqual(result["freeze_input_update_interval_ms"]["max"], 40)

    def test_cpu_completions_count_without_gpu_timings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cpu.jsonl"
            self.write_capture(path, [
                {"kind":"physics_submit", "data":{"tick":1,"sequence":1,"route":"cpu"}},
                {"kind":"physics_cpu_tick", "data":{"tick":1,"duration_ms":1.5,"degraded":False}},
                {"kind":"physics_publication", "data":{"tick":1,"route":"cpu"}},
                {"kind":"physics_readback", "data":{"tick":1,"sequence":1,"route":"cpu","error_flags":0}},
            ])
            result = summary.summarize(path)
            self.assertEqual(result["physics_routes"], ["cpu"])
            self.assertTrue(result["drain_complete"])
            self.assertEqual(result["cpu_tick_ms"]["p95"], 1.5)
            self.assertEqual(result["physics_gpu_ms"]["count"], 0)
            self.assertEqual(result["cpu_degraded_ticks"], 0)

    def test_replay_includes_drained_states_without_measuring_idle_drain_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            records = [
                {"kind": "metadata", "schema": 3, "duration_seconds": 2, "submission_duration_seconds": 1,
                 "metadata": {"replay_ticks": 1}},
                {"kind": "physics_submit", "phase": "measurement", "data": {"tick": 1, "sequence": 1}},
                {"kind": "physics_readback", "phase": "drain", "data": {"tick": 1, "sequence": 1, "gpu_tick_ms": 4}},
                {"kind": "physics_publication", "phase": "drain", "data": {"tick": 1, "state_hash": "state"}},
                {"kind": "frame", "phase": "drain", "data": {"frame_ms": 1000}},
                {"kind": "result", "valid": True},
            ]
            path.write_text("\n".join(map(json.dumps, records)))
            result = summary.summarize(path)
            self.assertTrue(result["drain_complete"])
            self.assertEqual(result["drained_readbacks"], 1)
            self.assertEqual(result["physics_gpu_ms"]["count"], 1)
            self.assertEqual(result["frame_ms"]["count"], 0)
            self.assertEqual(result["completed_tps"], 0.5)
            self.assertEqual(result["submitted_tps"], 1)
            records.pop(3)
            path.write_text("\n".join(map(json.dumps, records)))
            with self.assertRaisesRegex(ValueError, "incomplete fixed-duration replay"):
                summary.summarize(path)

    def test_drain_rejects_new_submissions_and_cannot_resume_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            for events in (
                [{"kind": "physics_submit", "phase": "drain", "data": {"tick": 1, "sequence": 1}}],
                [{"kind": "frame", "phase": "drain", "data": {}}, {"kind": "frame", "phase": "measurement", "data": {}}],
            ):
                self.write_capture(path, events)
                with self.assertRaises(ValueError):
                    summary.summarize(path)

    def test_driving_requires_input_for_each_dispatched_tick_including_batched_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            events = []
            for tick in range(359, 363):
                events += [
                    {"kind": "driving_input", "data": {"tick": tick, "script_tick": tick - 359}},
                    {"kind": "physics_submit", "data": {"tick": tick, "sequence": tick}},
                ]
            self.write_capture(path, events)
            self.assertTrue(summary.summarize(path)["driving_input"]["matches_submitted_ticks"])
            # Old per-frame input could cover only the first of several submissions.
            self.write_capture(path, [events[0]] + [event for event in events if event["kind"] == "physics_submit"])
            self.assertFalse(summary.summarize(path)["driving_input"]["matches_submitted_ticks"])
            events[2]["data"]["script_tick"] = 2
            self.write_capture(path, events)
            self.assertFalse(summary.summarize(path)["driving_input"]["matches_submitted_ticks"])

    def test_car_trace_rejects_inverted_nested_gpu_timestamps(self):
        spec = importlib.util.spec_from_file_location("car_summary", Path(__file__).with_name("summarize-suspension-benchmark.py"))
        car_summary = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(car_summary)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text(json.dumps({"type": "tick", "tick": 181, "recovery_ms": 4,
                                        "recovery_projection_ms": 4295}) + "\n" +
                            json.dumps({"type": "summary", "samples": 1}))
            with self.assertRaisesRegex(ValueError, "invalid nested GPU timestamps"):
                car_summary.summarize(path)

    def test_percentiles_use_raw_nearest_rank(self):
        self.assertEqual(summary.percentile(list(range(1, 101)), 95), 95)
        self.assertIsNone(summary.percentile([], 95))

    def test_missing_or_invalid_footer_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            for records in ([], [{"kind": "result", "valid": False}]):
                path.write_text("\n".join(json.dumps(dict(event, phase=event.get("phase", "measurement"))) for event in records))
                with self.assertRaises(ValueError):
                    summary.summarize(path)

    def write_capture(self, path, events):
        records = [{"kind": "metadata", "schema": 3, "metadata": {}, "submission_duration_seconds": 1, "duration_seconds": 1}]
        records += events + [{"kind": "result", "valid": True}]
        path.write_text("\n".join(json.dumps(dict(event, phase=event.get("phase", "measurement"))) for event in records))

    def test_tick_reset_or_missing_completion_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            for ticks, sequences, message in (((4, 4), (1, 2), "backwards"),
                                               ((4, 1), (1, 2), "backwards"),
                                               ((4, 6), (1, 3), "skips")):
                self.write_capture(path, [{"kind": "physics_readback", "data": {"tick": tick, "sequence": seq}}
                                          for tick, seq in zip(ticks, sequences)])
                with self.assertRaisesRegex(ValueError, message):
                    summary.summarize(path)

    def test_delayed_completion_gaps_do_not_use_capture_frame_drop_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            self.write_capture(path, [
                {"kind": "physics_readback", "data": {"tick": 4, "sequence": 1}},
                {"kind": "physics_readback", "data": {"tick": 22, "sequence": 2}},
                {"kind": "physics_submit", "data": {"tick": 23, "sequence": 3}},
                {"kind": "physics_readback", "data": {"tick": 23, "sequence": 3}},
                {"kind": "physics_drop", "data": {"first_tick": 24, "count": 5}},
                {"kind": "physics_submit", "data": {"tick": 29, "sequence": 4}},
            ])
            result = summary.summarize(path)
            self.assertEqual(result["dropped_ticks"], 5)
            self.assertEqual(result["completed_interval_skipped_ticks"], 17)
            self.assertEqual(result["boundary_readbacks"], 2)
            self.assertEqual(result["pending_at_capture_end"], 1)

    def test_scheduler_drops_cannot_hide_missing_readbacks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            self.write_capture(path, [
                {"kind": "physics_drop", "data": {"first_tick": 5, "count": 100}},
                {"kind": "physics_readback", "data": {"tick": 4, "sequence": 1}},
                {"kind": "physics_readback", "data": {"tick": 106, "sequence": 3}},
            ])
            with self.assertRaisesRegex(ValueError, "skips"):
                summary.summarize(path)

    def test_disagreeing_tick_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            self.write_capture(path, [
                {"kind": "physics_submit", "data": {"tick": 4, "sequence": 1}},
                {"kind": "physics_readback", "data": {"tick": 5, "sequence": 1}},
            ])
            with self.assertRaisesRegex(ValueError, "disagree"):
                summary.summarize(path)

    def test_submission_gap_requires_the_matching_drop_range(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            for first in (6, 100):
                self.write_capture(path, [
                    {"kind": "physics_submit", "data": {"tick": 4, "sequence": 1}},
                    {"kind": "physics_drop", "data": {"first_tick": first, "count": 1}},
                    {"kind": "physics_submit", "data": {"tick": 6, "sequence": 2}},
                ])
                with self.assertRaisesRegex(ValueError, "exact scheduler"):
                    summary.summarize(path)

    def test_repeated_async_sample_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            sample = {"kind": "render_gpu", "data": {"sample_id": 7}}
            records = [{"kind": "metadata", "schema": 3, "duration_seconds": 1, "submission_duration_seconds": 1}, sample, sample, {"kind": "result", "valid": True}]
            path.write_text("\n".join(json.dumps(dict(event, phase=event.get("phase", "measurement"))) for event in records))
            with self.assertRaisesRegex(ValueError, "duplicated"):
                summary.summarize(path)

    def test_callback_and_publication_metrics_remain_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            records = [{"kind": "metadata", "schema": 3, "metadata": {}, "submission_duration_seconds": 60, "duration_seconds": 60},
                       {"kind": "physics_readback", "data": {"tick": 1, "sequence": 1, "latency_ms": 100, "submission_to_callbacks_ms": 80, "callbacks_during_poll": False}},
                       {"kind": "physics_publication", "data": {"tick": 1, "callback_to_publication_ms": 20}},
                       {"kind": "physics_poll", "data": {"duration_ms": 0.1, "outcome": "empty"}},
                       {"kind": "result", "valid": True}]
            path.write_text("\n".join(json.dumps(dict(event, phase=event.get("phase", "measurement"))) for event in records))
            result = summary.summarize(path)
            self.assertEqual(result["submission_to_callbacks_ms"]["p50"], 80)
            self.assertEqual(result["callback_to_publication_ms"]["p50"], 20)
            self.assertEqual(result["callbacks_during_poll_counts"], {"true": 0, "false": 1})
            self.assertEqual(result["physics_poll_outcomes"]["empty"], 1)

    def test_gpu_distributions_preserve_missing_and_overlapping_spans(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            records = [{"kind": "metadata", "schema": 3, "metadata": {}, "submission_duration_seconds": 60, "duration_seconds": 60}]
            for sample_id, opaque in enumerate((20, 40)):
                records.append({"kind": "render_gpu", "elapsed_ms": 100,
                                "data": {"sample_id": sample_id, "status": "Complete",
                                         "sample_age_ms": 90, "opaque_ms": opaque,
                                         "tracked_span_ms": 45, "transparent_ms": None}})
            records.append({"kind": "result", "valid": True})
            path.write_text("\n".join(json.dumps(dict(event, phase=event.get("phase", "measurement"))) for event in records))
            result = summary.summarize(path)
            self.assertEqual(result["render_gpu_ms"]["opaque_ms"]["p95"], 40)
            self.assertEqual(result["render_gpu_ms"]["tracked_span_ms"]["p95"], 45)
            self.assertEqual(result["render_gpu_ms"]["transparent_ms"]["count"], 0)
            self.assertIsNone(result["render_gpu_ms"]["transparent_ms"]["p95"])


if __name__ == "__main__":
    unittest.main()
