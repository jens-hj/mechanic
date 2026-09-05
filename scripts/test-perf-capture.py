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
    def test_percentiles_use_raw_nearest_rank(self):
        self.assertEqual(summary.percentile(list(range(1, 101)), 95), 95)
        self.assertIsNone(summary.percentile([], 95))

    def test_missing_or_invalid_footer_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            for records in ([], [{"kind": "result", "valid": False}]):
                path.write_text("\n".join(map(json.dumps, records)))
                with self.assertRaises(ValueError):
                    summary.summarize(path)

    def test_tick_reset_or_missing_completion_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            for ticks in ((4, 4), (4, 1), (4, 6)):
                records = [{"kind": "metadata"}] + [{"kind": "physics_readback", "data": {"tick": tick}} for tick in ticks] + [{"kind": "result", "valid": True}]
                path.write_text("\n".join(map(json.dumps, records)))
                with self.assertRaisesRegex(ValueError, "nonconsecutive"):
                    summary.summarize(path)

    def test_repeated_async_sample_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            sample = {"kind": "render_gpu", "data": {"sample_id": 7}}
            records = [{"kind": "metadata"}, sample, sample, {"kind": "result", "valid": True}]
            path.write_text("\n".join(map(json.dumps, records)))
            with self.assertRaisesRegex(ValueError, "duplicated"):
                summary.summarize(path)

    def test_callback_and_publication_metrics_remain_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            records = [{"kind": "metadata", "metadata": {}, "duration_seconds": 60},
                       {"kind": "physics_readback", "data": {"tick": 1, "latency_ms": 100, "submission_to_callbacks_ms": 80, "callbacks_during_poll": False}},
                       {"kind": "physics_publication", "data": {"tick": 1, "callback_to_publication_ms": 20}},
                       {"kind": "physics_poll", "data": {"duration_ms": 0.1, "outcome": "empty"}},
                       {"kind": "result", "valid": True}]
            path.write_text("\n".join(map(json.dumps, records)))
            result = summary.summarize(path)
            self.assertEqual(result["submission_to_callbacks_ms"]["p50"], 80)
            self.assertEqual(result["callback_to_publication_ms"]["p50"], 20)
            self.assertEqual(result["callbacks_during_poll_counts"], {"true": 0, "false": 1})
            self.assertEqual(result["physics_poll_outcomes"]["empty"], 1)

    def test_gpu_distributions_preserve_missing_and_overlapping_spans(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.jsonl"
            records = [{"kind": "metadata", "metadata": {}, "duration_seconds": 60}]
            for sample_id, opaque in enumerate((20, 40)):
                records.append({"kind": "render_gpu", "elapsed_ms": 100,
                                "data": {"sample_id": sample_id, "status": "Complete",
                                         "sample_age_ms": 90, "opaque_ms": opaque,
                                         "tracked_span_ms": 45, "transparent_ms": None}})
            records.append({"kind": "result", "valid": True})
            path.write_text("\n".join(map(json.dumps, records)))
            result = summary.summarize(path)
            self.assertEqual(result["render_gpu_ms"]["opaque_ms"]["p95"], 40)
            self.assertEqual(result["render_gpu_ms"]["tracked_span_ms"]["p95"], 45)
            self.assertEqual(result["render_gpu_ms"]["transparent_ms"]["count"], 0)
            self.assertIsNone(result["render_gpu_ms"]["transparent_ms"]["p95"])


if __name__ == "__main__":
    unittest.main()
