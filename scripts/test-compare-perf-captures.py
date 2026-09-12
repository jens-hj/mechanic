#!/usr/bin/env python3
"""Paired captures must represent the same complete physical workload."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("comparison", Path(__file__).with_name("compare-perf-captures.py"))
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class PairedCaptureTests(unittest.TestCase):
    def fixture(self, directory, name, ticks=3, mismatch=False, pending=False):
        folder = directory / name
        folder.mkdir()
        path = folder / "capture.jsonl"
        records = [{"kind": "metadata", "schema": 3, "submission_duration_seconds": 1, "duration_seconds": 1,
                    "metadata": {"initial_state_hash": "initial", "initial_camera_matrix": [1, 0],
                                 "initial_terrain_fingerprint": "terrain", "initial_terrain_layout_fingerprint": "layout", "adapter": "test", "foreground_requested": True}}]
        records.append({"kind": "frame", "data": {
            "frame_ms": 10, "focused": True, "f3": True, "present_mode": "AutoNoVsync",
            "window_pixels": [4112, 2524], "target_pixels": [4112, 2524],
            "viewport_pixels": [4112, 2524], "msaa": 4, "backlog": 0, "terrain_backlog": 0,
        }})
        for tick in range(ticks):
            records.extend([
                {"kind": "physics_drive_rows", "data": {"tick": tick + 1, "hash": "drives"}},
                {"kind": "driving_input", "data": {"tick": tick + 1, "script_tick": tick, "held": ["W"]}},
                {"kind": "physics_submit", "data": {"tick": tick + 1, "sequence": tick + 1}},
            ])
            if not pending or tick != ticks - 1:
                records.extend([
                    {"kind": "physics_readback", "data": {"tick": tick + 1, "sequence": tick + 1, "error_flags": 0}},
                    {"kind": "physics_publication", "data": {"tick": tick + 1, "state_hash": f"state-{tick}-{mismatch and tick == 1}"}},
                ])
        records.append({"kind": "result", "valid": True})
        for event in records[1:-1]:
            event.setdefault("phase", "measurement")
        path.write_text("\n".join(map(json.dumps, records)))
        (folder / "run.json").write_text(json.dumps({"binary_sha256": "binary", "source_world_files_sha256": {"world": "hash"},
                                                   "assets_files_sha256": {"texture": "hash"}}))
        return path

    def test_identical_complete_runs_match_without_claiming_physics_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = comparison.compare(self.fixture(root, "a"), self.fixture(root, "b"), repeat=True)
            self.assertTrue(result["comparable"])
            self.assertFalse(result["physics_acceptance_passed"])

    def test_shared_prefix_cannot_hide_different_simulated_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = comparison.compare(self.fixture(root, "a"), self.fixture(root, "b", ticks=2), repeat=True)
            self.assertFalse(result["comparable"])
            self.assertEqual(result["shared_script_ticks"], 2)

    def test_state_divergence_and_unpublished_tail_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = self.fixture(root, "a")
            result = comparison.compare(a, self.fixture(root, "b", mismatch=True), repeat=True)
            self.assertFalse(result["comparable"])
            self.assertEqual(result["first_state_hash_mismatch_script_tick"], 1)
            result = comparison.compare(a, self.fixture(root, "c", pending=True), repeat=True)
            self.assertFalse(result["comparable"])
            self.assertEqual(result["missing_published_states"], 1)


if __name__ == "__main__":
    unittest.main()
