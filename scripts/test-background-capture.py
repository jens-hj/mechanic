#!/usr/bin/env python3
"""Verify the launcher isolates source saves and cleans up after failure."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("runner", Path(__file__).with_name("run-background-capture.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class WorldIsolation(unittest.TestCase):
    def test_freeze_capture_requires_accepted_movement_and_release(self):
        def state(height, aligned=True):
            return {"kind": "freeze_state", "data": {
                "held": height is not None, "aligned": aligned, "height": height,
            }}
        records = [state(None, False), state(2.0, False), state(2.0), state(2.25), state(2.0), state(None, False)]
        self.assertEqual(runner.freeze_rejections(records), [])
        self.assertIn("scripted release did not clear the hold", runner.freeze_rejections(records[:-1]))
        self.assertTrue(runner.freeze_rejections([]))
        self.assertIn("scripted raise never changed the accepted target",
                      runner.freeze_rejections([state(2.0), state(None, False)]))

    def test_failed_run_preserves_source_and_removes_only_its_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            world = root / "test4"
            world.mkdir()
            source = '(\n    name: "TEST4",\n    version: 4,\n)\n'
            (world / "world.ron").write_text(source)
            (world / "blocks").write_bytes(b"original blocks")
            binary = root / "mechanic-app"
            binary.touch()
            assets = root / "app"
            (assets / "assets").mkdir(parents=True)
            def fail(command, **kwargs):
                env = kwargs["env"]
                self.assertEqual(env["MECHANIC_AUTO_WORLD_STORE"], str(root.resolve() / "output/world-store"))
                self.assertNotIn("MECHANIC_PERF_CAPTURE_FROM_START", env)
                self.assertEqual(env["MECHANIC_PHYSICS"], "cpu")
                copy = Path(env["MECHANIC_AUTO_WORLD_STORE"]) / env["MECHANIC_AUTO_WORLD"]
                self.assertNotEqual(copy, world)
                self.assertIn(copy.name, (copy / "world.ron").read_text())
                (copy / "blocks").write_bytes(b"autosaved changes")
                raise subprocess.TimeoutExpired(command, 360)
            with patch.object(runner.subprocess, "run", side_effect=fail), patch.dict(runner.os.environ, {"MECHANIC_PERF_CAPTURE_FROM_START": "1"}):
                with self.assertRaises(subprocess.TimeoutExpired):
                    runner.run(binary, world, root / "output", assets, physics="cpu")
            self.assertEqual((world / "world.ron").read_text(), source)
            self.assertEqual((world / "blocks").read_bytes(), b"original blocks")
            self.assertEqual(list(root.glob("mechanic-auto-*")), [])
            self.assertEqual(list((root / "output/world-store").iterdir()), [])
            self.assertTrue((root / "output" / "app.log").exists())
            record = json.loads((root / "output" / "run.json").read_text())
            self.assertEqual(record["source_world_files_sha256"], runner.file_hashes(world))
            self.assertEqual(record["binary_sha256"], runner.digest(binary))

    def test_foreground_rejects_lost_focus_and_resolution_changes(self):
        result = {
            "metadata": {"foreground_requested": True, "automated_background": False},
            "frame_ms": {"count": 10}, "focused_frame_counts": {"true": 10, "false": 0},
            "settings_seen": {"window_pixels": ["[4112, 2524]"], "target_pixels": ["[4112, 2524]"],
                              "viewport_pixels": ["[4112, 2524]"], "present_mode": ['"AutoNoVsync"'],
                              "msaa": ["4"], "f3": ["true"]},
        }
        self.assertEqual(runner.foreground_rejections(result), [])
        result["focused_frame_counts"]["true"] = 9
        self.assertIn("every measured frame must be focused", runner.foreground_rejections(result))
        result["focused_frame_counts"]["true"] = 10
        result["settings_seen"]["target_pixels"].append("[1920, 1080]")
        self.assertIn("target_pixels", runner.foreground_rejections(result)[0])

    def test_foreground_rejects_background_metadata_even_if_window_is_focused(self):
        result = {
            "metadata": {"foreground_requested": False, "automated_background": True},
            "frame_ms": {"count": 0}, "focused_frame_counts": {"true": 0, "false": 0},
            "settings_seen": {key: [] for key in ("window_pixels", "target_pixels", "viewport_pixels", "present_mode", "msaa", "f3")},
        }
        errors = runner.foreground_rejections(result)
        self.assertIn("application did not acknowledge foreground capture", errors)
        self.assertIn("every measured frame must be focused", errors)


if __name__ == "__main__":
    unittest.main()
