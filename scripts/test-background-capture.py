#!/usr/bin/env python3
"""Verify the launcher isolates source saves and cleans up after failure."""
import importlib.util
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
                copy = root / env["MECHANIC_AUTO_WORLD"]
                self.assertNotEqual(copy, world)
                self.assertIn(copy.name, (copy / "world.ron").read_text())
                (copy / "blocks").write_bytes(b"autosaved changes")
                raise subprocess.TimeoutExpired(command, 360)
            with patch.object(runner.subprocess, "run", side_effect=fail):
                with self.assertRaises(subprocess.TimeoutExpired):
                    runner.run(binary, world, root / "output", assets)
            self.assertEqual((world / "world.ron").read_text(), source)
            self.assertEqual((world / "blocks").read_bytes(), b"original blocks")
            self.assertEqual(list(root.glob("mechanic-auto-*")), [])
            self.assertTrue((root / "output" / "app.log").exists())


if __name__ == "__main__":
    unittest.main()
