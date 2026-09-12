#!/usr/bin/env python3
"""Reference identity must come from an isolated build of the copied sources."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("reference", Path(__file__).with_name("build-performance-reference.py"))
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


class ReferenceBuildTests(unittest.TestCase):
    def test_existing_target_binary_cannot_be_misattributed_to_frozen_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "worktree"
            root.mkdir()
            (root / "Cargo.toml").write_text("frozen manifest")
            shared = root / "target/release/mechanic-app"
            shared.parent.mkdir(parents=True)
            shared.write_bytes(b"stale executable")
            output = Path(directory) / "reference"

            def output_for(command, **kwargs):
                return "rustc fixture" if command[0] == "rustc" else b"Cargo.toml\0"

            def compile_frozen(command, **kwargs):
                source = Path(kwargs["cwd"])
                self.assertEqual((source / "Cargo.toml").read_text(), "frozen manifest")
                (root / "Cargo.toml").write_text("new worktree edits")
                target = Path(command[command.index("--target-dir") + 1])
                self.assertFalse(target.exists())
                self.assertNotEqual(target, root / "target")
                (target / "release").mkdir(parents=True)
                (target / "release/mechanic-app").write_bytes(b"frozen executable")

            with patch.object(reference.subprocess, "check_output", side_effect=output_for), patch.object(reference.subprocess, "run", side_effect=compile_frozen):
                reference.build(root, output)
            self.assertEqual((output / "mechanic-app").read_bytes(), b"frozen executable")
            identity = json.loads((output / "identity.json").read_text())
            self.assertTrue(identity["build_complete"])
            self.assertEqual(identity["binary_sha256"], reference.digest(output / "mechanic-app"))
            self.assertEqual(identity["source_files_sha256"]["Cargo.toml"], reference.digest(output / "source/Cargo.toml"))


if __name__ == "__main__":
    unittest.main()
