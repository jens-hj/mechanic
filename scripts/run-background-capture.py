#!/usr/bin/env python3
"""Capture a disposable copy of a saved world without OS keyboard/mouse automation.

Background timings are diagnostic, not controlled foreground benchmark results.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid


def run(binary, world, output, assets):
    binary, world, output, assets = [p.resolve() for p in (binary, world, output, assets)]
    if not binary.is_file() or not (world / "world.ron").is_file():
        raise ValueError("binary and world/world.ron must exist")
    if not (assets / "assets").is_dir():
        raise ValueError("asset root must contain the application's assets directory")
    output.mkdir(parents=True, exist_ok=False)
    # Same store, distinct generated identity: autosaves cannot reach the source world.
    copy = world.parent / ("mechanic-auto-" + uuid.uuid4().hex)
    copy.mkdir()
    try:
        shutil.copytree(world, copy, dirs_exist_ok=True)
        manifest = copy / "world.ron"
        source = manifest.read_text()
        changed, count = re.subn(r'(?m)^(\s*name:\s*)"(?:[^"\\]|\\.)*"',
                                lambda m: m[1] + json.dumps(copy.name), source, count=1)
        if count != 1:
            raise ValueError("world manifest has no name")
        manifest.write_text(changed)
        (output / "run.json").write_text(json.dumps({
            "source_world": str(world), "temporary_world": str(copy),
            "manifest_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "binary": str(binary), "assets": str(assets),
            "automated_background": True, "controlled_foreground_benchmark": False,
        }, indent=2))
        env = os.environ.copy()
        env.update(MECHANIC_AUTO_WORLD=copy.name, MECHANIC_PERF_CAPTURE_DIR=str(output),
                   MECHANIC_PERF_LABEL=world.name + "-background", BEVY_ASSET_ROOT=str(assets),
                   MECHANIC_RENDER_EXPERIMENT="baseline")
        with (output / "app.log").open("w") as log:
            # subprocess.run kills and reaps on timeout; cleanup runs only afterward.
            subprocess.run([str(binary)], env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=360, check=True)
        captures = list(output.glob("capture-*.jsonl"))
        if len(captures) != 1:
            raise ValueError("expected exactly one completed capture")
        records = [json.loads(line) for line in captures[0].read_text().splitlines()]
        if not records[-1].get("valid") or not captures[0].with_suffix(".png").is_file():
            raise ValueError("missing valid capture or screenshot")
        summary = subprocess.run([sys.executable, str(Path(__file__).with_name("summarize-perf-capture.py")),
                                  str(captures[0])], check=True, capture_output=True, text=True)
        (output / "summary.json").write_text(summary.stdout)
        result = json.loads(summary.stdout)[0]
        if result["physics_error_flags"] != [0]:
            raise ValueError("capture has missing physics readbacks or nonzero failure flags")
        print(output)
    finally:
        shutil.rmtree(copy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/release/mechanic-app"))
    parser.add_argument("--world", type=Path, required=True, help="Saved world directory")
    parser.add_argument("--output", type=Path, required=True, help="New results directory")
    parser.add_argument("--assets", type=Path, default=Path("crates/mechanic-app"))
    args = parser.parse_args()
    run(args.binary, args.world, args.output, args.assets)
