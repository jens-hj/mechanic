#!/usr/bin/env python3
"""Freeze the worktree, build the frozen sources, and identify the exact executable.

The output must be outside target and ignored by Git. Builds are offline/locked
and use a new target directory to exclude artifacts from other dirty worktrees.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def build(root, output):
    root, output = root.resolve(), output.resolve()
    if output == root / "target" or root / "target" in output.parents:
        raise ValueError("reference must survive cargo clean")
    if root in output.parents:
        ignored = subprocess.run(["git", "check-ignore", str(output)], cwd=root,
                                 capture_output=True, check=False)
        if ignored.returncode != 0:
            raise ValueError("in-repository reference output must be ignored")
    files = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root
    ).decode().split("\0")
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    hashes = {}
    for name in sorted(set(files) - {""}):
        original = root / name
        if not original.is_file():
            continue
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        hashes[name] = digest(destination)
        if hashes[name] != digest(original):
            raise ValueError(f"source changed while freezing: {name}")
    build_target = output / "build"
    command = ["cargo", "build", "--release", "--offline", "--locked", "-p", "mechanic-app",
               "--manifest-path", str(source / "Cargo.toml"), "--target-dir", str(build_target)]
    identity = {
        "source_files_sha256": hashes,
        "source_tree_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "command": command,
        "rustc": subprocess.check_output(["rustc", "-vV"], cwd=source, text=True),
        "build_environment": {key: value for key, value in os.environ.items()
                              if key in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTUP_TOOLCHAIN")
                              or key.startswith("CARGO_PROFILE_")},
        "build_complete": False,
    }
    identity_path = output / "identity.json"
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    with (output / "build.log").open("w") as log:
        subprocess.run(command, cwd=source, stdout=log, stderr=subprocess.STDOUT, check=True)
    for name, expected in hashes.items():
        if digest(source / name) != expected:
            raise ValueError(f"frozen source changed during build: {name}")
    binary = output / "mechanic-app"
    shutil.copy2(build_target / "release/mechanic-app", binary)
    identity.update(binary_sha256=digest(binary), build_complete=True)
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    print(identity_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(Path(__file__).resolve().parent.parent, args.output)
