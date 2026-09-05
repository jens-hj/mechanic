#!/usr/bin/env python3
"""Freeze the dirty source and build matched release binaries; baseline uses pristine 29.0.4 internals."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("destination", type=Path, help="new directory outside the repository")
parser.add_argument("--build", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
destination = args.destination.resolve()
if destination.is_relative_to(root):
    parser.error("destination must be outside the repository")
destination.mkdir(parents=True, exist_ok=False)
files = set(subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=root).decode().split("\0")) - {""}
cargo_home = Path(os.environ.get("CARGO_HOME", Path.home() / ".cargo"))
for variant in ("baseline", "candidate"):
    source = destination / variant
    for name in sorted(files):
        original = root / name
        if original.is_file():
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
    if variant == "baseline":
        for package in ("wgpu-core", "wgpu-hal"):
            candidates = list((cargo_home / "registry/src").glob(f"*/{package}-29.0.4"))
            if len(candidates) != 1:
                raise RuntimeError(f"Expected one cached pristine {package} 29.0.4 source, found {candidates}")
            shutil.rmtree(source / "vendor" / package)
            shutil.copytree(candidates[0], source / "vendor" / package)
    fingerprints = {str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.rglob("*")) if p.is_file()}
    (destination / f"{variant}-sources.json").write_text(json.dumps(fingerprints, indent=2)+"\n")
if args.build:
    for variant in ("baseline", "candidate"):
        source = destination / variant
        with (destination / f"{variant}-build.log").open("w") as log:
            subprocess.run(["cargo", "build", "--release", "--locked", "--offline", "-p", "mechanic-app", "--target-dir", str(destination / "target")], cwd=source, stdout=log, stderr=subprocess.STDOUT, check=True)
        binary = destination / "target/release/mechanic-app"
        shutil.copy2(binary, destination / f"mechanic-app-{variant}")
        (destination / f"{variant}-binary.sha256").write_text(hashlib.sha256(binary.read_bytes()).hexdigest()+"\n")
print(destination)
