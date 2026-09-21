#!/usr/bin/env python3
"""Capture a disposable copy of a saved world without OS keyboard/mouse automation.

Background timings are diagnostic. --foreground requests and verifies focus
and native dimensions; it does not establish physical or performance acceptance.
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


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def file_hashes(root):
    return {str(path.relative_to(root)): digest(path)
            for path in sorted(root.rglob("*")) if path.is_file()}


def foreground_rejections(result):
    errors = []
    if not result["metadata"].get("foreground_requested") or result["metadata"].get("automated_background"):
        errors.append("application did not acknowledge foreground capture")
    if result["focused_frame_counts"]["true"] != result["frame_ms"]["count"] or not result["frame_ms"]["count"]:
        errors.append("every measured frame must be focused")
    expected = {"window_pixels": ["[4112, 2524]"], "target_pixels": ["[4112, 2524]"],
                "viewport_pixels": ["[4112, 2524]"], "present_mode": ['"AutoNoVsync"'],
                "msaa": ["4"], "f3": ["true"]}
    for key, value in expected.items():
        if result["settings_seen"][key] != value:
            errors.append(f"unexpected or changing {key}: {result['settings_seen'][key]}")
    if result["metadata"].get("capture_from_start"):
        errors.append("capture includes streaming warm-up")
    return errors


def freeze_rejections(records):
    states = [record["data"] for record in records if record.get("kind") == "freeze_state"]
    heights = [state["height"] for state in states if state["held"]]
    errors = []
    if not any(state["held"] and state["aligned"] for state in states):
        errors.append("scripted freeze never reached its aligned hold")
    if not any(b > a for a, b in zip(heights, heights[1:])):
        errors.append("scripted raise never changed the accepted target")
    if not any(b < a for a, b in zip(heights, heights[1:])):
        errors.append("scripted lower never changed the accepted target")
    if not states or states[-1]["held"]:
        errors.append("scripted release did not clear the hold")
    return errors


def run(binary, world, output, assets, drive=False, demonstration=False, place=None, straight=False,
        foreground=False, identity=None, replay_ticks=None, physics="cpu", freeze=False, hammer=None,
        from_start=False):
    if physics not in ("gpu", "cpu"):
        raise ValueError("physics must be gpu or cpu")
    binary, world, output, assets = [p.resolve() for p in (binary, world, output, assets)]
    if not binary.is_file() or not (world / "world.ron").is_file():
        raise ValueError("binary and world/world.ron must exist")
    if not (assets / "assets").is_dir():
        raise ValueError("asset root must contain the application's assets directory")
    if foreground and (demonstration or place is not None):
        raise ValueError("foreground comparison forbids demonstration and placement diagnostics")
    if foreground and from_start:
        raise ValueError("foreground comparison requires settled streaming")
    if foreground and identity is None:
        raise ValueError("foreground comparison requires --identity from a source-matched build")
    build_identity = json.loads(identity.read_text()) if identity else None
    if build_identity and (not build_identity.get("build_complete") or not build_identity.get("source_files_sha256")):
        raise ValueError("build identity must identify completed frozen-source build")
    if build_identity and build_identity["binary_sha256"] != digest(binary):
        raise ValueError("binary does not match build identity")
    asset_hashes = file_hashes(assets / "assets")
    if build_identity:
        prefix = "crates/mechanic-app/assets/"
        expected_assets = {name[len(prefix):]: value for name, value in build_identity["source_files_sha256"].items()
                           if name.startswith(prefix)}
        if asset_hashes != expected_assets:
            raise ValueError("assets do not match frozen build sources")
    if replay_ticks is not None and (not foreground or not 1 <= replay_ticks <= 3600):
        raise ValueError("replay requires foreground mode and 1..3600 ticks")
    source_hashes = file_hashes(world)
    output.mkdir(parents=True, exist_ok=False)
    # The application never opens a store containing the source world. Even its
    # startup/default-world selection can only see this disposable copy.
    store = output / "world-store"
    store.mkdir()
    copy = store / ("mechanic-auto-" + uuid.uuid4().hex)
    copy.mkdir()
    try:
        shutil.copytree(world, copy, dirs_exist_ok=True)
        if file_hashes(copy) != source_hashes:
            raise ValueError("world changed while copying the fixture")
        manifest = copy / "world.ron"
        source = manifest.read_text()
        changed, count = re.subn(r'(?m)^(\s*name:\s*)"(?:[^"\\]|\\.)*"',
                                lambda m: m[1] + json.dumps(copy.name), source, count=1)
        if count != 1:
            raise ValueError("world manifest has no name")
        manifest.write_text(changed)
        run_record = {
            "source_world": str(world), "temporary_world": str(copy),
            "manifest_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "scripted_freeze": freeze, "scripted_hammer": hammer, "physics_route": physics, "binary": str(binary), "binary_sha256": digest(binary), "assets": str(assets),
            "source_world_files_sha256": source_hashes,
            "assets_files_sha256": asset_hashes,
            "launcher_sha256": digest(Path(__file__)), "build_identity": build_identity,
            "automated_background": not foreground, "foreground_requested": foreground,
            "scripted_driving": drive, "straight_driving": straight, "replay_ticks": replay_ticks,
            "demonstration_frames": demonstration, "scripted_placement_seconds": place,
            "controlled_foreground_benchmark": False,
        }
        (output / "run.json").write_text(json.dumps(run_record, indent=2))
        env = os.environ.copy()
        # Do not inherit diagnostic switches that silently change the workload.
        for key in list(env):
            if key.startswith("MECHANIC_"):
                del env[key]
        env.update(MECHANIC_PHYSICS=physics, MECHANIC_AUTO_WORLD=copy.name, MECHANIC_PERF_CAPTURE_DIR=str(output),
                   MECHANIC_AUTO_WORLD_STORE=str(store),
                   MECHANIC_AUTO_FOREGROUND="1" if foreground else "0",
                   MECHANIC_PERF_LABEL=world.name + ("-foreground" if foreground else "-background"), BEVY_ASSET_ROOT=str(assets),
                   MECHANIC_RENDER_EXPERIMENT="baseline", MECHANIC_AUTO_DRIVE="1" if drive else "0",
                   MECHANIC_AUTO_DRIVE_STRAIGHT="1" if straight else "0",
                   MECHANIC_AUTO_DRIVING_FRAMES="1" if demonstration else "0",
                   MECHANIC_AUTO_PLACE=str(place) if place else "",
                   MECHANIC_AUTO_FREEZE="1" if freeze else "0")
        if from_start:
            # A world whose terrain keeps changing never settles its streaming.
            env["MECHANIC_PERF_CAPTURE_FROM_START"] = "1"
        if hammer is not None:
            env["MECHANIC_AUTO_HAMMER"] = json.dumps(hammer)
        if replay_ticks is not None:
            env["MECHANIC_AUTO_REPLAY_TICKS"] = str(replay_ticks)
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
        if freeze:
            errors = freeze_rejections(records)
            (output / "freeze-protocol.json").write_text(json.dumps({
                "passed": not errors, "rejections": errors,
            }, indent=2))
            if errors:
                raise ValueError("; ".join(errors))
        # Scripted placements republish the scene, so the acceptance summary,
        # which requires one uninterrupted tick sequence, does not apply.
        if place is None:
            summary = subprocess.run([sys.executable, str(Path(__file__).with_name("summarize-perf-capture.py")),
                                      str(captures[0])], check=True, capture_output=True, text=True)
            (output / "summary.json").write_text(summary.stdout)
            result = json.loads(summary.stdout)[0]
            if result["metadata"].get("replay_ticks") != replay_ticks:
                raise ValueError("application replay length differs from requested workload")
            if result["physics_routes"] != [physics]:
                raise ValueError("actual physics route differs from requested route")
            if result["cpu_degraded_ticks"]:
                raise ValueError("capture contains degraded CPU ticks")
            if not result["drain_complete"]:
                raise ValueError("capture did not publish every submitted state")
            if drive and not result["driving_input"]["matches_submitted_ticks"]:
                raise ValueError("scripted input was not applied at every submitted tick")
            if foreground:
                errors = foreground_rejections(result)
                (output / "foreground-protocol.json").write_text(json.dumps({
                    "passed": not errors, "rejections": errors,
                    "physics_acceptance_passed": False,
                    "reason": "complete kernel coverage and physical acceptance remain separate gates",
                }, indent=2))
                if errors:
                    raise ValueError("; ".join(errors))
            if result["physics_error_flags"] != [0]:
                raise ValueError("capture has missing physics readbacks or nonzero failure flags")
        print(output)
    except Exception as error:
        (output / "failure.json").write_text(json.dumps({
            "type": type(error).__name__, "message": str(error), "acceptance_passed": False,
        }, indent=2))
        raise
    finally:
        shutil.rmtree(copy)
        if file_hashes(world) != source_hashes:
            raise ValueError("source world changed during capture")
        if file_hashes(assets / "assets") != asset_hashes:
            raise ValueError("assets changed during capture")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/release/mechanic-app"))
    parser.add_argument("--world", type=Path, required=True, help="Saved world directory")
    parser.add_argument("--output", type=Path, required=True, help="New results directory")
    parser.add_argument("--assets", type=Path, default=Path("crates/mechanic-app"))
    parser.add_argument("--physics", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--drive", action="store_true", help="Drive the sole input-linked seat during capture")
    parser.add_argument("--straight", action="store_true", help="Hold full throttle without scripted steering")
    parser.add_argument("--demonstration", action="store_true", help="Save driving frames every five seconds; perturbs frame timing")
    parser.add_argument("--place", type=float, help="Commit one scripted block placement every N seconds; diagnostic only")
    parser.add_argument("--foreground", action="store_true", help="Request focus and verify every measured frame at native resolution")
    parser.add_argument("--identity", type=Path, help="Source-matched build identity JSON")
    parser.add_argument("--replay-ticks", type=int, help="Replay exactly N contiguous 60 Hz ticks, retaining backlog, then drain publication")
    parser.add_argument("--hammer", type=json.loads, help="Capture hammer JSON: body_index, local_point, impulse; optional repeat and body_local_impulse")
    parser.add_argument("--freeze", action="store_true", help="Freeze, raise, lower and release the linked creation")
    parser.add_argument("--from-start", action="store_true", help="Record from world entry instead of waiting for settled streaming; includes warm-up")
    args = parser.parse_args()
    run(args.binary, args.world, args.output, args.assets,
        args.drive or args.straight, args.demonstration, args.place, args.straight,
        args.foreground, args.identity, args.replay_ticks, args.physics, args.freeze, args.hammer,
        args.from_start)
