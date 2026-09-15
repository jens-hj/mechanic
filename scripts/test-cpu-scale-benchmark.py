#!/usr/bin/env python3
"""Evidence reuse must never silently compare different binaries or protocols."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('scale', Path(__file__).with_name('run-cpu-scale-benchmark.py'))
scale = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scale)


class EvidenceReuse(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binary = self.root / 'binary'
        self.binary.write_bytes(b'unchanged binary')
        self.previous = self.root / 'previous'
        self.previous.mkdir()
        self.manifest = {'binaries': {'baseline': {'sha256': scale.digest(self.binary)}},
                         'warmup_ticks': 0, 'measured_ticks': 1, 'runs': []}
        for connected in (False, True):
            label = f'1x-{"connected" if connected else "separated"}-baseline-1'
            (self.previous / f'{label}.jsonl').write_text(self.records(connected))
            self.manifest['runs'].append({'label': label, 'exit_code': 0})
        self.write_manifest()
        self.args = argparse.Namespace(output=self.root / 'output', baseline=self.binary,
                                       candidate=self.binary, copies=[1], repeats=1,
                                       warmup=0, ticks=1, floor=False, reuse_baseline=self.previous,
                                       assemblies=["separated", "connected"], no_progress_seconds=300)

    def write_manifest(self):
        (self.previous / 'manifest.json').write_text(json.dumps(self.manifest))

    @staticmethod
    def records(connected):
        return '\n'.join(map(json.dumps, [
            {'kind': 'metadata', 'copies': 1, 'connected': connected,
             'terrain': 'saved-seed-leaf-mesh', 'hold': False, 'route': 'cpu'},
            {'kind': 'tick', 'tick': 1, 'state_hash': 'same'},
            {'kind': 'summary', 'maximum_penetration_m': 0, 'closure_gap_m': 0,
             'closure_angle_rad': 0, 'timing_gate': False}])) + '\n'

    def test_identical_baseline_is_copied_with_provenance(self):
        def candidate(command, stdout, _timeout):
            stdout.write(self.records('--connected' in command))
            return argparse.Namespace(returncode=0), False
        with patch.object(scale, 'run_until_idle', side_effect=candidate) as run:
            scale.run(self.args)
        self.assertEqual(run.call_count, 2)
        result = json.loads((self.args.output / 'manifest.json').read_text())
        self.assertEqual(sum('copied_from' in r for r in result['runs']), 2)

    def test_changed_replay_hash_fails_gate_even_when_quality_matches(self):
        def candidate(command, stdout, _timeout):
            records = self.records('--connected' in command).replace('"same"', '"different"').replace('"timing_gate": false', '"timing_gate": true')
            stdout.write(records)
            return argparse.Namespace(returncode=0), False
        with patch.object(scale, 'run_until_idle', side_effect=candidate):
            scale.run(self.args)
        result = json.loads((self.args.output / 'summary.json').read_text())
        for comparison in result['comparisons'].values():
            self.assertFalse(comparison['matching_replay_state_hashes'])
            self.assertFalse(comparison['headless_gate_passed'])

    def test_silent_child_is_stopped_without_waiting_for_a_long_run_timeout(self):
        with tempfile.TemporaryFile('w') as output:
            result, stalled = scale.run_until_idle([sys.executable, '-c', 'import time; time.sleep(30)'], output, 0.05)
        self.assertTrue(stalled)
        self.assertNotEqual(result.returncode, 0)

    def test_changed_binary_is_rejected(self):
        self.binary.write_bytes(b'different binary')
        with self.assertRaisesRegex(ValueError, 'binary differs'):
            scale.run(self.args)

    def test_changed_tick_protocol_is_rejected(self):
        self.args.warmup = 1
        with self.assertRaisesRegex(ValueError, 'tick protocol differs'):
            scale.run(self.args)

    def test_incomplete_baseline_is_rejected(self):
        (self.previous / '1x-separated-baseline-1.jsonl').write_text(self.records(False).splitlines()[0]+'\n')
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            scale.run(self.args)


if __name__ == '__main__':
    unittest.main()
