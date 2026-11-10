"""Evidence guards reject modified or incomparable saved evaluations."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('compare_motion_runs', ROOT / 'scripts/compare_motion_runs.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class ComparisonGuards(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.names = ['baseline', 'motion-cnn', 'cnn-warp', 'cnn-warp-map']
        for name in self.names:
            target = self.directory / name
            target.mkdir()
            for filename in ['manifest.json', 'report.json']:
                shutil.copyfile(ROOT / 'experiments/learned-motion-v1/runs' / name / filename, target / filename)

    def run_comparison(self):
        argv = ['compare_motion_runs']
        for name in self.names:
            argv += ['--run', name, str(self.directory / name)]
        argv += ['--out', str(self.directory / 'comparison.md')]
        with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
            comparison.main()

    def mutate_manifest(self, name, update):
        path = self.directory / name / 'manifest.json'
        value = json.loads(path.read_text())
        update(value)
        path.write_text(json.dumps(value))

    def test_archived_comparison_is_complete(self):
        self.run_comparison()
        result = json.loads((self.directory / 'comparison.json').read_text())
        self.assertEqual([row['completed'] for row in result], [10] * 4)

    def test_report_tampering_is_rejected(self):
        path = self.directory / 'motion-cnn/report.json'
        path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'report hash mismatch'):
            self.run_comparison()

    def test_runtime_mismatch_is_rejected(self):
        self.mutate_manifest('motion-cnn', lambda m: m.update(runtime_sha256='different'))
        with self.assertRaisesRegex(ValueError, 'incomparable runtime'):
            self.run_comparison()

    def test_incomplete_run_is_rejected(self):
        self.mutate_manifest('motion-cnn', lambda m: m.update(finished_utc=None))
        with self.assertRaisesRegex(ValueError, 'not finished'):
            self.run_comparison()

    def test_changed_map_settings_are_rejected(self):
        self.mutate_manifest('cnn-warp-map', lambda m: m['embedded_model_metadata'].update(diagnostics_interval=20))
        with self.assertRaisesRegex(ValueError, 'differ only'):
            self.run_comparison()

    def test_actual_limits_mismatch_is_rejected(self):
        path = self.directory / 'motion-cnn/report.json'
        report = json.loads(path.read_text())
        report['limits']['step_fuel'] += 1
        path.write_text(json.dumps(report))
        self.mutate_manifest('motion-cnn', lambda m: m.update(report_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        with self.assertRaisesRegex(ValueError, 'actual limits differ'):
            self.run_comparison()


if __name__ == '__main__':
    unittest.main()
