"""Audit the recorded epoch-zero fallback and its actual public WASM outputs."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_run(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    reports = list(directory.glob('report-*/report.json'))
    if len(reports) != 1:
        raise ValueError('Expected exactly one complete report')
    path = reports[0]
    report = json.loads(path.read_text())
    if (manifest['exit_code'] != 0 or digest(path) != manifest['report_sha256']
            or report['summary']['completed'] != 10
            or any(e['metrics']['filled_frames'] for e in report['episodes'])):
        raise ValueError('Incomplete or modified report: ' + str(path))
    return path.parent, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    expected = json.loads((ROOT / 'configs/rotation-focus-v3.json').read_text())['initial_onnx_sha256']
    for variant in ('control', 'focus'):
        model = ROOT / f'models/rotation-{variant}-v3-retained'
        run = json.loads((model / 'run.json').read_text())
        if run['best_epoch'] != 0 or digest(model / 'net.onnx') != expected:
            raise ValueError('This audit applies only to the recorded epoch-zero fallback')
    checks = []
    for name, old, new in [
        ('baseline_repeat', 'ablation-v2-baseline', 'rotation-v3-baseline'),
        ('current_repeat', 'ablation-v2-current', 'rotation-v3-current'),
        ('control_fallback', 'rotation-v3-current', 'rotation-v3-control'),
        ('focus_fallback', 'rotation-v3-current', 'rotation-v3-focus'),
    ]:
        adir, before = read_run(ROOT / 'results' / old)
        bdir, after = read_run(ROOT / 'results' / new)
        if [e['hash'] for e in before['episodes']] != [e['hash'] for e in after['episodes']]:
            raise ValueError('Episode selection mismatch')
        episodes = []
        for a, b in zip(before['episodes'], after['episodes']):
            if a['metrics'] != b['metrics']:
                raise ValueError('Accuracy changed in ' + name)
            files = {}
            for filename in ('estimate.tum', 'twist.txt'):
                left, right = (p / 'logs' / a['hash'] / filename for p in (adir, bdir))
                if digest(left) != digest(right):
                    raise ValueError('Prediction changed in ' + name)
                files[filename] = digest(right)
            episodes.append({'hash': a['hash'], 'accuracy_identical': True, 'output_sha256': files})
        checks.append({'comparison': name, 'before': old, 'after': new, 'episodes': episodes})
    record = {'scope': 'Selected epoch-zero fallback only; trained epoch20 candidates remain validation diagnostics.',
              'initial_and_selected_onnx_sha256': expected, 'comparisons': checks}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + '\n')
    print('Verified all selected outputs and repeated baseline/current predictions byte-for-byte.')


if __name__ == '__main__':
    main()
