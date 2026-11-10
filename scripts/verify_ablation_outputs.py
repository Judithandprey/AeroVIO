"""Verify unchanged controls and full/compact parity on actual saved WASM output."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report(directory):
    path = next(directory.glob('report-*/report.json'))
    value = json.loads(path.read_text())
    manifest = json.loads((directory / 'manifest.json').read_text())
    if (digest(path) != manifest['report_sha256'] or value['summary']['completed'] != 10
            or any(e['metrics']['filled_frames'] for e in value['episodes'])):
        raise ValueError('Report integrity/completion failed: ' + str(path))
    return path.parent, value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    records = {'unchanged_controls': {}, 'compact_equivalence': []}
    for label, original, current in [
        ('baseline', 'baseline-comparison-v1', 'ablation-v2-baseline'),
        ('image_imu_warp', 'motion-warp-v1', 'ablation-v2-current'),
    ]:
        before_dir, before = report(ROOT / 'results' / original)
        after_dir, after = report(ROOT / 'results' / current)
        checks = []
        if [e['hash'] for e in before['episodes']] != [e['hash'] for e in after['episodes']]:
            raise ValueError('Control episode order mismatch')
        for a, b in zip(before['episodes'], after['episodes']):
            paths = [(before_dir / 'logs' / a['hash'] / f, after_dir / 'logs' / b['hash'] / f)
                     for f in ('estimate.tum', 'twist.txt')]
            same_files = all(digest(x) == digest(y) for x, y in paths)
            if a['metrics'] != b['metrics'] or not same_files:
                raise ValueError('Control predictions changed: ' + label + '/' + a['hash'])
            checks.append({'episode': a['hash'], 'all_metrics_identical': True,
                           'trajectory_sha256': digest(paths[0][1]), 'twist_sha256': digest(paths[1][1])})
        records['unchanged_controls'][label] = checks

    full_dir, full = report(ROOT / 'results/ablation-v2-imu-only')
    compact_dir, compact = report(ROOT / 'results/ablation-v2-imu-compact')
    if [e['hash'] for e in full['episodes']] != [e['hash'] for e in compact['episodes']]:
        raise ValueError('Compact episode order mismatch')
    tolerance = 1e-4
    for a, b in zip(full['episodes'], compact['episodes']):
        left, right = (p / 'logs' / a['hash'] for p in (full_dir, compact_dir))
        p, q = (np.loadtxt(directory / 'estimate.tum') for directory in (left, right))
        u, v = (np.loadtxt(directory / 'twist.txt') for directory in (left, right))
        if not all(np.isfinite(array).all() for array in (p, q, u, v)):
            raise ValueError('Non-finite compact/full outputs')
        if p.shape != q.shape or u.shape != v.shape or not np.array_equal(p[:, 0], q[:, 0]) or not np.array_equal(u[:, 0], v[:, 0]):
            raise ValueError('Compact trajectory shapes or timestamps differ')
        position = float(np.linalg.norm(p[:, 1:4] - q[:, 1:4], axis=1).max())
        pq = p[:, 4:8] / np.linalg.norm(p[:, 4:8], axis=1, keepdims=True)
        qq = q[:, 4:8] / np.linalg.norm(q[:, 4:8], axis=1, keepdims=True)
        angle = float((2 * np.arccos(np.clip(np.abs((pq * qq).sum(1)), 0, 1))).max())
        twist = float(np.abs(u[:, 1:] - v[:, 1:]).max())
        if max(position, angle, twist) > tolerance:
            raise ValueError('Compact public-output equivalence tolerance failed')
        records['compact_equivalence'].append({'episode': a['hash'], 'frames': len(p),
            'max_position_difference_m': position, 'max_rotation_difference_rad': angle,
            'max_twist_component_difference': twist,
            'full_trajectory_sha256': digest(left / 'estimate.tum'),
            'compact_trajectory_sha256': digest(right / 'estimate.tum')})
    records['compact_tolerance'] = {'position_m': tolerance, 'rotation_rad': tolerance, 'twist_component': tolerance}
    records['scope'] = 'Actual saved WASM outputs; TUM serialization has finite precision. Numerical equivalence, not bitwise identity, for the compact model.'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=2) + '\n')
    print('Unchanged baseline/current predictions and compact public-output parity verified.')


if __name__ == '__main__':
    main()
