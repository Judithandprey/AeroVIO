"""Verify completed v4 outputs and report paired calibration effects, without tuning."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from viobench.tools.geometry import poses_to_matrices

from analyze_rotation_ablation import load_run

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def same_outputs(before, after, require_fuel=False):
    checks = []
    for a, b in zip(before['report']['episodes'], after['report']['episodes']):
        if a['hash'] != b['hash'] or a['metrics'] != b['metrics']:
            raise ValueError('Prediction accuracy changed during repeat')
        files = {}
        for name in ('estimate.tum', 'twist.txt'):
            p, q = [run['report_path'].parent / 'logs' / a['hash'] / name for run in (before, after)]
            if sha(p) != sha(q):
                raise ValueError('Repeated output differs: ' + name)
            files[name] = sha(q)
        if require_fuel:
            p, q = [run['report_path'].parent / 'logs' / a['hash'] / 'fuel.npy' for run in (before, after)]
            if not np.array_equal(np.load(p), np.load(q)):
                raise ValueError('Repeated per-frame fuel differs')
            files['fuel.npy'] = sha(q)
        checks.append({'episode': a['hash'], 'accuracy_identical': True, 'sha256': files})
    return checks


def local_translations(path):
    rows = np.loadtxt(path)
    if rows.shape != (1200, 8) or not np.isfinite(rows).all():
        raise ValueError('Unexpected public trajectory shape or nonfinite output')
    pose = poses_to_matrices(rows[:, 1:8])
    return np.einsum('nji,nj->ni', pose[:-1, :3, :3], np.diff(pose[:, :3, 3], axis=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.with_suffix('.json').exists() or args.out.with_suffix('.md').exists():
        parser.error('Preserve earlier evidence; choose a new output prefix')
    expected = [e['hash'] for e in json.loads((ROOT / 'configs/public-episodes.json').read_text())]
    names = ('baseline', 'current', 'gyro-only', 'calibrated-residual', 'calibrated-gyro', 'selected-repeat', 'calibrated-gyro-repeat')
    runs = {n: load_run(n, ROOT / ('results/gyro-v4-' + n), expected) for n in names}
    selection_path = ROOT / 'experiments/gyro-bias-v4/selection.json'
    selected = json.loads(selection_path.read_text())['selected_variant']
    if runs[selected]['manifest']['wasm_sha256'] != runs['selected-repeat']['manifest']['wasm_sha256']:
        raise ValueError('Selected repeat used a different artifact')
    repeat = same_outputs(runs[selected], runs['selected-repeat'], require_fuel=True)
    if runs['calibrated-gyro']['manifest']['wasm_sha256'] != runs['calibrated-gyro-repeat']['manifest']['wasm_sha256']:
        raise ValueError('Diagnostic repeat used a different artifact')
    diagnostic_repeat = same_outputs(runs['calibrated-gyro'], runs['calibrated-gyro-repeat'], require_fuel=True)
    controls = {}
    for name, directory in [('baseline', 'rotation-v3-baseline'), ('current', 'rotation-v3-current'),
                            ('gyro-only', 'ablation-v2-no-rotation')]:
        old = load_run('old-' + name, ROOT / 'results' / directory, expected)
        controls[name] = same_outputs(old, runs[name])
    episodes = []
    for index, digest in enumerate(expected):
        metrics = {n: runs[n]['report']['episodes'][index]['metrics'] for n in names}
        local = {n: local_translations(runs[n]['report_path'].parent / 'logs' / digest / 'estimate.tum')
                 for n in names[1:5]}
        maximum = max(float(np.linalg.norm(a - local['current'], axis=1).max()) for a in local.values())
        if maximum > 2e-6:
            raise ValueError('Local translation changed beyond saved-text precision')
        changes = {}
        for key, factor in [('rotation_rmse', 180 / np.pi), ('sim3_ate_rmse', 1.0), ('ate_rmse', 1.0)]:
            on = (metrics['calibrated-residual'][key] - metrics['current'][key]) * factor
            off = (metrics['calibrated-gyro'][key] - metrics['gyro-only'][key]) * factor
            changes[key] = {'calibration_effect_residual_on': on, 'calibration_effect_residual_off': off,
                            'difference_of_effects': on - off}
        episodes.append({'hash': digest, 'max_local_translation_difference_m': maximum,
                         'effects': changes})
    result = {'selected_variant': selected, 'selection_sha256': sha(selection_path),
              'unchanged_controls': controls, 'selected_exact_repeat': repeat,
              'unselected_calibrated_gyro_exact_repeat': diagnostic_repeat,
              'per_episode': episodes,
              'interpretation': 'Effects are calibrated minus uncalibrated error; negative helps. Interaction is descriptive, not proof of a double-compensation mechanism.',
              'evidence_scope': 'Full repeated public outputs establish deterministic reproduction, not independent-startup or physical-robot generalization.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Paired calibration effects and output verification', '',
             'All three uncalibrated controls reproduce earlier pose/twist files and accuracy exactly.',
             f'The selected `{selected}` artifact reproduces all ten pose/twist files, accuracy and per-frame fuel exactly.',
             'The unselected calibrated-gyro diagnostic also reproduces all ten pose/twist files, accuracy and per-frame fuel exactly.',
             'Wall time is measured separately and need not repeat exactly.', '',
             '| Episode | Rotation change with residual ON (deg) | Rotation change with residual OFF (deg) | Difference of changes (deg) |',
             '| --- | ---: | ---: | ---: |']
    for e in episodes:
        v = e['effects']['rotation_rmse']
        lines.append(f"| {e['hash'][:8]} | {v['calibration_effect_residual_on']:.6f} | {v['calibration_effect_residual_off']:.6f} | {v['difference_of_effects']:.6f} |")
    lines += ['', result['interpretation'], '', result['evidence_scope'], '',
              f"Maximum camera-local translation discrepancy across four cells is {max(e['max_local_translation_difference_m'] for e in episodes):.3g} m, within saved-pose serialization tolerance."]
    args.out.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print('Verified unchanged controls, exact selected repeat including fuel, and all four local-translation contracts.')


if __name__ == '__main__':
    main()
