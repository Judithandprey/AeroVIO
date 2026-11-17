"""Causal, balanced three-arm evaluation of frozen V8 and structural lines.

Run development first, then confirmation after generation/builds finish. This
runner never tunes parameters, selects flights by scores, or promotes a model.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
from viobench.tools.datasets.storage import read_episode
from viobench.tools.geometry import poses_to_matrices, rotation_vectors
from viobench.tools.runner.engine import Engine
from viobench.tools.runner.metrics import evaluate_trajectory
from viobench.tools.runner.sandbox import Limits, Sandbox

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/structural-lines-v13'
ARCHIVE = ROOT / 'experiments/structural-lines-v13'
NAMES = ('baseline', 'detect', 'correct')
BASELINE_SHA = '50fac7f6d17e37f0cc5f473a326bca70d0fc6bc602a502fc0216bb17a5ae19ac'
COUNTER_COUNT = 16


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def balanced_orders(frame_count, seed):
    """All six orders per block; a cyclic tail balances 200-frame episodes."""
    rng = np.random.default_rng(seed)
    orders = list(itertools.permutations(NAMES))
    result = []
    for _ in range(frame_count // 6):
        result.extend(orders[int(i)] for i in rng.permutation(6))
    base = orders[int(rng.integers(6))]
    # Three cyclic orders give each artifact every position exactly once;
    # their reverse family completes all six unique permutations.
    tail = [base[i:] + base[:i] for i in range(3)]
    reverse = (base[0], base[2], base[1])
    tail += [reverse[i:] + reverse[:i] for i in range(3)]
    result.extend(tail[:frame_count % 6])
    counts = {n: [sum(order[p] == n for order in result) for p in range(3)] for n in NAMES}
    assert all(max(c) - min(c) <= 1 for c in counts.values())
    return result, counts


def source_snapshot(cfg):
    paths = [ROOT / 'configs/structural-lines-v13.json', ROOT / cfg['limits'],
             ROOT / cfg['checkpoint'], (ROOT / cfg['checkpoint']).with_name('meta.json'),
             ROOT / 'upstream.lock.json', Path(__file__), ROOT / 'scripts/report_structural_lines.py']
    paths += [ROOT / p for p in ('configs/long-validation-v13.json',
              'scripts/build_structural_lines.py', 'scripts/generate_line_validation.py',
              'scripts/probe_structural_lines.py', 'scripts/test_native_fusion.py', 'scripts/test_structural_lines.py',
              'scripts/generate_long_validation.py', 'scripts/generate_training_data.py')
              if (ROOT / p).exists()]
    paths += sorted(p for p in (ROOT / 'workspace/structural_lines_native').iterdir() if p.is_file())
    paths += sorted(p for p in (ROOT / 'tests').glob('structural_*') if p.is_file())
    assert sha(ROOT / cfg['checkpoint']) == cfg['checkpoint_sha256']
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}


def artifact_snapshot(artifacts):
    result = {}
    for name, entry in artifacts.items():
        path = Path(entry['path'])
        if not path.is_absolute():
            path = ROOT / path
        assert path.is_file() and sha(path) == entry['sha256'], name
        result[name] = entry['sha256']
    assert result['baseline'] == BASELINE_SHA
    return result


def diagnostic_reader(sandbox, cfg, required):
    exports = sandbox.instance.exports(sandbox.store)
    if 'line_diagnostics' not in exports:
        assert not required, 'missing line_diagnostics export'
        return lambda: None
    size = len(cfg['diag_keys']) * 8
    assert size == 21 * 8
    pointer = sandbox.alloc(size)

    def sample():
        exports['line_diagnostics'](sandbox.store, pointer)
        values = np.frombuffer(sandbox.read(pointer, size), dtype='<f8').copy()
        assert np.isfinite(values).all()
        return values

    return sample


def should_sample(index, frame_count):
    return index % 8 == 6 or index == frame_count - 1


def diagnostic_summary(values, cfg):
    if not values:
        return None
    array = np.asarray(values)
    assert np.all(np.diff(array[:, :COUNTER_COUNT], axis=0) >= -1e-10)
    final = dict(zip(cfg['diag_keys'], array[-1].tolist()))
    attempts = final['decision_attempts']
    sampled = final['sampled_images']
    return {'final': final,
            'acceptance_rate_per_decision': final['accepted'] / attempts if attempts else None,
            'usable_two_direction_rate_per_sample': final['usable_two_direction_images'] / sampled if sampled else None,
            'rejection_rates_per_decision': {k: v / attempts if attempts else None
                                           for k, v in final.items() if k.startswith('reject_')}}


def save_diagnostics(directory, indices, times, values, cfg):
    previous = np.zeros(len(cfg['diag_keys']))
    path = directory / 'decisions.jsonl'
    with path.open('x', encoding='utf-8') as stream:
        for frame, row in zip(indices, values, strict=True):
            delta = row - previous
            record = {'frame': frame, 'phase': frame % 8, 'time_s': float(times[frame]),
                      'values': dict(zip(cfg['diag_keys'], row.tolist())),
                      'counter_delta': dict(zip(cfg['diag_keys'][:COUNTER_COUNT], delta[:COUNTER_COUNT].tolist())),
                      'reason': cfg['reason_codes'].get(str(int(row[-1])), 'unknown'),
                      'has_decision_since_previous_sample': bool(delta[cfg['diag_keys'].index('decision_attempts')] > 0)}
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            previous = row
    return sha(path)


def repeat_episode(sandbox, ep, cfg):
    """Same diagnostic schedule/allocations as timed arm, but no timing claim."""
    sample = diagnostic_reader(sandbox, cfg, required=True)
    outputs, fuel, diagnostics = [], [], []
    for i, (time, frame) in enumerate(zip(ep.frame_times, ep.adapt((320, 240))[1], strict=True)):
        step = sandbox.step(float(time), frame, ep.imu_window(i).copy())
        outputs.append(np.r_[step.pose, step.twist])
        fuel.append(step.fuel)
        if should_sample(i, len(ep.frame_times)):
            diagnostics.append(sample())
    return np.asarray(outputs), np.asarray(fuel), np.asarray(diagnostics)


def phase_fuel(values):
    values = np.asarray(values)
    return {str(phase): {'frames': len(values[phase::8]),
                        'mean': float(values[phase::8].mean()),
                        'maximum': int(values[phase::8].max())} for phase in range(8)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('development', 'confirmation'), required=True)
    args = parser.parse_args()
    phase = args.phase
    cfg = read(ROOT / 'configs/structural-lines-v13.json')
    assert tuple(cfg['variants']) == NAMES
    artifacts = read(ARCHIVE / 'artifacts.json')
    identity = {'source_sha256': source_snapshot(cfg), 'artifact_sha256': artifact_snapshot(artifacts),
                'artifacts_manifest_sha256': sha(ARCHIVE / 'artifacts.json')}
    freeze_path = ARCHIVE / 'frozen-before-confirmation.json'
    if phase == 'confirmation':
        frozen = read(freeze_path)
        assert frozen['identity'] == identity, 'frozen source/artifact changed'
        assert sha(ARCHIVE / 'development.json') == frozen['development_sha256']
    else:
        assert not freeze_path.exists(), 'development already frozen'
    manifest_path = ROOT / cfg[phase + '_manifest']
    manifest = read(manifest_path)
    entries = manifest['validation']
    assert manifest['status'] == 'complete' and len(entries) == 4
    expected_frames = 200 if phase == 'development' else 1200
    expected_scenes = {'ProcTHOR-Train-5', 'ProcTHOR-Train-6'} if phase == 'development' else {'ProcTHOR-Train-8', 'ProcTHOR-Train-9'}
    assert {e['scene'] for e in entries} == expected_scenes
    assert len({(e['scene'], e['seed']) for e in entries}) == len(entries)
    seed_disjointness = None
    if phase == 'confirmation':
        development = read(ARCHIVE / 'development-manifest.json')['validation']
        assert not {e['scene'] for e in entries}.intersection(e['scene'] for e in development)
        prior_seeds, prior_manifests = set(), {}
        for old_path in sorted((ROOT / 'data').glob('*/data_manifest.json')):
            if old_path.resolve() == manifest_path.resolve():
                continue
            old = read(old_path)
            prior_manifests[str(old_path.relative_to(ROOT))] = sha(old_path)
            for part in ('train', 'validation', 'attempts'):
                prior_seeds.update(e['seed'] for e in old.get(part, []) if isinstance(e, dict) and e.get('seed') is not None)
        requested_seeds = {e['seed'] for e in entries}
        assert not requested_seeds.intersection(prior_seeds), 'confirmation seed reused from earlier data or attempt'
        seed_disjointness = {'confirmation_seeds': sorted(requested_seeds), 'prior_unique_seed_count': len(prior_seeds),
                             'prior_manifest_sha256': prior_manifests, 'development_scenes_disjoint': True}
    output = OUT / phase
    assert not (ARCHIVE / (phase + '.json')).exists(), 'preserve existing phase evidence'
    output.mkdir(parents=True, exist_ok=False)
    upstream = ROOT / 'external/viobench'
    upstream_commit = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    assert upstream_commit == read(ROOT / 'upstream.lock.json')['commit']
    assert not subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip()
    environment = {'python': sys.version, 'host': platform.platform(), 'upstream_commit': upstream_commit,
                   'source_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                   'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'wasmtime', 'viobench', 'opencv-python-headless')},
                   'started_utc': datetime.now(timezone.utc).isoformat()}
    write(ARCHIVE / ('environment-' + phase + '.json'), environment, exclusive=True)
    engine = Engine()
    limits = Limits.from_dict(read(ROOT / cfg['limits']))
    rows = {n: {'episodes': []} for n in NAMES}
    all_fuel, all_wall = {n: [] for n in NAMES}, {n: [] for n in NAMES}
    checks, fingerprints = [], set()
    if phase == 'confirmation':
        fingerprints.update(c['gt_prefix_sha256'] for c in read(ARCHIVE / 'development.json')['checks'])
    for episode_index, entry in enumerate(entries):
        ep = read_episode(ROOT / entry['path'], frames='lazy')
        config, frames = ep.adapt((320, 240))
        times = ep.frame_times
        assert len(times) == expected_frames
        sandboxes = {n: Sandbox(Path(artifacts[n]['path']), engine, limits=limits,
                               log_dir=output / str(episode_index) / n) for n in NAMES}
        for sandbox in sandboxes.values():
            sandbox.init(config)
        readers = {n: diagnostic_reader(sandboxes[n], cfg, n != 'baseline') for n in NAMES}
        outputs, fuels, walls, diagnostics = ({n: [] for n in NAMES} for _ in range(4))
        diagnostic_indices = []
        orders, counts = balanced_orders(len(times), cfg['order_seed'] + episode_index)
        input_digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
        for i, (time, frame) in enumerate(zip(times, frames, strict=True)):
            imu = ep.imu_window(i).copy()
            for array in (np.asarray(time), frame, imu):
                input_digest.update(np.ascontiguousarray(array).tobytes())
            for name in orders[i]:
                step = sandboxes[name].step(float(time), frame, imu)
                outputs[name].append(np.r_[step.pose, step.twist])
                fuels[name].append(step.fuel)
                walls[name].append(step.wall_seconds)
            # Do not insert diagnostic calls between the three measured steps.
            if should_sample(i, len(times)):
                diagnostic_indices.append(i)
                for name in NAMES:
                    value = readers[name]()
                    if value is not None:
                        diagnostics[name].append(value)
        outputs = {n: np.asarray(v) for n, v in outputs.items()}
        assert np.array_equal(outputs['detect'], outputs['baseline']), 'detector-only changes pose/twist'
        extra = Sandbox(Path(artifacts['correct']['path']), engine, limits=limits,
                        log_dir=output / str(episode_index) / 'correct-repeat')
        extra.init(config)
        repeated, repeated_fuel, repeated_diag = repeat_episode(extra, ep, cfg)
        assert np.array_equal(repeated, outputs['correct'])
        assert np.array_equal(repeated_fuel, np.asarray(fuels['correct']))
        assert np.array_equal(repeated_diag, np.asarray(diagnostics['correct']))
        control_exact = None
        if phase == 'development' and episode_index == 0 and 'control' in artifacts:
            control = Sandbox(Path(artifacts['control']['path']), engine, limits=limits,
                              log_dir=output / str(episode_index) / 'control-check')
            control.init(config)
            actual = []
            for i, (time, frame) in enumerate(zip(times, ep.adapt((320, 240))[1], strict=True)):
                step = control.step(float(time), frame, ep.imu_window(i).copy())
                actual.append(np.r_[step.pose, step.twist])
            control_exact = bool(np.array_equal(np.asarray(actual), outputs['baseline']))
            assert control_exact
        # Ground truth is first accessed after all causal executions for this flight.
        truth = ep.ground_truth_state()
        fingerprint = hashlib.sha256(truth[:200].tobytes()).hexdigest()
        assert fingerprint not in fingerprints, 'duplicate trajectory prefix'
        fingerprints.add(fingerprint)
        consistency = {}
        for name in NAMES:
            actual = outputs[name]
            assert np.isfinite(actual).all()
            rotations = poses_to_matrices(actual[:, :7])[:, :3, :3]
            velocity = np.einsum('nji,nj->ni', rotations[1:], np.diff(actual[:, :3], axis=0)) / np.diff(times)[:, None]
            angular = rotation_vectors(rotations[:-1].transpose(0, 2, 1) @ rotations[1:]) / np.diff(times)[:, None]
            consistency[name] = {'linear_max_difference': float(np.max(np.abs(velocity - actual[1:, 7:10]))),
                                 'angular_max_difference': float(np.max(np.abs(angular - actual[1:, 10:13])))}
            assert consistency[name]['linear_max_difference'] < 1e-8
            assert consistency[name]['angular_max_difference'] < 1e-6
            metrics = evaluate_trajectory(truth, actual, times).as_dict()
            assert metrics['frames'] == expected_frames and metrics['filled_frames'] == 0
            directory = output / str(episode_index) / name
            diag_hash = save_diagnostics(directory, diagnostic_indices if diagnostics[name] else [],
                                         times, diagnostics[name], cfg)
            np.savez_compressed(directory / 'trace.npz', times=times, outputs=actual,
                                fuel=fuels[name], wall=walls[name],
                                diagnostic_indices=diagnostic_indices if diagnostics[name] else [],
                                diagnostics=np.asarray(diagnostics[name]))
            peak = int(np.argmax(fuels[name]))
            rows[name]['episodes'].append({'scene': entry['scene'], 'seed': entry['seed'], 'metrics': metrics,
                'rotation_rmse_deg': float(np.degrees(metrics['rotation_rmse'])),
                'rpe_rotation_1s_deg': float(np.degrees(metrics['rpe_rotation_rmse'])),
                'max_fuel': int(fuels[name][peak]), 'mean_fuel': float(np.mean(fuels[name])),
                'peak_frame': peak, 'peak_phase': peak % 8, 'phase_fuel': phase_fuel(fuels[name]),
                'median_step_ms': float(np.median(walls[name]) * 1000),
                'diagnostics_summary': diagnostic_summary(diagnostics[name], cfg),
                'input_sha256': input_digest.hexdigest(), 'trace_sha256': sha(directory / 'trace.npz'),
                'decision_log_sha256': diag_hash})
            all_fuel[name].extend(fuels[name])
            all_wall[name].extend(walls[name])
        checks.append({'scene': entry['scene'], 'seed': entry['seed'], 'gt_prefix_sha256': fingerprint,
                       'execution_positions': counts, 'detect_outputs_exact_baseline': True,
                       'correct_repeat_outputs_fuel_diagnostics_exact': True,
                       'control_rebuild_outputs_exact_baseline': control_exact, 'twist_consistency': consistency})
        write(output / 'progress.json', {'completed': episode_index + 1, 'total': len(entries)})
        print('EVALUATED', phase, episode_index, entry['scene'], entry['seed'], flush=True)
    for name in NAMES:
        episodes = rows[name]['episodes']
        counters = {k: sum(e['diagnostics_summary']['final'][k] for e in episodes)
                    for k in cfg['diag_keys'][:COUNTER_COUNT]} if name != 'baseline' else None
        rows[name]['summary'] = {
            'mean_sim3_ate_m': float(np.mean([e['metrics']['sim3_ate_rmse'] for e in episodes])),
            'mean_raw_ate_m': float(np.mean([e['metrics']['ate_rmse'] for e in episodes])),
            'mean_rotation_rmse_deg': float(np.mean([e['rotation_rmse_deg'] for e in episodes])),
            'mean_rpe_rotation_1s_deg': float(np.mean([e['rpe_rotation_1s_deg'] for e in episodes])),
            'mean_fuel': float(np.mean(all_fuel[name])), 'max_fuel': int(max(all_fuel[name])),
            'median_step_ms': float(np.median(all_wall[name]) * 1000), 'diagnostic_counters': counters,
            'acceptance_rate_per_decision': counters['accepted'] / counters['decision_attempts']
                if counters and counters['decision_attempts'] else None}
    assert source_snapshot(cfg) == identity['source_sha256'], 'sources changed during evaluation'
    assert artifact_snapshot(artifacts) == identity['artifact_sha256']
    result = {'scope': phase, 'variants': rows, 'checks': checks, 'manifest_sha256': sha(manifest_path),
              'confirmation_seed_disjointness': seed_disjointness,
              'input_contract': 'Identical current RGB/IMU/calibration; no truth/future data in inference.',
              'wall_time_comparable': phase == 'confirmation', 'identity': identity,
              'timing_note': 'Confirmation must run after all generation/build work ends; diagnostics/repeats excluded.',
              'finished_utc': datetime.now(timezone.utc).isoformat()}
    write(ARCHIVE / (phase + '.json'), result, exclusive=True)
    write(ARCHIVE / (phase + '-manifest.json'), manifest, exclusive=True)
    if phase == 'development':
        write(freeze_path, {'identity': identity, 'development_sha256': sha(ARCHIVE / 'development.json'),
                           'frozen_utc': datetime.now(timezone.utc).isoformat(),
                           'policy': 'No rule or artifact changes after this freeze; confirmation cannot tune or promote.'}, exclusive=True)
    print('SUMMARY', phase, json.dumps({n: v['summary'] for n, v in rows.items()}), flush=True)


if __name__ == '__main__':
    main()
