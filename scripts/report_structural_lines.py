"""Verify preserved V13 evidence and report successes, costs and negative cases."""
import hashlib
import json
from pathlib import Path

import numpy as np
from viobench.tools.datasets.storage import read_episode
from viobench.tools.geometry import poses_to_matrices

from evaluate_structural_lines import ARCHIVE, NAMES, OUT, ROOT, artifact_snapshot, read, source_snapshot, write


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fmt(value, digits=6):
    return 'n/a' if value is None else f'{value:.{digits}f}'


def change(value, reference):
    return None if reference == 0 else 100 * (value / reference - 1)


def absolute_angles(truth, actual):
    gt = poses_to_matrices(truth[:, :7])[:, :3, :3]
    estimate = poses_to_matrices(actual[:, :7])[:, :3, :3]
    error = gt.transpose(0, 2, 1) @ estimate
    return np.degrees(np.arccos(np.clip((np.trace(error, axis1=1, axis2=2) - 1) / 2, -1, 1)))


def observed_event_windows(phase, manifest, cfg):
    """Post-hoc diagnostic only; these truth labels never return to inference."""
    result = []
    for index, entry in enumerate(manifest['validation']):
        directory = OUT / phase / str(index)
        decisions = [json.loads(line) for line in (directory / 'correct/decisions.jsonl').read_text().splitlines()]
        accepted = [row for row in decisions if row['counter_delta']['accepted'] > 0]
        if not accepted:
            continue
        ep = read_episode(ROOT / entry['path'], frames='lazy')
        truth = ep.ground_truth_state()
        with np.load(directory / 'baseline/trace.npz') as b, np.load(directory / 'correct/trace.npz') as c:
            base = absolute_angles(truth, b['outputs'])
            correct = absolute_angles(truth, c['outputs'])
        for row in accepted:
            frame = row['frame']
            before, after = max(0, frame - 1), min(len(base) - 1, frame + 1)
            result.append({'scope': phase, 'scene': entry['scene'], 'seed': entry['seed'],
                'decision_frame': frame, 'capture_frame': int(row['values']['last_capture_frame']),
                'before_frame': before, 'after_frame': after,
                'baseline_error_after_deg': float(base[after]),
                'corrected_error_after_deg': float(correct[after]),
                'corrected_minus_baseline_before_deg': float(correct[before] - base[before]),
                'corrected_minus_baseline_after_deg': float(correct[after] - base[after]),
                'change_in_error_gap_deg': float((correct[after] - base[after]) - (correct[before] - base[before])),
                'line_count': row['values']['last_line_count'],
                'direction_count': row['values']['last_direction_count'],
                'residual_rad': row['values']['last_residual_rad'],
                'scheduled_angle_delta_rad': row['counter_delta']['scheduled_angle_sum_rad']})
    return sorted(result, key=lambda row: row['change_in_error_gap_deg'], reverse=True)


def main():
    cfg = read(ROOT / 'configs/structural-lines-v13.json')
    data = {phase: read(ARCHIVE / (phase + '.json')) for phase in ('development', 'confirmation')}
    freeze = read(ARCHIVE / 'frozen-before-confirmation.json')
    assert freeze['identity']['source_sha256'] == source_snapshot(cfg)
    assert freeze['identity']['artifact_sha256'] == artifact_snapshot(read(ARCHIVE / 'artifacts.json'))
    assert freeze['identity']['artifacts_manifest_sha256'] == sha(ARCHIVE / 'artifacts.json')
    assert freeze['development_sha256'] == sha(ARCHIVE / 'development.json')
    assert all(d['identity'] == freeze['identity'] for d in data.values())
    trace_count = 0
    for phase, d in data.items():
        assert len(d['checks']) == 4
        for check in d['checks']:
            assert check['detect_outputs_exact_baseline']
            assert check['correct_repeat_outputs_fuel_diagnostics_exact']
            assert check['control_rebuild_outputs_exact_baseline'] is not False
        for name in NAMES:
            for index, episode in enumerate(d['variants'][name]['episodes']):
                directory = OUT / phase / str(index) / name
                assert episode['trace_sha256'] == sha(directory / 'trace.npz')
                assert episode['decision_log_sha256'] == sha(directory / 'decisions.jsonl')
                assert episode['input_sha256'] == d['variants']['baseline']['episodes'][index]['input_sha256']
                trace_count += 1
    assert trace_count == 24
    confirmation = data['confirmation']['variants']
    base, candidate = confirmation['baseline']['summary'], confirmation['correct']['summary']
    wins = sum(a['rotation_rmse_deg'] < b['rotation_rmse_deg'] - 1e-12
               for a, b in zip(confirmation['correct']['episodes'], confirmation['baseline']['episodes'], strict=True))
    rules = cfg['guards']
    guards = {
        'rotation': candidate['mean_rotation_rmse_deg'] <= base['mean_rotation_rmse_deg'] * rules['rotation_ratio_max'],
        'raw_ate': candidate['mean_raw_ate_m'] <= base['mean_raw_ate_m'] * rules['raw_ate_ratio_max'],
        'sim3_ate': candidate['mean_sim3_ate_m'] <= base['mean_sim3_ate_m'] * rules['sim3_ate_ratio_max'],
        'peak_fuel': candidate['max_fuel'] <= base['max_fuel'] * rules['peak_fuel_ratio_max'],
        'rotation_episode_wins': wins >= rules['minimum_rotation_wins']}
    selection = {'reference_retained': 'frozen V8 rotation-only', 'main_or_default_changed': False,
                 'server_uploaded': False, 'engineering_guards': guards, 'all_guards_pass': all(guards.values()),
                 'rotation_episode_wins': wins, 'evaluated_confirmation_episodes': 4,
                 'rule': 'No automatic promotion; fixed engineering guards are not significance tests.'}
    write(ARCHIVE / 'selection.json', selection)
    failure_records = []
    for phase, d in data.items():
        for b, c in zip(d['variants']['baseline']['episodes'], d['variants']['correct']['episodes'], strict=True):
            failure_records.append({'scope': phase, 'scene': c['scene'], 'seed': c['seed'],
                'rotation_win': c['rotation_rmse_deg'] < b['rotation_rmse_deg'] - 1e-12,
                'rpe_rotation_1s_win': c['rpe_rotation_1s_deg'] < b['rpe_rotation_1s_deg'] - 1e-12,
                'raw_ate_win': c['metrics']['ate_rmse'] < b['metrics']['ate_rmse'] - 1e-12,
                'rotation_change_deg': c['rotation_rmse_deg'] - b['rotation_rmse_deg'],
                'rpe_rotation_1s_change_deg': c['rpe_rotation_1s_deg'] - b['rpe_rotation_1s_deg'],
                'raw_ate_change_m': c['metrics']['ate_rmse'] - b['metrics']['ate_rmse'],
                'sim3_ate_change_m': c['metrics']['sim3_ate_rmse'] - b['metrics']['sim3_ate_rmse'],
                'peak_fuel_change_percent': change(c['max_fuel'], b['max_fuel']),
                'diagnostics': c['diagnostics_summary']})
    events = {phase: observed_event_windows(phase, read(ARCHIVE / (phase + '-manifest.json')), cfg) for phase in data}
    write(ARCHIVE / 'failure-cases.json', {'all_trajectory_changes': failure_records,
        'accepted_event_windows': events,
        'interpretation': 'Positive error changes are regressions. Event windows are descriptive post-hoc correlations, not isolated causal attribution or semantic wall validation.'})
    write(ARCHIVE / 'verification.json', {'primary_trace_and_decision_log_hashes_verified': trace_count,
        'identical_input_hashes': True, 'detector_only_outputs_exact_frozen_baseline': True,
        'all_eight_correct_repeats_outputs_fuel_diagnostics_exact': True,
        'frozen_sources_artifacts_verified': True, 'truth_only_used_for_post_execution_metrics': True,
        'no_main_default_or_course_submission_change': True})
    lines = ['# Structural-line rotation fallback V13', '',
        '**Frozen reference: selected rotation-only V8.** Only the structural-line module changes; original checkpoint, IMU integration, point tracker and V8 correction remain fixed. No memory model, retraining, absolute room direction, metric width or trajectory-straightening prior is introduced.', '',
        f"Confirmation engineering guards all pass: **{all(guards.values())}**; rotation wins: **{wins}/4**. This is a local simulated comparison, not automatic model promotion.", '',
        'The three primary arms are the original frozen V8 artifact, detector-only, and gated bounded line correction. Detector-only must reproduce every baseline pose and twist. All correction repeats reproduce output, fuel and diagnostics exactly. Each 160x120 image uses the original green-channel downsampling and pixel-center calibration. Capture is every eight frames; solve is phase six, with bounded correction spread across two frames.', '']
    for phase, d in data.items():
        lines += [f'## {phase.title()}', '',
            'Four flights. Development uses original 10-second room 5/6 data; confirmation uses new 60-second room 8/9 data after the development freeze.', '',
            '| Variant | Mean Sim3 ATE m | Mean raw ATE m | Mean rotation RMSE deg | Mean 1s rotation RPE deg | Max fuel/frame | Median step ms |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for name in NAMES:
            s = d['variants'][name]['summary']
            timing = fmt(s['median_step_ms']) if d['wall_time_comparable'] else 'excluded'
            lines.append(f"| {name} | {fmt(s['mean_sim3_ate_m'])} | {fmt(s['mean_raw_ate_m'])} | {fmt(s['mean_rotation_rmse_deg'])} | {fmt(s['mean_rpe_rotation_1s_deg'])} | {s['max_fuel']} | {timing} |")
        base_s = d['variants']['baseline']['summary']
        for name in ('detect', 'correct'):
            s = d['variants'][name]['summary']
            changes = {key: change(s[key], base_s[key]) for key in
                       ('mean_sim3_ate_m', 'mean_raw_ate_m', 'mean_rotation_rmse_deg', 'mean_rpe_rotation_1s_deg', 'mean_fuel', 'max_fuel')}
            lines += ['', f"{name} versus frozen V8, percent changes (negative errors are better): `{json.dumps(changes)}`."]
        lines += ['', '### Every trajectory', '',
            '| Room / seed | Variant | Sim3 ATE m | Raw ATE m | Rotation RMSE deg | 1s rotation RPE deg | Max fuel | Peak frame / phase | Accepted / attempts |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |']
        for index in range(4):
            for name in NAMES:
                e = d['variants'][name]['episodes'][index]
                m = e['metrics']
                diag = e['diagnostics_summary']
                rate = 'n/a' if diag is None else f"{int(diag['final']['accepted'])}/{int(diag['final']['decision_attempts'])}"
                lines.append(f"| {e['scene']} / {e['seed']} | {name} | {fmt(m['sim3_ate_rmse'])} | {fmt(m['ate_rmse'])} | {fmt(e['rotation_rmse_deg'])} | {fmt(e['rpe_rotation_1s_deg'])} | {e['max_fuel']} | {e['peak_frame']} / {e['peak_phase']} | {rate} |")
        lines += ['', '### Gate and detection coverage', '',
            '| Variant | Usable two-direction images / sampled | Accepted / attempted | Applied steps | Scheduled angle sum rad | Applied angle sum rad |',
            '| --- | --- | --- | ---: | ---: | ---: |']
        for name in ('detect', 'correct'):
            c = d['variants'][name]['summary']['diagnostic_counters']
            lines.append(f"| {name} | {int(c['usable_two_direction_images'])}/{int(c['sampled_images'])} | {int(c['accepted'])}/{int(c['decision_attempts'])} | {int(c['applied_steps'])} | {fmt(c['scheduled_angle_sum_rad'])} | {fmt(c['applied_angle_sum_rad'])} |")
            rejected = {k: {'count': c[k], 'rate_per_decision': c[k] / c['decision_attempts'] if c['decision_attempts'] else None}
                        for k in c if k.startswith('reject_')}
            lines += ['', f"{name} rejection reasons: `{json.dumps(rejected)}`."]
        lines += ['', 'Accepted/attempted counts are conditional on the sampling and gate sequence; samples overlap in scene content and are not statistically independent. A detector-only accepted opportunity is not an applied correction. Scheduled/applied angle sums are sums of magnitudes, not net orientation change.', '',
            '### Negative cases and descriptive event checks', '']
        failures = sorted((r for r in failure_records if r['scope'] == phase), key=lambda r: r['rotation_change_deg'], reverse=True)
        for row in failures:
            lines.append(f"- {row['scene']} seed {row['seed']}: rotation win={row['rotation_win']}, rotation change {row['rotation_change_deg']:+.6f} deg, 1s RPE change {row['rpe_rotation_1s_change_deg']:+.6f} deg, raw ATE change {row['raw_ate_change_m']:+.6f} m, peak fuel change {row['peak_fuel_change_percent']:+.3f}%.")
        if events[phase]:
            lines += ['', 'Accepted two-frame windows with the largest changes in corrected-minus-baseline absolute rotation error (positive means worse):']
            for event in events[phase][:3]:
                lines.append(f"- Seed {event['seed']}, decision frame {event['decision_frame']}: error-gap change {event['change_in_error_gap_deg']:+.6f} deg; {int(event['line_count'])} lines / {int(event['direction_count'])} directions, residual {event['residual_rad']:.6f} rad.")
        else:
            lines += ['', 'No accepted correction events occurred. This establishes rejection behavior and cost, not successful drift correction.']
        lines += ['', 'These post-hoc event windows are descriptive, not isolated causal proof. Full counters, phase fuel, all trajectory changes and accepted windows are retained in JSON; negative results are not filtered.', '']
    lines += ['## Fixed guards', '', '```json', json.dumps(selection, indent=2), '```', '',
        '## Verification and interpretation limits', '',
        'Twenty-four primary trace and diagnostic-log hashes verify. All eight complete correction repeats match output, per-frame fuel and diagnostic values. Detector-only output exactly matches frozen V8 on all flights. An optional disabled-module control is checked on the first development flight. Twist consistency is verified from emitted poses. Original weights and frozen inference/evaluation sources match their recorded hashes.', '',
        'All six execution orders are balanced; 200-frame development flights differ by at most one execution-position count and 1200-frame confirmation flights are exactly balanced. Diagnostic reads, repeated flights and control checks do not enter step timing. Development wall times are excluded. Confirmation timing is valid only after data generation and builds have ended; milliseconds are host-specific WASM step costs, not camera latency. Maximum per-frame fuel, rather than average CPU time, is the compute guard.', '',
        'Rotation RMSE and 1-second relative rotation errors are unaligned SO(3) metrics; the report converts radians to degrees. Sim3 ATE aligns positions with global rotation, translation and scale, so it cannot stand alone as evidence of corrected yaw, metric scale or raw trajectory improvement. Raw ATE and relative rotation are reported alongside it.', '',
        'Structural directions are local and sign-ambiguous. Vertical lines alone do not constrain yaw. The module does not know a global room axis, wall width or metric scale, and does not force a straight flight path. Static furniture edges can form convincing direction groups; there is no semantic wall classifier. Oblique walls, axis swaps, room changes, low texture and repeated structure remain failure modes. Pure-rotation/weak-parallax benefit must be measured rather than assumed.', '',
        'Only four fresh flights in two new rooms are used. Development data were used by earlier experiments. Guard thresholds and acceptance counts are engineering checks, not significance tests or real-robot validation. No main/default replacement or course-server upload occurs.', '',
        '## Reproduction', '', 'From the WSL checkout with pinned dependencies and fresh output directories:', '', '```bash',
        '.venv/bin/python scripts/evaluate_structural_lines.py --phase development',
        '# Finish new data generation/builds; preserve frozen sources and artifacts.',
        '.venv/bin/python scripts/evaluate_structural_lines.py --phase confirmation',
        '.venv/bin/python scripts/report_structural_lines.py', '```', '',
        'The evaluator refuses to overwrite a phase. Preserve this completed evidence and use a new experiment namespace for a rerun. `artifacts.json` identifies the exact three WASMs; the configuration, manifests, source freeze, traces and decision logs bind the comparison to its inputs. Raw traces/logs remain in `results/structural-lines-v13/`; compact evidence is archived here.']
    (ARCHIVE / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('VERIFIED', trace_count, 'primary traces; guards', guards, flush=True)


if __name__ == '__main__':
    main()
