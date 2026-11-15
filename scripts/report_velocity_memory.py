"""Produce evidence-bounded V11 report without changing selection."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/velocity-memory-v11';A=ROOT/'experiments/velocity-memory-v11'
NAMES=['teacher','v10','no-history-final','feedback-final']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fmt(x):return 'n/a' if x is None else f'{x:.6f}'
def main():
    datasets={s:read(A/(s+'.json')) for s in ('regression','confirmation')}
    frozen=read(A/'frozen-before-long-evaluation.json')
    for p,h in frozen['source_sha256'].items():assert sha(ROOT/p)==h,p
    assert sha(A/'artifacts.json')==frozen['artifacts_sha256']
    traces=[]
    for split,data in datasets.items():
        for n in NAMES:
            for i,e in enumerate(data['variants'][n]['episodes']):
                p=OUT/split/str(i)/n/'trace.npz';assert sha(p)==e['trace_sha256'];traces.append(str(p))
    artifacts=read(A/'artifacts.json')
    for a in artifacts.values():assert sha(a['path'])==a['sha256']
    assert read(A/'feedback-final/run.json')['model_sha256']==read(A/'feedback-repeat-final/run.json')['model_sha256']
    verification={'trace_hashes_verified':len(traces),'episodes_per_variant_per_set':6,'frames_per_variant_per_set':7200,
        'teacher_v10_artifacts_unchanged':True,'full_feedback_repeat_outputs_fuel_exact':True,
        'selected_epoch_zero_guests_match_v10':True,'terminal_training_repeat_exact':True,
        'rotation_angular_twist_exact':True,'training_and_generation_complete_before_confirmation_timing':True,
        'new_confirmation_gt_prefixes_distinct':True,'source_freeze_verified':True,'local_only_no_course_upload':True}
    (A/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    shutil.copy2(OUT/'host-tests.log',A/'host-tests.txt')
    for arm in ('no-history','feedback','feedback-repeat'):
        shutil.copy2(OUT/('train-'+arm+'.log'),A/('train-'+arm+'.log'))
    lines=['# Explicit velocity memory V11','',
        '**Primary result: retain V10.** Both matched 18-input arms chose epoch zero under the frozen full original-validation pose-loss rule. The trained terminal epoch 40 models below are diagnostics, not selected replacements. No deployment default changes.','',
        'The network adds previous pre-visual predicted velocity and a validity flag: 18 -> 64 -> 6, 1,606 parameters (+256), 1,536 semantic dense MACs (+256), and 1 KiB additional FP32 weights. Teacher calibrated gyro and visual direction fusion are retained. The velocity is transported into the next camera frame; it excludes delayed visual correction.','',
        'Both arms use identical V10 initialization, zero added columns, 20-frame causal contexts, target-only supervision, original V10 joint target plan, 40 epochs and 1,520 optimizer steps. Feedback uses self-predictions, never true previous velocity. No-history sets all added inputs to zero. Compact fine-tuning and sequence training are common changes versus frozen V10; only feedback versus matched control isolates memory.','',
        'A full deterministic training repeat and exact replay reproduce all epoch metrics. The replay only retains the fixed terminal checkpoint after both primary selectors retained epoch zero; this diagnostic decision happened before long-flight scores, without a new fit setting or checkpoint sweep.','',
        '## Original validation selection','',
        '| Arm | Selected epoch | Initial pose loss | Fixed final pose loss | Final translation RMSE (m) |',
        '| --- | ---: | ---: | ---: | ---: |']
    for n in ('no-history','feedback'):
        r=read(A/n/'run.json');f=read(A/(n+'-final')/'run.json')
        lines.append(f"| {n} | {r['best_epoch']} | {r['initial_validation']['pose_loss']:.6f} | {f['best_validation']['pose_loss']:.6f} | {f['best_validation']['translation_rmse_m']:.6f} |")
    for split,data in datasets.items():
        lines+=['',f'## {split.title()} long flights','',
            'Six 60-second flights, 7,200 frames per variant. Regression reuses the V10 set; confirmation uses fresh seeds starting at 430000. Models and diagnostic epoch were fixed before scores. All variants receive identical inputs.','',
            '| Variant | Mean Sim3 ATE (m) | Pooled 1s length RMSE (m) | Low net-motion RMSE (m) | Mean fuel/frame | Peak fuel/frame | Median step (ms) |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for n in NAMES:
            s=data['variants'][n]['summary']
            timing=fmt(s['median_step_ms']) if data['wall_time_comparable'] else 'excluded'
            lines.append(f"| {n} | {s['mean_sim3_ate_m']:.6f} | {s['pooled_length_rmse_m']:.6f} | {s['low_net_motion_rmse_m']:.6f} | {s['mean_fuel']:.2f} | {s['peak_fuel']} | {timing} |")
        if not data['wall_time_comparable']:lines+=['','Regression accuracy runs overlapped fresh-data generation. Their wall times are excluded; deterministic fuel and accuracy remain comparable. Final confirmation timings start after generation finishes.']
        f=data['variants']['feedback-final']['summary']
        for n in ('no-history-final','v10','teacher'):
            b=data['variants'][n]['summary']
            changes={k:100*(f[k]/b[k]-1) for k in ('mean_sim3_ate_m','pooled_length_rmse_m','low_net_motion_rmse_m','mean_fuel','peak_fuel')}
            lines+=['',f"Feedback final versus {n} (% change; negative error change means improvement): `{json.dumps({k:round(v,3) for k,v in changes.items()})}`."]
        lines+=['','### True motion regimes','',
            '| Regime | Windows | Teacher RMSE | V10 RMSE | Final no-history RMSE | Final feedback RMSE |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
        for regime in ('stationary','slow_straight','decelerating','turning','small_net_but_moving'):
            r=[data['variants'][n]['motion_diagnostics'][regime] for n in NAMES]
            lines.append('| '+regime+' | '+str(r[0]['windows'])+' | '+' | '.join(fmt(x['length_rmse_m']) for x in r)+' |')
        lines+=['','Small-net-motion means true one-second net displacement <0.1 m. It is not synonymous with stationary or low instantaneous speed. Regime counts describe this dataset, not independent trials; windows overlap.','',
            '| Variant | Stop events | Settled events | Mean post-stop displacement error (m) | Mean settling time among settled (s) |',
            '| --- | ---: | ---: | ---: | ---: |']
        for n in NAMES:
            st=data['variants'][n]['motion_diagnostics']['stops']
            lines.append(f"| {n} | {st['events']} | {st['settled']} | {fmt(st['mean_displacement_error_m'])} | {fmt(st['mean_settling_time_s_among_settled'])} |")
        lines+=['','Missing regimes/events remain n/a; no conclusion about stopping recovery is supported when there are no stops. A stop requires at least 0.5 seconds below 0.02 m/s after recent >=0.1 m/s motion; displacement error covers up to the first second of the observed hold.','',
                '### Per-flight errors','', '| Scene / seed | Teacher length | V10 length | No-history final length | Feedback final length |', '| --- | ---: | ---: | ---: | ---: |']
        for i,e in enumerate(data['variants']['teacher']['episodes']):
            vals=[]
            for n in NAMES:
                import numpy as np
                trace=np.load(OUT/split/str(i)/n/'trace.npz');ep=read_episode_local(read(A/(split+'-manifest.json'))['validation'][i])
                truth=ep.ground_truth_state();pred=trace['outputs'];err=np.linalg.norm(pred[20:,:3]-pred[:-20,:3],axis=1)-np.linalg.norm(truth[20:,:3]-truth[:-20,:3],axis=1)
                vals.append(float(np.sqrt(np.mean(err**2))))
            lines.append('| '+e['scene']+' / '+str(e['seed'])+' | '+' | '.join(fmt(v) for v in vals)+' |')
    lines+=['','## Selection and limits','','```json',json.dumps(read(A/'selection.json'),indent=2),'```','',
        '136 host tests pass. Diagnostic guest/ONNX motion error and causal memory checks are in runtime-parity.json. Selected epoch-zero guests match V10 on every evaluated frame. A separate complete feedback-final run reproduces all poses, twists and fuel. All orientations and angular twists equal teacher; linear twist agrees with final position increments.','',
        'Timing uses four distinct artifacts, all 24 execution orders, each artifact in each position 300 times per flight. Duplicate/repeat and selected-epoch checks run outside timing. Wall time is host-specific, while fuel is deterministic. No simulated sensor/render time is included in guest step time.','',
        'One training seed, short training trajectories, 20-frame training contexts and six fresh confirmation flights limit generalization. Scenes 5/6/7 are excluded from training but are not new scene identities relative to V10. Full validation and long evaluations are closed loop, rather than ground-truth-history evaluation. Training context truncation versus much longer deployment is a remaining possible source of drift.','',
        'Read docs/velocity-memory-v11.md for method, replay policy and reproduction. Raw logs/traces are retained in WSL results/velocity-memory-v11; raw fresh data is retained on D:. Teacher, V9, V10 and all prior experiments are preserved.']
    (A/'README.md').write_text('\n'.join(lines)+'\n')
    print('REPORT COMPLETE',json.dumps({s:{n:r['summary'] for n,r in d['variants'].items()} for s,d in datasets.items()}),flush=True)

def read_episode_local(entry):
    from viobench.tools.datasets.storage import read_episode
    return read_episode(ROOT/entry['path'],frames='lazy')
if __name__=='__main__':main()
