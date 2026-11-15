"""Verified gate experiment report; no model selection from long-flight scores."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/memory-gate-v12';A=ROOT/'experiments/memory-gate-v12'
NAMES=['v10','v11','constant','adaptive']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fmt(x):return 'n/a' if x is None else f'{x:.6f}'
def delta(x,y):return None if x is None or y in (None,0) else 100*(x/y-1)

def main():
    data={s:read(A/(s+'.json')) for s in ('regression','confirmation')}
    frozen=read(A/'frozen-before-long-evaluation.json');count=0
    for p,h in frozen['source_sha256'].items():assert sha(ROOT/p)==h,p
    assert sha(A/'artifacts.json')==frozen['artifacts_sha256']
    for a in read(A/'artifacts.json').values():assert sha(a['path'])==a['sha256']
    for split,d in data.items():
        assert len(d['checks'])==6
        for c in d['checks']:
            assert c['extra_checks']['adaptive']['maximum_output_difference']==0 and c['extra_checks']['adaptive']['fuel_exact']
        for n in NAMES:
            for i,e in enumerate(d['variants'][n]['episodes']):
                assert sha(OUT/split/str(i)/n/'trace.npz')==e['trace_sha256'];count+=1
    assert read(A/'adaptive/run.json')['model_sha256']==read(A/'adaptive-repeat/run.json')['model_sha256']
    for i in range(6):
        for name,old_name in (('v10','v10'),('v11','feedback-final')):
            old=np.load(ROOT/'results/velocity-memory-v11/confirmation'/str(i)/old_name/'trace.npz')
            current=np.load(OUT/'regression'/str(i)/name/'trace.npz')
            assert np.array_equal(old['outputs'],current['outputs']) and np.array_equal(old['fuel'],current['fuel'])
    endpoint=data['regression']['checks'][0]['extra_checks']
    assert endpoint['endpoint-v10']['maximum_output_difference']<1e-8 and endpoint['endpoint-v11']['maximum_output_difference']<1e-8
    selection=read(A/'selection.json')
    verification={'primary_traces_verified':count,'frames_per_variant_per_set':7200,'original_predictors_unchanged':True,
        'adaptive_training_repeat_exact':True,'all_adaptive_output_and_fuel_repeats_exact':True,
        'forced_gate_endpoints_match_original_outputs':True,'orientation_and_twist_invariants_passed':True,
        'source_freeze_verified':True,'local_only_no_server_upload':True,'confirmation_timing_after_other_work_complete':True}
    verification['v10_v11_outputs_and_fuel_match_preserved_v11_confirmation']=True
    (A/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    lines=['# Frozen-predictor memory gate V12','',
        '**Reference retained: V10. No main merge or deployment-default change.** This experiment adds only a learned scalar gate between frozen V10 and V11 predictors. The comparison uses independently selected updated gate candidates, not a claim that either is a validated replacement.','',
        'The adaptive 24 -> 8 -> 1 sigmoid gate has 209 trainable parameters. Its inputs are normalized IMU/intrinsics/dt/history plus both frozen candidate velocities. It blends pre-visual translation and feeds back the resulting velocity in the correct camera axes. Both predictor weights, calibrated gyro, visual tracking/fusion and memory validity/clip/reset rules remain unchanged. Total network parameters are 3,165, with 3,016 semantic dense MACs. No zero-velocity update, stationary classifier, new visual input or new training data is added.','',
        'The constant-mixture control trains only one sigmoid bias. Both arms start at g=0.5 and use the same target sampling plan, 20-frame causal contexts, self-predicted history, loss, 40 epochs and 1,520 updates. Only gate parameters enter the optimizer. The raw rotation residual used by the training warp stays V10 for all mixtures; deployed orientation always uses the unchanged gyro.','',
        '## Original validation and gate behavior','',
        '| Gate | Selected updated epoch | Initial pose loss | Best updated pose loss | Mean gate | Gate standard deviation |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for arm in ('constant','adaptive'):
        r=read(A/arm/'run.json');v=r['best_validation']
        lines.append(f"| {arm} | {r['best_epoch']} | {r['initial_validation']['pose_loss']:.6f} | {v['pose_loss']:.6f} | {v['gate_mean']:.6f} | {v['gate_std']:.6f} |")
    r=read(A/'adaptive/run.json')
    lines+=['',f"Hard V10 endpoint validation pose loss: {r['endpoint_validation_references']['0']['pose_loss']:.6f}. The trained adaptive candidate beats V10 on this selection metric: {selection['candidate_original_validation_beats_v10']}; beats its initial 50/50 mixture: {selection['candidate_original_validation_beats_initial_mixture']}.",'',
        'As predeclared before training, candidate checkpoints are selected among updated epochs 1..40. Initial and endpoint references are reported separately and V10 remains available. A best updated checkpoint is not necessarily better than initialization. Gate statistics above describe original validation, not independent evidence of turn/stop recognition. No epoch is selected using long-flight metrics.','']
    for split,d in data.items():
        lines+=[f'## {split.title()} set','',
            'Six 60-second flights, 7,200 frames per artifact. Regression uses the six earlier V11 flights; confirmation uses fresh seeds starting at 440000 in the same scene identities 5/6/7. All variants receive identical data and settings.','',
            '| Variant | Mean Sim3 ATE (m) | Pooled 1s length RMSE (m) | Turning length RMSE (m) | Cruise length RMSE (m) | Stop displacement error (m) |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
        for n in NAMES:
            v=d['variants'][n];s=v['summary'];m=v['motion_diagnostics']
            lines.append(f"| {n} | {fmt(s['mean_sim3_ate_m'])} | {fmt(s['pooled_length_rmse_m'])} | {fmt(m['turning']['length_rmse_m'])} | {fmt(v['cruise']['length_rmse_m'])} | {fmt(m['stops']['mean_displacement_error_m'])} |")
        v=d['variants']['adaptive'];m=v['motion_diagnostics']
        lines+=['',f"Coverage: {m['turning']['windows']} turning windows, {v['cruise']['windows']} cruise windows, {m['stationary']['windows']} stationary windows, {m['stops']['events']} stop events. Windows overlap and are not independent trials.",'',
            '| Variant | Stops settled | False-stop runs while moving | Cruise length bias (m) | Stationary length RMSE (m) |',
            '| --- | ---: | ---: | ---: | ---: |']
        for n in NAMES:
            v=d['variants'][n];m=v['motion_diagnostics']
            lines.append(f"| {n} | {m['stops']['settled']} / {m['stops']['events']} | {v['cruise']['false_stop_events']} | {fmt(v['cruise']['length_bias_m'])} | {fmt(m['stationary']['length_rmse_m'])} |")
        lines+=['','Cruise means one-second true-motion windows with mean speed >=0.1 m/s, CV <=0.1, speed range <=0.05 m/s, path efficiency >=0.95 and mean angular rate <15 deg/s. False-stop runs are tested on all true-moving intervals >=0.1 m/s, requiring five consecutive predicted increments below 0.02 m/s. These are evaluation-only truth labels.','',
            '| Variant | Mean fuel/frame | Peak fuel/frame | Median guest step (ms) |',
            '| --- | ---: | ---: | ---: |']
        for n in NAMES:
            s=d['variants'][n]['summary'];wall=fmt(s['median_step_ms']) if d['wall_time_comparable'] else 'excluded'
            lines.append(f"| {n} | {s['mean_fuel']:.2f} | {s['peak_fuel']} | {wall} |")
        if not d['wall_time_comparable']:lines+=['','Regression overlapped simulation; its wall times are excluded. Fuel and outputs remain deterministic.']
        adaptive=d['variants']['adaptive']
        for n in ('v10','v11','constant'):
            b=d['variants'][n];changes={k:delta(adaptive['summary'][k],b['summary'][k]) for k in ('mean_sim3_ate_m','pooled_length_rmse_m','mean_fuel')}
            changes.update({'turning_error':delta(adaptive['motion_diagnostics']['turning']['length_rmse_m'],b['motion_diagnostics']['turning']['length_rmse_m']),
                'cruise_error':delta(adaptive['cruise']['length_rmse_m'],b['cruise']['length_rmse_m']),
                'stop_error':delta(adaptive['motion_diagnostics']['stops']['mean_displacement_error_m'],b['motion_diagnostics']['stops']['mean_displacement_error_m'])})
            lines+=['',f"Adaptive versus {n}, percentage change (negative error change is better): `{json.dumps({k:None if v is None else round(v,3) for k,v in changes.items()})}`."]
        lines+=['','### Individual stops','','| Seed / start frame | V10 | V11 | Constant | Adaptive |','| --- | ---: | ---: | ---: | ---: |']
        events=0
        for i,e in enumerate(d['variants']['v10']['episodes']):
            for j,stop in enumerate(e['motion_diagnostics']['stop_events']):
                values=[d['variants'][n]['episodes'][i]['motion_diagnostics']['stop_events'][j]['displacement_error_m'] for n in NAMES]
                lines.append('| '+str(e['seed'])+' / '+str(stop['start_frame'])+' | '+' | '.join(fmt(x) for x in values)+' |');events+=1
        if not events:lines+=['| No qualifying events | n/a | n/a | n/a | n/a |']
        lines+=['','Stop errors are net displacement errors over up to one second of the observed hold. Stops require >=0.5 seconds below 0.02 m/s after recent >=0.1 m/s movement. No settled event means no settling-time estimate; do not call a reduced drift error successful stopping.','']
    lines+=['## Frozen confirmation guards','','```json',json.dumps(selection,indent=2),'```','',
        'Fewer than five new stop events makes that guard inconclusive, even if the measured mean improves. Empty bins are not scored as zero. The earlier stopping audit found only one qualifying transition in the training set; this gate-only experiment deliberately keeps that data unchanged.','',
        '## Verification and limits','',
        '143 host tests pass, including frozen-parameter optimization, endpoint blending, history gradients, resets, and cruise/false-stop definitions. Five 80-frame WASM checks and independent C++/ONNX gate/input/memory parity pass. A complete same-seed training repeat matches weights and every epoch metric. Native hard-zero and hard-one endpoint guests reproduce V10 and V11 outputs on a full flight. Complete adaptive repeats match every output and per-frame fuel on all twelve long flights. Forty-eight primary trace hashes and all frozen sources/artifacts are verified.','',
        'Fresh wall timing begins only after generation, training, compilation and regression complete. Every model occupies every execution position 300 times per flight. Endpoint/repeat/diagnostic runs are excluded from timing. Reported milliseconds are host-specific guest-step times and exclude simulation/rendering. No course-server submission or real-robot test occurred.','',
        'One training seed, the original short training data, and these simulated flights limit generalization. Both mixture candidates have independently optimized validation-selected weights; differences versus the constant control are not a perfectly mean-matched gate ablation. Additional predictors raise neural MACs, while vision dominates much of the system cost. No independent stationary detector or new visual confidence feature was added.','',
        'See `docs/memory-gate-v12.md` for the fixed protocol, `runtime-parity.json` for numerical checks, and `artifacts.json` for exact WASM identities. Raw data is retained on D: and logs/traces in WSL `results/memory-gate-v12`. All prior models and branches remain preserved.']
    (A/'README.md').write_text('\n'.join(lines)+'\n')
    print('REPORT VERIFIED',count,'primary traces',flush=True)

if __name__=='__main__':main()
