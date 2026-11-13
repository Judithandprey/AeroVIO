"""Verify complete V9 runs and summarize distance, trajectory and compute."""
import json
import math
from pathlib import Path
import statistics

import numpy as np
from viobench.tools.datasets.sources import episodes
from viobench.tools.geometry import poses_to_matrices

from run_data_balance import ROOT, OUT, ARCHIVE, PROTOCOL, sha, write, source_hashes, distance_metrics


def load(name):
    folder=OUT/('public-'+name)
    paths=list(folder.glob('report-*/report.json'))
    assert len(paths)==1
    report=json.loads(paths[0].read_text())
    manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['report_sha256']==sha(paths[0])
    assert manifest['selection_sha256']==sha(ARCHIVE/'selection.json')
    return report,paths[0].parent


def verify_same(first,second):
    a,da=first;b,db=second
    for ea,eb in zip(a['episodes'],b['episodes'],strict=True):
        assert ea['hash']==eb['hash'] and ea['metrics']==eb['metrics']
        for file in ('estimate.tum','twist.txt','fuel.npy'):
            assert sha(da/'logs'/ea['hash']/file)==sha(db/'logs'/eb['hash']/file)
    return {'all_10_episodes_pose_twist_metrics_fuel_exact':True}


def pooled_stats(truths,estimates,times):
    lengths=[];directions=[];speeds=[]
    for g,e,t in zip(truths,estimates,times,strict=True):
        dg=g[20:,:3]-g[:-20,:3];de=e[20:,:3]-e[:-20,:3]
        lg=np.linalg.norm(dg,axis=1);le=np.linalg.norm(de,axis=1)
        rg=poses_to_matrices(g[:,:7])[:-20,:3,:3];re=poses_to_matrices(e[:,:7])[:-20,:3,:3]
        dg=np.einsum('nji,nj->ni',rg,dg);de=np.einsum('nji,nj->ni',re,de)
        angle=np.degrees(np.arccos(np.clip(np.sum(dg*de,axis=1)/np.maximum(lg*le,1e-30),-1,1)))
        angle[(lg<.02)|(le<=1e-6)]=np.nan
        lengths.append(np.c_[lg,le]);directions.append(angle);speeds.append(lg/(t[20:]-t[:-20]))
    a=np.concatenate(lengths);angle=np.concatenate(directions);speed=np.concatenate(speeds)
    def stats(mask):
        valid=mask&(a[:,0]>=.02)&(a[:,1]>1e-6)
        return {'windows':int(mask.sum()),'length_rmse_m':float(np.sqrt(np.mean((a[mask,1]-a[mask,0])**2))),
                'length_bias_m':float(np.mean(a[mask,1]-a[mask,0])),
                'median_predicted_true_ratio':float(np.median(a[valid,1]/a[valid,0])),
                'median_local_direction_deg':float(np.nanmedian(angle[mask])),
                'median_true_speed_m_s':float(np.median(speed[mask])),
                'median_predicted_speed_m_s':float(np.median(a[mask,1]))}
    return {'all':stats(np.ones(len(a),bool)), 'speed_bins':{
        n:stats((speed>=lo)&(speed<hi)) for n,lo,hi in
        [('below_0.1',0,.1),('0.1_to_0.25',.1,.25),('0.25_to_0.5',.25,.5),('above_0.5',.5,np.inf)]}}


def main():
    cfg=json.loads(PROTOCOL.read_text()); selection=json.loads((ARCHIVE/'selection.json').read_text())
    assert selection['source_sha256']==source_hashes()
    runs={name:load(name) for name in selection['public_order']}
    verification={
        'teacher_control':verify_same(runs['teacher'],runs['control']),
        'teacher_repeat':verify_same(runs['teacher'],runs['teacher-repeat']),
        'balanced_repeat':verify_same(runs['balanced'],runs['balanced-repeat'])}
    old=ROOT/'artifacts/teacher-0c4f778-results/native-160/report-AeroVIO-teacher-native-160-0c4f778'
    verification['preserved_teacher']=verify_same(runs['teacher'],(json.loads((old/'report.json').read_text()),old))
    refs=json.loads((ROOT/cfg['public_episodes']).read_text())
    dataset=list(episodes(refs,frames='lazy'))
    truths=[ep.ground_truth_state() for ep in dataset]; times=[ep.frame_times for ep in dataset]
    artifacts=json.loads((ARCHIVE/'artifacts.json').read_text())
    assert artifacts['teacher']['sha256']==artifacts['control']['sha256']==cfg['reference_wasm_sha256']
    verification['rebuilt_control_wasm_byte_identical_to_teacher']=True
    rows=[];all_estimates={};details={}
    for name,(report,folder) in runs.items():
        assert report['summary']['completed']==10 and report['summary']['frames_attempted']==12000
        assert report['limits']==json.loads((ROOT/cfg['limits']).read_text())
        assert report['wasmtime']==runs['teacher'][0]['wasmtime']
        estimates=[];trace_hashes={};per_episode=[]
        for ep,entry,g,t in zip(dataset,report['episodes'],truths,times,strict=True):
            assert ep.hash==entry['hash'] and entry['completed'] and not entry['failure'] and entry['metrics']['filled_frames']==0
            log=folder/'logs'/ep.hash
            pose=np.loadtxt(log/'estimate.tum');twist=np.loadtxt(log/'twist.txt')
            assert len(pose)==len(t) and np.max(np.abs(pose[:,0]-t))<1e-6
            output=np.c_[pose[:,1:],twist[:,1:]];estimates.append(output)
            trace_hashes[ep.hash]={f:sha(log/f) for f in ('estimate.tum','twist.txt','fuel.npy')}
            per_episode.append({'hash':ep.hash,**distance_metrics(g,output,t)})
        all_estimates[name]=estimates
        s=report['summary'];distance=pooled_stats(truths,estimates,times)
        details[name]={'pooled_1s':distance,'episodes_1s':per_episode}
        row={'name':name,'mean_sim3_ate_m':s['sim3_ate_rmse_mean'],
            'mean_unaligned_ate_m':statistics.mean(e['metrics']['ate_rmse'] for e in report['episodes']),
            'median_rotation_deg':math.degrees(s['rotation_rmse_median']),
            'length_rmse_1s_m':distance['all']['length_rmse_m'],
            'median_distance_ratio':distance['all']['median_predicted_true_ratio'],
            'median_local_direction_deg':distance['all']['median_local_direction_deg'],
            'mean_fuel':s['fuel_per_frame_mean'],'max_fuel':s['fuel_per_frame_max'],
            'median_step_ms':s['wall_per_frame_median_ms'],
            'wasm_bytes':artifacts[name.removesuffix('-repeat')]['bytes']}
        rows.append(row)
        write(ARCHIVE/'runs'/(name+'.json'),{**report,'trace_sha256':trace_hashes})
    for teacher,balanced in zip(all_estimates['teacher'],all_estimates['balanced'],strict=True):
        assert np.array_equal(teacher[:,3:7],balanced[:,3:7])
        assert np.array_equal(teacher[:,10:13],balanced[:,10:13])
    verification['all_balanced_rotations_and_angular_twists_exact']=True
    write(ARCHIVE/'verification.json',verification);write(ARCHIVE/'comparison.json',rows)
    write(ARCHIVE/'distance-diagnostics.json',details)
    baseline=next(r for r in rows if r['name']=='teacher');candidate=next(r for r in rows if r['name']=='balanced')
    changes={k:100*(candidate[k]/baseline[k]-1) for k in
             ('mean_sim3_ate_m','mean_unaligned_ate_m','length_rmse_1s_m','median_local_direction_deg','mean_fuel','max_fuel')}
    write(ARCHIVE/'changes-vs-teacher.json',changes)
    audit=json.loads((ARCHIVE/'sampling-audit.json').read_text())
    with np.load(ROOT/'data/motion-balance-v9/train-labels.npz') as z:
        training_speed=z['speed'];training_turn=z['turn_deg_s']
    with np.load(ROOT/'data/motion-balance-v9/balanced-plan.npz') as z:
        exposure=z['order'].ravel()
    exposure_stats={}
    for name,indices in [('original',np.arange(len(training_speed))),('balanced',exposure)]:
        speed=training_speed[indices];turn=training_turn[indices]
        exposure_stats[name]={'exposures':len(indices),'median_speed_m_s':float(np.median(speed)),
            'fraction_below_0_1_m_s':float(np.mean(speed<.1)),
            'fraction_above_0_5_m_s':float(np.mean(speed>.5)),
            'fraction_turn_at_least_15_deg_s':float(np.mean(turn>=15))}
    with np.load(ROOT/'data/motion-balance-v9/train-labels.npz') as z:
        labels=z['labels']
    exposure_stats['original_group_speed']={name:{
        'median_speed_m_s':float(np.median(training_speed[labels==i])),
        'fraction_below_0_1_m_s':float(np.mean(training_speed[labels==i]<.1))}
        for i,name in enumerate(cfg['groups']['names'])}
    write(ARCHIVE/'exposure-distribution.json',exposure_stats)
    training=json.loads((ARCHIVE/'training-verification.json').read_text())
    validation=json.loads((ARCHIVE/'validation.json').read_text())
    lines=['# Data balance V9: completed matched sampling contrast','',
        'Teacher reference: `0c4f778`. Branch: `experiment/native-fusion-v8`. Same inference code, network, normalization, losses and training budget. Only training example exposure changes. No server submission.','',
        '## Fixed treatment','',
        f"Original training counts: startup/transition {audit['splits']['train']['counts'][0]}, cruise {audit['splits']['train']['counts'][1]}, turn {audit['splits']['train']['counts'][2]}. Balanced exposure is 796/796/796 per epoch, still 2,388 examples and 38 optimizer steps. Forty epochs give 1,520 steps in both arms.",
        '',f"Measured low-speed (<0.1 m/s) exposure changes from {exposure_stats['original']['fraction_below_0_1_m_s']:.1%} to {exposure_stats['balanced']['fraction_below_0_1_m_s']:.1%}. Median exposed true speed changes from {exposure_stats['original']['median_speed_m_s']:.3f} to {exposure_stats['balanced']['median_speed_m_s']:.3f} m/s. See `exposure-distribution.json`.",
        '','First second is startup. Afterwards angular speed >=15 deg/s takes precedence as turn; speed <0.1 m/s or absolute tangential acceleration >0.2 m/s2 is startup/transition; the remainder is approximate cruise. Acceleration uses a fixed five-frame speed average. Cruise is not a claim of perfectly constant 3D velocity. These labels only choose training examples.',
        '','Training uses the original zero-image graph and folds it into the same 1,350-parameter state-only graph. Same seed 17, batch 64, Adam 0.001, cosine schedule, warping weight 2, CPU four threads. Normalization is computed from the unchanged original cache, not the resampled exposures. Validation retains its original distribution and normalized pose-loss checkpoint criterion.',
        '',f"Control chooses epoch {training['runs']['control']['best_epoch']} and reproduces the teacher ONNX exactly. Balanced chooses epoch {training['runs']['balanced']['best_epoch']}; a complete same-seed rerun reproduces its ONNX and validation metrics exactly. This repetition establishes determinism, not across-seed statistical significance.",
        '', '## Validation decision before public evaluation','',
        f"Selected reference: **{selection['selected_variant']}**. Decision flags: `{json.dumps(selection['checks'])}`.",
        '', '| Variant | Mean Sim3 ATE (m) | Mean 1s length RMSE (m) | Mean rotation RMSE (deg) |',
        '| --- | ---: | ---: | ---: |']
    for name in ('teacher','control','balanced'):
        s=validation['variants'][name]['summary']
        lines.append(f"| {name} | {s['mean_sim3_ate_m']:.6f} | {s['mean_length_rmse_1s_m']:.6f} | {s['mean_rotation_deg']:.4f} |")
    lines+=['','## Full public comparison: 10 flights x 60 seconds','',
        '| Variant | Sim3 ATE (m) | 1s length RMSE (m) | Predicted/true distance median | Rotation RMSE median (deg) | Max fuel/frame | Median step (ms) |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['name']} | {r['mean_sim3_ate_m']:.6f} | {r['length_rmse_1s_m']:.6f} | {r['median_distance_ratio']:.4f} | {r['median_rotation_deg']:.4f} | {r['max_fuel']:,} | {r['median_step_ms']:.4f} |")
    lines+=['',f"Balanced changes 1s distance-length RMSE by {changes['length_rmse_1s_m']:+.2f}%, aligned ATE by {changes['mean_sim3_ate_m']:+.2f}%, mean unaligned ATE by {changes['mean_unaligned_ate_m']:+.2f}%, mean fuel by {changes['mean_fuel']:+.2f}%, and peak fuel by {changes['max_fuel']:+.2f}%. Negative error changes are improvements.",
        '', '## Speed-conditioned distance','',
        '| True net speed (m/s) | Teacher length RMSE (m) | Balanced length RMSE (m) | Teacher distance ratio | Balanced distance ratio |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name,original in details['teacher']['pooled_1s']['speed_bins'].items():
        new=details['balanced']['pooled_1s']['speed_bins'][name]
        lines.append(f"| {name} | {original['length_rmse_m']:.6f} | {new['length_rmse_m']:.6f} | {original['median_predicted_true_ratio']:.4f} | {new['median_predicted_true_ratio']:.4f} |")
    lines+=['','## Interpretation of this contrast','',
        'The balanced arm does not establish an improvement in displacement magnitude. Its public one-second length RMSE rises while mean aligned trajectory ATE falls slightly. The validation distance gain is below the predeclared 2% threshold and aligned ATE slightly regresses, so teacher remains the selected reference.',
        '', 'Motion-category balance did not balance speed: many turn examples are themselves slow. Low-speed exposure rose rather than fell. Therefore this experiment does not test a successful removal of the low-speed bias, and does not refute the broader data-distribution hypothesis. A future speed-by-turn stratification would be a new experiment, with fixed thresholds and fresh independent long-flight validation before another public comparison.',
        '', '## Verification and interpretation','',
        '- All five runs complete 10/10 flights, 12,000 frames each, without filled frames. Two new WASMs pass 80-frame checks. The original teacher and the rebuilt control have identical poses, twists, accuracy and per-frame fuel. The original archived teacher run also matches.',
        '- Both complete public repeats reproduce accuracy and per-frame fuel exactly. The balanced model preserves every teacher orientation and angular twist; learned rotation residual remains disabled. Geometry and tracking behavior may still react to the changed translation predictions.',
        '- Distance RMSE measures error in one-second net displacement lengths, without alignment. Direction and length-ratio summaries exclude true displacements below 2 cm and degenerate predicted directions. Windows overlap; they are descriptive, not 11,800 independent trials.',
        '- Sim3 scoring removes one global scale per episode. Distance and trajectory scores can move in different directions; a constant scale change cannot improve Sim3 ATE.',
        '- Timing was measured with all training stopped and public evaluations running serially. Wall time remains host-dependent; compare repeated-run ranges and deterministic fuel instead of claiming a hardware-independent speedup.',
        '- Rebalancing repeated existing examples; it did not add scenes, longer trajectories, temporal memory or new motion information. The previously inspected one-scene public set is diagnostic, not fresh independent generalization evidence. No threshold, checkpoint rule or default was changed after seeing public results.',
        '', 'See `sampling-audit.json`, `training-verification.json`, `validation.json`, `selection.json`, `distance-diagnostics.json`, `runs/`, and `verification.json` for exact values. Raw traces and build sources remain in `results/data-balance-v9/` in WSL.']
    if (ARCHIVE/'paired-timing.json').exists():
        paired=json.loads((ARCHIVE/'paired-timing.json').read_text())
        lines+=['','## Paired timing follow-up','',
            'The byte-identical teacher/control binary had substantially different serial wall times. To avoid interpreting host variability as a model speedup, a timing-only follow-up alternated which fixed artifact stepped first on each identical frame. All 12,000 pairs preserve prior outputs and per-frame fuel.',
            '',f"Pooled median step: teacher {paired['median_step_ms']['teacher']:.4f} ms; balanced {paired['median_step_ms']['balanced']:.4f} ms. Median per-episode balanced/teacher timing ratio: {paired['median_episode_ratio']:.4f}. This is a warm-process host measurement; peak fuel remains the stronger reproducible compute comparison. See `paired-timing.json`."]
    (ARCHIVE/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'selection':selection['selected_variant'],'changes':changes,'comparison':rows},indent=2))


if __name__=='__main__':main()
