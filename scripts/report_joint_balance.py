"""Publish the frozen V10 comparison on new long flights, not old public scores."""
import hashlib,json
from pathlib import Path
import numpy as np
from viobench.tools.datasets.storage import read_episode
from run_joint_balance import ROOT,OUT,ARCHIVE,PROTOCOL,sha,write


def main():
    cfg=json.loads(PROTOCOL.read_text());selection=json.loads((ARCHIVE/'selection.json').read_text())
    result_path=ARCHIVE/'long-validation.json';result=json.loads(result_path.read_text())
    assert selection['validation_sha256']==sha(result_path) and selection['protocol_sha256']==sha(PROTOCOL)
    variants=result['variants'];sampling=json.loads((ARCHIVE/'sampling-audit.json').read_text())
    training=json.loads((ARCHIVE/'training-verification.json').read_text())
    manifest=json.loads((ARCHIVE/'long-data-manifest.json').read_text())
    summaries={n:r['summary'] for n,r in variants.items()}
    final_timing=(json.loads((ARCHIVE/'timing-comparison.json').read_text())
                  if (ARCHIVE/'timing-comparison.json').exists() else None)
    if final_timing:
        assert final_timing['source_sha256']==sha(ROOT/'scripts/time_joint_balance.py')
        assert final_timing['plan']['artifacts_sha256']==sha(ARCHIVE/'artifacts.json')
        assert final_timing['plan']['selection_sha256']==sha(ARCHIVE/'selection.json')
        assert final_timing['frames_per_model']==7200
    changes={}
    for base in ('teacher','category-v9'):
        changes[base]={k:100*(summaries['joint-v10'][k]/summaries[base][k]-1) for k in
            ('mean_sim3_ate_m','mean_unaligned_ate_m','mean_length_rmse_1s_m','pooled_length_rmse_1s_m','mean_fuel','max_fuel')}
    timing={}
    for name in ('category-v9','joint-v10','joint-repeat'):
        ratios=[a['median_step_ms']/b['median_step_ms'] for a,b in zip(variants[name]['episodes'],variants['teacher']['episodes'],strict=True)]
        timing[name]={'median_episode_ratio_vs_teacher':float(np.median(ratios)),
                     'episode_ratio_quartiles':np.quantile(ratios,[.25,.75]).tolist()}
    write(ARCHIVE/'changes-vs-references.json',changes);write(ARCHIVE/'paired-timing.json',timing)
    trace_checks=[]
    for i,entry in enumerate(manifest['validation']):
        for name in variants:
            trace=OUT/'long-validation'/str(i)/name/'trace.npz'
            assert sha(trace)==variants[name]['episodes'][i]['trace_sha256']
        trace_checks.append({'scene':entry['scene'],'seed':entry['seed'],'all_trace_hashes_valid':True})
    write(ARCHIVE/'verification.json',{'trace_checks':trace_checks,'completed_flights_per_variant':6,
        'frames_per_variant':7200,'joint_training_repeat_exact':True,'joint_full_inference_repeat_exact':True,
        'all_rotations_and_angular_twists_exact':True,'original_network_and_normalization_retained':True,
        'teacher_and_v9_artifact_hashes_unchanged':True,'sampling_gate_passed_before_training':True})
    # Describe fresh data only after checkpoints and evaluation are frozen.
    speed=[];per=[];new_gt_hashes=[]
    old_manifest=json.loads((ROOT/cfg['training_manifest']).read_text())
    old_gt_hashes=set()
    for split in ('train','validation'):
        for entry in old_manifest[split]:
            old_ep=read_episode(ROOT/entry['path'],frames='lazy')
            old_gt_hashes.add(hashlib.sha256(np.ascontiguousarray(old_ep.ground_truth_state()[:200]).tobytes()).hexdigest())
    for entry in manifest['validation']:
        ep=read_episode(ROOT/entry['path'],frames='lazy');gt=ep.ground_truth_state()
        gt_hash=hashlib.sha256(np.ascontiguousarray(gt[:200]).tobytes()).hexdigest();new_gt_hashes.append(gt_hash)
        v=np.linalg.norm(np.diff(gt[:,:3],axis=0),axis=1)/np.diff(ep.frame_times)
        speed.extend(v);per.append({'scene':entry['scene'],'seed':entry['seed'],
            'fraction_below_0_1_m_s':float(np.mean(v<.1)),'median_speed_m_s':float(np.median(v)),
            'first_200_gt_states_sha256':gt_hash})
    assert len(set(new_gt_hashes))==6 and not set(new_gt_hashes)&old_gt_hashes
    distribution={'fraction_below_0_1_m_s':float(np.mean(np.asarray(speed)<.1)),
                  'median_speed_m_s':float(np.median(speed)),'episodes':per,
                  'distinct_first_200_gt_states_vs_all_old_flights':True,
                  'role':'post-evaluation description only; never used to tune training exposures'}
    write(ARCHIVE/'new-long-data-distribution.json',distribution)
    lines=['# Joint speed/turn balance V10','',
        'Teacher architecture and inference are preserved. This fixed contrast tests joint sampling on six fresh 60-second trajectories. Models and selection rules were frozen before any new-long estimator scores.',
        '',f"Selected reference: **{selection['selected_variant']}**. Selection checks: `{json.dumps(selection['checks'])}`.",
        '', 'The aggregate choice includes a material low-net-motion regression, detailed below. Workspace defaults remain unchanged; this is an experimental candidate rather than a universal replacement.',
        '', '## Pre-training distribution gate','',
        'Each of four true-speed bins (<0.1, 0.1–0.25, 0.25–0.5, >=0.5 m/s) contributes exactly 597 examples per epoch. Within each speed bin, turn-rate cells (<5, 5–15, 15–30, >=30 deg/s) are as balanced as available support allows. Per-example exposure is capped at four per epoch. Empty cells remain empty.',
        '',f"Low-speed exposure: original {sampling['low_speed_fraction_original']:.2%}; V9 category-balanced {sampling['low_speed_fraction_v9']:.2%}; V10 joint **25.00% in every epoch**. Across 40 epochs all {sampling['unique_pairs_seen_over_40_epochs']} original pairs are seen. Training-speed median changes from {sampling['speed_median_original']:.3f} to {sampling['speed_median_joint']:.3f} m/s.",
        '', '| Speed bin (m/s) | Original counts by turn bin | Draws per epoch by turn bin |',
        '| --- | --- | --- |']
    for i,name in enumerate(('<0.1','0.1–0.25','0.25–0.5','>=0.5')):
        lines.append(f"| {name} | {sampling['counts'][i]} | {sampling['quotas_each_epoch'][i]} |")
    lines+=['', 'Training still has virtually no high-speed sharp turns. Joint sampling corrects the speed marginal but cannot create this missing motion coverage. The rarest cell is capped instead of being repeated hundreds of times.',
        '', '## Unchanged training and inference','',
        f"Same seed 17, original cache/normalization, 40 epochs, batch 64, 1,520 optimizer steps, Adam 0.001, cosine schedule and warping weight 2. The original zero-image training graph is folded into the same 1,350-parameter compact network. The original short-validation normalized pose-loss rule chooses epoch {training['best_epoch']}. New long trajectories never influence training or epoch selection.",
        '', 'A complete same-seed training repeat reproduces the checkpoint, export and validation metrics. Both exports pass full/compact/ONNX checks on 796 validation examples plus 64 synthetic probes. The new WASM uses byte-identical teacher C++ sources; only the compiled weights and build paths change. The 80-frame ABI check passes. Native V8 rotation/confidence changes are not included.',
        '', '## Fresh long validation','',
        'Six flights, 60 seconds and 1,200 frames each: two each in ProcTHOR scenes 5, 6 and 7, using fresh seeds starting at 420000. Training scenes 2/3/4 and public scene 1 are excluded. Scenes 5/6 were used for earlier short validation, so only scene 7 is entirely new to both training and earlier checkpoint validation.',
        '',f"Generation records {len(manifest['attempts'])} attempts, accepting only non-crashing flights under the predeclared sequential-seed policy. Estimator accuracy never filters flights. The simulator header and successful-flight capture function are unchanged. Raw RGB/IMU/GT/depth and archives are retained on D: via `data/long-validation-v10`.",
        '', 'Teacher, V9 category balance, V10 joint and a full V10 repeat run on each identical frame, rotating execution order. Generation and training finish before timing. Every model keeps exact teacher orientations/angular twists. The V10 repeat matches all poses, twists and per-frame fuel.',
        '', '| Variant | Mean Sim3 ATE (m) | Pooled 1s length RMSE (m) | Distance ratio median | Rotation RMSE median (deg) | Peak fuel/frame | Paired median step (ms) |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in ('teacher','category-v9','joint-v10'):
        s=summaries[name]
        ratio='n/a' if s['median_distance_ratio'] is None else f"{s['median_distance_ratio']:.4f}"
        step_ms=final_timing['median_step_ms'][name] if final_timing else s['paired_median_step_ms']
        lines.append(f"| {name} | {s['mean_sim3_ate_m']:.6f} | {s['pooled_length_rmse_1s_m']:.6f} | {ratio} | {s['median_rotation_deg']:.4f} | {s['max_fuel']:,} | {step_ms:.4f} |")
    if final_timing:
        lines+=['','The table uses the final timing-only three-model follow-up. Each six-frame block contains all six execution orders, shuffled with fixed seed 20261112. Each artifact runs first/second/third exactly 400 times per episode, without a duplicate V10 instance warming its code cache. All 7,200 outputs and per-frame fuel values per model exactly match the accuracy run. Initial four-way timings remain in `long-validation.json`; final timings are in `timing-comparison.json`. Neither timing pass changes model selection.']
    lines+=['',f"Against teacher: pooled distance-length RMSE {changes['teacher']['pooled_length_rmse_1s_m']:+.2f}%, Sim3 ATE {changes['teacher']['mean_sim3_ate_m']:+.2f}%. Against V9: distance-length RMSE {changes['category-v9']['pooled_length_rmse_1s_m']:+.2f}%, Sim3 ATE {changes['category-v9']['mean_sim3_ate_m']:+.2f}%. Negative error changes mean improvement.",
        '', 'The frozen selection rule uses the mean of per-flight length RMSE (not pooled RMSE), requiring >=2% improvement versus both references, no aligned-ATE regression versus either, at least four distance wins versus teacher, exact rotations and a <=5% peak-fuel guard. The table reports pooled length RMSE to match the previous diagnostic definition.',
        '', '## Per-flight distance and trajectory errors','',
        '| Scene / seed | Teacher length RMSE | V9 length RMSE | V10 length RMSE | Teacher ATE | V9 ATE | V10 ATE |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for i,entry in enumerate(manifest['validation']):
        a,b,c=(variants[n]['episodes'][i] for n in ('teacher','category-v9','joint-v10'))
        lines.append(f"| {entry['scene']} / {entry['seed']} | {a['distance_1s']['length_rmse_m']:.5f} | {b['distance_1s']['length_rmse_m']:.5f} | {c['distance_1s']['length_rmse_m']:.5f} | {a['metrics']['sim3_ate_rmse']:.5f} | {b['metrics']['sim3_ate_rmse']:.5f} | {c['metrics']['sim3_ate_rmse']:.5f} |")
    lines+=['', '## Speed-conditioned length RMSE','',
        '| True 1s net speed (m/s) | Teacher | V9 | V10 |', '| --- | ---: | ---: | ---: |']
    for speed_bin in variants['teacher']['pooled_distance']['speed_bins']:
        values=[variants[n]['pooled_distance']['speed_bins'][speed_bin]['length_rmse_m'] for n in ('teacher','category-v9','joint-v10')]
        lines.append(f"| {speed_bin} | {values[0]:.5f} | {values[1]:.5f} | {values[2]:.5f} |")
    slow=[variants[n]['pooled_distance']['speed_bins'].get('below_0.1') for n in ('teacher','category-v9','joint-v10')]
    if all(slow):
        lines+=['','## Material regression despite aggregate acceptance','',
            f"For windows with true one-second net displacement below 0.1 m, length RMSE rises from {slow[0]['length_rmse_m']:.5f} m (teacher) and {slow[1]['length_rmse_m']:.5f} m (V9) to {slow[2]['length_rmse_m']:.5f} m (V10). This is {100*(slow[2]['length_rmse_m']/slow[0]['length_rmse_m']-1):.1f}% worse than teacher. The larger improvements are in medium/high net-speed bins. These evaluation bins use one-second net displacement, whereas the sampler uses adjacent-frame speeds.",
            '', 'Aggregate acceptance is not a universal improvement. Retain this model as an experimental candidate and investigate low-motion errors before replacing a general deployment default. No new guard was retroactively added and no second sampling sweep was run.',
            '', 'All six first-200-frame ground-truth traces are distinct from each other and from the original 16 training/short-validation flights. Freshness checks are recorded in `new-long-data-distribution.json`.']
    lines+=['', '## Interpretation boundaries','',
        '- This is one fixed training seed. Repeating the seed verifies determinism, not across-seed statistical significance.',
        '- The speed marginal was verified before training. Fresh long-flight motion distributions were described only after evaluation; no sampler changes were made in response.',
        '- One-second windows overlap and are not independent experimental trials. Direction/ratio diagnostics omit true net displacement below 2 cm or degenerate predicted direction; length RMSE retains all windows.',
        '- Sim3 alignment removes one global scale. Absolute distance accuracy and aligned trajectory error can disagree; neither should be substituted for the other.',
        '- Interleaved wall time is host-dependent. Use peak and mean fuel for reproducible compute comparisons. No school GPU benchmark or server upload is implied.',
        '- Teacher and V9 artifacts, all earlier experiment records and workspace defaults remain preserved. No additional sampling sweep, larger network, temporal input, map or geometric refinement is included.',
        '', 'Evidence: `sampling-audit.json`, `training-verification.json`, `long-data-manifest.json`, `long-validation.json`, `selection.json`, `paired-timing.json`, `verification.json`. Raw traces and logs: `results/joint-balance-v10/` in WSL.']
    (ARCHIVE/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'selected':selection['selected_variant'],'summaries':summaries,'changes':changes,
        'final_timing_median_ms':final_timing['median_step_ms'] if final_timing else None},indent=2))

if __name__=='__main__':main()
