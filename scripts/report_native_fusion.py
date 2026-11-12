"""Verify V8 public results and publish evidence plus hash-checked test WASMs."""
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from run_native_fusion import ROOT,OUT,ARCHIVE,PROTOCOL,sha,write,source_hashes


def load(directory):
    paths=list(directory.glob('report-*/report.json'))
    assert len(paths)==1
    path=paths[0]; report=json.loads(path.read_text())
    manifest=json.loads((directory/'manifest.json').read_text())
    assert sha(path)==manifest['report_sha256']
    assert report['summary']['completed']==10 and report['summary']['frames_attempted']==12000
    assert all(e['completed'] and not e['failure'] and e['metrics']['filled_frames']==0 for e in report['episodes'])
    return report,path.parent,manifest


def same_outputs(a,b,fuel):
    ra,da,_=a; rb,db,_=b; checks=[]
    for ea,eb in zip(ra['episodes'],rb['episodes'],strict=True):
        assert ea['hash']==eb['hash'] and ea['metrics']==eb['metrics']
        files=['estimate.tum','twist.txt']+(['fuel.npy'] if fuel else [])
        for f in files: assert sha(da/'logs'/ea['hash']/f)==sha(db/'logs'/eb['hash']/f)
        checks.append({'episode':ea['hash'],'pose_twist_metrics_exact':True,'fuel_exact':fuel})
    return checks


def main():
    import numpy as np
    from viobench.tools.geometry import poses_to_matrices
    protocol=json.loads(PROTOCOL.read_text())
    selection=json.loads((ARCHIVE/'selection.json').read_text())
    assert selection['source_sha256']==source_hashes()
    names=['teacher','control','confidence','rotation','combined','combined-repeat']
    runs={n:load(OUT/('public-'+n)) for n in names}
    artifacts=json.loads((OUT/'artifacts.json').read_text())
    build_records={}
    for variant in protocol['variants']:
        directory=ROOT/f"workspace/visual_native/build/v8-{variant['name']}"
        cache={}
        for line in (directory/'CMakeCache.txt').read_text().splitlines():
            if line and not line.startswith(('#','//')) and '=' in line:
                key,value=line.split('=',1); cache[key.split(':',1)[0]]=value
        assert cache['VISUAL_DOWNSAMPLE']=='2' and cache['VISUAL_SPARSE_DIRECTION']=='ON'
        assert cache['VISUAL_CONFIDENCE_WEIGHTING']==('ON' if variant['confidence'] else 'OFF')
        assert cache['VISUAL_ROTATION_CORRECTION']==('ON' if variant['rotation'] else 'OFF')
        build_records[variant['name']]={
            'settings':{k:v for k,v in cache.items() if k.startswith(('VISUAL_','CMAKE_CXX_COMPILER','CMAKE_TOOLCHAIN_FILE'))},
            'exported_weights_sha256':sha(directory/'weights.h')}
    assert len({r['exported_weights_sha256'] for r in build_records.values()})==1
    write(ARCHIVE/'build-configuration.json',build_records)
    selected=selection['selected_variant']
    plan=json.loads((ARCHIVE/'public-repeat-plan.json').read_text())
    assert plan['selected_variant']==selected
    if selected not in ('teacher','combined'):
        name=selected+'-repeat'; directory=OUT/('public-'+name)
        if not directory.exists():
            directory.mkdir()
            wasm=Path(artifacts[selected]['path'])
            episodes=json.loads((ROOT/protocol['public_episodes']).read_text())
            command=[sys.executable,'-u','-m','viobench','evaluate',str(wasm),
                     *['conta:'+e['hash'] for e in episodes],'--limits',str(ROOT/protocol['limits'])]
            print('START',name,flush=True)
            with (OUT/('evaluate-public-'+name+'.log')).open('x') as log:
                subprocess.run(command,cwd=directory,stdout=log,stderr=subprocess.STDOUT,check=True)
            path=directory/('report-'+wasm.stem)/'report.json'
            write(directory/'manifest.json',{'wasm_sha256':sha(wasm),'report_sha256':sha(path),
                'selection_sha256':sha(ARCHIVE/'selection.json'),'protocol_sha256':sha(PROTOCOL),
                'finished_utc':datetime.now(timezone.utc).isoformat()})
            print('END',name,flush=True)
        runs[name]=load(directory); names.append(name)
    hashes=[e['hash'] for e in json.loads((ROOT/protocol['public_episodes']).read_text())]
    rows=[]
    for name,(r,d,m) in runs.items():
        assert [e['hash'] for e in r['episodes']]==hashes
        assert r['limits']==json.loads((ROOT/protocol['limits']).read_text())
        assert r['wasmtime']==runs['teacher'][0]['wasmtime']
        assert m['selection_sha256']==sha(ARCHIVE/'selection.json') and m['protocol_sha256']==sha(PROTOCOL)
        original=name.removesuffix('-repeat'); artifact=artifacts[original]
        assert m['wasm_sha256']==artifact['sha256']
        s=r['summary']
        rows.append({'name':name,'mean_sim3_ate_m':s['sim3_ate_rmse_mean'],
            'median_rotation_deg':math.degrees(s['rotation_rmse_median']),
            'mean_rotation_deg':math.degrees(statistics.mean(e['metrics']['rotation_rmse'] for e in r['episodes'])),
            'median_unaligned_ate_m':s['ate_rmse_median'],'median_velocity_rmse_m_s':s['velocity_rmse_median'],
            'median_scale':statistics.median(e['metrics']['sim3_scale'] for e in r['episodes']),
            'max_fuel':s['fuel_per_frame_max'],'mean_fuel':s['fuel_per_frame_mean'],'p95_fuel':s['fuel_per_frame_p95'],
            'median_step_ms':s['wall_per_frame_median_ms'],'wasm_bytes':artifact['bytes'],
            'wasm_sha256':artifact['sha256'],'completed':s['completed']})
        # Compact upstream evidence, including per-episode metrics and exact trace hashes.
        copy={**r,'trace_sha256':{e['hash']:{f:sha(d/'logs'/e['hash']/f)
            for f in ('estimate.tum','twist.txt','fuel.npy')} for e in r['episodes']}}
        write(ARCHIVE/'runs'/(name+'.json'),copy)
    verification={'control_parity':same_outputs(runs['teacher'],runs['control'],False),
                  'combined_repeat':same_outputs(runs['combined'],runs['combined-repeat'],True)}
    if selected+'-repeat' in runs:
        verification['selected_repeat']=same_outputs(runs[selected],runs[selected+'-repeat'],True)
    prior=ROOT/'artifacts/teacher-0c4f778-results/native-160'
    verification['preserved_teacher_repeat']=same_outputs(load(prior),runs['teacher'],True)
    # Independent pose-derived angular velocity check on every validation trace.
    max_error=0.
    for path in (OUT/'validation').glob('*/*/trace.npz'):
        with np.load(path) as z:
            poses=poses_to_matrices(z['outputs'][:,:7]); dt=np.diff(z['times'])
            relative=np.einsum('nji,njk->nik',poses[:-1,:3,:3],poses[1:,:3,:3])
            skew=.5*np.stack((relative[:,2,1]-relative[:,1,2],relative[:,0,2]-relative[:,2,0],
                             relative[:,1,0]-relative[:,0,1]),axis=1)
            sine=np.linalg.norm(skew,axis=1); cosine=.5*(np.trace(relative,axis1=1,axis2=2)-1)
            angle=np.arctan2(sine,cosine)
            omega=skew*(np.where(sine<1e-5,0.,angle/np.maximum(sine,1e-30))/dt)[:,None]
            error=float(np.max(np.abs(omega-z['outputs'][1:,10:13])))
            max_error=max(max_error,error)
            assert error<1e-7,(path,error)
    verification['max_angular_twist_consistency_error_rad_s']=max_error
    verification['scope']='Local public set; no server upload or private evaluation.'
    write(ARCHIVE/'verification.json',verification); write(ARCHIVE/'comparison.json',rows)
    base=next(x for x in rows if x['name']=='teacher')
    changes={r['name']:{k:100*(r[k]/base[k]-1) for k in
        ('mean_sim3_ate_m','median_rotation_deg','mean_rotation_deg','max_fuel','mean_fuel','median_step_ms')}
        for r in rows if r['name'] not in ('teacher','control')}
    write(ARCHIVE/'changes-vs-teacher.json',changes)
    validation=json.loads((ARCHIVE/'validation.json').read_text())
    lines=['# Native fusion V8: completed fixed ablation','',
        'Teacher reference: `0c4f778`. Branch: `experiment/native-fusion-v8`. No retraining, map, online bias estimator, or server upload.',
        '',f'Validation selected **{selected}** before public evaluation. Source defaults and previous artifacts remain unchanged.',
        '', '## Validation: 4 scene-disjoint 10-second flights', '',
        '| Variant | Mean Sim(3) ATE (m) | Mean rotation RMSE (deg) | Max fuel/frame |',
        '| --- | ---: | ---: | ---: |']
    for n in ('teacher','confidence','rotation','combined'):
        s=validation['variants'][n]['summary']
        lines.append(f"| {n} | {s['mean_sim3_ate_m']:.6f} | {s['mean_rotation_deg']:.4f} | {s['max_fuel']:,} |")
    lines+=['','## Public: 10 fixed 60-second episodes','',
        '| Variant | Mean Sim(3) ATE (m) | Median rotation RMSE (deg) | Max fuel/frame | Median step (ms) |',
        '| --- | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['name']} | {r['mean_sim3_ate_m']:.6f} | {r['median_rotation_deg']:.4f} | {r['max_fuel']:,} | {r['median_step_ms']:.3f} |")
    selected_change=changes.get(selected,{k:0. for k in ('mean_sim3_ate_m','median_rotation_deg','max_fuel','mean_fuel')})
    lines+=['','## Interpretation and boundaries','',
        f"The validation-selected {selected} changes public mean aligned ATE by {selected_change['mean_sim3_ate_m']:+.3f}%, "
        f"median rotation error by {selected_change['median_rotation_deg']:+.3f}%, peak fuel by {selected_change['max_fuel']:+.3f}%, "
        f"and mean fuel by {selected_change['mean_fuel']:+.3f}% "
        'relative to the preserved teacher artifact. Negative error changes indicate improvement.', '',
        f"Confidence weighting alone changes public mean aligned ATE by {changes['confidence']['mean_sim3_ate_m']:+.3f}% "
        'while leaving orientation identical. It was rejected on validation before these scores were observed.', '',
        'All runs finish completely with zero filled frames. The frozen selection is not recomputed from public scores. '
        'The confidence weighting is an uncalibrated heuristic, and its failure does not establish that uncertainty-aware fusion is generally ineffective.',
        '', 'The rotation correction jointly refines translation direction and a bounded rotation residual, but is gated on '
        'a successful translation-direction fit. It does not independently solve pure-rotation or weak-parallax periods. '
        'The learned translation magnitudes, original gyro integration and startup calibration remain frozen.',
        '', 'Capture interval / fitting delay / correction spread remain 8 / 4 / 8 frames (0.4 / 0.2 / 0.4 seconds at 20 Hz). '
        'Step time is local execution time and can vary with host load. Max fuel is deterministic computation, not milliseconds.',
        '', 'Sim(3) alignment removes global scale. The following diagnostics must accompany any mapping or physical-robot claims:', '',
        '| Variant | Median unaligned ATE (m) | Median velocity RMSE (m/s) | Median Sim(3) scale |',
        '| --- | ---: | ---: | ---: |']
    for r in rows[:5]:
        lines.append(f"| {r['name']} | {r['median_unaligned_ate_m']:.4f} | {r['median_velocity_rmse_m_s']:.4f} | {r['median_scale']:.4f} |")
    lines+=['','## Verification','',
        '- 119 existing host tests and 11 new C++ geometry checks pass; the geometry checks execute inside Wasmtime.',
        '- All four new modules pass 80-frame ABI checks.',
        '- Control rebuild has exactly the teacher poses, twists and metrics on validation and public data.',
        '- Teacher public rerun matches its prior trajectories, twists, metrics and per-frame fuel exactly.',
        '- Combined and validation-selected public repeats reproduce poses, twists, metrics and every frame fuel exactly.',
        '- Validation confirms unchanged confidence-arm orientation, no changes before the first causal correction, and pose/twist consistency.',
        '', 'An initial preflight invoked training tests in the evaluation environment and stopped on missing PyTorch/ONNX. '
        'The runner was corrected to use the existing training environment before validation; no dependency was installed or test skipped. '
        'That failed preflight is retained under `results/native-fusion-v8-preflight-env/`.',
        '', 'See `../../docs/native-fusion-v8.md`, `selection.json`, `validation.json`, `comparison.json`, '
        '`verification.json`, `runs/`, and `windows-artifacts.json`. Raw traces and logs remain in ignored `results/native-fusion-v8/`.']
    (ARCHIVE/'README.md').write_text('\n'.join(lines)+'\n')
    for n in ('geometry-tests','host-tests'):
        shutil.copy2(OUT/(n+'.log'),ARCHIVE/(n+'.txt'))
    shutil.copy2(OUT/'pipeline.jsonl',ARCHIVE/'pipeline.jsonl')
    windows=Path('/mnt/c/Users/ROG/Documents/GitHub/AeroVIO/artifacts')
    records=[]
    for name in ('confidence','rotation','combined'):
        source=Path(artifacts[name]['path']); target=windows/source.name
        if target.exists(): assert sha(target)==sha(source)
        else: shutil.copy2(source,target)
        assert sha(target)==sha(source)
        records.append({'variant':name,'selected_on_validation':name==selected,
            'path':'C:/Users/ROG/Documents/GitHub/AeroVIO/artifacts/'+target.name,
            'sha256':sha(target),'bytes':target.stat().st_size})
    write(ARCHIVE/'windows-artifacts.json',records)
    print('\n'.join(lines[:30]),flush=True)

if __name__=='__main__': main()
