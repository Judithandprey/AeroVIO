"""Reproduce V8 native fusion ablation, freezing validation before public runs.

Run with the evaluation venv in WSL. All raw outputs go in a fresh results folder.
The teacher's WASM, weights, upstream runner and existing reports remain intact.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'configs/native-fusion-v8.json'
OUT=ROOT/'results/native-fusion-v8'
ARCHIVE=ROOT/'experiments/native-fusion-v8'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def source_hashes():
    paths=[PROTOCOL,Path(__file__),ROOT/'scripts/test_native_fusion.py',ROOT/'tests/native_fusion_geometry.cpp',
           *[p for p in (ROOT/'workspace/visual_native').iterdir() if p.is_file()]]
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}
def run(label,command,cwd=ROOT):
    print('START',label,flush=True); start=time.perf_counter()
    with (OUT/(label+'.log')).open('x') as log:
        result=subprocess.run(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT)
    record={'stage':label,'command':command,'cwd':str(cwd),'exit_code':result.returncode,
            'elapsed_seconds':time.perf_counter()-start}
    with (OUT/'pipeline.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
    print('END',label,result.returncode,flush=True)
    if result.returncode: raise RuntimeError(f'{label} failed; see {OUT/(label+".log")}')

def verify_environment(protocol):
    upstream=ROOT/'external/viobench'
    revision=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    assert revision==json.loads((ROOT/'upstream.lock.json').read_text())['commit']
    assert subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'],text=True)==''
    import viobench
    assert Path(viobench.__file__).resolve().is_relative_to(upstream)
    assert sha(ROOT/protocol['checkpoint'])==protocol['checkpoint_sha256']
    return {'upstream_commit':revision,'host':platform.platform(),'python':sys.version,
            'packages':{p:importlib.metadata.version(p) for p in ('wasmtime','numpy','opencv-python-headless')},
            'source_sha256':source_hashes(),'protocol_sha256':sha(PROTOCOL),
            'limits_sha256':sha(ROOT/protocol['limits']),
            'public_selection_sha256':sha(ROOT/protocol['public_episodes'])}

def build(protocol):
    artifacts={}
    teacher=ROOT/'artifacts/teacher-0c4f778-results/AeroVIO-teacher-native-160-0c4f778.wasm'
    assert sha(teacher)=='9d32a8164130a1d13fb65ed376e360925ce615db9f3e43a238686e4b57d156e8'
    artifacts['teacher']=teacher
    for v in protocol['variants']:
        name=v['name']; directory=ROOT/f'workspace/visual_native/build/v8-{name}'
        command=['cmake','-S',str(ROOT/'workspace/visual_native'),'-B',str(directory),
            '-DCMAKE_TOOLCHAIN_FILE='+str(ROOT/'workspace/build/toolchain/wasi-sdk/share/cmake/wasi-sdk-p1.cmake'),
            '-DPython3_EXECUTABLE='+sys.executable,
            '-DVIOBENCH_OPENCV_DIR='+str(ROOT/'workspace/build/runtime/opencv-build'),
            '-DVISUAL_CONFIDENCE_WEIGHTING='+('ON' if v['confidence'] else 'OFF'),
            '-DVISUAL_ROTATION_CORRECTION='+('ON' if v['rotation'] else 'OFF'),
            '-DBUILD_FUSION_TESTS='+('ON' if name=='control' else 'OFF')]
        run('configure-'+name,command)
        run('build-'+name,['cmake','--build',str(directory),'-j','4'])
        artifact=OUT/f'AeroVIO-native-v8-{name}.wasm'
        shutil.copy2(directory/'solution.wasm',artifact)
        artifacts[name]=artifact
        run('check-'+name,[sys.executable,'-m','viobench','check',str(artifact),'--frames','80',
                          '--logs',str(OUT/('check-'+name))])
    run('geometry-tests',[sys.executable,str(ROOT/'scripts/test_native_fusion.py'),
        str(ROOT/'workspace/visual_native/build/v8-control/fusion_tests.wasm')])
    # Existing tests exercise training exports; keep their PyTorch environment
    # separate from the lighter, pinned evaluation environment.
    run('host-tests',[str(ROOT/'.train-venv/bin/python'),'-m','unittest','discover','-s','tests','-v'])
    write(OUT/'artifacts.json',{k:{'path':str(v),'sha256':sha(v),'bytes':v.stat().st_size} for k,v in artifacts.items()})
    return artifacts

def select(rows,protocol):
    rules=protocol['selection']; base=rows['teacher']['summary']; checks={}; eligible=[]
    tol=rules['absolute_tolerance']
    for name,row in rows.items():
        s=row['summary']
        wins=sum(a['metrics']['sim3_ate_rmse']<b['metrics']['sim3_ate_rmse']-tol
                 for a,b in zip(row['episodes'],rows['teacher']['episodes'],strict=True))
        passed={
            'aligned_ate':s['mean_sim3_ate_m']<=base['mean_sim3_ate_m']*rules['mean_sim3_ate_ratio_max']+tol,
            'rotation':s['mean_rotation_deg']<=base['mean_rotation_deg']*rules['mean_rotation_rmse_ratio_max']+tol,
            'unaligned_ate':s['mean_unaligned_ate_m']<=base['mean_unaligned_ate_m']*rules['mean_unaligned_ate_ratio_max']+tol,
            'compute':s['max_fuel']<=base['max_fuel']*rules['max_fuel_ratio_max'],
            'episode_wins':wins>=rules['minimum_sim3_episode_wins'],
            'material_improvement':s['mean_sim3_ate_m']<=(1-rules['minimum_relative_improvement'])*base['mean_sim3_ate_m']+tol
                or s['mean_rotation_deg']<=(1-rules['minimum_relative_improvement'])*base['mean_rotation_deg']+tol}
        accepted=name not in ('teacher','control','combined-repeat') and all(passed.values())
        checks[name]={'conditions':passed,'sim3_episode_wins':wins,'eligible':accepted}
        if accepted: eligible.append((s['mean_sim3_ate_m'],s['mean_rotation_deg'],name))
    return {'selected_variant':min(eligible)[2] if eligible else 'teacher','checks':checks,
            'frozen_utc':datetime.now(timezone.utc).isoformat(),'public_data_used':False}

def validate(protocol,artifacts):
    import numpy as np
    from viobench.tools.datasets.storage import read_episode
    from viobench.tools.runner.engine import Engine
    from viobench.tools.runner.sandbox import Sandbox,Limits
    from viobench.tools.runner.metrics import evaluate_trajectory
    from viobench.tools.geometry import poses_to_matrices
    manifest=json.loads((ROOT/protocol['validation_manifest']).read_text())
    entries=manifest['validation']
    assert manifest['status']=='complete' and len(entries)==4
    assert {e['scene'] for e in entries}==set(protocol['validation_scenes'])
    order={**artifacts,'combined-repeat':artifacts['combined']}
    rows={name:{'episodes':[]} for name in order}; checks=[]
    engine=Engine(); limits=Limits.from_dict(json.loads((ROOT/protocol['limits']).read_text()))
    for i,entry in enumerate(entries):
        episode=read_episode(ROOT/entry['path'],frames='lazy')
        config,adapted=episode.adapt((320,240)); frames=list(adapted)
        times=episode.frame_times; windows=[episode.imu_window(j).copy() for j in range(len(times))]
        assert len(times)==200
        input_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode())
        for t,f,w in zip(times,frames,windows,strict=True):
            for array in (np.asarray(t),f,w): input_hash.update(np.ascontiguousarray(array).tobytes())
        runs={}
        for name,wasm in order.items():
            sandbox=Sandbox(wasm,engine,limits=limits,log_dir=OUT/'validation'/str(i)/name)
            sandbox.init(config)
            outputs=[]; fuels=[]; walls=[]
            for t,f,w in zip(times,frames,windows,strict=True):
                result=sandbox.step(float(t),f,w)
                outputs.append(np.r_[result.pose,result.twist]); fuels.append(result.fuel); walls.append(result.wall_seconds)
            outputs=np.asarray(outputs); fuels=np.asarray(fuels,dtype=np.int64)
            diag=None
            exports=sandbox.instance.exports(sandbox.store)
            if 'fusion_diagnostics' in exports:
                ptr=sandbox.alloc(5*8); exports['fusion_diagnostics'](sandbox.store,ptr)
                d=np.frombuffer(sandbox.read(ptr,5*8),dtype='<f8')
                diag=dict(zip(('direct_fits','rotation_fits','weight_sum','carry_fits','applied_rotation_norm_sum_rad'),d.tolist()))
            poses=poses_to_matrices(outputs[:,:7])
            velocity=np.einsum('nji,nj->ni',poses[1:,:3,:3],np.diff(outputs[:,:3],axis=0))/np.diff(times)[:,None]
            assert np.max(np.abs(velocity-outputs[1:,7:10]))<1e-8
            assert np.isfinite(outputs).all() and np.array_equal(outputs[0],np.r_[np.zeros(6),1.,np.zeros(6)])
            runs[name]={'outputs':outputs,'fuel':fuels,'wall':np.asarray(walls),'diagnostics':diag}
            np.savez_compressed(OUT/'validation'/str(i)/name/'trace.npz',times=times,outputs=outputs,fuel=fuels,wall=walls)
        assert np.array_equal(runs['teacher']['outputs'],runs['control']['outputs'])
        assert np.array_equal(runs['teacher']['outputs'][:,3:7],runs['confidence']['outputs'][:,3:7])
        for key in ('outputs','fuel'):
            assert np.array_equal(runs['combined'][key],runs['combined-repeat'][key])
        # First usable pair: capture frame 8, finish frame 12. No correction before it.
        assert all(np.array_equal(r['outputs'][:12],runs['teacher']['outputs'][:12]) for r in runs.values())
        checks.append({'episode':i,'control_pose_twist_exact':True,'confidence_orientation_exact':True,
                       'combined_repeat_outputs_fuel_exact':True,'causal_prefix_exact':True,'twist_consistent':True,
                       'input_sha256':input_hash.hexdigest(),'episode_json_sha256':sha(ROOT/entry['path']/'episode.json')})
        truth=episode.ground_truth_state() # only after all estimator executions
        for name,r in runs.items():
            metrics=evaluate_trajectory(truth,r['outputs'],times).as_dict()
            assert metrics['filled_frames']==0 and metrics['frames']==200
            rows[name]['episodes'].append({'scene':entry['scene'],'seed':entry['seed'],'metrics':metrics,
                'max_fuel':int(r['fuel'].max()),'median_step_ms':float(np.median(r['wall'])*1000),
                'diagnostics':r['diagnostics']})
        print('VALIDATED',i,{n:round(r['episodes'][-1]['metrics']['sim3_ate_rmse'],5) for n,r in rows.items()},flush=True)
    for name,row in rows.items():
        es=row['episodes']
        row['summary']={'mean_sim3_ate_m':statistics.mean(e['metrics']['sim3_ate_rmse'] for e in es),
            'mean_rotation_deg':math.degrees(statistics.mean(e['metrics']['rotation_rmse'] for e in es)),
            'mean_unaligned_ate_m':statistics.mean(e['metrics']['ate_rmse'] for e in es),
            'max_fuel':max(e['max_fuel'] for e in es)}
    validation={'scope':'4 original scene-disjoint 10-second validation flights, actual WASM',
        'manifest_sha256':sha(ROOT/protocol['validation_manifest']),'variants':rows,'checks':checks}
    write(ARCHIVE/'validation.json',validation)
    selection=select(rows,protocol)
    selection.update(source_sha256=source_hashes(),validation_sha256=sha(ARCHIVE/'validation.json'),protocol_sha256=sha(PROTOCOL))
    write(ARCHIVE/'selection.json',selection)
    print('FROZEN SELECTION',selection['selected_variant'],flush=True)

def public(protocol,artifacts):
    selection=json.loads((ARCHIVE/'selection.json').read_text())
    assert selection['source_sha256']==source_hashes()
    episodes=json.loads((ROOT/protocol['public_episodes']).read_text())
    for name,wasm in {**artifacts,'combined-repeat':artifacts['combined']}.items():
        directory=OUT/('public-'+name); directory.mkdir()
        command=[sys.executable,'-u','-m','viobench','evaluate',str(wasm),
                 *['conta:'+e['hash'] for e in episodes],'--limits',str(ROOT/protocol['limits'])]
        run('evaluate-public-'+name,command,directory)
        path=directory/('report-'+wasm.stem)/'report.json'
        report=json.loads(path.read_text())
        assert report['summary']['completed']==10 and report['summary']['frames_attempted']==12000
        assert all(e['completed'] and not e['failure'] and not e['metrics']['filled_frames'] for e in report['episodes'])
        write(directory/'manifest.json',{'wasm_sha256':sha(wasm),'report_sha256':sha(path),
            'selection_sha256':sha(ARCHIVE/'selection.json'),'protocol_sha256':sha(PROTOCOL),
            'finished_utc':datetime.now(timezone.utc).isoformat()})

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('validation','public'),required=True)
    args=p.parse_args(); protocol=json.loads(PROTOCOL.read_text())
    env=verify_environment(protocol)
    if args.phase=='validation':
        OUT.mkdir(); ARCHIVE.mkdir()
        write(ARCHIVE/'environment.json',env)
        artifacts=build(protocol)
        validate(protocol,artifacts)
        assert env['source_sha256']==source_hashes()
        shutil.copy2(OUT/'artifacts.json',ARCHIVE/'artifacts.json')
    else:
        artifacts={n:Path(r['path']) for n,r in json.loads((OUT/'artifacts.json').read_text()).items()}
        for n,r in json.loads((OUT/'artifacts.json').read_text()).items(): assert sha(artifacts[n])==r['sha256']
        public(protocol,artifacts)
    print('COMPLETE',args.phase,flush=True)

if __name__=='__main__': main()
