"""Matched V9 sampling experiment using exact teacher inference sources.

Run validation first, then public. Each phase refuses to overwrite its evidence.
Training commands and the original cache remain separate from this evaluation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/data-balance-v9'
ARCHIVE = ROOT/'experiments/data-balance-v9'
PROTOCOL = ROOT/'configs/data-balance-v9.json'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def source_hashes():
    files = [PROTOCOL, Path(__file__), ROOT/'scripts/train_motion.py', ROOT/'scripts/prepare_motion_balance.py',
             ROOT/'training/motion_sampling.py', ROOT/'training/motion_model.py', ROOT/'tests/test_motion_sampling.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in files}


def run(label, command, cwd=ROOT):
    print('START',label,flush=True)
    start = time.perf_counter()
    with (OUT/(label+'.log')).open('x') as f:
        result = subprocess.run(command, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    with (OUT/'pipeline.jsonl').open('a') as f:
        f.write(json.dumps({'stage':label,'command':command,'cwd':str(cwd),
                            'exit_code':result.returncode,'elapsed_s':time.perf_counter()-start})+'\n')
    print('END',label,result.returncode,flush=True)
    if result.returncode: raise RuntimeError('failed '+label)


def distance_metrics(truth, outputs, times, horizon=20):
    """One-second displacement magnitudes and local directions; no Sim3 fit."""
    from viobench.tools.geometry import poses_to_matrices
    gt = truth[horizon:,:3]-truth[:-horizon,:3]
    est = outputs[horizon:,:3]-outputs[:-horizon,:3]
    a,b = np.linalg.norm(gt,axis=1),np.linalg.norm(est,axis=1)
    usable = (a>=.02)&(b>1e-6)
    rg = poses_to_matrices(truth[:,:7])[:-horizon,:3,:3]
    re = poses_to_matrices(outputs[:,:7])[:-horizon,:3,:3]
    local_g = np.einsum('nji,nj->ni',rg,gt)
    local_e = np.einsum('nji,nj->ni',re,est)
    cosine = np.sum(local_g*local_e,axis=1)/np.maximum(a*b,1e-30)
    angle = np.degrees(np.arccos(np.clip(cosine,-1,1)))
    ratio = b[usable]/a[usable]
    speed = a/(times[horizon:]-times[:-horizon])
    def summarize(mask):
        return {'windows':int(mask.sum()), 'length_rmse_m':float(np.sqrt(np.mean((b[mask]-a[mask])**2))),
                'length_bias_m':float(np.mean(b[mask]-a[mask])),
                'median_predicted_true_ratio':float(np.median(b[mask&usable]/a[mask&usable])) if np.any(mask&usable) else None,
                'median_direction_deg':float(np.median(angle[mask&usable])) if np.any(mask&usable) else None}
    row = summarize(np.ones(len(a),bool))
    row['speed_bins'] = {name:summarize((speed>=lo)&(speed<hi))
                        for name,lo,hi in [('slow',0,.1),('medium',.1,.25),('cruise',.25,.5),('fast',.5,np.inf)]
                        if np.any((speed>=lo)&(speed<hi))}
    return row


def verify_training(cfg):
    names = ('control','balanced','balanced-repeat')
    runs,metas = {},{}
    for name in names:
        folder = ROOT/f'models/balance-v9-{name}'
        compact = ROOT/f'models/balance-v9-{name}-compact'
        runs[name] = json.loads((folder/'run.json').read_text())
        metas[name] = json.loads((compact/'meta.json').read_text())
        assert sha(compact/'net.onnx') == metas[name]['model_sha256']
        assert json.loads((compact/'run.json').read_text())['parameters'] == 1350
        for field in ('run.json','meta.json','history.json'):
            destination = ARCHIVE/'training'/name/field
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(folder/field,destination)
        for field in ('run.json','meta.json','net.onnx'):
            destination = ARCHIVE/name/field
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(compact/field,destination)
    reference = json.loads((ROOT/'experiments/motion-ablation-v2/imu-only/run.json').read_text())
    for field in ('model_sha256','initial_weights_sha256','minibatch_order_sha256','best_epoch','best_validation'):
        assert runs['control'][field] == reference[field],field
    assert metas['control']['model_sha256'] == cfg['reference_model_sha256']
    for name in names:
        for key in ('seed','epochs','batch','learning_rate','warp_weight','threads','device','input_mode',
                    'initial_weights_sha256','train_cache_sha256','validation_cache_sha256','packages','selection'):
            assert runs[name][key] == runs['control'][key],(name,key)
        for key in ('state_mean','state_std','target_mean','target_std','input_mode','network_inputs','map'):
            assert metas[name][key] == metas['control'][key],(name,key)
        assert runs[name]['epochs']*math.ceil(runs[name]['train_pairs']/runs[name]['batch']) == 1520
    for field in ('model_sha256','best_epoch','best_validation','minibatch_order_sha256'):
        assert runs['balanced'][field] == runs['balanced-repeat'][field],field
    assert metas['balanced']['model_sha256'] == metas['balanced-repeat']['model_sha256']
    assert runs['control']['minibatch_order_sha256'] != runs['balanced']['minibatch_order_sha256']
    audit = json.loads((ROOT/'data/motion-balance-v9/sampling-audit.json').read_text())
    assert audit['protocol_sha256'] == sha(PROTOCOL)
    assert runs['balanced']['sampling']['sha256'] == audit['balanced_plan_sha256']
    write(ARCHIVE/'sampling-audit.json',audit)
    write(ARCHIVE/'training-verification.json',{'teacher_control_model_exact':True,'balanced_retraining_exact':True,
         'normalization_and_settings_exact':True,'optimizer_steps_each':1520,'epochs_each':40,
         'source_sha256':source_hashes(),'runs':{n:{k:runs[n][k] for k in
             ('best_epoch','best_validation','model_sha256','minibatch_order_sha256','elapsed_s')} for n in names}})


def build(cfg):
    teacher = ROOT/'artifacts/teacher-0c4f778-results/AeroVIO-teacher-native-160-0c4f778.wasm'
    assert sha(teacher) == cfg['reference_wasm_sha256']
    artifacts = {'teacher':teacher}
    paths = subprocess.check_output(['git','ls-tree','--name-only',cfg['reference_commit'],'workspace/visual_native/'],cwd=ROOT,text=True).splitlines()
    source_record = {}
    for name in ('control','balanced'):
        src = OUT/('source-'+name); src.mkdir()
        checkpoint = ROOT/f'models/balance-v9-{name}-compact'
        for path in paths:
            data = subprocess.check_output(['git','show',cfg['reference_commit']+':'+path],cwd=ROOT)
            destination = src/Path(path).name
            destination.write_bytes(data)
            if destination.suffix in ('.cpp','.h'):
                source_record[path] = hashlib.sha256(data).hexdigest()
        # Build plumbing only: choose weights, preserving every C++ source byte.
        exporter = src/'export_weights.py'
        text = exporter.read_text()
        old = 'CHECKPOINT = ROOT / "experiments/motion-ablation-v2/imu-only-compact"'
        assert text.count(old) == 1
        exporter.write_text(text.replace(old,'CHECKPOINT = Path('+repr(str(checkpoint))+')'))
        cmake = src/'CMakeLists.txt'; text=cmake.read_text()
        old = 'set(checkpoint "${CMAKE_CURRENT_SOURCE_DIR}/../../experiments/motion-ablation-v2/imu-only-compact")'
        assert text.count(old) == 1
        cmake.write_text(text.replace(old,'set(checkpoint "'+str(checkpoint)+'")'))
        directory = OUT/('build-'+name)
        run('configure-'+name,['cmake','-S',str(src),'-B',str(directory),
            '-DCMAKE_TOOLCHAIN_FILE='+str(ROOT/'workspace/build/toolchain/wasi-sdk/share/cmake/wasi-sdk-p1.cmake'),
            '-DPython3_EXECUTABLE='+sys.executable,'-DVIOBENCH_NATIVE_DIR='+str(ROOT/'workspace/runtime/native'),
            '-DVIOBENCH_OPENCV_DIR='+str(ROOT/'workspace/build/runtime/opencv-build'),
            '-DVISUAL_DOWNSAMPLE=2','-DVISUAL_SPARSE_DIRECTION=ON'])
        run('build-'+name,['cmake','--build',str(directory),'-j','4'])
        artifact = OUT/f'AeroVIO-native-v9-{name}.wasm'
        shutil.copy2(directory/'solution.wasm',artifact); artifacts[name]=artifact
        run('check-'+name,[sys.executable,'-m','viobench','check',str(artifact),'--frames','80','--logs',str(OUT/('check-'+name))])
    write(ARCHIVE/'inference-source.json',{'teacher_commit':cfg['reference_commit'],'cpp_sha256':source_record,
        'changes':'Only build paths to new checkpoints. All C++ files byte-identical to teacher.'})
    rows={n:{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for n,p in artifacts.items()}
    write(ARCHIVE/'artifacts.json',rows)
    return artifacts


def validate(cfg,artifacts):
    from viobench.tools.datasets.storage import read_episode
    from viobench.tools.runner.engine import Engine
    from viobench.tools.runner.sandbox import Sandbox,Limits
    from viobench.tools.runner.metrics import evaluate_trajectory
    from viobench.tools.geometry import poses_to_matrices
    manifest = json.loads((ROOT/cfg['manifest']).read_text())
    rows={name:{'episodes':[]} for name in artifacts}; checks=[]
    engine=Engine(); limits=Limits.from_dict(json.loads((ROOT/cfg['limits']).read_text()))
    for i,entry in enumerate(manifest['validation']):
        ep=read_episode(ROOT/entry['path'],frames='lazy'); config,adapted=ep.adapt((320,240))
        frames=list(adapted); times=ep.frame_times
        windows=[ep.imu_window(j).copy() for j in range(len(times))]
        runs={}
        for name,wasm in artifacts.items():
            folder=OUT/'validation'/str(i)/name
            sandbox=Sandbox(wasm,engine,limits=limits,log_dir=folder); sandbox.init(config)
            outputs=[];fuels=[];walls=[]
            for t,f,w in zip(times,frames,windows,strict=True):
                result=sandbox.step(float(t),f,w)
                outputs.append(np.r_[result.pose,result.twist]);fuels.append(result.fuel);walls.append(result.wall_seconds)
            outputs=np.asarray(outputs);fuels=np.asarray(fuels,dtype=np.int64)
            assert np.isfinite(outputs).all()
            poses=poses_to_matrices(outputs[:,:7])
            velocity=np.einsum('nji,nj->ni',poses[1:,:3,:3],np.diff(outputs[:,:3],axis=0))/np.diff(times)[:,None]
            assert np.max(np.abs(velocity-outputs[1:,7:10]))<1e-8
            runs[name]={'outputs':outputs,'fuel':fuels,'wall':np.asarray(walls)}
            np.savez_compressed(folder/'trace.npz',times=times,outputs=outputs,fuel=fuels,wall=walls)
        assert np.array_equal(runs['teacher']['outputs'],runs['control']['outputs'])
        assert np.array_equal(runs['teacher']['fuel'],runs['control']['fuel'])
        assert np.array_equal(runs['teacher']['outputs'][:,3:7],runs['balanced']['outputs'][:,3:7])
        checks.append({'episode':i,'teacher_control_outputs_fuel_exact':True,'balanced_orientation_exact':True,'twist_consistent':True})
        truth=ep.ground_truth_state()
        for name,r in runs.items():
            metrics=evaluate_trajectory(truth,r['outputs'],times).as_dict()
            assert metrics['filled_frames']==0 and metrics['frames']==200
            rows[name]['episodes'].append({'scene':entry['scene'],'seed':entry['seed'],'metrics':metrics,
                'distance_1s':distance_metrics(truth,r['outputs'],times),
                'max_fuel':int(r['fuel'].max()),'median_step_ms':float(np.median(r['wall'])*1000)})
        print('VALIDATED',i,flush=True)
    for name,row in rows.items():
        episodes=row['episodes'];row['summary']={
            'mean_sim3_ate_m':statistics.mean(e['metrics']['sim3_ate_rmse'] for e in episodes),
            'mean_length_rmse_1s_m':statistics.mean(e['distance_1s']['length_rmse_m'] for e in episodes),
            'mean_rotation_deg':math.degrees(statistics.mean(e['metrics']['rotation_rmse'] for e in episodes)),
            'max_fuel':max(e['max_fuel'] for e in episodes)}
    write(ARCHIVE/'validation.json',{'variants':rows,'checks':checks,'scope':'4 unchanged 10-second scene-disjoint flights'})
    b,c=rows['teacher']['summary'],rows['balanced']['summary'];s=cfg['selection'];tol=s['absolute_tolerance']
    passed={'distance_improvement':c['mean_length_rmse_1s_m']<=b['mean_length_rmse_1s_m']*(1-s['minimum_distance_rmse_improvement'])+tol,
            'aligned_ate_nonregression':c['mean_sim3_ate_m']<=b['mean_sim3_ate_m']*s['mean_sim3_ate_ratio_max']+tol,
            'rotation_unchanged':c['mean_rotation_deg']==b['mean_rotation_deg'],
            'fuel':c['max_fuel']<=b['max_fuel']*s['max_fuel_ratio_max']}
    write(ARCHIVE/'selection.json',{'selected_variant':'balanced' if all(passed.values()) else 'teacher',
        'checks':passed,'source_sha256':source_hashes(),'validation_sha256':sha(ARCHIVE/'validation.json'),
        'protocol_sha256':sha(PROTOCOL),'public_data_used':False,'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'public_order':['teacher','control','balanced','balanced-repeat','teacher-repeat']})
    print('SELECTION',passed,flush=True)


def public(cfg):
    selection=json.loads((ARCHIVE/'selection.json').read_text())
    assert selection['source_sha256']==source_hashes()
    artifacts=json.loads((ARCHIVE/'artifacts.json').read_text())
    references=json.loads((ROOT/cfg['public_episodes']).read_text())
    for name in selection['public_order']:
        artifact=artifacts[name.removesuffix('-repeat')];wasm=Path(artifact['path'])
        assert sha(wasm)==artifact['sha256']
        directory=OUT/('public-'+name);directory.mkdir()
        run('evaluate-'+name,[sys.executable,'-u','-m','viobench','evaluate',str(wasm),
            *['conta:'+e['hash'] for e in references],'--limits',str(ROOT/cfg['limits'])],directory)
        path=directory/('report-'+wasm.stem)/'report.json'
        report=json.loads(path.read_text())
        assert report['summary']['completed']==10 and report['summary']['frames_attempted']==12000
        assert all(e['completed'] and not e['failure'] and e['metrics']['filled_frames']==0 for e in report['episodes'])
        write(directory/'manifest.json',{'wasm_sha256':sha(wasm),'report_sha256':sha(path),
             'selection_sha256':sha(ARCHIVE/'selection.json'),'finished_utc':datetime.now(timezone.utc).isoformat()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',required=True,choices=['validation','public'])
    args=parser.parse_args();cfg=json.loads(PROTOCOL.read_text())
    upstream=ROOT/'external/viobench'
    revision=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    assert revision==json.loads((ROOT/'upstream.lock.json').read_text())['commit']
    assert not subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'],text=True)
    if args.phase=='validation':
        ARCHIVE.mkdir()
        write(ARCHIVE/'environment.json',{'upstream_commit':revision,'python':sys.version,
              'packages':{p:importlib.metadata.version(p) for p in ('wasmtime','numpy','opencv-python-headless')},
              'limits_sha256':sha(ROOT/cfg['limits']),'public_selection_sha256':sha(ROOT/cfg['public_episodes']),
              'source_sha256':source_hashes()})
        verify_training(cfg)
        artifacts=build(cfg)
        validate(cfg,artifacts)
    else: public(cfg)


if __name__=='__main__':main()
