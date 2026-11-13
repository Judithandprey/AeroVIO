"""Build the V10 checkpoint with exact teacher C++; evaluate only fresh long flights."""
import argparse
from datetime import datetime,timezone
import importlib.metadata,json,math,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
from run_data_balance import sha,write,distance_metrics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/joint-balance-v10'
ARCHIVE=ROOT/'experiments/joint-balance-v10'
PROTOCOL=ROOT/'configs/joint-balance-v10.json'


def run(label,command):
    print('START',label,flush=True);started=time.perf_counter()
    with (OUT/(label+'.log')).open('x') as log:
        result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    with (OUT/'pipeline.jsonl').open('a') as f:f.write(json.dumps({'stage':label,'command':command,
        'exit_code':result.returncode,'elapsed_s':time.perf_counter()-started})+'\n')
    if result.returncode:raise RuntimeError(label+' failed')
    print('END',label,flush=True)


def frozen_sources():
    record=json.loads((OUT/'frozen-before-training.json').read_text())
    for name,digest in record['source_sha256'].items():assert sha(ROOT/name)==digest,name
    return record


def pooled_distance(truths,outputs,times):
    from viobench.tools.geometry import poses_to_matrices
    lengths=[];angles=[];speeds=[]
    for gt,out,t in zip(truths,outputs,times,strict=True):
        dg=gt[20:,:3]-gt[:-20,:3];de=out[20:,:3]-out[:-20,:3]
        a=np.linalg.norm(dg,axis=1);b=np.linalg.norm(de,axis=1)
        rg=poses_to_matrices(gt[:,:7])[:-20,:3,:3];re=poses_to_matrices(out[:,:7])[:-20,:3,:3]
        dg=np.einsum('nji,nj->ni',rg,dg);de=np.einsum('nji,nj->ni',re,de)
        angle=np.degrees(np.arccos(np.clip(np.sum(dg*de,axis=1)/np.maximum(a*b,1e-30),-1,1)))
        lengths.append(np.c_[a,b]);angles.append(angle);speeds.append(a/(t[20:]-t[:-20]))
    ab=np.concatenate(lengths);angle=np.concatenate(angles);speed=np.concatenate(speeds)
    def stats(mask):
        valid=mask&(ab[:,0]>=.02)&(ab[:,1]>1e-6)
        return {'windows':int(mask.sum()),'length_rmse_m':float(np.sqrt(np.mean((ab[mask,1]-ab[mask,0])**2))),
            'length_bias_m':float(np.mean(ab[mask,1]-ab[mask,0])),
            'median_predicted_true_ratio':float(np.median(ab[valid,1]/ab[valid,0])) if np.any(valid) else None,
            'median_local_direction_deg':float(np.median(angle[valid])) if np.any(valid) else None}
    return {'all':stats(np.ones(len(speed),bool)), 'speed_bins':{
        name:stats((speed>=low)&(speed<high)) for name,low,high in
        [('below_0.1',0,.1),('0.1_to_0.25',.1,.25),('0.25_to_0.5',.25,.5),('above_0.5',.5,np.inf)]
        if np.any((speed>=low)&(speed<high))}}


def training_audit(cfg):
    frozen=frozen_sources();control=json.loads((ROOT/'experiments/data-balance-v9/training/control/run.json').read_text())
    base_meta=json.loads((ROOT/'experiments/motion-ablation-v2/imu-only-compact/meta.json').read_text())
    runs={};compact_runs={}
    for name in ('joint','joint-repeat'):
        directory=ROOT/f'models/joint-v10-{name}';compact=ROOT/f'models/joint-v10-{name}-compact'
        r=json.loads((directory/'run.json').read_text());cr=json.loads((compact/'run.json').read_text())
        meta=json.loads((compact/'meta.json').read_text());runs[name]=r;compact_runs[name]=cr
        for key in ('seed','epochs','batch','learning_rate','warp_weight','threads','device','input_mode',
                    'initial_weights_sha256','train_cache_sha256','validation_cache_sha256','packages','selection'):
            assert r[key]==control[key],(name,key)
        for key in ('state_mean','state_std','target_mean','target_std','network_inputs','input_mode','map'):
            assert meta[key]==base_meta[key],(name,key)
        assert cr['parameters']==1350 and r['parameters']==29398
        assert r['epochs']*math.ceil(r['train_pairs']/r['batch'])==1520
        assert r['sampling']['sha256']==sha(ROOT/'data/joint-sampling-v10/joint-plan.npz')
        assert r['source_sha256']['scripts/train_motion.py']==sha(ROOT/'scripts/train_motion.py')
        assert meta['model_sha256']==sha(compact/'net.onnx')
        for key in ('maximum_torch_difference','maximum_opencv_difference'):assert cr['equivalence'][key]<1e-5
        for filename in ('run.json','meta.json','history.json'):
            dest=ARCHIVE/'training'/name/filename;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(directory/filename,dest)
        for filename in ('run.json','meta.json','net.onnx'):
            dest=ARCHIVE/name/filename;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(compact/filename,dest)
    for key in ('model_sha256','best_epoch','best_validation','minibatch_order_sha256'):
        assert runs['joint'][key]==runs['joint-repeat'][key],key
    assert compact_runs['joint']['model_sha256']==compact_runs['joint-repeat']['model_sha256']
    assert runs['joint']['minibatch_order_sha256']!=control['minibatch_order_sha256']
    write(ARCHIVE/'training-verification.json',{'unchanged_settings_and_normalization':True,
        'repeat_model_metrics_minibatches_exact':True,'optimizer_steps_each':1520,'frozen':frozen,
        'compact_model_sha256':compact_runs['joint']['model_sha256'],
        'best_epoch':runs['joint']['best_epoch'],'best_validation':runs['joint']['best_validation']})
    shutil.copy2(ROOT/'data/joint-sampling-v10/sampling-audit.json',ARCHIVE/'sampling-audit.json')
    shutil.copy2(OUT/'host-tests.log',ARCHIVE/'host-tests.txt')


def build(cfg):
    ARCHIVE.mkdir();training_audit(cfg)
    for src in (PROTOCOL,ROOT/'configs/long-validation-v10.json'):shutil.copy2(src,ARCHIVE/src.name)
    upstream=ROOT/'external/viobench'
    revision=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()
    assert revision==json.loads((ROOT/'upstream.lock.json').read_text())['commit']
    assert not subprocess.check_output(['git','-C',str(upstream),'status','--porcelain'],text=True)
    write(ARCHIVE/'environment.json',{'upstream_commit':revision,'python':sys.version,
        'packages':{p:importlib.metadata.version(p) for p in ('wasmtime','numpy','opencv-python-headless')},
        'limits_sha256':sha(ROOT/cfg['limits'])})
    artifacts={'teacher':ROOT/cfg['teacher_wasm'],'category-v9':ROOT/cfg['v9_wasm']}
    assert sha(artifacts['teacher'])==cfg['teacher_wasm_sha256']
    assert sha(artifacts['category-v9'])==cfg['v9_wasm_sha256']
    src=OUT/'native-source';src.mkdir();checkpoint=ROOT/'models/joint-v10-joint-compact'
    source_hashes={}
    paths=subprocess.check_output(['git','ls-tree','--name-only',cfg['teacher_commit'],'workspace/visual_native/'],cwd=ROOT,text=True).splitlines()
    for name in paths:
        data=subprocess.check_output(['git','show',cfg['teacher_commit']+':'+name],cwd=ROOT)
        path=src/Path(name).name;path.write_bytes(data)
        if path.suffix in ('.cpp','.h'):source_hashes[name]=sha(path)
    path=src/'export_weights.py';text=path.read_text()
    old='CHECKPOINT = ROOT / "experiments/motion-ablation-v2/imu-only-compact"';assert text.count(old)==1
    path.write_text(text.replace(old,'CHECKPOINT = Path('+repr(str(checkpoint))+')'))
    path=src/'CMakeLists.txt';text=path.read_text()
    old='set(checkpoint "${CMAKE_CURRENT_SOURCE_DIR}/../../experiments/motion-ablation-v2/imu-only-compact")'
    assert text.count(old)==1;path.write_text(text.replace(old,'set(checkpoint "'+str(checkpoint)+'")'))
    directory=OUT/'build-joint'
    run('configure',['cmake','-S',str(src),'-B',str(directory),
        '-DCMAKE_TOOLCHAIN_FILE='+str(ROOT/'workspace/build/toolchain/wasi-sdk/share/cmake/wasi-sdk-p1.cmake'),
        '-DPython3_EXECUTABLE='+sys.executable,'-DVIOBENCH_NATIVE_DIR='+str(ROOT/'workspace/runtime/native'),
        '-DVIOBENCH_OPENCV_DIR='+str(ROOT/'workspace/build/runtime/opencv-build'),
        '-DVISUAL_DOWNSAMPLE=2','-DVISUAL_SPARSE_DIRECTION=ON'])
    run('build',['cmake','--build',str(directory),'-j','4'])
    wasm=OUT/'AeroVIO-native-v10-joint.wasm';shutil.copy2(directory/'solution.wasm',wasm);artifacts['joint-v10']=wasm
    run('check',[sys.executable,'-m','viobench','check',str(wasm),'--frames','80','--logs',str(OUT/'check-joint')])
    write(ARCHIVE/'inference-source.json',{'teacher_commit':cfg['teacher_commit'],'cpp_sha256':source_hashes,
         'changes':'Only build paths selecting new weights; every C++ source byte is from teacher.'})
    write(ARCHIVE/'artifacts.json',{n:{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for n,p in artifacts.items()})
    write(ARCHIVE/'frozen-before-long-inference.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':sha(PROTOCOL),'source_sha256':sha(__file__),'artifacts_sha256':sha(ARCHIVE/'artifacts.json'),
        'evaluation_order':'rotate [teacher, category-v9, joint-v10, joint-repeat] by frame+episode index',
        'new_long_scores_used_to_select_weights':False})


def evaluate(cfg):
    from viobench.tools.datasets.storage import read_episode
    from viobench.tools.runner.engine import Engine
    from viobench.tools.runner.sandbox import Sandbox,Limits
    from viobench.tools.runner.metrics import evaluate_trajectory
    from viobench.tools.geometry import poses_to_matrices
    frozen_sources();frozen=json.loads((ARCHIVE/'frozen-before-long-inference.json').read_text())
    assert frozen['protocol_sha256']==sha(PROTOCOL) and frozen['source_sha256']==sha(__file__)
    assert frozen['artifacts_sha256']==sha(ARCHIVE/'artifacts.json')
    manifest_path=ROOT/cfg['long_manifest'];manifest=json.loads(manifest_path.read_text())
    assert manifest['status']=='complete' and len(manifest['validation'])==6 and not manifest['train']
    assert manifest['config_sha256']==sha(ROOT/'configs/long-validation-v10.json')
    assert len({e['seed'] for e in manifest['validation']})==6
    assert all(e['frames']==1200 and e['seed']>=420000 for e in manifest['validation'])
    old=json.loads((ROOT/cfg['training_manifest']).read_text())
    assert not {e['scene'] for e in old['train']}&{e['scene'] for e in manifest['validation']}
    for scene in cfg['new_long_validation']['scenes']:assert sum(e['scene']==scene for e in manifest['validation'])==2
    output=OUT/'long-validation';output.mkdir()
    artifacts=json.loads((ARCHIVE/'artifacts.json').read_text());variants=list(artifacts)+['joint-repeat']
    engine=Engine();limits=Limits.from_dict(json.loads((ROOT/cfg['limits']).read_text()))
    rows={n:{'episodes':[]} for n in variants};checks=[];all_walls={n:[] for n in variants}
    all_truth=[];all_times=[];all_outputs={n:[] for n in variants}
    for ei,entry in enumerate(manifest['validation']):
        path=ROOT/entry['path'];episode=read_episode(path,frames='lazy');config,adapted=episode.adapt((320,240))
        times=episode.frame_times;assert len(times)==1200
        sandboxes={};results={n:[] for n in variants};fuel={n:[] for n in variants};walls={n:[] for n in variants}
        for name in variants:
            artifact=artifacts['joint-v10' if name=='joint-repeat' else name];wasm=Path(artifact['path'])
            assert sha(wasm)==artifact['sha256']
            sb=Sandbox(wasm,engine,limits=limits,log_dir=output/str(ei)/name);sb.init(config);sandboxes[name]=sb
        for i,(t,frame) in enumerate(zip(times,adapted,strict=True)):
            imu=episode.imu_window(i).copy();start=(ei+i)%len(variants)
            for name in variants[start:]+variants[:start]:
                step=sandboxes[name].step(float(t),frame,imu)
                results[name].append(np.r_[step.pose,step.twist]);fuel[name].append(step.fuel);walls[name].append(step.wall_seconds)
        results={n:np.asarray(v) for n,v in results.items()};fuel={n:np.asarray(v,dtype=np.int64) for n,v in fuel.items()}
        assert np.array_equal(results['joint-v10'],results['joint-repeat'])
        assert np.array_equal(fuel['joint-v10'],fuel['joint-repeat'])
        for name in variants:
            assert np.isfinite(results[name]).all()
            assert np.array_equal(results['teacher'][:,3:7],results[name][:,3:7])
            assert np.array_equal(results['teacher'][:,10:13],results[name][:,10:13])
            poses=poses_to_matrices(results[name][:,:7])
            velocity=np.einsum('nji,nj->ni',poses[1:,:3,:3],np.diff(results[name][:,:3],axis=0))/np.diff(times)[:,None]
            assert np.max(np.abs(velocity-results[name][1:,7:10]))<1e-8
            np.savez_compressed(output/str(ei)/name/'trace.npz',times=times,outputs=results[name],fuel=fuel[name],wall=walls[name])
            all_outputs[name].append(results[name]);all_walls[name].extend(walls[name])
        # Ground truth is obtained only after every guest finishes this episode.
        truth=episode.ground_truth_state();all_truth.append(truth);all_times.append(times)
        for name in variants:
            metrics=evaluate_trajectory(truth,results[name],times).as_dict()
            assert metrics['frames']==1200 and metrics['filled_frames']==0
            rows[name]['episodes'].append({'scene':entry['scene'],'seed':entry['seed'],'metrics':metrics,
                'distance_1s':distance_metrics(truth,results[name],times),'max_fuel':int(fuel[name].max()),
                'mean_fuel':float(fuel[name].mean()),'median_step_ms':float(np.median(walls[name])*1000),
                'trace_sha256':sha(output/str(ei)/name/'trace.npz')})
        checks.append({'scene':entry['scene'],'seed':entry['seed'],'joint_repeat_outputs_fuel_exact':True,
            'all_orientations_angular_twists_exact':True,'translation_twist_consistent':True,
            'episode_json_sha256':sha(path/'episode.json')})
        write(output/'progress.json',{'completed_episodes':ei+1,'total':6})
        print('LONG EVALUATED',ei,entry['scene'],entry['seed'],flush=True)
    for name,row in rows.items():
        es=row['episodes'];row['pooled_distance']=pooled_distance(all_truth,all_outputs[name],all_times)
        row['summary']={'mean_sim3_ate_m':float(np.mean([e['metrics']['sim3_ate_rmse'] for e in es])),
            'mean_unaligned_ate_m':float(np.mean([e['metrics']['ate_rmse'] for e in es])),
            'median_rotation_deg':float(np.degrees(np.median([e['metrics']['rotation_rmse'] for e in es]))),
            'mean_length_rmse_1s_m':float(np.mean([e['distance_1s']['length_rmse_m'] for e in es])),
            'pooled_length_rmse_1s_m':row['pooled_distance']['all']['length_rmse_m'],
            'median_distance_ratio':row['pooled_distance']['all']['median_predicted_true_ratio'],
            'max_fuel':max(e['max_fuel'] for e in es),'mean_fuel':float(np.mean([e['mean_fuel'] for e in es])),
            'paired_median_step_ms':float(np.median(all_walls[name])*1000)}
    write(ARCHIVE/'long-validation.json',{'scope':'6 new 60-second generated flights; rotating interleaved runtime order',
        'manifest_sha256':sha(manifest_path),'variants':rows,'checks':checks})
    shutil.copy2(manifest_path,ARCHIVE/'long-data-manifest.json')
    selected=rows['joint-v10']['summary'];s=cfg['selection'];passed={}
    for base in ('teacher','category-v9'):
        b=rows[base]['summary']
        passed[base+'_distance']=selected['mean_length_rmse_1s_m']<=b['mean_length_rmse_1s_m']*(1-s['minimum_length_rmse_improvement'])
        passed[base+'_aligned_ate']=selected['mean_sim3_ate_m']<=b['mean_sim3_ate_m']*s['max_sim3_ate_ratio']+1e-9
    wins=sum(a['distance_1s']['length_rmse_m']<b['distance_1s']['length_rmse_m'] for a,b in
        zip(rows['joint-v10']['episodes'],rows['teacher']['episodes'],strict=True))
    passed['episode_wins']=wins>=s['minimum_episode_length_wins_vs_teacher']
    passed['rotations_exact']=True;passed['fuel']=selected['max_fuel']<=rows['teacher']['summary']['max_fuel']*s['max_fuel_ratio']
    write(ARCHIVE/'selection.json',{'selected_variant':'joint-v10' if all(passed.values()) else 'teacher',
        'checks':passed,'length_episode_wins_vs_teacher':wins,'validation_sha256':sha(ARCHIVE/'long-validation.json'),
        'protocol_sha256':sha(PROTOCOL),'new_long_used_for_training_or_epoch_selection':False,
        'finished_utc':datetime.now(timezone.utc).isoformat()})
    print('COMPLETE',json.dumps({n:r['summary'] for n,r in rows.items()},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['build','evaluate'],required=True)
    args=p.parse_args();cfg=json.loads(PROTOCOL.read_text())
    if args.phase=='build':build(cfg)
    else:evaluate(cfg)

if __name__=='__main__':main()
