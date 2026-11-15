"""Frozen gate-only comparison; original regressions and fresh confirmation."""
import argparse,hashlib,importlib.metadata,itertools,json,platform,sys
from pathlib import Path
import numpy as np
from viobench.tools.datasets.storage import read_episode
from viobench.tools.runner.engine import Engine
from viobench.tools.runner.sandbox import Sandbox,Limits
from viobench.tools.runner.metrics import evaluate_trajectory
from viobench.tools.geometry import poses_to_matrices
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.gate_metrics import cruise_metrics,combine_cruise
from evaluate_velocity_memory import regime_metrics,combine_regimes,execute
from run_joint_balance import pooled_distance
OUT=ROOT/'results/memory-gate-v12';A=ROOT/'experiments/memory-gate-v12'
NAMES=['v10','v11','constant','adaptive']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['regression','confirmation'],required=True);args=ap.parse_args()
    cfg=read(ROOT/'configs/memory-gate-v12.json');artifacts=read(A/'artifacts.json')
    for a in artifacts.values():assert sha(a['path'])==a['sha256']
    source_paths=['configs/memory-gate-v12.json','configs/long-validation-v12.json','training/memory_gate.py','training/gate_metrics.py',
        'scripts/train_memory_gate.py','scripts/build_memory_gate.py','scripts/verify_gate_runtime.py','scripts/evaluate_memory_gate.py',
        'scripts/evaluate_velocity_memory.py','scripts/generate_gate_validation.py','scripts/generate_long_validation.py',
        'workspace/memory_gate_native/inference.h','workspace/memory_gate_native/export_weights.py']
    frozen={'source_sha256':{p:sha(ROOT/p) for p in source_paths},'artifacts_sha256':sha(A/'artifacts.json'),
        'selected_models_sha256':{n:sha(A/n/'net.onnx') for n in ('constant','adaptive')},
        'protocol':'fixed predictors, train gate only; best updated epoch chosen on original validation before long scores',
        'timing':'four distinct artifacts, all 24 orders; regression times excluded while generation overlaps',
        'motion_definitions':cfg['motion'],'confirmation_guards':cfg['confirmation_guards']}
    frozen_path=A/'frozen-before-long-evaluation.json'
    if frozen_path.exists():assert read(frozen_path)==frozen
    else:write(frozen_path,frozen)
    environment={'python':sys.version,'host':platform.platform(),
        'packages':{p:importlib.metadata.version(p) for p in ('numpy','wasmtime','viobench','opencv-python-headless')}}
    write(A/('environment-'+args.phase+'.json'),environment)
    split=args.phase;manifest_path=ROOT/cfg['evaluation'][split+'_manifest'];manifest=read(manifest_path)
    assert manifest['status']=='complete' and len(manifest['validation'])==6
    output=OUT/split;output.mkdir();engine=Engine();limits=Limits.from_dict(read(ROOT/cfg['evaluation']['limits']))
    orders=list(itertools.permutations(NAMES));rng=np.random.default_rng(cfg['evaluation']['order_seed'])
    fingerprints=set()
    for name in ('data/hyperdrone-motion-v2/data_manifest.json','data/long-validation-v10/data_manifest.json','data/long-validation-v11/data_manifest.json'):
        old=read(ROOT/name)
        for part in ('train','validation'):
            for e in old[part]:
                ep=read_episode(ROOT/e['path'],frames='lazy');fingerprints.add(hashlib.sha256(ep.ground_truth_state()[:200].tobytes()).hexdigest())
    rows={n:{'episodes':[]} for n in NAMES};predictions={n:[] for n in NAMES}
    walls={n:[] for n in NAMES};fuels={n:[] for n in NAMES};truths=[];alltimes=[];checks=[]
    for ei,entry in enumerate(manifest['validation']):
        ep=read_episode(ROOT/entry['path'],frames='lazy');config,frames=ep.adapt((320,240));times=ep.frame_times
        assert len(times)==1200
        sb={n:Sandbox(Path(artifacts[n]['path']),engine,limits=limits,log_dir=output/str(ei)/n) for n in NAMES}
        for s in sb.values():s.init(config)
        outputs={n:[] for n in NAMES};efuel={n:[] for n in NAMES};ewall={n:[] for n in NAMES};counts={n:[0]*4 for n in NAMES}
        for i,(t,frame) in enumerate(zip(times,frames,strict=True)):
            if i%24==0:block=rng.permutation(24)
            imu=ep.imu_window(i).copy()
            for position,n in enumerate(orders[int(block[i%24])]):
                step=sb[n].step(float(t),frame,imu);outputs[n].append(np.r_[step.pose,step.twist])
                efuel[n].append(step.fuel);ewall[n].append(step.wall_seconds);counts[n][position]+=1
        outputs={n:np.asarray(o) for n,o in outputs.items()}
        assert all(c==[300]*4 for c in counts.values())
        for n in NAMES:
            assert np.isfinite(outputs[n]).all()
            assert np.array_equal(outputs[n][:,3:7],outputs['v10'][:,3:7])
            assert np.array_equal(outputs[n][:,10:13],outputs['v10'][:,10:13])
            rotation=poses_to_matrices(outputs[n][:,:7])[:, :3,:3]
            velocity=np.einsum('nji,nj->ni',rotation[1:],np.diff(outputs[n][:,:3],axis=0))/np.diff(times)[:,None]
            assert np.abs(velocity-outputs[n][1:,7:10]).max()<1e-8
            np.savez_compressed(output/str(ei)/n/'trace.npz',times=times,outputs=outputs[n],fuel=efuel[n],wall=ewall[n])
            predictions[n].append(outputs[n]);walls[n].extend(ewall[n]);fuels[n].extend(efuel[n])
        # Duplicate and endpoint identities never enter the balanced timings.
        extras=[('adaptive','adaptive')]
        if split=='regression' and ei==0:extras+=[('endpoint-v10','v10'),('endpoint-v11','v11')]
        exact={}
        for n,reference in extras:
            s=Sandbox(Path(artifacts[n]['path']),engine,limits=limits,log_dir=output/str(ei)/(n+'-separate'));s.init(config)
            actual,fuel=execute(s,ep,ep.adapt((320,240))[1])
            difference=float(np.abs(actual-outputs[reference]).max());assert difference<1e-8,(n,difference)
            if n=='adaptive':assert np.array_equal(actual,outputs[n]) and np.array_equal(fuel,np.asarray(efuel[n]))
            exact[n]={'maximum_output_difference':difference,'frames':len(actual),'fuel_exact':bool(np.array_equal(fuel,np.asarray(efuel[reference])))}
        truth=ep.ground_truth_state();fingerprint=hashlib.sha256(truth[:200].tobytes()).hexdigest()
        if split=='confirmation':assert fingerprint not in fingerprints
        fingerprints.add(fingerprint);truths.append(truth);alltimes.append(times)
        for n in NAMES:
            metrics=evaluate_trajectory(truth,outputs[n],times).as_dict();assert metrics['frames']==1200 and metrics['filled_frames']==0
            rows[n]['episodes'].append({'scene':entry['scene'],'seed':entry['seed'],'metrics':metrics,
                'motion_diagnostics':regime_metrics(truth,outputs[n],times),'cruise':cruise_metrics(truth,outputs[n],times,cfg['motion']),
                'trace_sha256':sha(output/str(ei)/n/'trace.npz'),'median_step_ms':float(np.median(ewall[n])*1000),
                'max_fuel':max(efuel[n])})
        checks.append({'scene':entry['scene'],'seed':entry['seed'],'gt_prefix_sha256':fingerprint,'extra_checks':exact,
            'orientation_angular_twist_exact':True,'linear_twist_consistent':True,'execution_positions':counts})
        write(output/'progress.json',{'completed':ei+1,'total':6});print('EVALUATED',split,ei,entry['seed'],flush=True)
    for n in NAMES:
        es=rows[n]['episodes'];distance=pooled_distance(truths,predictions[n],alltimes)
        rows[n]['pooled_distance']=distance;rows[n]['motion_diagnostics']=combine_regimes(es)
        rows[n]['cruise']=combine_cruise([e['cruise'] for e in es])
        rows[n]['summary']={'mean_sim3_ate_m':float(np.mean([e['metrics']['sim3_ate_rmse'] for e in es])),
            'mean_unaligned_ate_m':float(np.mean([e['metrics']['ate_rmse'] for e in es])),
            'pooled_length_rmse_m':distance['all']['length_rmse_m'],'low_net_motion_rmse_m':distance['speed_bins']['below_0.1']['length_rmse_m'],
            'mean_fuel':float(np.mean(fuels[n])),'peak_fuel':max(fuels[n]),'median_step_ms':float(np.median(walls[n])*1000)}
    result={'scope':split,'wall_time_comparable':split=='confirmation','variants':rows,'checks':checks,'manifest_sha256':sha(manifest_path)}
    write(A/(split+'.json'),result);write(A/(split+'-manifest.json'),manifest)
    print('SUMMARY',split,json.dumps({n:r['summary'] for n,r in rows.items()}),flush=True)
    if split=='confirmation':
        s=cfg['confirmation_guards'];a=rows['adaptive'];v=rows['v10'];old=rows['v11'];stop=a['motion_diagnostics']['stops']
        ratio=lambda x,y,k: None if x is None or y is None else x<=y*k
        guards={
            'turning':ratio(a['motion_diagnostics']['turning']['length_rmse_m'],old['motion_diagnostics']['turning']['length_rmse_m'],s['turning_ratio_vs_v11']),
            'stopping':ratio(stop['mean_displacement_error_m'],old['motion_diagnostics']['stops']['mean_displacement_error_m'],s['stop_error_ratio_vs_v11']) if stop['events']>=s['minimum_stop_events'] else None,
            'cruise_vs_v10':ratio(a['cruise']['length_rmse_m'],v['cruise']['length_rmse_m'],s['cruise_ratio_vs_v10']),
            'cruise_vs_v11':ratio(a['cruise']['length_rmse_m'],old['cruise']['length_rmse_m'],s['cruise_ratio_vs_v11']),
            'ate':ratio(a['summary']['mean_sim3_ate_m'],v['summary']['mean_sim3_ate_m'],s['ate_ratio_vs_v10']),
            'overall_distance':ratio(a['summary']['pooled_length_rmse_m'],v['summary']['pooled_length_rmse_m'],s['length_ratio_vs_v10']),
            'mean_fuel':ratio(a['summary']['mean_fuel'],v['summary']['mean_fuel'],s['fuel_ratio_vs_v10']),
            'peak_fuel':ratio(a['summary']['peak_fuel'],v['summary']['peak_fuel'],s['fuel_ratio_vs_v10'])}
        train=read(A/'adaptive/run.json')
        write(A/'selection.json',{'reference_retained':'v10','main_or_default_changed':False,'confirmation_guards':guards,
            'stop_events':stop['events'],'minimum_stop_events':s['minimum_stop_events'],
            'candidate_original_validation_beats_v10':train['best_validation']['pose_loss']<train['endpoint_validation_references']['0']['pose_loss'],
            'candidate_original_validation_beats_initial_mixture':train['best_validation']['pose_loss']<train['initial_validation']['pose_loss'],
            'no_long_scores_used_for_checkpoint_selection':True})
        print('COMPLETE',guards,flush=True)

if __name__=='__main__':main()
