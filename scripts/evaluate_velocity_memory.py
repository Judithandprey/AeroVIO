"""Frozen regression and fresh confirmation; fixed terminal models are diagnostic."""
import argparse,hashlib,itertools,json,sys
from pathlib import Path
import numpy as np
from viobench.tools.datasets.storage import read_episode
from viobench.tools.runner.engine import Engine
from viobench.tools.runner.sandbox import Sandbox,Limits
from viobench.tools.runner.metrics import evaluate_trajectory
from viobench.tools.geometry import poses_to_matrices
from run_joint_balance import pooled_distance
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/velocity-memory-v11';ARCHIVE=ROOT/'experiments/velocity-memory-v11'
NAMES=['teacher','v10','no-history-final','feedback-final']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def regime_metrics(truth,pred,times):
    """Report physical speed regimes separately from small one-second net motion."""
    dt=np.diff(times);delta=np.diff(truth[:,:3],axis=0);speed=np.linalg.norm(delta,axis=1)/dt
    matrices=poses_to_matrices(truth[:,:7]);rot=matrices[:-1,:3,:3].transpose(0,2,1)@matrices[1:,:3,:3]
    angular=np.degrees(np.arccos(np.clip((np.trace(rot,axis1=1,axis2=2)-1)/2,-1,1)))/dt
    starts=np.arange(len(times)-20);dg=truth[20:,:3]-truth[:-20,:3];dp=pred[20:,:3]-pred[:-20,:3]
    actual=np.linalg.norm(dg,axis=1);estimate=np.linalg.norm(dp,axis=1)
    speeds=np.stack([speed[i:i+20] for i in starts]);turns=np.stack([angular[i:i+20] for i in starts])
    path=np.array([np.linalg.norm(delta[i:i+20],axis=1).sum() for i in starts])
    masks={'stationary':speeds.max(1)<.02,
           'slow_straight':(speeds.max(1)<.1)&(speeds.mean(1)>=.02)&(actual/np.maximum(path,1e-12)>.8)&(turns.mean(1)<15),
           'decelerating':(speeds[:,:5].mean(1)-speeds[:,-5:].mean(1)>.05)&(speeds[:,-5:].mean(1)<.1),
           'turning':turns.mean(1)>=15,
           'small_net_but_moving':(actual<.1)&(path>=.1)}
    result={}
    for name,mask in masks.items():
        errors=estimate[mask]-actual[mask]
        result[name]={'windows':int(mask.sum()),'length_squared_error_sum':float((errors**2).sum()),
                      'length_error_sum':float(errors.sum()),'length_rmse_m':float(np.sqrt(np.mean(errors**2))) if len(errors) else None}
    low=speed<.02;events=[];i=0
    predspeed=np.linalg.norm(np.diff(pred[:,:3],axis=0),axis=1)/dt
    while i<len(low):
        if not low[i]:i+=1;continue
        end=i
        while end<len(low) and low[end]:end+=1
        prior=speed[max(0,i-10):i]
        if end-i>=10 and len(prior) and np.max(prior)>=.1:
            stopend=min(end,i+20);true_move=truth[stopend,:3]-truth[i,:3];estimated_move=pred[stopend,:3]-pred[i,:3]
            settling=None
            for j in range(i,end-4):
                if np.all(predspeed[j:j+5]<.02):settling=float(times[j]-times[i]);break
            events.append({'start_frame':int(i),'duration_s':float(times[stopend]-times[i]),
                'displacement_error_m':float(np.linalg.norm(estimated_move-true_move)),
                'settled':settling is not None,'settling_time_s':settling})
        i=end
    return {'regimes':result,'stop_events':events}

def combine_regimes(episodes):
    result={}
    for name in ('stationary','slow_straight','decelerating','turning','small_net_but_moving'):
        rows=[e['motion_diagnostics']['regimes'][name] for e in episodes];n=sum(r['windows'] for r in rows)
        result[name]={'windows':n,'length_rmse_m':float(np.sqrt(sum(r['length_squared_error_sum'] for r in rows)/n)) if n else None,
                      'length_bias_m':sum(r['length_error_sum'] for r in rows)/n if n else None}
    stops=[s for e in episodes for s in e['motion_diagnostics']['stop_events']]
    result['stops']={'events':len(stops),'settled':sum(s['settled'] for s in stops),
        'mean_displacement_error_m':float(np.mean([s['displacement_error_m'] for s in stops])) if stops else None,
        'mean_settling_time_s_among_settled':float(np.mean([s['settling_time_s'] for s in stops if s['settled']])) if any(s['settled'] for s in stops) else None}
    return result

def execute(sb,episode,frames):
    outputs=[];fuel=[]
    for i,(t,frame) in enumerate(zip(episode.frame_times,frames,strict=True)):
        step=sb.step(float(t),frame,episode.imu_window(i).copy());outputs.append(np.r_[step.pose,step.twist]);fuel.append(step.fuel)
    return np.asarray(outputs),np.asarray(fuel)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['regression','confirmation','both'],default='both');args=parser.parse_args()
    cfg=json.loads((ROOT/'configs/velocity-memory-v11.json').read_text());artifacts=json.loads((ARCHIVE/'artifacts.json').read_text())
    assert all(json.loads((ARCHIVE/n/'run.json').read_text())['best_epoch']==0 for n in ('no-history','feedback'))
    for a in artifacts.values():assert sha(a['path'])==a['sha256']
    source_paths=['configs/velocity-memory-v11.json','configs/long-validation-v11.json','scripts/evaluate_velocity_memory.py',
                  'scripts/build_velocity_memory.py','scripts/replay_velocity_diagnostics.py','training/velocity_memory.py',
                  'scripts/train_velocity_memory.py','scripts/generate_memory_validation.py','scripts/verify_velocity_runtime.py']
    frozen=ARCHIVE/'frozen-before-long-evaluation.json'
    frozen_record={'source_sha256':{p:sha(ROOT/p) for p in source_paths},'artifacts_sha256':sha(ARCHIVE/'artifacts.json'),
        'primary_selection':'both arms retain epoch zero by original full-episode validation; V10 remains selected',
        'diagnostic_policy':'fixed final epoch 40 saved by exact training replay after seeing original validation, before long evaluation; no hyperparameter change or long-score selection',
        'order':'all 24 permutations of four distinct artifacts per 24 frames, shuffled seed 20261114; selected-arm equality and full feedback repeat run separately after timing',
        'timed_variants':NAMES,'regression_wall_time_not_used':True,'confirmation_requires_completed_generation':True}
    if frozen.exists():assert json.loads(frozen.read_text())==frozen_record
    else:write(frozen,frozen_record)
    engine=Engine();limits=Limits.from_dict(json.loads((ROOT/cfg['evaluation']['limits']).read_text()))
    rng=np.random.default_rng(cfg['evaluation']['order_seed']);orders=list(itertools.permutations(NAMES));all_fingerprints=set()
    for name in ('data/hyperdrone-motion-v2/data_manifest.json','data/long-validation-v10/data_manifest.json'):
        old=json.loads((ROOT/name).read_text())
        for split in ('train','validation'):
            for e in old[split]:
                ep=read_episode(ROOT/e['path'],frames='lazy')
                all_fingerprints.add(hashlib.sha256(ep.ground_truth_state()[:200].tobytes()).hexdigest())
    for split,manifest_key in [('regression','regression_manifest'),('confirmation','confirmation_manifest')]:
        if args.phase!='both' and split!=args.phase:continue
        manifest_path=ROOT/cfg['evaluation'][manifest_key];manifest=json.loads(manifest_path.read_text())
        assert manifest['status']=='complete' and len(manifest['validation'])==6 and not manifest['train']
        output=OUT/split;output.mkdir();rows={n:{'episodes':[]} for n in NAMES};checks=[];truths=[];alltimes=[]
        predictions={n:[] for n in NAMES};walls={n:[] for n in NAMES};peak={n:[] for n in NAMES};fuels={n:[] for n in NAMES}
        for ei,entry in enumerate(manifest['validation']):
            ep=read_episode(ROOT/entry['path'],frames='lazy');config,frames=ep.adapt((320,240));times=ep.frame_times
            assert len(times)==1200
            sb={n:Sandbox(Path(artifacts[n]['path']),engine,limits=limits,log_dir=output/str(ei)/n) for n in NAMES}
            for s in sb.values():s.init(config)
            outputs={n:[] for n in NAMES};efuel={n:[] for n in NAMES};ewalls={n:[] for n in NAMES};counts={n:[0]*4 for n in NAMES}
            for i,(t,frame) in enumerate(zip(times,frames,strict=True)):
                if i%24==0:block=rng.permutation(24)
                order=orders[int(block[i%24])];imu=ep.imu_window(i).copy()
                for pos,n in enumerate(order):
                    step=sb[n].step(float(t),frame,imu);outputs[n].append(np.r_[step.pose,step.twist])
                    efuel[n].append(step.fuel);ewalls[n].append(step.wall_seconds);counts[n][pos]+=1
            outputs={n:np.asarray(v) for n,v in outputs.items()}
            assert all(v==[300]*4 for v in counts.values())
            for n in NAMES:
                assert np.isfinite(outputs[n]).all()
                assert np.array_equal(outputs[n][:,3:7],outputs['teacher'][:,3:7])
                assert np.array_equal(outputs[n][:,10:13],outputs['teacher'][:,10:13])
                matrices=poses_to_matrices(outputs[n][:,:7])
                velocity=np.einsum('nji,nj->ni',matrices[1:,:3,:3],np.diff(outputs[n][:,:3],axis=0))/np.diff(times)[:,None]
                assert np.max(np.abs(velocity-outputs[n][1:,7:10]))<1e-8
                np.savez_compressed(output/str(ei)/n/'trace.npz',times=times,outputs=outputs[n],fuel=efuel[n],wall=ewalls[n])
                predictions[n].append(outputs[n]);walls[n].extend(ewalls[n]);fuels[n].extend(efuel[n]);peak[n].append(max(efuel[n]))
            # These additional runs occur outside the timed interleave.
            extra_checks={}
            for n,expected in [('no-history','v10'),('feedback','v10'),('feedback-final','feedback-final')]:
                logname=n+'-separate';s=Sandbox(Path(artifacts[n]['path']),engine,limits=limits,log_dir=output/str(ei)/logname);s.init(config)
                actual,fuel=execute(s,ep,ep.adapt((320,240))[1]);difference=float(np.max(np.abs(actual-outputs[expected])))
                assert difference<1e-8,(n,difference)
                if n=='feedback-final':assert np.array_equal(actual,outputs[n]) and np.array_equal(fuel,efuel[n])
                extra_checks[n]={'maximum_pose_twist_difference':difference,'frames':len(actual)}
            truth=ep.ground_truth_state();fingerprint=hashlib.sha256(truth[:200].tobytes()).hexdigest()
            if split=='confirmation':assert fingerprint not in all_fingerprints,'repeated GT prefix'
            all_fingerprints.add(fingerprint);truths.append(truth);alltimes.append(times)
            for n in NAMES:
                metrics=evaluate_trajectory(truth,outputs[n],times).as_dict();assert metrics['frames']==1200 and metrics['filled_frames']==0
                rows[n]['episodes'].append({'scene':entry['scene'],'seed':entry['seed'],'metrics':metrics,
                    'motion_diagnostics':regime_metrics(truth,outputs[n],times),'trace_sha256':sha(output/str(ei)/n/'trace.npz'),
                    'median_step_ms':float(np.median(ewalls[n])*1000),'max_fuel':max(efuel[n])})
            checks.append({'scene':entry['scene'],'seed':entry['seed'],'gt_prefix_sha256':fingerprint,'extra_checks':extra_checks,
                'orientation_angular_twist_exact':True,'linear_twist_consistent':True,'execution_positions':counts})
            write(output/'progress.json',{'completed':ei+1,'total':6});print('EVALUATED',split,ei,entry['scene'],entry['seed'],flush=True)
        for n in NAMES:
            es=rows[n]['episodes'];rows[n]['pooled_distance']=pooled_distance(truths,predictions[n],alltimes)
            rows[n]['motion_diagnostics']=combine_regimes(es)
            rows[n]['summary']={'mean_sim3_ate_m':float(np.mean([e['metrics']['sim3_ate_rmse'] for e in es])),
                'mean_unaligned_ate_m':float(np.mean([e['metrics']['ate_rmse'] for e in es])),
                'pooled_length_rmse_m':rows[n]['pooled_distance']['all']['length_rmse_m'],
                'low_net_motion_rmse_m':rows[n]['pooled_distance']['speed_bins']['below_0.1']['length_rmse_m'],
                'mean_fuel':float(np.mean(fuels[n])),'peak_fuel':max(peak[n]),'median_step_ms':float(np.median(walls[n])*1000)}
        result={'scope':split,'wall_time_comparable':split=='confirmation','manifest_sha256':sha(manifest_path),'variants':rows,'checks':checks}
        write(ARCHIVE/(split+'.json'),result);write(ARCHIVE/(split+'-manifest.json'),manifest)
        print('SUMMARY',split,json.dumps({n:r['summary'] for n,r in rows.items()}),flush=True)
    if args.phase=='regression':return
    confirm=json.loads((ARCHIVE/'confirmation.json').read_text())['variants'];f=confirm['feedback-final']['summary'];c=confirm['no-history-final']['summary'];v=confirm['v10']['summary'];t=confirm['teacher']['summary'];s=cfg['selection']
    guards={'low_motion_vs_control':f['low_net_motion_rmse_m']<=c['low_net_motion_rmse_m']*s['feedback_vs_control_low_motion_ratio'],
        'low_motion_vs_v10':f['low_net_motion_rmse_m']<=v['low_net_motion_rmse_m']*s['feedback_vs_v10_low_motion_ratio'],
        'low_motion_vs_teacher':f['low_net_motion_rmse_m']<=t['low_net_motion_rmse_m']*s['teacher_low_motion_ratio'],
        'ate':f['mean_sim3_ate_m']<=v['mean_sim3_ate_m']*s['max_ate_ratio_vs_v10'],
        'distance':f['pooled_length_rmse_m']<=v['pooled_length_rmse_m']*s['max_length_ratio_vs_v10'],
        'mean_fuel':f['mean_fuel']<=v['mean_fuel']*s['max_mean_fuel_ratio_vs_v10'],
        'peak_fuel':f['peak_fuel']<=v['peak_fuel']*s['max_peak_fuel_ratio_vs_v10']}
    write(ARCHIVE/'selection.json',{'selected_variant':'v10','reason':'Both full original-validation selectors retained epoch zero; terminal-epoch models remain diagnostic regardless of long scores.',
         'fixed_final_epoch_confirmation_guards':guards,'no_long_scores_used_for_checkpoint_selection':True})
    print('COMPLETE selected V10',guards,flush=True)
if __name__=='__main__':main()
