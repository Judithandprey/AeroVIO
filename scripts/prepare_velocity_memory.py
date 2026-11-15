"""Archive causal gyro transports and chronology audit without estimator scores."""
import hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'workspace/solution')]
from gyro_bias import InitialGyroBias, integrate_debiased_rotation
from preintegrate import exp_so3
from viobench.tools.datasets.storage import read_episode
from training.velocity_memory import sequence_indices

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    out=ROOT/'data/velocity-memory-v11';out.mkdir()
    cfg=json.loads((ROOT/'configs/velocity-memory-v11.json').read_text())
    cache=ROOT/cfg['cache'];record=json.loads((cache/'cache.json').read_text())
    audit={'predictor_inputs_use_ground_truth':False,'splits':{},'source_sha256':{}}
    for split in ('train','validation'):
        with np.load(cache/(split+'.npz')) as z: data={k:z[k] for k in z.files}
        rotations=[];offset=0;episodes=[]
        for eid,entry in enumerate(record['splits'][split]['episodes']):
            ep=read_episode(entry['path'],frames='lazy');config,_=ep.adapt((320,240))
            calibration=InitialGyroBias();times=ep.frame_times
            for i,t in enumerate(times):
                imu=ep.imu_window(i)
                if i:
                    idx=offset+i-1
                    assert int(data['episode_id'][idx])==eid
                    assert abs(float(data['state'][idx,13])-(t-times[i-1]))<1e-6
                    delta=(integrate_debiased_rotation(imu,config['T_cam_imu'],times[i-1],t,calibration.bias_imu)
                           if calibration.accepted else exp_so3(data['state'][idx,:3].astype(np.float64)))
                    rotations.append(delta)
                calibration.observe_after_output(t,imu)
            episodes.append({'episode_id':eid,'start':offset,'pairs':len(times)-1,'path':entry['path'],
                             'episode_sha256':sha(Path(entry['path'])/'episode.json'),'calibration':calibration.diagnostics()})
            offset+=len(times)-1
        assert offset==len(data['state'])
        indices,active=sequence_indices(data['episode_id'],np.arange(offset),cfg['training']['sequence_length'])
        np.savez_compressed(out/(split+'.npz'),rotation=np.asarray(rotations,dtype=np.float32),indices=indices,active=active)
        audit['splits'][split]={'cache_sha256':sha(cache/(split+'.npz')),'history_sha256':sha(out/(split+'.npz')),
            'targets':offset,'context_frames':int(active.sum()-offset),'episodes':episodes,'no_cross_episode_context':True}
    plan=ROOT/cfg['sampling_plan'];p=np.load(plan)['order'];assert p.shape==(40,2388)
    audit['sampling_plan_sha256']=sha(plan)
    audit['same_target_exposure_as_v10']=True
    for name in ('configs/velocity-memory-v11.json','configs/long-validation-v11.json','training/velocity_memory.py',
                 'scripts/prepare_velocity_memory.py','scripts/train_velocity_memory.py','training/motion_model.py'):
        audit['source_sha256'][name]=sha(ROOT/name)
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('PREPARATION PASSED',[(k,v['targets']) for k,v in audit['splits'].items()],flush=True)

if __name__=='__main__':main()
