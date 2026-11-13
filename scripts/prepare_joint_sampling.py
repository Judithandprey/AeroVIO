"""Audit all joint-sampling exposures before any fit; preserve original caches."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.joint_sampling import cell_quotas,make_order,audit_order

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    protocol=ROOT/'configs/joint-balance-v10.json';cfg=json.loads(protocol.read_text())
    target=ROOT/'data/joint-sampling-v10';target.mkdir()
    labels=ROOT/cfg['training_labels'];cache=ROOT/cfg['cache']
    with np.load(labels) as z:speed,turn,ids=z['speed'],z['turn_deg_s'],z['episode_id']
    old=json.loads((ROOT/'experiments/motion-ablation-v2/imu-only/run.json').read_text())
    assert sha(cache/'train.npz')==old['train_cache_sha256']
    assert sha(cache/'validation.npz')==old['validation_cache_sha256']
    s=cfg['sampling'];sb=np.digitize(speed,s['speed_edges_m_s']);tb=np.digitize(turn,s['turn_edges_deg_s'])
    counts=np.zeros((4,4),dtype=np.int64);np.add.at(counts,(sb,tb),1)
    quotas=cell_quotas(counts,s['speed_quota_each_epoch'],s['max_exposure_per_example_per_epoch'])
    train=cfg['training']
    order=np.stack([make_order(sb,tb,quotas,e,train['seed']) for e in range(train['epochs'])])
    assert order.shape==(train['epochs'],train['samples_per_epoch'])
    cap=max(audit_order(o,sb,tb,quotas,s['max_exposure_per_example_per_epoch']) for o in order)
    low=np.mean(speed[order]<.1,axis=1)
    with np.load(ROOT/'data/motion-balance-v9/balanced-plan.npz') as z:v9_low=float(np.mean(speed[z['order']]<.1))
    assert np.all(low==.25) and .25<float(np.mean(speed<.1)) and .25<v9_low
    np.savez_compressed(target/'joint-plan.npz',order=order)
    episode_counts=[[len(np.unique(ids[(sb==i)&(tb==j)])) for j in range(4)] for i in range(4)]
    record={'status':'passed_before_training','protocol_sha256':sha(protocol),'labels_sha256':sha(labels),
        'train_cache_sha256':sha(cache/'train.npz'),'validation_cache_sha256':sha(cache/'validation.npz'),
        'counts':counts.tolist(),'unique_episodes_per_cell':episode_counts,'quotas_each_epoch':quotas.tolist(),
        'speed_exposure_each_epoch':quotas.sum(axis=1).tolist(),'low_speed_fraction_original':float(np.mean(speed<.1)),
        'low_speed_fraction_v9':v9_low,'low_speed_fraction_joint_each_epoch':low.tolist(),
        'max_sample_repeats_each_epoch':cap,'unique_pairs_seen_over_40_epochs':len(np.unique(order)),
        'plan_sha256':sha(target/'joint-plan.npz'),
        'speed_median_original':float(np.median(speed)),'speed_median_joint':float(np.median(speed[order])),
        'missing_cell_policy':'zero quota, never fabricated examples; support holes remain',
        'public_or_new_long_scores_used':False}
    (target/'sampling-audit.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
