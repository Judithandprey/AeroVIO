"""Run the gated fixed V10 fit and one exact repeat; never read new-long scores."""
from datetime import datetime,timezone
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    protocol=ROOT/'configs/joint-balance-v10.json';cfg=json.loads(protocol.read_text());t=cfg['training']
    out=ROOT/'results/joint-balance-v10';audit=ROOT/'data/joint-sampling-v10/sampling-audit.json'
    plan=ROOT/'data/joint-sampling-v10/joint-plan.npz';a=json.loads(audit.read_text())
    assert a['status']=='passed_before_training' and a['protocol_sha256']==sha(protocol)
    assert a['plan_sha256']==sha(plan) and a['low_speed_fraction_joint_each_epoch']==[.25]*40
    assert a['max_sample_repeats_each_epoch']<=4
    frozen=out/'frozen-before-training.json'
    if frozen.exists():raise FileExistsError('preserve previous fits')
    paths=[protocol,ROOT/'configs/long-validation-v10.json',ROOT/'scripts/train_motion.py',
           ROOT/'training/motion_model.py',ROOT/'training/joint_sampling.py',ROOT/'scripts/prepare_joint_sampling.py',
           ROOT/'scripts/train_joint_balance.py',ROOT/'scripts/generate_long_validation.py',audit,plan]
    frozen.write_text(json.dumps({'frozen_utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths},'new_long_scores_used':False},indent=2)+'\n')
    for name in ('joint','joint-repeat'):
        command=[str(ROOT/'.train-venv/bin/python'),str(ROOT/'scripts/train_motion.py'),
            '--cache',str(ROOT/cfg['cache']),'--out',str(ROOT/f'models/joint-v10-{name}'),
            '--compact-out',str(ROOT/f'models/joint-v10-{name}-compact'),
            '--input-mode','imu_only','--warp-weight',str(t['warp_weight']),'--epochs',str(t['epochs']),
            '--batch',str(t['batch']),'--lr',str(t['learning_rate']),'--seed',str(t['seed']),
            '--threads',str(t['threads']),'--sampling-plan',str(plan)]
        print('TRAIN',name,flush=True)
        with (out/('train-'+name+'.log')).open('x') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('TRAINING COMPLETE',flush=True)

if __name__=='__main__':main()
