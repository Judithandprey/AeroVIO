"""Exactly replay the fixed fit to retain its final epoch; never change selection."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
from train_velocity_memory import load,validation,export
from training.velocity_memory import VelocityMemoryNet

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    arm=sys.argv[sys.argv.index('--arm')+1]
    assert arm in ('no-history','feedback','feedback-repeat')
    source_path=ROOT/'scripts/train_velocity_memory.py';source=source_path.read_text()
    old="out=ROOT/'models'/('velocity-v11-'+args.arm);out.mkdir()"
    assert source.count(old)==1
    source=source.replace(old,"out=ROOT/'models'/('velocity-v11-replay-'+args.arm);out.mkdir()")
    old='net.load_state_dict(best);net.eval();parity=export'
    assert source.count(old)==1
    source=source.replace(old,"torch.save(net.state_dict(),out/'final.pt');net.load_state_dict(best);net.eval();parity=export")
    exec(compile(source,str(source_path),'exec'),{'__name__':'__main__','__file__':str(source_path)})
    original=ROOT/'models'/('velocity-v11-'+arm);replay=ROOT/'models'/('velocity-v11-replay-'+arm)
    a=json.loads((original/'run.json').read_text());b=json.loads((replay/'run.json').read_text())
    for key in ('model_sha256','best_epoch','best_validation','initial_weights_sha256'):assert a[key]==b[key],key
    ha=json.loads((original/'history.json').read_text());hb=json.loads((replay/'history.json').read_text())
    assert [{k:v for k,v in r.items() if k!='elapsed_s'} for r in ha]==[{k:v for k,v in r.items() if k!='elapsed_s'} for r in hb]
    cfg=json.loads((ROOT/'configs/velocity-memory-v11.json').read_text());meta=json.loads((ROOT/cfg['parent_model']/'meta.json').read_text())
    norm=tuple(torch.tensor(meta[k]) for k in ('state_mean','state_std','target_mean','target_std'))
    net=VelocityMemoryNet();net.load_state_dict(torch.load(replay/'final.pt',weights_only=True));net.eval()
    out=ROOT/'models'/('velocity-v11-'+arm+'-final');out.mkdir();val=load('validation',cfg);feedback=arm!='no-history'
    parity=export(net,out,meta,feedback,val,norm)
    record={**b,'arm':arm+'-final','diagnostic_only':True,'best_epoch':40,'best_validation':validation(net,val,norm,feedback),
            'model_sha256':sha(out/'net.onnx'),'parity':parity,'selection':'fixed terminal epoch 40; not validation-selected',
            'exact_replay_history_match':True,'replay_source_sha256':sha(__file__)}
    (out/'run.json').write_text(json.dumps(record,indent=2)+'\n');(out/'history.json').write_text(json.dumps(hb,indent=2)+'\n')
    print('FIXED FINAL DIAGNOSTIC',arm,record['best_validation'],flush=True)
if __name__=='__main__':main()
