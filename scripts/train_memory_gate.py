"""Frozen V10/V11 predictors; fit only constant/adaptive memory trust."""
import argparse,copy,hashlib,json,math,sys,time
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.memory_gate import GatedMotion
from training.velocity_memory import advance_memory
from training.motion_model import photometric_loss
from train_velocity_memory import load

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def weights_hash(module):return hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in module.state_dict().values())).hexdigest()

def sequence(net,data,indices,active):
    memory=torch.zeros(len(indices),3);valid=torch.zeros(len(indices),dtype=torch.bool)
    for t in range(indices.shape[1]):
        idx=indices[:,t];state=data['state'][idx]
        raw,g,_=net(state,memory,valid)
        memory,valid=advance_memory(raw,data['rotation'][idx],state[:,13],active[:,t])
    return raw,g

@torch.no_grad()
def validate(net,data,force=None):
    prediction=torch.zeros_like(data['target']);gates=[]
    for eid in data['episode_id'].unique():
        memory=torch.zeros(1,3);valid=torch.zeros(1,dtype=torch.bool)
        for idx in torch.where(data['episode_id']==eid)[0]:
            state=data['state'][idx:idx+1];raw,g,_=net(state,memory,valid,force)
            prediction[idx]=raw[0];gates.append(float(g[0,0]))
            memory,valid=advance_memory(raw,data['rotation'][idx:idx+1],state[:,13],torch.ones(1,dtype=torch.bool))
    loss=F.smooth_l1_loss((prediction-net.target_mean)/net.target_std,(data['target']-net.target_mean)/net.target_std)
    return {'pose_loss':float(loss),'translation_rmse_m':float((prediction[:,:3]-data['target'][:,:3]).square().sum(1).mean().sqrt()),
            'gate_mean':float(np.mean(gates)),'gate_std':float(np.std(gates)),'gate_min':min(gates),'gate_max':max(gates)}

def export(net,out,cfg,data):
    torch.save(net.gate.state_dict(),out/'best.pt')
    torch.onnx.export(net.gate,torch.zeros(1,24),str(out/'net.onnx'),input_names=['features'],output_names=['trust'],opset_version=13,dynamo=False)
    cvnet=cv2.dnn.readNetFromONNX(str(out/'net.onnx'));maximum=0.;steps=0
    rng=np.random.default_rng(1712)
    for x in rng.normal(size=(128,24)).astype(np.float32):
        cvnet.setInput(x[None]);actual=cvnet.forward()
        with torch.no_grad():expected=net.gate(torch.from_numpy(x[None])).numpy()
        maximum=max(maximum,float(np.abs(actual-expected).max()))
    with torch.no_grad():
        for eid in data['episode_id'].unique():
            memory=torch.zeros(1,3);valid=torch.zeros(1,dtype=torch.bool)
            for idx in torch.where(data['episode_id']==eid)[0]:
                state=data['state'][idx:idx+1];raw,g,x=net(state,memory,valid)
                cvnet.setInput(x.numpy());maximum=max(maximum,abs(float(cvnet.forward()[0,0])-float(g[0,0])))
                memory,valid=advance_memory(raw,data['rotation'][idx:idx+1],state[:,13],torch.ones(1,dtype=torch.bool));steps+=1
    assert maximum<1e-5,maximum
    meta={'network':'gate_only_24_8_1_sigmoid','model_sha256':sha(out/'net.onnx'),
          'predictors':{k:{'path':cfg[k],'sha256':sha(ROOT/cfg[k]/'net.onnx')} for k in ('v10','v11')},
          'gate_parameters':209,'semantic_dense_macs_total':3016,'parameters_total':3165,
          'candidate_velocity_clip_m_s':2.,'runtime_rotation':'unchanged calibrated gyro',
          'normalization_source':cfg['v10']+'/meta.json','onnx_gate_parity':{'maximum_difference':maximum,'validation_steps':steps,'random_probes':128}}
    write(out/'meta.json',meta);return meta

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['constant','adaptive','adaptive-repeat'],required=True);args=ap.parse_args()
    cfg=json.loads((ROOT/'configs/memory-gate-v12.json').read_text());tc=cfg['training']
    out=ROOT/'models'/('gate-v12-'+args.arm);out.mkdir()
    torch.manual_seed(tc['seed']);np.random.seed(tc['seed']);torch.set_num_threads(tc['threads']);torch.use_deterministic_algorithms(True)
    net=GatedMotion(ROOT,cfg,constant=args.arm=='constant')
    frozen={k:weights_hash(getattr(net,k)) for k in ('base','memory')}
    frozen_paths=[cfg[k]+'/net.onnx' for k in ('v10','v11')]+[cfg[k]+'/meta.json' for k in ('v10','v11')]
    source_paths=['configs/memory-gate-v12.json','configs/long-validation-v12.json','training/memory_gate.py','scripts/train_memory_gate.py',
                  'training/velocity_memory.py','training/motion_model.py','scripts/train_velocity_memory.py']
    provenance={p:sha(ROOT/p) for p in source_paths+frozen_paths}
    previous=json.loads((ROOT/'data/velocity-memory-v11/audit.json').read_text())
    for split,r in previous['splits'].items():
        assert sha(ROOT/cfg['cache']/(split+'.npz'))==r['cache_sha256']
        assert sha(ROOT/'data/velocity-memory-v11'/(split+'.npz'))==r['history_sha256']
    assert sha(ROOT/cfg['sampling_plan'])==previous['sampling_plan_sha256']
    train=load('train',cfg);validation=load('validation',cfg);plan=np.load(ROOT/cfg['sampling_plan'])['order']
    assert plan.shape==(40,2388)
    gate_params=[p for p in net.parameters() if p.requires_grad]
    count=sum(p.numel() for p in gate_params);assert count==(1 if args.arm=='constant' else 209)
    opt=torch.optim.Adam(gate_params,lr=tc['lr']);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,tc['epochs'])
    initial=validate(net,validation);references={str(g):validate(net,validation,g) for g in (0,1)}
    initial_hash=weights_hash(net.gate);history=[];bestloss=float('inf');bestepoch=None;best=None;started=time.perf_counter()
    print('INITIAL',args.arm,initial,'REFERENCES',references,flush=True)
    for epoch,order in enumerate(plan):
        total=0.;net.train()
        for b in range(0,len(order),tc['batch']):
            targets=torch.from_numpy(order[b:b+tc['batch']].copy()).long()
            raw,g=sequence(net,train,train['indices'][targets],train['active'][targets])
            loss=F.smooth_l1_loss((raw-net.target_mean)/net.target_std,(train['target'][targets]-net.target_mean)/net.target_std)
            loss+=tc['warp_weight']*photometric_loss(train['images'][targets],train['depth'][targets],train['intrinsics'][targets],
                                                   raw,train['state'][targets,:3],train['visibility'][targets])
            if not torch.isfinite(loss):raise RuntimeError('nonfinite gate loss')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(gate_params,5.);opt.step()
            total+=float(loss.detach())*len(targets)
        scheduler.step();net.eval();metrics=validate(net,validation)
        assert all(weights_hash(getattr(net,k))==v for k,v in frozen.items())
        row={'epoch':epoch+1,'train_objective':total/len(order),**metrics};history.append(row)
        if metrics['pose_loss']<bestloss:bestloss=metrics['pose_loss'];bestepoch=epoch+1;best=copy.deepcopy(net.gate.state_dict())
        write(out/'history.json',history);print(args.arm,json.dumps(row),flush=True)
    torch.save(net.gate.state_dict(),out/'final.pt');net.gate.load_state_dict(best)
    meta=export(net,out,cfg,validation)
    assert all(sha(ROOT/p)==h for p,h in provenance.items())
    run={'arm':args.arm,'trainable_parameters':count,'steps':1520,'settings':tc,'best_epoch':bestepoch,
         'initial_gate_sha256':initial_hash,'initial_validation':initial,'endpoint_validation_references':references,
         'best_validation':validate(net,validation),'source_sha256':provenance,'frozen_predictor_tensor_sha256':frozen,
         'predictors_unchanged':True,'sampling_plan_sha256':previous['sampling_plan_sha256'],
         'cache_sha256':{s:r['cache_sha256'] for s,r in previous['splits'].items()},'model_sha256':meta['model_sha256'],
         'selection':cfg['selection'],'elapsed_s':time.perf_counter()-started}
    write(out/'run.json',run);print('COMPLETE',args.arm,bestepoch,run['best_validation'],flush=True)

if __name__=='__main__':main()
