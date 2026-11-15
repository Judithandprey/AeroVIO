"""Matched compact continuation with self-predicted history and full-episode validation."""
import argparse, copy, hashlib, json, math, sys, time
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.velocity_memory import VelocityMemoryNet, memory_input, advance_memory
from training.motion_model import photometric_loss

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def load(split,cfg):
    with np.load(ROOT/cfg['cache']/(split+'.npz')) as z:
        data={k:torch.from_numpy(z[k].copy()) for k in z.files}
    with np.load(ROOT/'data/velocity-memory-v11'/(split+'.npz')) as z:
        data.update({k:torch.from_numpy(z[k].copy()) for k in z.files})
    return data

def sequence(net,data,indices,active,norm,feedback):
    mean,std,tm,ts=norm
    memory=torch.zeros(len(indices),3);valid=torch.zeros(len(indices),dtype=torch.bool)
    for t in range(indices.shape[1]):
        idx=indices[:,t];state=data['state'][idx]
        inputs=memory_input(state,memory,valid,mean,std,feedback)
        pred=net(inputs);raw=pred*ts+tm
        memory,valid=advance_memory(raw,data['rotation'][idx],state[:,13],active[:,t])
    return pred

@torch.no_grad()
def validation(net,data,norm,feedback):
    ids=data['episode_id'];mean,std,tm,ts=norm;predictions=torch.zeros_like(data['target'])
    for eid in ids.unique():
        rows=torch.where(ids==eid)[0];memory=torch.zeros(1,3);valid=torch.zeros(1,dtype=torch.bool)
        for idx in rows:
            state=data['state'][idx:idx+1]
            pred=net(memory_input(state,memory,valid,mean,std,feedback));predictions[idx]=pred[0]
            memory,valid=advance_memory(pred*ts+tm,data['rotation'][idx:idx+1],state[:,13],torch.ones(1,dtype=torch.bool))
    loss=float(F.smooth_l1_loss(predictions,(data['target']-tm)/ts))
    raw=predictions*ts+tm
    return {'pose_loss':loss,'translation_rmse_m':float((raw[:,:3]-data['target'][:,:3]).square().sum(1).mean().sqrt()),
            'rotation_residual_rmse_rad':float((raw[:,3:]-data['target'][:,3:]).square().sum(1).mean().sqrt())}

def export(net,out,meta,feedback,validation_data,norm):
    torch.save(net.state_dict(),out/'best.pt')
    torch.onnx.export(net,torch.zeros(1,18),str(out/'net.onnx'),input_names=['state'],output_names=['motion'],opset_version=13,dynamo=False)
    cvnet=cv2.dnn.readNetFromONNX(str(out/'net.onnx'));rng=torch.Generator().manual_seed(17029)
    probes=torch.randn(128,18,generator=rng);probes[:,17]=(probes[:,17]>0).float()
    maxdiff=0.
    with torch.no_grad():
        for row in probes:
            cvnet.setInput(row[None].numpy());actual=cvnet.forward()
            maxdiff=max(maxdiff,float(np.abs(actual-net(row[None]).numpy()).max()))
        # Autoregressive ONNX/Torch parity over all original validation episodes.
        mean,std,tm,ts=norm;seqdiff=0.
        for eid in validation_data['episode_id'].unique():
            m1=torch.zeros(1,3);m2=m1.clone();v1=torch.zeros(1,dtype=torch.bool);v2=v1.clone()
            for idx in torch.where(validation_data['episode_id']==eid)[0]:
                state=validation_data['state'][idx:idx+1];rot=validation_data['rotation'][idx:idx+1]
                p1=net(memory_input(state,m1,v1,mean,std,feedback))
                cvnet.setInput(memory_input(state,m2,v2,mean,std,feedback).numpy());p2=torch.from_numpy(cvnet.forward())
                seqdiff=max(seqdiff,float((p1-p2).abs().max()))
                m1,v1=advance_memory(p1*ts+tm,rot,state[:,13],torch.ones(1,dtype=torch.bool))
                m2,v2=advance_memory(p2*ts+tm,rot,state[:,13],torch.ones(1,dtype=torch.bool))
    assert maxdiff<1e-5 and seqdiff<1e-4,(maxdiff,seqdiff)
    result={**meta,'state_mean':meta['state_mean']+[0.]*4,'state_std':meta['state_std']+[1.]*4,
            'network_inputs':'state','history_enabled':feedback,'velocity_clip_m_s':2.,'max_dt_s':.2,
            'model_sha256':sha(out/'net.onnx'),'source_model_sha256':meta['model_sha256']}
    write(out/'meta.json',result)
    return {'random_probe_max_normalized_difference':maxdiff,'closed_loop_max_normalized_difference':seqdiff,
            'validation_pairs':len(validation_data['state']),'random_probes':len(probes)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['no-history','feedback','feedback-repeat'],required=True)
    args=ap.parse_args();feedback=args.arm!='no-history'
    cfg=json.loads((ROOT/'configs/velocity-memory-v11.json').read_text());tc=cfg['training']
    audit=json.loads((ROOT/'data/velocity-memory-v11/audit.json').read_text())
    for name,digest in audit['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for split,row in audit['splits'].items():
        assert sha(ROOT/cfg['cache']/(split+'.npz'))==row['cache_sha256']
        assert sha(ROOT/'data/velocity-memory-v11'/(split+'.npz'))==row['history_sha256']
    assert sha(ROOT/cfg['sampling_plan'])==audit['sampling_plan_sha256']
    out=ROOT/'models'/('velocity-v11-'+args.arm);out.mkdir()
    torch.manual_seed(tc['seed']);np.random.seed(tc['seed']);torch.set_num_threads(tc['threads']);torch.use_deterministic_algorithms(True)
    parent=ROOT/cfg['parent_model'];meta=json.loads((parent/'meta.json').read_text())
    assert sha(parent/'net.onnx')==meta['model_sha256']
    weights=torch.load(parent/'best.pt',weights_only=True);net=VelocityMemoryNet(weights)
    initialhash=hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in net.state_dict().values())).hexdigest()
    norm=tuple(torch.tensor(meta[k]) for k in ('state_mean','state_std','target_mean','target_std'))
    train=load('train',cfg);val=load('validation',cfg);plan=np.load(ROOT/cfg['sampling_plan'])['order']
    opt=torch.optim.Adam(net.parameters(),lr=tc['lr']);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,tc['epochs'])
    start=time.perf_counter();initial=validation(net,val,norm,feedback)
    bestloss=initial['pose_loss'];bestepoch=0;best=copy.deepcopy(net.state_dict());history=[]
    print('INITIAL',args.arm,initial,flush=True)
    for epoch,order in enumerate(plan):
        net.train();total=0.
        for startidx in range(0,len(order),tc['batch']):
            targets=torch.from_numpy(order[startidx:startidx+tc['batch']].copy()).long()
            prediction=sequence(net,train,train['indices'][targets],train['active'][targets],norm,feedback)
            mean,std,tm,ts=norm
            loss=F.smooth_l1_loss(prediction,(train['target'][targets]-tm)/ts)
            loss=loss+tc['warp_weight']*photometric_loss(train['images'][targets],train['depth'][targets],train['intrinsics'][targets],
                                                       prediction*ts+tm,train['state'][targets,:3],train['visibility'][targets])
            if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
            total+=float(loss.detach())*len(targets)
        scheduler.step();net.eval();metrics=validation(net,val,norm,feedback)
        row={'epoch':epoch+1,'train_objective':total/len(order),**metrics,'elapsed_s':time.perf_counter()-start};history.append(row)
        if metrics['pose_loss']<bestloss:bestloss=metrics['pose_loss'];bestepoch=epoch+1;best=copy.deepcopy(net.state_dict())
        write(out/'history.json',history);print(args.arm,json.dumps(row),flush=True)
    net.load_state_dict(best);net.eval();parity=export(net,out,meta,feedback,val,norm)
    run={'arm':args.arm,'feedback':feedback,'initial_weights_sha256':initialhash,'parent_model_sha256':meta['model_sha256'],
         'parameters':sum(p.numel() for p in net.parameters()),'settings':tc,'steps':tc['epochs']*math.ceil(len(plan[0])/tc['batch']),
         'sampling_plan_sha256':audit['sampling_plan_sha256'],'preparation_audit_sha256':sha(ROOT/'data/velocity-memory-v11/audit.json'),
         'model_sha256':sha(out/'net.onnx'),'best_epoch':bestepoch,'best_validation':validation(net,val,norm,feedback),
         'initial_validation':initial,'selection':'minimum full original validation-episode normalized pose loss, epoch 0 included',
         'parity':parity,'elapsed_s':time.perf_counter()-start,'source_sha256':audit['source_sha256']}
    write(out/'run.json',run);print('COMPLETE',args.arm,bestepoch,run['best_validation'],flush=True)

if __name__=='__main__':main()
