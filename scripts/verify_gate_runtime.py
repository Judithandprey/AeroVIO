"""Compare native logged features, gate and recurrent state to archived ONNX."""
import hashlib,json
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/memory-gate-v12';A=ROOT/'experiments/memory-gate-v12'

def main():
    cfg=json.loads((ROOT/'configs/memory-gate-v12.json').read_text())
    meta=json.loads((ROOT/cfg['v10']/'meta.json').read_text())
    mean=np.r_[np.array(meta['state_mean'],np.float32),np.zeros(4,np.float32)]
    std=np.r_[np.array(meta['state_std'],np.float32),np.ones(4,np.float32)]
    tm=np.array(meta['target_mean'],np.float32);ts=np.array(meta['target_std'],np.float32)
    models=[cv2.dnn.readNetFromONNX(str(p/'net.onnx')) for p in (ROOT/cfg['v10'],ROOT/cfg['v11'],A/'adaptive')]
    path=OUT/'check-adaptive-parity/stderr.txt'
    rows=np.asarray([line.split()[1:] for line in path.read_text().splitlines() if line.startswith('GATE ')],np.float64)
    assert rows.shape==(79,61),rows.shape
    errors={k:0. for k in ('gate_features','gate_output','motion','memory_transport','causal_input')}
    memory=np.zeros(3);valid=False
    for row in rows:
        x=row[:18].astype(np.float32);features=row[18:42];g=row[42];raw=row[43:49];next_memory=row[49:52];rot=row[52:].reshape(3,3)
        expected=np.r_[memory if valid else np.zeros(3),float(valid)]
        errors['causal_input']=max(errors['causal_input'],float(np.abs(x[14:]-expected).max()))
        xx=(x-mean)/std;models[0].setInput(xx[None,:14]);base=models[0].forward()[0]*ts+tm
        models[1].setInput(xx[None]);candidate=models[1].forward()[0]*ts+tm
        f=np.r_[xx,np.clip(base[:3]/x[13],-2,2),np.clip(candidate[:3]/x[13],-2,2)].astype(np.float32)
        errors['gate_features']=max(errors['gate_features'],float(np.abs(f-features).max()))
        models[2].setInput(f[None]);predicted_g=float(models[2].forward()[0,0])
        errors['gate_output']=max(errors['gate_output'],abs(predicted_g-g))
        motion=base.copy();motion[:3]=(1-predicted_g)*base[:3]+predicted_g*candidate[:3]
        errors['motion']=max(errors['motion'],float(np.abs(motion-raw).max()))
        expected_next=np.clip(rot.T@raw[:3]/float(x[13]),-2,2)
        errors['memory_transport']=max(errors['memory_transport'],float(np.abs(expected_next-next_memory).max()))
        memory=next_memory;valid=0<x[13]<=.2
    assert max(errors.values())<1e-5,errors
    result={'network_steps':len(rows),'maximum_differences':errors,'log_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'diagnostic_guest_excluded_from_timing':True}
    (A/'runtime-parity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
