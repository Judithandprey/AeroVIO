"""Check C++/ONNX predictions and recurrent transport independently from training."""
import hashlib,json
from pathlib import Path
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/velocity-memory-v11';ARCHIVE=ROOT/'experiments/velocity-memory-v11'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    source=OUT/'check-feedback-final-parity/stderr.txt'
    lines=[line.split()[1:] for line in source.read_text().splitlines() if line.startswith('MEMORY ')]
    data=np.asarray(lines,dtype=np.float64);assert data.shape==(79,36),data.shape
    model=ROOT/'models/velocity-v11-feedback-final';meta=json.loads((model/'meta.json').read_text());net=cv2.dnn.readNetFromONNX(str(model/'net.onnx'))
    mean=np.array(meta['state_mean'],dtype=np.float32);std=np.array(meta['state_std'],dtype=np.float32)
    tm=np.array(meta['target_mean'],dtype=np.float32);ts=np.array(meta['target_std'],dtype=np.float32)
    memory=np.zeros(3);previous_valid=False;max_pred=max_transport=max_input=0.
    for row in data:
        x=row[:18];raw=row[18:24];actual_next=row[24:27];rotation=row[27:].reshape(3,3);dt=x[13]
        valid=previous_valid and 0<dt<=.2
        expected_input=np.r_[memory if valid else np.zeros(3),float(valid)]
        max_input=max(max_input,float(np.max(np.abs(x[14:]-expected_input))))
        net.setInput(((x.astype(np.float32)-mean)/std)[None]);prediction=net.forward()[0]*ts+tm
        max_pred=max(max_pred,float(np.max(np.abs(prediction-raw))))
        expected_next=np.clip(rotation.T@raw[:3]/dt,-2,2).astype(np.float32)
        max_transport=max(max_transport,float(np.max(np.abs(expected_next-actual_next))))
        memory=actual_next;previous_valid=0<dt<=.2
    assert max_pred<1e-5 and max_transport<1e-5 and max_input<1e-6,(max_pred,max_transport,max_input)
    record={'frames':80,'network_steps':79,'max_motion_difference_cpp_onnx':max_pred,
        'max_memory_transport_difference':max_transport,'max_causal_input_difference':max_input,
        'log_sha256':sha(source),'model_sha256':sha(model/'net.onnx'),
        'diagnostic_guest_not_used_for_timing':True,'history_is_pre_visual_translation':True}
    (ARCHIVE/'runtime-parity.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
if __name__=='__main__':main()
