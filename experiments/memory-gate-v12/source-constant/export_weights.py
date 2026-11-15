"""Emit immutable predictor and gate weights with explicit model provenance."""
import argparse,hashlib,json
from pathlib import Path
import cv2
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);args=ap.parse_args()
    paths=json.loads(Path(__file__).with_name('model-paths.json').read_text())
    lines=['#pragma once','#include <array>']
    def literals(row):return ', '.join(float(v).hex()+'f' for v in row)
    def vector(name,row):lines.append(f'constexpr std::array<float,{len(row)}> {name} = {{{literals(row)}}};')
    for label,inputs,hidden,outputs in [('base',14,64,6),('memory',18,64,6),('gate',24,8,1)]:
        model=Path(paths[label]);meta=json.loads((model/'meta.json').read_text());blob=(model/'net.onnx').read_bytes()
        assert hashlib.sha256(blob).hexdigest()==meta['model_sha256']
        net=cv2.dnn.readNetFromONNX(np.frombuffer(blob,np.uint8))
        layers=[net.getLayer(net.getLayerId(n)) for n in net.getLayerNames()]
        assert [l.type for l in layers]==(['Gemm','ReLU','Gemm','Sigmoid'] if label=='gate' else ['Gemm','ReLU','Gemm'])
        lines+=['namespace '+label+'_weights {','// ONNX '+meta['model_sha256']]
        if label!='gate':
            for key in ('state_mean','state_std','target_mean','target_std'):vector(key,np.asarray(meta[key],np.float32))
        for number,layer,ni,no in [(1,layers[0],inputs,hidden),(2,layers[2],hidden,outputs)]:
            w,b=layer.blobs;assert w.shape==(no,ni) and b.size==no
            w=np.pad(w.T,((0,0),(0,-no%4)));b=np.pad(b.reshape(-1),(0,-no%4))
            vector('bias'+str(number),b)
            lines.append(f'constexpr std::array<std::array<float,{len(b)}>,{ni}> weight{number} = {{{{')
            lines.extend('    {'+literals(row)+'},' for row in w);lines+=['}};']
        lines+=['}']
    args.output.write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
