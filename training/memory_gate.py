"""Train only a scalar trust gate between two immutable motion predictors."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from training.velocity_memory import memory_input


def frozen_compact(path, inputs):
    net=cv2.dnn.readNetFromONNX(str(Path(path)/'net.onnx'))
    layers=[net.getLayer(net.getLayerId(n)) for n in net.getLayerNames()]
    assert [l.type for l in layers]==['Gemm','ReLU','Gemm']
    result=nn.Sequential(nn.Linear(inputs,64),nn.ReLU(),nn.Linear(64,6))
    with torch.no_grad():
        for module,layer in ((result[0],layers[0]),(result[2],layers[2])):
            w,b=layer.blobs;module.weight.copy_(torch.from_numpy(w));module.bias.copy_(torch.from_numpy(b.reshape(-1)))
    return result.requires_grad_(False)


class TrustGate(nn.Module):
    def __init__(self,constant=False):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(24,8),nn.ReLU(),nn.Linear(8,1),nn.Sigmoid())
        nn.init.zeros_(self.layers[2].weight);nn.init.zeros_(self.layers[2].bias)
        if constant:
            with torch.no_grad():
                self.layers[0].weight.zero_();self.layers[0].bias.zero_()
            self.requires_grad_(False);self.layers[2].bias.requires_grad_(True)

    def forward(self,x):return self.layers(x)


class GatedMotion(nn.Module):
    def __init__(self,root,cfg,constant=False):
        super().__init__();root=Path(root)
        self.base=frozen_compact(root/cfg['v10'],14)
        self.memory=frozen_compact(root/cfg['v11'],18)
        self.gate=TrustGate(constant)
        meta=json.loads((root/cfg['v10']/'meta.json').read_text())
        other=json.loads((root/cfg['v11']/'meta.json').read_text())
        for key in ('state_mean','state_std'):
            assert meta[key]==other[key][:14]
        for key in ('target_mean','target_std'):assert meta[key]==other[key]
        for key in ('state_mean','state_std','target_mean','target_std'):
            self.register_buffer(key,torch.tensor(meta[key]))

    def forward(self,state,memory,valid,force=None):
        x=memory_input(state,memory,valid,self.state_mean,self.state_std,True)
        base=self.base(x[:,:14])*self.target_std+self.target_mean
        candidate=self.memory(x)*self.target_std+self.target_mean
        dt=state[:,13:14].clamp_min(1e-6)
        gate_input=torch.cat((x,(base[:,:3]/dt).clamp(-2,2),(candidate[:,:3]/dt).clamp(-2,2)),1)
        g=self.gate(gate_input) if force is None else torch.full_like(dt,float(force))
        translation=(1-g)*base[:,:3]+g*candidate[:,:3]
        raw=torch.cat((translation,base[:,3:]),1)
        return raw,g,gate_input
