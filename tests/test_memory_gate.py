import json,unittest
from pathlib import Path
import torch
from training.memory_gate import GatedMotion,TrustGate
from training.velocity_memory import memory_input,advance_memory
ROOT=Path(__file__).resolve().parents[1]

class MemoryGateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.net=GatedMotion(ROOT,json.loads((ROOT/'configs/memory-gate-v12.json').read_text()))
        self.state=self.net.state_mean.repeat(4,1).clone();self.state[:,13]=.05
        self.memory=torch.randn(4,3)*.2;self.valid=torch.ones(4,dtype=torch.bool)

    def test_endpoints_and_convex_translation(self):
        x=memory_input(self.state,self.memory,self.valid,self.net.state_mean,self.net.state_std,True)
        base=self.net.base(x[:,:14])*self.net.target_std+self.net.target_mean
        old=self.net.memory(x)*self.net.target_std+self.net.target_mean
        raw0,_,_=self.net(self.state,self.memory,self.valid,0)
        raw1,_,_=self.net(self.state,self.memory,self.valid,1)
        torch.testing.assert_close(raw0,base,atol=0,rtol=0)
        torch.testing.assert_close(raw1[:,:3],old[:,:3],atol=0,rtol=0)
        raw,g,_=self.net(self.state,self.memory,self.valid)
        torch.testing.assert_close(raw[:,:3],.5*(base[:,:3]+old[:,:3]))
        torch.testing.assert_close(raw[:,3:],base[:,3:],atol=0,rtol=0)
        self.assertTrue(bool(((g>=0)&(g<=1)).all()))

    def test_optimizer_cannot_change_predictors(self):
        before={k:v.clone() for k,v in self.net.state_dict().items() if k.startswith(('base.','memory.'))}
        params=[p for p in self.net.parameters() if p.requires_grad]
        self.assertEqual(sum(p.numel() for p in params),209)
        opt=torch.optim.Adam(params,lr=.01)
        raw,_,_=self.net(self.state,self.memory,self.valid);raw.square().sum().backward();opt.step()
        for k,v in before.items():torch.testing.assert_close(v,self.net.state_dict()[k],atol=0,rtol=0)

    def test_history_gradient_and_gap(self):
        raw,_,_=self.net(self.state,self.memory,self.valid);raw.retain_grad()
        m,v=advance_memory(raw,torch.eye(3).repeat(4,1,1),self.state[:,13],self.valid)
        later,_,_=self.net(self.state,m,v);later[:,:3].sum().backward()
        self.assertGreater(float(raw.grad.abs().sum()),0)
        gap=self.state.clone();gap[:,13]=.3
        _,_,a=self.net(gap,self.memory,self.valid);_,_,b=self.net(gap,torch.zeros_like(self.memory),~self.valid)
        torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_constant_gate_has_no_state_dependence(self):
        gate=TrustGate(constant=True)
        self.assertEqual(sum(p.numel() for p in gate.parameters() if p.requires_grad),1)
        with torch.no_grad():gate.layers[2].bias.fill_(.7)
        values=gate(torch.randn(32,24)*100)
        torch.testing.assert_close(values,values[0].expand_as(values),atol=0,rtol=0)

if __name__=='__main__':unittest.main()
