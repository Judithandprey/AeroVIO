import unittest
import numpy as np
import torch
from training.velocity_memory import VelocityMemoryNet, memory_input, advance_memory, sequence_indices

class VelocityMemoryTests(unittest.TestCase):
    def test_rotating_camera_transports_velocity(self):
        rot=torch.tensor([[[0.,-1,0],[1,0,0],[0,0,1]]])
        raw=torch.tensor([[.05,0,0,0,0,0]])
        v,ok=advance_memory(raw,rot,torch.tensor([.05]),torch.tensor([True]))
        torch.testing.assert_close(v,torch.tensor([[0.,-1.,0.]]));self.assertTrue(ok.item())

    def test_boundaries_and_padding(self):
        ids=np.array([0,0,0,1,1]);idx,active=sequence_indices(ids,[2,3,4],3)
        np.testing.assert_array_equal(idx,[[0,1,2],[3,3,3],[3,3,4]])
        np.testing.assert_array_equal(active,[[True,True,True],[False,False,True],[False,True,True]])

    def test_unknown_is_distinct_from_known_zero(self):
        s=torch.zeros(2,14);s[:,13]=.05;m=torch.zeros(2,3)
        x=memory_input(s,m,torch.tensor([False,True]),torch.zeros(14),torch.ones(14),True)
        self.assertEqual(x[0,17],0);self.assertEqual(x[1,17],1)
        self.assertEqual(memory_input(s,m,torch.ones(2,dtype=torch.bool),torch.zeros(14),torch.ones(14),False)[:,14:].count_nonzero(),0)

    def test_gap_resets_and_clip_is_bounded(self):
        r=torch.eye(3).repeat(2,1,1);raw=torch.ones(2,6)
        v,ok=advance_memory(raw,r,torch.tensor([.05,.3]),torch.tensor([True,True]))
        torch.testing.assert_close(v,torch.tensor([[2.,2.,2.],[0.,0.,0.]]))
        self.assertEqual(ok.tolist(),[True,False])

    def test_parent_function_preserved_and_gradient_reaches_history(self):
        parent=torch.nn.Sequential(torch.nn.Linear(14,64),torch.nn.ReLU(),torch.nn.Linear(64,6))
        weights={'head.'+k:v for k,v in parent.state_dict().items()};net=VelocityMemoryNet(weights)
        x=torch.randn(8,18);torch.testing.assert_close(net(x),parent(x[:,:14]),atol=1e-6,rtol=1e-5)
        self.assertEqual(sum(p.numel() for p in net.parameters()),1606)
        net(x).square().sum().backward();self.assertGreater(float(net.head[0].weight.grad[:,14:].abs().sum()),0)

if __name__=='__main__':unittest.main()
