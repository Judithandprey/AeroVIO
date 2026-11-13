import unittest
import numpy as np
from training.joint_sampling import cell_quotas,make_order,audit_order

class JointSamplingTest(unittest.TestCase):
    def test_enforces_speed_marginal_and_rare_cell_cap(self):
        counts=np.array([[761,179,83,165],[277,81,16,20],[466,72,0,9],[258,1,0,0]])
        quotas=cell_quotas(counts,[597]*4,4)
        np.testing.assert_array_equal(quotas.sum(axis=1),[597]*4)
        self.assertEqual(quotas[3,1],4)
        self.assertTrue(np.all(quotas[counts==0]==0))
        self.assertTrue(np.all(quotas<=counts*4))
        speed=[];turn=[]
        for s in range(4):
            for t in range(4):speed.extend([s]*counts[s,t]);turn.extend([t]*counts[s,t])
        speed,turn=np.array(speed),np.array(turn)
        order=make_order(speed,turn,quotas,0,17)
        self.assertLessEqual(audit_order(order,speed,turn,quotas,4),4)
        self.assertEqual(np.mean(speed[order]==0),.25)
        np.testing.assert_array_equal(order,make_order(speed,turn,quotas,0,17))
        self.assertFalse(np.array_equal(order,make_order(speed,turn,quotas,1,17)))

    def test_refuses_unsupported_speed_marginal(self):
        with self.assertRaises(ValueError):cell_quotas([[1,0],[20,20]],[5,5],4)

    def test_gate_catches_corrupt_or_overweighted_plan(self):
        speed=np.array([0,0,1,1]);turn=np.zeros(4,int);q=np.array([[2],[2]])
        for order in (np.array([0,0,0,3]),np.array([0,1,2,4]),np.array([0.,1.,2.,3.])):
            with self.assertRaises(ValueError):audit_order(order,speed,turn,q,1)

if __name__=='__main__':unittest.main()
