"""Synthetic metric contracts: distinguish absolute length from aligned shape."""
from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_joint_balance import pooled_distance

class JointMetricTest(unittest.TestCase):
    def trace(self):
        t=np.arange(81)*.05
        gt=np.zeros((len(t),13));gt[:,0]=.3*t;gt[:,6]=1
        return t,gt

    def test_half_scale_keeps_direction_but_has_metric_length_error(self):
        t,gt=self.trace();out=gt.copy();out[:,:3]*=.5
        r=pooled_distance([gt],[out],[t])
        self.assertAlmostEqual(r['all']['length_rmse_m'],.15)
        self.assertAlmostEqual(r['all']['median_predicted_true_ratio'],.5)
        self.assertAlmostEqual(r['all']['median_local_direction_deg'],0.)
        self.assertEqual(set(r['speed_bins']),{'0.25_to_0.5'})
        self.assertEqual(r['all']['windows'],61)

    def test_degenerate_prediction_retains_distance_but_omits_angle(self):
        t,gt=self.trace();out=gt.copy();out[:,:3]=0
        r=pooled_distance([gt],[out],[t])['all']
        self.assertAlmostEqual(r['length_rmse_m'],.3)
        self.assertIsNone(r['median_predicted_true_ratio'])
        self.assertIsNone(r['median_local_direction_deg'])

if __name__=='__main__':unittest.main()
