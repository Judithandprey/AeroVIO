import json,unittest
from pathlib import Path
import numpy as np
from training.gate_metrics import cruise_metrics
CFG=json.loads((Path(__file__).resolve().parents[1]/'configs/memory-gate-v12.json').read_text())['motion']

class GateMetricTests(unittest.TestCase):
    def trajectory(self):
        t=np.arange(80)*.05;x=np.zeros((80,13));x[:,0]=.3*t;x[:,6]=1
        return t,x

    def test_exact_cruise_and_false_stop(self):
        t,x=self.trajectory();r=cruise_metrics(x,x,t,CFG)
        self.assertEqual(r['windows'],60);self.assertAlmostEqual(r['length_rmse_m'],0)
        p=x.copy();p[:,0]=0;r=cruise_metrics(x,p,t,CFG)
        self.assertEqual(len(r['false_stop_events']),1);self.assertAlmostEqual(r['length_rmse_m'],.3)

    def test_stationary_is_not_cruise(self):
        t,x=self.trajectory();x[:,0]=0;r=cruise_metrics(x,x,t,CFG)
        self.assertEqual(r['windows'],0);self.assertIsNone(r['length_rmse_m'])

    def test_braking_is_not_constant_speed(self):
        t,x=self.trajectory();x[:,0]=.3*t+.1*t*t
        self.assertEqual(cruise_metrics(x,x,t,CFG)['windows'],0)
