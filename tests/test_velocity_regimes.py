import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_velocity_memory import regime_metrics,combine_regimes

def states(n):
    x=np.zeros((n,13));x[:,6]=1.;return x

class VelocityRegimeTests(unittest.TestCase):
    def test_stationary_drift_is_detected(self):
        times=np.arange(50)*.05;gt=states(50);pred=gt.copy();pred[:,0]=times*.2
        r=regime_metrics(gt,pred,times)['regimes']
        self.assertEqual(r['stationary']['windows'],30)
        self.assertAlmostEqual(r['stationary']['length_rmse_m'],.2)
        self.assertIsNone(r['slow_straight']['length_rmse_m'])

    def test_stop_after_cruise_and_censored_settling(self):
        times=np.arange(60)*.05;gt=states(60);gt[:,0]=np.minimum(times,.8)*.2
        pred=gt.copy();pred[:,0]=times*.2
        report=regime_metrics(gt,pred,times)
        self.assertEqual(len(report['stop_events']),1)
        self.assertFalse(report['stop_events'][0]['settled'])
        self.assertAlmostEqual(report['stop_events'][0]['displacement_error_m'],.2)
        result=combine_regimes([{'motion_diagnostics':report}])
        self.assertIsNone(result['stops']['mean_settling_time_s_among_settled'])

    def test_small_net_motion_does_not_imply_stationary(self):
        times=np.arange(41)*.05;gt=states(41);gt[:,0]=.2*np.sin(times*2*np.pi)
        r=regime_metrics(gt,gt,times)['regimes']
        self.assertEqual(r['stationary']['windows'],0)
        self.assertGreater(r['small_net_but_moving']['windows'],0)
        self.assertEqual(r['small_net_but_moving']['length_rmse_m'],0.)
if __name__=='__main__':unittest.main()
