import unittest
import numpy as np
from training.motion_sampling import classify_motion, balanced_order, validate_plan


SETTINGS = {'startup_seconds':1., 'low_speed_m_s':.1, 'turn_deg_s':15.,
            'tangential_acceleration_m_s2':.2, 'speed_smoothing_frames':5}


class MotionSamplingTest(unittest.TestCase):
    def test_startup_precedes_turn_then_stationary_and_cruise(self):
        times = np.arange(101)*.05
        group,_ = classify_motion(times,np.full(100,.3),np.zeros(100),SETTINGS)
        np.testing.assert_array_equal(group[:20],0)
        np.testing.assert_array_equal(group[20:],1)
        group,_ = classify_motion(times,np.zeros(100),np.full(100,20.),SETTINGS)
        np.testing.assert_array_equal(group[:20],0)
        np.testing.assert_array_equal(group[20:],2)

    def test_acceleration_is_not_cruise(self):
        times = np.arange(101)*.05
        group,_ = classify_motion(times,times[1:]*.5,np.zeros(100),SETTINGS)
        np.testing.assert_array_equal(group[25:-5],0)

    def test_balances_without_extra_optimizer_examples(self):
        labels = np.repeat([0,1,2],[20,8,2])
        a = balanced_order(labels,0,17)
        np.testing.assert_array_equal(np.bincount(labels[a]),[10,10,10])
        self.assertEqual(len(a),len(labels))
        np.testing.assert_array_equal(a,balanced_order(labels,0,17))
        self.assertFalse(np.array_equal(a,balanced_order(labels,1,17)))
        self.assertTrue({28,29}.issubset(set(a)))

    def test_rejects_missing_group_and_invalid_plan(self):
        with self.assertRaises(ValueError): balanced_order(np.zeros(30,int),0,17)
        for order in (np.array([[0,-1,1]]),np.array([[0,1,3]]),np.array([[0.,1.,2.]])):
            with self.assertRaises(ValueError): validate_plan(order,1,3)
        np.testing.assert_array_equal(validate_plan(np.array([[2,2,0]]),1,3),[[2,2,0]])


if __name__ == '__main__': unittest.main()
