"""Check the shuffle intervention and SO(3) probe independently of trained nets."""
import importlib.util
from pathlib import Path
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("visual_probe", ROOT / "scripts/rotation_visual_probe.py")
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class RotationVisualProbeTests(unittest.TestCase):
    def test_shuffle_keeps_whole_pairs_and_states_in_original_order(self):
        images = np.zeros((30, 2, 2, 2), np.float32)
        images[:, 0] = np.arange(30)[:, None, None]
        images[:, 1] = 100 + np.arange(30)[:, None, None]
        states = np.zeros((30, 14), np.float32)
        states[:, 0] = np.arange(30)
        class Recorder:
            def predict(self, pair, state):
                return np.array([pair[0, 0, 0], pair[1, 0, 0], state[0], 0, 0, 0])
        order = PROBE.shuffle_indices(30)
        np.testing.assert_array_equal(order, PROBE.shuffle_indices(30))
        result = PROBE.predict_motion(Recorder(), images, states, order)
        np.testing.assert_array_equal(result[:, 0], order)
        np.testing.assert_array_equal(result[:, 1] - result[:, 0], 100)
        np.testing.assert_array_equal(result[:, 2], np.arange(30))
        self.assertGreater(int((order != np.arange(30)).sum()), 20)

    def test_state_only_predictions_are_invariant_to_shuffle(self):
        class StateOnly:
            def predict(self, pair, state): return state[:6].copy()
        generator = np.random.default_rng(2)
        images, states = generator.normal(size=(20, 2, 3, 3)), generator.normal(size=(20, 14))
        np.testing.assert_array_equal(PROBE.predict_motion(StateOnly(), images, states),
                                      PROBE.predict_motion(StateOnly(), images, states, PROBE.shuffle_indices(20)))

    def test_geodesic_uses_composition_and_retains_tiny_angles(self):
        a = cv2.Rodrigues(np.array([np.pi / 2, 0., 0.]))[0]
        b = cv2.Rodrigues(np.array([0., np.pi / 2, 0.]))[0]
        row = PROBE.error_summary(a[None], a[None], b[None])
        self.assertAlmostEqual(row['paired']['rotation_rmse_deg'], 0)
        self.assertAlmostEqual(row['shuffled']['rotation_rmse_deg'], 120)
        self.assertAlmostEqual(row['paired_shuffled_prediction_difference']['rotation_rms_deg'], 120)
        tiny = cv2.Rodrigues(np.array([1e-7, -2e-7, 3e-7]))[0]
        np.testing.assert_allclose(PROBE.rotation_vectors(tiny[None])[0], [1e-7, -2e-7, 3e-7], atol=1e-14)

    def test_real_and_aux_label_conventions_agree(self):
        state, target = np.zeros((2, 14)), np.zeros((2, 6))
        state[:, :3] = [.2, 0, .1]
        target[:, 3:] = [0, -.3, .1]
        truth = PROBE.true_rotations({'state': state, 'target': target})
        expected = cv2.Rodrigues(state[0, :3])[0] @ cv2.Rodrigues(target[0, 3:])[0]
        np.testing.assert_allclose(truth[0], expected, atol=1e-12)
        aux = {'state': state, 'rotation_gt': truth, 'rotation_residual': target[:, 3:]}
        np.testing.assert_allclose(PROBE.true_rotations(aux), truth)
        aux['rotation_gt'] = np.repeat(np.eye(3)[None], 2, axis=0)
        with self.assertRaises(ValueError): PROBE.true_rotations(aux)

    def test_group_membership_uses_labels_not_predictions(self):
        state, target = np.zeros((4, 14)), np.zeros((4, 6))
        target[:, 3:] = np.radians([[.05, 0, 0], [0, .25, 0], [0, 0, .75], [1, 1, 0]])
        data = {'state': state, 'target': target, 'axis_id': np.arange(4)}
        truth = PROBE.true_rotations(data)
        zero = np.repeat(np.eye(3)[None], 4, axis=0)
        row = PROBE.grouped_errors(data, truth, zero)
        self.assertEqual([v['pairs'] for v in row['by_axis'].values()], [1, 1, 1, 1])
        self.assertEqual(row['by_angle_deg']['[0,0.1)']['pairs'], 1)
        self.assertEqual(row['by_angle_deg']['[1,2)']['pairs'], 1)
        self.assertGreater(row['overall']['shuffled_minus_paired_rmse_deg'], 0)


if __name__ == '__main__':
    unittest.main()
