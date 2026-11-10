"""Independent geometry and checkpoint guards for rotation-focused training."""
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.rotation_focus import (
    eligible_checkpoint, geodesic_angle, rollout_metrics, rotation_auxiliary_loss,
)


class RotationFocusTests(unittest.TestCase):
    def test_noncommuting_gyro_residual_order_matches_reference(self):
        gyro = np.array([0.3, -0.2, 0.1])
        correction = np.array([-0.1, 0.2, 0.25])
        rg, rc = cv2.Rodrigues(gyro)[0], cv2.Rodrigues(correction)[0]
        state = torch.zeros(1, 14, dtype=torch.float64)
        state[0, :3] = torch.from_numpy(gyro)
        raw = torch.tensor([[0, 0, 0, *correction]], dtype=torch.float64)
        good = rotation_auxiliary_loss(raw, state, torch.from_numpy((rg @ rc)[None]), .1)
        wrong = rotation_auxiliary_loss(raw, state, torch.from_numpy((rc @ rg)[None]), .1)
        self.assertLess(float(good), 1e-24)
        self.assertGreater(float(wrong), .1)

    def test_auxiliary_gradient_near_zero_and_translation_is_unused(self):
        state = torch.zeros(1, 14, dtype=torch.float64)
        truth = torch.from_numpy(cv2.Rodrigues(np.array([0.003, -.002, .004]))[0][None])
        for epsilon in [0., 1e-10, 1e-6]:
            raw = torch.full((1, 6), epsilon, dtype=torch.float64, requires_grad=True)
            loss = rotation_auxiliary_loss(raw, state, truth, .01)
            gradient, = torch.autograd.grad(loss, raw)
            self.assertTrue(torch.isfinite(gradient).all())
            torch.testing.assert_close(gradient[:, :3], torch.zeros_like(gradient[:, :3]), atol=0, rtol=0)
            self.assertGreater(float(gradient[:, 3:].abs().min()), 1)
        raw = torch.zeros(1, 6, dtype=torch.float64, requires_grad=True)
        loss = rotation_auxiliary_loss(raw, state, torch.eye(3, dtype=torch.float64)[None], .01)
        loss.backward()
        self.assertEqual(float(loss.detach()), 0)
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertEqual(float(raw.grad.abs().max()), 0)

    def test_geodesic_is_rotation_angle_not_component_difference(self):
        a = cv2.Rodrigues(np.array([np.pi / 2, 0., 0.]))[0]
        b = cv2.Rodrigues(np.array([0., np.pi / 2, 0.]))[0]
        angle = geodesic_angle(torch.from_numpy(a[None]), torch.from_numpy(b[None]))
        self.assertAlmostEqual(float(angle), 2 * np.pi / 3, places=12)
        same = geodesic_angle(torch.from_numpy(a[None]), torch.from_numpy(a[None]))
        self.assertEqual(float(same), 0)

    def test_translation_composes_before_rotation_and_initial_frame_counts(self):
        raw = np.array([[1, 0, 0, 0, 0, np.pi / 2], [1, 0, 0, 0, 0, np.pi / 2]])
        target = raw.copy()
        target[:, 3:] = 0
        result = rollout_metrics(raw, target, np.zeros((2, 14)), np.array([0, 0]))
        self.assertEqual(result['pair_translation_rmse_m'], 0)
        self.assertAlmostEqual(result['trajectory_position_rmse_mean_m'], np.sqrt(2 / 3), places=12)
        self.assertAlmostEqual(result['episodes'][0]['final_position_error_m'], np.sqrt(2), places=12)
        self.assertAlmostEqual(result['rotation_rmse_mean_rad'], np.pi / 2 * np.sqrt(5 / 3), places=12)

    def test_episode_resets_and_aggregate_weights_episodes_equally(self):
        raw = np.zeros((4, 6))
        raw[:, 5] = .1
        result = rollout_metrics(raw, np.zeros_like(raw), np.zeros((4, 14)), np.array([2, 5, 5, 5]))
        expected = (.1 / np.sqrt(2) + .1 * np.sqrt(14 / 4)) / 2
        self.assertAlmostEqual(result['rotation_rmse_mean_rad'], expected, places=12)
        self.assertAlmostEqual(result['episodes'][1]['final_rotation_error_rad'], .3, places=12)
        self.assertEqual([e['episode_id'] for e in result['episodes']], [2, 5])

    def test_rollout_rejects_empty_nonfinite_and_interleaved_labels(self):
        for labels in [np.array([0, 1, 0]), np.array([-1, 0, 0]), np.array([0, .5, 1])]:
            with self.assertRaises(ValueError):
                rollout_metrics(np.zeros((3, 6)), np.zeros((3, 6)), np.zeros((3, 14)), labels)
        with self.assertRaises(ValueError):
            rollout_metrics(np.zeros((0, 6)), np.zeros((0, 6)), np.zeros((0, 14)), np.zeros(0))
        bad = np.zeros((2, 6)); bad[0, 1] = np.nan
        with self.assertRaises(ValueError):
            rollout_metrics(bad, np.zeros((2, 6)), np.zeros((2, 14)), np.zeros(2))

    def test_both_translation_guards_and_initial_checkpoint_eligibility(self):
        base = {'pair_translation_rmse_m': .01, 'trajectory_position_rmse_mean_m': .1,
                'rotation_rmse_mean_rad': .2}
        self.assertTrue(eligible_checkpoint(base, base))
        for key in ['pair_translation_rmse_m', 'trajectory_position_rmse_mean_m']:
            good = {**base, key: base[key] * 1.05, 'rotation_rmse_mean_rad': .1}
            bad = {**base, key: base[key] * 1.051, 'rotation_rmse_mean_rad': .01}
            self.assertTrue(eligible_checkpoint(good, base))
            self.assertFalse(eligible_checkpoint(bad, base))
        self.assertFalse(eligible_checkpoint({**base, 'rotation_rmse_mean_rad': np.nan}, base))
        with self.assertRaises(ValueError):
            eligible_checkpoint(base, base, .99)


if __name__ == '__main__':
    unittest.main()
