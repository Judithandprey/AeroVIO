"""Independent projection, subpixel labels, masks and descriptor-gradient contracts."""
from pathlib import Path
import sys
import unittest
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.correspondence_supervision import correspondence_targets, correspondence_loss, matching_displacement, volume
from training.feature_correlation import CorrelationMotionNet
from training.motion_model import MotionNet


class CorrespondenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def inputs(self):
        return (torch.full((1, 1, 48, 64), 2., dtype=torch.float64),
                torch.tensor([[20., 24., 31.5, 23.5]], dtype=torch.float64),
                torch.zeros(1, 6, dtype=torch.float64), torch.zeros(1, 3, dtype=torch.float64),
                torch.ones(1, 1, 48, 64, dtype=torch.bool))

    def test_identity_uses_exact_lattice_and_only_full_support(self):
        result = correspondence_targets(*self.inputs())
        self.assertEqual(int(result["valid"].sum()), 140)
        torch.testing.assert_close(result["soft_target"][:, 4], result["valid"].double(), atol=1e-12, rtol=0)
        torch.testing.assert_close(result["displacement"], torch.zeros_like(result["displacement"]), atol=1e-12, rtol=0)
        self.assertFalse(result["valid"][:, 0].any())
        self.assertFalse(result["valid"][:, :, -1].any())

    def test_subpixel_translation_direction_and_bilinear_mass(self):
        depth, k, motion, raw, visibility = self.inputs()
        motion[0, :2] = torch.tensor([.1, -.1], dtype=torch.float64)
        result = correspondence_targets(depth, k, motion, raw, visibility)
        torch.testing.assert_close(result["displacement"][0, :, 5, 6], torch.tensor([.25, -.3], dtype=torch.float64), atol=1e-12, rtol=0)
        expected = torch.zeros(9, dtype=torch.float64)
        expected[1], expected[2], expected[4], expected[5] = .225, .075, .525, .175
        torch.testing.assert_close(result["soft_target"][0, :, 5, 6], expected, atol=1e-12, rtol=0)
        torch.testing.assert_close(result["soft_target"].sum(1), result["valid"].double(), atol=1e-12, rtol=0)

    def test_rotation_translation_matches_independent_opencv_projection(self):
        depth, k, motion, raw, visibility = self.inputs()
        raw[0] = torch.tensor([.01, -.02, .005])
        motion[0] = torch.tensor([.03, -.02, .01, -.005, .012, -.007])
        result = correspondence_targets(depth, k, motion, raw, visibility)
        rotation = cv2.Rodrigues(raw[0].numpy())[0] @ cv2.Rodrigues(motion[0, 3:].numpy())[0]
        camera = np.array([[20., 0, 31.5], [0, 24., 23.5], [0, 0, 1.]])
        x, y = 24., 20.
        point = np.array([[(x-31.5)*2/20, (y-23.5)*2/24, 2.]])
        projected, _ = cv2.projectPoints(point, cv2.Rodrigues(rotation)[0], motion[0, :3].numpy(), camera, None)
        expected = (projected.reshape(2)-[x, y])/4
        np.testing.assert_allclose(result["displacement"][0, :, 5, 6].numpy(), expected, atol=1e-12)

    def test_visibility_out_of_window_and_behind_camera_are_excluded(self):
        for reason in ("visibility", "outside", "behind", "depth"):
            depth, k, motion, raw, visibility = self.inputs()
            if reason == "visibility": visibility.zero_()
            if reason == "outside": motion[0, 0] = 2
            if reason == "behind": motion[0, 2] = -3
            if reason == "depth": depth.zero_()
            result = correspondence_targets(depth, k, motion, raw, visibility)
            self.assertFalse(result["valid"].any())
            self.assertTrue(all(torch.isfinite(v).all() for v in result.values()))
            self.assertEqual(float(result["soft_target"].sum()), 0.)

    def test_soft_argmax_recovers_subpixel_label_expectation(self):
        depth, k, motion, raw, visibility = self.inputs(); motion[0, 0] = .13
        target = correspondence_targets(depth, k, motion, raw, visibility)
        logits = .1 * target["soft_target"].clamp_min(1e-15).log()
        predicted = matching_displacement(logits)
        torch.testing.assert_close(predicted[:, :, 1:-1, 1:-1], target["displacement"][:, :, 1:-1, 1:-1], atol=1e-12, rtol=0)

    def test_invalid_locations_have_no_gradient_and_empty_mask_is_finite(self):
        target = correspondence_targets(*self.inputs())
        for empty in (False, True):
            scores = torch.randn(1, 9, 12, 16, dtype=torch.float64, requires_grad=True)
            mask = torch.zeros_like(target["valid"]) if empty else target["valid"]
            loss = correspondence_loss(scores, target["soft_target"], mask)
            loss.backward()
            self.assertTrue(torch.isfinite(scores.grad).all())
            self.assertEqual(float(scores.grad.masked_select(~mask[:, None].expand_as(scores)).abs().sum()), 0.)
            if empty: self.assertEqual(float(loss), 0.)
            else: self.assertGreater(float(scores.grad.abs().sum()), 0.)

    def test_matching_supervision_trains_encoder_with_zero_rotation_head(self):
        torch.manual_seed(17)
        net = CorrelationMotionNet(MotionNet(), [0]*6, [1]*6)
        image = torch.rand(1, 2, 48, 64, requires_grad=True)
        target = correspondence_targets(*self.inputs())
        loss = correspondence_loss(volume(net, image), target["soft_target"].float(), target["valid"])
        loss.backward()
        self.assertGreater(float(net.encoder[0].weight.grad.abs().sum()), 0.)
        self.assertTrue(all(float(image.grad[:, i].abs().sum()) > 0 for i in (0, 1)))
        self.assertTrue(all(p.grad is None for p in net.base.parameters()))


if __name__ == "__main__": unittest.main()
