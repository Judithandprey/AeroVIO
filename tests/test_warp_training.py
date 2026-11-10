"""Independent synthetic geometry checks for training-only view reconstruction."""
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.motion_model import MotionNet, backward_warp, photometric_loss, so3_exp


def camera(width, height, dtype=torch.float64):
    return torch.tensor([[width / 2, height / 2, (width - 1) / 2, (height - 1) / 2]], dtype=dtype)


def ramp(width, height, dtype=torch.float64):
    yy, xx = torch.meshgrid(torch.arange(height, dtype=dtype), torch.arange(width, dtype=dtype), indexing="ij")
    return ((xx + .2 * yy) / (width + .2 * height))[None, None]


class WarpGeometryTests(unittest.TestCase):
    def test_identity_preserves_pixels_and_validity(self):
        source = ramp(32, 24)
        depth = torch.full_like(source, 4)
        motion = torch.zeros(1, 6, dtype=source.dtype)
        image, valid, z, _ = backward_warp(source, depth, camera(32, 24), motion, motion[:, 3:])
        torch.testing.assert_close(image, source, atol=1e-12, rtol=0)
        self.assertTrue(valid.all())
        torch.testing.assert_close(z, depth)

    def test_known_translation_samples_correct_source_direction(self):
        # A camera +0.5 m to the right viewing a plane at z=4 with fx=16
        # must sample source pixels two columns to the right of target pixels.
        source = ramp(32, 24)
        depth = torch.full_like(source, 4)
        motion = torch.tensor([[.5, 0, 0, 0, 0, 0]], dtype=source.dtype)
        image, valid, _, _ = backward_warp(source, depth, camera(32, 24), motion, torch.zeros(1, 3, dtype=source.dtype))
        torch.testing.assert_close(image[:, :, :, :-2], source[:, :, :, 2:], atol=1e-12, rtol=0)
        self.assertTrue(valid[:, :, :, :-2].all())
        self.assertFalse(valid[:, :, :, -2:].any())

    def test_correct_motion_beats_identity_and_reversed_motion(self):
        source = ramp(32, 24)
        target = torch.zeros_like(source)
        target[:, :, :, :-2] = source[:, :, :, 2:]
        images = torch.cat((source, target), dim=1)
        visibility = torch.ones_like(source, dtype=torch.bool)
        visibility[:, :, :, -2:] = False
        depth = torch.full_like(source, 4)
        motion = torch.tensor([[.5, 0, 0, 0, 0, 0]], dtype=source.dtype)
        gyro = torch.zeros(1, 3, dtype=source.dtype)
        good = photometric_loss(images, depth, camera(32, 24), motion, gyro, visibility)
        still = photometric_loss(images, depth, camera(32, 24), torch.zeros_like(motion), gyro, visibility)
        reversed_motion = photometric_loss(images, depth, camera(32, 24), -motion, gyro, visibility)
        self.assertLess(float(good), 1e-12)
        self.assertGreater(float(still), .01)
        self.assertGreater(float(reversed_motion), float(still))

    def test_rotation_composition_matches_independent_opencv_projection(self):
        source = ramp(32, 24)
        depth = torch.full_like(source, 4)
        imu = torch.tensor([[.04, -.02, .01]], dtype=source.dtype)
        motion = torch.tensor([[.1, -.02, .03, -.01, .05, -.02]], dtype=source.dtype)
        _, _, _, grid = backward_warp(source, depth, camera(32, 24), motion, imu)
        u, v = 17, 10
        intrinsic = camera(32, 24)[0].numpy()
        point = np.array([(u - intrinsic[2]) * 4 / intrinsic[0],
                          (v - intrinsic[3]) * 4 / intrinsic[1], 4])
        rotation = cv2.Rodrigues(imu[0].numpy())[0] @ cv2.Rodrigues(motion[0, 3:].numpy())[0]
        previous = rotation @ point + motion[0, :3].numpy()
        expected_pixel = np.array([intrinsic[0] * previous[0] / previous[2] + intrinsic[2],
                                   intrinsic[1] * previous[1] / previous[2] + intrinsic[3]])
        actual_grid = grid[0, v, u].numpy()
        actual_pixel = (actual_grid + 1) * np.array([32, 24]) / 2 - .5
        np.testing.assert_allclose(actual_pixel, expected_pixel, atol=1e-10, rtol=0)

    def test_so3_has_finite_nonzero_gradient_at_zero_and_near_zero(self):
        for scale in [0., 1e-9, 1e-5]:
            vector = torch.tensor([[scale, -scale, scale]], dtype=torch.float64, requires_grad=True)
            matrix = so3_exp(vector)
            objective = matrix[0, 0, 1] + 2 * matrix[0, 1, 2] + 3 * matrix[0, 2, 0]
            gradient, = torch.autograd.grad(objective, vector)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().min()), .5)
        torch.autograd.gradcheck(so3_exp, (torch.zeros(1, 3, dtype=torch.float64, requires_grad=True),), atol=1e-5)

    def test_photo_gradient_reaches_motion_near_zero(self):
        source = ramp(32, 24)
        target = source.clone()
        target[:, :, :, :-1] = source[:, :, :, 1:]
        images = torch.cat((source, target), dim=1)
        visibility = torch.zeros_like(source, dtype=torch.bool)
        visibility[:, :, 2:-2, 2:-2] = True
        motion = torch.zeros(1, 6, dtype=source.dtype, requires_grad=True)
        loss = photometric_loss(images, torch.full_like(source, 4), camera(32, 24), motion,
                                torch.zeros(1, 3, dtype=source.dtype), visibility)
        loss.backward()
        self.assertTrue(torch.isfinite(motion.grad).all())
        self.assertGreater(float(motion.grad[0, 0].abs()), .01)
        self.assertGreater(float(motion.grad[0, 3:].norm()), .01)

    def test_fixed_visibility_is_unchanged_and_detached(self):
        source = ramp(32, 24)
        images = torch.cat((source, source + .1), dim=1)
        visibility = torch.ones_like(source, requires_grad=True)
        original = visibility.detach().clone()
        motion = torch.tensor([[20., 0, 0, 0, 0, 0]], dtype=source.dtype, requires_grad=True)
        loss = photometric_loss(images, torch.full_like(source, 4), camera(32, 24), motion,
                                torch.zeros(1, 3, dtype=source.dtype), visibility)
        self.assertGreater(float(loss.detach()), .1)  # no overlap must not hide valid target pixels
        loss.backward()
        torch.testing.assert_close(visibility.detach(), original)
        self.assertIsNone(visibility.grad)
        self.assertTrue(torch.isfinite(motion.grad).all())

    def test_no_visible_pixels_returns_finite_zero_with_finite_gradients(self):
        source = ramp(32, 24)
        images = torch.cat((source, source), dim=1)
        motion = torch.zeros(1, 6, dtype=source.dtype, requires_grad=True)
        visibility = torch.zeros_like(source, dtype=torch.bool)
        loss = photometric_loss(images, torch.full_like(source, 4), camera(32, 24), motion,
                                torch.zeros(1, 3, dtype=source.dtype), visibility)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(motion.grad).all())
        torch.testing.assert_close(motion.grad, torch.zeros_like(motion))

    def test_intrinsics_resizing_preserves_normalized_projection(self):
        high = ramp(64, 48)
        low = ramp(32, 24)
        high_K = camera(64, 48)
        low_K = high_K.clone()
        low_K[:, :2] *= .5
        low_K[:, 2:] = (low_K[:, 2:] + .5) * .5 - .5
        motion = torch.tensor([[.1, -.1, .03, 0, 0, .05]], dtype=high.dtype)
        imu = torch.tensor([[0, 0, .03]], dtype=high.dtype)
        _, _, _, high_grid = backward_warp(high, torch.full_like(high, 4), high_K, motion, imu)
        _, _, _, low_grid = backward_warp(low, torch.full_like(low, 4), low_K, motion, imu)
        # Here projection is affine (constant depth, optical-axis rotation), so
        # averaging high-resolution pixel centres gives the exact low-res grid.
        sampled_high = F.avg_pool2d(high_grid.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
        torch.testing.assert_close(sampled_high, low_grid, atol=1e-12, rtol=0)


class ValidationAggregationTests(unittest.TestCase):
    def test_photo_metric_is_invariant_to_batch_size_with_empty_masks(self):
        from scripts.train_motion import evaluate

        class ZeroNet(torch.nn.Module):
            def forward(self, images, state):
                return torch.zeros(len(images), 6, dtype=images.dtype)

        images = torch.zeros(3, 2, 24, 32)
        images[:, 1] = .2
        visibility = torch.zeros(3, 1, 24, 32, dtype=torch.bool)
        visibility[0] = True
        data = {"images": images, "state": torch.zeros(3, 14),
                "target": torch.zeros(3, 6), "depth": torch.ones(3, 1, 24, 32),
                "intrinsics": camera(32, 24, torch.float32).repeat(3, 1),
                "visibility": visibility}
        normalization = (torch.zeros(14), torch.ones(14), torch.zeros(6), torch.ones(6))
        metrics = [evaluate(ZeroNet(), data, normalization, size) for size in [1, 2, 3]]
        for result in metrics:
            self.assertAlmostEqual(result["photo_loss"], .2, places=6)


class OnnxTrainingContractTests(unittest.TestCase):
    def test_motion_model_onnx_matches_torch_on_varied_inputs(self):
        torch.manual_seed(94)
        model = MotionNet().eval()
        with tempfile.TemporaryDirectory(prefix="aerovio-onnx-test-") as directory:
            path = Path(directory) / "motion.onnx"
            torch.onnx.export(model, (torch.zeros(1, 2, 48, 64), torch.zeros(1, 14)), str(path),
                              input_names=["images", "state"], output_names=["motion"],
                              opset_version=13, dynamo=False)
            cv_model = cv2.dnn.readNetFromONNX(str(path))
            for scale in [0., .1, 1., 10.]:
                images = torch.rand(1, 2, 48, 64)
                state = torch.randn(1, 14) * scale
                with torch.no_grad(): expected = model(images, state).numpy()
                cv_model.setInput(images.numpy(), "images")
                cv_model.setInput(state.numpy(), "state")
                actual = cv_model.forward()
                np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
