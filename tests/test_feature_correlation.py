"""Geometric correspondence, frozen-function and export contract checks."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "workspace/solution"), str(ROOT / "tests")]
from training.feature_correlation import CorrelationMotionNet, local_correlation
from training.motion_model import MotionNet
import test_motion_runtime as runtime_tests
from test_motion_runtime import CONFIG, FRAME, EMPTY
from motion_inference import MotionPredictor
from solution import Solution
from preintegrate import log_so3


class FeatureCorrelationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_known_displacement_sign_and_axis_order(self):
        generator = torch.Generator().manual_seed(82)
        previous = torch.randn(1, 64, 12, 16, generator=generator)
        for dy, dx in ((-1, 0), (0, 1), (1, -1)):
            current = torch.roll(previous, shifts=(-dy, -dx), dims=(-2, -1))
            scores = local_correlation(previous, current)[:, :, 1:-1, 1:-1]
            self.assertTrue(torch.all(scores.argmax(1) == (dy+1)*3+dx+1))

    def test_zero_features_finite_and_border_has_no_wraparound(self):
        zeros = torch.zeros(1, 4, 12, 16)
        self.assertTrue(torch.equal(local_correlation(zeros, zeros), torch.zeros(1, 9, 12, 16)))
        ones = torch.ones_like(zeros)
        volume = local_correlation(ones, ones)
        self.assertEqual(float(volume[0, 0, 0, 0]), 0.)
        self.assertGreater(float(volume[0, 4, 0, 0]), .99)

    def model(self, variant="temporal"):
        torch.manual_seed(17)
        return CorrelationMotionNet(MotionNet(), [.01]*6, [.1]*6, variant)

    def test_zero_start_preserves_translation_and_calibrated_rotation(self):
        net = self.model()
        images, state, prior = torch.rand(3, 2, 48, 64), torch.rand(3, 14), torch.rand(3, 3)
        with torch.no_grad():
            output = net(images, state, prior)
            torch.testing.assert_close(output[:, :3], net.base(images, state)[:, :3], atol=0, rtol=0)
            torch.testing.assert_close(output[:, 3:] * .1 + .01, torch.zeros(3, 3), atol=1e-8, rtol=0)
        self.assertFalse(any(p.requires_grad for p in net.base.parameters()))

    def test_control_has_same_weights_but_no_previous_image_path(self):
        one, two = self.model(), self.model("single-frame")
        self.assertEqual(sum(p.numel() for p in one.parameters()), sum(p.numel() for p in two.parameters()))
        for a, b in zip(one.state_dict().values(), two.state_dict().values()): self.assertTrue(torch.equal(a, b))
        torch.nn.init.normal_(two.rotation_head[-1].weight)
        images = torch.rand(3, 2, 48, 64, requires_grad=True)
        two.correction(images, torch.rand(3, 14), torch.rand(3, 3)).sum().backward()
        self.assertEqual(float(images.grad[:, 0].abs().sum()), 0.)
        self.assertGreater(float(images.grad[:, 1].abs().sum()), 0.)

    def test_nonzero_branch_gradient_reaches_both_frames_without_base_updates(self):
        net = self.model()
        torch.nn.init.normal_(net.rotation_head[-1].weight)
        images = torch.rand(2, 2, 48, 64, requires_grad=True)
        net.correction(images, torch.rand(2, 14), torch.rand(2, 3)).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in net.base.parameters()))
        self.assertTrue(all(float(images.grad[:, i].abs().sum()) > 0 for i in (0, 1)))

    def test_export_opencv_parity_with_nonzero_head_for_both_variants(self):
        for variant in ("temporal", "single-frame"):
            net = self.model(variant).eval()
            torch.nn.init.normal_(net.rotation_head[-1].weight, std=.01)
            inputs = (torch.rand(1, 2, 48, 64), torch.rand(1, 14), torch.rand(1, 3)*.01)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "net.onnx"
                torch.onnx.export(net, inputs, str(path), opset_version=13, dynamo=False,
                                  input_names=["images", "state", "rotation_prior"], output_names=["motion"])
                exported = cv2.dnn.readNetFromONNX(str(path))
                for key, value in zip(("images", "state", "rotation_prior"), inputs): exported.setInput(value.numpy(), key)
                with torch.no_grad(): expected = net(*inputs).numpy()
                np.testing.assert_allclose(exported.forward(), expected, atol=1e-5, rtol=1e-5)

    def test_prior_input_shape_and_causal_solution_integration(self):
        fixture = runtime_tests.InferenceContractTests()
        network = fixture.input_sensitive_network()
        meta = dict(fixture.metadata(), rotation_prior_input=True)
        with patch("cv2.dnn.readNetFromONNX", return_value=network):
            predictor = MotionPredictor(fixture.bundle(meta))
        for bad in (None, [1, 2], [0, 0, np.nan]):
            with self.assertRaisesRegex(ValueError, "rotation prior"):
                predictor.predict(np.zeros((2, 48, 64)), np.zeros(14), bad)
        predictor.predict(np.zeros((2, 48, 64)), np.zeros(14), [.01, .02, .03])
        np.testing.assert_allclose(network.inputs["rotation_prior"], [[.01, .02, .03]])
        class Spy:
            size = (64, 48)
            meta = {"gyro_bias": {"enabled": True}}
            requires_rotation_prior = True
            def predict(self, pair, state, rotation_prior):
                self.prior = rotation_prior.copy()
                return np.zeros(6)
        spy = Spy()
        solution = Solution(CONFIG, spy)
        for i in range(22):
            t = (i+1)*.05
            imu = np.array([[t-.045+j*.005, 0, 0, 9.81, 0, 0, .01] for j in range(10)])
            before = solution.T.copy()
            solution.step(t, FRAME, imu)
            if i:
                np.testing.assert_allclose(spy.prior, log_so3(before[:3, :3].T @ solution.T[:3, :3]), atol=1e-12)
        self.assertTrue(solution.gyro_calibration.accepted)
        np.testing.assert_allclose(spy.prior, [0, 0, 0], atol=1e-10)


if __name__ == "__main__": unittest.main()
