"""Information-control and algebraic-equivalence checks using synthetic inputs."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
import onnx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.motion_model import MotionNet, fold_zero_image_features, network_image_input
from scripts.train_motion import evaluate


class InputAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_initialization_and_architecture_match_original_control(self):
        torch.manual_seed(17)
        net = MotionNet()
        initial = hashlib.sha256(b"".join(v.numpy().tobytes() for v in net.state_dict().values())).hexdigest()
        self.assertEqual(initial, "99d9364370fcf237ba685f6ab1c381e954464789b22218eaa0c65bbb38a03d29")
        self.assertEqual(sum(parameter.numel() for parameter in net.parameters()), 29398)

    def test_image_selection_preserves_real_photometric_labels(self):
        images = torch.rand(2, 2, 48, 64)
        original = images.clone()
        self.assertIs(network_image_input(images), images)
        selected = network_image_input(images, "imu_only")
        self.assertEqual(int(torch.count_nonzero(selected)), 0)
        torch.testing.assert_close(images, original, atol=0, rtol=0)
        with self.assertRaises(ValueError):
            network_image_input(images, "invalid")

    def test_imu_predictions_and_gradients_cannot_use_images(self):
        torch.manual_seed(92)
        net = MotionNet()
        state = torch.randn(2, 14)
        predictions, gradients = [], []
        for image_value in (0., 1.):
            net.zero_grad(set_to_none=True)
            images = torch.full((2, 2, 48, 64), image_value)
            prediction = net(network_image_input(images, "imu_only"), state)
            prediction.square().sum().backward()
            predictions.append(prediction.detach())
            gradients.append([parameter.grad.clone() for parameter in net.parameters()])
        torch.testing.assert_close(predictions[0], predictions[1], atol=0, rtol=0)
        for first, second in zip(*gradients):
            torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_validation_masks_network_images_but_uses_real_photo_labels(self):
        class SpyNet(torch.nn.Module):
            def forward(self, images, state):
                self.received_images = images.clone()
                return torch.zeros(len(state), 6)

        net = SpyNet()
        images = torch.zeros(2, 2, 48, 64)
        images[:, 1] = .2
        data = {"images": images, "state": torch.zeros(2, 14), "target": torch.zeros(2, 6),
                "depth": torch.ones(2, 1, 48, 64),
                "intrinsics": torch.tensor([[32., 32., 31.5, 23.5]]).repeat(2, 1),
                "visibility": torch.ones(2, 1, 48, 64, dtype=torch.bool)}
        normalization = (torch.zeros(14), torch.ones(14), torch.zeros(6), torch.ones(6))
        result = evaluate(net, data, normalization, 2, input_mode="imu_only")
        self.assertEqual(int(torch.count_nonzero(net.received_images)), 0)
        self.assertAlmostEqual(result["photo_loss"], .2, places=6)
        self.assertAlmostEqual(result["pose_loss"], 0., places=6)

    def test_compact_folding_preserves_biased_visual_features_and_original_weights(self):
        torch.manual_seed(34)
        full = MotionNet().eval()
        original = {key: tensor.clone() for key, tensor in full.state_dict().items()}
        compact = fold_zero_image_features(full)
        for scale in (0., .1, 1., 10.):
            state = torch.randn(8, 14) * scale
            with torch.no_grad():
                expected = full(torch.zeros(8, 2, 48, 64), state)
                actual = compact(state)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        for key, tensor in full.state_dict().items():
            torch.testing.assert_close(tensor, original[key], atol=0, rtol=0)
        self.assertEqual(sum(p.numel() for p in compact.parameters()), 1350)

    def test_compact_and_full_onnx_contract_and_opencv_equivalence(self):
        torch.manual_seed(8)
        full = MotionNet().eval()
        compact = fold_zero_image_features(full)
        with tempfile.TemporaryDirectory(prefix="aerovio-input-ablation-") as directory:
            full_path, compact_path = (Path(directory) / name for name in ("full.onnx", "compact.onnx"))
            torch.onnx.export(full, (torch.zeros(1, 2, 48, 64), torch.zeros(1, 14)), str(full_path),
                              input_names=["images", "state"], output_names=["motion"], opset_version=13, dynamo=False)
            torch.onnx.export(compact, torch.zeros(1, 14), str(compact_path), input_names=["state"],
                              output_names=["motion"], opset_version=13, dynamo=False)
            self.assertEqual([item.name for item in onnx.load(full_path).graph.input], ["images", "state"])
            self.assertEqual([item.name for item in onnx.load(compact_path).graph.input], ["state"])
            full_cv, compact_cv = (cv2.dnn.readNetFromONNX(str(path)) for path in (full_path, compact_path))
            for scale in (0., .1, 1., 10.):
                state = (torch.randn(1, 14) * scale).numpy()
                full_cv.setInput(np.zeros((1, 2, 48, 64), np.float32), "images")
                full_cv.setInput(state, "state")
                compact_cv.setInput(state, "state")
                np.testing.assert_allclose(compact_cv.forward(), full_cv.forward(), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
