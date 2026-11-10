"""Coordinate, causality and fail-fast tests for the learned motion guest."""
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workspace/solution"))
from motion_inference import MotionPredictor
from preintegrate import camera_features, exp_so3, log_so3, make_state, preintegrate
from solution import Solution, quaternion_from_matrix


CONFIG = {"width": 320, "height": 240, "fx": 200.0, "fy": 200.0,
          "cx": 159.5, "cy": 119.5, "T_cam_imu": np.eye(4).tolist()}
FRAME = np.zeros((240, 320, 3), dtype=np.uint8)
EMPTY = np.empty((0, 7))


class Predictor:
    size = (64, 48)

    def __init__(self, motion, meta=None):
        self.motion = np.asarray(motion, dtype=np.float64)
        self.meta = meta or {}
        self.calls = []

    def predict(self, pair, state):
        self.calls.append((pair.copy(), state.copy()))
        return self.motion.copy()


class MotionRuntimeTests(unittest.TestCase):
    def test_first_sample_interval_is_integrated(self):
        window = np.array([[0.05, 2, 0, 0, 0, 0, 1.0]])
        features = preintegrate(window, np.eye(4), 0.0, 0.05)
        np.testing.assert_allclose(features[:3], [0, 0, 0.05], atol=1e-7)
        np.testing.assert_allclose(features[3:6], [0.1, 0, 0], atol=1e-7)
        np.testing.assert_allclose(features[6:], [0.0025, 0, 0], atol=1e-7)

    def test_timestamp_roundoff_is_clipped_but_future_samples_rejected(self):
        # Stored float32 timestamps can round slightly beyond a float64 frame time;
        # the upstream reader includes samples within its 1-microsecond tolerance.
        timestamp = float(np.float32(.6))
        self.assertGreater(timestamp, .6)
        vector = preintegrate([[timestamp, 0, 0, 0, 0, 0, 1]], np.eye(4), .55, .6)
        np.testing.assert_allclose(vector[:3], [0, 0, .05], atol=1e-7)
        with self.assertRaisesRegex(ValueError, "future"):
            preintegrate([[.6 + 2e-6, 0, 0, 0, 0, 0, 1]], np.eye(4), .55, .6)

    def test_imu_features_transform_into_camera_axes(self):
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = exp_so3([0, 0, np.pi / 2])
        features = preintegrate([[0.1, 2, 0, 0, 1, 0, 0]], extrinsic, 0, 0.1)
        np.testing.assert_allclose(features[:3], [0, 0.1, 0], atol=1e-7)
        np.testing.assert_allclose(features[3:6], [0, 0.2, 0], atol=1e-7)

    def test_frame_size_normalization_preserves_intrinsics(self):
        full = dict(CONFIG, width=640, height=480, fx=400, fy=400,
                    cx=319.5, cy=239.5)
        np.testing.assert_array_equal(camera_features(CONFIG), camera_features(full))
        state = make_state(EMPTY, np.eye(4), CONFIG, 1, 1.05)
        self.assertEqual(state.shape, (14,))
        np.testing.assert_allclose(state[9:13], [0.625, 200 / 240, 0.5, 0.5])
        self.assertAlmostEqual(float(state[-1]), 0.05, places=7)

    def test_pose_composition_and_current_camera_twist(self):
        predictor = Predictor([0.1, 0, 0, 0, 0, np.pi / 2])
        solution = Solution(CONFIG, predictor)
        initial = solution.step(0, FRAME, EMPTY)
        np.testing.assert_array_equal(initial, [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        first = solution.step(0.05, FRAME, EMPTY)
        np.testing.assert_allclose(first[:3], [0.1, 0, 0], atol=1e-8)
        np.testing.assert_allclose(first[7:10], [0, -2, 0], atol=1e-8)
        np.testing.assert_allclose(first[10:], [0, 0, np.pi / 0.1], atol=1e-6)
        second = solution.step(0.1, FRAME, EMPTY)
        np.testing.assert_allclose(second[:3], [0.1, 0.1, 0], atol=1e-8)
        self.assertAlmostEqual(float(np.linalg.norm(second[3:7])), 1.0)
        # Missing IMU does not reset the already accumulated visual pose.
        self.assertEqual(len(predictor.calls), 2)

    def test_gyro_prior_and_residual_rotation_order(self):
        residual = [0.1, 0, 0]
        solution = Solution(CONFIG, Predictor([0, 0, 0, *residual]))
        solution.step(0, FRAME, EMPTY)
        solution.step(0.05, FRAME, [[0.05, 0, 0, 0, 0, 0, 2]])
        expected = exp_so3([0, 0, 0.1]) @ exp_so3(residual)
        np.testing.assert_allclose(solution.T[:3, :3], expected, atol=1e-7)

    def test_new_episode_resets_pose_and_history(self):
        one = Solution(CONFIG, Predictor([0.1, 0, 0, 0, 0, 0]))
        one.step(0, FRAME, EMPTY)
        one.step(0.05, FRAME, EMPTY)
        two = Solution(CONFIG, Predictor([0.1, 0, 0, 0, 0, 0]))
        state = two.step(100, FRAME, EMPTY)
        np.testing.assert_array_equal(state[:3], [0, 0, 0])
        self.assertEqual(len(two.predictor.calls), 0)

    def test_rejects_nonfinite_prediction_without_advancing_state(self):
        solution = Solution(CONFIG, Predictor([np.nan, 0, 0, 0, 0, 0]))
        solution.step(0, FRAME, EMPTY)
        with self.assertRaisesRegex(ValueError, "finite"):
            solution.step(0.05, FRAME, EMPTY)
        self.assertEqual(solution.previous_t, 0)
        np.testing.assert_array_equal(solution.T, np.eye(4))

    def test_rejects_nonmonotonic_time_and_future_imu(self):
        solution = Solution(CONFIG, Predictor(np.zeros(6)))
        solution.step(0, FRAME, EMPTY)
        with self.assertRaisesRegex(ValueError, "increase"):
            solution.step(0, FRAME, EMPTY)
        with self.assertRaisesRegex(ValueError, "future"):
            solution.step(0.05, FRAME, [[0.06, 0, 0, 0, 0, 0, 0]])
        with self.assertRaisesRegex(ValueError, "finite"):
            preintegrate([[0.05, np.inf, 0, 0, 0, 0, 0]], np.eye(4), 0, 0.05)

    def test_quaternion_half_turn(self):
        rotation = exp_so3([np.pi, 0, 0])
        quaternion = quaternion_from_matrix(rotation)
        self.assertAlmostEqual(float(np.linalg.norm(quaternion)), 1)
        np.testing.assert_allclose(abs(quaternion[0]), 1, atol=1e-9)
        np.testing.assert_allclose(exp_so3(log_so3(rotation)), rotation, atol=1e-9)

    def test_map_correction_updates_velocity_and_next_prior(self):
        class Mapper:
            def __init__(self, **settings):
                self.frames = 0

            def update(self, gray, intrinsics, prior):
                self.frames += 1
                pose = prior.copy()
                if self.frames == 2:
                    pose[:3, :3] = exp_so3([0, 0, np.pi / 2])
                    pose[:3, 3] = [0.2, 0, 0]
                return pose, {"frames": self.frames}

        module = types.SimpleNamespace(SparseLocalMap=Mapper)
        with patch.dict(sys.modules, {"local_map": module}):
            solution = Solution(CONFIG, Predictor([0.1, 0, 0, 0, 0, 0], {"map": {"enabled": True}}))
            solution.step(0, FRAME, EMPTY)
            out = solution.step(0.1, FRAME, EMPTY)
            np.testing.assert_allclose(out[7:10], [0, -2, 0], atol=1e-8)
            out = solution.step(0.2, FRAME, EMPTY)
            np.testing.assert_allclose(out[:3], [0.2, 0.1, 0], atol=1e-8)


class InferenceContractTests(unittest.TestCase):
    def metadata(self):
        return {"schema_version": 1, "input_width": 64, "input_height": 48,
                "state_mean": [1] * 14, "state_std": [2] * 14,
                "target_mean": [0.1] * 6, "target_std": [3] * 6}

    def bundle(self, metadata, missing=False):
        class Bundle:
            def read_text(self, name):
                return json.dumps(metadata)

            def read_bytes(self, name):
                if missing:
                    raise FileNotFoundError(name)
                return b"unused: network mocked by contract test"
        return Bundle()

    def test_normalization_and_output_contract(self):
        class Network:
            def __init__(self): self.inputs = {}
            def setPreferableBackend(self, value): pass
            def setPreferableTarget(self, value): pass
            def enableFusion(self, value): pass
            def setInput(self, data, name): self.inputs[name] = data
            def forward(self): return np.ones((1, 6), dtype=np.float32)
        network = Network()
        with patch("cv2.dnn.readNetFromONNX", return_value=network):
            predictor = MotionPredictor(self.bundle(self.metadata()))
        output = predictor.predict(np.zeros((2, 48, 64)), np.full(14, 5))
        np.testing.assert_array_equal(network.inputs["state"], np.full((1, 14), 2))
        np.testing.assert_allclose(output, np.full(6, 3.1), atol=1e-6)
        self.assertEqual(network.inputs["images"].shape, (1, 2, 48, 64))

    def test_missing_weights_and_invalid_normalization_fail(self):
        with self.assertRaises(FileNotFoundError):
            MotionPredictor(self.bundle(self.metadata(), missing=True))
        metadata = self.metadata()
        metadata["state_std"][0] = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            MotionPredictor(self.bundle(metadata))


if __name__ == "__main__":
    unittest.main()
