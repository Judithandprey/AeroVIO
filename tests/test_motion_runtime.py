"""Coordinate, causality and fail-fast tests for the learned motion guest."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workspace/solution"))
from motion_inference import MotionPredictor, validate_motion_options
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

    def test_rotation_ablation_preserves_predicted_camera_translation(self):
        motion = [0.1, -0.02, 0.03, 0.15, -0.05, 0.02]
        enabled = Solution(CONFIG, Predictor(motion))
        disabled = Solution(CONFIG, Predictor(motion, {"rotation_residual_enabled": False}))
        for solution in (enabled, disabled):
            solution.step(0, FRAME, EMPTY)
        for frame in (1, 2):
            t = frame * .05
            imu = [[t, 0, 0, 0, 0, 0, 2]]
            old_enabled, old_disabled = enabled.T.copy(), disabled.T.copy()
            out_enabled = enabled.step(t, FRAME, imu)
            out_disabled = disabled.step(t, FRAME, imu)
            delta_enabled = np.linalg.inv(old_enabled) @ enabled.T
            delta_disabled = np.linalg.inv(old_disabled) @ disabled.T
            np.testing.assert_allclose(delta_enabled[:3, 3], motion[:3], atol=1e-9)
            np.testing.assert_allclose(delta_disabled[:3, 3], motion[:3], atol=1e-9)
            np.testing.assert_allclose(delta_disabled[:3, :3], exp_so3([0, 0, .1]), atol=1e-7)
            self.assertFalse(np.allclose(delta_enabled[:3, :3], delta_disabled[:3, :3]))
            np.testing.assert_allclose(out_disabled[10:], [0, 0, 2], atol=1e-7)
            if frame == 1:
                np.testing.assert_array_equal(out_enabled[:3], out_disabled[:3])
            # Subsequent world-axis position may differ because its reference
            # orientation changed; the predicted local translation must not.
            for a, b in zip(enabled.predictor.calls[-1], disabled.predictor.calls[-1]):
                np.testing.assert_array_equal(a, b)

    def test_rotation_ablation_requires_boolean_and_disabled_map(self):
        for value in (0, 1, "false", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Boolean"):
                Solution(CONFIG, Predictor(np.zeros(6), {"rotation_residual_enabled": value}))
        with self.assertRaisesRegex(ValueError, "map correction disabled"):
            Solution(CONFIG, Predictor(np.zeros(6),
                                       {"rotation_residual_enabled": False, "map": {"enabled": True}}))

    def test_new_episode_resets_pose_and_history(self):
        one = Solution(CONFIG, Predictor([0.1, 0, 0, 0, 0, 0]))
        one.step(0, FRAME, EMPTY)
        one.step(0.05, FRAME, EMPTY)
        two = Solution(CONFIG, Predictor([0.1, 0, 0, 0, 0, 0]))
        state = two.step(100, FRAME, EMPTY)
        np.testing.assert_array_equal(state[:3], [0, 0, 0])
        self.assertEqual(len(two.predictor.calls), 0)

    def test_state_only_solution_never_prepares_a_frame_pair(self):
        predictor = Predictor([0.1, 0, 0, 0, 0, 0], {"input_mode": "imu_only", "network_inputs": "state"})
        solution = Solution(CONFIG, predictor)
        with patch("solution.prepare_frame", side_effect=AssertionError("unexpected image preprocessing")), \
             patch.object(predictor, "predict", return_value=predictor.motion) as predict:
            initial = solution.step(0.0, FRAME, EMPTY)
            first = solution.step(0.05, FRAME, EMPTY)
            second = solution.step(0.10, FRAME, EMPTY)
        np.testing.assert_array_equal(initial[:7], [0, 0, 0, 0, 0, 0, 1])
        np.testing.assert_allclose(first[:3], [0.1, 0, 0])
        np.testing.assert_allclose(second[:3], [0.2, 0, 0])
        self.assertEqual(predict.call_count, 2)
        for call in predict.call_args_list:
            self.assertIsNone(call.args[0])
            self.assertEqual(call.args[1].shape, (14,))

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

    def input_sensitive_network(self):
        class Network:
            def __init__(self): self.inputs = {}
            def setPreferableBackend(self, value): pass
            def setPreferableTarget(self, value): pass
            def enableFusion(self, value): pass
            def setInput(self, data, name): self.inputs[name] = data.copy()
            def forward(self):
                value = self.inputs["state"].mean()
                if "images" in self.inputs:
                    value += self.inputs["images"].mean()
                return np.full((1, 6), value, dtype=np.float32)
        return Network()

    def test_imu_only_zeros_images_without_mutating_caller_inputs(self):
        network = self.input_sensitive_network()
        meta = dict(self.metadata(), input_mode="imu_only")
        with patch("cv2.dnn.readNetFromONNX", return_value=network):
            predictor = MotionPredictor(self.bundle(meta))
        pair = np.ones((2, 48, 64), dtype=np.float32)
        state = np.full(14, 5, dtype=np.float32)
        one = predictor.predict(pair, state)
        two = predictor.predict(pair * 0.25, state)
        np.testing.assert_array_equal(one, two)
        np.testing.assert_array_equal(network.inputs["images"], np.zeros((1, 2, 48, 64)))
        np.testing.assert_array_equal(pair, np.ones((2, 48, 64)))
        np.testing.assert_array_equal(state, np.full(14, 5))
        with self.assertRaisesRegex(ValueError, "finite"):
            predictor.predict(pair * np.nan, state)

    def test_default_image_mode_preserves_visual_input(self):
        network = self.input_sensitive_network()
        with patch("cv2.dnn.readNetFromONNX", return_value=network):
            predictor = MotionPredictor(self.bundle(self.metadata()))
        one = predictor.predict(np.ones((2, 48, 64)), np.full(14, 5))
        two = predictor.predict(np.zeros((2, 48, 64)), np.full(14, 5))
        self.assertFalse(np.array_equal(one, two))
        self.assertEqual(validate_motion_options({}), ("image_imu", "images_state", True))

    def test_compact_imu_graph_receives_only_normalized_state(self):
        network = self.input_sensitive_network()
        meta = dict(self.metadata(), input_mode="imu_only", network_inputs="state")
        with patch("cv2.dnn.readNetFromONNX", return_value=network):
            predictor = MotionPredictor(self.bundle(meta))
        one = predictor.predict(np.ones((2, 48, 64)), np.full(14, 5))
        two = predictor.predict(np.zeros((2, 48, 64)), np.full(14, 5))
        np.testing.assert_array_equal(one, two)
        np.testing.assert_array_equal(one, predictor.predict(None, np.full(14, 5)))
        self.assertEqual(set(network.inputs), {"state"})
        np.testing.assert_array_equal(network.inputs["state"], np.full((1, 14), 2))

    def test_invalid_ablation_metadata_fails_before_loading_weights(self):
        cases = [({"input_mode": "images"}, "input_mode"),
                 ({"input_mode": None}, "input_mode"),
                 ({"network_inputs": "both"}, "network_inputs"),
                 ({"network_inputs": "state"}, "requires input_mode imu_only"),
                 ({"rotation_residual_enabled": 0}, "Boolean"),
                 ({"rotation_residual_enabled": "false"}, "Boolean"),
                 ({"rotation_residual_enabled": False, "map": {"enabled": True}}, "map correction disabled")]
        for options, error in cases:
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, error):
                MotionPredictor(self.bundle(dict(self.metadata(), **options), missing=True))

    def test_imu_only_actual_onnx_is_independent_of_images(self):
        folder = Path(__file__).resolve().parents[1] / "experiments/learned-motion-v1/warp"
        metadata = json.loads((folder / "meta.json").read_text())
        metadata["input_mode"] = "imu_only"
        class Bundle:
            def read_text(self, name): return json.dumps(metadata)
            def read_bytes(self, name): return (folder / name).read_bytes()
        predictor = MotionPredictor(Bundle())
        state = np.asarray(metadata["state_mean"], dtype=np.float32)
        pair = np.random.default_rng(7).random((2, 48, 64), dtype=np.float32)
        one = predictor.predict(pair, state)
        two = predictor.predict(1 - pair, state)
        np.testing.assert_array_equal(one, two)
        self.assertTrue(np.isfinite(one).all())

    def test_missing_weights_and_invalid_normalization_fail(self):
        with self.assertRaises(FileNotFoundError):
            MotionPredictor(self.bundle(self.metadata(), missing=True))
        metadata = self.metadata()
        metadata["state_std"][0] = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            MotionPredictor(self.bundle(metadata))


class PackagingAblationTests(unittest.TestCase):
    def package_module(self):
        path = Path(__file__).resolve().parents[1] / "scripts/package_motion.py"
        spec = importlib.util.spec_from_file_location("package_motion_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_package_preserves_checkpoint_and_inherits_input_mode(self):
        module = self.package_module()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "workspace/solution"
            source.mkdir(parents=True)
            (source / "solution.py").write_text("# packaging test fixture\n")
            model = root / "model"
            model.mkdir()
            weights = b"fixed weight fixture"
            (model / "net.onnx").write_bytes(weights)
            metadata = {"model_sha256": hashlib.sha256(weights).hexdigest(), "input_mode": "imu_only"}
            original = json.dumps(metadata)
            (model / "meta.json").write_text(original)
            def bundle(command, **kwargs):
                Path(command[command.index("-o") + 1]).write_bytes(b"bundled fixture")
            argv = ["package_motion.py", "--model", str(model), "--name", "imu-rotation-off",
                    "--disable-rotation-residual"]
            with patch.object(module, "ROOT", root), patch.object(sys, "argv", argv), \
                 patch.object(sys, "path", sys.path.copy()), patch.object(module.subprocess, "run", side_effect=bundle), \
                 contextlib.redirect_stdout(io.StringIO()):
                module.main()
            stage = root / "artifacts/imu-rotation-off-solution"
            saved = json.loads((stage / "meta.json").read_text())
            self.assertIs(saved["rotation_residual_enabled"], False)
            self.assertEqual(saved["input_mode"], "imu_only")
            self.assertEqual(saved["map"], {"enabled": False})
            self.assertEqual((stage / "net.onnx").read_bytes(), weights)
            self.assertEqual((model / "meta.json").read_text(), original)

    def test_packaging_rejects_map_with_rotation_ablation(self):
        module = self.package_module()
        argv = ["package_motion.py", "--model", "unused", "--name", "invalid",
                "--map", "--disable-rotation-residual"]
        with patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()), \
             self.assertRaises(SystemExit) as raised:
            module.main()
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
