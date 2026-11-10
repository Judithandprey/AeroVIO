"""Fixed-window, frame-causal and unchanged-network contracts for gyro calibration."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workspace/solution"))
from gyro_bias import InitialGyroBias, integrate_debiased_rotation
from motion_inference import validate_gyro_bias_options, validate_motion_options
from preintegrate import exp_so3
from solution import Solution

CONFIG = {"width": 320, "height": 240, "fx": 200., "fy": 200.,
          "cx": 159.5, "cy": 119.5, "T_cam_imu": np.eye(4).tolist()}
FRAME = np.zeros((240, 320, 3), dtype=np.uint8)
BIAS = np.array([.01, -.012, .007])


def samples(count=200, bias=BIAS):
    rows = np.zeros((count, 7))
    rows[:, 0] = np.arange(1, count + 1) * .005
    rows[:, 2] = -9.81
    rows[:, 4:7] = bias
    return rows


class Predictor:
    size = (64, 48)

    def __init__(self, calibrated=False, residual=True, explicit=True):
        self.meta = {"rotation_residual_enabled": residual}
        if explicit:
            self.meta["gyro_bias"] = {"enabled": calibrated}
        self.calls = []
        self.motion = np.array([.01, -.002, .003, .0004, -.0003, .0002])

    def predict(self, pair, state):
        self.calls.append((pair.copy(), state.copy(), self.motion.copy()))
        return self.motion.copy()


class GyroCalibrationTests(unittest.TestCase):
    def test_full_window_accepts_known_bias_with_population_moments(self):
        rows = samples()
        jitter = np.tile([-1., 1.], 100)
        rows[:, 1] += jitter * .1
        rows[:, 4] += jitter * .02
        original = rows.copy()
        calibration = InitialGyroBias()
        for start in range(0, 200, 10):
            batch = rows[start:start + 10]
            calibration.observe_after_output(float(batch[-1, 0]), batch)
            self.assertEqual(calibration.accepted, start == 190)
        info = calibration.diagnostics()
        self.assertEqual(info["samples"], 200)
        self.assertAlmostEqual(info["anchor_t"], 0)
        self.assertAlmostEqual(info["sample_span_s"], .995)
        self.assertAlmostEqual(info["activation_after_t"], 1)
        np.testing.assert_allclose(info["bias_imu"], BIAS, atol=1e-15)
        self.assertAlmostEqual(info["accel_std_norm"], np.linalg.norm(rows[:, 1:4].std(0)))
        self.assertAlmostEqual(info["gyro_std_norm"], np.linalg.norm(rows[:, 4:7].std(0)))
        np.testing.assert_array_equal(rows, original)
        frozen = calibration.diagnostics()
        calibration.observe_after_output(2, samples(400, np.ones(3)))
        self.assertEqual(calibration.diagnostics(), frozen)

    def test_boundary_tolerance_and_next_interval_activation_without_pose_reset(self):
        enabled = Solution(CONFIG, Predictor(True, False))
        disabled = Solution(CONFIG, Predictor(False, False))
        rows = samples(210)
        old_outputs = []
        for start in range(0, 200, 10):
            t = rows[start + 9, 0]
            if start == 190:
                t -= 2.2e-8
            a = enabled.step(t, FRAME, rows[start:start + 10])
            b = disabled.step(t, FRAME, rows[start:start + 10])
            np.testing.assert_array_equal(a, b)
            old_outputs.append((a, a.copy()))
        before = enabled.T.copy()
        self.assertEqual(enabled.gyro_bias_diagnostics["status"], "accepted")
        self.assertAlmostEqual(enabled.gyro_bias_diagnostics["activation_after_t"], 1 - 2.2e-8)
        enabled.step(1.05, FRAME, rows[200:])
        disabled.step(1.05, FRAME, rows[200:])
        np.testing.assert_allclose(enabled.T[:3, :3], before[:3, :3], atol=1e-14)
        self.assertGreater(np.linalg.norm(before[:3, :3] - np.eye(3)), .001)
        self.assertGreater(np.linalg.norm(enabled.T[:3, :3] - disabled.T[:3, :3]), 1e-4)
        for returned, frozen in old_outputs:
            np.testing.assert_array_equal(returned, frozen)

    def test_factorial_cells_keep_network_inputs_outputs_and_local_translation(self):
        cells = {(bias, residual): Solution(CONFIG, Predictor(bias, residual))
                 for bias in (False, True) for residual in (False, True)}
        rows = samples(230)
        for start in range(0, len(rows), 10):
            t = float(rows[start + 9, 0])
            prior = {key: value.T.copy() for key, value in cells.items()}
            frame = FRAME + (start // 10)
            outputs = {key: value.step(t, frame, rows[start:start + 10]) for key, value in cells.items()}
            if start == 0:
                continue
            expected = cells[(False, True)].predictor.calls[-1]
            for key, value in cells.items():
                for actual, wanted in zip(value.predictor.calls[-1], expected):
                    np.testing.assert_array_equal(actual, wanted)
                local = np.linalg.inv(prior[key]) @ value.T
                np.testing.assert_allclose(local[:3, 3], value.predictor.motion[:3], atol=1e-14)
            if t <= 1:
                for residual in (False, True):
                    np.testing.assert_array_equal(outputs[(True, residual)], outputs[(False, residual)])

    def test_noncrossing_and_crossing_frames_never_use_new_bias(self):
        a, b = Solution(CONFIG, Predictor(True, False)), Solution(CONFIG, Predictor(False, False))
        rows = samples(220)
        # The last frame interval crosses the deadline; all of that interval
        # still uses the old rule even though the observations close calibration.
        for start, end in [(0, 10), (10, 100), (100, 190), (190, 210)]:
            t = float(rows[end - 1, 0])
            np.testing.assert_array_equal(a.step(t, FRAME, rows[start:end]), b.step(t, FRAME, rows[start:end]))
        self.assertEqual(a.gyro_bias_diagnostics["samples"], 200)
        self.assertEqual(a.gyro_bias_diagnostics["activation_after_t"], 1.05)
        a.step(1.1, FRAME, rows[210:]); b.step(1.1, FRAME, rows[210:])
        self.assertFalse(np.array_equal(a.T[:3, :3], b.T[:3, :3]))

    def test_per_sample_debias_respects_rotation_order_extrinsics_and_input_copy(self):
        rows = samples(3)
        rows[:, 4:7] += [[.8, -.3, .2], [-.1, .7, .5], [.4, .2, -.6]]
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = exp_so3([.2, -.4, .1])
        expected = np.eye(3)
        for row in rows:
            expected = expected @ exp_so3((row[4:7] - BIAS) * .005)
        expected = extrinsic[:3, :3] @ expected @ extrinsic[:3, :3].T
        original = rows.copy()
        actual = integrate_debiased_rotation(rows, extrinsic, 0, .015, BIAS)
        np.testing.assert_allclose(actual, expected, atol=1e-14)
        np.testing.assert_array_equal(rows, original)
        np.testing.assert_allclose(integrate_debiased_rotation([], extrinsic, 0, .015, BIAS), np.eye(3), atol=1e-15)
        with self.assertRaisesRegex(ValueError, "future"):
            integrate_debiased_rotation([[.02, 0, 0, 0, 0, 0, 0]], extrinsic, 0, .015, BIAS)

    def test_postwindow_samples_excluded_and_duplicates_not_counted(self):
        rows = samples(210)
        rows[200:, 4:] = 100
        calibration = InitialGyroBias()
        calibration.observe_after_output(1.05, np.repeat(rows, 2, axis=0))
        self.assertEqual(calibration.samples, 200)
        self.assertTrue(calibration.accepted)
        np.testing.assert_allclose(calibration.bias_imu, BIAS, atol=1e-15)

    def test_bad_anchor_missing_count_and_short_span_reject_permanently(self):
        for rows, reason in [(np.empty((0, 7)), "bad_initial_anchor"),
                             (samples(1), "bad_initial_anchor"),
                             (np.repeat(samples(1), 10, axis=0), "bad_initial_anchor"),
                             (samples(179), "insufficient_coverage")]:
            calibration = InitialGyroBias()
            calibration.observe_after_output(1.0, rows)
            self.assertEqual(calibration.reason, reason)
            frozen = calibration.diagnostics()
            calibration.observe_after_output(2.0, samples(400))
            self.assertEqual(calibration.diagnostics(), frozen)
        rows = samples()
        rows[:, 0] *= .8
        calibration = InitialGyroBias()
        calibration.observe_after_output(1, rows)
        self.assertEqual(calibration.reason, "insufficient_coverage")

    def test_variation_and_mean_prior_gates_with_known_constant_motion_limitation(self):
        for channel, amplitude in [(1, 1.01), (4, .201)]:
            rows = samples()
            rows[:, channel] += np.tile([-amplitude, amplitude], 100)
            calibration = InitialGyroBias()
            calibration.observe_after_output(1, rows)
            self.assertEqual(calibration.reason, "variation_gate")
        bound = .02 + 3 * .07 / np.sqrt(200)
        calibration = InitialGyroBias()
        calibration.observe_after_output(1, samples(bias=[bound, 0, 0]))
        self.assertTrue(calibration.accepted)
        calibration = InitialGyroBias()
        calibration.observe_after_output(1, samples(bias=[bound + 1e-6, 0, 0]))
        self.assertEqual(calibration.reason, "mean_prior_gate")
        # A small constant real yaw produces the same gyro observations as bias.
        # Acceptance cannot establish true rest; the documented hover is assumed.
        calibration = InitialGyroBias()
        calibration.observe_after_output(1, samples(bias=[0, 0, .01]))
        self.assertTrue(calibration.accepted)

    def test_nonfinite_future_and_bad_prediction_do_not_advance_calibration(self):
        calibration = InitialGyroBias()
        for rows in [np.array([[.1, 0, 0, 0, np.nan, 0, 0]]), samples(12)]:
            with self.assertRaises(ValueError):
                calibration.observe_after_output(.05, rows)
            self.assertIsNone(calibration.anchor_t)
            self.assertEqual(calibration.samples, 0)
        solution = Solution(CONFIG, Predictor(True))
        rows = samples(20)
        solution.step(.05, FRAME, rows[:10])
        frozen = solution.gyro_bias_diagnostics
        solution.predictor.motion[0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            solution.step(.1, FRAME, rows[10:])
        self.assertEqual(solution.gyro_bias_diagnostics, frozen)
        self.assertEqual(solution.previous_t, .05)

    def test_new_episode_and_default_disabled_contract(self):
        explicit = Solution(CONFIG, Predictor(False))
        legacy = Solution(CONFIG, Predictor(explicit=False))
        for start in range(0, 220, 10):
            rows = samples(220)[start:start + 10]
            np.testing.assert_array_equal(explicit.step(rows[-1, 0], FRAME, rows),
                                          legacy.step(rows[-1, 0], FRAME, rows))
        fresh = Solution(CONFIG, Predictor(True))
        self.assertEqual(fresh.gyro_bias_diagnostics["samples"], 0)
        self.assertEqual(fresh.gyro_bias_diagnostics["status"], "collecting")
        self.assertEqual(validate_motion_options({}), ("image_imu", "images_state", True))
        for settings in [None, True, {"enabled": 1}, {"enabled": "false"}, {"enabled": True, "duration": 2}]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                validate_gyro_bias_options({"gyro_bias": settings})
        with self.assertRaisesRegex(ValueError, "map correction disabled"):
            validate_motion_options({"gyro_bias": {"enabled": True}, "map": {"enabled": True}})

    def test_package_allows_both_calibration_and_residual_switches_without_model_edit(self):
        spec = importlib.util.spec_from_file_location("package_gyro_test", ROOT / "scripts/package_motion.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root / "workspace/solution"; source.mkdir(parents=True)
            (source / "gyro_bias.py").write_text("# source fixture\n")
            model = root / "model"; model.mkdir(); weights = b"weights"
            (model / "net.onnx").write_bytes(weights)
            original = json.dumps({"model_sha256": hashlib.sha256(weights).hexdigest()})
            (model / "meta.json").write_text(original)
            def bundle(command, **kwargs):Path(command[command.index("-o") + 1]).write_bytes(b"wasm")
            args = ["package_motion.py", "--model", str(model), "--name", "calibrated-gyro",
                    "--calibrate-gyro", "--disable-rotation-residual"]
            with patch.object(module, "ROOT", root), patch.object(sys, "argv", args), \
                 patch.object(sys, "path", sys.path.copy()), patch.object(module.subprocess, "run", side_effect=bundle), \
                 contextlib.redirect_stdout(io.StringIO()):module.main()
            staged = json.loads((root / "artifacts/calibrated-gyro-solution/meta.json").read_text())
            self.assertEqual(staged["gyro_bias"], {"enabled": True})
            self.assertIs(staged["rotation_residual_enabled"], False)
            self.assertEqual((model / "meta.json").read_text(), original)
        args = ["package_motion.py", "--model", "unused", "--name", "invalid", "--map", "--calibrate-gyro"]
        with patch.object(sys, "argv", args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            module.main()


if __name__ == "__main__":
    unittest.main()
