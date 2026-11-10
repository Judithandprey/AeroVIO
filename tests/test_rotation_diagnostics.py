"""Synthetic checks for fixed-axis SO(3) diagnostics and controlled-pair guards."""
import copy
import importlib.util
from pathlib import Path
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("rotation_diagnostics", ROOT / "scripts/analyze_rotation_ablation.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def rotations(vectors):
    return np.stack([cv2.Rodrigues(np.asarray(v, dtype=float))[0] for v in vectors])


class RotationDiagnosticsTests(unittest.TestCase):
    def test_constant_signed_bias_accumulates_exactly_about_one_axis(self):
        times = np.arange(201) * 0.05
        truth = np.repeat(np.eye(3)[None], len(times), axis=0)
        estimate = rotations([[0, 0, 0.01 * t] for t in times])
        row = MODULE.rotation_diagnostics(times, truth, estimate)
        self.assertAlmostEqual(row["absolute_final_deg"], np.degrees(0.1), places=8)
        self.assertAlmostEqual(row["absolute_rmse_deg"], np.degrees(np.sqrt(np.mean((0.01 * times) ** 2))), places=8)
        np.testing.assert_allclose(row["increment_error_first_camera"]["duration_weighted_mean_rate_deg_s"], [0, 0, np.degrees(0.01)], atol=1e-9)
        self.assertAlmostEqual(row["increment_error_first_camera"]["directional_coherence"], 1)

    def test_common_axes_transport_changes_camera_x_to_world_y(self):
        rz = cv2.Rodrigues(np.array([0., 0., np.pi / 2]))[0]
        truth = np.stack([rz, rz])
        rx = cv2.Rodrigues(np.array([0.02, 0., 0.]))[0]
        estimate = np.stack([rz, rz @ rx])
        row = MODULE.rotation_diagnostics(np.array([0., 0.05]), truth, estimate)
        np.testing.assert_allclose(row["increment_error_first_camera"]["first_order_sum_deg"], [0, np.degrees(0.02), 0], atol=1e-8)
        np.testing.assert_allclose(row["increment_error_camera_channels"]["first_order_sum_deg"], [np.degrees(0.02), 0, 0], atol=1e-8)

    def test_paired_outputs_recover_residual_with_identical_local_translation(self):
        times = np.arange(21) * 0.05
        on = np.repeat(np.eye(4)[None], len(times), axis=0)
        off = on.copy()
        for k in range(1, len(times)):
            on[k, :3, :3] = on[k-1, :3, :3] @ cv2.Rodrigues(np.array([0., 0., 0.003]))[0]
            off[k, :3, :3] = off[k-1, :3, :3] @ cv2.Rodrigues(np.array([0., 0., 0.002]))[0]
            on[k, :3, 3] = on[k-1, :3, 3] + on[k-1, :3, :3] @ np.array([0.1, 0, 0])
            off[k, :3, 3] = off[k-1, :3, 3] + off[k-1, :3, :3] @ np.array([0.1, 0, 0])
        row = MODULE.paired_diagnostics(times, off, on, off)
        np.testing.assert_allclose(row["recovered_residual_camera_channels"]["duration_weighted_mean_rate_deg_s"], [0, 0, np.degrees(0.02)], atol=1e-8)
        self.assertLess(row["max_local_translation_difference_m"], 1e-12)
        self.assertGreater(row["final_world_position_difference_m"], 0.01)
        np.testing.assert_allclose(row["required_correction_first_camera"]["first_order_sum_deg"], [0, 0, 0], atol=1e-9)

    def test_pair_guard_rejects_weights_map_and_input_changes(self):
        manifest = {key: "same" for key in ["upstream_commit", "runtime_sha256", "selection_sha256", "limits_sha256", "host", "python", "packages"]}
        manifest.update(bundle_files_sha256={"solution.py": "source", "net.onnx": "weights", "meta.json": "meta"}, embedded_model_metadata={"map": {"enabled": False}})
        on = {"manifest": copy.deepcopy(manifest)}
        off = {"manifest": copy.deepcopy(manifest)}
        off["manifest"]["embedded_model_metadata"]["rotation_residual_enabled"] = False
        MODULE.validate_pair(on, off)
        for changed in ["weights", "map", "inputs"]:
            bad = copy.deepcopy(off)
            if changed == "weights": bad["manifest"]["bundle_files_sha256"]["net.onnx"] = "other"
            if changed == "map": bad["manifest"]["embedded_model_metadata"]["map"]["enabled"] = True
            if changed == "inputs": bad["manifest"]["embedded_model_metadata"]["input_mode"] = "imu_only"
            with self.assertRaises(ValueError): MODULE.validate_pair(on, bad)


if __name__ == "__main__":
    unittest.main()
