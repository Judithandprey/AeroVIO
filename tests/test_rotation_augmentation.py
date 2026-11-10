"""Independent ray geometry, IMU mechanics and provenance-friendly determinism."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.rotation_augmentation import (camera_matrix, exp_so3, make_rotation_pair,
                                             project_pixels, sample_rotation, synthetic_imu,
                                             virtual_view)


def configuration():
    return {"width": 320, "height": 240, "fx": 110., "fy": 110., "cx": 159.5, "cy": 119.5,
            "T_cam_imu": np.eye(4).reshape(-1).tolist()}


def sensor():
    return {"rate_hz": 200., "gravity": [0., 0., -9.81],
            "accel_bias_init_max": .2, "gyro_bias_init_max": .02,
            "accel_bias_random_walk": .001, "gyro_bias_random_walk": .0006,
            "accel_noise_std": .28, "gyro_noise_std": .07}


class RotationAugmentationTests(unittest.TestCase):
    def test_yaw_homography_matches_independent_3d_rays_and_direction(self):
        angle = .04
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
        config = configuration()
        K = camera_matrix(config)
        pixels = np.array([[159.5, 119.5], [80., 40.], [245., 180.]])
        projected, _ = project_pixels(K @ rotation @ np.linalg.inv(K), pixels)
        for index, (u, v) in enumerate(pixels):
            x, y = (u - 159.5) / 110., (v - 119.5) / 110.
            rotated = np.array([c * x + s, y, -s * x + c])
            expected = rotated[:2] / rotated[2] * 110 + [159.5, 119.5]
            np.testing.assert_allclose(projected[index], expected, atol=1e-12)
        self.assertGreater(projected[0, 0], 159.5)  # target centre samples to source right
        self.assertGreater(abs(projected[1, 1] - pixels[1, 1]), .5)  # yaw is not an image-plane spin/shift

    def test_supported_views_preserve_constant_pixels_without_border_fill(self):
        source = np.full((240, 320, 3), 173, dtype=np.uint8)
        row = make_rotation_pair(source, configuration(), np.eye(3), sensor(),
                                 np.array([.02, -.03, .01]), np.random.default_rng(1))
        np.testing.assert_allclose(row["images"], 173 / 255, atol=1e-7)
        self.assertGreaterEqual(row["source_clearance_pixels"], 1.)
        self.assertGreater(row["visibility"].mean(), .9)
        self.assertAlmostEqual(float(row["state"][9]), 110 / .85 / 320, places=7)
        self.assertAlmostEqual(float(row["intrinsics"][0]), 110 / .85 / 5, places=5)

    def test_missing_source_rays_are_rejected_not_padded(self):
        K = camera_matrix(configuration())
        with self.assertRaisesRegex(ValueError, "outside source support"):
            virtual_view(np.full((240, 320, 3), 173, np.uint8), K, K, exp_so3([0., .1, 0.]))

    def test_corner_certificate_bounds_all_interior_rays(self):
        K = camera_matrix(configuration())
        Kv = K.copy()
        Kv[0, 0] /= .85
        Kv[1, 1] /= .85
        R = exp_so3([.01, -.02, .015])
        _, clearance = virtual_view(np.ones((240, 320, 3), np.uint8), K, Kv, R)
        yy, xx = np.mgrid[:240, :320]
        projected, z = project_pixels(K @ R @ np.linalg.inv(Kv), np.column_stack((xx.ravel(), yy.ravel())))
        measured = min(projected[:, 0].min(), projected[:, 1].min(), 319 - projected[:, 0].max(), 239 - projected[:, 1].max())
        self.assertAlmostEqual(clearance, measured, places=10)
        self.assertTrue((z > 0).all())

    def test_stationary_sensor_measures_specific_force_not_zero(self):
        rows = synthetic_imu(np.zeros(3), np.eye(3), np.eye(4), sensor(), np.random.default_rng(2), noise=False)
        np.testing.assert_allclose(rows[:, 1:4], np.tile([0., 0., 9.81], (10, 1)), atol=1e-12)
        np.testing.assert_array_equal(rows[:, 4:7], np.zeros((10, 3)))
        self.assertEqual(rows.shape, (10, 7))
        self.assertAlmostEqual(rows[-1, 0], .05)

    def test_gravity_rotates_at_each_imu_sample(self):
        rows = synthetic_imu(np.array([.1, 0., 0.]), np.eye(3), np.eye(4), sensor(), np.random.default_rng(2), noise=False)
        angles = rows[:, 0] * 2.
        expected = np.column_stack((np.zeros(10), 9.81 * np.sin(angles), 9.81 * np.cos(angles)))
        np.testing.assert_allclose(rows[:, 1:4], expected, atol=1e-12)
        np.testing.assert_allclose(rows[:, 4:7], np.tile([2., 0., 0.], (10, 1)), atol=1e-12)

    def test_imu_lever_arm_centripetal_acceleration_has_correct_sign(self):
        transform = np.eye(4)
        transform[0, 3] = 1.
        priors = {**sensor(), "gravity": [0., 0., 0.]}
        rows = synthetic_imu(np.array([0., 0., .05]), np.eye(3), transform, priors, np.random.default_rng(2), noise=False)
        np.testing.assert_allclose(rows[:, 1:4], np.tile([-1., 0., 0.], (10, 1)), atol=1e-12)

    def test_true_rotation_and_imu_residual_compose_with_camera_extrinsic(self):
        config = configuration()
        transform = np.eye(4)
        transform[:3, :3] = exp_so3([.2, -.1, .3])
        config["T_cam_imu"] = transform.reshape(-1).tolist()
        vector = np.array([.02, -.03, .01])
        for noisy in (False, True):
            row = make_rotation_pair(np.full((240, 320, 3), 127, np.uint8), config,
                                     exp_so3([.1, .2, -.3]), sensor(), vector,
                                     np.random.default_rng(128), noise=noisy)
            np.testing.assert_allclose(row["rotation_gt"], exp_so3(vector), atol=4e-8)
            reconstructed = exp_so3(row["state"][:3]) @ exp_so3(row["rotation_residual"])
            np.testing.assert_allclose(reconstructed, exp_so3(vector), atol=2e-8)
            if noisy:
                self.assertGreater(np.linalg.norm(row["rotation_residual"]), 1e-5)
            else:
                self.assertLess(np.linalg.norm(row["rotation_residual"]), 1e-7)

    def test_seed_reproduces_images_labels_and_independent_sensor_noise(self):
        source = np.random.default_rng(4).integers(0, 256, (240, 320, 3), dtype=np.uint8)
        first = make_rotation_pair(source, configuration(), np.eye(3), sensor(), [.02, .01, -.01], np.random.default_rng(5))
        second = make_rotation_pair(source, configuration(), np.eye(3), sensor(), [.02, .01, -.01], np.random.default_rng(5))
        changed_noise = make_rotation_pair(source, configuration(), np.eye(3), sensor(), [.02, .01, -.01], np.random.default_rng(6))
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        np.testing.assert_array_equal(first["images"], changed_noise["images"])
        np.testing.assert_array_equal(first["rotation_gt"], changed_noise["rotation_gt"])
        self.assertFalse(np.array_equal(first["state"], changed_noise["state"]))

    def test_axis_sampling_uses_actual_camera_axes_and_bounded_angles(self):
        rng = np.random.default_rng(7)
        for axis_id in range(4):
            for _ in range(20):
                vector = sample_rotation(rng, axis_id)
                self.assertLessEqual(np.linalg.norm(vector), np.deg2rad(3.))
                if axis_id < 3:
                    self.assertEqual(np.count_nonzero(vector), 1)
                    self.assertNotEqual(vector[axis_id], 0.)


if __name__ == "__main__":
    unittest.main()
