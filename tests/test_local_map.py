"""Synthetic geometric and image-sequence checks; no training/evaluation GT enters runtime."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workspace/solution"))
from local_map import SparseLocalMap


K = np.array([[220.0, 0, 159.5], [0, 220.0, 119.5], [0, 0, 1]])


def pose(x=0.0, y=0.0, z=0.0, rotation=(0.0, 0.0, 0.0)):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rotation, np.float64))[0]
    T[:3, 3] = [x, y, z]
    return T


def observe(xyz, T):
    # Independent synthetic camera: transform homogeneous world points then divide.
    T_CW = np.linalg.inv(T)
    camera = (T_CW @ np.column_stack((xyz, np.ones(len(xyz)))).T).T[:, :3]
    uvh = (K @ camera.T).T
    return uvh[:, :2] / uvh[:, 2:3]


def scene(n=60):
    rng = np.random.default_rng(1907)
    return np.column_stack((rng.uniform(-1.8, 1.8, n), rng.uniform(-1.2, 1.2, n), rng.uniform(3, 6, n)))


def seeded_tracks(mapper, xyz, current, anchor=None, mapped=False):
    anchor = np.eye(4) if anchor is None else anchor
    mapper.points = observe(xyz, current).astype(np.float32)
    mapper.anchor_pixels = observe(xyz, anchor)
    mapper.anchor_ids = np.full(len(xyz), -1 if mapped else 0, np.int64)
    mapper.world = xyz.copy() if mapped else np.full_like(xyz, np.nan)
    mapper.anchors = {0: anchor.copy()}
    mapper.frame_index = 5


class LocalMapTests(unittest.TestCase):
    def test_triangulation_recovers_known_nonplanar_scene(self):
        xyz = scene()
        current = pose(0.28, 0.03, rotation=(0.01, 0.04, -0.01))
        mapper = SparseLocalMap()
        seeded_tracks(mapper, xyz, current)
        count = mapper._triangulate(K, current)
        self.assertGreater(count, 45)
        valid = np.isfinite(mapper.world).all(axis=1)
        np.testing.assert_allclose(mapper.world[valid], xyz[valid], atol=2e-5)

    def test_rotation_motion_does_not_masquerade_as_depth(self):
        xyz = scene()
        actual = pose(rotation=(0.0, 0.15, 0.0))
        mapper = SparseLocalMap()
        seeded_tracks(mapper, xyz, actual)
        self.assertGreater(np.median(np.linalg.norm(mapper.points - mapper.anchor_pixels, axis=1)), 20)
        # Even a bad translation prior cannot turn rotational image flow into parallax.
        incorrect_translation_prior = actual.copy()
        incorrect_translation_prior[0, 3] = 0.3
        self.assertEqual(mapper._triangulate(K, incorrect_translation_prior), 0)
        self.assertFalse(np.isfinite(mapper.world).any())

    def test_tiny_baseline_withholds_depth(self):
        xyz = scene()
        current = pose(0.001)
        mapper = SparseLocalMap()
        seeded_tracks(mapper, xyz, current)
        self.assertEqual(mapper._triangulate(K, current), 0)

    def test_pnp_corrects_known_pose_with_partial_bad_correspondences(self):
        xyz = scene()
        actual = pose(0.22, -0.02, rotation=(0.01, 0.04, -0.005))
        prior = pose(0.29, 0.01, rotation=(0.015, 0.055, -0.005))
        mapper = SparseLocalMap()
        seeded_tracks(mapper, xyz, actual, mapped=True)
        mapper.points[:8] += [45, -30]
        mapper.previous_pose = pose(0.18, -0.02)
        corrected, info = mapper._pose_update(K, prior, (240, 320))
        self.assertTrue(info['accepted'], info)
        self.assertGreaterEqual(info['inliers'], 45)
        np.testing.assert_allclose(corrected, actual, atol=1e-5)
        self.assertLess(info['reprojection_after_px'], info['reprojection_before_px'])
        self.assertLessEqual(len(mapper.points), 52)

    def test_large_innovation_is_rejected_without_changing_prior(self):
        xyz = scene()
        actual = pose(0.2)
        prior = pose(1.2)
        mapper = SparseLocalMap()
        seeded_tracks(mapper, xyz, actual, mapped=True)
        mapper.previous_pose = pose(1.19)
        corrected, info = mapper._pose_update(K, prior, (240, 320))
        self.assertFalse(info['accepted'])
        self.assertEqual(info['reason'], 'translation_innovation')
        np.testing.assert_array_equal(corrected, prior)

    def test_collinear_landmarks_cannot_trigger_pose_update(self):
        xyz = np.column_stack((np.linspace(-1, 1, 30), np.zeros(30), np.full(30, 4.0)))
        mapper = SparseLocalMap()
        seeded_tracks(mapper, xyz, pose(0.2), mapped=True)
        _, info = mapper._pose_update(K, pose(0.23), (240, 320))
        self.assertEqual(info['reason'], 'degenerate_geometry')

    def test_causal_image_tracking_map_and_partial_visibility(self):
        rng = np.random.default_rng(32)
        image = rng.integers(0, 256, (240, 320), dtype=np.uint8)
        image = cv2.GaussianBlur(image, (3, 3), 0.7)
        mapper = SparseLocalMap()
        accepted = False
        max_landmarks = 0
        for index in range(9):
            actual = pose(index * 0.045)
            # A textured stationary plane at Z=4, seen under exact camera translation.
            H = K @ np.column_stack((np.eye(3)[:, :2], np.array([-actual[0, 3] / 4, 0, 1]))) @ np.linalg.inv(K)
            frame = cv2.warpPerspective(image, H, (320, 240))
            if index >= 6:
                frame[:, :80] = 0  # Newly hidden region has no valid correspondences.
            prior = actual.copy()
            if index >= 6:
                prior[:3, 3] += [0.06, 0.02, 0]
            corrected, info = mapper.update(frame, K, prior)
            self.assertLessEqual(info['tracks'], mapper.max_tracks)
            self.assertLessEqual(info['keyframes'], mapper.max_keyframes)
            if index == 5:
                self.assertFalse(info['accepted'])  # New triangles cannot correct their creation frame.
            if info['accepted']:
                accepted = True
                self.assertLess(np.linalg.norm(corrected[:3, 3] - actual[:3, 3]), 0.04)
            max_landmarks = max(max_landmarks, info['landmarks'])
        self.assertGreater(max_landmarks, 30)
        self.assertTrue(accepted, 'Expected correction after map initialization and partial occlusion')

    def test_blank_frames_reset_and_no_cross_instance_state(self):
        blank = np.zeros((240, 320), np.uint8)
        first, second = SparseLocalMap(), SparseLocalMap()
        for index in range(25):
            prior = pose(index * 0.01)
            corrected, info = first.update(blank, K, prior)
            np.testing.assert_array_equal(corrected, prior)
            self.assertEqual(info['landmarks'], 0)
        self.assertIsNone(second.previous_pose)
        corrected, _ = second.update(blank, K, np.eye(4))
        np.testing.assert_array_equal(corrected, np.eye(4))

    def test_camera_change_and_nonfinite_pose_are_rejected(self):
        mapper = SparseLocalMap()
        image = np.zeros((240, 320), np.uint8)
        mapper.update(image, K, np.eye(4))
        changed = K.copy()
        changed[0, 0] += 1
        with self.assertRaises(ValueError):
            mapper.update(image, changed, np.eye(4))
        invalid = np.eye(4)
        invalid[0, 3] = np.nan
        with self.assertRaises(ValueError):
            mapper.update(image, K, invalid)


if __name__ == '__main__':
    unittest.main()
