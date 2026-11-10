"""Bounded, causal sparse local-map correction for the motion-network ablation.

Camera pose T_WC maps camera coordinates into the first-camera/world frame.
Only currently tracked landmarks survive. This is a short local map, without
loop closure, bundle adjustment, ground-truth input, or unseen-surface inference.
"""
import cv2
import numpy as np


def camera_matrix(K, T_WC):
    R_CW = T_WC[:3, :3].T
    return K @ np.column_stack((R_CW, -R_CW @ T_WC[:3, 3]))


def project_points(points, K, T_WC):
    camera = (points - T_WC[:3, 3]) @ T_WC[:3, :3]
    projected = camera @ K.T
    return projected[:, :2] / np.maximum(camera[:, 2:3], 1e-9), camera[:, 2]


class SparseLocalMap:
    def __init__(self, max_tracks=160, max_keyframes=4, keyframe_interval=5,
                 min_baseline=0.03, min_parallax_deg=1.5, min_depth=0.2,
                 max_depth=30.0, min_pnp_points=12, ransac_iterations=40,
                 pnp_reprojection_px=2.0, max_rotation_correction_deg=5.0,
                 max_translation_correction=0.15, correction_step_factor=2.0):
        self.max_tracks = int(max_tracks)
        self.max_keyframes = int(max_keyframes)
        self.keyframe_interval = int(keyframe_interval)
        self.min_baseline = float(min_baseline)
        self.min_parallax = np.deg2rad(min_parallax_deg)
        self.min_depth, self.max_depth = float(min_depth), float(max_depth)
        self.min_pnp_points = int(min_pnp_points)
        self.ransac_iterations = int(ransac_iterations)
        self.pnp_reprojection_px = float(pnp_reprojection_px)
        self.max_rotation_correction = np.deg2rad(max_rotation_correction_deg)
        self.max_translation_correction = float(max_translation_correction)
        self.correction_step_factor = float(correction_step_factor)
        if self.max_tracks < self.min_pnp_points or self.min_pnp_points < 6:
            raise ValueError("max_tracks >= min_pnp_points >= 6 is required")
        if min(self.max_keyframes, self.keyframe_interval, self.ransac_iterations) < 1:
            raise ValueError("map budgets must be positive")
        if not 0 < self.min_depth < self.max_depth:
            raise ValueError("depth bounds must be positive and ordered")
        self.points = np.empty((0, 2), np.float32)
        self.world = np.empty((0, 3), np.float64)
        self.anchor_pixels = np.empty((0, 2), np.float64)
        self.anchor_ids = np.empty(0, np.int64)
        self.anchors = {}
        self.previous_frame = None
        self.previous_pose = None
        self.K = None
        self.frame_index = -1

    def _keep(self, mask):
        self.points = self.points[mask]
        self.world = self.world[mask]
        self.anchor_pixels = self.anchor_pixels[mask]
        self.anchor_ids = self.anchor_ids[mask]

    def _track(self, gray):
        if not len(self.points):
            return
        old = self.points.reshape(-1, 1, 2)
        options = dict(winSize=(15, 15), maxLevel=2,
                       criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 12, 0.01))
        new, status, _ = cv2.calcOpticalFlowPyrLK(self.previous_frame, gray, old, None, **options)
        if new is None or status is None:
            self._keep(np.zeros(len(self.points), bool))
            return
        back, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous_frame, new, None, **options)
        if back is None or back_status is None:
            self._keep(np.zeros(len(self.points), bool))
            return
        new = new.reshape(-1, 2)
        back = back.reshape(-1, 2)
        h, w = gray.shape
        valid = (status.ravel() != 0) & (back_status.ravel() != 0)
        valid &= np.isfinite(new).all(axis=1) & np.isfinite(back).all(axis=1)
        valid &= np.linalg.norm(back - self.points, axis=1) < 1.0
        valid &= (new[:, 0] >= 4) & (new[:, 0] < w - 4)
        valid &= (new[:, 1] >= 4) & (new[:, 1] < h - 4)
        self.points = new.astype(np.float32)
        self._keep(valid)

    def _pose_update(self, K, prior, image_shape):
        """Estimate pose from pre-existing 3D landmarks only; never new triangles."""
        mapped = np.isfinite(self.world).all(axis=1)
        obj = np.ascontiguousarray(self.world[mapped], dtype=np.float64)
        uv = np.ascontiguousarray(self.points[mapped], dtype=np.float64)
        info = {"accepted": False, "inliers": 0, "reason": "insufficient_landmarks"}
        if len(obj) < self.min_pnp_points:
            return prior, info
        # A line of landmarks does not constrain six-degree-of-freedom pose.
        spatial_s = np.linalg.svd(obj - obj.mean(axis=0), compute_uv=False)
        if spatial_s[1] < 1e-4 * max(spatial_s[0], 1e-9):
            info["reason"] = "degenerate_geometry"
            return prior, info
        h, w = image_shape
        if cv2.contourArea(cv2.convexHull(uv.astype(np.float32))) < 0.01 * h * w:
            info["reason"] = "small_image_support"
            return prior, info
        R_CW = prior[:3, :3].T
        initial_r = cv2.Rodrigues(R_CW)[0]
        initial_t = (-R_CW @ prior[:3, 3]).reshape(3, 1)
        # OpenCV's RNG is per process; seed once per frame for reproducible RANSAC.
        cv2.setRNGSeed(741 + self.frame_index)
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj, uv, K, None, initial_r, initial_t, True,
                iterationsCount=self.ransac_iterations,
                reprojectionError=self.pnp_reprojection_px, confidence=0.99,
                flags=cv2.SOLVEPNP_EPNP)
            if not ok or inliers is None:
                info["reason"] = "pnp_failed"
                return prior, info
            index = inliers.ravel()
            info["inliers"] = int(len(index))
            if len(index) < self.min_pnp_points or len(index) < 0.6 * len(obj):
                info["reason"] = "weak_inliers"
                return prior, info
            rvec, tvec = cv2.solvePnPRefineLM(
                obj[index], uv[index], K, None, rvec, tvec,
                criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 5, 1e-5))
            R_CW = cv2.Rodrigues(rvec)[0]
        except cv2.error:
            info["reason"] = "pnp_error"
            return prior, info
        candidate = np.eye(4)
        candidate[:3, :3] = R_CW.T
        candidate[:3, 3] = -R_CW.T @ tvec.ravel()
        if not np.isfinite(candidate).all():
            info["reason"] = "nonfinite_pnp"
            return prior, info
        correction = np.linalg.norm(candidate[:3, 3] - prior[:3, 3])
        predicted_step = 0.0 if self.previous_pose is None else np.linalg.norm(
            prior[:3, 3] - self.previous_pose[:3, 3])
        angle = np.linalg.norm(cv2.Rodrigues(prior[:3, :3].T @ candidate[:3, :3])[0])
        if correction > self.max_translation_correction + self.correction_step_factor * predicted_step:
            info["reason"] = "translation_innovation"
            return prior, info
        if angle > self.max_rotation_correction:
            info["reason"] = "rotation_innovation"
            return prior, info
        before_uv, _ = project_points(obj[index], K, prior)
        after_uv, depths = project_points(obj[index], K, candidate)
        before = float(np.median(np.linalg.norm(before_uv - uv[index], axis=1)))
        after = float(np.median(np.linalg.norm(after_uv - uv[index], axis=1)))
        info.update(reprojection_before_px=before, reprojection_after_px=after,
                    translation_correction=float(correction), rotation_correction=float(angle))
        if np.any(depths <= self.min_depth) or after > self.pnp_reprojection_px:
            info["reason"] = "invalid_projection"
            return prior, info
        if after >= 0.9 * before or before - after < 0.05:
            info["reason"] = "no_material_improvement"
            return prior, info
        # Reject map outliers from future pose updates; retaining them only adds fuel.
        mapped_indices = np.flatnonzero(mapped)
        bad = np.ones(len(obj), bool)
        bad[index] = False
        keep = np.ones(len(self.points), bool)
        keep[mapped_indices[bad]] = False
        self._keep(keep)
        info.update(accepted=True, reason="accepted")
        return candidate, info

    def _triangulate(self, K, T_WC):
        """Triangulate old tracks only when translation and bearing parallax agree."""
        K_inv = np.linalg.inv(K)
        count = 0
        for key in list(self.anchors):
            anchor = self.anchors[key]
            if self.frame_index - key < 2:
                continue
            baseline = np.linalg.norm(T_WC[:3, 3] - anchor[:3, 3])
            if baseline < self.min_baseline:
                continue
            ids = np.flatnonzero((self.anchor_ids == key) & ~np.isfinite(self.world).all(axis=1))
            if not len(ids):
                continue
            a, b = self.anchor_pixels[ids], self.points[ids].astype(np.float64)
            bearing_a = np.column_stack((a, np.ones(len(a)))) @ K_inv.T @ anchor[:3, :3].T
            bearing_b = np.column_stack((b, np.ones(len(b)))) @ K_inv.T @ T_WC[:3, :3].T
            bearing_a /= np.linalg.norm(bearing_a, axis=1)[:, None]
            bearing_b /= np.linalg.norm(bearing_b, axis=1)[:, None]
            # Bearings in one world frame remove the apparent motion due to rotation.
            parallax = np.arccos(np.clip(np.sum(bearing_a * bearing_b, axis=1), -1, 1))
            ids, a, b = ids[parallax >= self.min_parallax], a[parallax >= self.min_parallax], b[parallax >= self.min_parallax]
            if not len(ids):
                continue
            homogeneous = cv2.triangulatePoints(camera_matrix(K, anchor), camera_matrix(K, T_WC), a.T, b.T).T
            good_w = np.abs(homogeneous[:, 3]) > 1e-9
            xyz = np.full((len(ids), 3), np.nan)
            xyz[good_w] = homogeneous[good_w, :3] / homogeneous[good_w, 3:4]
            reproj_a, depth_a = project_points(xyz, K, anchor)
            reproj_b, depth_b = project_points(xyz, K, T_WC)
            good = np.isfinite(xyz).all(axis=1)
            good &= (depth_a > self.min_depth) & (depth_b > self.min_depth)
            good &= (depth_a < self.max_depth) & (depth_b < self.max_depth)
            good &= np.linalg.norm(reproj_a - a, axis=1) < 1.5
            good &= np.linalg.norm(reproj_b - b, axis=1) < 1.5
            self.world[ids[good]] = xyz[good]
            self.anchor_ids[ids[good]] = -1
            count += int(good.sum())
        return count

    def _replenish(self, gray, pose):
        if len(self.points) >= self.max_tracks:
            return 0
        mask = np.full(gray.shape, 255, np.uint8)
        for x, y in self.points:
            cv2.circle(mask, (int(round(x)), int(round(y))), 7, 0, -1)
        new = cv2.goodFeaturesToTrack(gray, maxCorners=self.max_tracks - len(self.points),
                                     qualityLevel=0.01, minDistance=7, mask=mask, blockSize=5)
        if new is None:
            return 0
        new = new.reshape(-1, 2)
        self.anchors[self.frame_index] = pose.copy()
        self.points = np.concatenate((self.points, new.astype(np.float32)))
        self.world = np.concatenate((self.world, np.full((len(new), 3), np.nan)))
        self.anchor_pixels = np.concatenate((self.anchor_pixels, new.astype(np.float64)))
        self.anchor_ids = np.concatenate((self.anchor_ids, np.full(len(new), self.frame_index, np.int64)))
        while len(self.anchors) > self.max_keyframes:
            oldest = next(iter(self.anchors))
            del self.anchors[oldest]
            self._keep(self.anchor_ids != oldest)
        return len(new)

    def update(self, gray, K, T_WC_pred):
        gray = np.ascontiguousarray(gray, dtype=np.uint8)
        K, prior = np.asarray(K, np.float64), np.asarray(T_WC_pred, np.float64)
        if gray.ndim != 2 or K.shape != (3, 3) or prior.shape != (4, 4):
            raise ValueError("expected gray HxW, K 3x3, and T_WC 4x4")
        if not np.isfinite(K).all() or not np.isfinite(prior).all() or min(K[0, 0], K[1, 1]) <= 0:
            raise ValueError("camera parameters and pose must be finite; focal lengths positive")
        if self.K is not None and (gray.shape != self.previous_frame.shape or not np.array_equal(K, self.K)):
            raise ValueError("camera configuration changed inside an episode; create a new map")
        self.frame_index += 1
        if self.previous_frame is not None:
            self._track(gray)
        corrected, info = self._pose_update(K, prior.copy(), gray.shape)
        # Correction happens before triangulation, preventing self-confirmation on the creation frame.
        created = self._triangulate(K, corrected) if self.frame_index % self.keyframe_interval == 0 else 0
        if self.frame_index % self.keyframe_interval == 0 or len(self.points) < self.min_pnp_points:
            added = self._replenish(gray, corrected)
        else:
            added = 0
        self.previous_frame = gray.copy()
        self.previous_pose = corrected.copy()
        self.K = K.copy()
        info.update(tracks=int(len(self.points)), landmarks=int(np.isfinite(self.world).all(axis=1).sum()),
                    triangulated=int(created), added_tracks=int(added), keyframes=int(len(self.anchors)))
        return corrected, info
