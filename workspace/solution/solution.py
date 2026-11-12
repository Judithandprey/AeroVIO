"""Causal small-network visual-inertial odometry for the viobench ABI.

The network predicts camera translation and an IMU rotation residual. Pose is
T_camera0_current. Twist is the finite-interval mean in current camera axes,
not an instantaneous velocity measurement. Optional sparse-map correction uses
the same network weights and is applied before recomputing pose and twist.
"""
import cv2
import numpy as np

from motion_inference import MotionPredictor, validate_motion_options, validate_gyro_bias_options
from gyro_bias import InitialGyroBias, integrate_debiased_rotation
from preintegrate import camera_features, exp_so3, log_so3, make_state, prepare_frame


def quaternion_from_matrix(rotation):
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = np.sqrt(trace + 1.0) * 2.0
        q = np.array([(rotation[2, 1] - rotation[1, 2]) / scale,
                      (rotation[0, 2] - rotation[2, 0]) / scale,
                      (rotation[1, 0] - rotation[0, 1]) / scale, 0.25 * scale])
    else:
        i = int(np.argmax(np.diag(rotation)))
        j, k = (i + 1) % 3, (i + 2) % 3
        scale = np.sqrt(max(0.0, 1 + rotation[i, i] - rotation[j, j] - rotation[k, k])) * 2
        q = np.empty(4)
        q[i] = 0.25 * scale
        q[j] = (rotation[j, i] + rotation[i, j]) / scale
        q[k] = (rotation[k, i] + rotation[i, k]) / scale
        q[3] = (rotation[k, j] - rotation[j, k]) / scale
    q /= np.linalg.norm(q)
    return q if q[3] >= 0 else -q


class Solution:
    frame_width = 320

    def __init__(self, config, predictor=None):
        cv2.setRNGSeed(0)
        self.config = dict(config)
        camera_features(config)  # validate intrinsics before accepting observations
        self.K = np.array([[config["fx"], 0, config["cx"]],
                           [0, config["fy"], config["cy"]], [0, 0, 1]], dtype=np.float64)
        self.T_cam_imu = np.asarray(config["T_cam_imu"], dtype=np.float64).reshape(4, 4)
        if not np.isfinite(self.T_cam_imu).all():
            raise ValueError("camera/IMU transform must be finite")
        if predictor is None:
            import solution_bundle as bundle
            predictor = MotionPredictor(bundle)
        self.predictor = predictor
        _, network_inputs, self.rotation_residual_enabled = validate_motion_options(predictor.meta)
        self.uses_image_pair = network_inputs == "images_state"
        self.gyro_calibration = InitialGyroBias() if validate_gyro_bias_options(predictor.meta) else None
        self.size = tuple(predictor.size)
        self.previous = None
        self.previous_t = None
        self.T = np.eye(4)
        self.frames = 0
        self.last_map_diagnostics = None
        self.map = None
        settings = dict(predictor.meta.get("map", {}))
        if settings.pop("enabled", False):
            from local_map import SparseLocalMap
            self.map = SparseLocalMap(**settings)
        self.diagnostics_interval = int(predictor.meta.get("diagnostics_interval", 0))

    @property
    def gyro_bias_diagnostics(self):
        if self.gyro_calibration is None:
            return {"enabled": False, "status": "disabled", "bias_imu": [0.0, 0.0, 0.0]}
        return self.gyro_calibration.diagnostics()

    def step(self, t, frame, imu):
        t = float(t)
        if not np.isfinite(t):
            raise ValueError("frame time must be finite")
        if self.previous_t is not None and t <= self.previous_t:
            raise ValueError("frame times must strictly increase")
        small = prepare_frame(frame, self.size) if self.uses_image_pair else None
        if self.previous_t is None:
            if self.map is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                self.T, self.last_map_diagnostics = self.map.update(gray, self.K, self.T.copy())
            if self.gyro_calibration is not None:
                self.gyro_calibration.observe_after_output(t, imu)
            self.previous, self.previous_t = small, t
            self.frames += 1
            return np.array([0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0, 0])
        interval = t - self.previous_t
        state = make_state(imu, self.T_cam_imu, self.config, self.previous_t, t)
        delta = np.eye(4)
        delta[:3, :3] = exp_so3(state[:3].astype(np.float64))
        if self.gyro_calibration is not None and self.gyro_calibration.accepted:
            delta[:3, :3] = integrate_debiased_rotation(imu, self.T_cam_imu, self.previous_t, t,
                                                       self.gyro_calibration.bias_imu)
        pair = np.stack([self.previous, small]) if self.uses_image_pair else None
        if getattr(self.predictor, "requires_rotation_prior", False):
            prediction = self.predictor.predict(pair, state, rotation_prior=log_so3(delta[:3, :3]))
        else:
            prediction = self.predictor.predict(pair, state)
        motion = np.asarray(prediction, dtype=np.float64)
        if motion.shape != (6,) or not np.isfinite(motion).all():
            raise ValueError("motion prediction must contain six finite values")
        if self.rotation_residual_enabled:
            delta[:3, :3] = delta[:3, :3] @ exp_so3(motion[3:])
        delta[:3, 3] = motion[:3]
        previous_pose = self.T
        pose = previous_pose @ delta
        if self.map is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            pose, self.last_map_diagnostics = self.map.update(gray, self.K, pose)
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("pose update must be a finite 4x4 transform")
        # Compute twist from the final corrected pose, in the current camera axes.
        rotation_delta = previous_pose[:3, :3].T @ pose[:3, :3]
        velocity = pose[:3, :3].T @ (pose[:3, 3] - previous_pose[:3, 3]) / interval
        angular_velocity = log_so3(rotation_delta) / interval
        result = np.concatenate([pose[:3, 3], quaternion_from_matrix(pose[:3, :3]),
                                 velocity, angular_velocity])
        if not np.isfinite(result).all():
            raise ValueError("non-finite ABI output")
        # Freeze calibration only after this frame's output has been computed.
        # Even the deadline/crossing frame uses the previously active rotation.
        if self.gyro_calibration is not None:
            self.gyro_calibration.observe_after_output(t, imu)
        self.T = pose.copy()
        self.previous, self.previous_t = small, t
        self.frames += 1
        if self.map is not None and self.diagnostics_interval > 0 and self.frames % self.diagnostics_interval == 0:
            print("map", self.frames, self.last_map_diagnostics)
        return result
