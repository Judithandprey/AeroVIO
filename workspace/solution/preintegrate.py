"""Camera-frame IMU features shared by training and the WASM estimator.

Adapted from the pinned viobench tiny_cnn example. Unlike that example, the
previous frame timestamp is explicit so the first IMU sample interval is kept.
Acceleration is raw specific force (gravity included), not world acceleration.
"""
import cv2
import numpy as np

# Matches upstream Episode.imu_window boundary tolerance for stored timestamps.
IMU_TIMESTAMP_TOLERANCE_SECONDS = 1e-6


def exp_so3(vector):
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    if not np.isfinite(vector).all():
        raise ValueError("rotation vector must be finite")
    return cv2.Rodrigues(vector)[0]


def log_so3(rotation):
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(rotation).all():
        raise ValueError("rotation matrix must be finite")
    return cv2.Rodrigues(rotation)[0].reshape(3)


def preintegrate(window, T_cam_imu, start_t, end_t):
    """Return [rotvec(3), dv(3), dp(3)] in the previous camera axes.

    Each row is [timestamp, accel(3), gyro(3)] at its interval's right endpoint.
    Empty windows produce zero features. Trailing time without an observation
    is not extrapolated. The camera lever arm is not part of these raw IMU
    features; the learned translation target is at the camera centre.
    """
    if not np.isfinite([start_t, end_t]).all() or end_t <= start_t:
        raise ValueError("IMU interval must have finite increasing endpoints")
    window = np.asarray(window, dtype=np.float64).reshape(-1, 7)
    if not np.isfinite(window).all():
        raise ValueError("IMU observations must be finite")
    if len(window) and np.any(window[:, 0] > end_t + IMU_TIMESTAMP_TOLERANCE_SECONDS):
        raise ValueError("IMU window contains a future observation")
    R_ci = np.asarray(T_cam_imu, dtype=np.float64).reshape(4, 4)[:3, :3]
    rotation = np.eye(3)
    velocity = np.zeros(3)
    position = np.zeros(3)
    previous = float(start_t)
    for row in window:
        timestamp = min(float(row[0]), float(end_t))
        dt = timestamp - previous
        if dt <= 0:
            continue
        acceleration = rotation @ row[1:4]
        position += velocity * dt + 0.5 * acceleration * dt * dt
        velocity += acceleration * dt
        rotation = rotation @ exp_so3(row[4:7] * dt)
        previous = timestamp
        if timestamp >= end_t:
            break
    camera_rotation = R_ci @ rotation @ R_ci.T
    return np.concatenate([log_so3(camera_rotation), R_ci @ velocity,
                           R_ci @ position]).astype(np.float32)


def camera_features(config):
    """Focal lengths and principal point in frame-size-normalized coordinates."""
    width, height = float(config["width"]), float(config["height"])
    if width <= 0 or height <= 0:
        raise ValueError("camera dimensions must be positive")
    features = np.array([config["fx"] / width, config["fy"] / height,
                         (config["cx"] + 0.5) / width,
                         (config["cy"] + 0.5) / height], dtype=np.float32)
    if not np.isfinite(features).all() or np.any(features[:2] <= 0):
        raise ValueError("camera intrinsics must be finite with positive focal lengths")
    return features


def make_state(window, T_cam_imu, config, start_t, end_t):
    """Network state: IMU9, normalized camera intrinsics4, frame interval1."""
    return np.concatenate([preintegrate(window, T_cam_imu, start_t, end_t),
                           camera_features(config), [end_t - start_t]]).astype(np.float32)


def prepare_frame(frame, size):
    """RGB uint8 frame -> gray float32 image [0,1] at (width,height)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, tuple(size), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32) / 255.0
