"""Training-only pure camera-centre SO(3) reprojection and coherent virtual IMU.

Both views come from one existing RGB observation. They reveal no scene outside
that observation. Cropping supplies a margin; pairs with any unsupported rays
are rejected rather than filled. Synthetic acceleration follows a fixed camera
centre, gravity and the actual IMU lever arm. Only rotation is a training label.
"""
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workspace/solution"))
from preintegrate import exp_so3, log_so3, make_state, prepare_frame


def camera_matrix(config):
    return np.array([[config["fx"], 0., config["cx"]],
                     [0., config["fy"], config["cy"]], [0., 0., 1.]], dtype=np.float64)


def project_pixels(homography, pixels):
    homogeneous = homography @ np.vstack((np.asarray(pixels).T, np.ones(len(pixels))))
    return (homogeneous[:2] / homogeneous[2:]).T, homogeneous[2]


def virtual_view(source, source_K, virtual_K, source_from_virtual, margin=1.):
    """Backward map: u_source ~ K_source R_source_virtual inv(K_virtual) u_virtual."""
    height, width = source.shape[:2]
    homography = source_K @ source_from_virtual @ np.linalg.inv(virtual_K)
    corners = np.array([[0., 0.], [width - 1., 0.], [width - 1., height - 1.], [0., height - 1.]])
    projected, z = project_pixels(homography, corners)
    # With positive projective denominator throughout the rectangle, each
    # coordinate's extrema are at a corner; this certifies every output pixel.
    lower = projected.min(0)
    upper = projected.max(0)
    clearance = min(lower[0], lower[1], width - 1 - upper[0], height - 1 - upper[1])
    if not np.isfinite(projected).all() or z.min() <= 0 or clearance < margin:
        raise ValueError("virtual view includes rays outside source support")
    result = cv2.warpPerspective(source, homography, (width, height),
                                 flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return result, float(clearance)


def synthetic_imu(rotation_vector, world_from_previous, T_cam_imu, imu_config, rng,
                  duration=.05, noise=True):
    """Right-endpoint IMU samples for constant angular velocity about camera centre.

    T_cam_imu maps IMU coordinates into camera coordinates; its translation is
    the IMU location relative to the stationary camera centre. Bias is sampled
    once per pair, then evolves by the configured random walk. Noise and bias
    are independent of image texture and the sampled true rotation.
    """
    rate = float(imu_config["rate_hz"])
    steps = int(round(duration * rate))
    if steps < 1 or duration <= 0:
        raise ValueError("positive duration and at least one IMU sample required")
    sample_dt = duration / steps
    transform = np.asarray(T_cam_imu, dtype=np.float64).reshape(4, 4)
    R_ci, lever = transform[:3, :3], transform[:3, 3]
    omega = np.asarray(rotation_vector, dtype=np.float64) / duration
    gravity_previous = np.asarray(world_from_previous).T @ np.asarray(imu_config["gravity"])
    centripetal_camera = np.cross(omega, np.cross(omega, lever))
    accel_bias = rng.uniform(-imu_config["accel_bias_init_max"], imu_config["accel_bias_init_max"], 3) if noise else np.zeros(3)
    gyro_bias = rng.uniform(-imu_config["gyro_bias_init_max"], imu_config["gyro_bias_init_max"], 3) if noise else np.zeros(3)
    rows = []
    for index in range(1, steps + 1):
        timestamp = index * sample_dt
        R_previous_current = exp_so3(omega * timestamp)
        force_camera = centripetal_camera - R_previous_current.T @ gravity_previous
        accel = R_ci.T @ force_camera
        gyro = R_ci.T @ omega
        if noise:
            accel_bias += rng.normal(0., imu_config["accel_bias_random_walk"] * np.sqrt(sample_dt), 3)
            gyro_bias += rng.normal(0., imu_config["gyro_bias_random_walk"] * np.sqrt(sample_dt), 3)
            accel = accel + accel_bias + rng.normal(0., imu_config["accel_noise_std"], 3)
            gyro = gyro + gyro_bias + rng.normal(0., imu_config["gyro_noise_std"], 3)
        rows.append(np.concatenate(([timestamp], accel, gyro)))
    return np.asarray(rows, dtype=np.float64)


def sample_rotation(rng, axis_id, max_angle_degrees=3.):
    """Balanced x/y/z/mixed classes with independent sign and uniform magnitude."""
    if axis_id not in (0, 1, 2, 3):
        raise ValueError("axis_id must be x=0, y=1, z=2 or mixed=3")
    axis = np.eye(3)[axis_id] if axis_id < 3 else rng.normal(size=3)
    axis = axis / np.linalg.norm(axis)
    magnitude = np.deg2rad(rng.uniform(0., max_angle_degrees))
    return axis * magnitude * rng.choice([-1., 1.])


def make_rotation_pair(source_rgb, source_config, world_from_source, imu_config,
                       rotation_vector, rng, crop=.85, duration=.05, noise=True):
    """Return images/state/exact R plus overlap evidence; no translation label."""
    if source_rgb.shape != (240, 320, 3) or not 0 < crop < 1:
        raise ValueError("use upstream-adapted 320x240 RGB and a crop in (0,1)")
    source_K = camera_matrix(source_config)
    virtual_K = source_K.copy()
    virtual_K[0, 0] /= crop
    virtual_K[1, 1] /= crop
    # Q maps each virtual camera's coordinates into the original source camera.
    Q_previous = exp_so3(-np.asarray(rotation_vector) / 2.)
    Q_current = exp_so3(np.asarray(rotation_vector) / 2.)
    previous_rgb, previous_clearance = virtual_view(source_rgb, source_K, virtual_K, Q_previous)
    current_rgb, current_clearance = virtual_view(source_rgb, source_K, virtual_K, Q_current)
    R_previous_current = Q_previous.T @ Q_current
    virtual_config = {**source_config, "fx": float(virtual_K[0, 0]), "fy": float(virtual_K[1, 1])}
    imu = synthetic_imu(rotation_vector, np.asarray(world_from_source) @ Q_previous,
                        source_config["T_cam_imu"], imu_config, rng, duration, noise)
    state = make_state(imu, source_config["T_cam_imu"], virtual_config, 0., duration)
    residual = log_so3(exp_so3(state[:3]).T @ R_previous_current).astype(np.float32)
    images = np.stack((prepare_frame(previous_rgb, (64, 48)), prepare_frame(current_rgb, (64, 48))))
    small_K = virtual_K.copy()
    small_K[:2, :2] /= 5.
    small_K[0, 2] = (virtual_K[0, 2] + .5) / 5. - .5
    small_K[1, 2] = (virtual_K[1, 2] + .5) / 5. - .5
    yy, xx = np.mgrid[:48, :64]
    previous_pixels, z = project_pixels(small_K @ R_previous_current @ np.linalg.inv(small_K),
                                        np.column_stack((xx.ravel(), yy.ravel())))
    visible = ((z > 0) & (previous_pixels[:, 0] >= 0) & (previous_pixels[:, 0] <= 63) &
               (previous_pixels[:, 1] >= 0) & (previous_pixels[:, 1] <= 47)).reshape(1, 48, 64)
    return {"images": images, "state": state, "rotation_gt": R_previous_current.astype(np.float32),
            "rotation_residual": residual, "visibility": visible,
            "intrinsics": small_K[[0, 1, 0, 1], [0, 1, 2, 2]].astype(np.float32),
            "source_clearance_pixels": np.float32(min(previous_clearance, current_clearance))}
