"""Fixed causal startup gyro calibration for the predeclared v4 experiment.

The documented initial hover supplies the assumption of rest. Low variation and
mean-prior gates cannot distinguish every small constant true rotation from bias.
No observation or pose emitted before calibration completion is corrected.
"""
import numpy as np

from preintegrate import IMU_TIMESTAMP_TOLERANCE_SECONDS, exp_so3

CALIBRATION_SECONDS = 1.0
MINIMUM_SAMPLES = 180
MINIMUM_SPAN_SECONDS = 0.9
REST_ACCEL_STD = 1.0
REST_GYRO_STD = 0.2
GYRO_BIAS_ABS_PRIOR = 0.02
GYRO_NOISE_STD = 0.07
MEAN_GUARD_SIGMAS = 3.0


def _window(imu, end_t):
    values = np.asarray(imu, dtype=np.float64).reshape(-1, 7)
    if not np.isfinite(end_t) or not np.isfinite(values).all():
        raise ValueError("gyro calibration observations and time must be finite")
    if len(values) and np.any(values[:, 0] > end_t + IMU_TIMESTAMP_TOLERANCE_SECONDS):
        raise ValueError("gyro calibration window contains a future observation")
    return values


def integrate_debiased_rotation(imu, T_cam_imu, start_t, end_t, bias_imu):
    """Integrate gyro minus fixed bias per sample, then rotate into camera axes.

    This is separate from the raw feature preintegration used by the network.
    Timestamp clipping, skipped stale samples and lack of trailing extrapolation
    match preintegrate.py; the caller's IMU array is never modified.
    """
    if not np.isfinite([start_t, end_t]).all() or end_t <= start_t:
        raise ValueError("IMU interval must have finite increasing endpoints")
    values = _window(imu, end_t)
    bias = np.asarray(bias_imu, dtype=np.float64).reshape(3)
    R_ci = np.asarray(T_cam_imu, dtype=np.float64).reshape(4, 4)[:3, :3]
    if not np.isfinite(bias).all() or not np.isfinite(R_ci).all():
        raise ValueError("gyro bias and camera transform must be finite")
    rotation = np.eye(3)
    previous = float(start_t)
    for row in values:
        timestamp = min(float(row[0]), float(end_t))
        dt = timestamp - previous
        if dt <= 0:
            continue
        rotation = rotation @ exp_so3((row[4:7] - bias) * dt)
        previous = timestamp
        if timestamp >= end_t:
            break
    return R_ci @ rotation @ R_ci.T


class InitialGyroBias:
    """Bounded online moments over one fixed initial window; never retries."""
    def __init__(self):
        self.status = "collecting"
        self.reason = "collecting"
        self.anchor_t = None
        self.deadline_t = None
        self.samples = 0
        self.mean = np.zeros(6)
        self.m2 = np.zeros(6)
        self.first_sample_t = None
        self.last_sample_t = None
        self.bias_imu = np.zeros(3)
        self.finalized_at_t = None
        self.activation_after_t = None

    @property
    def accepted(self):
        return self.status == "accepted"

    def _reject(self, reason, t):
        self.status = "rejected"
        self.reason = reason
        self.finalized_at_t = float(t)

    def observe_after_output(self, t, imu):
        """Call after computing this frame's output; bias applies next interval."""
        if self.status != "collecting":
            return
        values = _window(imu, t)  # validate the whole batch before mutating moments
        if self.anchor_t is None:
            positive = np.diff(values[:, 0])
            positive = positive[positive > 0]
            if len(values) < 2 or not len(positive):
                self._reject("bad_initial_anchor", t)
                return
            self.anchor_t = float(values[0, 0] - positive[0])
            self.deadline_t = self.anchor_t + CALIBRATION_SECONDS
        for row in values:
            timestamp = float(row[0])
            if timestamp <= self.anchor_t or timestamp > self.deadline_t + IMU_TIMESTAMP_TOLERANCE_SECONDS:
                continue
            timestamp = min(timestamp, self.deadline_t)
            if self.last_sample_t is not None and timestamp <= self.last_sample_t:
                continue
            self.samples += 1
            difference = row[1:7] - self.mean
            self.mean += difference / self.samples
            self.m2 += difference * (row[1:7] - self.mean)
            if self.first_sample_t is None:
                self.first_sample_t = timestamp
            self.last_sample_t = timestamp
        if t + IMU_TIMESTAMP_TOLERANCE_SECONDS < self.deadline_t:
            return
        span = 0.0 if self.samples < 2 else self.last_sample_t - self.first_sample_t
        if self.samples < MINIMUM_SAMPLES or span < MINIMUM_SPAN_SECONDS:
            self._reject("insufficient_coverage", t)
            return
        std = np.sqrt(np.maximum(self.m2 / self.samples, 0.0))  # population std, matching upstream
        if np.linalg.norm(std[:3]) >= REST_ACCEL_STD or np.linalg.norm(std[3:]) >= REST_GYRO_STD:
            self._reject("variation_gate", t)
            return
        bound = GYRO_BIAS_ABS_PRIOR + MEAN_GUARD_SIGMAS * GYRO_NOISE_STD / np.sqrt(self.samples)
        if np.any(np.abs(self.mean[3:]) > bound):
            self._reject("mean_prior_gate", t)
            return
        self.bias_imu = self.mean[3:].copy()
        self.bias_imu.setflags(write=False)
        self.status = self.reason = "accepted"
        self.finalized_at_t = self.activation_after_t = float(t)

    def diagnostics(self):
        std = np.sqrt(np.maximum(self.m2 / max(self.samples, 1), 0.0))
        span = 0.0 if self.samples < 2 else self.last_sample_t - self.first_sample_t
        bound = GYRO_BIAS_ABS_PRIOR + MEAN_GUARD_SIGMAS * GYRO_NOISE_STD / np.sqrt(self.samples) if self.samples else None
        return {"enabled": True, "status": self.status, "reason": self.reason,
                "anchor_t": self.anchor_t, "deadline_t": self.deadline_t,
                "samples": self.samples, "sample_span_s": span,
                "accel_std_norm": float(np.linalg.norm(std[:3])),
                "gyro_std_norm": float(np.linalg.norm(std[3:])),
                "gyro_mean_imu": self.mean[3:].tolist(), "mean_guard_limit": bound,
                "bias_imu": self.bias_imu.tolist(), "finalized_at_t": self.finalized_at_t,
                "activation_after_t": self.activation_after_t}
