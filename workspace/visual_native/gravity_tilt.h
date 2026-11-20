// Gravity tilt correction for the integrated gyro rotation (stage 2: no start hover).
//
// A quadrotor's accelerometer mostly sees thrust, so a single frame says little about
// gravity. Over seconds, though, the mean acceleration of a bounded flight is near zero,
// so the time average of the specific force expressed in the camera-0 frame points along
// gravity. We anchor that direction over the first `anchor_seconds`, then keep a slower
// running average of the current specific force (rotated by our estimate) and nudge the
// estimate so the two agree. The correction axis is perpendicular to gravity, so it only
// fixes roll/pitch drift; yaw is untouched.
#pragma once
#include "rotation.h"
#include <opencv2/core.hpp>
#include <cmath>

#ifndef GRAVITY_TILT_GAIN
#define GRAVITY_TILT_GAIN 0.
#endif
#ifndef GRAVITY_TILT_TAU
#define GRAVITY_TILT_TAU 3.
#endif
#ifndef GRAVITY_TILT_ANCHOR
#define GRAVITY_TILT_ANCHOR 5.
#endif

struct GravityTilt {
    static constexpr double gain = GRAVITY_TILT_GAIN;          // correction rate (1/s)
    static constexpr double tau = GRAVITY_TILT_TAU;            // running-average time constant (s)
    static constexpr double anchor_seconds = GRAVITY_TILT_ANCHOR;
    cv::Vec3d anchor_sum{0., 0., 0.};
    double anchor_time = 0.;
    double elapsed = 0.;
    cv::Vec3d recent{0., 0., 0.};
    bool recent_ready = false;

    // rotation: camera_k in camera_0. imu rows: [t, ax, ay, az, gx, gy, gz].
    cv::Matx33d correct(const cv::Matx33d& rotation, const double* imu, int count,
                        const cv::Matx33d& camera_from_imu, double dt) {
        if (gain <= 0. || count <= 0 || dt <= 0.) return rotation;
        cv::Vec3d mean{0., 0., 0.};
        for (int i = 0; i < count; ++i) mean += cv::Vec3d(imu[7 * i + 1], imu[7 * i + 2], imu[7 * i + 3]);
        mean *= 1. / count;
        const cv::Vec3d world = rotation * (camera_from_imu * mean);
        elapsed += dt;
        if (elapsed <= anchor_seconds) {
            anchor_sum += world * dt;
            anchor_time += dt;
        }
        const double alpha = dt / (tau + dt);
        recent = recent_ready ? recent + alpha * (world - recent) : world;
        recent_ready = true;
        if (elapsed <= anchor_seconds || anchor_time <= 0.) return rotation;
        const double a = cv::norm(anchor_sum), r = cv::norm(recent);
        if (a < 1e-9 || r < 1e-9) return rotation;
        const cv::Vec3d target = anchor_sum * (1. / a), current = recent * (1. / r);
        // Rotate the estimate (left-multiply, camera-0 frame) so `current` moves toward `target`.
        const cv::Vec3d axis = current.cross(target);
        return rotation_matrix(axis * (gain * dt)) * rotation;
    }
};
