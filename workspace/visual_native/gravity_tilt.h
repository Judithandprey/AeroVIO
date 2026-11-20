// Optional Mahony-style gravity tilt correction (LEO_GRAVITY_TILT, default off).
//
// The startup reference up0 is the mean accelerometer specific force (about +9.81 m/s^2
// pointing up at rest) over the first LEO_GRAVITY_TILT_INIT_SECONDS, expressed in
// frame-0 camera axes. During integration each accepted sample adds
// sign * K * (measured x predicted) to the gyro rate, where both vectors are unit
// specific-force directions in the IMU frame. The error is perpendicular to the up
// direction, so it corrects roll/pitch only; yaw is untouched.
#pragma once
#include <opencv2/core.hpp>
#include <cmath>

#ifndef LEO_GRAVITY_TILT
#define LEO_GRAVITY_TILT 0
#endif
#ifndef LEO_GRAVITY_TILT_GAIN
#define LEO_GRAVITY_TILT_GAIN 0.01  // K in 1/s; the per-sample correction is K * e * dt
#endif
#ifndef LEO_GRAVITY_TILT_GATE
#define LEO_GRAVITY_TILT_GATE 0.5  // accept only | |a| - 9.81 | below this, m/s^2
#endif
#ifndef LEO_GRAVITY_TILT_SIGN
#define LEO_GRAVITY_TILT_SIGN 1  // diagnostic only: -1 flips the correction to check the convention
#endif
#ifndef LEO_GRAVITY_TILT_INIT_SECONDS
#define LEO_GRAVITY_TILT_INIT_SECONDS 0.5
#endif

struct GravityTilt {
    static constexpr double gravity = 9.81;
    static constexpr double gain = LEO_GRAVITY_TILT_SIGN * LEO_GRAVITY_TILT_GAIN;
    static constexpr double gate = LEO_GRAVITY_TILT_GATE;
    static constexpr double window = LEO_GRAVITY_TILT_INIT_SECONDS;
    cv::Vec3d sum{0., 0., 0.};
    cv::Vec3d up0{0., 0., 0.};  // unit specific force at startup, frame-0 camera axes
    double first = 0.;
    int samples = 0;
    bool ready = false;

    // camera0_from_imu: current estimate of the IMU-to-frame-0-camera rotation.
    void observe(const cv::Matx33d& camera0_from_imu, const double* imu, int count) {
        if (ready) return;
        for (int i = 0; i < count; ++i) {
            const auto* row = imu + 7 * i;
            if (samples == 0) first = row[0];
            if (row[0] - first > window) {
                const double n = cv::norm(sum);
                if (n > 0.) {
                    up0 = sum / n;
                    ready = true;
                }
                return;
            }
            sum += camera0_from_imu * cv::Vec3d(row[1], row[2], row[3]);
            ++samples;
        }
    }

    // Corrected IMU-frame rate. up_imu: predicted up direction in the current IMU frame.
    static cv::Vec3d correct(const cv::Vec3d& omega, const cv::Vec3d& accel, const cv::Vec3d& up_imu) {
        const double n = cv::norm(accel);
        if (!(std::abs(n - gravity) < gate)) return omega;
        return omega + gain * (accel / n).cross(up_imu);
    }
};
