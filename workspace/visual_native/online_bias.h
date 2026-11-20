// Optional online gyro bias estimation from V8's visual rotation refinement (LEO_ONLINE_BIAS, default off).
//
// Each accepted refinement measures the rotation the gyro got wrong over one VISUAL_INTERVAL, in anchor
// camera axes. A remaining bias b (IMU axes) shows up as residual ~ -R b dt (R = camera_from_imu), so
// the rate error is -R^T residual / dt. The bias moves a fraction GAIN of that per accepted fit (a slow
// integral loop); fits implying a rate error above REJECT rad/s are skipped as outliers, and each axis is
// clamped to LIMIT rad/s. Updates start once the startup calibration has finished, so they add to its
// bias when it is accepted and replace it when it is rejected. This also observes yaw.
#pragma once
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>

#ifndef LEO_ONLINE_BIAS
#define LEO_ONLINE_BIAS 0
#endif
#ifndef LEO_ONLINE_BIAS_GAIN
#define LEO_ONLINE_BIAS_GAIN 0.05  // fraction of the measured rate error applied per accepted fit
#endif
#ifndef LEO_ONLINE_BIAS_REJECT
#define LEO_ONLINE_BIAS_REJECT 0.03  // rad/s; larger implied rate errors are outliers
#endif
#ifndef LEO_ONLINE_BIAS_SIGN
#define LEO_ONLINE_BIAS_SIGN 1  // diagnostic only: -1 flips the update to check the convention
#endif

struct OnlineGyroBias {
    static constexpr double gain = LEO_ONLINE_BIAS_SIGN * LEO_ONLINE_BIAS_GAIN;
    static constexpr double reject = LEO_ONLINE_BIAS_REJECT;
    static constexpr double limit = 0.05;  // rad/s per axis; the simulator's initial bias is at most 0.02
    cv::Vec3d bias{0., 0., 0.};  // IMU axes, subtracted in addition to the startup calibration
    int updates = 0, rejected = 0;

    void update(const cv::Vec3d& residual_camera, const cv::Matx33d& camera_from_imu, double dt) {
        if (!(dt > 0.)) return;
        const cv::Vec3d rate_error = camera_from_imu.t() * residual_camera * (-1. / dt);
        if (!(cv::norm(rate_error) < reject)) { ++rejected; return; }
        bias += gain * rate_error;
        for (int i = 0; i < 3; ++i) bias[i] = std::clamp(bias[i], -limit, limit);
        ++updates;
    }
};
