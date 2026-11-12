// Port of AeroVIO's fixed startup gyro-bias calibration.
#pragma once
#include <opencv2/core.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

struct InitialGyroBias {
    static constexpr double tolerance = 1e-6;
    cv::Vec3d bias{0., 0., 0.};
    bool accepted = false;
    bool finished = false;
    bool anchored = false;
    double anchor = 0.;
    double first = 0.;
    double last = -std::numeric_limits<double>::infinity();
    int samples = 0;
    std::array<double, 6> mean{}, m2{};

    void observe(double time, const double* imu, int count) {
        if (finished) return;
        if (!anchored) {
            for (int i = 1; i < count; ++i) {
                const double spacing = imu[7 * i] - imu[7 * (i - 1)];
                if (spacing > 0.) {
                    anchor = imu[0] - spacing;
                    anchored = true;
                    break;
                }
            }
            if (!anchored) {
                finished = true;
                return;
            }
        }
        const double deadline = anchor + 1.;
        for (int i = 0; i < count; ++i) {
            const auto* row = imu + 7 * i;
            if (row[0] <= anchor || row[0] > deadline + tolerance) continue;
            const double t = std::min(row[0], deadline);
            if (t <= last) continue;
            ++samples;
            if (samples == 1) first = t;
            last = t;
            for (int axis = 0; axis < 6; ++axis) {
                const double difference = row[axis + 1] - mean[axis];
                mean[axis] += difference / samples;
                m2[axis] += difference * (row[axis + 1] - mean[axis]);
            }
        }
        if (time + tolerance < deadline) return;
        finished = true;
        if (samples < 180 || last - first < .9) return;
        double accel_variance = 0., gyro_variance = 0.;
        for (int axis = 0; axis < 3; ++axis) {
            accel_variance += std::max(0., m2[axis] / samples);
            gyro_variance += std::max(0., m2[axis + 3] / samples);
        }
        if (std::sqrt(accel_variance) >= 1. || std::sqrt(gyro_variance) >= .2) return;
        const double bound = .02 + 3. * .07 / std::sqrt(samples);
        for (int axis = 3; axis < 6; ++axis) {
            if (std::abs(mean[axis]) > bound) return;
        }
        bias = {mean[3], mean[4], mean[5]};
        accepted = true;
    }
};
