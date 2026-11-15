#pragma once

#include <cmath>
#include <opencv2/calib3d.hpp>

inline cv::Matx33d rotation_matrix(const cv::Vec3d& vector) {
    if constexpr (USE_SMALL_ANGLE_ROTATION) {
        const double x = vector[0], y = vector[1], z = vector[2];
        const double xx = x * x, yy = y * y, zz = z * z;
        const double angle_squared = xx + yy + zz;
        double a, b;
        if (angle_squared <= 0.01) {
            // Taylor expansions of sin(theta)/theta and (1-cos(theta))/theta^2.
            a = 1. + angle_squared * (-1. / 6. + angle_squared
                * (1. / 120. + angle_squared * (-1. / 5040. + angle_squared / 362880.)));
            b = 0.5 + angle_squared * (-1. / 24. + angle_squared
                * (1. / 720. + angle_squared * (-1. / 40320. + angle_squared / 3628800.)));
        } else {
            const double angle = std::sqrt(angle_squared);
            a = std::sin(angle) / angle;
            b = (1. - std::cos(angle)) / angle_squared;
        }
        const double xy = b * x * y, xz = b * x * z, yz = b * y * z;
        const double ax = a * x, ay = a * y, az = a * z;
        return {1. - b * (yy + zz), xy - az, xz + ay,
                xy + az, 1. - b * (xx + zz), yz - ax,
                xz - ay, yz + ax, 1. - b * (xx + yy)};
    }
    cv::Matx33d matrix;
    cv::Rodrigues(vector, matrix);
    return matrix;
}

inline cv::Vec3d rotation_vector(const cv::Matx33d& rotation, bool proper_rotation = true) {
    if constexpr (USE_SMALL_ANGLE_ROTATION) {
        if (proper_rotation) {
            const cv::Vec3d skew{
                0.5 * (rotation(2, 1) - rotation(1, 2)),
                0.5 * (rotation(0, 2) - rotation(2, 0)),
                0.5 * (rotation(1, 0) - rotation(0, 1))};
            const double sine_squared = skew.dot(skew);
            const double cosine = 0.5 * (rotation(0, 0) + rotation(1, 1) + rotation(2, 2) - 1.);
            if (cosine > 0. && sine_squared <= 0.01) {
                // Match Rodrigues' zero cutoff and compute asin(sin(theta))/sin(theta).
                if (sine_squared < 1e-10) return {0., 0., 0.};
                const double factor = 1. + sine_squared * (1. / 6. + sine_squared
                    * (3. / 40. + sine_squared * (5. / 112. + sine_squared
                    * (35. / 1152. + sine_squared * (63. / 2816.)))));
                return skew * factor;
            }
            if (sine_squared >= 1e-10) {
                const double sine = std::sqrt(sine_squared);
                return skew * (std::atan2(sine, cosine) / sine);
            }
        }
        // Keep the reference convention at the half-turn singularity or if
        // the supplied calibration is too far from a rotation matrix.
    }
    cv::Vec3d vector;
    cv::Rodrigues(rotation, vector);
    return vector;
}
