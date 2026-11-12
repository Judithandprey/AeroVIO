// Native compact IMU-only guest with the original weights compiled into SIMD kernels.
#include "viobench_abi.h"
#include "inference.h"
#include "gyro_bias.h"

#include "rotation.h"
#include "visual_direction.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
using cv::Matx33d;
using cv::Vec3d;

struct State {
    Matx33d camera_from_imu = Matx33d::eye();
    std::array<float, 4> camera_features{};
    Matx33d rotation = Matx33d::eye();
    Vec3d position{0., 0., 0.};
    double previous_time = 0.;
    bool started = false;
    bool rigid_extrinsics = true;
    InitialGyroBias calibration;
    visual_direction::Tracker tracker;
    Vec3d imu_position{0., 0., 0.};
} state;

double number(const cv::FileNode& node) {
    if (!node.isInt() && !node.isReal()) throw std::runtime_error("expected numeric calibration value");
    const auto value = static_cast<double>(node);
    if (!std::isfinite(value)) throw std::runtime_error("nonfinite calibration value");
    return value;
}

std::array<double, 4> quaternion(const Matx33d& rotation) {
    std::array<double, 4> result{};
    const double trace = rotation(0, 0) + rotation(1, 1) + rotation(2, 2);
    if (trace > 0.) {
        const double scale = 2. * std::sqrt(1. + trace);
        result = {(rotation(2, 1) - rotation(1, 2)) / scale,
                  (rotation(0, 2) - rotation(2, 0)) / scale,
                  (rotation(1, 0) - rotation(0, 1)) / scale, 0.25 * scale};
    } else {
        int i = 0;
        for (int axis = 1; axis < 3; ++axis) {
            if (rotation(axis, axis) > rotation(i, i)) i = axis;
        }
        const int j = (i + 1) % 3;
        const int k = (i + 2) % 3;
        const double scale = 2. * std::sqrt(std::max(0., 1. + rotation(i, i) - rotation(j, j) - rotation(k, k)));
        result[i] = 0.25 * scale;
        result[j] = (rotation(j, i) + rotation(i, j)) / scale;
        result[k] = (rotation(k, i) + rotation(i, k)) / scale;
        result[3] = (rotation(k, j) - rotation(j, k)) / scale;
    }
    double square_norm = 0.;
    for (const double value : result) square_norm += value * value;
    const double norm = (result[3] >= 0. ? 1. : -1.) * std::sqrt(square_norm);
    for (double& value : result) value /= norm;
    return result;
}

struct Features {
    std::array<float, 14> input;
    Matx33d gyro_rotation;
};

Features features(double end_time, const double* imu, int count) {
    Matx33d rotation = Matx33d::eye(), corrected = Matx33d::eye();
    Vec3d velocity{0., 0., 0.}, position{0., 0., 0.};
    double previous = state.previous_time;
    constexpr double timestamp_tolerance = 1.e-6;
    for (int sample = 0; sample < count; ++sample) {
        const auto* row = imu + VIOBENCH_IMU_ROW_VALUES * sample;
        for (int i = 0; i < VIOBENCH_IMU_ROW_VALUES; ++i) {
            if (!std::isfinite(row[i])) throw std::runtime_error("nonfinite IMU sample");
        }
        if (row[0] > end_time + timestamp_tolerance) throw std::runtime_error("future IMU sample");
        const double time = std::min(row[0], end_time);
        const double dt = time - previous;
        if (dt <= 0.) continue;
        const auto accel = rotation * Vec3d(row[1], row[2], row[3]);
        position += velocity * dt + (0.5 * accel) * dt * dt;
        velocity += accel * dt;
        const Vec3d omega{row[4], row[5], row[6]};
        rotation = rotation * rotation_matrix(omega * dt);
        if (state.calibration.accepted)
            corrected = corrected * rotation_matrix((omega - state.calibration.bias) * dt);
        previous = time;
    }
    const auto& r = state.camera_from_imu;
    const auto angle = rotation_vector(r * rotation * r.t(), state.rigid_extrinsics);
    velocity = r * velocity;
    position = r * position;
    const std::array<float, 14> input{
        float(angle[0]), float(angle[1]), float(angle[2]),
        float(velocity[0]), float(velocity[1]), float(velocity[2]),
        float(position[0]), float(position[1]), float(position[2]),
        state.camera_features[0], state.camera_features[1],
        state.camera_features[2], state.camera_features[3],
        float(end_time - state.previous_time)};
    const auto gyro_rotation = state.calibration.accepted
        ? r * corrected * r.t()
        : rotation_matrix({input[0], input[1], input[2]});
    return {input, gyro_rotation};
}

} // namespace

int32_t abi_version() { return VIOBENCH_ABI_VERSION; }
void* alloc(uint32_t size) { return std::malloc(size ? size : 1); }
int32_t frame_width() { return 320; }

int32_t init(const char* json, uint32_t size) {
    try {
        const cv::FileStorage config(std::string(json, size),
            cv::FileStorage::READ | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);
        const double w = number(config["width"]), h = number(config["height"]);
        const double fx = number(config["fx"]), fy = number(config["fy"]);
        if (w != 320 || h != 240 || fx <= 0. || fy <= 0.) throw std::runtime_error("invalid camera calibration");
        state.tracker.configure(fx, fy, number(config["cx"]), number(config["cy"]));
        state.camera_features = {float(fx / w), float(fy / h),
            float((number(config["cx"]) + 0.5) / w), float((number(config["cy"]) + 0.5) / h)};
        const auto extrinsics = config["T_cam_imu"];
        if (!extrinsics.isSeq() || extrinsics.size() != 4) throw std::runtime_error("invalid camera/IMU transform");
        for (int i = 0; i < 4; ++i) {
            const auto row = extrinsics[i];
            if (!row.isSeq() || row.size() != 4) throw std::runtime_error("invalid camera/IMU transform row");
            for (int j = 0; j < 4; ++j) {
                const double value = number(row[j]);
                if (i < 3 && j < 3) state.camera_from_imu(i, j) = value;
            }
        }
        const auto& r = state.camera_from_imu;
        const auto orthogonality = r * r.t() - Matx33d::eye();
        state.rigid_extrinsics = cv::norm(orthogonality, cv::NORM_INF) < 1e-10
            && std::abs(cv::determinant(r) - 1.) < 1e-10;
        return VIOBENCH_STATUS_OK;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "init: %s\n", error.what());
        return VIOBENCH_STATUS_ERROR;
    }
}

int32_t step(double time, const uint8_t* frame, const double* imu, int32_t count, double* out) {
    try {
        if (!std::isfinite(time) || count < 0 || (state.started && time <= state.previous_time))
            throw std::runtime_error("invalid frame time or sample count");
        if (!state.started) {
            std::fill_n(out, VIOBENCH_STATE_VALUES, 0.);
            out[6] = 1.;
            state.tracker.step(frame, state.rotation, state.imu_position);
        } else {
            const auto preintegrated = features(time, imu, count);
            const auto motion = infer(preintegrated.input);
            Matx33d delta = preintegrated.gyro_rotation;
            if constexpr (USE_LEARNED_ROTATION_RESIDUAL)
                delta = delta * rotation_matrix({motion[3], motion[4], motion[5]});
            const Matx33d rotation = state.rotation * delta;
            const Vec3d increment = state.rotation * Vec3d(motion[0], motion[1], motion[2]);
            const Vec3d imu_position = state.imu_position + increment;
            const Vec3d correction = state.tracker.step(frame, rotation, imu_position);
            const Vec3d position = state.position + increment + correction;
            const double dt = time - state.previous_time;
            const Vec3d linear = (rotation.t() * (position - state.position)) / dt;
            const Vec3d angular = rotation_vector(state.rotation.t() * rotation, state.rigid_extrinsics) / dt;
            const auto q = quaternion(rotation);
            std::copy_n(position.val, 3, out);
            std::copy(q.begin(), q.end(), out + 3);
            std::copy_n(linear.val, 3, out + 7);
            std::copy_n(angular.val, 3, out + 10);
            for (int i = 0; i < VIOBENCH_STATE_VALUES; ++i) {
                if (!std::isfinite(out[i])) throw std::runtime_error("nonfinite output");
            }
            state.rotation = rotation;
            state.position = position;
            state.imu_position = imu_position;
        }
        state.calibration.observe(time, imu, count);
        state.previous_time = time;
        state.started = true;
        return VIOBENCH_STATUS_OK;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "step: %s\n", error.what());
        return VIOBENCH_STATUS_ERROR;
    }
}
