// Native compact IMU-only guest with the original weights compiled into SIMD kernels.
#include "viobench_abi.h"
#include "inference.h"
#include "gyro_bias.h"
#include "gravity_tilt.h"
#ifndef ESEKF
#define ESEKF 0
#endif
#if ESEKF
#include "esekf.h"
#endif
#ifndef VIO_KF
#define VIO_KF 0
#endif
#if VIO_KF
#include "vio_kf.h"
#endif
#ifndef WINDOW_IMU
#define WINDOW_IMU 0
#endif
#if WINDOW_IMU
#include "window_inference.h"
#endif

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

// LEO_IMU_ONLY=1 never calls the visual tracker: pose comes only from the compact
// IMU network and (calibrated) gyro rotation. Default 0 keeps the teacher build.
#ifndef LEO_IMU_ONLY
#define LEO_IMU_ONLY 0
#endif
// POSITION_SCALE multiplies the learned translation increment. Stage 2 scores
// unaligned ATE, and the Sim(3)-trained network under-estimates path length.
#ifndef ONLINE_GYRO_BIAS_GAIN
#define ONLINE_GYRO_BIAS_GAIN 0.3
#endif
#ifndef ONLINE_GYRO_BIAS_SIGN
#define ONLINE_GYRO_BIAS_SIGN 1.0
#endif
static_assert(!(LEO_IMU_ONLY && ONLINE_GYRO_BIAS), "online gyro bias needs the visual tracker");

#ifndef POSITION_SCALE
#define POSITION_SCALE 1.0
#endif
static_assert(!(LEO_IMU_ONLY && VISUAL_ROTATION_CORRECTION), "LEO_IMU_ONLY has no visual rotation");

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
    GravityTilt tilt;
    Vec3d online_bias{0., 0., 0.};  // IMU frame, rad/s, added to the startup calibration
    int residual_seen = 0;
#if ESEKF
    Esekf ekf;
    int ekf_heading_seen = 0, ekf_residual_seen = 0;
#endif
    int tracker_steps = 0;  // tracker.step calls so far; captures happen when % VISUAL_INTERVAL == 0
#if VIO_KF
    VioKf kf;
    Vec3d specific_force_sum{0., 0., 0.};  // world-axes specific-force velocity change, for gravity
    double specific_force_time = 0.;
    int heading_seen = 0;
#endif
#if WINDOW_IMU
    WindowImu window;
#endif
    visual_direction::Tracker tracker;
    Vec3d imu_position{0., 0., 0.};
    Matx33d visual_alignment = Matx33d::eye();
    Matx33d output_rotation = Matx33d::eye();
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
    const Vec3d bias = (state.calibration.accepted ? state.calibration.bias : Vec3d(0., 0., 0.)) + state.online_bias;
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
        if (state.calibration.accepted || ONLINE_GYRO_BIAS)
            corrected = corrected * rotation_matrix((omega - bias) * dt);
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
    const auto gyro_rotation = (state.calibration.accepted || ONLINE_GYRO_BIAS)
        ? r * corrected * r.t()
        : rotation_matrix({input[0], input[1], input[2]});
    return {input, gyro_rotation};
}

} // namespace

int32_t abi_version() { return VIOBENCH_ABI_VERSION; }
void* alloc(uint32_t size) { return std::malloc(size ? size : 1); }
int32_t frame_width() { return 320; }

// Optional read-only host diagnostics; it does not participate in estimation.
extern "C" VIOBENCH_EXPORT(fusion_diagnostics) void fusion_diagnostics(double* out) {
    const auto values = state.tracker.diagnostics();
    std::copy(values.begin(), values.end(), out);
}

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

#if ESEKF
// Full error-state EKF path: the filter owns rotation and position; heuristics (tilt filter, online bias,
// visual alignment, dead-reckoned increments) are not used.
static void esekf_step(double time, const uint8_t* frame, const double* imu, int32_t count, double* out) {
    auto& f = state.ekf;
    if (!state.started) {
        std::fill_n(out, VIOBENCH_STATE_VALUES, 0.);
        out[6] = 1.;
        f.init(imu, count, state.camera_from_imu);
        f.capture();
#if !LEO_IMU_ONLY
        state.tracker.step(frame, f.R, f.p);
        ++state.tracker_steps;
#endif
        return;
    }
    const double dt = time - state.previous_time;
    const Matx33d R_prev = f.R;
    const Vec3d p_prev = f.p;
    const auto pre = features(time, imu, count);
    f.propagate(imu, count, state.previous_time, time, state.camera_from_imu);
#if WINDOW_IMU
    const auto& in = pre.input;
    const Vec3d t_net = state.window.step(R_prev, Vec3d(in[3], in[4], in[5]), Vec3d(in[0], in[1], in[2]), dt);
#else
    const auto motion = infer(pre.input);
    const Vec3d t_net(motion[0], motion[1], motion[2]);
#endif
    f.update_velocity(R_prev * t_net * (POSITION_SCALE / dt), ESEKF_SIGMA_NET);
#if !LEO_IMU_ONLY
    if (state.tracker_steps % VISUAL_INTERVAL == 0) f.capture();
    state.tracker.step(frame, f.R, f.p);
    ++state.tracker_steps;
    if (state.tracker.heading_count() != state.ekf_heading_seen) {
        state.ekf_heading_seen = state.tracker.heading_count();
        f.update_heading(state.tracker.last_direction(), state.tracker.last_heading_weight());
    }
    if (state.tracker.residual_count() != state.ekf_residual_seen) {
        state.ekf_residual_seen = state.tracker.residual_count();
        f.update_rotation(state.tracker.last_residual());
    }
#endif
#ifdef ESEKF_DEBUG
    if (state.tracker_steps % 200 == 0)
        std::fprintf(stderr, "ekf t=%.1f head %d/%d rot %d/%d |v|=%.2f bg=(%.4f %.4f %.4f) |ba|=%.3f g=(%.2f %.2f %.2f)\n",
            time, f.heading_ok, f.heading_soft, f.rot_ok, f.rot_soft, cv::norm(f.v), f.bg[0], f.bg[1], f.bg[2],
            cv::norm(f.ba), f.g[0], f.g[1], f.g[2]);
#endif
    const Vec3d linear = (f.R.t() * (f.p - p_prev)) / dt;
    const Vec3d angular = rotation_vector(R_prev.t() * f.R, state.rigid_extrinsics) / dt;
    const auto q = quaternion(f.R);
    std::copy_n(f.p.val, 3, out);
    std::copy(q.begin(), q.end(), out + 3);
    std::copy_n(linear.val, 3, out + 7);
    std::copy_n(angular.val, 3, out + 10);
    for (int i = 0; i < VIOBENCH_STATE_VALUES; ++i)
        if (!std::isfinite(out[i])) throw std::runtime_error("nonfinite output");
}
#endif

int32_t step(double time, const uint8_t* frame, const double* imu, int32_t count, double* out) {
    try {
#if ESEKF
        if (!std::isfinite(time) || count < 0 || (state.started && time <= state.previous_time))
            throw std::runtime_error("invalid frame time or sample count");
        esekf_step(time, frame, imu, count, out);
        state.calibration.observe(time, imu, count);
        state.previous_time = time;
        state.started = true;
        return VIOBENCH_STATUS_OK;
#endif
        if (!std::isfinite(time) || count < 0 || (state.started && time <= state.previous_time))
            throw std::runtime_error("invalid frame time or sample count");
        if (!state.started) {
            std::fill_n(out, VIOBENCH_STATE_VALUES, 0.);
            out[6] = 1.;
#if VIO_KF
            state.kf.capture();
#endif
#if !LEO_IMU_ONLY
            state.tracker.step(frame, state.rotation, state.imu_position);
            ++state.tracker_steps;
#endif
        } else {
            const auto preintegrated = features(time, imu, count);
            const auto motion = infer(preintegrated.input);
            Matx33d delta = preintegrated.gyro_rotation;
            if constexpr (USE_LEARNED_ROTATION_RESIDUAL)
                delta = delta * rotation_matrix({motion[3], motion[4], motion[5]});
            const Matx33d rotation = state.tilt.correct(state.rotation * delta, imu, count,
                state.camera_from_imu, time - state.previous_time);
#if WINDOW_IMU
            // Windowed network: sees the last frames of IMU, so it can infer the current speed.
            const auto& in = preintegrated.input;
            const Vec3d translation = state.window.step(state.rotation, Vec3d(in[3], in[4], in[5]),
                Vec3d(in[0], in[1], in[2]), time - state.previous_time);
            const Vec3d increment = state.rotation * (POSITION_SCALE * translation);
#else
            const Vec3d increment = state.rotation * (POSITION_SCALE * Vec3d(motion[0], motion[1], motion[2]));
#endif
#if VIO_KF
            {
                const double frame_dt = time - state.previous_time;
                const auto& in = preintegrated.input;
                const Vec3d dv_cam(in[3], in[4], in[5]), dp_cam(in[6], in[7], in[8]);
                if (state.specific_force_time < 5.) {
                    state.specific_force_sum += state.rotation * dv_cam;
                    state.specific_force_time += frame_dt;
                }
                const double sf = cv::norm(state.specific_force_sum);
                const Vec3d gravity = sf > 1e-9 ? state.specific_force_sum * (-9.81 / sf) : Vec3d(0., 9.81, 0.);
                state.kf.predict(state.rotation, dv_cam, dp_cam, frame_dt, gravity);
                state.kf.update_velocity(increment / frame_dt, VIO_KF_SIGMA_NET);
#if !LEO_IMU_ONLY
                if (state.tracker_steps % VISUAL_INTERVAL == 0) state.kf.capture();
#endif
            }
            const Vec3d imu_position = state.kf.get(0);
#else
            const Vec3d imu_position = state.imu_position + increment;
#endif
#if LEO_IMU_ONLY
            const Vec3d correction{0., 0., 0.};
#else
            const Vec3d correction = state.tracker.step(frame, rotation, imu_position);
            ++state.tracker_steps;
#if VIO_KF
            if (state.tracker.heading_count() != state.heading_seen) {
                state.heading_seen = state.tracker.heading_count();
                state.kf.update_heading(state.tracker.last_heading(), state.tracker.last_heading_weight());
            }
#endif
            if constexpr (ONLINE_GYRO_BIAS) {
                if (state.tracker.residual_count() != state.residual_seen) {
                    state.residual_seen = state.tracker.residual_count();
                    // Residual rotation left over a visual interval ~ remaining bias x interval.
                    const double interval = VISUAL_INTERVAL * (time - state.previous_time);
                    const Vec3d measured = state.camera_from_imu.t() * state.tracker.last_residual()
                        * (ONLINE_GYRO_BIAS_SIGN / interval);
                    state.online_bias += ONLINE_GYRO_BIAS_GAIN * measured;
                    for (int axis = 0; axis < 3; ++axis)
                        state.online_bias[axis] = std::clamp(state.online_bias[axis], -.05, .05);
                }
            }
#endif
            Vec3d position = state.position + increment + correction;
            Matx33d output_rotation = rotation;
            Matx33d previous_output_rotation = state.rotation;
            if constexpr (VISUAL_ROTATION_CORRECTION) {
                // Tracking and the learned IMU prior stay in the original gyro
                // frame. Apply delayed, cumulative visual rotation only to the
                // emitted pose and its translation increments. This prevents
                // stale captured rays from mixing corrected/uncorrected axes.
                position = state.position + state.visual_alignment * (increment + correction);
                state.visual_alignment = state.visual_alignment * rotation_matrix(state.tracker.rotation_step());
                output_rotation = state.visual_alignment * rotation;
                previous_output_rotation = state.output_rotation;
            }
#if VIO_KF && defined(VIO_KF_DEBUG)
            if (state.tracker_steps % 200 == 0) {
                const Vec3d v = state.kf.get(3), b = state.kf.get(6);
                std::fprintf(stderr, "kf t=%.1f upd=%d rej=%d |v|=%.2f v=(%.2f %.2f %.2f) |b|=%.3f sv=%.3f\n", time,
                    state.kf.heading_updates, state.kf.heading_rejects, cv::norm(v), v[0], v[1], v[2], cv::norm(b),
                    std::sqrt(state.kf.P(3, 3)));
            }
#endif
#if VIO_KF
            position = state.kf.get(0);  // the filter replaces dead-reckoned increments and visual nudges
#endif
            const double dt = time - state.previous_time;
            const Vec3d linear = (output_rotation.t() * (position - state.position)) / dt;
            const Vec3d angular = rotation_vector(previous_output_rotation.t() * output_rotation, state.rigid_extrinsics) / dt;
            const auto q = quaternion(output_rotation);
            std::copy_n(position.val, 3, out);
            std::copy(q.begin(), q.end(), out + 3);
            std::copy_n(linear.val, 3, out + 7);
            std::copy_n(angular.val, 3, out + 10);
            for (int i = 0; i < VIOBENCH_STATE_VALUES; ++i) {
                if (!std::isfinite(out[i])) throw std::runtime_error("nonfinite output");
            }
            state.rotation = rotation;
            state.output_rotation = output_rotation;
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
