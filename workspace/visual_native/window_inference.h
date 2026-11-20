// Windowed IMU motion network (scripts/train_window_imu.py). Mirrors the training features exactly:
// per frame, the velocity change and rotation vector in the previous camera's axes are rotated into
// camera-0 axes with the running rotation, kept for the last window_frames frames, and rotated back
// into the previous camera's axes at inference. Output: camera translation of this frame, previous
// camera axes (same convention as the compact network).
#pragma once
#include WINDOW_WEIGHTS_HEADER
#include <opencv2/core.hpp>
#include <algorithm>
#include <array>

struct WindowImu {
    std::array<cv::Vec3d, window_frames> dv{}, rv{};  // camera-0 axes, ring buffer
    int count = 0, head = 0;

    // previous_rotation: camera k-1 in camera 0. dv_cam/rv_cam: this frame's features in camera k-1 axes.
    cv::Vec3d step(const cv::Matx33d& previous_rotation, const cv::Vec3d& dv_cam, const cv::Vec3d& rv_cam, double dt) {
        dv[head] = previous_rotation * dv_cam;
        rv[head] = previous_rotation * rv_cam;
        head = (head + 1) % window_frames;
        count = std::min(count + 1, window_frames);
        std::array<float, window_inputs> x{};
        const cv::Matx33d to_camera = previous_rotation.t();
        // oldest -> newest; missing history stays zero (as in training).
        for (int slot = 0; slot < window_frames; ++slot) {
            const int age = window_frames - 1 - slot;  // 0 = newest
            if (age >= count) continue;
            const int index = (head - 1 - age + 2 * window_frames) % window_frames;
            const cv::Vec3d a = to_camera * dv[index], b = to_camera * rv[index];
            for (int i = 0; i < 3; ++i) { x[slot * 6 + i] = float(a[i]); x[slot * 6 + 3 + i] = float(b[i]); }
        }
        x[window_inputs - 1] = float(dt);
        for (int i = 0; i < window_inputs; ++i) x[i] = (x[i] - window_x_mean[i]) / window_x_std[i];
        std::array<float, window_hidden> h1{}, h2{};
        for (int o = 0; o < window_hidden; ++o) {
            float s = window_b1[o];
            const float* w = window_w1.data() + o * window_inputs;
            for (int i = 0; i < window_inputs; ++i) s += w[i] * x[i];
            h1[o] = std::max(0.f, s);
        }
        for (int o = 0; o < window_hidden; ++o) {
            float s = window_b2[o];
            const float* w = window_w2.data() + o * window_hidden;
            for (int i = 0; i < window_hidden; ++i) s += w[i] * h1[i];
            h2[o] = std::max(0.f, s);
        }
        cv::Vec3d out;
        for (int o = 0; o < 3; ++o) {
            float s = window_b3[o];
            const float* w = window_w3.data() + o * window_hidden;
            for (int i = 0; i < window_hidden; ++i) s += w[i] * h2[i];
            out[o] = double(s * window_y_std[o] + window_y_mean[o]);
        }
        return out;
    }
};
