// Linear position/velocity Kalman filter for stage 2 (unaligned ATE needs metric scale).
//
// Rotation comes from the existing gyro chain (online bias + gravity tilt), so with rotation
// treated as known the filter is linear. Accelerometer integration gives metric scale; the
// visual tracker gives the direction of travel between capture frames; the learned network
// is a weak velocity pseudo-measurement that keeps the filter from wandering.
//
// State (camera-0 axes): p(3) v(3) b(3) c1(3) c0(3)
//   b  : world-frame acceleration correction (gravity direction error + accelerometer bias)
//   c1 : position at the latest capture frame, c0: at the capture before (stochastic clones)
#pragma once
#include <opencv2/core.hpp>
#include <cmath>

#ifndef VIO_KF_SIGMA_V
#define VIO_KF_SIGMA_V 0.02      // velocity process noise per frame (m/s)
#endif
#ifndef VIO_KF_SIGMA_B
#define VIO_KF_SIGMA_B 0.002     // acceleration-correction random walk per frame (m/s^2)
#endif
#ifndef VIO_KF_SIGMA_HEADING
#define VIO_KF_SIGMA_HEADING 0.05 // visual direction noise (rad)
#endif
#ifndef VIO_KF_GATE
#define VIO_KF_GATE 9.0
#endif
#ifndef VIO_KF_SIGMA_NET
#define VIO_KF_SIGMA_NET 0.3     // learned velocity pseudo-measurement noise (m/s)
#endif

struct VioKf {
    using V15 = cv::Vec<double, 15>;
    using M15 = cv::Matx<double, 15, 15>;
    V15 x = V15::all(0.);
    M15 P = M15::zeros();
    int captures = 0;
    int heading_updates = 0, heading_rejects = 0;

    VioKf() {
        for (int i = 0; i < 3; ++i) {
            P(i, i) = 1e-6;          // position starts at the origin
            P(3 + i, 3 + i) = 0.25;  // unknown initial velocity (no start hover)
            P(6 + i, 6 + i) = 0.25;  // gravity direction / bias correction
            P(9 + i, 9 + i) = 1e-6;
            P(12 + i, 12 + i) = 1e-6;
        }
    }

    cv::Vec3d get(int i) const { return {x[i], x[i + 1], x[i + 2]}; }
    void set(int i, const cv::Vec3d& v) { x[i] = v[0]; x[i + 1] = v[1]; x[i + 2] = v[2]; }

    // R: camera k-1 in camera 0. dv_cam / dp_cam: specific-force velocity and position change over the
    // frame in camera k-1 axes (IMU preintegration). g: gravity in camera-0 axes.
    void predict(const cv::Matx33d& R, const cv::Vec3d& dv_cam, const cv::Vec3d& dp_cam, double dt,
                 const cv::Vec3d& g) {
        const cv::Vec3d p = get(0), v = get(3), b = get(6);
        const cv::Vec3d a_dt = R * dv_cam;  // specific-force velocity change, world axes
        set(0, p + v * dt + R * dp_cam + (g + b) * (0.5 * dt * dt));
        set(3, v + a_dt + (g + b) * dt);
        M15 F = M15::eye();
        for (int i = 0; i < 3; ++i) {
            F(i, 3 + i) = dt;
            F(i, 6 + i) = 0.5 * dt * dt;
            F(3 + i, 6 + i) = dt;
        }
        P = F * P * F.t();
        const double qv = VIO_KF_SIGMA_V * VIO_KF_SIGMA_V, qb = VIO_KF_SIGMA_B * VIO_KF_SIGMA_B;
        for (int i = 0; i < 3; ++i) {
            P(3 + i, 3 + i) += qv;
            P(i, i) += qv * dt * dt * 0.25;
            P(6 + i, 6 + i) += qb;
        }
    }

    // Called on capture frames: c0 <- c1, c1 <- p.
    void capture() {
        M15 T = M15::zeros();
        for (int i = 0; i < 9; ++i) T(i, i) = 1.;
        for (int i = 0; i < 3; ++i) {
            T(9 + i, i) = 1.;       // c1 <- p
            T(12 + i, 9 + i) = 1.;  // c0 <- c1
        }
        x = T * x;
        P = T * P * T.t();
        ++captures;
    }

    template <int N>
    bool update(const cv::Matx<double, N, 15>& H, const cv::Vec<double, N>& residual,
                const cv::Matx<double, N, N>& R, double gate) {
        const cv::Matx<double, N, N> S = H * P * H.t() + R;
        bool ok = false;
        const cv::Matx<double, N, N> Si = S.inv(cv::DECOMP_CHOLESKY, &ok);
        if (!ok) return false;
        const double chi2 = (residual.t() * Si * residual)(0);
        if (!std::isfinite(chi2)) return false;
        // Soft gating: an inconsistent measurement is down-weighted, not dropped, so the filter
        // can recover after its velocity drifted (hard gating would lock out the camera for good).
        cv::Matx<double, N, N> R_used = R;
        cv::Matx<double, N, N> Si_used = Si;
        if (chi2 > gate) {
            R_used = R * (chi2 / gate);
            const cv::Matx<double, N, N> S2 = H * P * H.t() + R_used;
            Si_used = S2.inv(cv::DECOMP_CHOLESKY, &ok);
            if (!ok) return false;
        }
        const cv::Matx<double, 15, N> K = P * H.t() * Si_used;
        x += K * residual;
        const M15 I_KH = M15::eye() - K * H;
        P = I_KH * P * I_KH.t() + K * R_used * K.t();  // Joseph form
        return chi2 <= gate;
    }

    // Visual direction d (unit, world) of c1 - c0: the perpendicular components must vanish.
    void update_heading(const cv::Vec3d& d, double weight) {
        if (captures < 2 || weight <= 0.) return;
        const cv::Vec3d axis = std::abs(d[2]) < .9 ? cv::Vec3d(0, 0, 1) : cv::Vec3d(1, 0, 0);
        cv::Vec3d u = d.cross(axis); u *= 1. / cv::norm(u);
        const cv::Vec3d w = d.cross(u);
        const cv::Vec3d delta = get(9) - get(12);
        // A displacement opposite to the visual direction is a sign/outlier problem, not a small error.
        cv::Matx<double, 2, 15> H = cv::Matx<double, 2, 15>::zeros();
        for (int i = 0; i < 3; ++i) {
            H(0, 9 + i) = u[i]; H(0, 12 + i) = -u[i];
            H(1, 9 + i) = w[i]; H(1, 12 + i) = -w[i];
        }
        const cv::Vec2d residual(-u.dot(delta), -w.dot(delta));
        const double sigma = VIO_KF_SIGMA_HEADING / weight * std::max(cv::norm(delta), 0.05);
        const cv::Matx22d R(sigma * sigma, 0., 0., sigma * sigma);
        if (update<2>(H, residual, R, VIO_KF_GATE)) ++heading_updates; else ++heading_rejects;
    }

    // Weak velocity measurement (world axes) from the learned network.
    void update_velocity(const cv::Vec3d& v_meas, double sigma) {
        cv::Matx<double, 3, 15> H = cv::Matx<double, 3, 15>::zeros();
        for (int i = 0; i < 3; ++i) H(i, 3 + i) = 1.;
        const cv::Vec3d residual = v_meas - get(3);
        const cv::Matx33d R = cv::Matx33d::eye() * (sigma * sigma);
        update<3>(H, residual, R, 1e9);
    }
};
