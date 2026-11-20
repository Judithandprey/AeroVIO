// Error-state EKF for stage 2: orientation, position, velocity, gyro/accel bias and gravity in one filter.
//
// The earlier linear filter (vio_kf.h) took rotation from outside; leftover tilt leaked gravity into the
// integrated acceleration. Here tilt is a filter state, so accelerometer integration and the visual
// constraints correct it together.
//
// World axes = camera 0 (the output frame); gravity in those axes is a state.
// Error state (30): dtheta(3) dp(3) dv(3) dbg(3) dba(3) dg(3) | clone1: dtheta(3) dp(3) | clone0: dtheta(3) dp(3)
//   orientation errors are global: R_true = Exp(dtheta) R. Biases are in camera axes.
// Measurements:
//   - visual translation direction between the last two capture frames (perpendicular components)
//   - visual rotation residual between the same frames
//   - learned network velocity (weak pseudo-measurement)
#pragma once
#include "rotation.h"
#include <opencv2/core.hpp>
#include <algorithm>
#include <cmath>

#ifndef ESEKF_SIGMA_THETA
#define ESEKF_SIGMA_THETA 0.002     // rad per frame
#endif
#ifndef ESEKF_SIGMA_V
#define ESEKF_SIGMA_V 0.01          // m/s per frame
#endif
#ifndef ESEKF_SIGMA_BG
#define ESEKF_SIGMA_BG 0.00015      // rad/s per frame
#endif
#ifndef ESEKF_SIGMA_BA
#define ESEKF_SIGMA_BA 0.0005       // m/s^2 per frame
#endif
#ifndef ESEKF_SIGMA_HEADING
#define ESEKF_SIGMA_HEADING 0.05    // rad
#endif
#ifndef ESEKF_SIGMA_ROT
#define ESEKF_SIGMA_ROT 0.004       // rad, visual rotation residual
#endif
#ifndef ESEKF_ROT_SIGN
#define ESEKF_ROT_SIGN 1.0
#endif
#ifndef ESEKF_SIGMA_NET
#define ESEKF_SIGMA_NET 0.3         // m/s
#endif
#ifndef ESEKF_GATE
#define ESEKF_GATE 9.0
#endif

struct Esekf {
    static constexpr int N = 30;
    using VN = cv::Vec<double, N>;
    using MN = cv::Matx<double, N, N>;
    using M3 = cv::Matx33d;
    using V3 = cv::Vec3d;
    enum { TH = 0, P = 3, V = 6, BG = 9, BA = 12, G = 15, TH1 = 18, P1 = 21, TH0 = 24, P0 = 27 };

    M3 R = M3::eye(), R1 = M3::eye(), R0 = M3::eye();
    V3 p{0, 0, 0}, v{0, 0, 0}, bg{0, 0, 0}, ba{0, 0, 0}, g{0, 9.81, 0}, p1{0, 0, 0}, p0{0, 0, 0};
    MN Cov = MN::zeros();
    int captures = 0, heading_ok = 0, heading_soft = 0, rot_ok = 0, rot_soft = 0;
    double last_time = 0.;

    static M3 skew(const V3& a) { return {0, -a[2], a[1], a[2], 0, -a[0], -a[1], a[0], 0}; }
    void block(MN& m, int r, int c, const M3& b) const {
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) m(r + i, c + j) = b(i, j);
    }

    // First frame: gravity direction from the mean specific force (camera axes = world axes at frame 0).
    void init(const double* imu, int count, const M3& camera_from_imu) {
        V3 f{0, 0, 0};
        for (int i = 0; i < count; ++i) f += V3(imu[7 * i + 1], imu[7 * i + 2], imu[7 * i + 3]);
        if (count > 0 && cv::norm(f) > 1e-6) g = (camera_from_imu * f) * (-9.81 / cv::norm(f));
        for (int i = 0; i < 3; ++i) {
            Cov(V + i, V + i) = 0.25;          // initial velocity unknown (no start hover)
            Cov(BG + i, BG + i) = 0.02 * 0.02; // simulator gyro bias up to 0.02 rad/s
            Cov(BA + i, BA + i) = 0.2 * 0.2;   // accelerometer bias up to 0.2 m/s^2
            Cov(G + i, G + i) = 1.5 * 1.5;     // gravity direction from one (possibly accelerating) window
        }
    }

    void propagate(const double* imu, int count, double start, double end, const M3& camera_from_imu) {
        M3 dR = M3::eye();
        V3 dv{0, 0, 0}, dp{0, 0, 0};
        double previous = start;
        for (int s = 0; s < count; ++s) {
            const double* row = imu + 7 * s;
            const double t = std::min(row[0], end), h = t - previous;
            if (h <= 0.) continue;
            const V3 f = camera_from_imu * V3(row[1], row[2], row[3]) - ba;
            const V3 w = camera_from_imu * V3(row[4], row[5], row[6]) - bg;
            const V3 a = dR * f;
            dp += dv * h + a * (0.5 * h * h);
            dv += a * h;
            dR = dR * rotation_matrix(w * h);
            previous = t;
        }
        const double dt = end - start;
        const M3 Rk = R;
        MN F = MN::eye();
        block(F, TH, BG, Rk * (-dt));
        block(F, V, TH, skew(Rk * dv) * -1.);
        block(F, V, BA, Rk * (-dt));
        block(F, V, G, M3::eye() * dt);
        block(F, P, V, M3::eye() * dt);
        block(F, P, TH, skew(Rk * dp) * -1.);
        block(F, P, BA, Rk * (-0.5 * dt * dt));
        block(F, P, G, M3::eye() * (0.5 * dt * dt));
        p += v * dt + g * (0.5 * dt * dt) + Rk * dp;
        v += g * dt + Rk * dv;
        R = Rk * dR;
        Cov = F * Cov * F.t();
        const double qt = ESEKF_SIGMA_THETA * ESEKF_SIGMA_THETA, qv = ESEKF_SIGMA_V * ESEKF_SIGMA_V;
        const double qbg = ESEKF_SIGMA_BG * ESEKF_SIGMA_BG, qba = ESEKF_SIGMA_BA * ESEKF_SIGMA_BA;
        for (int i = 0; i < 3; ++i) {
            Cov(TH + i, TH + i) += qt;
            Cov(V + i, V + i) += qv;
            Cov(P + i, P + i) += qv * dt * dt * 0.25;
            Cov(BG + i, BG + i) += qbg;
            Cov(BA + i, BA + i) += qba;
            Cov(G + i, G + i) += 1e-8;
        }
    }

    // On capture frames: clone0 <- clone1, clone1 <- current pose.
    void capture() {
        MN T = MN::zeros();
        for (int i = 0; i < 18; ++i) T(i, i) = 1.;
        for (int i = 0; i < 3; ++i) {
            T(TH1 + i, TH + i) = 1.; T(P1 + i, P + i) = 1.;
            T(TH0 + i, TH1 + i) = 1.; T(P0 + i, P1 + i) = 1.;
        }
        Cov = T * Cov * T.t();
        R0 = R1; p0 = p1; R1 = R; p1 = p;
        ++captures;
    }

    void inject(const VN& d) {
        auto get = [&](int i) { return V3(d[i], d[i + 1], d[i + 2]); };
        R = rotation_matrix(get(TH)) * R;
        p += get(P); v += get(V); bg += get(BG); ba += get(BA);
        g += get(G); g *= 9.81 / cv::norm(g);
        R1 = rotation_matrix(get(TH1)) * R1; p1 += get(P1);
        R0 = rotation_matrix(get(TH0)) * R0; p0 += get(P0);
    }

    template <int M>
    bool update(const cv::Matx<double, M, N>& H, const cv::Vec<double, M>& r, cv::Matx<double, M, M> Rn, bool soft) {
        cv::Matx<double, M, M> S = H * Cov * H.t() + Rn;
        bool ok = false;
        cv::Matx<double, M, M> Si = S.inv(cv::DECOMP_CHOLESKY, &ok);
        if (!ok) return false;
        const double chi2 = (r.t() * Si * r)(0);
        if (!std::isfinite(chi2)) return false;
        const bool consistent = chi2 <= ESEKF_GATE;
        if (!consistent && soft) {  // down-weight instead of dropping, so the filter can recover
            Rn = Rn * (chi2 / ESEKF_GATE);
            S = H * Cov * H.t() + Rn;
            Si = S.inv(cv::DECOMP_CHOLESKY, &ok);
            if (!ok) return false;
        }
        const cv::Matx<double, N, M> K = Cov * H.t() * Si;
        inject(K * r);
        const MN IKH = MN::eye() - K * H;
        Cov = IKH * Cov * IKH.t() + K * Rn * K.t();
        return consistent;
    }

    // direction: unit translation direction clone0 -> clone1 in clone0 camera axes.
    void update_heading(const V3& direction, double weight) {
        if (captures < 2 || weight <= 0.) return;
        const V3 h = R0 * direction;
        const V3 axis = std::abs(h[2]) < .9 ? V3(0, 0, 1) : V3(1, 0, 0);
        V3 u = h.cross(axis); u *= 1. / cv::norm(u);
        const V3 w = h.cross(u);
        const V3 delta = p1 - p0;
        cv::Matx<double, 2, N> H = cv::Matx<double, 2, N>::zeros();
        const V3 bases[2] = {u, w};
        for (int k = 0; k < 2; ++k) {
            const V3 dth0 = bases[k].cross(delta);
            for (int i = 0; i < 3; ++i) {
                H(k, P1 + i) = bases[k][i];
                H(k, P0 + i) = -bases[k][i];
                H(k, TH0 + i) = dth0[i];
            }
        }
        const cv::Vec2d r(-u.dot(delta), -w.dot(delta));
        const double sigma = ESEKF_SIGMA_HEADING / weight * std::max(cv::norm(delta), 0.05);
        if (update<2>(H, r, cv::Matx22d::eye() * (sigma * sigma), true)) ++heading_ok; else ++heading_soft;
    }

    // residual: visual relative-rotation residual between clone0 and clone1, in clone0 camera axes.
    void update_rotation(const V3& residual) {
        if (captures < 2) return;
        const V3 z = (R0 * residual) * ESEKF_ROT_SIGN;
        cv::Matx<double, 3, N> H = cv::Matx<double, 3, N>::zeros();
        for (int i = 0; i < 3; ++i) { H(i, TH1 + i) = 1.; H(i, TH0 + i) = -1.; }
        const double s = ESEKF_SIGMA_ROT;
        if (update<3>(H, z, cv::Matx33d::eye() * (s * s), true)) ++rot_ok; else ++rot_soft;
    }

    void update_velocity(const V3& v_meas, double sigma) {
        cv::Matx<double, 3, N> H = cv::Matx<double, 3, N>::zeros();
        for (int i = 0; i < 3; ++i) H(i, V + i) = 1.;
        update<3>(H, v_meas - v, cv::Matx33d::eye() * (sigma * sigma), true);
    }
};
