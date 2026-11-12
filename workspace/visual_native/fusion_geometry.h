#pragma once

#include "rotation.h"
#include <opencv2/core.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace visual_direction {
using cv::Matx33d;
using cv::Vec3d;

struct Match {
    Vec3d anchor, current, normal, unit_normal;
    double backward_error = 0.;
    double photometric_error = 0.;
};

inline double length(const Vec3d& v) { return std::sqrt(v.dot(v)); }

// A bounded engineering quality score, not a calibrated probability.
inline double fusion_quality(const std::vector<Match>& matches, const Vec3d& direction,
                             double fx, double fy) {
    double n = 0., error = 0., backward = 0., photo = 0., parallax = 0.;
    Vec3d mean{0., 0., 0.};
    Matx33d second = Matx33d::zeros();
    for (const auto& m : matches) {
        const double r = std::abs(m.unit_normal.dot(direction));
        if (r >= .22) continue;
        ++n;
        error += r * r;
        backward += m.backward_error * m.backward_error;
        photo += m.photometric_error;
        parallax += std::hypot((m.anchor[0] - m.current[0]) * fx,
                              (m.anchor[1] - m.current[1]) * fy);
        mean += m.anchor;
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) second(i,j) += m.anchor[i] * m.anchor[j];
    }
    if (n == 0.) return 0.;
    mean *= 1. / n;
    const double xx = std::max(0., second(0,0) / n - mean[0] * mean[0]);
    const double yy = std::max(0., second(1,1) / n - mean[1] * mean[1]);
    const double xy = second(0,1) / n - mean[0] * mean[1];
    const double minor = std::max(0., .5 * (xx + yy - std::hypot(xx - yy, 2. * xy)));
    const double support = std::min(1., n / (0.85 * matches.size()));
    const double spread = std::clamp(std::sqrt(minor) / .20, 0., 1.);
    const double fit = std::clamp(1. - std::sqrt(error / n) / .22, 0., 1.);
    const double tracking = std::clamp(1. - std::sqrt(backward / n) / .70, 0., 1.);
    const double appearance = std::clamp(1. - photo / (n * 24.), 0., 1.);
    const double motion = std::clamp(parallax / (n * 2.), 0., 1.);
    return std::sqrt(support * spread * fit * tracking * appearance * motion);
}

// Sampson-style epipolar distance; both rays start in the gyro anchor axes.
// Translation direction is optimized with rotation, so translation flow is
// never treated as a pure rotational correspondence.
inline double epipolar_residual(const Match& m, const Matx33d& correction,
                               const Vec3d& direction) {
    Vec3d b = correction * m.current;
    if (!(b[2] > .1)) return 1e3;
    b *= 1. / b[2];
    const auto la = direction.cross(b), lb = direction.cross(m.anchor);
    const double denom = std::sqrt(la[0]*la[0] + la[1]*la[1]
                                 + lb[0]*lb[0] + lb[1]*lb[1] + 1e-12);
    return direction.dot(m.anchor.cross(b)) / denom;
}

inline double robust_cost(double r) {
    const double a = std::abs(r);
    return a <= 1.5 ? .5 * r * r : 1.5 * (a - .75);
}

// Four bounded Gauss-Newton iterations over 3 rotation + 2 direction DOF.
// The gyro prior regularizes rotation. A Schur-complement check rejects
// rotation that cannot be separated from translation in the observed rays.
inline bool refine_rotation(const std::vector<Match>& matches, const Vec3d& initial_direction,
                            double focal, Vec3d& result) {
    result = {0., 0., 0.};
    if (matches.size() < 12 || !(focal > 0.) || !std::isfinite(focal)) return false;
    const Vec3d axis = std::abs(initial_direction[2]) < .9 ? Vec3d(0.,0.,1.) : Vec3d(1.,0.,0.);
    Vec3d u = initial_direction.cross(axis);
    if (length(u) < 1e-8) return false;
    u *= 1. / length(u);
    const Vec3d v = initial_direction.cross(u);
    using V5 = cv::Vec<double,5>;
    using M5 = cv::Matx<double,5,5>;
    V5 parameters = V5::all(0.);
    const double sigma = .5 / focal;
    constexpr double prior_information = 10000.; // 0.01 rad prior standard deviation
    constexpr double epsilon = 1e-5;
    auto pose = [&](const V5& p, Matx33d& r, Vec3d& t) {
        r = rotation_matrix({p[0],p[1],p[2]});
        t = initial_direction + u * p[3] + v * p[4];
        t *= 1. / length(t);
    };
    auto cost = [&](const V5& p) {
        Matx33d r; Vec3d t; pose(p,r,t);
        double sum = 0.;
        for (const auto& m : matches) sum += robust_cost(epipolar_residual(m,r,t) / sigma);
        return sum;
    };
    const double before = cost(parameters);
    if (!std::isfinite(before) || before < 1e-6) return false;
    M5 data_information = M5::zeros();
    for (int iteration = 0; iteration < 4; ++iteration) {
        Matx33d r; Vec3d t; pose(parameters,r,t);
        std::array<Matx33d,5> shifted_r;
        std::array<Vec3d,5> shifted_t;
        for (int j=0;j<5;++j) {
            auto p=parameters; p[j]+=epsilon; pose(p,shifted_r[j],shifted_t[j]);
        }
        M5 h = M5::zeros(); V5 g = V5::all(0.);
        for (const auto& m : matches) {
            const double residual = epipolar_residual(m,r,t) / sigma;
            const double weight = std::min(1., 1.5 / std::max(1e-12,std::abs(residual)));
            V5 jacobian;
            for (int j=0;j<5;++j)
                jacobian[j]=(epipolar_residual(m,shifted_r[j],shifted_t[j])/sigma-residual)/epsilon;
            for(int i=0;i<5;++i) {
                g[i]+=weight*jacobian[i]*residual;
                for(int j=0;j<5;++j) h(i,j)+=weight*jacobian[i]*jacobian[j];
            }
        }
        data_information=h;
        for(int j=0;j<3;++j) { h(j,j)+=prior_information; g[j]+=prior_information*parameters[j]; }
        h(3,3)+=1e-6; h(4,4)+=1e-6;
        V5 delta;
        if(!cv::solve(h,-g,delta,cv::DECOMP_CHOLESKY)) return false;
        parameters+=delta;
        for(double x:parameters.val) if(!std::isfinite(x)) return false;
        if(length({parameters[0],parameters[1],parameters[2]})>.015 ||
           std::hypot(parameters[3],parameters[4])>.5) return false;
    }
    const cv::Matx22d tt(data_information(3,3),data_information(3,4),
                        data_information(4,3),data_information(4,4));
    bool invertible=false;
    const auto tt_inverse=tt.inv(cv::DECOMP_CHOLESKY,&invertible);
    if(!invertible) return false;
    cv::Matx<double,3,2> rt;
    Matx33d rr;
    for(int i=0;i<3;++i) {
        for(int j=0;j<3;++j) rr(i,j)=data_information(i,j);
        for(int j=0;j<2;++j) rt(i,j)=data_information(i,j+3);
    }
    Vec3d eigenvalues;
    if(!cv::eigen(rr-rt*tt_inverse*rt.t(),eigenvalues) || eigenvalues[2]<25.) return false;
    Matx33d r; Vec3d t; pose(parameters,r,t);
    size_t inliers=0;
    for(const auto& m:matches) inliers+=std::abs(epipolar_residual(m,r,t))<=2.*sigma;
    if(4*inliers<3*matches.size() || cost(parameters)>.95*before) return false;
    // Conservative application gain; never use evaluation ground truth here.
    result=Vec3d(parameters[0],parameters[1],parameters[2])*.25;
    return length(result)>1e-8;
}
} // namespace visual_direction
