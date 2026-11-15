#pragma once

#include "split_corners.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace visual_direction {
using cv::Matx33d;
using cv::Point2f;
using cv::Vec3d;

struct Match {
    Vec3d anchor;
    Vec3d current;
    Vec3d normal;
    Vec3d unit_normal;
};

inline double length(const Vec3d& v) { return std::sqrt(v.dot(v)); }

// The two rays are expressed in the anchor camera's axes. For any valid
// match, its epipolar-plane normal is perpendicular to camera translation.
inline bool estimate_direction(const std::vector<Match>& matches, bool sparse, Vec3d& direction) {
    constexpr double inlier_threshold = 0.22;
    const size_t minimum_tracks = sparse ? 8 : 10;
    const size_t count = matches.size();
    if (count < minimum_tracks) return false;
    Vec3d best;
    size_t best_count = 0;
    double best_error = 0.;
    uint32_t seed = 7919u;
    for (int trial = 0; trial < 36; ++trial) {
        seed = 1664525u * seed + 1013904223u;
        const size_t i = (seed >> 8) % count;
        seed = 1664525u * seed + 1013904223u;
        const size_t j = (seed >> 8) % count;
        Vec3d candidate = matches[i].unit_normal.cross(matches[j].unit_normal);
        const double magnitude = length(candidate);
        if (magnitude < 0.15) continue;
        candidate *= 1. / magnitude;
        size_t inliers = 0;
        double error = 0.;
        for (const auto& match : matches) {
            const double residual = std::abs(match.unit_normal.dot(candidate));
            if (residual < inlier_threshold) {
                ++inliers;
                error += residual * residual;
            }
        }
        if (inliers > best_count || (inliers == best_count && error < best_error)) {
            best = candidate;
            best_count = inliers;
            best_error = error;
        }
    }
    if (best_count < minimum_tracks || 5 * best_count < 3 * count) return false;

    Matx33d normal_matrix = Matx33d::zeros();
    for (const auto& match : matches) {
        if (std::abs(match.unit_normal.dot(best)) >= inlier_threshold) continue;
        const auto& normal = match.unit_normal;
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) normal_matrix(row, col) += normal[row] * normal[col];
        }
    }
    Vec3d eigenvalues;
    Matx33d eigenvectors;
    // With only eight or nine tracks, require better spread in the epipolar
    // planes and a clearer separation of the fitted direction from the noise.
    if (!cv::eigen(normal_matrix, eigenvalues, eigenvectors) ||
        eigenvalues[1] < (sparse ? .05 : .025) * best_count ||
        eigenvalues[2] > (sparse ? .10 : .25) * eigenvalues[1])
        return false;
    direction = {eigenvectors(2, 0), eigenvectors(2, 1), eigenvectors(2, 2)};

    // For r0*depth0 = t + r1*depth1, cross with r1/r0 respectively.
    // Choose the sign that puts most inliers in front of both cameras.
    size_t positive = 0, negative = 0, supported = 0;
    for (const auto& match : matches) {
        if (std::abs(match.unit_normal.dot(direction)) >= inlier_threshold) continue;
        const double depth0 = direction.cross(match.current).dot(match.normal);
        const double depth1 = direction.cross(match.anchor).dot(match.normal);
        positive += depth0 > 0. && depth1 > 0.;
        negative += depth0 < 0. && depth1 < 0.;
        ++supported;
    }
    const size_t consistent_depths = std::max(positive, negative);
    if (supported < minimum_tracks || 20 * consistent_depths < (sparse ? 17 : 14) * supported) return false;
    if (negative > positive) direction *= -1.;
    return true;
}

class Tracker {
public:
    static_assert(VISUAL_INTERVAL >= 8);
    static constexpr int scale = VISUAL_DOWNSAMPLE;
    static constexpr int image_width = 320;
    static constexpr int width = image_width / scale, height = 240 / scale;
    static constexpr int max_features = 48;
    static constexpr int levels = VISUAL_PYRAMID_LEVELS;

    void configure(double fx, double fy, double cx, double cy) {
        fx_ = fx / scale;
        fy_ = fy / scale;
        cx_ = (cx + .5) / scale - .5;
        cy_ = (cy + .5) / scale - .5;
    }

    Vec3d step(const uint8_t* rgb, const Matx33d& camera_to_world, const Vec3d& imu_position) {
        switch (frame_index_ % VISUAL_INTERVAL) {
        case 0:
            make_gray(rgb);
            cv::buildOpticalFlowPyramid(current_, current_pyramid_, patch_, levels, true,
                cv::BORDER_REFLECT_101, cv::BORDER_CONSTANT, false);
            capture_rotation_ = camera_to_world;
            capture_position_ = imu_position;
            current_to_anchor_ = anchor_rotation_.t() * capture_rotation_;
            observations_.clear();
            if (!anchor_pyramid_.empty()) observations_.resize(points_.size());
            if constexpr (scale == 2) track_forward(0);
            break;
        case 1:
            track_forward(scale == 2 ? 1 : 0);
            if constexpr (scale == 2) track_reverse(1);
            break;
        case 2:
            track_forward(scale == 2 ? 2 : 1);
            break;
        case 3:
            if constexpr (scale == 2) track_forward(3);
            track_reverse(3);
            break;
        case 4:
            if constexpr (scale == 2) track_forward(4);
            track_reverse(4);
            finish_observation();
            break;
        case 5:
            detector_.compute_half(anchor_, 0);
            break;
        case 6:
            detector_.compute_half(anchor_, 1);
            break;
        case 7:
            detector_.select(points_, max_features, 16 / scale);
            break;
        }
        ++frame_index_;
        if (correction_frames_ == 0) return {0., 0., 0.};
        --correction_frames_;
        return pending_;
    }

private:
    double fx_ = 1., fy_ = 1., cx_ = 0., cy_ = 0.;
    int frame_index_ = 0, correction_frames_ = 0;
    cv::Mat anchor_, current_;
    std::vector<cv::Mat> anchor_pyramid_, current_pyramid_;
    struct Observation {
        Point2f after;
        Match match;
        bool forward_ok = false;
        bool reverse_ok = false;
    };

    SplitCorners<width> detector_;
    std::vector<Point2f> points_;
    std::vector<Observation> observations_;
    const cv::Size patch_{scale == 2 ? 13 : 9, scale == 2 ? 13 : 9};
    const cv::TermCriteria criteria_{cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 10, .015};
    Matx33d anchor_rotation_ = Matx33d::eye(), capture_rotation_ = Matx33d::eye();
    Matx33d current_to_anchor_ = Matx33d::eye();
    Vec3d anchor_position_{0., 0., 0.}, capture_position_{0., 0., 0.}, pending_{0., 0., 0.};
    Vec3d previous_visual_heading_{0., 0., 0.}, previous_inertial_heading_{0., 0., 0.};
    double previous_baseline_ = 0.;
    bool previous_heading_valid_ = false;

    void make_gray(const uint8_t* rgb) {
        current_.create(height, width, CV_8UC1);
        constexpr int stride = image_width * 3;
        for (int y = 0; y < height; ++y) {
            const auto* source = rgb + scale * y * stride + 1;
            auto* dest = current_.ptr<uint8_t>(y);
            for (int x = 0; x < width; ++x, source += 3 * scale) {
                int sum = 0;
                for (int dy = 0; dy < scale; ++dy)
                    for (int dx = 0; dx < scale; ++dx)
                        sum += source[dy * stride + 3 * dx];
                dest[x] = (sum + scale * scale / 2) / (scale * scale);
            }
        }
    }

    Vec3d ray(const Point2f& p) const { return {(p.x - cx_) / fx_, (p.y - cy_) / fy_, 1.}; }

    Point2f pixel(const Vec3d& p) const {
        return {static_cast<float>(fx_ * p[0] / p[2] + cx_),
                static_cast<float>(fy_ * p[1] / p[2] + cy_)};
    }

    void track_forward(int batch_index) {
        // Ranks assigned to phases 3 and 4 are processed before their
        // respective backward pass. The match order is still the corner order.
        static constexpr std::array<int, 16> phase_for_rank{
            3, 4, 2, 1, 0, 2, 1, 2, 0, 1, 2, 1, 0, 2, 1, 2};
        std::vector<size_t> indices;
        std::vector<Point2f> before, after;
        const auto anchor_to_current = current_to_anchor_.t();
        for (size_t i = 0; i < observations_.size(); ++i) {
            const int batch = scale == 2 ? phase_for_rank[i % phase_for_rank.size()]
                                         : static_cast<int>(i >= (observations_.size() + 1) / 2);
            if (batch != batch_index) continue;
            const auto& point = points_[i];
            const auto mapped = anchor_to_current * ray(point);
            indices.push_back(i);
            before.push_back(point);
            after.push_back(mapped[2] > .2 ? pixel(mapped) : point);
        }
        if (before.empty()) return;

        std::vector<uint8_t> status;
        std::vector<float> errors;
        cv::calcOpticalFlowPyrLK(anchor_pyramid_, current_pyramid_, before, after,
            status, errors, patch_, levels, criteria_, cv::OPTFLOW_USE_INITIAL_FLOW);
        constexpr float border = scale == 2 ? 6.f : 4.f;
        constexpr float min_parallax_squared = scale == 2 ? .16f : .0625f;
        for (size_t i = 0; i < before.size(); ++i) {
            const auto& point = after[i];
            if (!status[i] || errors[i] > 24.f ||
                !(point.x > border && point.x < width - border &&
                  point.y > border && point.y < height - border)) continue;
            const auto a = ray(before[i]);
            auto b = current_to_anchor_ * ray(point);
            if (b[2] < .2) continue;
            b *= 1. / b[2];
            const double dx = (a[0] - b[0]) * fx_, dy = (a[1] - b[1]) * fy_;
            if (dx * dx + dy * dy < min_parallax_squared) continue;
            const auto normal = a.cross(b);
            const double magnitude = length(normal);
            if (!(magnitude > 1e-8)) continue;

            auto& observation = observations_[indices[i]];
            observation.after = point;
            observation.match = {a, b, normal, normal * (1. / magnitude)};
            observation.forward_ok = true;
        }
    }

    void track_reverse(int phase) {
        // Backward tracking is needed only for observations that passed the
        // forward photometric, image-boundary and parallax checks.
        std::vector<size_t> indices;
        std::vector<Point2f> current, reverse;
        for (size_t i = 0; i < observations_.size(); ++i) {
            // At 160x120, these ranks have already run forward in phase 1.
            const bool track_early = scale == 2 && i % 16 == 3;
            const int reverse_phase = track_early ? 1 : 3 + i % 2;
            if (phase != reverse_phase || !observations_[i].forward_ok) continue;
            indices.push_back(i);
            current.push_back(observations_[i].after);
            reverse.push_back(points_[i]);
        }
        if (indices.empty()) return;

        std::vector<uint8_t> status;
        cv::calcOpticalFlowPyrLK(current_pyramid_, anchor_pyramid_, current, reverse,
            status, cv::noArray(), patch_, levels, criteria_, cv::OPTFLOW_USE_INITIAL_FLOW);
        for (size_t i = 0; i < indices.size(); ++i) {
            const auto back_error = reverse[i] - points_[indices[i]];
            observations_[indices[i]].reverse_ok = status[i] && !(back_error.dot(back_error) > .49f);
        }
    }

    bool supports_previous_heading(const std::vector<Match>& matches,
                                   const Vec3d& prior, double baseline) const {
        if (!previous_heading_valid_ || matches.size() < 6 || baseline <= .005 ||
            baseline < .5 * previous_baseline_ || baseline > 2. * previous_baseline_ ||
            prior.dot(previous_inertial_heading_) < .90 * baseline)
            return false;
        const double rotation_trace = current_to_anchor_(0, 0) + current_to_anchor_(1, 1)
                                    + current_to_anchor_(2, 2);
        if (rotation_trace < 2.90) return false;
        const auto direction = anchor_rotation_.t() * previous_visual_heading_;
        size_t supporting = 0;
        for (const auto& match : matches) {
            if (std::abs(match.unit_normal.dot(direction)) < .30 &&
                direction.cross(match.current).dot(match.normal) > 0. &&
                direction.cross(match.anchor).dot(match.normal) > 0.)
                ++supporting;
        }
        return supporting >= 4 && 2 * supporting >= matches.size();
    }

    void finish_observation() {
        std::vector<Match> matches;
        matches.reserve(observations_.size());
        for (const auto& observation : observations_)
            if (observation.reverse_ok) matches.push_back(observation.match);
        Vec3d direction;
        const auto prior = capture_position_ - anchor_position_;
        const double baseline = length(prior);
        const bool sparse = VISUAL_SPARSE_DIRECTION && scale == 2 && matches.size() < 10;
        if (baseline > .005 && estimate_direction(matches, sparse, direction)) {
            const Vec3d heading = anchor_rotation_ * direction;
            const double weight = sparse ? .85 : 1.;
            pending_ = (heading * baseline - prior) * (weight / VISUAL_INTERVAL);
            correction_frames_ = VISUAL_INTERVAL;
            previous_visual_heading_ = heading;
            previous_inertial_heading_ = prior * (1. / baseline);
            previous_baseline_ = baseline;
            previous_heading_valid_ = true;
        } else {
            if (supports_previous_heading(matches, prior, baseline)) {
                pending_ = (previous_visual_heading_ * baseline - prior) * (.55 / VISUAL_INTERVAL);
                correction_frames_ = VISUAL_INTERVAL;
            }
            // Extrapolation can only follow a directly observed heading.
            previous_heading_valid_ = false;
        }
        std::swap(anchor_, current_);
        std::swap(anchor_pyramid_, current_pyramid_);
        anchor_rotation_ = capture_rotation_;
        anchor_position_ = capture_position_;
    }
};
} // namespace visual_direction
