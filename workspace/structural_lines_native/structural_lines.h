#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

// Single-frame structural evidence only. A direction is an unoriented camera-frame
// vanishing direction, not a known global axis or a semantic wall classification.
// Parallel image lines are supported: their homogeneous intersection is at infinity.
namespace structural_lines {
using cv::Matx33d;
using cv::Point2d;
using cv::Vec3d;

inline double norm3(const Vec3d& v) { return std::sqrt(v.dot(v)); }
inline Vec3d canonical(Vec3d v) {
    v *= 1. / std::max(1e-12, norm3(v));
    int k = 0;
    if (std::abs(v[1]) > std::abs(v[k])) k = 1;
    if (std::abs(v[2]) > std::abs(v[k])) k = 2;
    if (v[k] < 0.) v *= -1.;
    return v;
}

struct Config {
    int max_lines = 24;             // Hard post-detection cap; image itself is 160x120.
    int max_directions = 3;
    int min_support = 3;            // Three separate image lines, never three fragments.
    double min_length_px = 24.;
    double duplicate_angle_rad = 0.03490658503988659; // 2 degrees.
    double duplicate_distance_px = 3.;
    double inlier_angle_rad = 0.013962634015954637;    // 0.8 degrees to line plane.
    double minimum_pair_sine = 0.025;
    double minimum_conditioning = 0.004; // Middle eigenvalue / trace of sum(n n^T).
    double maximum_noise_ratio = 0.08;  // Smallest / middle eigenvalue.
    double minimum_center_span_px = 18.;
    double minimum_direction_separation_rad = 0.3490658503988659; // 20 degrees.
};

struct Line {
    Point2d first, last;
    Vec3d normal;                   // Unit camera-frame plane normal K^T*l.
    double length = 0.;
};

struct Direction {
    Vec3d axis{0., 0., 0.};
    int support = 0;
    double residual = 0.;           // RMS angle to supporting line planes, radians.
    double conditioning = 0.;       // Middle eigenvalue / trace; larger is better.
    double center_span_px = 0.;
    uint32_t support_mask = 0;      // Bits index Detection::lines, not raw LSD output.
};

struct Detection {
    std::vector<Line> lines;
    std::vector<Direction> directions;
    int raw_lines = 0;
    int rejected_short = 0;
    int rejected_duplicate = 0;
    int tested_pairs = 0;
    int rejected_geometry = 0;
    bool valid_image = false;
};

inline bool nearly_same_line(const Line& a, const Line& b, const Config& cfg) {
    const Point2d da = (a.last - a.first) * (1. / a.length);
    const Point2d db = (b.last - b.first) * (1. / b.length);
    if (std::abs(da.dot(db)) < std::cos(cfg.duplicate_angle_rad)) return false;
    const Point2d ma = (a.first + a.last) * .5, mb = (b.first + b.last) * .5;
    const Point2d separation = mb - ma;
    const double ea = std::abs(da.x * separation.y - da.y * separation.x);
    const double eb = std::abs(db.x * separation.y - db.y * separation.x);
    // Collinear fragments and the two LSD edges of one thin stripe count once,
    // even when their projected intervals do not overlap.
    return std::max(ea, eb) < cfg.duplicate_distance_px;
}

inline uint32_t support_mask(const std::vector<Line>& lines, uint32_t available,
                             const Vec3d& axis, double threshold) {
    uint32_t mask = 0;
    for (size_t i = 0; i < lines.size(); ++i)
        if ((available & (uint32_t(1) << i)) &&
            std::abs(lines[i].normal.dot(axis)) < threshold)
            mask |= uint32_t(1) << i;
    return mask;
}

inline int count_bits(uint32_t n) {
    int result = 0;
    while (n) { n &= n - 1; ++result; }
    return result;
}

inline bool fit_direction(const std::vector<Line>& lines, uint32_t mask,
                          const Config& cfg, Direction& out) {
    if (count_bits(mask) < cfg.min_support) return false;
    Matx33d information = Matx33d::zeros();
    double weight_sum = 0.;
    for (size_t i = 0; i < lines.size(); ++i) if (mask & (uint32_t(1) << i)) {
        const auto& line = lines[i];
        const double weight = std::min(80., line.length); // Bound one long edge's leverage.
        weight_sum += weight;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                information(r, c) += weight * line.normal[r] * line.normal[c];
    }
    Vec3d values;
    Matx33d vectors;
    if (!cv::eigen(information, values, vectors) || weight_sum <= 0. ||
        values[1] / weight_sum < cfg.minimum_conditioning ||
        values[2] > cfg.maximum_noise_ratio * values[1]) return false;
    const Vec3d axis = canonical({vectors(2, 0), vectors(2, 1), vectors(2, 2)});
    double error = 0., span = 0.;
    for (size_t i = 0; i < lines.size(); ++i) if (mask & (uint32_t(1) << i)) {
        const double angle = std::asin(std::min(1., std::abs(lines[i].normal.dot(axis))));
        if (angle >= cfg.inlier_angle_rad) return false;
        error += angle * angle;
        const Point2d center = (lines[i].first + lines[i].last) * .5;
        for (size_t j = 0; j < i; ++j) if (mask & (uint32_t(1) << j))
            span = std::max(span, cv::norm(center - (lines[j].first + lines[j].last) * .5));
    }
    if (span < cfg.minimum_center_span_px) return false;
    out = {axis, count_bits(mask), std::sqrt(error / count_bits(mask)),
           values[1] / weight_sum, span, mask};
    return true;
}

inline void extract_directions(Detection& result, const Config& cfg) {
    const auto& lines = result.lines;
    if (lines.size() < size_t(cfg.min_support)) return;
    uint32_t available = (uint32_t(1) << lines.size()) - 1;
    const double threshold = std::sin(cfg.inlier_angle_rad);
    struct Candidate { uint32_t mask; int support; double length; double error; };
    for (int group = 0; group < cfg.max_directions; ++group) {
        std::vector<Candidate> candidates;
        candidates.reserve(lines.size() * (lines.size() - 1) / 2);
        for (size_t i = 0; i < lines.size(); ++i) if (available & (uint32_t(1) << i))
            for (size_t j = 0; j < i; ++j) if (available & (uint32_t(1) << j)) {
                ++result.tested_pairs;
                Vec3d axis = lines[i].normal.cross(lines[j].normal);
                const double magnitude = norm3(axis);
                if (magnitude < cfg.minimum_pair_sine) continue;
                axis *= 1. / magnitude;
                const uint32_t mask = support_mask(lines, available, axis, threshold);
                const int count = count_bits(mask);
                if (count < cfg.min_support) continue;
                bool duplicate = false;
                for (const auto& candidate : candidates) if (candidate.mask == mask) { duplicate = true; break; }
                if (duplicate) continue;
                double length = 0., error = 0.;
                for (size_t k = 0; k < lines.size(); ++k) if (mask & (uint32_t(1) << k)) {
                    length += lines[k].length;
                    const double residual = lines[k].normal.dot(axis);
                    error += residual * residual;
                }
                candidates.push_back({mask, count, length, error});
            }
        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            if (a.support != b.support) return a.support > b.support;
            if (a.length != b.length) return a.length > b.length;
            if (a.error != b.error) return a.error < b.error;
            return a.mask < b.mask;
        });
        bool accepted = false;
        // Bound matrix solves even in textured/cluttered frames.
        const size_t tries = std::min(size_t(8), candidates.size());
        for (size_t k = 0; k < tries; ++k) {
            Direction direction;
            if (!fit_direction(lines, candidates[k].mask, cfg, direction)) {
                ++result.rejected_geometry; continue;
            }
            bool distinct = true;
            for (const auto& previous : result.directions)
                if (std::abs(previous.axis.dot(direction.axis)) >
                    std::cos(cfg.minimum_direction_separation_rad)) distinct = false;
            if (!distinct) { ++result.rejected_geometry; continue; }
            // Final support is the refined group. A line cannot support two axes.
            result.directions.push_back(direction);
            available &= ~direction.support_mask;
            accepted = true;
            break;
        }
        if (!accepted || count_bits(available) < cfg.min_support) break;
    }
}

class Detector {
public:
    explicit Detector(Config config = {}) : config_(config),
        lsd_(cv::createLineSegmentDetector(cv::LSD_REFINE_NONE, .8, .6, 2., 22.5, 0., .7, 64)) {
        // Mask and candidate work stay bounded even if caller supplies bad limits.
        config_.max_lines = std::clamp(config_.max_lines, 3, 24);
        config_.max_directions = std::clamp(config_.max_directions, 1, 3);
        config_.min_support = std::clamp(config_.min_support, 3, config_.max_lines);
    }

    Detection from_segments(const std::vector<cv::Vec4f>& segments,
                            double fx, double fy, double cx, double cy) const {
        Detection result;
        if (!std::isfinite(fx) || !std::isfinite(fy) || !std::isfinite(cx) ||
            !std::isfinite(cy) || fx <= 0. || fy <= 0.) return result;
        result.valid_image = true;
        result.raw_lines = int(segments.size());
        std::vector<Line> candidates;
        candidates.reserve(segments.size());
        for (const auto& segment : segments) {
            bool finite = true;
            for (int i = 0; i < 4; ++i) finite &= std::isfinite(segment[i]);
            if (!finite) { ++result.rejected_short; continue; }
            Point2d a(segment[0], segment[1]), b(segment[2], segment[3]);
            if (a.x > b.x || (a.x == b.x && a.y > b.y)) std::swap(a, b);
            const double length = cv::norm(b - a);
            if (length < config_.min_length_px) { ++result.rejected_short; continue; }
            const Vec3d ray_a{(a.x - cx) / fx, (a.y - cy) / fy, 1.};
            const Vec3d ray_b{(b.x - cx) / fx, (b.y - cy) / fy, 1.};
            const Vec3d normal = ray_a.cross(ray_b);
            const double magnitude = norm3(normal);
            if (magnitude < 1e-9) { ++result.rejected_short; continue; }
            candidates.push_back({a, b, normal / magnitude, length});
        }
        std::sort(candidates.begin(), candidates.end(), [](const Line& a, const Line& b) {
            if (a.length != b.length) return a.length > b.length;
            if (a.first.x != b.first.x) return a.first.x < b.first.x;
            if (a.first.y != b.first.y) return a.first.y < b.first.y;
            if (a.last.x != b.last.x) return a.last.x < b.last.x;
            return a.last.y < b.last.y;
        });
        for (const auto& candidate : candidates) {
            bool duplicate = false;
            for (const auto& line : result.lines)
                if (nearly_same_line(candidate, line, config_)) { duplicate = true; break; }
            if (duplicate) { ++result.rejected_duplicate; continue; }
            result.lines.push_back(candidate);
            if (int(result.lines.size()) >= config_.max_lines) break;
        }
        extract_directions(result, config_);
        return result;
    }

    Detection detect(const cv::Mat& gray, double fx, double fy, double cx, double cy) {
        if (gray.type() != CV_8UC1 || gray.cols != 160 || gray.rows != 120) return {};
        std::vector<cv::Vec4f> segments;
        lsd_->detect(gray, segments);
        return from_segments(segments, fx, fy, cx, cy);
    }

    const Config& config() const { return config_; }

private:
    Config config_;
    cv::Ptr<cv::LineSegmentDetector> lsd_;
};
} // namespace structural_lines
