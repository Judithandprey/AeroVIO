#pragma once

#include "fixed_lk.h"
#include "flow_pyramid.h"
#include "split_corners.h"
#include <opencv2/imgproc.hpp>
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

constexpr int max_features = 48;

// Reversing a pair negates the candidate and leaves the absolute residuals unchanged.
// Keep each pair's first orientation and its place in the original trial order.
constexpr int direction_trials = 36;
struct DirectionSample { uint8_t first, second; };
struct DirectionSamples {
    std::array<DirectionSample, direction_trials> pairs{};
    uint8_t size = 0;
};

constexpr auto make_direction_samples() {
    std::array<DirectionSamples, max_features + 1> table{};
    for (unsigned count = 8; count <= max_features; ++count) {
        std::array<uint64_t, max_features> seen{};
        uint32_t seed = 7919u;
        auto& list = table[count];
        for (int trial = 0; trial < direction_trials; ++trial) {
            seed = 1664525u * seed + 1013904223u;
            const unsigned i = (seed >> 8) % count;
            seed = 1664525u * seed + 1013904223u;
            const unsigned j = (seed >> 8) % count;
            if (i == j) continue;
            auto& row = seen[std::min(i, j)];
            const uint64_t bit = uint64_t(1) << std::max(i, j);
            if (row & bit) continue;
            row |= bit;
            list.pairs[list.size++] = {static_cast<uint8_t>(i), static_cast<uint8_t>(j)};
        }
    }
    return table;
}

constexpr auto direction_samples = make_direction_samples();

// The two rays are expressed in the anchor camera's axes. For any valid
// match, its epipolar-plane normal is perpendicular to camera translation.
inline bool estimate_direction(const Match* const* matches, size_t count, bool sparse, Vec3d& direction) {
    constexpr double inlier_threshold = 0.22;
    const size_t minimum_tracks = sparse ? 8 : 10;
    if (count < minimum_tracks) return false;
    struct NormalPair { v128_t x, y, z; };
    std::array<NormalPair, max_features / 2> normals;
    const size_t pairs = count / 2;
    for (size_t pair = 0; pair < pairs; ++pair) {
        const auto& first = matches[2 * pair]->unit_normal;
        const auto& second = matches[2 * pair + 1]->unit_normal;
        normals[pair] = {wasm_f64x2_make(first[0], second[0]), wasm_f64x2_make(first[1], second[1]),
                         wasm_f64x2_make(first[2], second[2])};
    }
    Vec3d best;
    size_t best_count = 0;
    double best_error = 0.;
    const auto threshold = wasm_f64x2_splat(inlier_threshold);
    const auto& samples = direction_samples[count];
    for (size_t trial = 0; trial < samples.size; ++trial) {
        const auto [i, j] = samples.pairs[trial];
        Vec3d candidate = matches[i]->unit_normal.cross(matches[j]->unit_normal);
        const double magnitude = length(candidate);
        if (magnitude < 0.15) continue;
        candidate *= 1. / magnitude;
        double error = 0.;
        const auto cx = wasm_f64x2_splat(candidate[0]);
        const auto cy = wasm_f64x2_splat(candidate[1]);
        const auto cz = wasm_f64x2_splat(candidate[2]);
        auto counts = wasm_i64x2_splat(0);
        const auto scored_count = [&] {
            return static_cast<size_t>(wasm_i64x2_extract_lane(counts, 0) + wasm_i64x2_extract_lane(counts, 1));
        };
        const auto score_pair = [&](size_t pair) {
            const auto& normal = normals[pair];
            const auto xy = wasm_f64x2_add(wasm_f64x2_mul(normal.x, cx), wasm_f64x2_mul(normal.y, cy));
            const auto residual = wasm_f64x2_abs(wasm_f64x2_add(xy, wasm_f64x2_mul(normal.z, cz)));
            const auto accepted = wasm_f64x2_lt(residual, threshold);
            const auto squared = wasm_v128_and(wasm_f64x2_mul(residual, residual), accepted);
            // Keep the scalar accumulation order for candidate tie breaks.
            error += wasm_f64x2_extract_lane(squared, 0);
            error += wasm_f64x2_extract_lane(squared, 1);
            // Accepted lanes are all ones, so subtraction increments their counts.
            counts = wasm_i64x2_sub(counts, accepted);
        };
        size_t pair = 0;
        bool discarded = false;
        for (; pair + 8 <= pairs; pair += 8) {
#pragma clang loop unroll(full)
            for (size_t offset = 0; offset < 8; ++offset) score_pair(pair + offset);
            // Discard candidates that cannot win even if all remaining matches agree.
            const auto possible = scored_count() + count - 2 * (pair + 8);
            if (possible < best_count || (possible == best_count && error >= best_error)) {
                discarded = true;
                break;
            }
        }
        if (discarded) continue;
        for (; pair < pairs; ++pair) score_pair(pair);
        size_t inliers = scored_count();
        if (count % 2) {
            const double residual = std::abs(matches[count - 1]->unit_normal.dot(candidate));
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
    for (size_t k = 0; k < count; ++k) {
        const auto& match = *matches[k];
        if (std::abs(match.unit_normal.dot(best)) >= inlier_threshold) continue;
        const auto& normal = match.unit_normal;
        for (int row = 0; row < 3; ++row) {
            for (int col = row; col < 3; ++col) normal_matrix(row, col) += normal[row] * normal[col];
        }
    }
    // Mirror the symmetric entries before passing the matrix to OpenCV.
    normal_matrix(1, 0) = normal_matrix(0, 1);
    normal_matrix(2, 0) = normal_matrix(0, 2);
    normal_matrix(2, 1) = normal_matrix(1, 2);
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
    for (size_t k = 0; k < count; ++k) {
        const auto& match = *matches[k];
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

struct TrackingBatch {
    std::array<uint8_t, max_features> ranks{};
    size_t count = 0;
};

template<int Scale, bool Forward>
constexpr auto make_tracking_batches() {
    std::array<TrackingBatch, 5> batches{};
    const auto append = [&](int batch, int rank) {
        auto& entry = batches[batch];
        entry.ranks[entry.count++] = static_cast<uint8_t>(rank);
    };
    for (int rank = 0; rank < max_features; ++rank) {
        int batch = 1 + rank % 4;
        if (rank == 15) batch = 2;
        if constexpr (Scale == 4) {
            if (rank == 1) batch = 3;
            if (rank == 31) batch = 2;
        }
        if constexpr (Scale == 2 && Forward) {
            if (rank == 0 || rank == 15 || rank % 24 == 1) batch = 0;
            if (rank == 5) append(0, rank);  // Prepare only the coarse estimate.
        }
        append(batch, rank);
    }
    return batches;
}

class Tracker {
public:
    static_assert(VISUAL_INTERVAL >= 8);
    static constexpr int scale = VISUAL_DOWNSAMPLE;
    static constexpr int image_width = 320;
    static constexpr int width = image_width / scale, height = 240 / scale;
    static constexpr int levels = VISUAL_PYRAMID_LEVELS;
    static constexpr int window = scale == 2 ? 13 : 9;

    void configure(double fx, double fy, double cx, double cy) {
        fx_ = fx / scale;
        fy_ = fy / scale;
        cx_ = (cx + .5) / scale - .5;
        cy_ = (cy + .5) / scale - .5;
        for (auto* pyramid : {&current_pyramid_, &anchor_pyramid_}) {
            pyramid->resize(2 * (levels + 1));
            for (int level = 0; level <= levels; ++level) {
                allocate_padded((*pyramid)[2 * level], width >> level, height >> level, window);
                allocate_gradient((*pyramid)[2 * level + 1], width >> level, height >> level, window);
            }
        }
    }

    Vec3d step(const uint8_t* rgb, const Matx33d& camera_to_world, const Vec3d& imu_position) {
        const int phase = frame_index_ % VISUAL_INTERVAL;
        switch (phase) {
        case 0:
            make_gray(rgb, current_pyramid_[0]);
            reflect_border<width, height, window>(current_pyramid_[0]);
            scharr<width, 0, height>(current_pyramid_[0], current_pyramid_[1]);
            if constexpr (levels == 1) {
                pyramid_down<width, height>(current_pyramid_[0], current_pyramid_[2]);
                reflect_border<width / 2, height / 2, window>(current_pyramid_[2]);
            }
            capture_rotation_ = camera_to_world;
            capture_position_ = imu_position;
            current_to_anchor_ = anchor_rotation_.t() * capture_rotation_;
            observations_.clear();
            observations_.resize(points_.size());
            if constexpr (levels == 1) {
                int work = 0;
                track_forward(0, &work);
                // The early forward pass reads only the current images, so its
                // completed iteration count can guide the gradient split.
                constexpr int coarse_height = height / 2;
                constexpr int half = height / 4;
                constexpr int full_budget = 55;
                constexpr int half_budget = 64;
                if (observations_.empty()) {
                    coarse_rows_ = 0;
                } else if (work <= full_budget) {
                    scharr<width / 2, 0, coarse_height>(current_pyramid_[2], current_pyramid_[3]);
                    coarse_rows_ = coarse_height;
                } else if (work <= half_budget) {
                    scharr<width / 2, 0, half>(current_pyramid_[2], current_pyramid_[3]);
                    coarse_rows_ = half;
                } else {
                    coarse_rows_ = 0;
                }
            }
            break;
        case 1:
            // Finish the coarse gradient before the first reverse batch.
            if constexpr (levels == 1) {
                if (coarse_rows_ == 0)
                    scharr<width / 2, 0, height / 2>(current_pyramid_[2], current_pyramid_[3]);
                else if (coarse_rows_ == height / 4)
                    scharr<width / 2, height / 4, height / 2>(current_pyramid_[2], current_pyramid_[3]);
            }
            [[fallthrough]];
        case 2:
        case 3:
        case 4:
            track_forward(phase);
            track_reverse(phase);
            if (phase == 4) finish_observation();
            break;
        case 5:
            detector_.compute_half(anchor_pyramid_[0], 0);
            break;
        case 6:
            detector_.compute_half(anchor_pyramid_[0], 1);
            break;
        case 7:
            detector_.select<16 / scale>(points_, max_features);
            prepare_anchor_patches();
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
    int coarse_rows_ = 0;
    std::vector<cv::Mat> anchor_pyramid_, current_pyramid_;
    // The binned selector leaves room to prepare all 160x120 anchor patches.
    static constexpr size_t cached_points = scale == 2 ? max_features : 36;
    static_assert(levels == 0 || cached_points == max_features);
    std::array<std::array<fixed_lk::SourcePatch<window>, levels + 1>, cached_points> anchor_patches_;
    // The finest forward check fills these samples before setting forward_ok.
    std::array<fixed_lk::IntensityPatch<window>, levels == 1 ? max_features : 0> sampled_forward_;
    struct Observation {
        // Holds the coarse estimate until the forward batch finishes.
        Point2f after;
        Match match;
        bool forward_ok = false;
        bool reverse_ok = false;
    };

    SplitCorners<width, height> detector_;
    std::vector<Point2f> points_;
    std::vector<Observation> observations_;
    Matx33d anchor_rotation_ = Matx33d::eye(), capture_rotation_ = Matx33d::eye();
    Matx33d current_to_anchor_ = Matx33d::eye();
    Vec3d anchor_position_{0., 0., 0.}, capture_position_{0., 0., 0.}, pending_{0., 0., 0.};
    Vec3d previous_visual_heading_{0., 0., 0.}, previous_inertial_heading_{0., 0., 0.};
    double previous_baseline_ = 0.;
    bool previous_heading_valid_ = false;

    static void make_gray(const uint8_t* rgb, cv::Mat& gray) {
        constexpr int stride = image_width * 3;
        static_assert(width % (32 / scale) == 0);
        for (int y = 0; y < height; ++y) {
            const auto* source = rgb + scale * y * stride;
            auto* dest = gray.ptr<uint8_t>(y);
#pragma clang loop unroll(full)
            for (int x = 0; x < width; x += 32 / scale, source += 96) {
                const auto first = downsample_green(source, stride);
                const auto second = downsample_green(source + 48, stride);
                if constexpr (scale == 2) {
                    wasm_v128_store(dest + x, wasm_u8x16_narrow_i16x8(first, second));
                } else {
                    const auto shorts = wasm_i16x8_narrow_i32x4(first, second);
                    wasm_v128_store64_lane(dest + x, wasm_u8x16_narrow_i16x8(shorts, shorts), 0);
                }
            }
        }
    }

    static v128_t downsample_green(const uint8_t* source, int stride) {
        const auto top = wasm_i16x8_add(green_pairs(source), green_pairs(source + stride));
        if constexpr (scale == 2) {
            return wasm_u16x8_shr(wasm_i16x8_add(top, wasm_i16x8_splat(2)), 2);
        } else {
            const auto bottom = wasm_i16x8_add(green_pairs(source + 2 * stride), green_pairs(source + 3 * stride));
            const auto pairs = wasm_u32x4_extadd_pairwise_u16x8(wasm_i16x8_add(top, bottom));
            return wasm_u32x4_shr(wasm_i32x4_add(pairs, wasm_i32x4_splat(8)), 4);
        }
    }

    // Sum the green values of adjacent pairs in a group of 16 RGB pixels.
    static v128_t green_pairs(const uint8_t* rgb) {
        const auto first = wasm_v128_load(rgb);
        const auto middle = wasm_v128_load(rgb + 16);
        const auto last = wasm_v128_load(rgb + 32);
        const auto head = wasm_i8x16_shuffle(first, middle, 1, 4, 7, 10, 13, 16, 19, 22,
                                            25, 28, 31, 0, 0, 0, 0, 0);
        const auto green = wasm_i8x16_shuffle(head, last, 0, 1, 2, 3, 4, 5, 6, 7,
                                             8, 9, 10, 18, 21, 24, 27, 30);
        return wasm_u16x8_extadd_pairwise_u8x16(green);
    }

    Vec3d ray(const Point2f& p) const { return {(p.x - cx_) / fx_, (p.y - cy_) / fy_, 1.}; }

    Point2f pixel(const Vec3d& p) const {
        return {static_cast<float>(fx_ * p[0] / p[2] + cx_),
                static_cast<float>(fy_ * p[1] / p[2] + cy_)};
    }

    void prepare_anchor_patches() {
        const size_t count = std::min(points_.size(), cached_points);
        for (size_t i = 0; i < count; ++i) {
            fixed_lk::prepare_source<window, width>(anchor_pyramid_[0], anchor_pyramid_[1],
                points_[i], anchor_patches_[i][0]);
            if constexpr (levels == 1)
                fixed_lk::prepare_source<window, width / 2>(anchor_pyramid_[2], anchor_pyramid_[3],
                    points_[i] * .5f, anchor_patches_[i][1]);
        }
    }

    // Start one of the second batch's coarse estimates at capture.
    static bool pretrack_coarse(size_t rank) {
        return scale == 2 && rank == 5;
    }

    void track_forward(int batch_index, int* work = nullptr) {
        // Use spare capture time at 160x120 for forward tracks.
        const auto anchor_to_current = current_to_anchor_.t();
        constexpr float border = scale == 2 ? 6.f : 4.f;
        constexpr float min_parallax_squared = scale == 2 ? .16f : .0625f;
        static constexpr auto batches = make_tracking_batches<scale, true>();
        const auto& jobs = batches[batch_index];
        for (size_t job = 0; job < jobs.count; ++job) {
            const size_t i = jobs.ranks[job];
            if (i >= observations_.size()) break;
            const bool split_coarse = pretrack_coarse(i);
            const bool coarse_only = batch_index == 0 && split_coarse;
            const auto& previous = points_[i];
            const auto a = ray(previous);
            auto& observation = observations_[i];
            auto point = observation.after;
            if (!split_coarse || coarse_only) {
                const auto mapped = anchor_to_current * a;
                point = mapped[2] > .2 ? pixel(mapped) : previous;
            }
            if constexpr (levels == 1) {
                if (coarse_only) {
                    fixed_lk::refine_coarse_prepared<window, width / 2>(current_pyramid_[2], anchor_patches_[i][1], point, work);
                    observation.after = point;
                    continue;
                }
            }
            Vec3d b;
            const auto passes_geometry = [&](const Point2f& tracked) {
                if (!(tracked.x > border && tracked.x < width - border &&
                      tracked.y > border && tracked.y < height - border)) return false;
                b = current_to_anchor_ * ray(tracked);
                if (b[2] < .2) return false;
                b *= 1. / b[2];
                const double dx = (a[0] - b[0]) * fx_, dy = (a[1] - b[1]) * fy_;
                return dx * dx + dy * dy >= min_parallax_squared;
            };
            bool tracked;
            if constexpr (levels == 1) {
                tracked = fixed_lk::track_prepared<window, width, levels, true>(
                    current_pyramid_, anchor_patches_[i], point, passes_geometry, split_coarse, &sampled_forward_[i], work);
            } else {
                tracked = i < cached_points
                    ? fixed_lk::track_prepared<window, width, levels>(
                        current_pyramid_, anchor_patches_[i], point, passes_geometry)
                    : fixed_lk::track<window, width, levels>(
                        anchor_pyramid_, current_pyramid_, previous, point, passes_geometry);
            }
            if (!tracked) continue;
            const auto normal = a.cross(b);
            const double magnitude = length(normal);
            if (!(magnitude > 1e-8)) continue;

            observation.after = point;
            observation.match = {a, b, normal, normal * (1. / magnitude)};
            observation.forward_ok = true;
        }
    }

    void track_reverse(int batch_index) {
        // Backward tracking is needed only for observations that passed the
        // forward photometric, image-boundary and parallax checks.
        static constexpr auto batches = make_tracking_batches<scale, false>();
        const auto& jobs = batches[batch_index];
        for (size_t job = 0; job < jobs.count; ++job) {
            const size_t i = jobs.ranks[job];
            if (i >= observations_.size()) break;
            if (!observations_[i].forward_ok) continue;
            auto& observation = observations_[i];
            auto reverse = points_[i];
            const bool tracked = fixed_lk::track<window, width, levels, levels == 1>(
                current_pyramid_, anchor_pyramid_, observation.after, reverse, nullptr,
                levels == 1 ? &sampled_forward_[i] : nullptr);
            const auto back_error = reverse - points_[i];
            observation.reverse_ok = tracked && !(back_error.dot(back_error) > .49f);
        }
    }

    bool supports_previous_heading(const Match* const* matches, size_t count,
                                    const Vec3d& prior, double baseline) const {
        if (!previous_heading_valid_ || count < 6 || baseline <= .005 ||
            baseline < .5 * previous_baseline_ || baseline > 2. * previous_baseline_ ||
            prior.dot(previous_inertial_heading_) < .90 * baseline)
            return false;
        const double rotation_trace = current_to_anchor_(0, 0) + current_to_anchor_(1, 1)
                                    + current_to_anchor_(2, 2);
        if (rotation_trace < 2.90) return false;
        const auto direction = anchor_rotation_.t() * previous_visual_heading_;
        size_t supporting = 0;
        for (size_t k = 0; k < count; ++k) {
            const auto& match = *matches[k];
            if (std::abs(match.unit_normal.dot(direction)) < .30 &&
                direction.cross(match.current).dot(match.normal) > 0. &&
                direction.cross(match.anchor).dot(match.normal) > 0.)
                ++supporting;
        }
        return supporting >= 4 && 2 * supporting >= count;
    }

    void finish_observation() {
        std::array<const Match*, max_features> matches;
        size_t count = 0;
        for (const auto& observation : observations_)
            if (observation.reverse_ok) matches[count++] = &observation.match;
        Vec3d direction;
        const auto prior = capture_position_ - anchor_position_;
        const double baseline = length(prior);
        const bool sparse = VISUAL_SPARSE_DIRECTION && scale == 2 && count < 10;
        if (baseline > .005 && estimate_direction(matches.data(), count, sparse, direction)) {
            const Vec3d heading = anchor_rotation_ * direction;
            const double weight = sparse ? .85 : 1.;
            pending_ = (heading * baseline - prior) * (weight / VISUAL_INTERVAL);
            correction_frames_ = VISUAL_INTERVAL;
            previous_visual_heading_ = heading;
            previous_inertial_heading_ = prior * (1. / baseline);
            previous_baseline_ = baseline;
            previous_heading_valid_ = true;
        } else {
            if (supports_previous_heading(matches.data(), count, prior, baseline)) {
                pending_ = (previous_visual_heading_ * baseline - prior) * (.55 / VISUAL_INTERVAL);
                correction_frames_ = VISUAL_INTERVAL;
            }
            // Extrapolation can only follow a directly observed heading.
            previous_heading_valid_ = false;
        }
        std::swap(anchor_pyramid_, current_pyramid_);
        anchor_rotation_ = capture_rotation_;
        anchor_position_ = capture_position_;
    }
};
} // namespace visual_direction
