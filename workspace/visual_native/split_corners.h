#pragma once

#include "flow_pyramid.h"
#include <opencv2/imgproc.hpp>
#include <wasm_simd128.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

namespace visual_direction {

// Evaluate each half with a three-row ring of Sobel products. Using integer
// products avoids intermediate float scaling. The score's constant scale
// cancels when ranking corners and applying a relative response threshold.
template<int width, int height>
class SplitCorners {
public:
    void compute_half(const cv::Mat& image, int half) {
        response_.create(image.size(), CV_32FC1);
        const int middle = image.rows / 2;
        half_maximum_[half] = compute_rows(image, half == 0 ? 0 : middle, half == 0 ? middle : image.rows);
        // The last row in the top half needs a neighbor from the bottom half.
        if (half == 0) {
            candidates_.clear();
            if (half_maximum_[0] > 0.f)
                collect_peaks(1, width == 160 ? middle - 1 : scan_ahead(middle), threshold_for(half_maximum_[0]));
        } else {
            const float maximum = std::max(half_maximum_[0], half_maximum_[1]);
            if constexpr (width == 160) {
                if (maximum > 0.f)
                    collect_peaks(middle - 1, height - 1, threshold_for(maximum));
            } else if (half_maximum_[1] > 0.f) {
                collect_peaks(middle, middle + scan_ahead(middle), threshold_for(maximum));
            }
        }
    }

    template<int spacing>
    void select(std::vector<cv::Point2f>& points, int maximum) {
        points.clear();
        const float maximum_response = std::max(half_maximum_[0], half_maximum_[1]);
        if (maximum_response <= 0.f) return;
        const float threshold = threshold_for(maximum_response);

        if constexpr (width == 160) {
            select_binned<spacing>(points, maximum, threshold, maximum_response);
        } else {
            const int middle = response_.rows / 2;
            collect_peaks(scan_ahead(middle), middle, threshold);
            collect_peaks(middle + scan_ahead(middle), response_.rows - 1, threshold);
            select_prefix<spacing>(points, maximum, threshold);
        }
    }

private:
    struct Corner { float score; int x, y; };
    std::vector<Corner> candidates_;
    using Key = uint64_t;
    std::vector<Key> ordered_;

    static uint32_t score_bits(float score) {
        uint32_t bits;
        std::memcpy(&bits, &score, sizeof(bits));
        return bits;
    }

    template<int spacing>
    void select_binned(std::vector<cv::Point2f>& points, int maximum, float threshold, float maximum_response) {
        static_assert(width <= 256 && height <= 256 && spacing <= 16);
        // Positive float bits have the same order as the numeric scores.
        // Bins partition that order; the last holds any remaining low scores.
        constexpr int bin_count = 32;
        constexpr int bin_shift = 21;
        const auto largest_bin = score_bits(maximum_response) >> bin_shift;
        const auto min_bits = score_bits(threshold);
        const auto bucket = [&](uint32_t bits) {
            return std::min(largest_bin - (bits >> bin_shift), static_cast<uint32_t>(bin_count - 1));
        };
        std::array<size_t, bin_count> offsets{};
        for (const auto& corner : candidates_) {
            const auto bits = score_bits(corner.score);
            if (bits > min_bits) ++offsets[bucket(bits)];
        }
        size_t count = 0;
        for (auto& offset : offsets) {
            const size_t n = offset;
            offset = count;
            count += n;
        }
        ordered_.resize(count);
        for (const auto& corner : candidates_) {
            const auto bits = score_bits(corner.score);
            if (bits <= min_bits) continue;
            const auto key = (Key(bits) << 16) | (uint32_t(corner.y) << 8) | uint32_t(corner.x);
            ordered_[offsets[bucket(bits)]++] = key;
        }

        // Coordinates are offset by one bitset word so each disk can update
        // two adjacent words, even next to the left or right image edge.
        constexpr int word_bits = 32;
        constexpr int words = (width + word_bits - 1) / word_bits + 2;
        constexpr int radius = spacing - 1;
        using Row = std::array<uint32_t, words>;
        std::array<Row, height> blocked{};
        std::array<uint32_t, spacing> spans;
        for (int dy = 0; dy < spacing; ++dy) {
            const int span = int(std::sqrt(spacing * spacing - dy * dy - 1));
            spans[dy] = ((uint32_t(1) << (2 * span + 1)) - 1) << (radius - span);
        }
        points.reserve(maximum);
        auto first = ordered_.begin();
        for (const auto offset : offsets) {
            const auto last = ordered_.begin() + offset;
            // Low bits break ties by descending y and then descending x.
            std::sort(first, last, std::greater<Key>());
            for (; first != last; ++first) {
                const int x = *first & 255;
                const int y = (*first >> 8) & 255;
                if (blocked[y][1 + x / word_bits] & (uint32_t(1) << (x % word_bits))) continue;
                points.emplace_back(float(x), float(y));
                if (static_cast<int>(points.size()) >= maximum) return;
                const int left = x + word_bits - radius;
                const int shift = left % word_bits;
                const int word = left / word_bits;
                const int y0 = std::max(0, y - radius), y1 = std::min(response_.rows - 1, y + radius);
                for (int row = y0; row <= y1; ++row) {
                    const uint64_t mask = uint64_t(spans[std::abs(row - y)]) << shift;
                    blocked[row][word] |= uint32_t(mask);
                    blocked[row][word + 1] |= uint32_t(mask >> word_bits);
                }
            }
        }
    }

    template<int spacing>
    void select_prefix(std::vector<cv::Point2f>& points, int maximum, float threshold) {
        auto& candidates = candidates_;
        candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
            [threshold](const Corner& corner) { return corner.score <= threshold; }), candidates.end());
        const auto stronger = [](const Corner& a, const Corner& b) {
            if (a.score != b.score) return a.score > b.score;
            if (a.y != b.y) return a.y > b.y;
            return a.x > b.x;
        };
        // Usually the strongest candidates fill the track list. Only order
        // the rest if the spacing constraint rejects too many of the first group.
        const size_t prefix_size = std::min(candidates.size(), static_cast<size_t>(4 * maximum));
        const auto prefix_end = candidates.begin() + prefix_size;
        if (prefix_end != candidates.end())
            std::nth_element(candidates.begin(), prefix_end, candidates.end(), stronger);
        std::sort(candidates.begin(), prefix_end, stronger);

        blocked_.create(response_.size(), CV_8UC1);
        blocked_.setTo(0);
        points.reserve(maximum);
        const int distance_squared = spacing * spacing;
        std::array<int, spacing> half_span;
        for (int dy = 0; dy < spacing; ++dy)
            half_span[dy] = int(std::sqrt(distance_squared - dy * dy - 1));
        for (auto candidate = candidates.begin(); candidate != candidates.end(); ++candidate) {
            if (candidate == prefix_end) std::sort(prefix_end, candidates.end(), stronger);
            const auto& corner = *candidate;
            if (blocked_.ptr<uint8_t>(corner.y)[corner.x]) continue;
            points.emplace_back(static_cast<float>(corner.x), static_cast<float>(corner.y));
            if (static_cast<int>(points.size()) >= maximum) break;
            const int y0 = std::max(0, corner.y - spacing + 1);
            const int y1 = std::min(response_.rows, corner.y + spacing);
            for (int y = y0; y < y1; ++y) {
                auto* blocked = blocked_.ptr<uint8_t>(y);
                const int span = half_span[std::abs(y - corner.y)];
                const int x0 = std::max(0, corner.x - span);
                const int x1 = std::min(response_.cols, corner.x + span + 1);
                std::memset(blocked + x0, 1, x1 - x0);
            }
        }
    }

    // Scan most of each half during Sobel scoring to balance the selection frame.
    static constexpr int scan_ahead(int rows) { return 2 * rows / 3; }

    static float threshold_for(double response) { return static_cast<float>(response * .018); }

    void collect_peaks(int begin, int end, float threshold) {
        const auto minimum = wasm_f32x4_splat(threshold);
        for (int y = begin; y < end; ++y) {
            const auto* top = response_.ptr<float>(y - 1);
            const auto* response = response_.ptr<float>(y);
            const auto* bottom = response_.ptr<float>(y + 1);
            int x = 1;
            for (; x + 4 < width; x += 4) {
                const auto score = wasm_v128_load(response + x);
                const auto eligible = wasm_f32x4_gt(score, minimum);
                if (wasm_i32x4_bitmask(eligible) == 0) continue;
                const auto top_max = neighborhood_max(top + x);
                const auto bottom_max = neighborhood_max(bottom + x);
                const auto sides = wasm_f32x4_max(wasm_v128_load(response + x - 1),
                                                  wasm_v128_load(response + x + 1));
                const auto neighbors = wasm_f32x4_max(wasm_f32x4_max(top_max, bottom_max), sides);
                unsigned mask = wasm_i32x4_bitmask(wasm_v128_and(eligible, wasm_f32x4_ge(score, neighbors)));
                while (mask != 0) {
                    const int col = x + __builtin_ctz(mask);
                    candidates_.push_back({response[col], col, y});
                    mask &= mask - 1;
                }
            }
            for (; x + 1 < width; ++x) {
                const float neighbors = std::max({top[x - 1], top[x], top[x + 1], response[x - 1],
                    response[x + 1], bottom[x - 1], bottom[x], bottom[x + 1]});
                if (response[x] > threshold && response[x] >= neighbors)
                    candidates_.push_back({response[x], x, y});
            }
        }
    }

    static v128_t neighborhood_max(const float* row) {
        return wasm_f32x4_max(wasm_f32x4_max(wasm_v128_load(row - 1), wasm_v128_load(row)),
                             wasm_v128_load(row + 1));
    }

    struct ProductBlock { v128_t xx, xy, yy; };
    using Products = std::array<ProductBlock, width / 4>;

    template<bool Upper>
    static ProductBlock products(v128_t dx, v128_t dy) {
        if constexpr (Upper)
            return {wasm_i32x4_extmul_high_i16x8(dx, dx), wasm_i32x4_extmul_high_i16x8(dx, dy),
                    wasm_i32x4_extmul_high_i16x8(dy, dy)};
        else
            return {wasm_i32x4_extmul_low_i16x8(dx, dx), wasm_i32x4_extmul_low_i16x8(dx, dy),
                    wasm_i32x4_extmul_low_i16x8(dy, dy)};
    }

    template<bool First = false, bool Last = false>
    static ProductBlock horizontal_sum(const ProductBlock& left, const ProductBlock& center,
                                       const ProductBlock& right) {
        // Reflect the two edge taps without materializing an extended row.
        const auto sum = [](v128_t l, v128_t c, v128_t r) {
            const auto before = First ? wasm_i32x4_shuffle(c, c, 1, 0, 1, 2)
                                      : wasm_i32x4_shuffle(l, c, 3, 4, 5, 6);
            const auto after = Last ? wasm_i32x4_shuffle(c, c, 1, 2, 3, 2)
                                     : wasm_i32x4_shuffle(c, r, 1, 2, 3, 4);
            return wasm_i32x4_add(wasm_i32x4_add(before, c), after);
        };
        return {sum(left.xx, center.xx, right.xx), sum(left.xy, center.xy, right.xy),
                sum(left.yy, center.yy, right.yy)};
    }

    static void sobel_products(const cv::Mat& image, int y, Products& row) {
        const auto* centre = image.ptr<uint8_t>(y);
        constexpr int stride = width + 2 * horizontal_padding;
        DerivativeColumns<1, 2> left(centre - 1, stride), columns(centre + 7, stride);
        const auto [dx0, dy0] = left.finish(columns);
        auto previous = products<false>(dx0, dy0);
        auto current = products<true>(dx0, dy0);
        row[0] = horizontal_sum<true>(previous, previous, current);
#pragma clang loop unroll(full)
        for (int i = 2; i < width / 4; i += 2) {
            const DerivativeColumns<1, 2> right(centre + 4 * i + 7, stride);
            const auto [dx, dy] = columns.finish(right);
            const auto low = products<false>(dx, dy);
            const auto high = products<true>(dx, dy);
            row[i - 1] = horizontal_sum(previous, current, low);
            row[i] = horizontal_sum(current, low, high);
            previous = low;
            current = high;
            columns = right;
        }
        row.back() = horizontal_sum<false, true>(previous, current, current);
    }

    static v128_t vertical_sum(v128_t top, v128_t center, v128_t bottom) {
        return wasm_f32x4_convert_i32x4(wasm_i32x4_add(wasm_i32x4_add(top, center), bottom));
    }

    float compute_rows(const cv::Mat& image, int begin, int end) {
        std::array<Products, 3> products;
        Products* top = &products[0];
        Products* centre = &products[1];
        Products* bottom = &products[2];
        sobel_products(image, begin == 0 ? 1 : begin - 1, *top);
        sobel_products(image, begin, *centre);
        auto maximum = wasm_f32x4_splat(0.f);
        const auto half = wasm_f32x4_splat(.5f);
        for (int y = begin; y < end; ++y) {
            sobel_products(image, y == image.rows - 1 ? y - 1 : y + 1, *bottom);
            float* response = response_.ptr<float>(y);
#pragma clang loop unroll(full)
            for (int i = 0; i < width / 4; ++i) {
                const auto& t = (*top)[i];
                const auto& m = (*centre)[i];
                const auto& btm = (*bottom)[i];
                const auto a = wasm_f32x4_mul(half, vertical_sum(t.xx, m.xx, btm.xx));
                const auto b = vertical_sum(t.xy, m.xy, btm.xy);
                const auto c = wasm_f32x4_mul(half, vertical_sum(t.yy, m.yy, btm.yy));
                const auto diff = wasm_f32x4_sub(a, c);
                const auto discriminant = wasm_f32x4_add(wasm_f32x4_mul(diff, diff), wasm_f32x4_mul(b, b));
                const auto score = wasm_f32x4_sub(wasm_f32x4_add(a, c), wasm_f32x4_sqrt(discriminant));
                wasm_v128_store(response + 4 * i, score);
                maximum = wasm_f32x4_max(maximum, score);
            }
            auto* spare = top;
            top = centre;
            centre = bottom;
            bottom = spare;
        }
        return std::max({wasm_f32x4_extract_lane(maximum, 0), wasm_f32x4_extract_lane(maximum, 1),
                         wasm_f32x4_extract_lane(maximum, 2), wasm_f32x4_extract_lane(maximum, 3)});
    }

    cv::Mat response_, blocked_;
    std::array<float, 2> half_maximum_{};
};

} // namespace visual_direction
