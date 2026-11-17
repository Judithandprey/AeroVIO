/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
// Single-channel, fixed-window adaptation of OpenCV 4.12's LKTrackerInvoker.
#pragma once
#include "flow_pyramid.h"
#include <opencv2/core.hpp>
#include <wasm_simd128.h>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace fixed_lk {
constexpr int weight_bits = 14;
constexpr int intensity_shift = weight_bits - 5;
constexpr float gradient_scale = 1.f / (1 << 20);
constexpr float photometric_error_limit = 24.f;

struct Bilinear {
    v128_t top, bottom;

    explicit Bilinear(v128_t fraction) {
        const auto complement = wasm_f32x4_sub(wasm_f32x4_splat(1.f), fraction);
        const auto horizontal = wasm_i32x4_shuffle(complement, fraction, 0, 4, 0, 4);
        const auto vertical = wasm_i32x4_shuffle(complement, fraction, 1, 1, 5, 5);
        const auto scaled = wasm_f32x4_mul(wasm_f32x4_mul(horizontal, vertical),
                                          wasm_f32x4_splat(1 << weight_bits));
        const auto rounded = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_nearest(scaled));
        // The first three rounded weights and their remainder all fit in
        // signed 16-bit lanes. Interleave their low bytes directly.
        const auto pairs = wasm_i32x4_add(rounded, wasm_i32x4_shuffle(rounded, rounded, 1, 0, 0, 0));
        const auto sum = wasm_i32x4_add(pairs, wasm_i32x4_shuffle(rounded, rounded, 2, 2, 2, 2));
        const auto remainder = wasm_i32x4_sub(wasm_i32x4_splat(1 << weight_bits), sum);
        top = wasm_i8x16_shuffle(rounded, rounded, 0, 1, 4, 5, 0, 1, 4, 5, 0, 1, 4, 5, 0, 1, 4, 5);
        bottom = wasm_i8x16_shuffle(rounded, remainder, 8, 9, 16, 17, 8, 9, 16, 17, 8, 9, 16, 17, 8, 9, 16, 17);
    }

    template<int Shift>
    v128_t sample_pairs(v128_t top_pixels, v128_t bottom_pixels,
                        v128_t bias = wasm_i32x4_splat(1 << (Shift - 1))) const {
        const auto weighted = wasm_i32x4_add(wasm_i32x4_dot_i16x8(top_pixels, top),
                                             wasm_i32x4_dot_i16x8(bottom_pixels, bottom));
        return wasm_i32x4_shr(wasm_i32x4_add(weighted, bias), Shift);
    }

    template<int Offset>
    static v128_t adjacent_pixels(v128_t row) {
        const auto zero = wasm_i32x4_splat(0);
        return wasm_i8x16_shuffle(row, zero, Offset, 16, Offset + 1, 16,
            Offset + 1, 16, Offset + 2, 16, Offset + 2, 16, Offset + 3, 16, Offset + 3, 16, Offset + 4, 16);
    }
};

struct GradientSamples {
    v128_t low, high, tail{};
};

template<int Window>
struct AxisPairs {
    v128_t low, high, tail{};

    AxisPairs(v128_t first, v128_t second) {
        low = wasm_i16x8_shuffle(first, first, 0, 1, 1, 2, 2, 3, 3, 4);
        high = wasm_i16x8_shuffle(first, second, 4, 5, 5, 6, 6, 7, 7, 8);
        if constexpr (Window == 13)
            tail = wasm_i16x8_shuffle(second, second, 0, 1, 1, 2, 2, 3, 3, 4);
    }

    GradientSamples interpolate(const Bilinear& w, const AxisPairs& bottom) const {
        GradientSamples out;
        out.low = w.sample_pairs<weight_bits>(low, bottom.low);
        out.high = w.sample_pairs<weight_bits>(high, bottom.high);
        if constexpr (Window == 13) out.tail = w.sample_pairs<weight_bits>(tail, bottom.tail);
        return out;
    }
};

template<int Window, bool IntegerOrigin>
struct GradientRow {
    AxisPairs<Window> horizontal, vertical;
    v128_t edge;

    GradientRow(const int16_t* dx, const int16_t* dy)
        : GradientRow(wasm_v128_load(dx), wasm_v128_load(dx + 8),
                      wasm_v128_load(dy), wasm_v128_load(dy + 8)) {}

private:
    static constexpr int last = Window - 9;

    GradientRow(v128_t dx_first, v128_t dx_second, v128_t dy_first, v128_t dy_second)
        : horizontal(dx_first, dx_second), vertical(dy_first, dy_second),
          edge(wasm_i16x8_shuffle(dx_second, dy_second,
              last, last + 1, 8 + last, 9 + last, last, last + 1, 8 + last, 9 + last)) {}
};

template<int Window>
struct GradientRow<Window, true> {
    GradientRow(const int16_t*, const int16_t*) {}
};

template<int Window>
GradientSamples integer_samples(const int16_t* values) {
    GradientSamples out;
    const auto first = wasm_v128_load(values);
    out.low = wasm_i32x4_extend_low_i16x8(first);
    out.high = wasm_i32x4_extend_high_i16x8(first);
    if constexpr (Window == 13) out.tail = wasm_i32x4_extend_low_i16x8(wasm_v128_load(values + 8));
    return out;
}

template<int Window>
struct PixelPairs {
    v128_t low, high, tail{}, edge;

    explicit PixelPairs(v128_t row)
        : low(Bilinear::adjacent_pixels<0>(row)), high(Bilinear::adjacent_pixels<4>(row)) {
        if constexpr (Window == 13) tail = Bilinear::adjacent_pixels<8>(row);
        constexpr int last = Window - 1;
        const auto zero = wasm_i32x4_splat(0);
        edge = wasm_i8x16_shuffle(row, zero, last, 16, last + 1, 16,
            last, 16, last + 1, 16, last, 16, last + 1, 16, last, 16, last + 1, 16);
    }

    v128_t interpolate8(const Bilinear& weights, const PixelPairs& bottom) const {
        const auto first = weights.sample_pairs<intensity_shift>(low, bottom.low);
        const auto second = weights.sample_pairs<intensity_shift>(high, bottom.high);
        return wasm_i16x8_narrow_i32x4(first, second);
    }
};

struct PatchOrigin {
    cv::Point2i pixel;
    Bilinear weights;

    explicit PatchOrigin(cv::Point2f point)
        : PatchOrigin(wasm_f32x4_make(point.x, point.y, 0.f, 0.f)) {}

    explicit PatchOrigin(v128_t coordinates)
        : PatchOrigin(coordinates, wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_floor(coordinates))) {}

private:
    PatchOrigin(v128_t coordinates, v128_t indices)
        : pixel(wasm_i32x4_extract_lane(indices, 0), wasm_i32x4_extract_lane(indices, 1)),
          weights(wasm_f32x4_sub(coordinates, wasm_f32x4_convert_i32x4(indices))) {}
};

template<int Window, int Width>
bool inside_padded_image(cv::Point2f point) {
    // Both native pyramid sizes retain the input camera's 4:3 aspect ratio.
    constexpr int height = Width * 3 / 4;
    return point.x >= -Window && point.x < Width && point.y >= -Window && point.y < height;
}

inline float reduce_sum(v128_t values) {
    return (wasm_f32x4_extract_lane(values, 0) + wasm_f32x4_extract_lane(values, 2)) +
           (wasm_f32x4_extract_lane(values, 1) + wasm_f32x4_extract_lane(values, 3));
}

// Accumulate the three scalar covariance terms in parallel. Every lane
// still adds one pixel at a time in the reference order.
inline v128_t accumulate_tail(v128_t sum, v128_t dx, v128_t dy) {
    const auto xx = wasm_i32x4_mul(dx, dx);
    const auto xy = wasm_i32x4_mul(dx, dy);
    const auto yy = wasm_i32x4_mul(dy, dy);
    const auto low = wasm_i32x4_shuffle(xx, xy, 0, 4, 1, 5);
    const auto high = wasm_i32x4_shuffle(xx, xy, 2, 6, 3, 7);
    sum = wasm_f32x4_add(sum, wasm_f32x4_convert_i32x4(wasm_i32x4_shuffle(low, yy, 0, 1, 4, 4)));
    sum = wasm_f32x4_add(sum, wasm_f32x4_convert_i32x4(wasm_i32x4_shuffle(low, yy, 2, 3, 5, 5)));
    sum = wasm_f32x4_add(sum, wasm_f32x4_convert_i32x4(wasm_i32x4_shuffle(high, yy, 0, 1, 6, 6)));
    return wasm_f32x4_add(sum, wasm_f32x4_convert_i32x4(wasm_i32x4_shuffle(high, yy, 2, 3, 7, 7)));
}

struct Covariance {
    float xx = 0.f, xy = 0.f, yy = 0.f;
};

// Moving the integer reference subtraction before the shift preserves rounding.
inline v128_t difference_bias(v128_t reference) {
    return wasm_i32x4_sub(wasm_i32x4_splat(1 << (intensity_shift - 1)),
                         wasm_i32x4_shl(reference, intensity_shift));
}

// Interpolated forward samples at the center used as the reverse source.
struct IntensityRow {
    v128_t pixels, tail_bias;
};

template<int Window>
struct IntensityPatch {
    std::array<IntensityRow, Window> rows;
    // Contiguous last-column biases let the photometric check store four at once.
    std::array<int, Window> edge_bias;
};

struct PatchRow {
    v128_t pixels;
    v128_t gradient_pairs_low, gradient_pairs_high;
    v128_t tail_bias, tail_gradients;
    // Lanes 0 and 1 hold the last column's gradients. Lane 2 holds its intensity bias.
    v128_t edge_sample;
};

// Cache the bias and gradients for two last-column pixels together.
struct PatchEdgePair {
    v128_t bias, gradients;

    PatchEdgePair() = default;
    PatchEdgePair(v128_t first, v128_t second)
        : bias(wasm_i32x4_shuffle(first, second, 2, 2, 6, 6)),
          gradients(wasm_i32x4_shuffle(first, second, 0, 1, 4, 5)) {}
};

template<int Window>
struct SourcePatch {
    std::array<PatchRow, Window> rows;
    // Cache last-column pairs for the 13x13 windows used at 160x120.
    static constexpr bool cache_edges = Window == 13;
    std::array<PatchEdgePair, cache_edges ? Window / 2 : 0> edge_pairs;
    Covariance matrix;
    float inverse = 0.f;

    PatchEdgePair edge_pair(int y) const {
        if constexpr (cache_edges) {
            return edge_pairs[y / 2];
        } else {
            return {rows[y].edge_sample, rows[y + 1].edge_sample};
        }
    }
};

inline void accumulate_covariance(v128_t gx, v128_t gy, v128_t& xx, v128_t& xy, v128_t& yy) {
    const auto dx = wasm_f32x4_convert_i32x4(gx);
    const auto dy = wasm_f32x4_convert_i32x4(gy);
    xx = wasm_f32x4_add(xx, wasm_f32x4_mul(dx, dx));
    xy = wasm_f32x4_add(xy, wasm_f32x4_mul(dx, dy));
    yy = wasm_f32x4_add(yy, wasm_f32x4_mul(dy, dy));
}

template<int Window, int Width, bool IntegerOrigin, bool ReuseIntensity>
Covariance prepare_patch(const cv::Mat& image, const cv::Mat& derivative,
                         const PatchOrigin& origin, SourcePatch<Window>& patch,
                         const IntensityPatch<Window>* samples) {
    // The tracker allocates both image and gradient planes with this padding.
    constexpr int image_stride = Width + 2 * visual_direction::horizontal_padding;
    constexpr int gradient_stride = 2 * image_stride;
    const auto* pixels = image.data + origin.pixel.y * image_stride + origin.pixel.x;
    auto* dx_row = reinterpret_cast<const int16_t*>(derivative.data)
                 + origin.pixel.y * gradient_stride + origin.pixel.x;
    // allocate_gradient places the two padded planes side by side in each row.
    const auto* dy_row = dx_row + gradient_stride / 2;
    auto xx = wasm_f32x4_splat(0.f), xy = xx, yy = xx;
    auto edge = wasm_f32x4_splat(0.f);
    auto previous_edge = wasm_i32x4_splat(0);
    const auto& w = origin.weights;
    auto top_row = wasm_v128_load(pixels);
    // The lower taps of this patch row become the upper taps of the next row.
    PixelPairs<Window> top_pixels(top_row);
    GradientRow<Window, IntegerOrigin> top_gradients(dx_row, dy_row);
#pragma clang loop unroll(full)
    for (int y = 0; y < Window; ++y) {
        auto& row = patch.rows[y];
        const auto bottom_row = wasm_v128_load(pixels + image_stride);
        const PixelPairs<Window> bottom_pixels(bottom_row);
        const GradientRow<Window, IntegerOrigin> bottom_gradients(dx_row + gradient_stride,
                                                                dy_row + gradient_stride);
        if constexpr (ReuseIntensity)
            row.pixels = samples->rows[y].pixels;
        else if constexpr (IntegerOrigin)
            row.pixels = wasm_i16x8_shl(wasm_u16x8_extend_low_u8x16(top_row), 5);
        else
            row.pixels = top_pixels.interpolate8(w, bottom_pixels);
        GradientSamples dx, dy;
        if constexpr (IntegerOrigin) {
            dx = integer_samples<Window>(dx_row);
            dy = integer_samples<Window>(dy_row);
        } else {
            dx = top_gradients.horizontal.interpolate(w, bottom_gradients.horizontal);
            dy = top_gradients.vertical.interpolate(w, bottom_gradients.vertical);
        }
        const auto gx = wasm_i16x8_narrow_i32x4(dx.low, dx.high);
        const auto gy = wasm_i16x8_narrow_i32x4(dy.low, dy.high);
        row.gradient_pairs_low = wasm_i16x8_shuffle(gx, gy, 0, 4, 8, 12, 1, 5, 9, 13);
        row.gradient_pairs_high = wasm_i16x8_shuffle(gx, gy, 2, 6, 10, 14, 3, 7, 11, 15);
        accumulate_covariance(dx.low, dy.low, xx, xy, yy);
        accumulate_covariance(dx.high, dy.high, xx, xy, yy);
        if constexpr (Window == 13) {
            if constexpr (ReuseIntensity) {
                row.tail_bias = samples->rows[y].tail_bias;
            } else {
                if constexpr (IntegerOrigin) {
                    const auto tail_bytes = wasm_i32x4_shuffle(top_row, top_row, 2, 3, 2, 3);
                    row.tail_bias = wasm_i32x4_shl(wasm_u32x4_extend_low_u16x8(wasm_u16x8_extend_low_u8x16(tail_bytes)), 5);
                } else {
                    row.tail_bias = w.sample_pairs<intensity_shift>(top_pixels.tail, bottom_pixels.tail);
                }
                row.tail_bias = difference_bias(row.tail_bias);
            }
            // Each interpolated Scharr gradient fits in a signed 16-bit lane.
            row.tail_gradients = wasm_i16x8_shuffle(dx.tail, dy.tail, 0, 8, 2, 10, 4, 12, 6, 14);
            edge = accumulate_tail(edge, dx.tail, dy.tail);
        }
        constexpr int last = Window - 1;
        int last_pixel = 0;
        if constexpr (IntegerOrigin) {
            row.edge_sample = wasm_i32x4_make(dx_row[last], dy_row[last], 0, 0);
            if constexpr (!ReuseIntensity)
                last_pixel = wasm_u8x16_extract_lane(top_row, last) << 5;
        } else {
            row.edge_sample = w.sample_pairs<weight_bits>(top_gradients.edge, bottom_gradients.edge);
            if constexpr (!ReuseIntensity) {
                const auto intensity = w.sample_pairs<intensity_shift>(top_pixels.edge, bottom_pixels.edge);
                last_pixel = wasm_i32x4_extract_lane(intensity, 0);
            }
        }
        const int edge_bias = ReuseIntensity ? samples->edge_bias[y]
                            : (1 << (intensity_shift - 1)) - last_pixel * (1 << intensity_shift);
        row.edge_sample = wasm_i32x4_replace_lane(row.edge_sample, 2, edge_bias);
        if constexpr (SourcePatch<Window>::cache_edges) {
            if (y % 2 == 1)
                patch.edge_pairs[y / 2] = {previous_edge, row.edge_sample};
            previous_edge = row.edge_sample;
        }
        const auto left = wasm_i32x4_shuffle(row.edge_sample, row.edge_sample, 0, 0, 1, 1);
        const auto right = wasm_i32x4_shuffle(row.edge_sample, row.edge_sample, 0, 1, 1, 1);
        edge = wasm_f32x4_add(edge, wasm_f32x4_convert_i32x4(wasm_i32x4_mul(left, right)));
        pixels += image_stride;
        dx_row += gradient_stride;
        dy_row += gradient_stride;
        top_row = bottom_row;
        if constexpr (!IntegerOrigin) {
            top_pixels = bottom_pixels;
            top_gradients = bottom_gradients;
        }
    }
    return {(wasm_f32x4_extract_lane(edge, 0) + reduce_sum(xx)) * gradient_scale,
            (wasm_f32x4_extract_lane(edge, 1) + reduce_sum(xy)) * gradient_scale,
            (wasm_f32x4_extract_lane(edge, 2) + reduce_sum(yy)) * gradient_scale};
}

inline void accumulate_inner_residual(v128_t diff, const PatchRow& row, v128_t& sum_low, v128_t& sum_high) {
    const auto low = wasm_i16x8_shuffle(diff, diff, 0, 4, 0, 4, 1, 5, 1, 5);
    const auto high = wasm_i16x8_shuffle(diff, diff, 2, 6, 2, 6, 3, 7, 3, 7);
    sum_low = wasm_f32x4_add(sum_low, wasm_f32x4_convert_i32x4(wasm_i32x4_dot_i16x8(low, row.gradient_pairs_low)));
    sum_high = wasm_f32x4_add(sum_high, wasm_f32x4_convert_i32x4(wasm_i32x4_dot_i16x8(high, row.gradient_pairs_high)));
}

inline v128_t accumulate_tail_residual(v128_t sum, v128_t error, v128_t gradients) {
    // The fixed-point errors fit in signed 16-bit lanes. Keep the running
    // x and y sums in the first two lanes and add each pixel in order.
    const auto duplicated = wasm_i16x8_shuffle(error, error, 0, 0, 2, 2, 4, 4, 6, 6);
    const auto first = wasm_f32x4_convert_i32x4(wasm_i32x4_extmul_low_i16x8(duplicated, gradients));
    const auto second = wasm_f32x4_convert_i32x4(wasm_i32x4_extmul_high_i16x8(duplicated, gradients));
    sum = wasm_f32x4_add(sum, first);
    sum = wasm_f32x4_add(sum, wasm_i32x4_shuffle(first, first, 2, 3, 2, 3));
    sum = wasm_f32x4_add(sum, second);
    return wasm_f32x4_add(sum, wasm_i32x4_shuffle(second, second, 2, 3, 2, 3));
}

inline v128_t finish_residual(v128_t sum_low, v128_t sum_high, v128_t tail_sum) {
    const auto sum = wasm_f32x4_add(sum_low, sum_high);
    const auto pair = wasm_f32x4_add(sum, wasm_i32x4_shuffle(sum, sum, 2, 3, 0, 1));
    return wasm_f32x4_mul(wasm_f32x4_add(tail_sum, pair), wasm_f32x4_splat(gradient_scale));
}

template<int Window, int Width>
v128_t residual(const cv::Mat& image, const PatchOrigin& origin,
                     const SourcePatch<Window>& source) {
    static_assert(Window % 2 == 1);
    const auto& patch = source.rows;
    constexpr int stride = Width + 2 * visual_direction::horizontal_padding;
    const auto* pixels = image.data + origin.pixel.y * stride + origin.pixel.x;
    auto sum_low = wasm_f32x4_splat(0.f), sum_high = sum_low;
    // Lanes 0 and 1 carry the reference x and y scalar sums in parallel.
    auto tail_sum = wasm_f32x4_splat(0.f);
    const auto& w = origin.weights;
    const auto accumulate_pixels = [&](const PixelPairs<Window>& top,
                                       const PixelPairs<Window>& bottom, const PatchRow& row) {
        const auto diff = wasm_i16x8_sub(top.interpolate8(w, bottom), row.pixels);
        accumulate_inner_residual(diff, row, sum_low, sum_high);
        if constexpr (Window == 13) {
            const auto error = w.sample_pairs<intensity_shift>(top.tail, bottom.tail, row.tail_bias);
            tail_sum = accumulate_tail_residual(tail_sum, error, row.tail_gradients);
        }
    };
    const auto two_edges = [&](const PixelPairs<Window>& top, const PixelPairs<Window>& middle,
                               const PixelPairs<Window>& bottom, const PatchEdgePair& edge) {
        const auto upper = wasm_i32x4_shuffle(top.edge, middle.edge, 0, 0, 4, 4);
        const auto lower = wasm_i32x4_shuffle(middle.edge, bottom.edge, 0, 0, 4, 4);
        const auto difference = w.sample_pairs<intensity_shift>(upper, lower, edge.bias);
        return wasm_f32x4_convert_i32x4(wasm_i32x4_mul(difference, edge.gradients));
    };
    PixelPairs<Window> top(wasm_v128_load(pixels));
#pragma clang loop unroll(full)
    for (int y = 0; y + 1 < Window; y += 2) {
        const PixelPairs<Window> middle(wasm_v128_load(pixels + (y + 1) * stride));
        const PixelPairs<Window> bottom(wasm_v128_load(pixels + (y + 2) * stride));
        const auto edge_products = two_edges(top, middle, bottom, source.edge_pair(y));
        // Add each edge product after its own row's other products.
        accumulate_pixels(top, middle, patch[y]);
        tail_sum = wasm_f32x4_add(tail_sum, edge_products);
        accumulate_pixels(middle, bottom, patch[y + 1]);
        tail_sum = wasm_f32x4_add(tail_sum, wasm_i32x4_shuffle(edge_products, edge_products, 2, 3, 2, 3));
        top = bottom;
    }
    const PixelPairs<Window> bottom(wasm_v128_load(pixels + Window * stride));
    const auto& last = patch.back();
    accumulate_pixels(top, bottom, last);
    const auto last_bias = wasm_i32x4_shuffle(last.edge_sample, last.edge_sample, 2, 2, 2, 2);
    const auto last_error = w.sample_pairs<intensity_shift>(top.edge, bottom.edge, last_bias);
    tail_sum = wasm_f32x4_add(tail_sum, wasm_f32x4_convert_i32x4(wasm_i32x4_mul(last_error, last.edge_sample)));
    // Keep the original reduction order in the first two lanes, x and y.
    return finish_residual(sum_low, sum_high, tail_sum);
}

template<bool YHalf, int Window, int Width>
v128_t residual_grid(const cv::Mat& image, cv::Point2i origin, bool x_half,
                     const std::array<PatchRow, Window>& patch) {
    constexpr int stride = Width + 2 * visual_direction::horizontal_padding;
    const auto left_x = x_half ? wasm_i8x16_make(0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8)
                               : wasm_i8x16_make(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    const auto right_x = wasm_i8x16_add(left_x, wasm_i8x16_splat(8));
    const auto* pixels = image.data + origin.y * stride + origin.x;
    const auto horizontal_sum = [](v128_t row, v128_t indices) {
        return wasm_u16x8_extadd_pairwise_u8x16(wasm_i8x16_swizzle(row, indices));
    };
    const auto raw = wasm_v128_load(pixels);
    auto top_left = horizontal_sum(raw, left_x), top_right = horizontal_sum(raw, right_x);
    auto sum_low = wasm_f32x4_splat(0.f), sum_high = sum_low, tail_sum = sum_low;
#pragma clang loop unroll(full)
    for (const auto& row : patch) {
        pixels += stride;
        const auto bottom = wasm_v128_load(pixels);
        const auto lower_left = horizontal_sum(bottom, left_x), lower_right = horizontal_sum(bottom, right_x);
        // Repeat pixels for integer axes. Dividing the four-tap integer sum
        // by four and scaling by 32 is exactly a left shift by three.
        const auto left = wasm_i16x8_add(top_left, YHalf ? lower_left : top_left);
        const auto right = wasm_i16x8_add(top_right, YHalf ? lower_right : top_right);
        const auto diff = wasm_i16x8_sub(wasm_i16x8_shl(left, 3), row.pixels);
        accumulate_inner_residual(diff, row, sum_low, sum_high);
        if constexpr (Window == 13) {
            const auto tail = wasm_i32x4_shr(wasm_i32x4_add(
                wasm_i32x4_shl(wasm_i32x4_extend_low_i16x8(right), 3 + intensity_shift), row.tail_bias), intensity_shift);
            tail_sum = accumulate_tail_residual(tail_sum, tail, row.tail_gradients);
        }
        const int edge_input = wasm_i16x8_extract_lane(right, Window - 9);
        const int edge_error = ((edge_input << (3 + intensity_shift)) + wasm_i32x4_extract_lane(row.edge_sample, 2))
                              >> intensity_shift;
        tail_sum = wasm_f32x4_add(tail_sum, wasm_f32x4_convert_i32x4(
            wasm_i32x4_mul(wasm_i32x4_splat(edge_error), row.edge_sample)));
        top_left = lower_left;
        top_right = lower_right;
    }
    return finish_residual(sum_low, sum_high, tail_sum);
}

template<int Window, int Width>
v128_t initial_reverse_residual(const cv::Mat& image, v128_t estimate,
                                const SourcePatch<Window>& patch) {
    const auto doubled = wasm_f32x4_add(estimate, estimate);
    const auto on_grid = wasm_f32x4_eq(doubled, wasm_f32x4_floor(doubled));
    if ((wasm_i32x4_bitmask(on_grid) & 3) != 3)
        return residual<Window, Width>(image, PatchOrigin(estimate), patch);
    const auto floor = wasm_f32x4_floor(estimate);
    const auto indices = wasm_i32x4_trunc_sat_f32x4(floor);
    const auto mask = wasm_i32x4_bitmask(wasm_f32x4_ne(estimate, floor));
    const cv::Point2i origin(wasm_i32x4_extract_lane(indices, 0), wasm_i32x4_extract_lane(indices, 1));
    return mask & 2 ? residual_grid<true, Window, Width>(image, origin, mask & 1, patch.rows)
                    : residual_grid<false, Window, Width>(image, origin, mask & 1, patch.rows);
}

template<int Window, int Width, bool SaveIntensity>
float mean_error(const cv::Mat& image, const PatchOrigin& origin,
                 const SourcePatch<Window>& source, IntensityPatch<Window>* samples) {
    const auto& patch = source.rows;
    constexpr int stride = Width + 2 * visual_direction::horizontal_padding;
    const auto* pixels = image.data + origin.pixel.y * stride + origin.pixel.x;
    auto vector_sum = wasm_i32x4_splat(0);
    auto edge_sum = vector_sum;
    const auto& w = origin.weights;
    const auto row_at = [&](int y) { return PixelPairs<Window>(wasm_v128_load(pixels + y * stride)); };
    const auto add_inner = [&](int y, const PixelPairs<Window>& top, const PixelPairs<Window>& bottom) {
        const auto& row = patch[y];
        const auto intensity = top.interpolate8(w, bottom);
        const auto diff = wasm_i16x8_abs(wasm_i16x8_sub(intensity, row.pixels));
        vector_sum = wasm_i32x4_add(vector_sum, wasm_u32x4_extadd_pairwise_u16x8(diff));
        if constexpr (SaveIntensity) samples->rows[y].pixels = intensity;
        if constexpr (Window == 13) {
            const auto error = w.sample_pairs<intensity_shift>(top.tail, bottom.tail, row.tail_bias);
            vector_sum = wasm_i32x4_add(vector_sum, wasm_i32x4_abs(error));
            if constexpr (SaveIntensity)
                samples->rows[y].tail_bias = wasm_i32x4_sub(row.tail_bias, wasm_i32x4_shl(error, intensity_shift));
        }
    };
    static_assert((Window - 1) % 4 == 0);
    auto first = row_at(0);
#pragma clang loop unroll(full)
    for (int y = 0; y + 3 < Window; y += 4) {
        const auto second = row_at(y + 1), third = row_at(y + 2);
        const auto fourth = row_at(y + 3), next = row_at(y + 4);
        add_inner(y, first, second);
        add_inner(y + 1, second, third);
        add_inner(y + 2, third, fourth);
        add_inner(y + 3, fourth, next);
        const auto join = [](v128_t a, v128_t b, v128_t c, v128_t d) {
            const auto low = wasm_i32x4_shuffle(a, b, 0, 4, 0, 4);
            const auto high = wasm_i32x4_shuffle(c, d, 0, 4, 0, 4);
            return wasm_i32x4_shuffle(low, high, 0, 1, 4, 5);
        };
        const auto top = join(first.edge, second.edge, third.edge, fourth.edge);
        const auto bottom = join(second.edge, third.edge, fourth.edge, next.edge);
        const auto bias = wasm_i32x4_shuffle(source.edge_pair(y).bias, source.edge_pair(y + 2).bias, 0, 2, 4, 6);
        const auto error = w.sample_pairs<intensity_shift>(top, bottom, bias);
        edge_sum = wasm_i32x4_add(edge_sum, wasm_i32x4_abs(error));
        if constexpr (SaveIntensity)
            wasm_v128_store(samples->edge_bias.data() + y,
                            wasm_i32x4_sub(bias, wasm_i32x4_shl(error, intensity_shift)));
        first = next;
    }
    const auto last = row_at(Window);
    add_inner(Window - 1, first, last);
    const auto bias = wasm_i32x4_shuffle(patch.back().edge_sample, patch.back().edge_sample, 2, 2, 2, 2);
    const auto edge = w.sample_pairs<intensity_shift>(first.edge, last.edge, bias);
    const int error = wasm_i32x4_extract_lane(edge, 0);
    if constexpr (SaveIntensity)
        samples->edge_bias.back() = wasm_i32x4_extract_lane(bias, 0) - error * (1 << intensity_shift);
    const auto sums = wasm_i32x4_add(vector_sum, edge_sum);
    // The sum of absolute patch errors fits exactly in float32.
    const int sum = std::abs(error) + wasm_i32x4_extract_lane(sums, 0)
                  + wasm_i32x4_extract_lane(sums, 1) + wasm_i32x4_extract_lane(sums, 2)
                  + wasm_i32x4_extract_lane(sums, 3);
    return float(sum) / (32 * Window * Window);
}

template<int Window, int Width, bool ReuseIntensity = false>
void prepare_source(const cv::Mat& previous, const cv::Mat& derivative, cv::Point2f source,
                    SourcePatch<Window>& patch, const IntensityPatch<Window>* samples = nullptr) {
    const cv::Point2f half((Window - 1) * .5f, (Window - 1) * .5f);
    const auto source_corner = source - half;
    patch.inverse = 0.f;
    if (!inside_padded_image<Window, Width>(source_corner)) return;
    const PatchOrigin origin(source_corner);
    const bool integer_origin = source_corner.x == origin.pixel.x && source_corner.y == origin.pixel.y;
    patch.matrix = integer_origin ? prepare_patch<Window, Width, true, ReuseIntensity>(previous, derivative, origin, patch, samples)
                                  : prepare_patch<Window, Width, false, ReuseIntensity>(previous, derivative, origin, patch, samples);
    const auto [xx, xy, yy] = patch.matrix;
    const float determinant = xx * yy - xy * xy;
    const float minimum_eigenvalue = (yy + xx - std::sqrt((xx - yy) * (xx - yy) + 4.f * xy * xy))
                                   / (2 * Window * Window);
    if (minimum_eigenvalue >= 1e-4f && determinant >= FLT_EPSILON)
        patch.inverse = 1.f / determinant;
}

template<int Window, int Width, bool SaveIntensity = false, bool ReverseGrid = false, class Filter>
void refine_point(const cv::Mat& current, const SourcePatch<Window>& patch, cv::Point2f& target,
                  bool finest_level, uint8_t& status, const Filter& filter, IntensityPatch<Window>* samples = nullptr,
                  int* work = nullptr) {
    if (patch.inverse == 0.f) {
        if (finest_level) status = 0;
        return;
    }
    constexpr float half = (Window - 1) * .5f;
    const auto radius = wasm_f32x4_splat(half);
    const auto lower = wasm_f32x4_splat(-Window);
    const auto upper = wasm_f32x4_make(Width, Width * 3 / 4, 0.f, 0.f);
    const auto [xx, xy, yy] = patch.matrix;
    const auto diagonal = wasm_f32x4_make(yy, xx, 0.f, 0.f);
    const auto cross = wasm_f32x4_splat(xy);
    const auto inverse = wasm_f32x4_splat(patch.inverse);
    auto fitted = wasm_f32x4_make(target.x, target.y, 0.f, 0.f);
    auto estimate = wasm_f32x4_sub(fitted, radius);
    auto previous_delta = wasm_f32x4_splat(0.f);
    int iteration = 0;
    for (; iteration < 10; ++iteration) {
        const auto valid = wasm_v128_and(wasm_f32x4_ge(estimate, lower), wasm_f32x4_lt(estimate, upper));
        if ((wasm_i32x4_bitmask(valid) & 3) != 3) {
            if (finest_level) status = 0;
            break;
        }
        const auto rhs = ReverseGrid && iteration == 0
            ? initial_reverse_residual<Window, Width>(current, estimate, patch)
            : residual<Window, Width>(current, PatchOrigin(estimate), patch);
        const auto swapped = wasm_i32x4_shuffle(rhs, rhs, 1, 0, 2, 3);
        const auto delta = wasm_f32x4_mul(wasm_f32x4_sub(wasm_f32x4_mul(cross, swapped),
                                                       wasm_f32x4_mul(diagonal, rhs)), inverse);
        estimate = wasm_f32x4_add(estimate, delta);
        fitted = wasm_f32x4_add(estimate, radius);
        // Keep the double-precision convergence comparisons.
        const auto wide = wasm_f64x2_promote_low_f32x4(delta);
        const auto squares = wasm_f64x2_mul(wide, wide);
        if (wasm_f64x2_extract_lane(squares, 0) + wasm_f64x2_extract_lane(squares, 1) <= .015 * .015) break;
        if (iteration > 0) {
            const auto oscillation = wasm_f64x2_promote_low_f32x4(wasm_f32x4_add(delta, previous_delta));
            if (wasm_i64x2_bitmask(wasm_f64x2_lt(wasm_f64x2_abs(oscillation), wasm_f64x2_splat(.01))) == 3) {
                fitted = wasm_f32x4_sub(fitted, wasm_f32x4_mul(delta, wasm_f32x4_splat(.5f)));
                break;
            }
        }
        previous_delta = delta;
    }
    target = {wasm_f32x4_extract_lane(fitted, 0), wasm_f32x4_extract_lane(fitted, 1)};
    // Count an attempted iteration conservatively if the estimate leaves the image.
    if (work) *work += std::min(iteration + 1, 10);
    if constexpr (!std::is_same_v<Filter, std::nullptr_t>) {
        if (status && finest_level) {
            const auto target_corner = target - cv::Point2f(half, half);
            status = inside_padded_image<Window, Width>(target_corner) && filter(target);
            if (status) {
                if (work) ++*work;
                status = mean_error<Window, Width, SaveIntensity>(
                    current, PatchOrigin(target_corner), patch, samples) <= photometric_error_limit;
            }
        }
    }
}

template<int Window, int Width, bool ReuseIntensity = false, bool ReverseGrid = false, class Filter>
void track_point(const cv::Mat& previous, const cv::Mat& derivative, const cv::Mat& current,
                 cv::Point2f source, cv::Point2f& target, bool finest_level, uint8_t& status,
                 const Filter& filter, const IntensityPatch<Window>* samples = nullptr) {
    SourcePatch<Window> patch;
    prepare_source<Window, Width, ReuseIntensity>(previous, derivative, source, patch, samples);
    refine_point<Window, Width, false, ReverseGrid>(current, patch, target, finest_level, status, filter);
}

template<int Window, int Width>
void refine_coarse_prepared(const cv::Mat& coarse, const SourcePatch<Window>& patch, cv::Point2f& target,
                            int* work = nullptr) {
    uint8_t status = 1;
    target *= .5f;
    refine_point<Window, Width>(coarse, patch, target, false, status, nullptr, nullptr, work);
    target *= 2.f;
}

template<int Window, int Width, int Levels, bool SaveIntensity = false, class Filter>
bool track_prepared(const std::vector<cv::Mat>& current, const std::array<SourcePatch<Window>, Levels + 1>& patches,
                    cv::Point2f& target, const Filter& filter, bool coarse_ready = false,
                    IntensityPatch<Window>* samples = nullptr, int* work = nullptr) {
    if constexpr (Levels == 1) {
        if (!coarse_ready) refine_coarse_prepared<Window, Width / 2>(current[2], patches[1], target, work);
    }
    uint8_t status = 1;
    refine_point<Window, Width, SaveIntensity>(current[0], patches[0], target, true, status, filter, samples, work);
    return status != 0;
}

template<int Window, int Width, int Levels, bool ReuseIntensity = false, class Filter = std::nullptr_t>
bool track(const std::vector<cv::Mat>& previous, const std::vector<cv::Mat>& current,
           cv::Point2f source, cv::Point2f& target, const Filter& filter = nullptr,
           const IntensityPatch<Window>* samples = nullptr) {
    static_assert(Window == 9 || Window == 13);
    static_assert(Levels == 0 || Levels == 1);
    uint8_t status = 1;
    if constexpr (Levels == 1) {
        target *= .5f;
        track_point<Window, Width / 2, false, ReuseIntensity>(previous[2], previous[3], current[2], source * .5f, target, false, status, nullptr);
        target *= 2.f;
    }
    track_point<Window, Width, ReuseIntensity>(previous[0], previous[1], current[0], source, target, true, status, filter, samples);
    return status != 0;
}
} // namespace fixed_lk
