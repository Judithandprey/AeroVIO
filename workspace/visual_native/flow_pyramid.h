#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <wasm_simd128.h>
#include <array>
#include <cstdint>
#include <cstring>

namespace visual_direction {

struct GradientBlock { v128_t dx, dy; };

template<int Side, int Center>
struct DerivativeColumns {
    v128_t smoothed, difference;

    DerivativeColumns(const uint8_t* center, int stride) {
        const auto top = wasm_u16x8_load8x8(center - stride);
        const auto middle = wasm_u16x8_load8x8(center);
        const auto bottom = wasm_u16x8_load8x8(center + stride);
        smoothed = wasm_i16x8_add(wasm_i16x8_mul(wasm_i16x8_splat(Side), wasm_i16x8_add(top, bottom)),
                                  wasm_i16x8_mul(wasm_i16x8_splat(Center), middle));
        difference = wasm_i16x8_sub(bottom, top);
    }

    GradientBlock finish(const DerivativeColumns& right) const {
        const auto smooth_right = wasm_i16x8_shuffle(smoothed, right.smoothed, 2, 3, 4, 5, 6, 7, 8, 9);
        const auto delta_center = wasm_i16x8_shuffle(difference, right.difference, 1, 2, 3, 4, 5, 6, 7, 8);
        const auto delta_right = wasm_i16x8_shuffle(difference, right.difference, 2, 3, 4, 5, 6, 7, 8, 9);
        return {wasm_i16x8_sub(smooth_right, smoothed),
                wasm_i16x8_add(wasm_i16x8_mul(wasm_i16x8_splat(Center), delta_center),
                    wasm_i16x8_mul(wasm_i16x8_splat(Side), wasm_i16x8_add(difference, delta_right)))};
    }
};

constexpr int horizontal_padding = 16;

// The x and y planes share a backing allocation, with a zero halo in each.
inline void allocate_gradient(cv::Mat& gradient, int width, int height, int padding) {
    if (gradient.empty()) {
        const int plane_width = width + 2 * horizontal_padding;
        cv::Mat backing = cv::Mat::zeros(height + 2 * padding, 2 * plane_width, CV_16SC1);
        gradient = backing(cv::Rect(horizontal_padding, padding, width, height));
    }
}

// The horizontal Scharr taps are shared by three successive output rows.
struct HorizontalScharr {
    v128_t smoothed, difference;

    explicit HorizontalScharr(const uint8_t* pixels) {
        const auto row = wasm_v128_load(pixels);
        const auto zero = wasm_i32x4_splat(0);
        const auto left = wasm_u16x8_extend_low_u8x16(row);
        const auto middle = wasm_i8x16_shuffle(row, zero, 1, 16, 2, 16, 3, 16, 4, 16, 5, 16, 6, 16, 7, 16, 8, 16);
        const auto right = wasm_i8x16_shuffle(row, zero, 2, 16, 3, 16, 4, 16, 5, 16, 6, 16, 7, 16, 8, 16, 9, 16);
        smoothed = weighted(left, middle, right);
        difference = wasm_i16x8_sub(right, left);
    }

    static v128_t weighted(v128_t first, v128_t middle, v128_t third) {
        return wasm_i16x8_add(wasm_i16x8_mul(wasm_i16x8_splat(3), wasm_i16x8_add(first, third)),
                             wasm_i16x8_mul(wasm_i16x8_splat(10), middle));
    }
};

inline void store_scharr(int16_t* dx_out, int16_t* dy_out, const HorizontalScharr& top,
                         const HorizontalScharr& middle, const HorizontalScharr& bottom) {
    const auto dx = HorizontalScharr::weighted(top.difference, middle.difference, bottom.difference);
    const auto dy = wasm_i16x8_sub(bottom.smoothed, top.smoothed);
    wasm_v128_store(dx_out, dx);
    wasm_v128_store(dy_out, dy);
}

// The image halo is reflected. Only the interior of the gradient is written.
template<int Width, int Begin, int End>
void scharr(const cv::Mat& image, cv::Mat& gradient) {
    static_assert(Width % 8 == 0 && Begin % 2 == 0 && End % 2 == 0);
    // Match the strides of allocate_padded and the two-plane allocate_gradient.
    constexpr int stride = Width + 2 * horizontal_padding;
    constexpr int derivative_stride = 2 * stride;
    for (int x = 0; x < Width; x += 8) {
        const auto* source = image.data + Begin * stride + x - 1;
        auto* dx = reinterpret_cast<int16_t*>(gradient.data) + Begin * derivative_stride + x;
        auto* dy = dx + derivative_stride / 2;
        HorizontalScharr top(source - stride);
        HorizontalScharr middle(source);
#pragma clang loop unroll(full)
        for (int y = Begin; y < End; y += 2) {
            const HorizontalScharr bottom(source + stride);
            const HorizontalScharr next(source + 2 * stride);
            store_scharr(dx, dy, top, middle, bottom);
            store_scharr(dx + derivative_stride, dy + derivative_stride, middle, bottom, next);
            top = bottom;
            middle = next;
            source += 2 * stride;
            dx += 2 * derivative_stride;
            dy += 2 * derivative_stride;
        }
    }
}

inline v128_t gaussian5(v128_t left, v128_t near_left, v128_t center,
                        v128_t near_right, v128_t right) {
    const auto near = wasm_i16x8_shl(wasm_i16x8_add(near_left, near_right), 2);
    const auto middle = wasm_i16x8_mul(center, wasm_i16x8_splat(6));
    return wasm_i16x8_add(wasm_i16x8_add(near, middle), wasm_i16x8_add(left, right));
}

// Group the symmetric byte pairs before widening for the 5-tap filter.
struct GaussianColumns {
    v128_t bytes;

    explicit GaussianColumns(const uint8_t* pixels) : bytes(wasm_v128_load(pixels)) {}

    v128_t horizontal(const GaussianColumns& next) const {
        const auto outer = wasm_i8x16_shuffle(bytes, next.bytes, 0, 4, 2, 6, 4, 8, 6, 10, 8, 12, 10, 14, 12, 16, 14, 18);
        const auto inner = wasm_i8x16_shuffle(bytes, next.bytes, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15, 17);
        const auto centers = wasm_i8x16_shuffle(bytes, next.bytes, 2, 4, 6, 8, 10, 12, 14, 16, 0, 0, 0, 0, 0, 0, 0, 0);
        const auto outer_sum = wasm_u16x8_extadd_pairwise_u8x16(outer);
        const auto inner_sum = wasm_i16x8_shl(wasm_u16x8_extadd_pairwise_u8x16(inner), 2);
        const auto center = wasm_i16x8_mul(wasm_u16x8_extend_low_u8x16(centers), wasm_i16x8_splat(6));
        return wasm_i16x8_add(wasm_i16x8_add(outer_sum, inner_sum), center);
    }
};

template<int Width, int Height>
void pyramid_down(const cv::Mat& source, cv::Mat& target) {
    static_assert(Width % 16 == 0);
    // Both pyramid levels are already allocated with horizontal padding.
    constexpr size_t blocks = Width / 16;
    using Row = std::array<v128_t, blocks>;
    std::array<Row, 5> rows;
    constexpr int stride = Width + 2 * horizontal_padding;
    constexpr int target_stride = Width / 2 + 2 * horizontal_padding;
    auto compute_row = [&](Row& row, int y) {
        const auto* pixels = source.data + y * stride;
        GaussianColumns left(pixels - 2);
#pragma clang loop unroll(full)
        for (size_t x = 0; x < row.size(); ++x) {
            const GaussianColumns right(pixels + 16 * x + 14);
            row[x] = left.horizontal(right);
            left = right;
        }
    };
    Row* previous2 = &rows[0];
    Row* previous1 = &rows[1];
    Row* center = &rows[2];
    Row* next1 = &rows[3];
    Row* next2 = &rows[4];
    compute_row(*previous2, -2);
    compute_row(*previous1, -1);
    compute_row(*center, 0);
    for (int y = 0; y < Height / 2; ++y) {
        compute_row(*next1, 2 * y + 1);
        compute_row(*next2, 2 * y + 2);
        auto* dest = target.data + y * target_stride;
        const auto rounded = [&](size_t x) {
            const auto sum = gaussian5((*previous2)[x], (*previous1)[x], (*center)[x], (*next1)[x], (*next2)[x]);
            return wasm_u16x8_shr(wasm_i16x8_add(sum, wasm_i16x8_splat(128)), 8);
        };
#pragma clang loop unroll(full)
        for (size_t x = 0; x + 1 < blocks; x += 2) {
            wasm_v128_store(dest + 8 * x, wasm_u8x16_narrow_i16x8(rounded(x), rounded(x + 1)));
        }
        if constexpr (blocks % 2 != 0) {
            constexpr size_t last = blocks - 1;
            const auto tail = rounded(last);
            wasm_v128_store64_lane(dest + 8 * last, wasm_u8x16_narrow_i16x8(tail, tail), 0);
        }
        auto* first = previous2;
        auto* second = previous1;
        previous2 = center;
        previous1 = next1;
        center = next2;
        next1 = first;
        next2 = second;
    }
}

inline void allocate_padded(cv::Mat& image, int width, int height, int padding) {
    if (image.empty()) {
        cv::Mat backing(height + 2 * padding, width + 2 * horizontal_padding, CV_8UC1);
        image = backing(cv::Rect(horizontal_padding, padding, width, height));
    }
}

inline v128_t reflected_bytes(const uint8_t* pixels) {
    const auto row = wasm_v128_load(pixels);
    return wasm_i8x16_shuffle(row, row, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
}

template<int Width, int Height, int Padding>
void reflect_border(cv::Mat& image) {
    // Reflect without repeating the outermost pixel (OpenCV BORDER_REFLECT_101).
    // The backing allocation has a fixed stride, so each halo copy has a fixed size.
    constexpr int stride = Width + 2 * horizontal_padding;
    auto* row = image.data;
    for (int y = 0; y < Height; ++y, row += stride) {
        wasm_v128_store(row - horizontal_padding, reflected_bytes(row + 1));
        wasm_v128_store(row + Width, reflected_bytes(row + Width - horizontal_padding - 1));
    }
    auto* first = image.data - horizontal_padding;
    auto* last = first + (Height - 1) * stride;
    for (int offset = stride; offset <= Padding * stride; offset += stride) {
        std::memcpy(first - offset, first + offset, stride);
        std::memcpy(last + offset, last - offset, stride);
    }
}

} // namespace visual_direction
