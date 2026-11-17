#pragma once

#include "weights.h"
#include <array>
#include <algorithm>
#include <wasm_simd128.h>

// Each SIMD lane is one output. Accumulate input terms in their original order
// and add the bias after the dot product, matching the WASM OpenCV Gemm kernel.
template<size_t Inputs, size_t Outputs>
std::array<float, Outputs> dense(const std::array<float, Inputs>& input,
                                const std::array<std::array<float, Outputs>, Inputs>& weight,
                                const std::array<float, Outputs>& bias) {
    static_assert(Outputs % 4 == 0);
    std::array<float, Outputs> output{};
    for (size_t unit = 0; unit < Outputs; unit += 4) {
        auto total = wasm_f32x4_splat(0.f);
        for (size_t i = 0; i < Inputs; ++i) {
            const auto term = wasm_f32x4_mul(wasm_f32x4_splat(input[i]),
                                            wasm_v128_load(weight[i].data() + unit));
            total = wasm_f32x4_add(total, term);
        }
        wasm_v128_store(output.data() + unit, wasm_f32x4_add(total, wasm_v128_load(bias.data() + unit)));
    }
    return output;
}

inline std::array<float, 6> infer(std::array<float, 14> input) {
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = (input[i] - state_mean[i]) / state_std[i];
    }
    auto hidden = dense(input, weight1, bias1);
    for (float& value : hidden) value = std::max(0.f, value);
    const auto raw = dense(hidden, weight2, bias2);
    std::array<float, 6> motion{};
    for (size_t i = 0; i < motion.size(); ++i) motion[i] = raw[i] * target_std[i] + target_mean[i];
    return motion;
}
