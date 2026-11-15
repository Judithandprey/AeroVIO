#pragma once
#include "weights.h"
#include <array>
#include <algorithm>
#include <cmath>
#include <wasm_simd128.h>

#ifndef GATE_FORCE
#define GATE_FORCE -1
#endif
#ifndef GATE_DIAGNOSTICS
#define GATE_DIAGNOSTICS 0
#endif

template<size_t Inputs,size_t Outputs>
std::array<float,Outputs> dense(const std::array<float,Inputs>& input,
    const std::array<std::array<float,Outputs>,Inputs>& weight,const std::array<float,Outputs>& bias) {
    static_assert(Outputs%4==0);
    std::array<float,Outputs> output{};
    for(size_t unit=0;unit<Outputs;unit+=4) {
        auto total=wasm_f32x4_splat(0.f);
        for(size_t i=0;i<Inputs;++i)
            total=wasm_f32x4_add(total,wasm_f32x4_mul(wasm_f32x4_splat(input[i]),wasm_v128_load(weight[i].data()+unit)));
        wasm_v128_store(output.data()+unit,wasm_f32x4_add(total,wasm_v128_load(bias.data()+unit)));
    }
    return output;
}

#if GATE_DIAGNOSTICS
inline std::array<float,24> last_gate_features{};
inline float last_gate_value=0.f;
#endif

inline std::array<float,6> infer(std::array<float,18> input) {
    const float dt=std::max(input[13],1.e-6f);
    for(size_t i=0;i<18;++i)input[i]=(input[i]-memory_weights::state_mean[i])/memory_weights::state_std[i];
    std::array<float,14> base_input{};
    std::copy_n(input.begin(),14,base_input.begin());
    auto bh=dense(base_input,base_weights::weight1,base_weights::bias1);
    for(auto& v:bh)v=std::max(0.f,v);
    const auto bn=dense(bh,base_weights::weight2,base_weights::bias2);
    std::array<float,6> base{};
    for(size_t i=0;i<6;++i)base[i]=bn[i]*base_weights::target_std[i]+base_weights::target_mean[i];
    if constexpr(GATE_FORCE==0)return base;
    auto mh=dense(input,memory_weights::weight1,memory_weights::bias1);
    for(auto& v:mh)v=std::max(0.f,v);
    const auto mn=dense(mh,memory_weights::weight2,memory_weights::bias2);
    std::array<float,6> memory{};
    for(size_t i=0;i<6;++i)memory[i]=mn[i]*memory_weights::target_std[i]+memory_weights::target_mean[i];
    if constexpr(GATE_FORCE==1)return memory;
    std::array<float,24> features{};
    std::copy(input.begin(),input.end(),features.begin());
    for(size_t i=0;i<3;++i) {
        features[18+i]=std::clamp(base[i]/dt,-2.f,2.f);
        features[21+i]=std::clamp(memory[i]/dt,-2.f,2.f);
    }
    auto gh=dense(features,gate_weights::weight1,gate_weights::bias1);
    for(auto& v:gh)v=std::max(0.f,v);
    const auto logit=dense(gh,gate_weights::weight2,gate_weights::bias2);
    const float g=1.f/(1.f+std::exp(-logit[0]));
#if GATE_DIAGNOSTICS
    last_gate_features=features;last_gate_value=g;
#endif
    auto motion=base;
    for(size_t i=0;i<3;++i)motion[i]=(1.f-g)*base[i]+g*memory[i];
    return motion;
}
