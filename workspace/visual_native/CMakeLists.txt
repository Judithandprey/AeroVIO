cmake_minimum_required(VERSION 3.20)
project(aerovio_visual_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(VIOBENCH_NATIVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../runtime/native" CACHE PATH "viobench native build helpers")
include("${VIOBENCH_NATIVE_DIR}/viobench_reactor.cmake")
viobench_find_opencv(core calib3d imgproc video)

if(DEFINED ENV{PYTHON3} AND NOT DEFINED Python3_EXECUTABLE)
    set(Python3_EXECUTABLE "$ENV{PYTHON3}")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(checkpoint "${CMAKE_CURRENT_SOURCE_DIR}/../../experiments/motion-ablation-v2/imu-only-compact")
set(weights "${CMAKE_CURRENT_BINARY_DIR}/weights.h")
add_custom_command(
    OUTPUT "${weights}"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/export_weights.py" "${weights}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/export_weights.py" "${checkpoint}/net.onnx" "${checkpoint}/meta.json"
    VERBATIM
)

option(LEO_WEIGHTS_NOHOVER "Use the MLP retrained on original + no-hover flights (weights_nohover.h)" OFF)
set(LEO_WEIGHTS_NOHOVER_CHECKPOINT "${CMAKE_CURRENT_SOURCE_DIR}/../../experiments/nohover-v11/compact" CACHE PATH
    "LEO_WEIGHTS_NOHOVER: compact checkpoint directory (net.onnx + meta.json)")
set(nohover_checkpoint "${LEO_WEIGHTS_NOHOVER_CHECKPOINT}")
if(LEO_WEIGHTS_NOHOVER)
    set(weights "${CMAKE_CURRENT_BINARY_DIR}/weights_nohover.h")
    add_custom_command(
        OUTPUT "${weights}"
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/export_weights.py" "${weights}"
                --checkpoint "${nohover_checkpoint}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/export_weights.py" "${nohover_checkpoint}/net.onnx" "${nohover_checkpoint}/meta.json"
        VERBATIM
    )
endif()

set(VISUAL_DOWNSAMPLE 2 CACHE STRING "Integer downsample factor for sparse visual tracking (2 or 4)")
if(VISUAL_DOWNSAMPLE STREQUAL "2")
    set(pyramid_levels 1)
elseif(VISUAL_DOWNSAMPLE STREQUAL "4")
    set(pyramid_levels 0)
else()
    message(FATAL_ERROR "VISUAL_DOWNSAMPLE must be 2 or 4")
endif()

set(VISUAL_INTERVAL 8 CACHE STRING "Frames per visual tracking cycle (>= 8); 8 is the teacher schedule, larger values only idle longer between cycles")
if(NOT VISUAL_INTERVAL MATCHES "^[0-9]+$" OR VISUAL_INTERVAL LESS 8)
    message(FATAL_ERROR "VISUAL_INTERVAL must be an integer >= 8")
endif()
option(LEO_IMU_ONLY "Cheap Pareto point: compact IMU network + gyro calibration only, never call the visual tracker" OFF)
if(LEO_IMU_ONLY AND (VISUAL_ROTATION_CORRECTION OR VISUAL_CONFIDENCE_WEIGHTING))
    message(FATAL_ERROR "LEO_IMU_ONLY disables the visual tracker; turn off the V8 visual options")
endif()

option(VISUAL_SPARSE_DIRECTION "Accept strongly constrained fits with eight or nine tracks at 160x120" ON)
option(VISUAL_CONFIDENCE_WEIGHTING "V8 heuristic quality weighting of accepted direction fits" OFF)
option(VISUAL_ROTATION_CORRECTION "V8 bounded joint epipolar rotation refinement" OFF)
option(BUILD_FUSION_TESTS "Build standalone geometry tests for Wasmtime" OFF)

viobench_solution(solution solution.cpp "${weights}")
target_include_directories(solution PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_definitions(solution PRIVATE
    VISUAL_DOWNSAMPLE=${VISUAL_DOWNSAMPLE} VISUAL_PYRAMID_LEVELS=${pyramid_levels}
    VISUAL_INTERVAL=${VISUAL_INTERVAL} VISUAL_SPARSE_DIRECTION=$<BOOL:${VISUAL_SPARSE_DIRECTION}>
    LEO_IMU_ONLY=$<BOOL:${LEO_IMU_ONLY}>
    VISUAL_CONFIDENCE_WEIGHTING=$<BOOL:${VISUAL_CONFIDENCE_WEIGHTING}>
    VISUAL_ROTATION_CORRECTION=$<BOOL:${VISUAL_ROTATION_CORRECTION}>
    USE_SMALL_ANGLE_ROTATION=1 USE_LEARNED_ROTATION_RESIDUAL=0
    _WASI_EMULATED_PROCESS_CLOCKS _WASI_EMULATED_SIGNAL _WASI_EMULATED_MMAN _WASI_EMULATED_GETPID
)
if(LEO_WEIGHTS_NOHOVER)
    target_compile_definitions(solution PRIVATE LEO_WEIGHTS_NOHOVER=1)
endif()
option(LEO_GRAVITY_TILT "Mahony-style accelerometer roll/pitch correction of the gyro rotation" OFF)
set(LEO_GRAVITY_TILT_GAIN 0.01 CACHE STRING "LEO_GRAVITY_TILT gain K in 1/s (per-sample correction K * e * dt)")
set(LEO_GRAVITY_TILT_GATE 0.5 CACHE STRING "LEO_GRAVITY_TILT: use samples with | |a| - 9.81 | below this (m/s^2)")
set(LEO_GRAVITY_TILT_SIGN 1 CACHE STRING "LEO_GRAVITY_TILT diagnostic: -1 flips the correction sign")
if(LEO_GRAVITY_TILT)
    if(NOT LEO_GRAVITY_TILT_SIGN MATCHES "^-?1$")
        message(FATAL_ERROR "LEO_GRAVITY_TILT_SIGN must be 1 or -1")
    endif()
    target_compile_definitions(solution PRIVATE LEO_GRAVITY_TILT=1
        LEO_GRAVITY_TILT_GAIN=${LEO_GRAVITY_TILT_GAIN} LEO_GRAVITY_TILT_GATE=${LEO_GRAVITY_TILT_GATE}
        LEO_GRAVITY_TILT_SIGN=${LEO_GRAVITY_TILT_SIGN})
endif()
option(LEO_ONLINE_BIAS "Online gyro bias estimation from the V8 visual rotation refinement" OFF)
set(LEO_ONLINE_BIAS_GAIN 0.05 CACHE STRING "LEO_ONLINE_BIAS: fraction of the measured rate error applied per accepted fit")
set(LEO_ONLINE_BIAS_REJECT 0.03 CACHE STRING "LEO_ONLINE_BIAS: reject fits implying a rate error above this (rad/s)")
set(LEO_ONLINE_BIAS_SIGN 1 CACHE STRING "LEO_ONLINE_BIAS diagnostic: -1 flips the update sign")
if(LEO_ONLINE_BIAS)
    if(NOT VISUAL_ROTATION_CORRECTION OR LEO_IMU_ONLY)
        message(FATAL_ERROR "LEO_ONLINE_BIAS needs VISUAL_ROTATION_CORRECTION=ON and LEO_IMU_ONLY=OFF")
    endif()
    if(NOT LEO_ONLINE_BIAS_SIGN MATCHES "^-?1$")
        message(FATAL_ERROR "LEO_ONLINE_BIAS_SIGN must be 1 or -1")
    endif()
    target_compile_definitions(solution PRIVATE LEO_ONLINE_BIAS=1 LEO_ONLINE_BIAS_GAIN=${LEO_ONLINE_BIAS_GAIN}
        LEO_ONLINE_BIAS_REJECT=${LEO_ONLINE_BIAS_REJECT} LEO_ONLINE_BIAS_SIGN=${LEO_ONLINE_BIAS_SIGN})
endif()
target_compile_options(solution PRIVATE -ffp-contract=off -fwasm-exceptions "SHELL:-mllvm -wasm-use-legacy-eh=false")
target_link_options(solution PRIVATE -fwasm-exceptions -Wl,-z,stack-size=8388608 -Wl,--stack-first)
target_link_libraries(solution PRIVATE ${OpenCV_LIBS} unwind c-printscan-long-double
    wasi-emulated-process-clocks wasi-emulated-signal wasi-emulated-mman wasi-emulated-getpid)

if(BUILD_FUSION_TESTS)
    viobench_solution(fusion_tests "${CMAKE_CURRENT_SOURCE_DIR}/../../tests/native_fusion_geometry.cpp")
    target_include_directories(fusion_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_definitions(fusion_tests PRIVATE USE_SMALL_ANGLE_ROTATION=1
        _WASI_EMULATED_PROCESS_CLOCKS _WASI_EMULATED_SIGNAL _WASI_EMULATED_MMAN _WASI_EMULATED_GETPID)
    target_compile_options(fusion_tests PRIVATE -ffp-contract=off -fwasm-exceptions "SHELL:-mllvm -wasm-use-legacy-eh=false")
    target_link_options(fusion_tests PRIVATE -fwasm-exceptions -Wl,-z,stack-size=8388608 -Wl,--stack-first)
    target_link_libraries(fusion_tests PRIVATE ${OpenCV_LIBS} unwind c-printscan-long-double
        wasi-emulated-process-clocks wasi-emulated-signal wasi-emulated-mman wasi-emulated-getpid)
endif()
