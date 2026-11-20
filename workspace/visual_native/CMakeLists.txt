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

set(POSITION_SCALE 1.0 CACHE STRING "Multiplier on the learned translation increment (stage 2 metric scale)")
set(GRAVITY_TILT_GAIN 0 CACHE STRING "Gravity tilt correction rate in 1/s (0 disables)")
set(GRAVITY_TILT_TAU 3 CACHE STRING "Gravity tilt running-average time constant in seconds")
set(GRAVITY_TILT_ANCHOR 5 CACHE STRING "Seconds used to anchor the gravity direction")
option(ONLINE_GYRO_BIAS "Re-estimate gyro bias online from the visual rotation residual" OFF)
set(ONLINE_GYRO_BIAS_GAIN 0.3 CACHE STRING "Online gyro-bias update gain per visual fit")
set(ONLINE_GYRO_BIAS_SIGN 1 CACHE STRING "Sign convention of the visual residual (1 or -1)")
option(VISUAL_SPARSE_DIRECTION "Accept strongly constrained fits with eight or nine tracks at 160x120" ON)
option(VISUAL_CONFIDENCE_WEIGHTING "V8 heuristic quality weighting of accepted direction fits" OFF)
option(VISUAL_ROTATION_CORRECTION "V8 bounded joint epipolar rotation refinement" OFF)
option(BUILD_FUSION_TESTS "Build standalone geometry tests for Wasmtime" OFF)

viobench_solution(solution solution.cpp "${weights}")
target_include_directories(solution PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_definitions(solution PRIVATE
    VISUAL_DOWNSAMPLE=${VISUAL_DOWNSAMPLE} VISUAL_PYRAMID_LEVELS=${pyramid_levels}
    VISUAL_INTERVAL=${VISUAL_INTERVAL} VISUAL_SPARSE_DIRECTION=$<BOOL:${VISUAL_SPARSE_DIRECTION}>
    LEO_IMU_ONLY=$<BOOL:${LEO_IMU_ONLY}> POSITION_SCALE=${POSITION_SCALE}
    GRAVITY_TILT_GAIN=${GRAVITY_TILT_GAIN} GRAVITY_TILT_TAU=${GRAVITY_TILT_TAU} GRAVITY_TILT_ANCHOR=${GRAVITY_TILT_ANCHOR}
    ONLINE_GYRO_BIAS=$<BOOL:${ONLINE_GYRO_BIAS}> ONLINE_GYRO_BIAS_GAIN=${ONLINE_GYRO_BIAS_GAIN} ONLINE_GYRO_BIAS_SIGN=${ONLINE_GYRO_BIAS_SIGN}
    VISUAL_CONFIDENCE_WEIGHTING=$<BOOL:${VISUAL_CONFIDENCE_WEIGHTING}>
    VISUAL_ROTATION_CORRECTION=$<BOOL:${VISUAL_ROTATION_CORRECTION}>
    USE_SMALL_ANGLE_ROTATION=1 USE_LEARNED_ROTATION_RESIDUAL=0
    _WASI_EMULATED_PROCESS_CLOCKS _WASI_EMULATED_SIGNAL _WASI_EMULATED_MMAN _WASI_EMULATED_GETPID
)
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
