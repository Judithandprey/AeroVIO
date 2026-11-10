// HyperDrone World specification of the challenge's data generator; rendered from hyperdrone_world.h.in by
// viobench.tools.datasets.generate.simulation (the placeholders become numbers) and handed to hyperdrone.Config(spec_header=...).
// The content hash of the rendered file is part of the JIT key: every distinct configuration compiles once.
//
// Preset x500_fpv_imu (200 Hz IMU in the dynamics state chain, FPV camera), visual-inertial localization task
// (autopilot flies a random waypoint route), rendering every tenth IMU step so the data pipeline can integrate any shutter
// time itself (viobench.tools.datasets.generate.camera), a hover at the start of each episode, randomised horizontal FOV,
// and a route length that the generator scales with the episode duration (the preset's EPISODE_STEP_LIMIT is not
// enforced anywhere; the generator steps for as long as it wants).
#pragma once
#include <rl_tools/rl/environments/hyperdrone/presets.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cpu.h>
// the environment only includes a task's operations for its built-in task selection; a user World brings its own
#include <rl_tools/rl/environments/hyperdrone/tasks/visual_inertial_localization/visual_inertial_localization.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/visual_inertial_localization/operations_cpu.h>
#include <cstddef>

#pragma once
#include <rl_tools/hyperlux/outputs/outputs.h>
namespace aerovio_camera_config {
    namespace out = rl_tools::hyperlux::outputs;
    constexpr unsigned int SCHEMA_VERSION = 1;
    struct Parameters0 { static constexpr float SCALE = 10.0f; };
    using Output0 = out::Depth<out::format::Float32, Parameters0>;
    using Output1 = out::RGB<out::format::Float32>;
    using OUTPUTS = out::List<Output0, Output1>;
    namespace h = rl_tools::rl::environments::hyperdrone;
    using CAMERA_OUTPUTS = h::CameraOutputs<out::List<Output0, Output1>>;
    using OBSERVATION = h::ImageObservation<0>;
}

namespace hyperdrone_user {
    using T = float;
    using TI = std::size_t;
    namespace hyperdrone = rl_tools::rl::environments::hyperdrone;

    struct SPEC: hyperdrone::presets::X500FPVIMU<T, TI> {
        using CAMERA_OUTPUTS = aerovio_camera_config::CAMERA_OUTPUTS;
        using OBSERVATION = aerovio_camera_config::OBSERVATION;
        static constexpr TI INSTANCES_PER_WORLD = 1;
        static constexpr TI N_CAMERA_AGENTS = 1;
        static constexpr TI MAX_DRONES = 1;
        static constexpr TI CAM_WIDTH = 640;
        static constexpr TI CAM_HEIGHT = 480;
        using SHADING = rl_tools::hyperlux::High;
        static constexpr bool ENABLE_ANTI_ALIASING = false;
        static constexpr TI ANTI_ALIASING_GRID_SIZE = 1;
        static constexpr bool ENABLE_MOTION_BLUR = false;  // exposure is integrated over rendered sub-frames in Python
        static constexpr TI MOTION_BLUR_SAMPLES = 1;
        static constexpr bool SELF_VISIBLE = false;  // like the C++ visual-inertial localization harness
        // the horizontal FOV is drawn per episode from nominal + U(-range, range) degrees;
        // Python sets the nominal through constructor parameters (--fov); the exact value is reported in the episode config
        static constexpr T CAMERA_FOV_RANDOMIZATION_RANGE = 15.0;
        static constexpr T CAMERA_MOUNT_ROTATION_RANDOMIZATION_RANGE = 0;
        static constexpr T CAMERA_MOUNT_OFFSET_RANDOMIZATION_RANGE = 0;
        static constexpr T BRIGHTNESS_RANDOMIZATION_RANGE = 0;  // photometry is handled by the ISP stage
    };
    using BASE_WORLD = hyperdrone::World<SPEC>;

    struct TASK_SPEC: hyperdrone::tasks::visual_inertial_localization::Specification<BASE_WORLD> {
        static constexpr TI FRAME_STRIDE = 10;  // training captures at 20 Hz; IMU remains 200 Hz
        static constexpr TI NUM_WAYPOINTS = 48;  // route length (the route cycles when exhausted); scaled with the duration
        static constexpr TI INITIALIZATION_HOLD_STEPS = 200;  // hover before the route starts
    };
    using WORLD = hyperdrone::tasks::visual_inertial_localization::World<TASK_SPEC>;
}
