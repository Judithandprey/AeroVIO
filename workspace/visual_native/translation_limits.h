#pragma once
// Stage 2 anti-runaway experiments on the learned per-frame translation.
// Position speed comes only from the MLP increment; the visual tracker redirects
// that increment but keeps its length. Each limit is compiled in only when its
// CMake option is ON (LEO_TRANSLATION_LIMITS is set when any of them is).
#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>

namespace translation_limits {

// t: learned camera-frame translation of one frame interval dt.
// cycles_without_fit: visual cycles since the last direct heading fit.
inline cv::Vec3d limit(cv::Vec3d t, double dt, int cycles_without_fit) {
    (void)dt; (void)cycles_without_fit;
#if LEO_OUTPUT_SCALE
    t *= double(LEO_OUTPUT_SCALE_FACTOR);
#endif
#if LEO_VISUAL_DAMPING
    // Shrink the increment geometrically once the tracker has missed more than
    // LEO_DAMPING_GRACE cycles in a row; one direct fit restores full gain.
    if (cycles_without_fit > LEO_DAMPING_GRACE) {
        const double gain = std::pow(double(LEO_DAMPING_FACTOR), cycles_without_fit - LEO_DAMPING_GRACE);
        t *= std::max(gain, double(LEO_DAMPING_MIN));
    }
#endif
#if LEO_SPEED_CAP
    const double length = cv::norm(t), cap = double(LEO_SPEED_CAP_MPS) * dt;
    if (length > cap) t *= cap / length;
#endif
    return t;
}

} // namespace translation_limits
