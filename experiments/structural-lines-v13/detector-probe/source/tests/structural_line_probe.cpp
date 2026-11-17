#include "viobench_abi.h"
#include "structural_lines.h"
#include "structural_detector_cases.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>

void* alloc(uint32_t size) { return std::malloc(size ? size : 1); }

// out[128]: 8 summary values, 24x4 retained pixel endpoints, 3x8 directions:
// [axis_x,axis_y,axis_z,support,residual_rad,conditioning,center_span_px,mask].
// This standalone diagnostic is not a VIO estimator and does not read truth.
extern "C" VIOBENCH_EXPORT(probe) int probe(const uint8_t* gray, double fx, double fy,
                                           double cx, double cy, double* out) {
    try {
        static structural_lines::Detector detector;
        std::fill_n(out, 128, 0.);
        const auto result = detector.detect(cv::Mat(120, 160, CV_8UC1, const_cast<uint8_t*>(gray)), fx, fy, cx, cy);
        out[0] = result.valid_image;
        out[1] = result.raw_lines;
        out[2] = result.lines.size();
        out[3] = result.directions.size();
        out[4] = result.rejected_short;
        out[5] = result.rejected_duplicate;
        out[6] = result.tested_pairs;
        out[7] = result.rejected_geometry;
        for (size_t i = 0; i < result.lines.size(); ++i) {
            const auto& l = result.lines[i];
            out[8 + 4*i] = l.first.x; out[9 + 4*i] = l.first.y;
            out[10 + 4*i] = l.last.x; out[11 + 4*i] = l.last.y;
        }
        for (size_t i = 0; i < result.directions.size(); ++i) {
            const auto& d = result.directions[i]; const size_t offset = 104 + 8*i;
            for (int j = 0; j < 3; ++j) out[offset+j] = d.axis[j];
            out[offset+3] = d.support; out[offset+4] = d.residual;
            out[offset+5] = d.conditioning; out[offset+6] = d.center_span_px;
            out[offset+7] = d.support_mask;
        }
        return result.valid_image ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "structural probe: %s\n", error.what());
        return 2;
    }
}

extern "C" VIOBENCH_EXPORT(run_tests) int run_tests() { return detector_cases(); }
