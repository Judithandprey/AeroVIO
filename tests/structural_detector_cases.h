#pragma once
#include "structural_lines.h"
#include <cstdio>
#include <limits>

inline int detector_cases() {
    using namespace structural_lines;
    int failed = 0;
    auto check = [&](bool ok, const char* label) {
        std::printf("%s detector: %s\n", ok ? "PASS" : "FAIL", label);
        failed += !ok;
    };
    Detector detector;
    const double fx = 90., fy = 90., cx = 79.5, cy = 59.5;
    auto contains = [](const Detection& d, const Vec3d& expected, double angle) {
        for (const auto& direction : d.directions)
            if (std::abs(direction.axis.dot(canonical(expected))) >= std::cos(angle)) return true;
        return false;
    };
    auto draw = [](const std::vector<cv::Vec4f>& segments) {
        cv::Mat image(120, 160, CV_8UC1, cv::Scalar(20));
        for (const auto& s : segments)
            cv::line(image, {int(std::round(s[0])), int(std::round(s[1]))},
                     {int(std::round(s[2])), int(std::round(s[3]))}, cv::Scalar(235), 1, cv::LINE_8);
        return image;
    };
    auto blank = detector.detect(cv::Mat(120, 160, CV_8UC1, cv::Scalar(40)), fx, fy, cx, cy);
    check(blank.valid_image && blank.lines.empty() && blank.directions.empty(), "blank image has no evidence");
    auto invalid = detector.detect(cv::Mat(60, 80, CV_8UC1, cv::Scalar(40)), fx, fy, cx, cy);
    check(!invalid.valid_image, "incorrect resolution rejects");
    auto one = detector.detect(draw({{20.f, 12.f, 20.f, 108.f}}), fx, fy, cx, cy);
    check(one.directions.empty(), "single stripe cannot invent a vanishing family");

    const std::vector<cv::Vec4f> parallel{{15.f, 10.f, 15.f, 110.f}, {50.f, 10.f, 50.f, 110.f},
        {95.f, 10.f, 95.f, 110.f}, {140.f, 10.f, 140.f, 110.f}};
    auto exact_parallel = detector.from_segments(parallel, fx, fy, cx, cy);
    check(exact_parallel.directions.size() == 1 && contains(exact_parallel, {0., 1., 0.}, 1e-6),
          "parallel lines recover a direction at infinity without dividing by z");
    auto detected_parallel = detector.detect(draw(parallel), fx, fy, cx, cy);
    check(detected_parallel.directions.size() == 1 && contains(detected_parallel, {0., 1., 0.}, .035),
          "LSD parallel image has exactly one direction, never an invented yaw axis");

    std::vector<cv::Vec4f> perspective;
    const double vpx = 240., vpy = 30.;
    for (double y : {12., 44., 76., 108.}) {
        const double end_y = y + (vpy - y) * (140. - 12.) / (vpx - 12.);
        perspective.push_back({12.f, float(y), 140.f, float(end_y)});
    }
    const Vec3d expected{(vpx - cx) / fx, (vpy - cy) / fy, 1.};
    auto exact_perspective = detector.from_segments(perspective, fx, fy, cx, cy);
    check(contains(exact_perspective, expected, 1e-5), "finite perspective VP uses calibrated camera rays");
    auto detected_perspective = detector.detect(draw(perspective), fx, fy, cx, cy);
    check(contains(detected_perspective, expected, .06), "LSD recovers finite perspective direction");

    std::vector<cv::Vec4f> oblique{{5.f, 12.f, 65.f, 12.f}, {5.f, 48.f, 65.f, 48.f},
        {5.f, 84.f, 65.f, 84.f}, {95.f, 8.f, 135.f, 48.f},
        {95.f, 38.f, 135.f, 78.f}, {95.f, 68.f, 135.f, 108.f}};
    auto exact_oblique = detector.from_segments(oblique, fx, fy, cx, cy);
    check(exact_oblique.directions.size() == 2 && contains(exact_oblique, {1., 0., 0.}, .001) &&
          contains(exact_oblique, {1., 1., 0.}, .001), "two oblique families do not require orthogonal room axes");
    auto detected_oblique = detector.detect(draw(oblique), fx, fy, cx, cy);
    check(contains(detected_oblique, {1., 0., 0.}, .06) && contains(detected_oblique, {1., 1., 0.}, .06),
          "LSD retains two separated oblique line families");

    std::vector<cv::Vec4f> duplicates{{30.f, 10.f, 30.f, 100.f}, {31.f, 10.f, 31.f, 100.f},
        {30.f, 15.f, 30.f, 95.f}, {30.f, 30.f, 30.f, 80.f}};
    auto duplicate_result = detector.from_segments(duplicates, fx, fy, cx, cy);
    check(duplicate_result.lines.size() == 1 && duplicate_result.directions.empty(),
          "duplicate edges and collinear fragments count once");
    auto near_degenerate = detector.from_segments({{50.f, 10.f, 50.f, 100.f},
        {54.f, 10.f, 54.f, 100.f}, {58.f, 10.f, 58.f, 100.f}}, fx, fy, cx, cy);
    check(near_degenerate.directions.empty(), "narrow bundle rejects poorly conditioned VP");
    auto short_lines = detector.from_segments({{10.f, 10.f, 20.f, 10.f}, {30.f, 20.f, 40.f, 30.f},
        {40.f, 50.f, 40.f, 65.f}}, fx, fy, cx, cy);
    check(short_lines.lines.empty() && short_lines.directions.empty(), "short clutter rejects");
    auto bad_focal = detector.from_segments(parallel, 0., fy, cx, cy);
    check(!bad_focal.valid_image && bad_focal.directions.empty(), "invalid calibration rejects");
    std::vector<cv::Vec4f> many;
    for (int y = 0; y < 120; y += 4) many.push_back({1.f, float(y), 158.f, float(y)});
    auto capped = detector.from_segments(many, fx, fy, cx, cy);
    check(capped.lines.size() <= 24 && capped.directions.size() <= 3 && capped.tested_pairs <= 828,
          "line, direction and pair-work caps hold");
    auto repeated = detector.detect(draw(perspective), fx, fy, cx, cy);
    bool same = repeated.lines.size() == detected_perspective.lines.size() &&
                repeated.directions.size() == detected_perspective.directions.size();
    for (size_t i = 0; same && i < repeated.directions.size(); ++i)
        same &= norm3(repeated.directions[i].axis - detected_perspective.directions[i].axis) == 0.;
    check(same, "same input produces deterministic directions");
    return failed;
}
