#pragma once

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace visual_direction {

// Evaluate each half with a three-row ring of Sobel products. Using integer
// products avoids intermediate float scaling. The score's constant scale
// cancels when ranking corners and applying a relative response threshold.
template<int width>
class SplitCorners {
public:
    void compute_half(const cv::Mat& image, int half) {
        response_.create(image.size(), CV_32FC1);
        const int middle = image.rows / 2;
        compute_rows(image, half == 0 ? 0 : middle, half == 0 ? middle : image.rows);
    }

    void select(std::vector<cv::Point2f>& points, int maximum, int spacing) {
        double maximum_response = 0.;
        cv::minMaxLoc(response_, nullptr, &maximum_response);
        const float threshold = static_cast<float>(maximum_response * .018);
        cv::dilate(response_, dilated_, cv::Mat());

        struct Corner { float score; int x, y; };
        std::vector<Corner> candidates;
        for (int y = 1; y + 1 < response_.rows; ++y) {
            const auto* response = response_.ptr<float>(y);
            const auto* local_max = dilated_.ptr<float>(y);
            for (int x = 1; x + 1 < response_.cols; ++x) {
                if (response[x] > threshold && response[x] == local_max[x])
                    candidates.push_back({response[x], x, y});
            }
        }
        std::sort(candidates.begin(), candidates.end(), [](const Corner& a, const Corner& b) {
            if (a.score != b.score) return a.score > b.score;
            if (a.y != b.y) return a.y > b.y;
            return a.x > b.x;
        });

        blocked_.create(response_.size(), CV_8UC1);
        blocked_.setTo(0);
        points.clear();
        points.reserve(maximum);
        const int distance_squared = spacing * spacing;
        for (const auto& corner : candidates) {
            if (blocked_.ptr<uint8_t>(corner.y)[corner.x]) continue;
            points.emplace_back(static_cast<float>(corner.x), static_cast<float>(corner.y));
            if (static_cast<int>(points.size()) >= maximum) break;
            const int y0 = std::max(0, corner.y - spacing + 1);
            const int y1 = std::min(response_.rows, corner.y + spacing);
            const int x0 = std::max(0, corner.x - spacing + 1);
            const int x1 = std::min(response_.cols, corner.x + spacing);
            for (int y = y0; y < y1; ++y) {
                auto* blocked = blocked_.ptr<uint8_t>(y);
                const int dy = y - corner.y;
                for (int x = x0; x < x1; ++x) {
                    const int dx = x - corner.x;
                    if (dx * dx + dy * dy < distance_squared) blocked[x] = 1;
                }
            }
        }
    }

private:
    struct Products {
        std::array<int, width + 2> xx, xy, yy;
    };

    static void sobel_products(const cv::Mat& image, int y, Products& row) {
        const auto* top = image.ptr<uint8_t>(y == 0 ? 1 : y - 1);
        const auto* centre = image.ptr<uint8_t>(y);
        const auto* bottom = image.ptr<uint8_t>(y == image.rows - 1 ? y - 1 : y + 1);
        for (int x = 1; x < width - 1; ++x) {
            const int dx = top[x + 1] - top[x - 1] + 2 * (centre[x + 1] - centre[x - 1])
                         + bottom[x + 1] - bottom[x - 1];
            const int dy = bottom[x - 1] + 2 * bottom[x] + bottom[x + 1]
                         - top[x - 1] - 2 * top[x] - top[x + 1];
            row.xx[x + 1] = dx * dx;
            row.xy[x + 1] = dx * dy;
            row.yy[x + 1] = dy * dy;
        }
        for (const int x : {0, width - 1}) {
            const int neighbor = x == 0 ? 1 : width - 2;
            const int dy = 2 * (bottom[neighbor] - top[neighbor] + bottom[x] - top[x]);
            row.xx[x + 1] = row.xy[x + 1] = 0;
            row.yy[x + 1] = dy * dy;
        }
        for (auto* product : {&row.xx, &row.xy, &row.yy}) {
            (*product)[0] = (*product)[2];
            (*product)[width + 1] = (*product)[width - 1];
        }
    }

    static int sum3(const std::array<int, width + 2>& row, int x) {
        return row[x] + row[x + 1] + row[x + 2];
    }

    void compute_rows(const cv::Mat& image, int begin, int end) {
        std::array<Products, 3> products;
        Products* top = &products[0];
        Products* centre = &products[1];
        Products* bottom = &products[2];
        sobel_products(image, begin == 0 ? 1 : begin - 1, *top);
        sobel_products(image, begin, *centre);
        for (int y = begin; y < end; ++y) {
            sobel_products(image, y == image.rows - 1 ? y - 1 : y + 1, *bottom);
            float* response = response_.ptr<float>(y);
            for (int x = 0; x < width; ++x) {
                const float a = .5f * (sum3(top->xx, x) + sum3(centre->xx, x) + sum3(bottom->xx, x));
                const float b = sum3(top->xy, x) + sum3(centre->xy, x) + sum3(bottom->xy, x);
                const float c = .5f * (sum3(top->yy, x) + sum3(centre->yy, x) + sum3(bottom->yy, x));
                const float diff = a - c;
                response[x] = (a + c) - std::sqrt(diff * diff + b * b);
            }
            auto* spare = top;
            top = centre;
            centre = bottom;
            bottom = spare;
        }
    }

    cv::Mat response_, dilated_, blocked_;
};

} // namespace visual_direction
