/* FAST corner detector algorithm testing.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <catch2/catch.hpp>

#include "cvlib.hpp"

#include <algorithm>
#include <tuple>

using namespace cvlib;

namespace
{
const int default_threshold = 20;
const cv::Size image_size(60, 60);
const cv::Rect square(20, 20, 20, 20);
const uchar dark = 50;
const uchar bright = 200;
const int border = 3; // радиус окружности FAST, ближе к краю она не помещается в изображение

cv::Mat make_square(uchar background, uchar foreground, const cv::Rect& rect = square)
{
    cv::Mat image(image_size, CV_8UC1, cv::Scalar(background));
    image(rect).setTo(foreground);
    return image;
}

std::vector<cv::Point> square_vertices(const cv::Rect& rect = square)
{
    return {rect.tl(), {rect.br().x - 1, rect.y}, {rect.x, rect.br().y - 1}, rect.br() - cv::Point(1, 1)};
}

std::vector<cv::Point> detect_points(const cv::Mat& image, int threshold = default_threshold, const cv::Mat& mask = cv::Mat())
{
    auto fast = corner_detector_fast::create();
    fast->set_threshold(threshold);
    std::vector<cv::KeyPoint> keypoints;
    fast->detect(image, keypoints, mask);

    std::vector<cv::Point> points;
    for (const auto& keypoint : keypoints)
        points.emplace_back(keypoint.pt);
    return points;
}

bool same_points(std::vector<cv::Point> lhs, std::vector<cv::Point> rhs)
{
    const auto less = [](const cv::Point& a, const cv::Point& b) { return std::tie(a.y, a.x) < std::tie(b.y, b.x); };
    std::sort(lhs.begin(), lhs.end(), less);
    std::sort(rhs.begin(), rhs.end(), less);
    return lhs == rhs;
}
} // namespace

TEST_CASE("simple check", "[corner_detector_fast]")
{
    auto fast = corner_detector_fast::create();
    // без инициализации в изображении остаётся мусор из памяти, и в нём могут найтись углы
    cv::Mat image = cv::Mat::zeros(10, 10, CV_8UC1);
    SECTION("empty image")
    {
        std::vector<cv::KeyPoint> out;
        fast->detect(image, out);
        REQUIRE(out.empty());
    }

    SECTION("flat image")
    {
        REQUIRE(detect_points(cv::Mat(image_size, CV_8UC1, cv::Scalar(dark))).empty());
    }

    SECTION("bright square on dark background")
    {
        // без подавления немаксимумов у каждой вершины нашлось бы по 6 точек
        REQUIRE(same_points(square_vertices(), detect_points(make_square(dark, bright))));
    }

    SECTION("dark square on bright background")
    {
        REQUIRE(same_points(square_vertices(), detect_points(make_square(bright, dark))));
    }

    SECTION("straight edge is not a corner")
    {
        // по одну сторону границы лежат только 7 точек окружности из 16, а нужно 9 подряд
        const cv::Rect right_half(image_size.width / 2, 0, image_size.width / 2, image_size.height);
        REQUIRE(detect_points(make_square(dark, bright, right_half)).empty());
    }

    SECTION("threshold")
    {
        const uchar low_contrast = dark + 10;
        const int sensitive_threshold = 5;
        const auto image = make_square(dark, low_contrast);
        REQUIRE(detect_points(image, default_threshold).empty());
        REQUIRE(same_points(square_vertices(), detect_points(image, sensitive_threshold)));
    }

    SECTION("mask")
    {
        cv::Mat left_half = cv::Mat::zeros(image_size, CV_8UC1);
        left_half(cv::Rect(0, 0, image_size.width / 2, image_size.height)).setTo(255);
        const std::vector<cv::Point> left_vertices = {square.tl(), {square.x, square.br().y - 1}};
        REQUIRE(same_points(left_vertices, detect_points(make_square(dark, bright), default_threshold, left_half)));
    }

    SECTION("corner near image border")
    {
        const cv::Rect touching_border(0, 0, 20, 20);
        const auto points = detect_points(make_square(dark, bright, touching_border));
        const cv::Rect allowed(border, border, image_size.width - 2 * border, image_size.height - 2 * border);
        REQUIRE(same_points({{19, 19}}, points));
        REQUIRE(std::all_of(points.begin(), points.end(), [&](const cv::Point& p) { return allowed.contains(p); }));
    }

    SECTION("color image")
    {
        const auto gray = make_square(dark, bright);
        cv::Mat color;
        cv::cvtColor(gray, color, cv::COLOR_GRAY2BGR);
        REQUIRE(same_points(detect_points(gray), detect_points(color)));
    }
}
