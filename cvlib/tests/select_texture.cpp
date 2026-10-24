/* Тестирование текстурной сегментации.
 */

#include <catch2/catch.hpp>

#include "cvlib.hpp"

#include <cmath>

using namespace cvlib;

namespace
{
const cv::Size half_size(128, 128);
const cv::Rect sample(32, 48, 32, 32); // внутри левой половины
// области достаточно далеко от краёв и от границы между половинами,
// где ни окно, ни ядра фильтров не достают до другой текстуры
const cv::Rect left_inner(16, 16, 48, 96);
const cv::Rect right_inner(192, 16, 48, 96);
// при таком контрасте расстояние внутри одной текстуры меньше 0.25, между разными - больше 20
const double eps = 10;

const double default_mean = 128;
const double contrast = 50;

/// \brief Строит синусоидальные полосы: вертикальные меняются вдоль x, горизонтальные - вдоль y
cv::Mat make_stripes(double period, bool vertical, double mean = default_mean)
{
    cv::Mat image(half_size, CV_8UC1);
    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            const int coord = vertical ? x : y;
            image.at<uchar>(y, x) = cv::saturate_cast<uchar>(mean + contrast * std::cos(2 * CV_PI * coord / period));
        }
    }
    return image;
}

cv::Mat join(const cv::Mat& left, const cv::Mat& right)
{
    cv::Mat image;
    cv::hconcat(left, right, image);
    return image;
}

bool is_selected(const cv::Mat& mask, const cv::Rect& region)
{
    return cv::countNonZero(mask(region)) == region.area();
}

bool is_rejected(const cv::Mat& mask, const cv::Rect& region)
{
    return cv::countNonZero(mask(region)) == 0;
}
} // namespace

TEST_CASE("mask format", "[select_texture]")
{
    const auto image = join(make_stripes(8, true), make_stripes(8, false));
    const auto mask = select_texture(image, sample, eps);

    REQUIRE(image.size() == mask.size());
    REQUIRE(CV_8UC1 == mask.type());
    REQUIRE(0 == cv::countNonZero(mask != 0 & mask != 255));
}

TEST_CASE("different textures", "[select_texture]")
{
    SECTION("orientation")
    {
        const auto mask = select_texture(join(make_stripes(8, true), make_stripes(8, false)), sample, eps);
        REQUIRE(is_selected(mask, left_inner));
        REQUIRE(is_rejected(mask, right_inner));
    }

    SECTION("frequency")
    {
        const auto mask = select_texture(join(make_stripes(4, true), make_stripes(16, true)), sample, eps);
        REQUIRE(is_selected(mask, left_inner));
        REQUIRE(is_rejected(mask, right_inner));
    }
}

TEST_CASE("brightness independence", "[select_texture]")
{
    SECTION("same texture")
    {
        const double dark = 70;
        const double bright = 180;
        const auto mask = select_texture(join(make_stripes(8, true, dark), make_stripes(8, true, bright)), sample, eps);
        REQUIRE(is_selected(mask, left_inner));
        REQUIRE(is_selected(mask, right_inner));
    }

    SECTION("flat regions")
    {
        const cv::Mat dark(half_size, CV_8UC1, cv::Scalar(50));
        const cv::Mat bright(half_size, CV_8UC1, cv::Scalar(200));
        const auto mask = select_texture(join(dark, bright), sample, eps);
        REQUIRE(is_selected(mask, left_inner));
        REQUIRE(is_selected(mask, right_inner));
    }
}

TEST_CASE("invalid roi", "[select_texture]")
{
    const auto image = make_stripes(8, true);

    SECTION("empty")
    {
        REQUIRE_THROWS_AS(select_texture(image, cv::Rect(10, 10, 0, 0), eps), cv::Exception);
    }

    SECTION("outside of the image")
    {
        REQUIRE_THROWS_AS(select_texture(image, cv::Rect(100, 100, 50, 50), eps), cv::Exception);
    }
}
