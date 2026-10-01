/* Split and merge segmentation algorithm testing.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <catch2/catch.hpp>

#include "cvlib.hpp"

using namespace cvlib;

namespace
{
const int block_size = 10;
const uchar dark = 50;
const uchar bright = 150;
const uchar texture = 2;

bool is_equal(const cv::Mat& lhs, const cv::Mat& rhs)
{
    return lhs.size() == rhs.size() && lhs.type() == rhs.type() && 0 == cv::countNonZero(lhs != rhs);
}

/// \brief Builds grid x grid chessboard of blocks, the right half of every block is brighter by shift
cv::Mat make_blocks(int grid, uchar shift)
{
    cv::Mat image(grid * block_size, grid * block_size, CV_8UC1);
    for (int y = 0; y < image.rows; ++y)
    {
        for (int x = 0; x < image.cols; ++x)
        {
            const bool is_dark = (y / block_size + x / block_size) % 2 == 0;
            const bool is_right_half = x % block_size >= block_size / 2;
            image.at<uchar>(y, x) = static_cast<uchar>((is_dark ? dark : bright) + (is_right_half ? shift : 0));
        }
    }
    return image;
}

/// \brief Blocks are not aligned with the split grid, so merge has to collect every block from pieces
void check_blocks(int grid)
{
    const double stddev = 1.5; // above the texture stddev inside a block (1.0), far below the contrast between blocks
    const cv::Mat expected = make_blocks(grid, 0) + texture / 2;

    REQUIRE(is_equal(expected, split_and_merge(make_blocks(grid, texture), stddev)));
}
} // namespace

TEST_CASE("constant image", "[split_and_merge]")
{
    const cv::Mat image(100, 100, CV_8UC1, cv::Scalar{15});

    const auto res = split_and_merge(image, 1);
    REQUIRE(image.size() == res.size());
    REQUIRE(image.type() == res.type());
    REQUIRE(cv::Scalar(15) == cv::mean(res));
}

TEST_CASE("simple regions", "[split_and_merge]")
{
    SECTION("2x2")
    {
        const cv::Mat reference = (cv::Mat_<char>(2, 2) << 2, 2, 2, 2);
        cv::Mat image = (cv::Mat_<char>(2, 2) << 0, 1, 2, 3);
        auto res = split_and_merge(image, 10);
        REQUIRE(image.size() == res.size());
        REQUIRE(image.type() == res.type());
        REQUIRE(0 == cv::countNonZero(reference - res));

        // split gives four pixels, then 0, 1 and 2 are merged (stddev 0.82), adding 3 would give stddev 1.12
        const cv::Mat merged = (cv::Mat_<char>(2, 2) << 1, 1, 1, 3);
        res = split_and_merge(image, 1);
        REQUIRE(is_equal(merged, res));
    }

    SECTION("3x3")
    {
        const cv::Mat image = (cv::Mat_<uchar>(3, 3) << 10, 10, 200, 10, 10, 200, 10, 10, 200);
        REQUIRE(is_equal(image, split_and_merge(image, 1)));
    }

    SECTION("1xN")
    {
        const cv::Mat image = (cv::Mat_<uchar>(1, 5) << 0, 0, 0, 100, 100);
        REQUIRE(is_equal(image, split_and_merge(image, 1)));
    }

    SECTION("input is not modified")
    {
        const cv::Mat image = (cv::Mat_<uchar>(2, 2) << 0, 1, 2, 3);
        const cv::Mat copy = image.clone();
        split_and_merge(image, 10);
        REQUIRE(is_equal(copy, image));
    }

    SECTION("zero threshold keeps image")
    {
        const uint64 seed = 42;
        cv::Mat image(16, 16, CV_8UC1);
        cv::RNG(seed).fill(image, cv::RNG::UNIFORM, 0, 256);
        REQUIRE(is_equal(image, split_and_merge(image, 0)));
    }
}

TEST_CASE("compex regions", "[split_and_merge]")
{
    SECTION("regions from different split branches")
    {
        // quadrants 10 and 12 are separate after split, merge joins them into one region with mean 11
        const cv::Mat image = (cv::Mat_<uchar>(4, 4) << 10, 10, 12, 12, 10, 10, 12, 12, 100, 100, 100, 100, 100, 100, 100, 100);
        const cv::Mat expected = (cv::Mat_<uchar>(4, 4) << 11, 11, 11, 11, 11, 11, 11, 11, 100, 100, 100, 100, 100, 100, 100, 100);
        REQUIRE(is_equal(expected, split_and_merge(image, 2)));
    }

    SECTION("2x2")
    {
        check_blocks(2);
    }

    SECTION("3x3")
    {
        check_blocks(3);
    }

    SECTION("4x4")
    {
        check_blocks(4);
    }
}
