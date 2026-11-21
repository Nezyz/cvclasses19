/* Тестирование вычитания фона.
 * @file
 * @date 2018-09-18
 * @author Anonymous
 */

#include <catch2/catch.hpp>

#include "cvlib.hpp"

using namespace cvlib;

namespace
{
const cv::Size frame_size(64, 64);
const cv::Rect object(16, 16, 16, 16);
const double background_level = 100;
const double object_level = 200;
// разница с фоном 100 уровней, а сигма модели после обучения не больше нескольких уровней
const int warmup_frames = 30;
const double noise_stddev = 2; // типичный шум камеры, заметно меньше минимальной сигмы модели (4)

cv::Mat make_frame(double level, cv::RNG& rng, double noise = noise_stddev)
{
    cv::Mat frame(frame_size, CV_32FC1);
    rng.fill(frame, cv::RNG::NORMAL, level, noise);
    frame.convertTo(frame, CV_8U);
    return frame;
}

cv::Mat with_object(cv::Mat frame)
{
    frame(object).setTo(object_level);
    return frame;
}

void warm_up(motion_segmentation& mseg, cv::RNG& rng)
{
    cv::Mat mask;
    for (int i = 0; i < warmup_frames; ++i)
        mseg.apply(make_frame(background_level, rng), mask);
}

bool is_foreground(const cv::Mat& mask, const cv::Rect& region)
{
    return cv::countNonZero(mask(region)) == region.area();
}

bool is_background(const cv::Mat& mask)
{
    return cv::countNonZero(mask) == 0;
}

/// \brief Маска вне объекта
cv::Mat outside_object(const cv::Mat& mask)
{
    cv::Mat outside = mask.clone();
    outside(object).setTo(0);
    return outside;
}
} // namespace

TEST_CASE("foreground mask format", "[motion_segmentation]")
{
    cv::RNG rng;
    motion_segmentation mseg;
    warm_up(mseg, rng);

    cv::Mat mask;
    mseg.apply(with_object(make_frame(background_level, rng)), mask);

    REQUIRE(frame_size == mask.size());
    REQUIRE(CV_8UC1 == mask.type());
    REQUIRE(0 == cv::countNonZero(mask != 0 & mask != 255));
}

TEST_CASE("static scene", "[motion_segmentation]")
{
    cv::RNG rng;
    motion_segmentation mseg;
    cv::Mat mask;

    SECTION("first frame is background")
    {
        mseg.apply(make_frame(background_level, rng), mask);
        REQUIRE(is_background(mask));
    }

    SECTION("camera noise is background")
    {
        warm_up(mseg, rng);
        mseg.apply(make_frame(background_level, rng), mask);
        REQUIRE(is_background(mask));
    }
}

TEST_CASE("moving object", "[motion_segmentation]")
{
    cv::RNG rng;
    motion_segmentation mseg;
    warm_up(mseg, rng);
    cv::Mat mask;

    SECTION("object is detected")
    {
        mseg.apply(with_object(make_frame(background_level, rng)), mask);
        REQUIRE(is_foreground(mask, object));
        REQUIRE(is_background(outside_object(mask)));
    }

    SECTION("stopped object is not absorbed into background")
    {
        const int long_stay = 300; // втрое больше постоянной времени обучения
        for (int i = 0; i < long_stay; ++i)
            mseg.apply(with_object(make_frame(background_level, rng)), mask);
        REQUIRE(is_foreground(mask, object));
    }
}

TEST_CASE("learning rate", "[motion_segmentation]")
{
    cv::RNG rng;
    motion_segmentation mseg;
    warm_up(mseg, rng);
    cv::Mat mask;

    SECTION("zero keeps the model")
    {
        cv::Mat before;
        mseg.getBackgroundImage(before);

        // сдвиг меньше порога, поэтому пиксели считаются фоном и при обычном обучении модель бы его выучила
        const double small_shift = 5;
        const int frames = 100;
        for (int i = 0; i < frames; ++i)
            mseg.apply(make_frame(background_level + small_shift, rng), mask, 0);

        cv::Mat after;
        mseg.getBackgroundImage(after);
        REQUIRE(0 == cv::norm(before, after, cv::NORM_INF));
    }

    SECTION("one reinitializes the model")
    {
        mseg.apply(with_object(make_frame(background_level, rng)), mask, 1);
        mseg.apply(with_object(make_frame(background_level, rng)), mask);
        REQUIRE(is_background(mask));
    }
}

TEST_CASE("slow illumination change", "[motion_segmentation]")
{
    // без шума, чтобы проверять только отставание модели: при скорости 1 уровень за 40 кадров
    // оно около 2.5 уровней, а порог не меньше 2.5 * 4 = 10 уровней
    const double noiseless = 0;
    const int frames_per_level = 40;
    const int levels = 20;

    cv::RNG rng;
    cv::Mat mask;

    SECTION("slow change is background")
    {
        motion_segmentation mseg;
        for (int i = 0; i <= levels * frames_per_level; ++i)
            mseg.apply(make_frame(background_level + i / frames_per_level, rng, noiseless), mask);
        REQUIRE(is_background(mask));
    }

    SECTION("same change at once is foreground")
    {
        motion_segmentation mseg;
        for (int i = 0; i < warmup_frames; ++i)
            mseg.apply(make_frame(background_level, rng, noiseless), mask);
        mseg.apply(make_frame(background_level + levels, rng, noiseless), mask);
        REQUIRE(is_foreground(mask, cv::Rect(cv::Point(), frame_size)));
    }
}

TEST_CASE("color frame", "[motion_segmentation]")
{
    const cv::Scalar gray(background_level, background_level, background_level);
    const cv::Scalar red(background_level, background_level, object_level);
    const cv::Mat background(frame_size, CV_8UC3, gray);
    cv::Mat frame = background.clone();
    frame(object).setTo(red);

    motion_segmentation mseg;
    cv::Mat mask;
    for (int i = 0; i < warmup_frames; ++i)
        mseg.apply(background, mask);
    mseg.apply(frame, mask);

    REQUIRE(CV_8UC1 == mask.type());
    REQUIRE(is_foreground(mask, object));
    REQUIRE(is_background(outside_object(mask)));

    cv::Mat model;
    mseg.getBackgroundImage(model);
    REQUIRE(CV_8UC3 == model.type());
    REQUIRE(0 == cv::norm(model, background, cv::NORM_INF));
}

TEST_CASE("known foreground", "[motion_segmentation]")
{
    cv::RNG rng;
    motion_segmentation mseg;
    warm_up(mseg, rng);

    cv::Mat known = cv::Mat::zeros(frame_size, CV_8UC1);
    known(object).setTo(255);

    cv::Mat mask;
    mseg.apply(make_frame(background_level, rng), known, mask);
    REQUIRE(is_foreground(mask, object));
    REQUIRE(is_background(outside_object(mask)));
}

TEST_CASE("invalid frame", "[motion_segmentation]")
{
    motion_segmentation mseg;
    cv::Mat mask;
    const cv::Mat float_frame(frame_size, CV_32FC1, cv::Scalar(background_level));
    REQUIRE_THROWS_AS(mseg.apply(float_frame, mask), cv::Exception);
}
