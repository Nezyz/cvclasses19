/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <algorithm>
#include <ctime>

namespace
{
const int radius = 3;
const int circle_size = 16;
// на прямой границе по одну сторону лежат только 7 точек окружности, поэтому при 9 граница не считается углом
const int arc_length = 9;
// окружность Брезенхэма радиуса 3 по часовой стрелке, начиная сверху
const cv::Point circle[circle_size] = {{0, -3}, {1, -3}, {2, -2}, {3, -1}, {3, 0},  {3, 1},   {2, 2},   {1, 3},
                                       {0, 3},  {-1, 3}, {-2, 2}, {-3, 1}, {-3, 0}, {-3, -1}, {-2, -2}, {-1, -3}};
const int compass[] = {0, 4, 8, 12};
// любые 9 подряд точек из 16 захватывают хотя бы 2 из 4 точек по сторонам света,
// при изменении arc_length это значение нужно пересчитать
const int min_compass_matches = 2;
const float keypoint_size = 2 * radius + 1;

/// \brief Оценка угла для дуги ярче (sign = 1) или темнее (sign = -1) центра:
///        сумма превышений порога по точкам окружности или 0, если 9 подряд точек не набралось
int arc_score(const uchar* center, const int* offsets, int threshold, int sign)
{
    int compass_matches = 0;
    for (const int i : compass)
        compass_matches += sign * (center[offsets[i]] - *center) > threshold;
    if (compass_matches < min_compass_matches)
        return 0;

    int run = 0;
    int score = 0;
    bool is_corner = false;
    // проход длиннее окружности на arc_length - 1, чтобы найти дугу, проходящую через начало
    for (int i = 0; i < circle_size + arc_length - 1; ++i)
    {
        const int excess = sign * (center[offsets[i % circle_size]] - *center) - threshold;
        run = excess > 0 ? run + 1 : 0;
        is_corner |= run >= arc_length;
        if (i < circle_size && excess > 0)
            score += excess;
    }
    return is_corner ? score : 0;
}

cv::Mat corner_scores(const cv::Mat& gray, int threshold)
{
    // смещения точек окружности в памяти считаются один раз, дальше пиксель читается как center[offset]
    int offsets[circle_size];
    for (int i = 0; i < circle_size; ++i)
        offsets[i] = circle[i].y * static_cast<int>(gray.step) + circle[i].x;

    cv::Mat scores = cv::Mat::zeros(gray.size(), CV_32FC1);
    for (int y = radius; y < gray.rows - radius; ++y)
    {
        const uchar* row = gray.ptr(y);
        auto* score_row = scores.ptr<float>(y);
        for (int x = radius; x < gray.cols - radius; ++x)
            score_row[x] = static_cast<float>(std::max(arc_score(row + x, offsets, threshold, 1), arc_score(row + x, offsets, threshold, -1)));
    }
    return scores;
}
} // namespace

namespace cvlib
{
// static
cv::Ptr<corner_detector_fast> corner_detector_fast::create()
{
    return cv::makePtr<corner_detector_fast>();
}

void corner_detector_fast::detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray mask)
{
    keypoints.clear();
    CV_Assert(image.depth() == CV_8U);
    cv::Mat gray = image.getMat();
    if (gray.channels() == 3)
        cv::cvtColor(gray, gray, cv::COLOR_BGR2GRAY);

    // подавление немаксимумов: остаются точки, у которых оценка максимальна в окрестности 3x3
    const cv::Mat scores = corner_scores(gray, threshold_);
    cv::Mat local_max;
    cv::dilate(scores, local_max, cv::Mat());

    std::vector<cv::Point> corners;
    cv::findNonZero(scores > 0 & scores == local_max, corners);
    for (const auto& corner : corners)
        keypoints.emplace_back(corner, keypoint_size, -1.f, scores.at<float>(corner));

    cv::KeyPointsFilter::runByPixelsMask(keypoints, mask.getMat());
}

void corner_detector_fast::compute(cv::InputArray, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    std::srand(unsigned(std::time(0))); // \todo remove me
    // \todo implement any binary descriptor
    const int desc_length = 2;
    descriptors.create(static_cast<int>(keypoints.size()), desc_length, CV_32S);
    auto desc_mat = descriptors.getMat();
    desc_mat.setTo(0);

    int* ptr = reinterpret_cast<int*>(desc_mat.ptr());
    for (const auto& pt : keypoints)
    {
        for (int i = 0; i < desc_length; ++i)
        {
            *ptr = std::rand();
            ++ptr;
        }
    }
}

void corner_detector_fast::detectAndCompute(cv::InputArray, cv::InputArray, std::vector<cv::KeyPoint>&, cv::OutputArray descriptors, bool /*= false*/)
{
    // \todo implement me
}
} // namespace cvlib
