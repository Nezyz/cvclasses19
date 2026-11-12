/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-18
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <cmath>
#include <vector>

namespace
{
const int orientations_count = 4; // амплитуда для theta и theta + pi одинакова, поэтому достаточно [0, pi)
const double wavelengths[] = {4.0, 8.0, 16.0}; // шаг между масштабами в одну октаву
const double sigma_to_wavelength = 0.56; // даёт полосу пропускания в одну октаву, соседние масштабы почти не перекрываются
const double aspect_ratio = 0.5;
const double kernel_radius_in_sigmas = 3.0; // дальше гауссова огибающая мала

/// \brief Чётная (косинус) и нечётная (синус) части фильтра Габора, образующие квадратурную пару
struct gabor_pair
{
    cv::Mat even;
    cv::Mat odd;
};

int odd_size(double radius)
{
    return 2 * static_cast<int>(std::ceil(radius)) + 1;
}

gabor_pair make_gabor_pair(double wavelength, double orientation)
{
    const double sigma = sigma_to_wavelength * wavelength;
    const int size = odd_size(kernel_radius_in_sigmas * sigma);
    const cv::Size ksize(size, size);

    cv::Mat even = cv::getGaborKernel(ksize, sigma, orientation, wavelength, aspect_ratio, 0, CV_32F);
    const cv::Mat odd = cv::getGaborKernel(ksize, sigma, orientation, wavelength, aspect_ratio, CV_PI / 2, CV_32F);

    // нулевое среднее даёт нулевой отклик на однотонных областях независимо от их яркости,
    // нечётная часть антисимметрична, поэтому её сумма уже равна нулю
    even -= cv::mean(even);

    // обе части масштабируются одним множителем, чтобы амплитуда пары сохраняла смысл,
    // после этого амплитуда отклика 8-битного изображения не превышает 255
    cv::Mat magnitude;
    cv::magnitude(even, odd, magnitude);
    const double scale = 1.0 / cv::sum(magnitude)[0];

    return {even * scale, odd * scale};
}

std::vector<gabor_pair> make_gabor_bank()
{
    std::vector<gabor_pair> bank;
    for (const double wavelength : wavelengths)
        for (int i = 0; i < orientations_count; ++i)
            bank.push_back(make_gabor_pair(wavelength, CV_PI * i / orientations_count));
    return bank;
}

/// \brief Локальная энергия текстуры на частоте и ориентации фильтра,
///        в отличие от сырого отклика она не колеблется вдоль полос
cv::Mat gabor_amplitude(const cv::Mat& image, const gabor_pair& filter)
{
    cv::Mat even;
    cv::Mat odd;
    cv::filter2D(image, even, CV_32F, filter.even);
    cv::filter2D(image, odd, CV_32F, filter.odd);

    cv::Mat amplitude;
    cv::magnitude(even, odd, amplitude);
    return amplitude;
}

/// \brief Добавляет среднее и стандартное отклонение амплитуды в окне с центром в каждом пикселе
void append_local_stats(const cv::Mat& amplitude, const cv::Size& window, std::vector<cv::Mat>& features)
{
    cv::Mat mean;
    cv::Mat mean_sq;
    cv::blur(amplitude, mean, window);
    cv::blur(amplitude.mul(amplitude), mean_sq, window);

    // из-за погрешности вычислений с плавающей точкой дисперсия может получиться отрицательной
    const cv::Mat variance = cv::max(mean_sq - mean.mul(mean), 0.0);
    cv::Mat stddev;
    cv::sqrt(variance, stddev);

    features.push_back(mean);
    features.push_back(stddev);
}

/// \brief L2-расстояние от дескриптора каждого пикселя до дескриптора опорного пикселя
cv::Mat distance_to(const std::vector<cv::Mat>& features, const cv::Point& reference)
{
    cv::Mat sum_sq = cv::Mat::zeros(features.front().size(), CV_32FC1);
    for (const auto& feature : features)
    {
        const cv::Mat diff = feature - feature.at<float>(reference);
        sum_sq += diff.mul(diff);
    }

    cv::Mat distance;
    cv::sqrt(sum_sq, distance);
    return distance;
}
} // namespace

namespace cvlib
{
cv::Mat select_texture(const cv::Mat& image, const cv::Rect& roi, double eps)
{
    CV_Assert(image.channels() == 1);
    // образец текстуры должен быть непустой частью изображения
    CV_Assert(roi.area() > 0 && (roi & cv::Rect(cv::Point(), image.size())) == roi);

    // дескриптор каждого пикселя считается по окну размером с ROI с центром в этом пикселе,
    // поэтому дескриптор центра ROI в точности совпадает с дескриптором образца
    std::vector<cv::Mat> features;
    for (const auto& filter : make_gabor_bank())
        append_local_stats(gabor_amplitude(image, filter), roi.size(), features);

    const cv::Point roi_center = roi.tl() + cv::Point(roi.width / 2, roi.height / 2);
    return distance_to(features, roi_center) <= eps;
}
} // namespace cvlib
