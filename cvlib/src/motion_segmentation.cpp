/* Вычитание фона: одна гауссиана (1G) на каждый пиксель.
 * @file
 * @date 2018-09-18
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <algorithm>

namespace
{
const double initial_variance = 15.0 * 15.0; // широкая гауссиана, пока модель не обучилась
// иначе на неподвижном пикселе дисперсия стремится к нулю и шум камеры становится движением
const double min_variance = 4.0 * 4.0;
const double default_learning_rate = 0.01; // постоянная времени около 100 кадров

cv::Mat mark_known_foreground(cv::Mat mask, cv::InputArray known_foreground)
{
    if (!known_foreground.empty())
        mask.setTo(255, known_foreground);
    return mask;
}
} // namespace

namespace cvlib
{
motion_segmentation::motion_segmentation(double threshold) : threshold_(threshold)
{
}

void motion_segmentation::apply(cv::InputArray _image, cv::OutputArray _fgmask, double learningRate)
{
    apply(_image, cv::noArray(), _fgmask, learningRate);
}

void motion_segmentation::apply(cv::InputArray _image, cv::InputArray _knownForegroundMask, cv::OutputArray _fgmask, double learningRate)
{
    // пороги и начальная дисперсия заданы в уровнях 8-битного изображения
    CV_Assert(_image.depth() == CV_8U);

    cv::Mat frame;
    _image.getMat().convertTo(frame, CV_32F);

    ++frames_count_;
    if (bg_model_.size() != frame.size() || bg_model_.type() != frame.type())
    {
        initialize(frame);
        _fgmask.assign(mark_known_foreground(cv::Mat::zeros(frame.size(), CV_8UC1), _knownForegroundMask));
        return;
    }

    // заранее известные объекты, как и найденные, не обучают модель
    const cv::Mat mask = mark_known_foreground(foreground_mask(frame), _knownForegroundMask);
    const double learning_rate = choose_learning_rate(learningRate);
    if (learning_rate >= 1)
        initialize(frame);
    else
        update(frame, ~mask, learning_rate);

    _fgmask.assign(mask);
}

void motion_segmentation::initialize(const cv::Mat& frame)
{
    bg_model_ = frame.clone();
    variance_ = cv::Mat(frame.size(), frame.type(), cv::Scalar::all(initial_variance));
    frames_count_ = 1;
}

cv::Mat motion_segmentation::foreground_mask(const cv::Mat& frame) const
{
    const cv::Mat diff = frame - bg_model_;
    const cv::Mat normalized_sq = diff.mul(diff) / variance_;

    // среднее по каналам квадратов отклонений в сигмах, чтобы порог не зависел от числа каналов
    const int channels = frame.channels();
    cv::Mat distance_sq;
    cv::transform(normalized_sq, distance_sq, cv::Mat(1, channels, CV_32F, cv::Scalar(1.0 / channels)));

    return distance_sq > threshold_ * threshold_;
}

void motion_segmentation::update(const cv::Mat& frame, const cv::Mat& background, double learning_rate)
{
    const cv::Mat diff = frame - bg_model_;
    const cv::Mat mean = bg_model_ + learning_rate * diff;
    const cv::Mat variance = cv::max((1 - learning_rate) * variance_ + learning_rate * diff.mul(diff), min_variance);

    // объекты не обучают модель, иначе остановившийся человек быстро растворится в фоне
    mean.copyTo(bg_model_, background);
    variance.copyTo(variance_, background);
}

double motion_segmentation::choose_learning_rate(double requested) const
{
    if (requested >= 0)
        return std::min(requested, 1.0);

    // на первых кадрах модель равна точному среднему всех кадров, поэтому обучается за секунду, а не за минуту
    return std::max(default_learning_rate, 1.0 / frames_count_);
}
} // namespace cvlib
