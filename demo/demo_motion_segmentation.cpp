/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

#include "utils.hpp"

namespace
{
const int threshold_scale = 10; // ползунок целочисленный, поэтому шаг порога 0.1 сигмы
const int initial_threshold = 25; // 2.5 сигмы
const int max_threshold = 100; // дальше 10 сигм движением не считается почти ничего

void on_threshold(int value, void* data)
{
    static_cast<cvlib::motion_segmentation*>(data)->set_threshold(static_cast<double>(value) / threshold_scale);
}
} // namespace

int demo_motion_segmentation(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    cvlib::motion_segmentation mseg(static_cast<double>(initial_threshold) / threshold_scale);
    const auto main_wnd = "orig";
    const auto demo_wnd = "demo";
    const auto background_wnd = "background";

    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);
    cv::namedWindow(background_wnd);
    cv::createTrackbar("th", demo_wnd, nullptr, max_threshold, on_threshold, &mseg);
    cv::setTrackbarPos("th", demo_wnd, initial_threshold);

    cv::Mat frame;
    cv::Mat frame_mseg;
    cv::Mat background;
    utils::fps_counter fps;
    while (cv::waitKey(30) != 27) // ESC
    {
        cap >> frame;
        mseg.apply(frame, frame_mseg);
        mseg.getBackgroundImage(background);

        utils::put_fps_text(frame, fps);
        cv::imshow(main_wnd, frame);
        cv::imshow(demo_wnd, frame_mseg);
        cv::imshow(background_wnd, background);
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);
    cv::destroyWindow(background_wnd);

    return 0;
}
