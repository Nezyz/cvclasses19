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
const int initial_threshold = 20;
const int max_threshold = 100; // при большем пороге углов почти не остаётся
const cv::Point count_text_origin(10, 20);
const double count_font_scale = 0.6;
const cv::Scalar green(0, 255, 0);
} // namespace

int demo_corner_detector(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    const auto main_wnd = "orig";
    const auto demo_wnd = "demo";

    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);

    int threshold = initial_threshold;
    cv::createTrackbar("th", demo_wnd, &threshold, max_threshold);

    cv::Mat frame;
    auto detector = cvlib::corner_detector_fast::create();
    std::vector<cv::KeyPoint> corners;

    utils::fps_counter fps;
    while (cv::waitKey(30) != 27) // ESC
    {
        cap >> frame;
        cv::imshow(main_wnd, frame);

        detector->set_threshold(threshold);
        detector->detect(frame, corners);
        cv::drawKeypoints(frame, corners, frame, cv::Scalar(0, 0, 255));
        utils::put_fps_text(frame, fps);
        cv::putText(frame, "corners: " + std::to_string(corners.size()), count_text_origin, cv::FONT_HERSHEY_SIMPLEX, count_font_scale, green);
        cv::imshow(demo_wnd, frame);
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);

    return 0;
}
