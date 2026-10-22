/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <opencv2/opencv.hpp>

#include <cvlib.hpp>

#include "utils.hpp"

namespace
{
// stddev of 8-bit image can not exceed 127.5 (half of pixels are 0, half are 255)
const int max_stddev = 128;
} // namespace

int demo_split_and_merge(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    cv::Mat frame;
    cv::Mat frame_gray;
    cv::Mat segmented;

    const auto origin_wnd = "origin";
    const auto demo_wnd = "demo";

    int stddev = 50;
    cv::namedWindow(demo_wnd, 1);
    cv::createTrackbar("stdev", demo_wnd, &stddev, max_stddev);

    utils::fps_counter fps;
    while (cv::waitKey(30) != 27) // ESC
    {
        cap >> frame;

        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
        cv::imshow(origin_wnd, frame);

        segmented = cvlib::split_and_merge(frame_gray, stddev);
        utils::put_fps_text(segmented, fps, cv::Scalar(255));
        cv::imshow(demo_wnd, segmented);
    }

    cv::destroyWindow(origin_wnd);
    cv::destroyWindow(demo_wnd);

    return 0;
}
