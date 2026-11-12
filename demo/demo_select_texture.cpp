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
// расстояние между контрастными полосами разной ориентации или частоты составляет 40-80,
// у реальных текстур контраст ниже, поэтому при большем пороге выделяется весь кадр
const int max_eps = 100;

struct user_data
{
    std::string wnd;
    cv::Point tl;
    cv::Point br;
    cv::Mat image;
};

void mouse(int event, int x, int y, int flags, void* param)
{
    user_data& data = *reinterpret_cast<user_data*>(param);
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        data.tl = {x, y};
    }
    else if (event == cv::EVENT_RBUTTONDOWN)
    {
        data.br = {x, y};
    }
}
} // namespace

int demo_select_texture(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    user_data data;

    data.wnd = "origin";
    const auto demo_wnd = "demo";

    int eps = 20;
    cv::namedWindow(data.wnd);
    cv::namedWindow(demo_wnd);
    cv::createTrackbar("eps", demo_wnd, &eps, max_eps);

    cv::setMouseCallback(data.wnd, mouse, &data);

    cv::Mat frame_gray;
    utils::fps_counter fps;
    while (cv::waitKey(30) != 27) // ESC
    {
        cap >> data.image;

        cv::cvtColor(data.image, frame_gray, cv::COLOR_BGR2GRAY);
        const cv::Rect roi = {data.tl, data.br};
        if (roi.area())
        {
            const auto mask = cvlib::select_texture(frame_gray, roi, eps);
            cv::Mat segmented = mask.clone();
            frame_gray.copyTo(segmented, mask);
            utils::put_fps_text(segmented, fps, cv::Scalar(255));
            cv::imshow(demo_wnd, segmented);
            cv::rectangle(data.image, data.tl, data.br, cv::Scalar(0, 0, 255));
        }
        cv::imshow(data.wnd, data.image);
    }

    cv::destroyWindow(data.wnd);
    cv::destroyWindow(demo_wnd);

    return 0;
}
