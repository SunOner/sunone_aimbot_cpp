#pragma once
#ifdef USE_CUDA

#include <opencv2/opencv.hpp>
#include <tuple>

namespace depth_anything
{
    std::tuple<cv::Mat, int, int> resize_depth(const cv::Mat& img, int w, int h);
}

#endif