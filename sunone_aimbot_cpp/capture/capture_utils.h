#pragma once
#include <opencv2/opencv.hpp>

cv::Mat apply_circle_mask(const cv::Mat& input);

void apply_circle_mask_inplace(cv::Mat& image);