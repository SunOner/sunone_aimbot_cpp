#ifndef CAPTURE_H
#define CAPTURE_H

#include <opencv2/opencv.hpp>

void captureThread(int CAPTURE_WIDTH, int CAPTURE_HEIGHT);

cv::Mat cropCenterCPU(const cv::Mat& src, int targetWidth, int targetHeight);

extern int screenWidth;
extern int screenHeight;

#endif // CAPTURE_H