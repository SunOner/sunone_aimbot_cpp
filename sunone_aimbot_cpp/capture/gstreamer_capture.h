#ifndef GSTREAMER_CAPTURE_H
#define GSTREAMER_CAPTURE_H

#include <opencv2/opencv.hpp>
#include <string>

#include "capture.h"

class GStreamerCapture : public IScreenCapture
{
public:
    GStreamerCapture(const std::string& pipeline, int targetWidth, int targetHeight);
    cv::Mat GetNextFrameCpu() override;

private:
    bool openCapture();

    std::string pipeline_;
    int targetWidth_;
    int targetHeight_;
    cv::VideoCapture capture_;
};

#endif // GSTREAMER_CAPTURE_H