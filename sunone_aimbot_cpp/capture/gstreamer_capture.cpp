#include <iostream>

#include "gstreamer_capture.h"

GStreamerCapture::GStreamerCapture(const std::string& pipeline, int targetWidth, int targetHeight)
    : pipeline_(pipeline)
    , targetWidth_(targetWidth)
    , targetHeight_(targetHeight)
{
    openCapture();
}

bool GStreamerCapture::openCapture()
{
    if (pipeline_.empty())
        return false;

    capture_.release();
    if (!capture_.open(pipeline_, cv::CAP_GSTREAMER))
    {
        std::cerr << "[Capture] Failed to open GStreamer pipeline." << std::endl;
        return false;
    }

    return true;
}

cv::Mat GStreamerCapture::GetNextFrameCpu()
{
    if (pipeline_.empty())
        return {};

    if (!capture_.isOpened())
    {
        if (!openCapture())
            return {};
    }

    cv::Mat frame;
    if (!capture_.read(frame))
    {
        capture_.release();
        return {};
    }

    if (!frame.empty() && targetWidth_ > 0 && targetHeight_ > 0
        && (frame.cols != targetWidth_ || frame.rows != targetHeight_))
    {
        cv::resize(frame, frame, cv::Size(targetWidth_, targetHeight_), 0, 0, cv::INTER_LINEAR);
    }

    return frame;
}