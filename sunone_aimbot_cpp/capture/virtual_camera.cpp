#include <opencv2/cudaimgproc.hpp>
#include <iostream>
#include <algorithm>

#include "virtual_camera.h"

namespace {

    inline int even(int v) { return (v % 2 == 0) ? v : v + 1; }

}

VirtualCameraCapture::VirtualCameraCapture(int w, int h)
{
    auto cams = GetAvailableVirtualCameras();
    if (cams.empty())
        throw std::runtime_error("[VirtualCamera] No capture devices found");

    int camIdx = 0;
    try
    {
        camIdx = std::stoi(config.virtual_camera_name);
    }
    catch (...)
    {
        camIdx = 0;
    }
    
    cap_ = std::make_unique<cv::VideoCapture>(camIdx, cv::CAP_MSMF);
    cap_->set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

    if (!cap_->isOpened())
    {
        for (int i = 0; i < 5 && !cap_->isOpened(); ++i)
        {
            cap_.reset(new cv::VideoCapture(i, cv::CAP_MSMF));
            if (cap_->isOpened())
                std::cerr << "[VirtualCamera] Opened by index: " << i << '\n';
        }
    }

    if (!cap_->isOpened())
        throw std::runtime_error("[VirtualCamera] Unable to open any capture device");

    bool autoMode = (w <= 0 || h <= 0);

    if (autoMode)
    {
        w = static_cast<int>(cap_->get(cv::CAP_PROP_FRAME_WIDTH));
        h = static_cast<int>(cap_->get(cv::CAP_PROP_FRAME_HEIGHT));
    }
    else
    {
        cap_->set(cv::CAP_PROP_FRAME_WIDTH, even(w));
        cap_->set(cv::CAP_PROP_FRAME_HEIGHT, even(h));
        w = static_cast<int>(cap_->get(cv::CAP_PROP_FRAME_WIDTH));
        h = static_cast<int>(cap_->get(cv::CAP_PROP_FRAME_HEIGHT));
    }

    cap_->set(cv::CAP_PROP_FPS, config.capture_fps);
    cap_->set(cv::CAP_PROP_BUFFERSIZE, 0);

    roiW_ = even(w);
    roiH_ = even(h);

    scratchGpu_.create(roiH_, roiW_, CV_8UC2);
    bgrGpu_.create(roiH_, roiW_, CV_8UC3);

    std::cout << "[VirtualCamera] Actual capture: "
        << roiW_ << 'x' << roiH_ << " @ "
        << cap_->get(cv::CAP_PROP_FPS) << " FPS\n";
}

VirtualCameraCapture::~VirtualCameraCapture()
{
    if (cap_)
    {
        if (cap_->isOpened())
        {
            cap_->release();
        }
        cap_.reset();
    }
}

cv::cuda::GpuMat VirtualCameraCapture::GetNextFrameGpu()
{
    if (!cap_ || !cap_->isOpened())
        return cv::cuda::GpuMat();

    cv::Mat frame;
    if (!cap_->read(frame) || frame.empty())
    {
        return lastGpu;
    }

    if (frame.channels() == 1)        cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
    else if (frame.channels() == 2)   cv::cvtColor(frame, frame, cv::COLOR_YUV2BGR_YUY2);
    else if (frame.channels() == 4)   cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);

    cv::cuda::GpuMat gpuFrame;
    gpuFrame.upload(frame);

    int target_width = config.virtual_camera_width;
    int target_height = config.virtual_camera_heigth;

    if (target_width > 0 && target_height > 0 && !gpuFrame.empty())
    {
        cv::cuda::resize(gpuFrame, gpuFrame, cv::Size(target_width, target_height));
    }

    lastGpu = gpuFrame;
    return gpuFrame;
}

cv::Mat VirtualCameraCapture::GetNextFrameCpu()
{
    if (!cap_ || !cap_->isOpened())
        return cv::Mat();

    cv::Mat frame;
    if (!cap_->read(frame) || frame.empty())
    {
        return cv::Mat();
    }

    switch (frame.channels())
    {
    case 1: cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR); break;
    case 4: cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR); break;
    case 3:                                                 break;
    default:
        std::cerr << "[VirtualCamera] Unexpected channel count: "
            << frame.channels() << std::endl;
        return cv::Mat();
    }

    frameCpu = frame;

    int target_width = config.virtual_camera_width;
    int target_height = config.virtual_camera_heigth;

    if (target_width > 0 && target_height > 0 && !frameCpu.empty())
    {
        cv::resize(frameCpu, frameCpu, cv::Size(target_width, target_height));
    }

    return frameCpu.clone();
}

std::vector<std::string> VirtualCameraCapture::GetAvailableVirtualCameras()
{
    std::vector<std::string> cams;

    for (int i = 0; i < 10; ++i)
    {
        cv::VideoCapture test(i, cv::CAP_DSHOW);
        if (test.isOpened())
        {
            cams.push_back(std::to_string(i));
            test.release();
        }
    }
    return cams;
}