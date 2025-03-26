#ifndef VIRTUAL_CAMERA_H
#define VIRTUAL_CAMERA_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudawarping.hpp>
#include "capture.h"

class VirtualCameraCapture : public IScreenCapture
{
public:
    VirtualCameraCapture(int width, int height);
    ~VirtualCameraCapture();

    cv::cuda::GpuMat GetNextFrameGpu() override;

    cv::Mat GetNextFrameCpu() override;

    static std::vector<std::string> GetAvailableVirtualCameras();

private:
    cv::VideoCapture* cap;
    int captureWidth;
    int captureHeight;
    
    cv::cuda::GpuMat frameGpu;
    cv::Mat          frameCpu;
};

#endif // VIRTUAL_CAMERA_H
