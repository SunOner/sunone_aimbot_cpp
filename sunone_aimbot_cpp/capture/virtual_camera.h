#ifndef VIRTUAL_CAMERA_H
#define VIRTUAL_CAMERA_H

#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>

#include "capture.h"
#include "sunone_aimbot_cpp.h"

class VirtualCameraCapture final : public IScreenCapture
{
public:
    VirtualCameraCapture(int width, int height);
    ~VirtualCameraCapture() override;

    cv::cuda::GpuMat GetNextFrameGpu() override;
    cv::Mat         GetNextFrameCpu() override;

    static std::vector<std::string> GetAvailableVirtualCameras(bool forceRescan = false);
    static void ClearCachedCameraList();

private:
    std::unique_ptr<cv::VideoCapture> cap_;
    int captureWidth{ 0 }, captureHeight{ 0 };

    int roiW_, roiH_;

    cv::Mat frameCpu;
    cv::cuda::GpuMat lastGpu;
    cv::cuda::GpuMat scratchGpu_;
    cv::cuda::GpuMat bgrGpu_;

};

#endif // VIRTUAL_CAMERA_H
