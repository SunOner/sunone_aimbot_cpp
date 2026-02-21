#ifndef VIRTUAL_CAMERA_H
#define VIRTUAL_CAMERA_H

#include <opencv2/opencv.hpp>

#include "capture.h"
#include "sunone_aimbot_cpp.h"

class VirtualCameraCapture final : public IScreenCapture
{
public:
    VirtualCameraCapture(int width, int height, const std::string& cameraName, int captureFps, bool verbose);
    ~VirtualCameraCapture() override;

    cv::Mat GetNextFrameCpu() override;

    static std::vector<std::string> GetAvailableVirtualCameras(bool forceRescan = false);
    static void ClearCachedCameraList();

private:
    std::unique_ptr<cv::VideoCapture> cap_;
    int captureWidth{ 0 }, captureHeight{ 0 };
    int targetWidth_{ 0 };
    int targetHeight_{ 0 };
    std::string selectedCameraName_;
    int captureFps_{ 0 };
    bool verbose_{ false };

    int roiW_, roiH_;

    cv::Mat frameCpu;
};

#endif // VIRTUAL_CAMERA_H
