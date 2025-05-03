#include "virtual_camera.h"
#include "sunone_aimbot_cpp.h"
#include <iostream>

VirtualCameraCapture::VirtualCameraCapture(int width, int height)
    : captureWidth(width)
    , captureHeight(height)
    , cap(nullptr)
{
    captureWidth = (captureWidth % 2 == 0) ? captureWidth : (captureWidth + 1);
    captureHeight = (captureHeight % 2 == 0) ? captureHeight : (captureHeight + 1);

    std::vector<std::string> cameras = GetAvailableVirtualCameras();

    int cameraIndex = -1;
    for (int i = 0; i < static_cast<int>(cameras.size()); ++i)
    {
        if (cameras[i] == config.virtual_camera_name)
        {
            cameraIndex = i;
            break;
        }
    }

    if (cameraIndex == -1 && !cameras.empty())
    {
        cameraIndex = 0;
        config.virtual_camera_name = cameras[0];
        config.saveConfig();
    }

    if (cameraIndex != -1)
    {
        std::vector<int> backends = {
            cv::CAP_DSHOW,
            cv::CAP_MSMF
        };

        bool cameraOpened = false;
        for (int backend : backends)
        {
            cap = new cv::VideoCapture(cameraIndex, backend);
            if (cap->isOpened())
            {
                cameraOpened = true;
                break;
            }
            delete cap;
            cap = nullptr;
        }

        if (cameraOpened && cap)
        {
            cap->set(cv::CAP_PROP_FRAME_WIDTH, captureWidth);
            cap->set(cv::CAP_PROP_FRAME_HEIGHT, captureHeight);

            double actualWidth = cap->get(cv::CAP_PROP_FRAME_WIDTH);
            double actualHeight = cap->get(cv::CAP_PROP_FRAME_HEIGHT);

            if (config.verbose)
            {
                std::cout << "[Virtual camera] Requested size: " << captureWidth
                    << "x" << captureHeight << std::endl;
                std::cout << "[Virtual camera] Actual camera size: "
                    << actualWidth << "x" << actualHeight << std::endl;
            }
        }
        else
        {
            std::cerr << "[Virtual camera] Error: Could not open camera with any backend" << std::endl;
        }
    }
}

VirtualCameraCapture::~VirtualCameraCapture()
{
    if (cap)
    {
        cap->release();
        delete cap;
        cap = nullptr;
    }
}

cv::cuda::GpuMat VirtualCameraCapture::GetNextFrameGpu()
{
    if (!cap || !cap->isOpened())
    {
        return cv::cuda::GpuMat();
    }

    cv::Mat frame;
    if (!cap->read(frame))
    {
        return cv::cuda::GpuMat();
    }
    if (frame.empty())
    {
        return cv::cuda::GpuMat();
    }

    try
    {
        cv::Mat processedFrame;
        if (frame.channels() == 1)
        {
            cv::cvtColor(frame, processedFrame, cv::COLOR_GRAY2BGR);
        }
        else if (frame.channels() == 4)
        {
            cv::cvtColor(frame, processedFrame, cv::COLOR_BGRA2BGR);
        }
        else if (frame.channels() == 3)
        {
            processedFrame = frame;
        }
        else
        {
            std::cerr << "[VirtualCamera] Unexpected number of channels: "
                << frame.channels() << std::endl;
            return cv::cuda::GpuMat();
        }

        frameGpu.upload(frame);
        return frameGpu;
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "[VirtualCamera] OpenCV exception: " << e.what() << std::endl;
        return cv::cuda::GpuMat();
    }
}

cv::Mat VirtualCameraCapture::GetNextFrameCpu()
{
    if (!cap || !cap->isOpened())
    {
        return cv::Mat();
    }

    cv::Mat frame;
    if (!cap->read(frame))
    {
        return cv::Mat();
    }
    if (frame.empty())
    {
        return cv::Mat();
    }

    try
    {
        cv::Mat processedFrame;
        if (frame.channels() == 1)
        {
            cv::cvtColor(frame, processedFrame, cv::COLOR_GRAY2BGR);
        }
        else if (frame.channels() == 4)
        {
            cv::cvtColor(frame, processedFrame, cv::COLOR_BGRA2BGR);
        }
        else if (frame.channels() == 3)
        {
            processedFrame = frame;
        }
        else
        {
            std::cerr << "[VirtualCamera] Unexpected channel count: "
                << frame.channels() << std::endl;
            return cv::Mat();
        }

        cv::Mat resizedFrame;
        cv::resize(
            processedFrame,
            resizedFrame,
            cv::Size(captureWidth, captureHeight),
            0,
            0,
            cv::INTER_LINEAR
        );

        cv::Mat evenFrame;
        cv::resize(
            resizedFrame,
            evenFrame,
            cv::Size(
                resizedFrame.cols + (resizedFrame.cols % 2),
                resizedFrame.rows + (resizedFrame.rows % 2)
            ),
            0,
            0,
            cv::INTER_LINEAR
        );
        resizedFrame = evenFrame;

        frameCpu = resizedFrame.clone();
        return frameCpu;
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "[VirtualCamera] OpenCV exception: " << e.what() << std::endl;
        return cv::Mat();
    }
}

std::vector<std::string> VirtualCameraCapture::GetAvailableVirtualCameras()
{
    std::vector<std::string> cameras;

    std::vector<int> backends = {
        cv::CAP_DSHOW,
        cv::CAP_MSMF,
        cv::CAP_ANY
    };

    for (int backend : backends)
    {
        for (int i = 0; i < 10; ++i)
        {
            try
            {
                cv::VideoCapture testCap(i, backend);
                if (testCap.isOpened())
                {
                    std::string deviceName =
                        "Camera " + std::to_string(i) + ":" +
                        (backend == cv::CAP_DSHOW ? "DirectShow" :
                            backend == cv::CAP_MSMF ? "MSMF" :
                            "Any");
                    cameras.push_back(deviceName);
                    testCap.release();
                }
            }
            catch (...)
            {
                
            }
        }
    }

    std::cout << "[Virtual camera] Available cameras:" << std::endl;
    for (const auto& camera : cameras)
    {
        std::cout << "  " << camera << std::endl;
    }

    return cameras;
}