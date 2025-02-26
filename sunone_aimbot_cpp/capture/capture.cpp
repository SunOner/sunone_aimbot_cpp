#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <iostream>
#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>
#include <timeapi.h>
#include <condition_variable>

#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.System.h>
#include <winrt/Windows.System.Threading.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>
#include <winrt/base.h>
#include <comdef.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

#include "capture.h"
#include "detector.h"
#include "sunone_aimbot_cpp.h"
#include "keycodes.h"
#include "keyboard_listener.h"
#include "other_tools.h"
#include "optical_flow.h"

#include "duplication_api_capture.h"
#include "winrt_capture.h"
#include "virtual_camera.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "windowsapp.lib")

cv::cuda::GpuMat latestFrameGpu;
cv::Mat latestFrameCpu;

std::mutex frameMutex;

int screenWidth = 0;
int screenHeight = 0;

std::atomic<int> captureFrameCount(0);
std::atomic<int> captureFps(0);
std::chrono::time_point<std::chrono::high_resolution_clock> captureFpsStartTime;

void captureThread(int CAPTURE_WIDTH, int CAPTURE_HEIGHT)
{
    try
    {
        if (config.verbose)
        {
            std::cout << "[Capture] OpenCV version: " << CV_VERSION << std::endl;
            std::cout << "[Capture] CUDA Support: " << cv::cuda::getCudaEnabledDeviceCount() << " devices found." << std::endl;
        }

        IScreenCapture* capturer = nullptr;

        if (config.capture_method == "duplication_api")
        {
            capturer = new DuplicationAPIScreenCapture(CAPTURE_WIDTH, CAPTURE_HEIGHT);
            if (config.verbose)
            {
                std::cout << "[Capture] Using Duplication API." << std::endl;
            }
        }
        else if (config.capture_method == "winrt")
        {
            winrt::init_apartment(winrt::apartment_type::multi_threaded);
            capturer = new WinRTScreenCapture(CAPTURE_WIDTH, CAPTURE_HEIGHT);
            if (config.verbose)
            {
                std::cout << "[Capture] Using WinRT." << std::endl;
            }
        }
        else if (config.capture_method == "virtual_camera")
        {
            capturer = new VirtualCameraCapture(CAPTURE_WIDTH, CAPTURE_HEIGHT);
            if (config.verbose)
            {
                std::cout << "[Capture] Using virtual camera input." << std::endl;
            }
        }
        else
        {
            std::cout << "[Capture] Unknown screen capture method. The default screen capture method is set." << std::endl;
            config.capture_method = "duplication_api";
            config.saveConfig();
            capturer = new DuplicationAPIScreenCapture(CAPTURE_WIDTH, CAPTURE_HEIGHT);
            if (config.verbose)
            {
                std::cout << "[Capture] Using Duplication API." << std::endl;
            }
        }

        cv::cuda::GpuMat latestFrameGpu;
        bool buttonPreviouslyPressed = false;

        auto lastSaveTime = std::chrono::steady_clock::now();

        std::optional<std::chrono::duration<double, std::milli>> frame_duration;
        bool frameLimitingEnabled = false;

        if (config.capture_fps > 0.0)
        {
            timeBeginPeriod(1);
            frame_duration = std::chrono::duration<double, std::milli>(1000.0 / config.capture_fps);
            frameLimitingEnabled = true;
        }

        captureFpsStartTime = std::chrono::high_resolution_clock::now();
        auto start_time = std::chrono::high_resolution_clock::now();

        while (!shouldExit)
        {
            if (capture_fps_changed.load())
            {
                if (config.capture_fps > 0.0)
                {
                    if (!frameLimitingEnabled)
                    {
                        timeBeginPeriod(1);
                        frameLimitingEnabled = true;
                    }
                    frame_duration = std::chrono::duration<double, std::milli>(1000.0 / config.capture_fps);
                }
                else
                {
                    if (frameLimitingEnabled)
                    {
                        timeEndPeriod(1);
                        frameLimitingEnabled = false;
                    }
                    frame_duration = std::nullopt;
                }
                capture_fps_changed.store(false);
            }

            if (detection_resolution_changed.load() ||
                capture_method_changed.load() ||
                capture_cursor_changed.load() ||
                capture_borders_changed.load())
            {
                delete capturer;
                int new_CAPTURE_WIDTH = config.detection_resolution;
                int new_CAPTURE_HEIGHT = config.detection_resolution;

                if (config.capture_method == "duplication_api")
                {
                    capturer = new DuplicationAPIScreenCapture(new_CAPTURE_WIDTH, new_CAPTURE_HEIGHT);
                    if (config.verbose)
                        std::cout << "[Capture] Using Duplication API." << std::endl;
                }
                else if (config.capture_method == "winrt")
                {
                    capturer = new WinRTScreenCapture(new_CAPTURE_WIDTH, new_CAPTURE_HEIGHT);
                    if (config.verbose)
                        std::cout << "[Capture] Using WinRT." << std::endl;
                }
                else if (config.capture_method == "virtual_camera")
                {
                    capturer = new VirtualCameraCapture(new_CAPTURE_WIDTH, new_CAPTURE_HEIGHT);
                    if (config.verbose)
                        std::cout << "[Capture] Using virtual camera input." << std::endl;
                }
                else
                {
                    std::cout << "[Capture] Unknown screen capture method. Setting default." << std::endl;
                    config.capture_method = "duplication_api";
                    config.saveConfig();
                    continue;
                }

                screenWidth = new_CAPTURE_WIDTH;
                screenHeight = new_CAPTURE_HEIGHT;

                detection_resolution_changed.store(false);
                capture_method_changed.store(false);
                capture_cursor_changed.store(false);
                capture_borders_changed.store(false);
            }

            cv::cuda::GpuMat screenshotGpu = capturer->GetNextFrame();

            if (!screenshotGpu.empty())
            {
                cv::cuda::GpuMat processedFrame;

                if (config.circle_mask)
                {
                    cv::Mat mask = cv::Mat::zeros(screenshotGpu.size(), CV_8UC1);
                    cv::Point center(mask.cols / 2, mask.rows / 2);
                    int radius = std::min(mask.cols, mask.rows) / 2;
                    cv::circle(mask, center, radius, cv::Scalar(255), -1);
                    cv::cuda::GpuMat maskGpu;
                    maskGpu.upload(mask);
                    cv::cuda::GpuMat maskedImageGpu;
                    screenshotGpu.copyTo(maskedImageGpu, maskGpu);
                    cv::cuda::resize(maskedImageGpu, processedFrame, cv::Size(640, 640), 0, 0, cv::INTER_LINEAR);
                }
                else
                {
                    cv::cuda::resize(screenshotGpu, processedFrame, cv::Size(640, 640));
                }

                {
                    std::lock_guard<std::mutex> lock(frameMutex);
                    latestFrameGpu = processedFrame.clone();
                }

                detector.processFrame(processedFrame);

                if (config.enable_optical_flow)
                {
                    cv::cuda::GpuMat opticFrame = processedFrame.clone();
                    if (opticFrame.channels() == 4)
                    {
                        cv::cuda::cvtColor(opticFrame, opticFrame, cv::COLOR_BGRA2BGR);
                    }
                    if (opticFrame.channels() == 3)
                    {
                        cv::cuda::GpuMat opticGray;
                        cv::cuda::cvtColor(opticFrame, opticGray, cv::COLOR_BGR2GRAY);
                        opticalFlow.enqueueFrame(opticGray);
                    }
                    else
                    {
                        opticalFlow.enqueueFrame(opticFrame);
                    }
                }

                processedFrame.download(latestFrameCpu);
                {
                    std::lock_guard<std::mutex> lock(frameMutex);
                    latestFrameCpu = latestFrameCpu.clone();
                }
                frameCV.notify_one();

                if (!config.screenshot_button.empty() && config.screenshot_button[0] != "None")
                {
                    bool buttonPressed = isAnyKeyPressed(config.screenshot_button);
                    auto now = std::chrono::steady_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastSaveTime).count();

                    if (buttonPressed && elapsed >= config.screenshot_delay)
                    {
                        cv::Mat resizedCpu;
                        processedFrame.download(resizedCpu);
                        auto epoch_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch()).count();
                        std::string filename = std::to_string(epoch_time) + ".jpg";
                        cv::imwrite("screenshots/" + filename, resizedCpu);
                        lastSaveTime = now;
                    }
                }

                captureFrameCount++;
                auto currentTime = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsedTime = currentTime - captureFpsStartTime;
                if (elapsedTime.count() >= 1.0)
                {
                    captureFps = static_cast<int>(captureFrameCount / elapsedTime.count());
                    captureFrameCount = 0;
                    captureFpsStartTime = currentTime;
                }
            }

            if (frame_duration.has_value())
            {
                auto end_time = std::chrono::high_resolution_clock::now();
                auto work_duration = end_time - start_time;
                auto sleep_duration = frame_duration.value() - work_duration;

                if (sleep_duration > std::chrono::duration<double, std::milli>(0))
                {
                    std::this_thread::sleep_for(sleep_duration);
                }
                start_time = std::chrono::high_resolution_clock::now();
            }
        }

        if (frameLimitingEnabled)
        {
            timeEndPeriod(1);
        }

        delete capturer;

        if (config.capture_method == "winrt")
        {
            winrt::uninit_apartment();
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Capture] Unhandled exception: " << e.what() << std::endl;
    }
}