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
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

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
            std::cout << "[Capture] CUDA devices found: " << cv::cuda::getCudaEnabledDeviceCount() << std::endl;
        }

        IScreenCapture* capturer = nullptr;
        if (config.capture_method == "duplication_api")
        {
            capturer = new DuplicationAPIScreenCapture(CAPTURE_WIDTH, CAPTURE_HEIGHT);
            if (config.verbose)
                std::cout << "[Capture] Using Duplication API" << std::endl;
        }
        else if (config.capture_method == "winrt")
        {
            winrt::init_apartment(winrt::apartment_type::multi_threaded);
            capturer = new WinRTScreenCapture(CAPTURE_WIDTH, CAPTURE_HEIGHT);
            if (config.verbose)
                std::cout << "[Capture] Using WinRT" << std::endl;
        }
        else if (config.capture_method == "virtual_camera")
        {
            capturer = new VirtualCameraCapture(CAPTURE_WIDTH, CAPTURE_HEIGHT);
            if (config.verbose)
                std::cout << "[Capture] Using Virtual Camera" << std::endl;
        }
        else
        {
            config.capture_method = "duplication_api";
            config.saveConfig();
            capturer = new DuplicationAPIScreenCapture(CAPTURE_WIDTH, CAPTURE_HEIGHT);
            std::cout << "[Capture] Unknown capture_method. Set to duplication_api by default." << std::endl;
        }

        bool frameLimitingEnabled = false;
        std::optional<std::chrono::duration<double, std::milli>> frame_duration;
        if (config.capture_fps > 0.0)
        {
            timeBeginPeriod(1);
            frame_duration = std::chrono::duration<double, std::milli>(1000.0 / config.capture_fps);
            frameLimitingEnabled = true;
        }

        captureFpsStartTime = std::chrono::high_resolution_clock::now();
        auto start_time = std::chrono::high_resolution_clock::now();
        auto lastSaveTime = std::chrono::steady_clock::now();

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
                    frame_duration.reset();
                }
                capture_fps_changed.store(false);
            }

            if (detection_resolution_changed.load() ||
                capture_method_changed.load() ||
                capture_cursor_changed.load() ||
                capture_borders_changed.load())
            {
                delete capturer;
                capturer = nullptr;

                int newWidth = config.detection_resolution;
                int newHeight = config.detection_resolution;

                if (config.capture_method == "duplication_api")
                {
                    capturer = new DuplicationAPIScreenCapture(newWidth, newHeight);
                    if (config.verbose)
                        std::cout << "[Capture] Re-init with Duplication API." << std::endl;
                }
                else if (config.capture_method == "winrt")
                {
                    capturer = new WinRTScreenCapture(newWidth, newHeight);
                    if (config.verbose)
                        std::cout << "[Capture] Re-init with WinRT." << std::endl;
                }
                else if (config.capture_method == "virtual_camera")
                {
                    capturer = new VirtualCameraCapture(newWidth, newHeight);
                    if (config.verbose)
                        std::cout << "[Capture] Re-init with Virtual Camera." << std::endl;
                }
                else
                {
                    config.capture_method = "duplication_api";
                    config.saveConfig();
                    capturer = new DuplicationAPIScreenCapture(newWidth, newHeight);
                    std::cout << "[Capture] Unknown capture_method. Set to duplication_api." << std::endl;
                }

                detection_resolution_changed.store(false);
                capture_method_changed.store(false);
                capture_cursor_changed.store(false);
                capture_borders_changed.store(false);
            }

            cv::cuda::GpuMat screenshotGpu;
            cv::Mat screenshotCpu;

            if (config.capture_use_cuda)
            {
                screenshotGpu = capturer->GetNextFrameGpu();
                if (screenshotGpu.empty())
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    continue;
                }

                cv::cuda::GpuMat maskedGpu;
                if (config.circle_mask)
                {
                    cv::Mat maskHost = cv::Mat::zeros(screenshotGpu.size(), CV_8UC1);
                    cv::Point center(maskHost.cols / 2, maskHost.rows / 2);
                    int radius = std::min(maskHost.cols, maskHost.rows) / 2;
                    cv::circle(maskHost, center, radius, cv::Scalar(255), -1);

                    cv::cuda::GpuMat maskGpu;
                    maskGpu.upload(maskHost);

                    cv::cuda::GpuMat maskedTemp;
                    screenshotGpu.copyTo(maskedTemp, maskGpu);

                    cv::cuda::resize(maskedTemp, maskedGpu, cv::Size(640, 640), 0, 0, cv::INTER_LINEAR);
                }
                else
                {
                    cv::cuda::resize(screenshotGpu, maskedGpu, cv::Size(640, 640), 0, 0, cv::INTER_LINEAR);
                }

                {
                    std::lock_guard<std::mutex> lock(frameMutex);
                    latestFrameGpu = maskedGpu.clone();
                    latestFrameCpu.release();
                }

                if (config.backend == "TRT")
                {
                    detector.processFrame(maskedGpu);
                }
                else if (dml_detector)
                {
                    cv::Mat cpuMat;
                    maskedGpu.download(cpuMat);

                    auto detections = dml_detector->detect(cpuMat);
                    {
                        std::lock_guard<std::mutex> lock(detector.detectionMutex);
                        detector.detectedBoxes.clear();
                        detector.detectedClasses.clear();
                        for (auto& d : detections)
                        {
                            detector.detectedBoxes.push_back(d.box);
                            detector.detectedClasses.push_back(d.classId);
                        }
                        detector.detectionVersion++;
                    }
                    detector.detectionCV.notify_one();
                }

                if (config.enable_optical_flow)
                {
                    cv::cuda::GpuMat flowFrame = maskedGpu;

                    if (flowFrame.channels() == 4)
                        cv::cuda::cvtColor(flowFrame, flowFrame, cv::COLOR_BGRA2BGR);

                    if (flowFrame.channels() == 3)
                    {
                        cv::cuda::GpuMat gray;
                        cv::cuda::cvtColor(flowFrame, gray, cv::COLOR_BGR2GRAY);
                        opticalFlow.enqueueFrame(gray);
                    }
                    else
                    {
                        opticalFlow.enqueueFrame(flowFrame);
                    }
                }

                {
                    cv::Mat tmpCpu;
                    maskedGpu.download(tmpCpu);
                    std::lock_guard<std::mutex> lock(frameMutex);
                    latestFrameCpu = tmpCpu.clone();
                }
            }
            else
            {
                screenshotCpu = capturer->GetNextFrameCpu();
                if (screenshotCpu.empty())
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    continue;
                }

                if (config.circle_mask)
                {
                    cv::Mat mask = cv::Mat::zeros(screenshotCpu.size(), CV_8UC1);
                    cv::Point center(mask.cols / 2, mask.rows / 2);
                    int radius = std::min(mask.cols, mask.rows) / 2;
                    cv::circle(mask, center, radius, cv::Scalar(255), -1);

                    cv::Mat maskedCpu;
                    screenshotCpu.copyTo(maskedCpu, mask);
                    cv::resize(maskedCpu, screenshotCpu, cv::Size(640, 640), 0, 0, cv::INTER_LINEAR);
                }
                else
                {
                    cv::resize(screenshotCpu, screenshotCpu, cv::Size(640, 640), 0, 0, cv::INTER_LINEAR);
                }

                {
                    std::lock_guard<std::mutex> lock(frameMutex);
                    latestFrameCpu = screenshotCpu.clone();
                    latestFrameGpu.release();
                }
                frameCV.notify_one();

                if (config.backend == "DML" && dml_detector)
                {
                    auto detections = dml_detector->detect(screenshotCpu);
                    {
                        std::lock_guard<std::mutex> lock(detector.detectionMutex);
                        detector.detectedBoxes.clear();
                        detector.detectedClasses.clear();
                        for (auto& d : detections)
                        {
                            detector.detectedBoxes.push_back(d.box);
                            detector.detectedClasses.push_back(d.classId);
                        }
                        detector.detectionVersion++;
                    }
                    detector.detectionCV.notify_one();
                }
                else if (config.backend == "TRT")
                {
                    cv::cuda::GpuMat toTrt;
                    toTrt.upload(screenshotCpu);
                    detector.processFrame(toTrt);
                }
            }

            if (!config.screenshot_button.empty() && config.screenshot_button[0] != "None")
            {
                bool buttonPressed = isAnyKeyPressed(config.screenshot_button);
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastSaveTime).count();
                if (buttonPressed && elapsed >= config.screenshot_delay)
                {
                    cv::Mat saveMat;
                    {
                        std::lock_guard<std::mutex> lock(frameMutex);
                        saveMat = latestFrameCpu.clone();
                    }

                    auto epoch_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count();
                    std::string filename = std::to_string(epoch_time) + ".jpg";
                    cv::imwrite("screenshots/" + filename, saveMat);

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

        if (capturer)
        {
            delete capturer;
            capturer = nullptr;
        }

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