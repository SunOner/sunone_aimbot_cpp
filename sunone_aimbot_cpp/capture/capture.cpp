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

#include "capture.h"
#ifdef USE_CUDA
#include "trt_detector.h"
#include "depth/depth_mask.h"
#include "tensorrt/nvinf.h"
#endif
#include "sunone_aimbot_cpp.h"
#include "keycodes.h"
#include "keyboard_listener.h"
#include "other_tools.h"
#include "duplication_api_capture.h"
#include "winrt_capture.h"
#include "virtual_camera.h"
#include "udp_capture.h"
#include "capture_utils.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "windowsapp.lib")

cv::Mat latestFrame;
std::mutex frameMutex;
std::mutex capturerMutex;

int screenWidth = 0;
int screenHeight = 0;

std::atomic<int> captureFrameCount(0);
std::atomic<int> captureFps(0);
std::chrono::time_point<std::chrono::high_resolution_clock> captureFpsStartTime;

std::deque<cv::Mat> frameQueue;

std::vector<cv::Mat> getBatchFromQueue(int batch_size)
{
    std::vector<cv::Mat> batch;
    std::lock_guard<std::mutex> lk(frameMutex);
    const size_t target_size = (batch_size > 0) ? static_cast<size_t>(batch_size) : 0;
    const size_t n = std::min(frameQueue.size(), target_size);

    for (size_t i = 0; i < n; ++i)
        batch.push_back(frameQueue[frameQueue.size() - n + i]);

    while (batch.size() < target_size && !batch.empty())
        batch.push_back(batch.back().clone());
    return batch;
}

void captureThread(int CAPTURE_WIDTH, int CAPTURE_HEIGHT)
{
    try
    {
        if (config.verbose)
            std::cout << "[Capture] OpenCV version: " << CV_VERSION << std::endl;

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
            {
                std::lock_guard<std::mutex> lock(capturerMutex);
                capturer = new VirtualCameraCapture(config.virtual_camera_width, config.virtual_camera_heigth);
            }
            if (config.verbose)
                std::cout << "[Capture] Using Virtual Camera" << std::endl;
        }
        else if (config.capture_method == "udp_capture")
        {
            capturer = new UDPCapture(CAPTURE_WIDTH, CAPTURE_HEIGHT, config.udp_ip, config.udp_port);
            if (config.verbose)
                std::cout << "[Capture] Using UDP capture" << std::endl;
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
                    {
                        std::lock_guard<std::mutex> lock(capturerMutex);
                        capturer = new VirtualCameraCapture(config.virtual_camera_width, config.virtual_camera_heigth);
                    }
                    if (config.verbose)
                        std::cout << "[Capture] Re-init with Virtual Camera." << std::endl;
                }
                else if (config.capture_method == "udp_capture")
                {
                    capturer = new UDPCapture(newWidth, newHeight, config.udp_ip, config.udp_port);
                    if (config.verbose)
                        std::cout << "[Capture] Re-init with UDP capture." << std::endl;
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

            cv::Mat screenshotCpu;
            {
                std::lock_guard<std::mutex> lock(capturerMutex);
                screenshotCpu = capturer->GetNextFrameCpu();
            }

            if (screenshotCpu.empty())
                continue;

            if (config.capture_method == "virtual_camera")
            {
                int x = (screenshotCpu.cols - CAPTURE_WIDTH) / 2;
                int y = (screenshotCpu.rows - CAPTURE_HEIGHT) / 2;
                x = std::max(x, 0);
                y = std::max(y, 0);
                screenshotCpu = screenshotCpu(cv::Rect(x, y, CAPTURE_WIDTH, CAPTURE_HEIGHT)).clone();
            }

            if (config.circle_mask)
                screenshotCpu = apply_circle_mask(screenshotCpu);

            cv::Mat detectionFrame = screenshotCpu;
#ifdef USE_CUDA
            static bool lastDepthInferenceEnabled = true;
            if (!config.depth_inference_enabled)
            {
                if (lastDepthInferenceEnabled)
                {
                    auto& depthMask = depth_anything::GetDepthMaskGenerator();
                    depthMask.reset();
                }
                lastDepthInferenceEnabled = false;
            }
            else
            {
                lastDepthInferenceEnabled = true;
            }

            if (config.depth_inference_enabled && config.depth_mask_enabled)
            {
                if (config.verbose)
                {
                    static auto lastMaskLog = std::chrono::steady_clock::time_point::min();
                    auto now = std::chrono::steady_clock::now();
                    if (now - lastMaskLog > std::chrono::seconds(2))
                    {
                        std::cout << "[DepthMask] update frame " << screenshotCpu.cols << "x" << screenshotCpu.rows
                                  << " model=" << config.depth_model_path
                                  << " fps=" << config.depth_mask_fps
                                  << " near=" << config.depth_mask_near_percent
                                  << " invert=" << (config.depth_mask_invert ? "true" : "false")
                                  << std::endl;
                        lastMaskLog = now;
                    }
                }

                depth_anything::DepthMaskOptions maskOptions;
                maskOptions.enabled = config.depth_mask_enabled;
                maskOptions.fps = config.depth_mask_fps;
                maskOptions.near_percent = config.depth_mask_near_percent;
                maskOptions.invert = config.depth_mask_invert;

                auto& depthMask = depth_anything::GetDepthMaskGenerator();
                depthMask.update(screenshotCpu, maskOptions, config.depth_model_path, gLogger);

                cv::Mat mask = depthMask.getMask();
                if (!mask.empty() && mask.size() == screenshotCpu.size())
                {
                    detectionFrame = screenshotCpu.clone();
                    detectionFrame.setTo(cv::Scalar(0, 0, 0), mask);
                }
            }
#endif

            {
                std::lock_guard<std::mutex> lock(frameMutex);
                latestFrame = screenshotCpu.clone();
                if (frameQueue.size() >= 1)
                    frameQueue.pop_front();
                frameQueue.push_back(latestFrame);
            }
            frameCV.notify_one();

            if (config.backend == "DML" && dml_detector)
            {
                dml_detector->processFrame(detectionFrame);
            }
#ifdef USE_CUDA
            else if (config.backend == "TRT")
            {
                trt_detector.processFrame(detectionFrame);
            }
#endif
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
                        saveMat = latestFrame.clone();
                    }

                    auto epoch_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()
                    ).count();
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
                    std::this_thread::sleep_for(sleep_duration);
                start_time = std::chrono::high_resolution_clock::now();
            }
        }

        if (frameLimitingEnabled)
            timeEndPeriod(1);

        if (capturer)
        {
            std::lock_guard<std::mutex> lock(capturerMutex);
            delete capturer;
            capturer = nullptr;
        }

        if (config.capture_method == "winrt")
            winrt::uninit_apartment();
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Capture] Unhandled exception: " << e.what() << std::endl;
    }
}
