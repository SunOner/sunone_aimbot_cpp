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
#include "timeapi.h"
#include <condition_variable>

#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudaimgproc.hpp>

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

#include "duplication_api_capture.h"
#include "winrt_capture.h"

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "windowsapp.lib")

using namespace std;

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

        if (config.duplication_api)
        {
            capturer = new DuplicationAPIScreenCapture(CAPTURE_WIDTH, CAPTURE_HEIGHT);
            if (config.verbose)
            {
                std::cout << "[Capture] Using Duplication API." << std::endl;
            }
        }
        else
        {
            winrt::init_apartment(winrt::apartment_type::multi_threaded);
            capturer = new WinRTScreenCapture(CAPTURE_WIDTH, CAPTURE_HEIGHT);
            if (config.verbose)
            {
                std::cout << "[Capture] Using WinRT." << std::endl;
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

            if (detection_resolution_changed.load()
                || capture_method_changed.load()
                || capture_cursor_changed.load()
                || capture_borders_changed.load())
            {
                delete capturer;

                int new_CAPTURE_WIDTH = config.detection_resolution;
                int new_CAPTURE_HEIGHT = config.detection_resolution;

                if (config.duplication_api)
                {
                    capturer = new DuplicationAPIScreenCapture(new_CAPTURE_WIDTH, new_CAPTURE_HEIGHT);
                    if (config.verbose)
                    {
                        std::cout << "[Capture] Using Duplication API." << std::endl;
                    }
                }
                else
                {
                    capturer = new WinRTScreenCapture(new_CAPTURE_WIDTH, new_CAPTURE_HEIGHT);
                    if (config.verbose)
                    {
                        std::cout << "[Capture] Using WinRT." << std::endl;
                    }
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
                {
                    std::lock_guard<std::mutex> lock(detector.scaleMutex);
                    detector.scaleX = 1;
                    detector.scaleY = 1;
                }

                {
                    std::lock_guard<std::mutex> lock(frameMutex);
                    latestFrameGpu = screenshotGpu.clone();
                }

                detector.processFrame(screenshotGpu);
                screenshotGpu.download(latestFrameCpu);

                {
                    std::lock_guard<std::mutex> lock(frameMutex);
                    latestFrameCpu = latestFrameCpu.clone();
                }
                frameCV.notify_one();


                if (config.screenshot_button.size() && config.screenshot_button[0] != "None")
                {
                    bool buttonPressed = isAnyKeyPressed(config.screenshot_button);

                    if (buttonPressed)
                    {
                        auto now = std::chrono::steady_clock::now();
                        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastSaveTime).count();

                        if (elapsed >= config.screenshot_delay)
                        {
                            cv::Mat resizedCpu;
                            screenshotGpu.download(resizedCpu);

                            auto epoch_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::system_clock::now().time_since_epoch()).count();
                            std::string filename = std::to_string(epoch_time) + ".jpg";

                            cv::imwrite("screenshots/" + filename, resizedCpu);

                            lastSaveTime = now;
                        }
                    }

                    buttonPreviouslyPressed = buttonPressed;
                }

                captureFrameCount++;
                auto currentTime = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = currentTime - captureFpsStartTime;

                if (elapsed.count() >= 1.0)
                {
                    captureFps = static_cast<int>(captureFrameCount / elapsed.count());
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

        if (!config.duplication_api)
        {
            winrt::uninit_apartment();
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Capture] Unhandled exception: " << e.what() << std::endl;
    }
}