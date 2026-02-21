#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <iostream>
#include <atomic>
#include <thread>
#include <mutex>
#include <algorithm>
#include <chrono>
#include <timeapi.h>
#include <condition_variable>
#include <filesystem>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

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

int screenWidth = 0;
int screenHeight = 0;

std::atomic<int> captureFrameCount(0);
std::atomic<int> captureFps(0);
std::chrono::time_point<std::chrono::high_resolution_clock> captureFpsStartTime;

std::deque<cv::Mat> frameQueue;

namespace
{
struct CaptureThreadConfig
{
    std::string capture_method;
    int capture_fps = 0;
    int detection_resolution = 0;
    int monitor_idx = 0;
    bool circle_mask = false;
    bool capture_borders = true;
    bool capture_cursor = true;
    std::string capture_target;
    std::string capture_window_title;
    std::string virtual_camera_name;
    int virtual_camera_width = 0;
    int virtual_camera_heigth = 0;
    std::string udp_ip;
    int udp_port = 0;
    std::string backend;
    std::vector<std::string> screenshot_button;
    int screenshot_delay = 0;
    bool verbose = false;
#ifdef USE_CUDA
    bool depth_inference_enabled = false;
    bool depth_mask_enabled = false;
    std::string depth_model_path;
    int depth_mask_fps = 0;
    int depth_mask_near_percent = 0;
    bool depth_mask_invert = false;
#endif
};

CaptureThreadConfig SnapshotCaptureConfig()
{
    std::lock_guard<std::mutex> cfgLock(configMutex);
    CaptureThreadConfig snapshot;
    snapshot.capture_method = config.capture_method;
    snapshot.capture_fps = config.capture_fps;
    snapshot.detection_resolution = config.detection_resolution;
    snapshot.monitor_idx = config.monitor_idx;
    snapshot.circle_mask = config.circle_mask;
    snapshot.capture_borders = config.capture_borders;
    snapshot.capture_cursor = config.capture_cursor;
    snapshot.capture_target = config.capture_target;
    snapshot.capture_window_title = config.capture_window_title;
    snapshot.virtual_camera_name = config.virtual_camera_name;
    snapshot.virtual_camera_width = config.virtual_camera_width;
    snapshot.virtual_camera_heigth = config.virtual_camera_heigth;
    snapshot.udp_ip = config.udp_ip;
    snapshot.udp_port = config.udp_port;
    snapshot.backend = config.backend;
    snapshot.screenshot_button = config.screenshot_button;
    snapshot.screenshot_delay = config.screenshot_delay;
    snapshot.verbose = config.verbose;
#ifdef USE_CUDA
    snapshot.depth_inference_enabled = config.depth_inference_enabled;
    snapshot.depth_mask_enabled = config.depth_mask_enabled;
    snapshot.depth_model_path = config.depth_model_path;
    snapshot.depth_mask_fps = config.depth_mask_fps;
    snapshot.depth_mask_near_percent = config.depth_mask_near_percent;
    snapshot.depth_mask_invert = config.depth_mask_invert;
#endif
    return snapshot;
}

std::string NormalizeCaptureMethod(const std::string& method)
{
    if (method == "duplication_api" || method == "winrt" || method == "virtual_camera" || method == "udp_capture")
        return method;
    return "duplication_api";
}

class TimerResolutionGuard
{
public:
    void Enable()
    {
        if (!enabled_)
        {
            timeBeginPeriod(1);
            enabled_ = true;
        }
    }

    void Disable()
    {
        if (enabled_)
        {
            timeEndPeriod(1);
            enabled_ = false;
        }
    }

    ~TimerResolutionGuard()
    {
        Disable();
    }

private:
    bool enabled_{ false };
};

class WinrtApartmentGuard
{
public:
    void Ensure(bool required)
    {
        if (required && !initialized_)
        {
            winrt::init_apartment(winrt::apartment_type::multi_threaded);
            initialized_ = true;
        }
        else if (!required && initialized_)
        {
            winrt::uninit_apartment();
            initialized_ = false;
        }
    }

    ~WinrtApartmentGuard()
    {
        if (initialized_)
            winrt::uninit_apartment();
    }

private:
    bool initialized_{ false };
};

class ScreenshotWriter
{
public:
    ScreenshotWriter()
    {
        writerThread_ = std::thread([this]() { Run(); });
    }

    ~ScreenshotWriter()
    {
        Stop();
    }

    void Enqueue(const std::string& filename, cv::Mat frame)
    {
        if (filename.empty() || frame.empty())
            return;

        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.size() >= maxPendingFrames_)
            queue_.pop();
        queue_.emplace(filename, std::move(frame));
        cv_.notify_one();
    }

private:
    void Stop()
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_one();

        if (writerThread_.joinable())
            writerThread_.join();
    }

    void Run()
    {
        std::error_code ec;
        std::filesystem::create_directories("screenshots", ec);

        while (true)
        {
            std::pair<std::string, cv::Mat> job;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this]() { return stop_ || !queue_.empty(); });
                if (stop_ && queue_.empty())
                    break;

                job = std::move(queue_.front());
                queue_.pop();
            }

            try
            {
                const std::filesystem::path outputPath = std::filesystem::path("screenshots") / job.first;
                cv::imwrite(outputPath.string(), job.second);
            }
            catch (const std::exception& e)
            {
                std::cerr << "[Capture] Screenshot save failed: " << e.what() << std::endl;
            }
            catch (...)
            {
                std::cerr << "[Capture] Screenshot save failed: unknown exception." << std::endl;
            }
        }
    }

private:
    static constexpr size_t maxPendingFrames_ = 8;

    std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<std::pair<std::string, cv::Mat>> queue_;
    std::thread writerThread_;
    bool stop_{ false };
};
} // namespace

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
        CaptureThreadConfig currentCfg = SnapshotCaptureConfig();
        if (currentCfg.verbose)
            std::cout << "[Capture] OpenCV version: " << CV_VERSION << std::endl;

        int captureWidth = std::max(1, CAPTURE_WIDTH);
        int captureHeight = std::max(1, CAPTURE_HEIGHT);
        if (currentCfg.detection_resolution > 0)
        {
            captureWidth = currentCfg.detection_resolution;
            captureHeight = currentCfg.detection_resolution;
        }

        WinrtApartmentGuard winrtApartment;
        auto createCapturer = [&](const CaptureThreadConfig& cfg, int width, int height) -> std::unique_ptr<IScreenCapture>
        {
            const std::string method = NormalizeCaptureMethod(cfg.capture_method);
            if (method != cfg.capture_method)
                std::cout << "[Capture] Unknown capture_method '" << cfg.capture_method << "'. Falling back to duplication_api." << std::endl;

            winrtApartment.Ensure(method == "winrt");

            if (method == "duplication_api")
            {
                if (cfg.verbose)
                    std::cout << "[Capture] Using Duplication API" << std::endl;
                return std::make_unique<DuplicationAPIScreenCapture>(width, height, cfg.monitor_idx);
            }

            if (method == "winrt")
            {
                if (cfg.verbose)
                    std::cout << "[Capture] Using WinRT" << std::endl;

                WinRTScreenCapture::Options options;
                options.target = cfg.capture_target;
                options.windowTitle = cfg.capture_window_title;
                options.monitorIndex = cfg.monitor_idx;
                options.captureBorders = cfg.capture_borders;
                options.captureCursor = cfg.capture_cursor;

                return std::make_unique<WinRTScreenCapture>(width, height, options);
            }

            if (method == "virtual_camera")
            {
                if (cfg.verbose)
                    std::cout << "[Capture] Using Virtual Camera" << std::endl;
                return std::make_unique<VirtualCameraCapture>(
                    cfg.virtual_camera_width,
                    cfg.virtual_camera_heigth,
                    cfg.virtual_camera_name,
                    cfg.capture_fps,
                    cfg.verbose
                );
            }

            if (cfg.verbose)
                std::cout << "[Capture] Using UDP capture" << std::endl;
            return std::make_unique<UDPCapture>(width, height, cfg.udp_ip, cfg.udp_port);
        };

        std::unique_ptr<IScreenCapture> capturer = createCapturer(currentCfg, captureWidth, captureHeight);

        TimerResolutionGuard timerResolution;
        std::optional<std::chrono::steady_clock::duration> frameDuration;
        auto updateFrameDuration = [&](int captureFpsSetting)
        {
            if (captureFpsSetting > 0)
            {
                timerResolution.Enable();
                const auto frameMs = std::chrono::duration<double, std::milli>(1000.0 / captureFpsSetting);
                frameDuration = std::chrono::duration_cast<std::chrono::steady_clock::duration>(frameMs);
            }
            else
            {
                timerResolution.Disable();
                frameDuration.reset();
            }
        };
        updateFrameDuration(currentCfg.capture_fps);

        captureFpsStartTime = std::chrono::high_resolution_clock::now();

        auto frameStartTime = std::chrono::steady_clock::now();
        auto applyFrameLimiter = [&]()
        {
            if (frameDuration.has_value())
            {
                const auto now = std::chrono::steady_clock::now();
                const auto elapsed = now - frameStartTime;
                if (elapsed < frameDuration.value())
                {
                    std::this_thread::sleep_for(frameDuration.value() - elapsed);
                }
            }
            frameStartTime = std::chrono::steady_clock::now();
        };

        ScreenshotWriter screenshotWriter;
        auto lastSaveTime = std::chrono::steady_clock::now();

        while (!shouldExit)
        {
            currentCfg = SnapshotCaptureConfig();

            if (capture_fps_changed.exchange(false))
            {
                updateFrameDuration(currentCfg.capture_fps);
            }

            const bool needsReinit =
                detection_resolution_changed.exchange(false) ||
                capture_method_changed.exchange(false) ||
                capture_cursor_changed.exchange(false) ||
                capture_borders_changed.exchange(false) ||
                capture_window_changed.exchange(false);

            if (needsReinit)
            {
                if (currentCfg.detection_resolution > 0)
                {
                    captureWidth = currentCfg.detection_resolution;
                    captureHeight = currentCfg.detection_resolution;
                }

                capturer = createCapturer(currentCfg, captureWidth, captureHeight);
                if (currentCfg.verbose)
                    std::cout << "[Capture] Reinitialized capture backend." << std::endl;
            }

            if (!capturer)
            {
                if (!frameDuration.has_value())
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                applyFrameLimiter();
                continue;
            }

            cv::Mat screenshotCpu;
            screenshotCpu = capturer->GetNextFrameCpu();

            if (screenshotCpu.empty())
            {
                if (!frameDuration.has_value())
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                applyFrameLimiter();
                continue;
            }

            if (NormalizeCaptureMethod(currentCfg.capture_method) == "virtual_camera")
            {
                const int targetW = std::max(1, captureWidth);
                const int targetH = std::max(1, captureHeight);
                const int roiW = std::min(targetW, screenshotCpu.cols);
                const int roiH = std::min(targetH, screenshotCpu.rows);

                if (roiW <= 0 || roiH <= 0)
                {
                    applyFrameLimiter();
                    continue;
                }

                const int x = std::max(0, (screenshotCpu.cols - roiW) / 2);
                const int y = std::max(0, (screenshotCpu.rows - roiH) / 2);
                cv::Mat centered = screenshotCpu(cv::Rect(x, y, roiW, roiH));

                if (roiW != targetW || roiH != targetH)
                {
                    cv::resize(centered, screenshotCpu, cv::Size(targetW, targetH), 0, 0, cv::INTER_LINEAR);
                }
                else
                {
                    screenshotCpu = centered;
                }
            }

            if (currentCfg.circle_mask)
                screenshotCpu = apply_circle_mask(screenshotCpu);

            cv::Mat detectionFrame = screenshotCpu;
#ifdef USE_CUDA
            static bool lastDepthInferenceEnabled = true;
            if (!currentCfg.depth_inference_enabled)
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

            if (currentCfg.depth_inference_enabled && currentCfg.depth_mask_enabled)
            {
                if (currentCfg.verbose)
                {
                    static auto lastMaskLog = std::chrono::steady_clock::time_point::min();
                    auto now = std::chrono::steady_clock::now();
                    if (now - lastMaskLog > std::chrono::seconds(2))
                    {
                        std::cout << "[DepthMask] update frame " << screenshotCpu.cols << "x" << screenshotCpu.rows
                                  << " model=" << currentCfg.depth_model_path
                                  << " fps=" << currentCfg.depth_mask_fps
                                  << " near=" << currentCfg.depth_mask_near_percent
                                  << " invert=" << (currentCfg.depth_mask_invert ? "true" : "false")
                                  << std::endl;
                        lastMaskLog = now;
                    }
                }

                depth_anything::DepthMaskOptions maskOptions;
                maskOptions.enabled = currentCfg.depth_mask_enabled;
                maskOptions.fps = currentCfg.depth_mask_fps;
                maskOptions.near_percent = currentCfg.depth_mask_near_percent;
                maskOptions.invert = currentCfg.depth_mask_invert;

                auto& depthMask = depth_anything::GetDepthMaskGenerator();
                depthMask.update(screenshotCpu, maskOptions, currentCfg.depth_model_path, gLogger);

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
                latestFrame = screenshotCpu;
                if (frameQueue.size() >= 1)
                    frameQueue.pop_front();
                frameQueue.push_back(latestFrame);
            }
            frameCV.notify_one();

            if (currentCfg.backend == "DML" && dml_detector)
            {
                dml_detector->processFrame(detectionFrame);
            }
#ifdef USE_CUDA
            else if (currentCfg.backend == "TRT")
            {
                trt_detector.processFrame(detectionFrame);
            }
#endif
            if (!currentCfg.screenshot_button.empty() && currentCfg.screenshot_button[0] != "None")
            {
                bool buttonPressed = isAnyKeyPressed(currentCfg.screenshot_button);
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastSaveTime).count();
                if (buttonPressed && elapsed >= currentCfg.screenshot_delay)
                {
                    cv::Mat saveMat = screenshotCpu.clone();

                    if (!saveMat.empty())
                    {
                        auto epoch_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch()
                        ).count();
                        std::string filename = std::to_string(epoch_time) + ".jpg";
                        screenshotWriter.Enqueue(filename, std::move(saveMat));
                    }

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

            applyFrameLimiter();
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Capture] Unhandled exception: " << e.what() << std::endl;
        throw;
    }
}
