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
#include "depth/depth_anything_trt.h"
#include "depth/depth_mask.h"
#include "tensorrt/nvinf.h"
#endif
#include "sunone_aimbot_2.h"
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

#ifdef USE_CUDA
namespace
{
std::mutex g_detectionSuppressionMaskMutex;
cv::Mat g_detectionSuppressionMask;
}

static void UpdateDetectionSuppressionMask(const cv::Mat& mask)
{
    std::lock_guard<std::mutex> lock(g_detectionSuppressionMaskMutex);
    if (!mask.empty() && mask.type() == CV_8UC1)
        g_detectionSuppressionMask = mask.clone();
    else
        g_detectionSuppressionMask.release();
}

cv::Mat getCurrentDetectionSuppressionMask()
{
    std::lock_guard<std::mutex> lock(g_detectionSuppressionMaskMutex);
    return g_detectionSuppressionMask.clone();
}
#endif

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
    bool show_window = false;
    bool verbose = false;
#ifdef USE_CUDA
    bool depth_inference_enabled = false;
    bool depth_mask_enabled = false;
    std::string depth_model_path;
    int depth_mask_fps = 0;
    int depth_mask_near_percent = 0;
    int depth_mask_expand = 0;
    bool depth_mask_invert = false;
    bool capture_use_cuda = true;
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
    snapshot.show_window = config.show_window;
    snapshot.verbose = config.verbose;
#ifdef USE_CUDA
    snapshot.depth_inference_enabled = config.depth_inference_enabled;
    snapshot.depth_mask_enabled = config.depth_mask_enabled;
    snapshot.depth_model_path = config.depth_model_path;
    snapshot.depth_mask_fps = config.depth_mask_fps;
    snapshot.depth_mask_near_percent = config.depth_mask_near_percent;
    snapshot.depth_mask_expand = config.depth_mask_expand;
    snapshot.depth_mask_invert = config.depth_mask_invert;
    snapshot.capture_use_cuda = config.capture_use_cuda;
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

#ifdef USE_CUDA
        depth_anything::DepthAnythingTrt depthMaskFallbackModel;
        std::string depthMaskFallbackModelPath;
#endif

        WinrtApartmentGuard winrtApartment;
        auto createCapturer = [&](const CaptureThreadConfig& cfg, int width, int height) -> std::unique_ptr<IScreenCapture>
        {
            try
            {
                const std::string method = NormalizeCaptureMethod(cfg.capture_method);
                if (method != cfg.capture_method)
                    std::cout << "[Capture] Unknown capture_method '" << cfg.capture_method << "'. Falling back to duplication_api." << std::endl;

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
            }
            catch (const std::exception& e)
            {
                std::cerr << "[Capture] Failed to initialize '" << cfg.capture_method
                    << "' capture: " << e.what() << std::endl;
                return nullptr;
            }
        };

        std::string desiredCaptureMethod = NormalizeCaptureMethod(currentCfg.capture_method);
        winrtApartment.Ensure(desiredCaptureMethod == "winrt");

        std::unique_ptr<IScreenCapture> capturer = createCapturer(currentCfg, captureWidth, captureHeight);
        std::string activeCapturerMethod = capturer ? desiredCaptureMethod : std::string();
        auto lastCapturerCreateAttempt = std::chrono::steady_clock::now();

        auto clearCaptureFrames = [&]()
        {
            std::lock_guard<std::mutex> lock(frameMutex);
            latestFrame.release();
            frameQueue.clear();
        };

        auto clearDetections = [&]()
        {
            std::lock_guard<std::mutex> lock(detectionBuffer.mutex);
            detectionBuffer.boxes.clear();
            detectionBuffer.classes.clear();
            detectionBuffer.version++;
            detectionBuffer.cv.notify_all();
        };

        auto markCaptureUnavailable = [&]()
        {
            clearCaptureFrames();
            clearDetections();
            frameCV.notify_one();
        };

        bool captureUnavailable = false;
        auto setCaptureUnavailable = [&]()
        {
            if (captureUnavailable)
                return;
            captureUnavailable = true;
            markCaptureUnavailable();
        };
        auto setCaptureAvailable = [&]()
        {
            captureUnavailable = false;
        };

        // Do not keep stale preview/detections from previous capture state.
        setCaptureUnavailable();

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
        auto lastSuccessfulFrameTime = std::chrono::steady_clock::now();
        constexpr auto staleFrameTimeout = std::chrono::milliseconds(500);

        while (!shouldExit)
        {
            try
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
                setCaptureUnavailable();

                if (currentCfg.detection_resolution > 0)
                {
                    captureWidth = currentCfg.detection_resolution;
                    captureHeight = currentCfg.detection_resolution;
                }

                const std::string nextMethod = NormalizeCaptureMethod(currentCfg.capture_method);
                desiredCaptureMethod = nextMethod;
                const bool nextNeedsWinrt = (nextMethod == "winrt");

                // Always teardown current backend first to avoid overlap between old/new capture objects.
                // WinRT must be destroyed before apartment teardown.
                if (capturer)
                {
                    const bool activeWasWinrt = (activeCapturerMethod == "winrt");
                    capturer.reset();
                    activeCapturerMethod.clear();
                    if (activeWasWinrt && !nextNeedsWinrt)
                        winrtApartment.Ensure(false);
                }

                winrtApartment.Ensure(nextNeedsWinrt);

                if (nextMethod == "virtual_camera")
                    VirtualCameraCapture::GetAvailableVirtualCameras(true);

                capturer = createCapturer(currentCfg, captureWidth, captureHeight);
                if (capturer)
                    activeCapturerMethod = nextMethod;
                else
                    activeCapturerMethod.clear();

                lastCapturerCreateAttempt = std::chrono::steady_clock::now();
                if (currentCfg.verbose)
                    std::cout << "[Capture] Reinitialized capture backend." << std::endl;
            }

            if (!capturer)
            {
                const auto now = std::chrono::steady_clock::now();
                if (now - lastCapturerCreateAttempt >= std::chrono::seconds(1))
                {
                    desiredCaptureMethod = NormalizeCaptureMethod(currentCfg.capture_method);
                    winrtApartment.Ensure(desiredCaptureMethod == "winrt");

                    if (desiredCaptureMethod == "virtual_camera")
                        VirtualCameraCapture::GetAvailableVirtualCameras(true);

                    capturer = createCapturer(currentCfg, captureWidth, captureHeight);
                    lastCapturerCreateAttempt = now;

                    if (capturer)
                    {
                        activeCapturerMethod = desiredCaptureMethod;
                        lastSuccessfulFrameTime = now;
                        if (currentCfg.verbose)
                            std::cout << "[Capture] Capture backend recovered." << std::endl;
                    }
                    else
                    {
                        activeCapturerMethod.clear();
                    }
                }

                setCaptureUnavailable();
                if (!frameDuration.has_value())
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                applyFrameLimiter();
                continue;
            }

            const bool screenshotEnabled =
                !currentCfg.screenshot_button.empty() && currentCfg.screenshot_button[0] != "None";
            const auto screenshotNow = std::chrono::steady_clock::now();
            const auto screenshotElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                screenshotNow - lastSaveTime
            ).count();
            const bool screenshotRequested =
                screenshotEnabled &&
                isAnyKeyPressed(currentCfg.screenshot_button) &&
                screenshotElapsedMs >= currentCfg.screenshot_delay;
#ifdef USE_CUDA
            const bool needCpuCopyFromGpu = screenshotRequested || currentCfg.show_window;
#endif

            cv::Mat screenshotCpu;
            cv::Mat detectionFrame;
            bool frameSubmittedToDetector = false;

#ifdef USE_CUDA
            static bool lastDepthInferenceEnabled = true;
            if (!currentCfg.depth_inference_enabled)
            {
                if (lastDepthInferenceEnabled)
                {
                    auto& depthMask = depth_anything::GetDepthMaskGenerator();
                    depthMask.reset();
                    depthMaskFallbackModel.reset();
                    depthMaskFallbackModelPath.clear();
                }
                UpdateDetectionSuppressionMask(cv::Mat());
                lastDepthInferenceEnabled = false;
            }
            else
            {
                lastDepthInferenceEnabled = true;
            }

            const bool depthMaskEnabled = currentCfg.depth_inference_enabled && currentCfg.depth_mask_enabled;
            const bool preferGpuCapturePath =
                currentCfg.backend == "TRT" &&
                NormalizeCaptureMethod(currentCfg.capture_method) == "duplication_api" &&
                currentCfg.capture_use_cuda &&
                !currentCfg.circle_mask &&
                !depthMaskEnabled;

            if (preferGpuCapturePath)
            {
                auto* duplicationCapture = dynamic_cast<DuplicationAPIScreenCapture*>(capturer.get());
                if (duplicationCapture)
                {
                    cv::cuda::GpuMat screenshotGpu;
                    if (duplicationCapture->GetNextFrameGpu(screenshotGpu))
                    {
                        trt_detector.processFrameGpu(screenshotGpu);
                        frameSubmittedToDetector = true;

                        if (needCpuCopyFromGpu)
                            screenshotGpu.download(screenshotCpu);
                    }
                }
            }
#endif

            if (!frameSubmittedToDetector)
            {
                screenshotCpu = capturer->GetNextFrameCpu();

                if (screenshotCpu.empty())
                {
                    const auto now = std::chrono::steady_clock::now();
                    if (now - lastSuccessfulFrameTime >= staleFrameTimeout)
                        setCaptureUnavailable();

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

                detectionFrame = screenshotCpu;
#ifdef USE_CUDA
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
                                      << " expand=" << currentCfg.depth_mask_expand
                                      << " invert=" << (currentCfg.depth_mask_invert ? "true" : "false")
                                      << std::endl;
                            lastMaskLog = now;
                        }
                    }

                    cv::Mat mask;
                    depth_anything::DepthMaskOptions maskOptions;
                    maskOptions.enabled = currentCfg.depth_mask_enabled;
                    maskOptions.fps = currentCfg.depth_mask_fps;
                    maskOptions.near_percent = currentCfg.depth_mask_near_percent;
                    maskOptions.expand = currentCfg.depth_mask_expand;
                    maskOptions.invert = currentCfg.depth_mask_invert;

                    auto& depthMask = depth_anything::GetDepthMaskGenerator();
                    depthMask.update(screenshotCpu, maskOptions, currentCfg.depth_model_path, gLogger);
                    mask = depthMask.getMask();

                    if (!mask.empty() && mask.size() != screenshotCpu.size())
                        mask.release();

                    if (mask.empty())
                    {
                        if (currentCfg.depth_model_path.empty())
                        {
                            if (depthMaskFallbackModel.ready())
                                depthMaskFallbackModel.reset();
                            depthMaskFallbackModelPath.clear();
                        }
                        else if (depthMaskFallbackModelPath != currentCfg.depth_model_path || !depthMaskFallbackModel.ready())
                        {
                            if (depthMaskFallbackModel.initialize(currentCfg.depth_model_path, gLogger))
                            {
                                depthMaskFallbackModelPath = currentCfg.depth_model_path;
                            }
                        }

                        if (depthMaskFallbackModel.ready())
                        {
                            cv::Mat depthLocal = depthMaskFallbackModel.predictDepth(screenshotCpu);
                            if (!depthLocal.empty())
                            {
                                const int nearPercent = std::clamp(currentCfg.depth_mask_near_percent, 1, 100);
                                const bool invertMask = currentCfg.depth_mask_invert;
                                const int total = depthLocal.rows * depthLocal.cols;
                                if (total > 0)
                                {
                                    int hist[256] = {};
                                    for (int y = 0; y < depthLocal.rows; ++y)
                                    {
                                        const uint8_t* row = depthLocal.ptr<uint8_t>(y);
                                        for (int x = 0; x < depthLocal.cols; ++x)
                                            hist[row[x]]++;
                                    }

                                    const int target = std::max(1, (total * nearPercent) / 100);
                                    int threshold = 0;
                                    if (!invertMask)
                                    {
                                        int count = 0;
                                        for (int i = 0; i < 256; ++i)
                                        {
                                            count += hist[i];
                                            if (count >= target)
                                            {
                                                threshold = i;
                                                break;
                                            }
                                        }
                                        cv::compare(depthLocal, threshold, mask, cv::CMP_LE);
                                    }
                                    else
                                    {
                                        int count = 0;
                                        for (int i = 255; i >= 0; --i)
                                        {
                                            count += hist[i];
                                            if (count >= target)
                                            {
                                                threshold = i;
                                                break;
                                            }
                                        }
                                        cv::compare(depthLocal, threshold, mask, cv::CMP_GE);
                                    }

                                    const int expand = std::clamp(currentCfg.depth_mask_expand, 0, 128);
                                    if (expand > 0)
                                    {
                                        const int kernelSize = 2 * expand + 1;
                                        cv::Mat kernel = cv::getStructuringElement(
                                            cv::MORPH_ELLIPSE, cv::Size(kernelSize, kernelSize));
                                        cv::dilate(mask, mask, kernel);
                                    }
                                }
                            }
                        }
                    }

                    UpdateDetectionSuppressionMask(mask);
                    if (!mask.empty() && mask.size() == screenshotCpu.size())
                    {
                        detectionFrame = screenshotCpu.clone();
                        detectionFrame.setTo(cv::Scalar(0, 0, 0), mask);
                    }
                }
                else
                {
                    UpdateDetectionSuppressionMask(cv::Mat());
                }
#endif

                if (currentCfg.backend == "DML" && dml_detector)
                {
                    dml_detector->processFrame(detectionFrame, screenshotCpu);
                }
#ifdef USE_CUDA
                else if (currentCfg.backend == "TRT")
                {
                    trt_detector.processFrame(detectionFrame, screenshotCpu);
                }
#endif
            }

            if (frameSubmittedToDetector || !screenshotCpu.empty())
            {
                lastSuccessfulFrameTime = std::chrono::steady_clock::now();
                setCaptureAvailable();
            }

            if (!screenshotCpu.empty())
            {
                std::lock_guard<std::mutex> lock(frameMutex);
                latestFrame = screenshotCpu;
                if (frameQueue.size() >= 1)
                    frameQueue.pop_front();
                frameQueue.push_back(latestFrame);
            }
            frameCV.notify_one();

            if (screenshotRequested)
            {
                cv::Mat saveMat = screenshotCpu.clone();
                if (!saveMat.empty())
                {
                    auto epoch_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()
                    ).count();
                    std::string filename = std::to_string(epoch_time) + ".jpg";
                    screenshotWriter.Enqueue(filename, std::move(saveMat));
                    lastSaveTime = screenshotNow;
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
            catch (const std::exception& e)
            {
                std::cerr << "[Capture] Loop exception: " << e.what() << std::endl;
                capturer.reset();
                activeCapturerMethod.clear();
                winrtApartment.Ensure(false);
                setCaptureUnavailable();
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            catch (...)
            {
                std::cerr << "[Capture] Loop exception: unknown." << std::endl;
                capturer.reset();
                activeCapturerMethod.clear();
                winrtApartment.Ensure(false);
                setCaptureUnavailable();
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Capture] Unhandled exception: " << e.what() << std::endl;
        throw;
    }
}
