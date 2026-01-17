#pragma once
#ifdef USE_CUDA

#include <opencv2/opencv.hpp>
#include <chrono>
#include <mutex>
#include <string>

namespace nvinfer1
{
    class ILogger;
}

namespace depth_anything
{
    class DepthAnythingTrt;

    struct DepthMaskOptions
    {
        bool enabled = false;
        int fps = 5;
        int near_percent = 20;
        bool invert = false;
    };

    struct DepthMaskDebugState
    {
        bool initialized = false;
        bool has_model = false;
        bool model_ready = false;
        std::string last_model_path;
    };

    class DepthMaskGenerator
    {
    public:
        void update(const cv::Mat& frame, const DepthMaskOptions& options,
            const std::string& modelPath, nvinfer1::ILogger& logger);
        cv::Mat getMask() const;
        bool ready() const;
        std::string lastError() const;
        std::chrono::steady_clock::time_point lastAttemptTime() const;
        std::pair<int, int> lastFrameSize() const;
        DepthMaskDebugState debugState() const;
        void reset();

    private:
        mutable std::mutex state_mutex;
        cv::Mat mask_binary;
        std::chrono::steady_clock::time_point last_update = std::chrono::steady_clock::time_point::min();
        std::chrono::steady_clock::time_point last_attempt = std::chrono::steady_clock::time_point::min();
        int last_frame_w = 0;
        int last_frame_h = 0;
        std::string last_model_path;
        std::string last_error;
        bool initialized = false;

        class DepthAnythingTrt* model = nullptr;
    };

    DepthMaskGenerator& GetDepthMaskGenerator();
}

#endif
