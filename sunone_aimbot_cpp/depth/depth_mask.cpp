#ifdef USE_CUDA

#include "depth_mask.h"

#include <algorithm>
#include <vector>

#include "depth_anything_trt.h"

namespace depth_anything
{
    void DepthMaskGenerator::reset()
    {
        std::lock_guard<std::mutex> lk(state_mutex);
        mask_binary.release();
        last_error.clear();
        last_model_path.clear();
        initialized = false;
        last_update = std::chrono::steady_clock::time_point::min();
        last_attempt = std::chrono::steady_clock::time_point::min();
        last_frame_w = 0;
        last_frame_h = 0;
        if (model)
        {
            delete model;
            model = nullptr;
        }
    }

    bool DepthMaskGenerator::ready() const
    {
        std::lock_guard<std::mutex> lk(state_mutex);
        return initialized && model && model->ready();
    }

    std::string DepthMaskGenerator::lastError() const
    {
        std::lock_guard<std::mutex> lk(state_mutex);
        return last_error;
    }

    std::chrono::steady_clock::time_point DepthMaskGenerator::lastAttemptTime() const
    {
        std::lock_guard<std::mutex> lk(state_mutex);
        return last_attempt;
    }

    std::pair<int, int> DepthMaskGenerator::lastFrameSize() const
    {
        std::lock_guard<std::mutex> lk(state_mutex);
        return { last_frame_w, last_frame_h };
    }

    DepthMaskDebugState DepthMaskGenerator::debugState() const
    {
        std::lock_guard<std::mutex> lk(state_mutex);
        DepthMaskDebugState state;
        state.initialized = initialized;
        state.has_model = (model != nullptr);
        state.model_ready = (model != nullptr) ? model->ready() : false;
        state.last_model_path = last_model_path;
        return state;
    }

    cv::Mat DepthMaskGenerator::getMask() const
    {
        std::lock_guard<std::mutex> lk(state_mutex);
        return mask_binary.clone();
    }

    void DepthMaskGenerator::update(const cv::Mat& frame, const DepthMaskOptions& options,
        const std::string& modelPath, nvinfer1::ILogger& logger)
    {
        if (!options.enabled)
            return;
        const auto now = std::chrono::steady_clock::now();
        if (frame.empty())
        {
            std::lock_guard<std::mutex> lk(state_mutex);
            last_error = "Depth mask frame is empty.";
            last_attempt = now;
            last_frame_w = 0;
            last_frame_h = 0;
            return;
        }

        std::lock_guard<std::mutex> lk(state_mutex);
        last_attempt = now;
        last_frame_w = frame.cols;
        last_frame_h = frame.rows;

        if (!model)
            model = new DepthAnythingTrt();

        if (modelPath.empty())
        {
            last_error = "Depth mask model path is empty.";
            return;
        }

        if (!initialized || modelPath != last_model_path || !model->ready())
        {
            if (!model->initialize(modelPath, logger))
            {
                last_error = model->lastError();
                initialized = false;
                return;
            }
            last_model_path = modelPath;
            initialized = true;
            last_error.clear();
        }

        const int fps = options.fps > 0 ? options.fps : 5;
        const auto interval = std::chrono::milliseconds(1000 / fps);
        if (now - last_update < interval)
            return;

        last_update = now;

        cv::Mat depth_norm = model->predictDepth(frame);
        if (depth_norm.empty())
        {
            last_error = model->lastError();
            if (last_error.empty())
                last_error = "Depth mask inference returned empty output.";
            return;
        }

        int near_percent = std::clamp(options.near_percent, 1, 100);

        const int total = depth_norm.rows * depth_norm.cols;
        if (total <= 0)
        {
            return;
        }

        const int target = std::max(1, (total * near_percent) / 100);
        int hist[256] = {};
        for (int y = 0; y < depth_norm.rows; ++y)
        {
            const uint8_t* row = depth_norm.ptr<uint8_t>(y);
            for (int x = 0; x < depth_norm.cols; ++x)
            {
                hist[row[x]]++;
            }
        }

        int threshold = 0;
        if (!options.invert)
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
        }

        cv::Mat mask;
        if (!options.invert)
            cv::compare(depth_norm, threshold, mask, cv::CMP_LE);
        else
            cv::compare(depth_norm, threshold, mask, cv::CMP_GE);

        mask_binary = std::move(mask);
    }

    DepthMaskGenerator& GetDepthMaskGenerator()
    {
        static DepthMaskGenerator generator;
        return generator;
    }
}

#endif
