#pragma once
#ifdef USE_CUDA

#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

#include "depth_utils.h"

namespace depth_anything
{
    class DepthAnythingTrt
    {
    public:
        DepthAnythingTrt();
        ~DepthAnythingTrt();

        bool initialize(const std::string& modelPath, nvinfer1::ILogger& logger);
        cv::Mat predict(const cv::Mat& image);
        bool ready() const;
        const std::string& lastError() const;
        void reset();

    private:
        int input_w;
        int input_h;
        int min_input_size;
        int max_input_size;
        bool dynamic_input;
        float mean[3];
        float std[3];

        std::unique_ptr<nvinfer1::IRuntime> runtime;
        std::unique_ptr<nvinfer1::ICudaEngine> engine;
        std::unique_ptr<nvinfer1::IExecutionContext> context;

        void* buffer[2];
        std::vector<float> depth_data;
        cudaStream_t stream;

        bool initialized;
        std::string last_error;

        std::vector<float> preprocess(const cv::Mat& image);
        int selectInputSize(const cv::Mat& image) const;
        bool setInputShape(int w, int h);
        bool loadEngine(const std::string& modelPath, nvinfer1::ILogger& logger);
        bool buildEngine(const std::string& onnxPath, nvinfer1::ILogger& logger);
        bool saveEngine(const std::string& onnxPath);
    };
}

#endif
