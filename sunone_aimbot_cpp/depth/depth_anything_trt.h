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
        cv::Mat predictDepth(const cv::Mat& image);
        void setColormap(int type);
        int colormapType() const;
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
        float stddev[3];
        int colormap_type;

        std::unique_ptr<nvinfer1::IRuntime> runtime;
        std::unique_ptr<nvinfer1::ICudaEngine> engine;
        std::unique_ptr<nvinfer1::IExecutionContext> context;

        std::string input_name;
        std::string output_name;
        void* input_buffer;
        void* output_buffer;
        size_t input_capacity;
        size_t output_capacity;
        int output_w;
        int output_h;
        std::vector<float> depth_data;
        cudaStream_t stream;

        bool initialized;
        std::string last_error;

        bool preprocess(const cv::Mat& image, std::vector<float>& input_tensor);
        bool getOutputShape(int& out_h, int& out_w, size_t& out_elements);
        bool ensureInputCapacity(size_t elements);
        bool ensureOutputCapacity(size_t elements);
        bool setTensorAddresses();
        int selectInputSize(const cv::Mat& image) const;
        bool setInputShape(int w, int h);
        bool runInference(const cv::Mat& image, cv::Mat& depth_norm);
        bool loadEngine(const std::string& modelPath, nvinfer1::ILogger& logger);
        bool buildEngine(const std::string& onnxPath, nvinfer1::ILogger& logger);
        bool saveEngine(const std::string& onnxPath);
    };
}

#endif
