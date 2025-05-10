#ifndef DIRECTML_DETECTOR_H
#define DIRECTML_DETECTOR_H

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <mutex>

#include "postProcess.h"

class DirectMLDetector
{
public:
    DirectMLDetector(const std::string& model_path);
    ~DirectMLDetector();

    std::vector<Detection> detect(const cv::Mat& input_frame);
    int getNumberOfClasses();
private:
    Ort::Env env;
    Ort::Session session{ nullptr };
    Ort::SessionOptions session_options;
    Ort::AllocatorWithDefaultOptions allocator;

    std::string input_name;
    std::string output_name;
    std::vector<int64_t> input_shape;

    std::mutex inference_mutex;

    void initializeModel(const std::string& model_path);
    Ort::MemoryInfo memory_info;
};

#endif // DIRECTML_DETECTOR_H