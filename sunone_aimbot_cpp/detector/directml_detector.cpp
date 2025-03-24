#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include "directml_detector.h"
#include <dml_provider_factory.h>

#include "sunone_aimbot_cpp.h"
#include "postProcess.h"

DirectMLDetector::DirectMLDetector(const std::string& model_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "DML_Detector")
{
    session_options.DisableMemPattern();
    session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0);
    initializeModel(model_path);
}

DirectMLDetector::~DirectMLDetector()
{
}

void DirectMLDetector::initializeModel(const std::string& model_path)
{
    std::wstring model_path_wide(model_path.begin(), model_path.end());
    session = Ort::Session(env, model_path_wide.c_str(), session_options);

    input_name = session.GetInputNameAllocated(0, allocator).get();
    output_name = session.GetOutputNameAllocated(0, allocator).get();

    Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    input_shape = input_tensor_info.GetShape();
}

std::vector<Detection> DirectMLDetector::detect(const cv::Mat& input_frame)
{
    std::lock_guard<std::mutex> lock(inference_mutex);

    if (input_frame.empty())
    {
        std::cerr << "[DirectMLDetector] Empty input frame." << std::endl;
        return {};
    }

    cv::Mat resized_frame;
    int target_width = 640;
    int target_height = 640;
    cv::resize(input_frame, resized_frame, cv::Size(target_width, target_height));
    resized_frame.convertTo(resized_frame, CV_32FC3, 1.0f / 255.0f);

    const size_t channels = 3;
    const size_t height = resized_frame.rows;
    const size_t width = resized_frame.cols;
    const size_t tensor_size = channels * height * width;

    std::vector<float> input_tensor_values(tensor_size);
    std::vector<cv::Mat> chw_planes(channels);
    cv::split(resized_frame, chw_planes);

    for (size_t c = 0; c < channels; ++c)
    {
        memcpy(input_tensor_values.data() + c * height * width,
            chw_planes[c].data,
            height * width * sizeof(float));
    }

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<int64_t> dynamic_input_shape = { 1, 3, 640, 640 };

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), tensor_size, dynamic_input_shape.data(), dynamic_input_shape.size());

    const char* input_names[] = { input_name.c_str() };
    const char* output_names[] = { output_name.c_str() };

    auto output_tensors = session.Run(Ort::RunOptions{ nullptr },
        input_names, &input_tensor, 1,
        output_names, 1);

    float* output_data = output_tensors.front().GetTensorMutableData<float>();

    Ort::TensorTypeAndShapeInfo output_info = output_tensors.front().GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = output_info.GetShape();

    std::vector<Detection> detections;

    int num_classes = output_shape[1] - 4;
    int rows = output_shape[1];
    int cols = output_shape[2];

    cv::Mat det_output(rows, cols, CV_32F, output_data);

    const float conf_threshold = config.confidence_threshold;
    const float nms_threshold = config.nms_threshold;
    const float img_scale_x = input_frame.cols / static_cast<float>(width);
    const float img_scale_y = input_frame.rows / static_cast<float>(height);

    for (int i = 0; i < cols; ++i)
    {
        cv::Mat classes_scores = det_output.col(i).rowRange(4, 4 + num_classes);

        cv::Point class_id_point;
        double score;
        cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        if (score > conf_threshold)
        {
            float cx = det_output.at<float>(0, i) * img_scale_x;
            float cy = det_output.at<float>(1, i) * img_scale_y;
            float ow = det_output.at<float>(2, i) * img_scale_x;
            float oh = det_output.at<float>(3, i) * img_scale_y;

            cv::Rect box(
                static_cast<int>(cx - 0.5f * ow),
                static_cast<int>(cy - 0.5f * oh),
                static_cast<int>(ow),
                static_cast<int>(oh)
            );

            detections.push_back(Detection{ box, static_cast<float>(score), class_id_point.y });
        }
    }

    NMS(detections, nms_threshold);

    return detections;
}

int DirectMLDetector::getNumberOfClasses()
{
    Ort::TypeInfo output_type_info = session.GetOutputTypeInfo(0);
    auto tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = tensor_info.GetShape();

    if (output_shape.size() == 3)
    {
        int num_classes = static_cast<int>(output_shape[1]) - 4;
        return num_classes;
    }
    else
    {
        std::cerr << "[DirectMLDetector] Unexpected output tensor shape." << std::endl;
        return -1;
    }
}