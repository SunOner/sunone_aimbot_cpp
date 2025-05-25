#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>
#include <dml_provider_factory.h>
#include <wrl/client.h>
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <dxgi.h>

#include "dml_detector.h"
#include "sunone_aimbot_cpp.h"
#include "postProcess.h"
#include "capture.h"

extern std::atomic<bool> detector_model_changed;
extern std::atomic<bool> detection_resolution_changed;

std::string GetDMLDeviceName(int deviceId)
{
    Microsoft::WRL::ComPtr<IDXGIFactory1> dxgiFactory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory))))
        return "Unknown";

    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
    if (FAILED(dxgiFactory->EnumAdapters1(deviceId, &adapter)))
        return "Invalid device ID";

    DXGI_ADAPTER_DESC1 desc;
    if (FAILED(adapter->GetDesc1(&desc)))
        return "Failed to get description";

    std::wstring wname(desc.Description);
    return std::string(wname.begin(), wname.end());
}

DirectMLDetector::DirectMLDetector(const std::string& model_path)
    :
    env(ORT_LOGGING_LEVEL_WARNING, "DML_Detector"),
    memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(session_options, config.dml_device_id));

    if (config.verbose)
        std::cout << "[DirectML] Using adapter: " << GetDMLDeviceName(config.dml_device_id) << std::endl;

    session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    session_options.SetIntraOpNumThreads(std::thread::hardware_concurrency());

    initializeModel(model_path);
}

DirectMLDetector::~DirectMLDetector()
{
    shouldExit = true;
    inferenceCV.notify_all();
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
    std::vector<cv::Mat> batch = { input_frame };
    auto batchResult = detectBatch(batch);
    if (!batchResult.empty())
        return batchResult[0];
    else
        return {};
}

std::vector<std::vector<Detection>> DirectMLDetector::detectBatch(const std::vector<cv::Mat>& frames)
{
    int batch_size = frames.size();
    int target_width = config.detection_resolution;
    int target_height = config.detection_resolution;

    std::vector<float> input_tensor_values(batch_size * 3 * target_height * target_width);

    for (int b = 0; b < batch_size; ++b)
    {
        cv::Mat resized_frame;
        cv::resize(frames[b], resized_frame, cv::Size(target_width, target_height));
        cv::cvtColor(resized_frame, resized_frame, cv::COLOR_BGR2RGB);
        resized_frame.convertTo(resized_frame, CV_32FC3, 1.0f / 255.0f);

        const float* input_data = reinterpret_cast<const float*>(resized_frame.data);
        for (int h = 0; h < target_height; ++h)
            for (int w = 0; w < target_width; ++w)
                for (int c = 0; c < 3; ++c)
                    input_tensor_values[
                        b * 3 * target_height * target_width +
                            c * target_height * target_width +
                            h * target_width + w
                    ] = input_data[(h * target_width + w) * 3 + c];
    }

    std::vector<int64_t> input_shape = { batch_size, 3, target_height, target_width };

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size());

    const char* input_names[] = { input_name.c_str() };
    const char* output_names[] = { output_name.c_str() };

    auto output_tensors = session.Run(Ort::RunOptions{ nullptr },
        input_names, &input_tensor, 1,
        output_names, 1);

    float* output_data = output_tensors.front().GetTensorMutableData<float>();
    Ort::TensorTypeAndShapeInfo output_info = output_tensors.front().GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = output_info.GetShape();

    int batch_out = output_shape[0];
    int rows = output_shape[1];
    int cols = output_shape[2];
    int num_classes = rows - 4;

    std::vector<std::vector<Detection>> batchDetections(batch_out);

    float conf_threshold = config.confidence_threshold;
    float nms_threshold = config.nms_threshold;

    for (int b = 0; b < batch_out; ++b)
    {
        const float* out_ptr = output_data + b * rows * cols;
        std::vector<Detection> detections;

        if (config.postprocess == "yolo10")
        {
            std::vector<int64_t> shape = { batch_out, rows, cols };
            detections = postProcessYolo10DML(
                out_ptr,
                shape,
                num_classes,
                conf_threshold,
                nms_threshold
            );
        }
        else
        {
            std::vector<int64_t> shape = { rows, cols };
            detections = postProcessYolo11DML(
                out_ptr,
                shape,
                num_classes,
                conf_threshold,
                nms_threshold
            );
        }

        batchDetections[b] = std::move(detections);
    }

    return batchDetections;
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

void DirectMLDetector::processFrame(const cv::Mat& frame)
{
    std::unique_lock<std::mutex> lock(inferenceMutex);
    currentFrame = frame.clone();
    frameReady = true;
    inferenceCV.notify_one();
}

void DirectMLDetector::dmlInferenceThread()
{
    while (!shouldExit)
    {
        if (detector_model_changed.load())
        {
            initializeModel("models/" + config.ai_model);
            detection_resolution_changed.store(true);
            detector_model_changed.store(false);
            std::cout << "[DML] Detector reloaded: " << config.ai_model << std::endl;
        }

        cv::Mat frame;
        bool hasNewFrame = false;
        {
            std::unique_lock<std::mutex> lock(inferenceMutex);
            if (!frameReady && !shouldExit)
                inferenceCV.wait(lock, [this] { return frameReady || shouldExit; });

            if (shouldExit) break;

            if (frameReady)
            {
                frame = std::move(currentFrame);
                frameReady = false;
                hasNewFrame = true;
            }
        }

        if (hasNewFrame && !frame.empty())
        {
            auto start = std::chrono::steady_clock::now();

            std::vector<cv::Mat> batchFrames = { frame };
            auto detectionsBatch = detectBatch(batchFrames);
            const std::vector<Detection>& detections = detectionsBatch.back();

            auto end = std::chrono::steady_clock::now();
            lastInferenceTimeDML = end - start;

            std::lock_guard<std::mutex> lock(detectionBuffer.mutex);
            detectionBuffer.boxes.clear();
            detectionBuffer.classes.clear();
            for (const auto& d : detections) {
                detectionBuffer.boxes.push_back(d.box);
                detectionBuffer.classes.push_back(d.classId);
            }
            detectionBuffer.version++;
            detectionBuffer.cv.notify_all();
        }
    }
}