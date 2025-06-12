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

std::chrono::duration<double, std::milli> lastPreprocessTimeDML{};
std::chrono::duration<double, std::milli> lastCopyTimeDML{};
std::chrono::duration<double, std::milli> lastPostprocessTimeDML{};
std::chrono::duration<double, std::milli> lastNmsTimeDML{};

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

    bool isStatic = true;
    for (auto d : input_shape) if (d <= 0) isStatic = false;

    if (isStatic != config.fixed_input_size)
    {
        config.fixed_input_size = isStatic;
        config.saveConfig();
        detector_model_changed.store(true);
        std::cout << "[DML] Automatically set fixed_input_size = " << (isStatic ? "true" : "false") << std::endl;
    }
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
    std::vector<std::vector<Detection>> empty;
    if (frames.empty()) return empty;

    const int batch_size = static_cast<int>(frames.size());

    int model_h = (input_shape.size() > 2) ? static_cast<int>(input_shape[2]) : -1;
    int model_w = (input_shape.size() > 3) ? static_cast<int>(input_shape[3]) : -1;
    const bool useFixed = config.fixed_input_size && model_h > 0 && model_w > 0;

    const int target_h = useFixed ? model_h : config.detection_resolution;
    const int target_w = useFixed ? model_w : config.detection_resolution;

    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> input_tensor_values(batch_size * 3 * target_h * target_w);

    for (int b = 0; b < batch_size; ++b)
    {
        cv::Mat resized;
        cv::resize(frames[b], resized, cv::Size(target_w, target_h));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        resized.convertTo(resized, CV_32FC3, 1.0f / 255.0f);

        const float* src = reinterpret_cast<const float*>(resized.data);
        for (int h = 0; h < target_h; ++h)
            for (int w = 0; w < target_w; ++w)
                for (int c = 0; c < 3; ++c)
                {
                    size_t dstIdx = b * 3 * target_h * target_w + c * target_h * target_w + h * target_w + w;
                    input_tensor_values[dstIdx] = src[(h * target_w + w) * 3 + c];
                }
    }
    auto t1 = std::chrono::steady_clock::now();

    std::vector<int64_t> ort_input_shape{ batch_size, 3, target_h, target_w };
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        ort_input_shape.data(), ort_input_shape.size());

    const char* input_names[] = { input_name.c_str() };
    const char* output_names[] = { output_name.c_str() };

    auto t2 = std::chrono::steady_clock::now();
    auto output_tensors = session.Run(Ort::RunOptions{ nullptr },
        input_names, &input_tensor, 1,
        output_names, 1);
    auto t3 = std::chrono::steady_clock::now();

    float* outData = output_tensors.front().GetTensorMutableData<float>();
    Ort::TensorTypeAndShapeInfo outInfo = output_tensors.front().GetTensorTypeAndShapeInfo();
    std::vector<int64_t> outShape = outInfo.GetShape(); // [B, rows, cols]

    int rows = static_cast<int>(outShape[1]);
    int cols = static_cast<int>(outShape[2]);
    const int num_classes = rows - 4;

    std::vector<std::vector<Detection>> batchDetections(batch_size);
    float conf_thr = config.confidence_threshold;
    float nms_thr = config.nms_threshold;

    auto t4 = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> nmsTimeTmp{ 0 };

    for (int b = 0; b < batch_size; ++b)
    {
        const float* ptr = outData + b * rows * cols;
        std::vector<Detection> detections;

        if (config.postprocess == "yolo10")
        {
            std::vector<int64_t> shp = { batch_size, rows, cols };
            detections = postProcessYolo10DML(ptr, shp, num_classes, conf_thr, nms_thr, &nmsTimeTmp);
        }
        else
        {
            std::vector<int64_t> shp = { rows, cols };
            detections = postProcessYolo11DML(ptr, shp, num_classes, conf_thr, nms_thr, &nmsTimeTmp);
        }

        if (useFixed && (target_w != config.detection_resolution))
        {
            float scale = static_cast<float>(config.detection_resolution) / target_w;
            for (auto& d : detections)
            {
                d.box.x = static_cast<int>(d.box.x * scale);
                d.box.y = static_cast<int>(d.box.y * scale);
                d.box.width = static_cast<int>(d.box.width * scale);
                d.box.height = static_cast<int>(d.box.height * scale);
            }
        }

        batchDetections[b] = std::move(detections);
    }
    auto t5 = std::chrono::steady_clock::now();

    lastPreprocessTimeDML = t1 - t0;
    lastInferenceTimeDML = t3 - t2;
    lastCopyTimeDML = t4 - t3;
    lastPostprocessTimeDML = t5 - t4;
    lastNmsTimeDML = nmsTimeTmp;

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