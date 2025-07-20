#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#define NOMINMAX

#include <winsock2.h>
#include <Windows.h>
#include <dml_provider_factory.h>
#include <wrl/client.h>
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <dxgi.h>
#include <opencv2/dnn.hpp>
#include <numeric>

#include "dml_detector.h"
#include "sunone_aimbot_cpp.h"
#include "postProcess.h"
#include "capture.h"

// Variables globales para telemetría
extern std::atomic<bool> detector_model_changed;
extern std::atomic<bool> detection_resolution_changed;
std::chrono::duration<double, std::milli> lastPreprocessTimeDML{};
std::chrono::duration<double, std::milli> lastCopyTimeDML{};
std::chrono::duration<double, std::milli> lastPostprocessTimeDML{};
std::chrono::duration<double, std::milli> lastNmsTimeDML{};

// ... (El resto de las funciones de ayuda y el constructor/destructor no cambian) ...
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
    : env(ORT_LOGGING_LEVEL_WARNING, "DML_Detector"),
      memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      frameReady(false), shouldExit(false)
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

    auto temp_input_name_ptr = session.GetInputNameAllocated(0, allocator);
    input_name = temp_input_name_ptr.get();
    
    auto temp_output_name_ptr = session.GetOutputNameAllocated(0, allocator);
    output_name = temp_output_name_ptr.get();

    Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    input_shape = input_tensor_info.GetShape();

    bool isStatic = true;
    for (auto d : input_shape) if (d <= 0) isStatic = false;

    if (isStatic != config.fixed_input_size) {
        config.fixed_input_size = isStatic;
        config.saveConfig();
        detector_model_changed.store(true);
        std::cout << "[DML] Automatically set fixed_input_size = " << (isStatic ? "true" : "false") << std::endl;
    }
}

std::vector<std::vector<Detection>> DirectMLDetector::detectBatch(const std::vector<cv::Mat>& frames)
{
    if (frames.empty() || frames[0].empty()) return {};

    auto t0_prep = std::chrono::steady_clock::now();

    const int batch_size = static_cast<int>(frames.size());
    const int model_h = (input_shape.size() > 2 && config.fixed_input_size) ? static_cast<int>(input_shape[2]) : config.detection_resolution;
    const int model_w = (input_shape.size() > 3 && config.fixed_input_size) ? static_cast<int>(input_shape[3]) : config.detection_resolution;

    cv::Mat blob;
    cv::dnn::blobFromImages(frames, blob, 1.0 / 255.0, cv::Size(model_w, model_h), cv::Scalar(), true, false, CV_32F);

    auto t1_prep = std::chrono::steady_clock::now();
    lastPreprocessTimeDML = t1_prep - t0_prep;

    std::vector<int64_t> ort_input_shape = { (int64_t)batch_size, 3, (int64_t)model_h, (int64_t)model_w };
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, blob.ptr<float>(), blob.total(),
        ort_input_shape.data(), ort_input_shape.size());

    const char* input_names[] = { input_name.c_str() };
    const char* output_names[] = { output_name.c_str() };

    auto t2_inf = std::chrono::steady_clock::now();
    auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 1);
    auto t3_inf = std::chrono::steady_clock::now();
    lastInferenceTimeDML = t3_inf - t2_inf;

    auto t4_post = std::chrono::steady_clock::now();
    
    float* outData = output_tensors.front().GetTensorMutableData<float>();
    std::vector<int64_t> outShape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();
    const int num_classes = (outShape.size() >= 3) ? (static_cast<int>(outShape[outShape.size() - 2] - 4)) : -1;

    std::vector<std::vector<Detection>> batchDetections(batch_size);
    std::chrono::duration<double, std::milli> nmsTimeTmp{ 0 };

    for (int b = 0; b < batch_size; ++b)
    {
        if (frames[b].empty()) continue;

        const int single_batch_output_size = std::accumulate(outShape.begin() + 1, outShape.end(), 1, std::multiplies<int64_t>());
        const float* ptr = outData + b * single_batch_output_size;
        
        std::vector<Detection> detections;
        
        // *** CORRECCIÓN: Se restaura la lógica original para pasar la forma (shape) correcta a cada función ***
        if (config.postprocess == "yolo10")
        {
            // La función postProcessYolo10DML espera la forma completa de 3D, ya que usa shape[1] y shape[2]
            detections = postProcessYolo10DML(ptr, outShape, num_classes, config.confidence_threshold, config.nms_threshold, &nmsTimeTmp);
        }
        else // Asume "yolo11"
        {
            // La función postProcessYolo11DML espera una forma de 2D, así que creamos un sub-vector
            std::vector<int64_t> shp_2d = { outShape[outShape.size() - 2], outShape[outShape.size() - 1] };
            detections = postProcessYolo11DML(ptr, shp_2d, num_classes, config.confidence_threshold, config.nms_threshold, &nmsTimeTmp);
        }

        if (model_w != config.detection_resolution) {
            float scale = static_cast<float>(config.detection_resolution) / model_w;
            for (auto& d : detections) {
                d.box.x = static_cast<int>(d.box.x * scale);
                d.box.y = static_cast<int>(d.box.y * scale);
                d.box.width = static_cast<int>(d.box.width * scale);
                d.box.height = static_cast<int>(d.box.height * scale);
            }
        }
        batchDetections[b] = std::move(detections);
    }
    
    auto t5_post = std::chrono::steady_clock::now();
    lastPostprocessTimeDML = t5_post - t4_post;
    lastNmsTimeDML = nmsTimeTmp;

    return batchDetections;
}

void DirectMLDetector::processFrame(cv::Mat&& frame)
{
    std::unique_lock<std::mutex> lock(inferenceMutex);
    currentFrame = std::move(frame);
    frameReady = true;
    inferenceCV.notify_one();
}

void DirectMLDetector::dmlInferenceThread()
{
    while (!shouldExit)
    {
        if (detector_model_changed.load()) {
            initializeModel("models/" + config.ai_model);
            detection_resolution_changed.store(true);
            detector_model_changed.store(false);
            std::cout << "[DML] Detector reloaded: " << config.ai_model << std::endl;
        }

        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lock(inferenceMutex);
            inferenceCV.wait(lock, [this] { return frameReady || shouldExit; });
            if (shouldExit) break;
            if (frameReady) {
                frame = std::move(currentFrame);
                frameReady = false;
            }
        }

        if (!frame.empty())
        {
            std::vector<cv::Mat> batchFrames;
            batchFrames.push_back(std::move(frame));
            
            auto detectionsBatch = detectBatch(batchFrames);

            if (!detectionsBatch.empty()) {
                const std::vector<Detection>& detections = detectionsBatch.front();
                std::lock_guard<std::mutex> lock(detectionBuffer.mutex);
                detectionBuffer.boxes.clear();
                detectionBuffer.classes.clear();
                detectionBuffer.boxes.reserve(detections.size());
                detectionBuffer.classes.reserve(detections.size());
                for (const auto& d : detections) {
                    detectionBuffer.boxes.push_back(d.box);
                    detectionBuffer.classes.push_back(d.classId);
                }
                detectionBuffer.version++;
                detectionBuffer.cv.notify_all();
            }
        }
    }
}