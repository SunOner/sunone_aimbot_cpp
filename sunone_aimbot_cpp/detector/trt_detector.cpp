#ifdef USE_CUDA
#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <algorithm>
#include <atomic>
#include <numeric>
#include <vector>
#include <queue>
#include <mutex>

#include "trt_detector.h"
#include "nvinf.h"
#include "sunone_aimbot_cpp.h"
#include "other_tools.h"
#include "postProcess.h"

extern std::atomic<bool> detectionPaused;
int model_quant;
std::vector<float> outputData;

extern std::atomic<bool> detector_model_changed;
extern std::atomic<bool> detection_resolution_changed;

static bool error_logged = false;

TrtDetector::TrtDetector()
    : frameReady(false),
    shouldExit(false),
    inputBufferDevice(nullptr),
    img_scale(1.0f),
    numClasses(0)
{
    cudaStreamCreate(&stream);
}

TrtDetector::~TrtDetector()
{
    for (auto& buffer : pinnedOutputBuffers)
    {
        if (buffer.second) cudaFreeHost(buffer.second);
    }
    for (auto& binding : inputBindings)
    {
        if (binding.second) cudaFree(binding.second);
    }
    for (auto& binding : outputBindings)
    {
        if (binding.second) cudaFree(binding.second);
    }
    if (inputBufferDevice)
    {
        cudaFree(inputBufferDevice);
    }
}

void TrtDetector::destroyCudaGraph()
{
    cudaStreamDestroy(stream);

    if (cudaGraphCaptured)
    {
        cudaGraphExecDestroy(cudaGraphExec);
        cudaGraphDestroy(cudaGraph);
        cudaGraphCaptured = false;
    }
}

void TrtDetector::captureCudaGraph()
{
    if (!useCudaGraph || cudaGraphCaptured) return;

    cudaStreamSynchronize(stream);

    cudaError_t st = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (st != cudaSuccess) {
        std::cerr << "[Detector] BeginCapture failed: "
            << cudaGetErrorString(st) << std::endl;
        return;
    }

    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);

    for (const auto& name : outputNames)
        if (pinnedOutputBuffers.count(name))
            cudaMemcpyAsync(pinnedOutputBuffers[name],
                outputBindings[name],
                outputSizes[name],
                cudaMemcpyDeviceToHost,
                stream);

    st = cudaStreamEndCapture(stream, &cudaGraph);
    if (st != cudaSuccess) {
        std::cerr << "[Detector] EndCapture failed: "
            << cudaGetErrorString(st) << std::endl;
        return;
    }

    st = cudaGraphInstantiate(&cudaGraphExec, cudaGraph, 0);
    if (st != cudaSuccess) {
        std::cerr << "[Detector] GraphInstantiate failed: "
            << cudaGetErrorString(st) << std::endl;
        return;
    }

    cudaGraphCaptured = true;
}

inline void TrtDetector::launchCudaGraph()
{
    auto err = cudaGraphLaunch(cudaGraphExec, stream);
    if (err != cudaSuccess)
    {
        std::cerr << "[Detector] GraphLaunch failed: " << cudaGetErrorString(err) << std::endl;
    }
}

void TrtDetector::getInputNames()
{
    inputNames.clear();
    inputSizes.clear();

    for (int i = 0; i < engine->getNbIOTensors(); ++i)
    {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
            inputNames.emplace_back(name);
            if (config.verbose)
            {
                std::cout << "[Detector] Detected input: " << name << std::endl;
            }
        }
    }
}

void TrtDetector::getOutputNames()
{
    outputNames.clear();
    outputSizes.clear();
    outputTypes.clear();
    outputShapes.clear();

    for (int i = 0; i < engine->getNbIOTensors(); ++i)
    {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT)
        {
            outputNames.emplace_back(name);
            outputTypes[name] = engine->getTensorDataType(name);
            
            if (config.verbose)
            {
                std::cout << "[Detector] Detected output: " << name << std::endl;
            }
        }
    }
}

void TrtDetector::getBindings()
{
    for (auto& binding : inputBindings)
    {
        if (binding.second) cudaFree(binding.second);
    }
    inputBindings.clear();

    for (auto& binding : outputBindings)
    {
        if (binding.second) cudaFree(binding.second);
    }
    outputBindings.clear();

    for (const auto& name : inputNames)
    {
        size_t size = inputSizes[name];
        if (size > 0)
        {
            void* ptr = nullptr;

            cudaError_t err = cudaMalloc(&ptr, size);
            if (err == cudaSuccess)
            {
                inputBindings[name] = ptr;
                if (config.verbose)
                {
                    std::cout << "[Detector] Allocated " << size << " bytes for input " << name << std::endl;
                }
            }
            else
            {
                std::cerr << "[Detector] Failed to allocate input memory: " << cudaGetErrorString(err) << std::endl;
            }
        }
    }

    for (const auto& name : outputNames)
    {
        size_t size = outputSizes[name];
        if (size > 0) {
            void* ptr = nullptr;
            cudaError_t err = cudaMalloc(&ptr, size);
            if (err == cudaSuccess)
            {
                outputBindings[name] = ptr;
                if (config.verbose)
                {
                    std::cout << "[Detector] Allocated " << size << " bytes for output " << name << std::endl;
                }
            } else
            {
                std::cerr << "[Detector] Failed to allocate output memory: " << cudaGetErrorString(err) << std::endl;
            }
        }
    }
}

void TrtDetector::initialize(const std::string& modelFile)
{
    runtime.reset(nvinfer1::createInferRuntime(gLogger));
    loadEngine(modelFile);
    if (!engine)
    {
        std::cerr << "[Detector] Engine loading failed" << std::endl;
        return;
    }

    context.reset(engine->createExecutionContext());
    if (!context)
    {
        std::cerr << "[Detector] Context creation failed" << std::endl;
        return;
    }

    getInputNames();
    getOutputNames();
    if (inputNames.empty())
    {
        std::cerr << "[Detector] No input tensors found" << std::endl;
        return;
    }
    inputName = inputNames[0];

    nvinfer1::Dims inputDims = context->getTensorShape(inputName.c_str());
    bool isStatic = true;
    for (int i = 0; i < inputDims.nbDims; ++i)
        if (inputDims.d[i] <= 0) isStatic = false;
    
    if (isStatic != config.fixed_input_size)
    {
        config.fixed_input_size = isStatic;
        config.saveConfig();
        detector_model_changed.store(true);
        std::cout << "[Detector] Automatically set fixed_input_size = " << (isStatic ? "true" : "false") << std::endl;
    }

    const int target = config.detection_resolution;
    if (!isStatic)
    {
        nvinfer1::Dims4 newShape{ 1, 3, target, target };
        context->setInputShape(inputName.c_str(), newShape);
        if (!context->allInputDimensionsSpecified())
        {
            std::cerr << "[Detector] Failed to set input dimensions" << std::endl;
            return;
        }
        inputDims = context->getTensorShape(inputName.c_str());
    }

    inputSizes.clear();
    outputSizes.clear();
    outputShapes.clear();
    outputTypes.clear();

    for (const auto& inName : inputNames)
    {
        nvinfer1::Dims d = context->getTensorShape(inName.c_str());
        nvinfer1::DataType dt = engine->getTensorDataType(inName.c_str());
        inputSizes[inName] = getSizeByDim(d) * getElementSize(dt);
    }
    for (const auto& outName : outputNames)
    {
        nvinfer1::Dims d = context->getTensorShape(outName.c_str());
        nvinfer1::DataType dt = engine->getTensorDataType(outName.c_str());
        outputSizes[outName] = getSizeByDim(d) * getElementSize(dt);
        std::vector<int64_t> shape(d.nbDims);
        for (int j = 0; j < d.nbDims; ++j) shape[j] = d.d[j];
        outputShapes[outName] = std::move(shape);
        outputTypes[outName] = dt;
    }

    getBindings();

    if (!outputNames.empty())
    {
        const std::string& mainOut = outputNames[0];
        nvinfer1::Dims outDims = context->getTensorShape(mainOut.c_str());
        numClasses = (config.postprocess == "yolo10") ? 11 : (outDims.d[1] - 4);
    }

    int c = inputDims.d[1];
    int h = inputDims.d[2];
    int w = inputDims.d[3];

    img_scale = static_cast<float>(config.detection_resolution) / w;

    resizedBuffer.create(h, w, CV_8UC3);
    floatBuffer.create(h, w, CV_32FC3);
    channelBuffers.clear();
    channelBuffers.resize(c);
    for (int i = 0; i < c; ++i)
        channelBuffers[i].create(h, w, CV_32F);

    for (const auto& n : inputNames)
        context->setTensorAddress(n.c_str(), inputBindings[n]);
    for (const auto& n : outputNames)
        context->setTensorAddress(n.c_str(), outputBindings[n]);

    if (config.verbose)
    {
        std::cout << "[Detector] Initialized. ModelStatic=" << std::boolalpha << isStatic
            << ", NetInput=" << h << "x" << w << " (scale=" << img_scale << ")" << std::endl;
    }
}

size_t TrtDetector::getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] < 0) return 0;
        size *= dims.d[i];
    }
    return size;
}

size_t TrtDetector::getElementSize(nvinfer1::DataType dtype)
{
    switch (dtype)
    {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kINT8: return 1;
        default: return 0;
    }
}

void TrtDetector::loadEngine(const std::string& modelFile)
{
    std::string engineFilePath;
    std::filesystem::path modelPath(modelFile);
    std::string extension = modelPath.extension().string();

    if (extension == ".engine")
    {
        engineFilePath = modelFile;
    }
    else if (extension == ".onnx")
    {
        engineFilePath = modelPath.replace_extension(".engine").string();

        if (!fileExists(engineFilePath))
        {
            std::cout << "[Detector] Building engine from ONNX model" << std::endl;

            nvinfer1::ICudaEngine* builtEngine = buildEngineFromOnnx(modelFile, gLogger);
            if (builtEngine)
            {
                nvinfer1::IHostMemory* serializedEngine = builtEngine->serialize();

                if (serializedEngine)
                {
                    std::ofstream engineFile(engineFilePath, std::ios::binary);
                    if (engineFile)
                    {
                        engineFile.write(reinterpret_cast<const char*>(serializedEngine->data()), serializedEngine->size());
                        engineFile.close();
                        
                        config.ai_model = std::filesystem::path(engineFilePath).filename().string();
                        config.saveConfig("config.ini");
                        
                        std::cout << "[Detector] Engine saved to: " << engineFilePath << std::endl;
                    }
                    delete serializedEngine;
                }
                delete builtEngine;
            }
        }
    }
    else
    {
        std::cerr << "[Detector] Unsupported model format: " << extension << std::endl;
        return;
    }

    std::cout << "[Detector] Loading engine: " << engineFilePath << std::endl;
    engine.reset(loadEngineFromFile(engineFilePath, runtime.get()));
}

void TrtDetector::processFrame(const cv::Mat& frame)
{
    if (config.backend == "DML") return;

    if (detectionPaused)
    {
        std::lock_guard<std::mutex> lock(detectionBuffer.mutex);
        detectionBuffer.boxes.clear();
        detectionBuffer.classes.clear();
        return;
    }

    std::unique_lock<std::mutex> lock(inferenceMutex);
    currentFrame = frame.clone();
    frameReady = true;
    inferenceCV.notify_one();
}

void TrtDetector::inferenceThread()
{
    while (!shouldExit)
    {
        if (detector_model_changed.load())
        {
            {
                std::unique_lock<std::mutex> lock(inferenceMutex);
                context.reset();
                engine.reset();
                for (auto& binding : inputBindings)
                    if (binding.second) cudaFree(binding.second);
                inputBindings.clear();
                for (auto& binding : outputBindings)
                    if (binding.second) cudaFree(binding.second);
                outputBindings.clear();
            }
            initialize("models/" + config.ai_model);
            detection_resolution_changed.store(true);
            detector_model_changed.store(false);
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

        if (!context)
        {
            if (!error_logged)
            {
                std::cerr << "[Detector] Context not initialized" << std::endl;
                error_logged = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        else
        {
            error_logged = false;
        }

        if (hasNewFrame && !frame.empty())
        {
            try
            {
                auto t0 = std::chrono::steady_clock::now();
                preProcess(frame);
                auto t1 = std::chrono::steady_clock::now();

                context->enqueueV3(stream);
                cudaStreamSynchronize(stream);

                auto t2 = std::chrono::steady_clock::now();

                for (const auto& name : outputNames)
                {
                    size_t size = outputSizes[name];
                    nvinfer1::DataType dtype = outputTypes[name];

                    auto t_copy_start = std::chrono::steady_clock::now();

                    if (dtype == nvinfer1::DataType::kHALF)
                    {
                        size_t numElements = size / sizeof(__half);
                        std::vector<__half>& outputDataHalf = outputDataBuffersHalf[name];
                        outputDataHalf.resize(numElements);

                        cudaMemcpy(
                            outputDataHalf.data(),
                            outputBindings[name],
                            size,
                            cudaMemcpyDeviceToHost
                        );

                        std::vector<float> outputDataFloat(outputDataHalf.size());
                        for (size_t i = 0; i < outputDataHalf.size(); ++i) {
                            outputDataFloat[i] = __half2float(outputDataHalf[i]);
                        }

                        auto t_copy_end = std::chrono::steady_clock::now();
                        lastCopyTime = t_copy_end - t_copy_start;

                        auto t_post_start = std::chrono::steady_clock::now();

                        postProcess(
                            outputDataFloat.data(),
                            name,
                            &lastNmsTime
                        );

                        auto t_post_end = std::chrono::steady_clock::now();
                        lastPostprocessTime = t_post_end - t_post_start;
                    }
                    else if (dtype == nvinfer1::DataType::kFLOAT)
                    {
                        std::vector<float>& outputData = outputDataBuffers[name];
                        outputData.resize(size / sizeof(float));

                        cudaMemcpy(
                            outputData.data(),
                            outputBindings[name],
                            size,
                            cudaMemcpyDeviceToHost
                        );

                        auto t_copy_end = std::chrono::steady_clock::now();
                        lastCopyTime = t_copy_end - t_copy_start;

                        auto t_post_start = std::chrono::steady_clock::now();

                        postProcess(
                            outputData.data(),
                            name,
                            &lastNmsTime
                        );

                        auto t_post_end = std::chrono::steady_clock::now();
                        lastPostprocessTime = t_post_end - t_post_start;
                    }
                }
                lastPreprocessTime = t1 - t0;
                lastInferenceTime = t2 - t1;
            }
            catch (const std::exception& e)
            {
                std::cerr << "[Detector] Error during inference: " << e.what() << std::endl;
            }
        }
    }
}

std::vector<std::vector<Detection>> TrtDetector::detectBatch(const std::vector<cv::Mat>& frames)
{
    std::vector<std::vector<Detection>> batchDetections;
    if (frames.empty() || !context) return batchDetections;

    int batch_size = static_cast<int>(frames.size());

    nvinfer1::Dims dims = context->getTensorShape(inputName.c_str());
    int c = dims.d[1];
    int h = dims.d[2];
    int w = dims.d[3];

    if (dims.d[0] != batch_size)
    {
        context->setInputShape(inputName.c_str(), nvinfer1::Dims4{ batch_size, c, h, w });
    }

    std::vector<float> batchInput(batch_size * c * h * w);

    for (int b = 0; b < batch_size; ++b)
    {
        cv::Mat resized;
        cv::resize(frames[b], resized, cv::Size(w, h));
        resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);
        std::vector<cv::Mat> channels;
        cv::split(resized, channels);

        for (int ch = 0; ch < c; ++ch)
        {
            float* dst = batchInput.data() + b * c * h * w + ch * h * w;
            memcpy(dst, channels[ch].ptr<float>(), h * w * sizeof(float));
        }
    }

    cudaMemcpy(inputBindings[inputName], batchInput.data(), batchInput.size() * sizeof(float), cudaMemcpyHostToDevice);

    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);

    std::vector<float> output;
    const auto& outName = outputNames[0];
    size_t outputElements = outputSizes[outName] / sizeof(float);
    output.resize(outputElements);
    cudaMemcpy(output.data(), outputBindings[outName], outputSizes[outName], cudaMemcpyDeviceToHost);

    const std::vector<int64_t>& shape = outputShapes[outName];
    int batch_out = static_cast<int>(shape[0]);
    int rows = static_cast<int>(shape[1]);
    int cols = static_cast<int>(shape[2]);

    for (int b = 0; b < batch_out; ++b)
    {
        const float* out_ptr = output.data() + b * rows * cols;
        std::vector<Detection> detections;

        if (config.postprocess == "yolo10")
        {
            std::vector<int64_t> shape = { batch_out, rows, cols };
            detections = postProcessYolo10(
                out_ptr,
                shape,
                numClasses,
                config.confidence_threshold,
                config.nms_threshold,
                &lastNmsTime
            );
        }
        else if (
            config.postprocess == "yolo8" ||
            config.postprocess == "yolo9" ||
            config.postprocess == "yolo11" ||
            config.postprocess == "yolo12"
        )
        {
            std::vector<int64_t> shape = { rows, cols };
            detections = postProcessYolo11(
                out_ptr,
                shape,
                numClasses,
                config.confidence_threshold,
                config.nms_threshold,
                &lastNmsTime
            );
        }

        batchDetections.push_back(std::move(detections));
    }

    return batchDetections;
}

void TrtDetector::preProcess(const cv::Mat& frame)
{
    if (frame.empty()) return;

    void* inputBuffer = inputBindings[inputName];
    if (!inputBuffer) return;

    nvinfer1::Dims dims = context->getTensorShape(inputName.c_str());
    int c = dims.d[1];
    int h = dims.d[2];
    int w = dims.d[3];

    cv::cuda::GpuMat gpuFrame, gpuResized, gpuFloat;
    gpuFrame.upload(frame);

    if (frame.channels() == 4)
        cv::cuda::cvtColor(gpuFrame, gpuFrame, cv::COLOR_BGRA2BGR);
    else if (frame.channels() == 1)
        cv::cuda::cvtColor(gpuFrame, gpuFrame, cv::COLOR_GRAY2BGR);

    cv::cuda::resize(gpuFrame, gpuResized, cv::Size(w, h));
    gpuResized.convertTo(gpuFloat, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::cuda::GpuMat> gpuChannels;
    cv::cuda::split(gpuFloat, gpuChannels);

    for (int i = 0; i < c; ++i)
        cudaMemcpy((float*)inputBuffer + i * h * w, gpuChannels[i].ptr<float>(), h * w * sizeof(float), cudaMemcpyDeviceToDevice);
}

void TrtDetector::postProcess(const float* output, const std::string& outputName, std::chrono::duration<double, std::milli>* nmsTime)
{
    if (numClasses <= 0) return;

    std::vector<Detection> detections;

    if (config.postprocess == "yolo10")
    {
        const std::vector<int64_t>& shape = outputShapes[outputName];
        detections = postProcessYolo10(
            output,
            shape,
            numClasses,
            config.confidence_threshold,
            config.nms_threshold,
            nmsTime
        );
    }
    else if(
        config.postprocess == "yolo8" ||
        config.postprocess == "yolo9" ||
        config.postprocess == "yolo11" ||
        config.postprocess == "yolo12"
        )
    {
        auto shape = context->getTensorShape(outputName.c_str());
        std::vector<int64_t> engineShape;
        for (int i = 0; i < shape.nbDims; ++i)
        {
            engineShape.push_back(shape.d[i]);
        }

        detections = postProcessYolo11(
            output,
            engineShape,
            numClasses,
            config.confidence_threshold,
            config.nms_threshold,
            nmsTime
        );
    }

    {
        std::lock_guard<std::mutex> lock(detectionBuffer.mutex);
        detectionBuffer.boxes.clear();
        detectionBuffer.classes.clear();

        for (const auto& det : detections)
        {
            detectionBuffer.boxes.push_back(det.box);
            detectionBuffer.classes.push_back(det.classId);
        }

        detectionBuffer.version++;
        detectionBuffer.cv.notify_all();
    }
}
#endif