#ifdef USE_CUDA
#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <algorithm>
#include <atomic>
#include <limits>
#include <numeric>
#include <vector>
#include <queue>
#include <mutex>

#include "trt_detector.h"
#include "nvinf.h"
#include "sunone_aimbot_cpp.h"
#include "other_tools.h"
#include "postProcess.h"
#include "cuda_preprocess.h"

extern std::atomic<bool> detectionPaused;
int model_quant;
std::vector<float> outputData;

extern std::atomic<bool> detector_model_changed;
extern std::atomic<bool> detection_resolution_changed;

static bool error_logged = false;

namespace {
bool tryGetDimInt(int64_t value, int* out)
{
    if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
        return false;
    *out = static_cast<int>(value);
    return true;
}

bool tryGetPositiveDimInt(int64_t value, int* out)
{
    if (value <= 0)
        return false;
    return tryGetDimInt(value, out);
}
} // namespace

TrtDetector::TrtDetector()
    : frameReady(false),
    shouldExit(false),
    useCudaGraph(false),
    cudaGraphCaptured(false),
    cudaGraph(nullptr),
    cudaGraphExec(nullptr),
    inputBufferDevice(nullptr),
    img_scale(1.0f),
    numClasses(0)
{
    stream = nullptr;
    cudaStreamCreate(&stream);
}

TrtDetector::~TrtDetector()
{
    destroyCudaGraph();
    freePinnedOutputs();

    for (auto& binding : inputBindings) if (binding.second) cudaFree(binding.second);
    for (auto& binding : outputBindings) if (binding.second) cudaFree(binding.second);
    if (inputBufferDevice) cudaFree(inputBufferDevice);
    if (preprocessStartEvent) cudaEventDestroy(preprocessStartEvent);
    if (inferenceStartEvent) cudaEventDestroy(inferenceStartEvent);
    if (inferenceCompleteEvent) cudaEventDestroy(inferenceCompleteEvent);
    if (copyCompleteEvent) cudaEventDestroy(copyCompleteEvent);
    if (stream) cudaStreamDestroy(stream);
}

void TrtDetector::freePinnedOutputs()
{
    for (auto& kv : pinnedOutputBuffers)
    {
        if (kv.second)
            cudaFreeHost(kv.second);
    }
    pinnedOutputBuffers.clear();
}

void TrtDetector::allocatePinnedOutputs()
{
    freePinnedOutputs();

    for (const auto& name : outputNames)
    {
        const size_t bytes = outputSizes[name];
        if (bytes == 0) continue;

        void* hostPtr = nullptr;
        cudaError_t err = cudaHostAlloc(&hostPtr, bytes, cudaHostAllocDefault);
        if (err != cudaSuccess)
        {
            std::cerr << "[Detector] cudaHostAlloc failed for output " << name
                << " (" << bytes << " bytes): " << cudaGetErrorString(err) << std::endl;
            continue;
        }

        pinnedOutputBuffers[name] = hostPtr;

        if (config.verbose)
        {
            std::cout << "[Detector] Allocated pinned host buffer for output " << name
                << ": " << bytes << " bytes" << std::endl;
        }
    }
}

void TrtDetector::destroyCudaGraph()
{
    if (cudaGraphExec)
    {
        cudaGraphExecDestroy(cudaGraphExec);
        cudaGraphExec = nullptr;
    }
    if (cudaGraph)
    {
        cudaGraphDestroy(cudaGraph);
        cudaGraph = nullptr;
    }
    cudaGraphCaptured = false;
}

void TrtDetector::captureCudaGraph()
{
    if (!useCudaGraph || cudaGraphCaptured) return;

    destroyCudaGraph();

    cudaStreamSynchronize(stream);

    cudaError_t st = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (st != cudaSuccess) {
        std::cerr << "[Detector] BeginCapture failed: "
            << cudaGetErrorString(st) << std::endl;
        return;
    }

    context->enqueueV3(stream);
    cudaEventRecord(inferenceCompleteEvent, stream);

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
        cudaGraphDestroy(cudaGraph);
        cudaGraph = nullptr;
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
            }
            else
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
    fp16OutputScratch.clear();

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

    allocatePinnedOutputs();

    if (!outputNames.empty())
    {
        const std::string& mainOut = outputNames[0];
        nvinfer1::Dims outDims = context->getTensorShape(mainOut.c_str());
        const int64_t outChannels = outDims.d[1];
        const int64_t classes64 = (outChannels > 4) ? (outChannels - 4) : 1;
        int classes = 0;
        if (!tryGetDimInt(classes64, &classes) || classes <= 0)
        {
            std::cerr << "[Detector] Invalid output dimensions for classes" << std::endl;
            return;
        }
        numClasses = classes;
    }

    int c = 0;
    int h = 0;
    int w = 0;
    if (!tryGetPositiveDimInt(inputDims.d[1], &c)
        || !tryGetPositiveDimInt(inputDims.d[2], &h)
        || !tryGetPositiveDimInt(inputDims.d[3], &w))
    {
        std::cerr << "[Detector] Invalid input dimensions" << std::endl;
        return;
    }

    // Pre-allocate GPU buffers
    gpuResizedBuffer.create(h, w, CV_8UC3);
    gpuFloatBuffer.create(h, w, CV_32FC3);
    gpuChannelBuffers.resize(c);
    for (int i = 0; i < c; ++i)
        gpuChannelBuffers[i].create(h, w, CV_32F);

    cvStream = cv::cuda::StreamAccessor::wrapStream(stream);

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

    if (preprocessStartEvent) cudaEventDestroy(preprocessStartEvent);
    if (inferenceStartEvent) cudaEventDestroy(inferenceStartEvent);
    if (inferenceCompleteEvent) cudaEventDestroy(inferenceCompleteEvent);
    if (copyCompleteEvent) cudaEventDestroy(copyCompleteEvent);

    preprocessStartEvent = nullptr;
    inferenceStartEvent = nullptr;
    inferenceCompleteEvent = nullptr;
    copyCompleteEvent = nullptr;

    cudaEventCreate(&preprocessStartEvent);
    cudaEventCreate(&inferenceStartEvent);
    cudaEventCreate(&inferenceCompleteEvent);
    cudaEventCreate(&copyCompleteEvent);

    useCudaGraph = config.use_cuda_graph;
    if (useCudaGraph)
    {
        captureCudaGraph();
    }

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
    currentFrame = frame;
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
                destroyCudaGraph();
                context.reset();
                engine.reset();

                freePinnedOutputs();

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

        if (useCudaGraph != config.use_cuda_graph)
        {
            useCudaGraph = config.use_cuda_graph;
            if (!useCudaGraph)
            {
                destroyCudaGraph();
            }
            else if (context)
            {
                captureCudaGraph();
            }
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
                cudaEventRecord(preprocessStartEvent, stream);
                preProcess(frame);
                cudaEventRecord(inferenceStartEvent, stream);
                bool usedGraph = useCudaGraph && cudaGraphCaptured;
                if (usedGraph)
                {
                    launchCudaGraph();
                    cudaEventRecord(copyCompleteEvent, stream);
                    cudaEventSynchronize(copyCompleteEvent);
                }
                else
                {
                    context->enqueueV3(stream);
                    cudaEventRecord(inferenceCompleteEvent, stream);

                    for (const auto& name : outputNames)
                    {
                        const size_t size = outputSizes[name];

                        auto itPinned = pinnedOutputBuffers.find(name);
                        if (itPinned == pinnedOutputBuffers.end() || !itPinned->second)
                            continue;

                        cudaMemcpyAsync(
                            itPinned->second,
                            outputBindings[name],
                            size,
                            cudaMemcpyDeviceToHost,
                            stream
                        );
                    }

                    cudaEventRecord(copyCompleteEvent, stream);
                    cudaEventSynchronize(copyCompleteEvent);
                }

                // Post-processing (CPU)
                auto t_post_start = std::chrono::steady_clock::now();

                for (const auto& name : outputNames)
                {
                    const auto itPinned = pinnedOutputBuffers.find(name);
                    if (itPinned == pinnedOutputBuffers.end() || !itPinned->second)
                        continue;

                    nvinfer1::DataType dtype = outputTypes[name];

                    if (dtype == nvinfer1::DataType::kHALF)
                    {
                        // Convert to float on CPU
                        const size_t numElements = outputSizes[name] / sizeof(__half);
                        const __half* halfPtr = reinterpret_cast<const __half*>(itPinned->second);

                        auto& outputDataFloat = fp16OutputScratch[name];
                        if (outputDataFloat.size() != numElements)
                            outputDataFloat.resize(numElements);

                        for (size_t i = 0; i < numElements; ++i)
                            outputDataFloat[i] = __half2float(halfPtr[i]);

                        postProcess(outputDataFloat.data(), name, &lastNmsTime);
                    }
                    else if (dtype == nvinfer1::DataType::kFLOAT)
                    {
                        const float* floatPtr = reinterpret_cast<const float*>(itPinned->second);
                        postProcess(floatPtr, name, &lastNmsTime);
                    }
                }

                auto t_post_end = std::chrono::steady_clock::now();

                float preprocessMs = 0.0f;
                float inferenceMs = 0.0f;
                float copyMs = 0.0f;

                cudaEventElapsedTime(&preprocessMs, preprocessStartEvent, inferenceStartEvent);
                cudaEventElapsedTime(&inferenceMs, inferenceStartEvent, inferenceCompleteEvent);
                cudaEventElapsedTime(&copyMs, inferenceCompleteEvent, copyCompleteEvent);

                lastPreprocessTime = std::chrono::duration<double, std::milli>(preprocessMs);
                lastInferenceTime = std::chrono::duration<double, std::milli>(inferenceMs);
                lastCopyTime = std::chrono::duration<double, std::milli>(copyMs);
                lastPostprocessTime = t_post_end - t_post_start;
            }
            catch (const std::exception& e)
            {
                std::cerr << "[Detector] Error during inference: " << e.what() << std::endl;
            }
        }
    }
}

void TrtDetector::preProcess(const cv::Mat& frame)
{
    if (frame.empty()) return;

    void* inputBuffer = inputBindings[inputName];
    if (!inputBuffer) return;

    nvinfer1::Dims dims = context->getTensorShape(inputName.c_str());
    int c = 0;
    int h = 0;
    int w = 0;
    if (!tryGetPositiveDimInt(dims.d[1], &c)
        || !tryGetPositiveDimInt(dims.d[2], &h)
        || !tryGetPositiveDimInt(dims.d[3], &w))
    {
        return;
    }

    if (c != 3) return;

    gpuFrameBuffer.upload(frame, cvStream);

    if (frame.channels() == 4)
        cv::cuda::cvtColor(gpuFrameBuffer, gpuFrameBuffer, cv::COLOR_BGRA2BGR, 0, cvStream);
    else if (frame.channels() == 1)
        cv::cuda::cvtColor(gpuFrameBuffer, gpuFrameBuffer, cv::COLOR_GRAY2BGR, 0, cvStream);

    cv::cuda::resize(gpuFrameBuffer, gpuResizedBuffer, cv::Size(w, h), 0, 0, cv::INTER_LINEAR, cvStream);

    gpuResizedBuffer.convertTo(gpuFloatBuffer, CV_32FC3, 1.0 / 255.0, 0.0, cvStream);

    launch_hwc_to_chw_norm(gpuFloatBuffer, reinterpret_cast<float*>(inputBuffer), w, h, stream);

    if (config.verbose)
    {
        auto err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "[Detector] preprocess kernel launch error: " << cudaGetErrorString(err) << std::endl;
        }
    }
}

void TrtDetector::postProcess(const float* output, const std::string& outputName, std::chrono::duration<double, std::milli>* nmsTime)
{
    if (numClasses <= 0) return;

    const auto shapeIt = outputShapes.find(outputName);
    if (shapeIt == outputShapes.end())
        return;

    std::vector<Detection> detections;
    
    detections = postProcessYolo(
        output,
        shapeIt->second,
        numClasses,
        config.confidence_threshold,
        config.nms_threshold,
        nmsTime
    );

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
