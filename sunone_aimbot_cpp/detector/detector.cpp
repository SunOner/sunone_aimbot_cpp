#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <fstream>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

#include <algorithm>
#include <cuda_fp16.h>
#include <atomic>
#include <numeric>

#include <boost/algorithm/string.hpp>

#include "detector.h"
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

Detector::Detector()
    : frameReady(false),
    shouldExit(false),
    detectionVersion(0),
    inputBufferDevice(nullptr),
    inputDims(),
    img_scale(1.0f),
    numClasses(0)
{
    cudaError_t err = cudaStreamCreate(&stream);

    if (err != cudaSuccess)
    {
        std::cout << "[Detector] Can't create CUDA stream!" << std::endl;
    }
}

Detector::~Detector()
{
    cudaStreamDestroy(stream);

    for (auto& binding : inputBindings)
    {
        cudaFree(binding.second);
    }
    inputBindings.clear();

    for (auto& binding : outputBindings)
    {
        cudaFree(binding.second);
    }
    outputBindings.clear();

    if (inputBufferDevice)
    {
        cudaFree(inputBufferDevice);
        inputBufferDevice = nullptr;
    }
}

void Detector::getInputNames()
{
    inputNames.clear();
    inputSizes.clear();

    for (int i = 0; i < engine->getNbIOTensors(); ++i)
    {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
            if (config.verbose)
            {
                std::cout << "[Detector] Detected model input name: " << name << std::endl;
            }
            inputNames.emplace_back(name);

            nvinfer1::DataType dtype = engine->getTensorDataType(name);
        }
    }
}

void Detector::getOutputNames()
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
            nvinfer1::DataType dtype = engine->getTensorDataType(name);
            outputTypes[name] = dtype;

            if (config.verbose)
            {
                std::cout << "[Detector] Model output name: " << name << std::endl;
            }
        }
    }
}

void Detector::getBindings()
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
        void* ptr = nullptr;
        if (size > 0)
        {
            cudaError_t err = cudaMalloc(&ptr, size);
            if (err != cudaSuccess)
            {
                std::cerr << "[CUDA] Failed to allocate " << size << " bytes for input " << name << ": " << cudaGetErrorString(err) << std::endl;
            }
            else
            {
                inputBindings[name] = ptr;
                std::cout << "[DEBUG] Allocated " << size << " bytes for input " << name << std::endl;
            }
        }
        else
        {
            std::cerr << "[ERROR] inputSize = 0 for " << name << " (perhaps -1 remained in dims)" << std::endl;
        }
    }

    for (const auto& name : outputNames)
    {
        size_t size = outputSizes[name];
        void* ptr = nullptr;
        if (size > 0)
        {
            cudaError_t err = cudaMalloc(&ptr, size);
            if (err != cudaSuccess)
            {
                std::cerr << "[CUDA] Failed to allocate " << size << " bytes for output " << name << ": " << cudaGetErrorString(err) << std::endl;
            }
            else
            {
                outputBindings[name] = ptr;
                std::cout << "[DEBUG] Allocated " << size << " bytes for output " << name << std::endl;
            }
        }
        else
        {
            std::cerr << "[ERROR] outputSize = 0 for " << name << " (perhaps -1 remained in dims)" << std::endl;
        }
    }
}

void Detector::initialize(const std::string& modelFile)
{
    runtime.reset(nvinfer1::createInferRuntime(gLogger));
    loadEngine(modelFile);

    if (!engine)
    {
        std::cerr << "[Detector] Error loading the engine from the file " << modelFile << std::endl;
        return;
    }

    context.reset(engine->createExecutionContext());
    if (!context)
    {
        std::cerr << "[Detector] Error creating the execution context" << std::endl;
        return;
    }

    getInputNames();
    getOutputNames();

    if (inputNames.empty())
    {
        std::cerr << "[Detector] No input tensors found!" << std::endl;
        return;
    }

    inputName = inputNames[0];

    bool success = context->setInputShape(inputName.c_str(), nvinfer1::Dims4{ 1, 3, 640, 640 });
    if (!success)
    {
        std::cerr << "[Detector] Failed to set input shape for " << inputName << std::endl;
        return;
    }

    if (!context->allInputDimensionsSpecified())
    {
        std::cerr << "[Detector] Not all input dimensions are specified." << std::endl;
        return;
    }

    for (const auto& inName : inputNames)
    {
        nvinfer1::Dims inDims = context->getTensorShape(inName.c_str());
        nvinfer1::DataType inType = engine->getTensorDataType(inName.c_str());
        size_t size = getSizeByDim(inDims) * getElementSize(inType);
        inputSizes[inName] = size;

        if (config.verbose)
        {
            std::cout << "[Detector] Real input '" << inName << "' shape: ";
            for (int j = 0; j < inDims.nbDims; j++)
                std::cout << inDims.d[j] << " ";
            std::cout << " => bytes: " << size << std::endl;
        }
    }

    for (const auto& outName : outputNames)
    {
        nvinfer1::Dims outDims = context->getTensorShape(outName.c_str());
        nvinfer1::DataType outType = engine->getTensorDataType(outName.c_str());
        size_t size = getSizeByDim(outDims) * getElementSize(outType);
        outputSizes[outName] = size;

        std::vector<int64_t> shapeVec;
        for (int j = 0; j < outDims.nbDims; j++) {
            shapeVec.push_back(outDims.d[j]);
        }
        outputShapes[outName] = shapeVec;

        if (config.verbose)
        {
            std::cout << "[Detector] Real output '" << outName << "' shape: ";
            for (int j = 0; j < outDims.nbDims; j++)
                std::cout << outDims.d[j] << " ";
            std::cout << " => bytes: " << size << std::endl;
        }
    }

    getBindings();

    numClasses = 0;
    if (!outputNames.empty())
    {
        const std::string& mainOut = outputNames[0];
        nvinfer1::Dims outDims = context->getTensorShape(mainOut.c_str());

        if (config.postprocess == "yolo10")
        {
            numClasses = 11;
        }
        else
        {
            int c = outDims.d[1];
            numClasses = c - 4;
        }
        
        if (numClasses < 1)
        {
            std::cerr << "[Detector] Invalid number of classes: " << numClasses << std::endl;
            numClasses = 0;
        }
        else
        {
            std::cout << "[Detector] Number of classes: " << numClasses << std::endl;
        }
    }
    else
    {
        std::cerr << "[Detector] No outputs found to compute classes!" << std::endl;
    }

    img_scale = static_cast<float>(config.detection_resolution) / 640;
    if (config.verbose)
    {
        std::cout << "[Detector] Image scale factor: " << img_scale << std::endl;
    }
}

size_t Detector::getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] < 0)
        {
            std::cerr << "[WARNING] Negative dimension detected: " << dims.d[i] << " in tensor shape!" << std::endl;
            return 0;
        }
        size *= dims.d[i];
    }
    return size;
}

size_t Detector::getElementSize(nvinfer1::DataType dtype)
{
    switch (dtype)
    {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT8:  return 1;
        default: return 0;
    }
}

void Detector::loadEngine(const std::string& modelFile)
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
            std::cout << "[Detector] ONNX model detected: " << modelFile << std::endl;

            nvinfer1::ICudaEngine* builtEngine = buildEngineFromOnnx(modelFile, gLogger);
            if (!builtEngine)
            {
                std::cerr << "[Detector] Could not build engine from ONNX model: " << modelFile << std::endl;
                return;
            }

            nvinfer1::IHostMemory* serializedEngine = builtEngine->serialize();
            if (!serializedEngine)
            {
                std::cerr << "[Detector] Engine serialization error for model: " << modelFile << std::endl;
                delete builtEngine;
                return;
            }

            std::ofstream engineFile(engineFilePath, std::ios::binary);
            if (!engineFile)
            {
                std::cerr << "[Detector] Could not open engine file for writing: " << engineFilePath << std::endl;
                delete serializedEngine;
                delete builtEngine;
                return;
            }

            engineFile.write(reinterpret_cast<const char*>(serializedEngine->data()), serializedEngine->size());
            engineFile.close();
            delete serializedEngine;
            delete builtEngine;

            config.ai_model = std::filesystem::path(engineFilePath).filename().string();
            config.saveConfig("config.ini");

            std::cout << "[Detector] Engine successfully built and saved to: " << engineFilePath << std::endl;
        }
    }
    else
    {
        std::cerr << "[Detector] Unsupported model format: " << modelFile << std::endl;
        return;
    }

    std::cout << "[Detector] Loading engine model from file: " << engineFilePath << std::endl;
    engine.reset(loadEngineFromFile(engineFilePath, runtime.get()));

    if (!engine)
    {
        std::cerr << "[Detector] Error loading engine from file: " << engineFilePath << std::endl;
        return;
    }

    std::cout << "[Detector] Engine loaded successfully: " << engineFilePath << std::endl;
}

void Detector::processFrame(const cv::cuda::GpuMat& frame)
{
    if (detectionPaused)
    {
        if (!detectedBoxes.empty())
        {
            detectedBoxes.clear();
            detectedClasses.clear();
        }
        return;
    }

    std::unique_lock<std::mutex> lock(inferenceMutex);
    currentFrame = frame.clone();
    frameReady = true;
    inferenceCV.notify_one();
}

void Detector::inferenceThread()
{
    while (!shouldExit)
    {
        if (detector_model_changed.load())
        {
            {
                std::unique_lock<std::mutex> lock(inferenceMutex);

                context.reset();
                engine.reset();
                runtime.reset();

                for (auto& binding : inputBindings)
                {
                    cudaFree(binding.second);
                }
                inputBindings.clear();

                for (auto& binding : outputBindings)
                {
                    cudaFree(binding.second);
                }
                outputBindings.clear();

                cudaStreamDestroy(stream);
                cudaStreamCreate(&stream);

                initialize("models/" + config.ai_model);
            }

            detection_resolution_changed.store(true);
            detector_model_changed.store(false);
        }

        cv::cuda::GpuMat frame;
        {
            std::unique_lock<std::mutex> lock(inferenceMutex);
            inferenceCV.wait(lock, [this] { return frameReady || shouldExit; });
            if (shouldExit) break;
            frame = std::move(currentFrame);
            frameReady = false;
        }

        if (!context)
        {
            if (!error_logged)
            {
                std::cerr << "[Detector] The context is not initialized. Please select an AI model." << std::endl;
                error_logged = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        else
        {
            error_logged = false;
        }

        preProcess(frame);

        const std::string& inputName = inputNames[0];
        void* inputAddress = inputBindings[inputName];
        
        if (!inputAddress)
        {
            std::cerr << "[ERROR] Input tensor '" << inputName << "' not allocated!" << std::endl;
            return;
        }

        context->setTensorAddress(inputName.c_str(), inputAddress);

        for (const auto& name : outputNames)
        {
            void* outputAddress = outputBindings[name];
            if (!outputAddress) {
                std::cerr << "[ERROR] Output tensor '" << name << "' not allocated!" << std::endl;
                return;
            }
            context->setTensorAddress(name.c_str(), outputAddress);
        }

        context->enqueueV3(stream);

        for (const auto& name : outputNames)
        {
            size_t size = outputSizes[name];
            nvinfer1::DataType dtype = outputTypes[name];

            if (dtype == nvinfer1::DataType::kHALF)
            {
                size_t numElements = size / sizeof(__half);
                std::vector<__half>& outputDataHalf = outputDataBuffersHalf[name];
                outputDataHalf.resize(numElements);
                cudaMemcpyAsync(outputDataHalf.data(), outputBindings[name], size, cudaMemcpyDeviceToHost, stream);
            }
            else if (dtype == nvinfer1::DataType::kFLOAT)
            {
                size_t numElements = size / sizeof(float);
                std::vector<float>& outputData = outputDataBuffers[name];
                outputData.resize(numElements);
                cudaMemcpyAsync(outputData.data(), outputBindings[name], size, cudaMemcpyDeviceToHost, stream);
            }
            else
            {
                std::cerr << "[Detector] Unsupported output tensor data type" << std::endl;
                return;
            }
        }
        
        cudaStreamSynchronize(stream);

        for (const auto& name : outputNames)
        {
            nvinfer1::DataType dtype = outputTypes[name];

            if (dtype == nvinfer1::DataType::kHALF)
            {
                const std::vector<__half>& outputDataHalf = outputDataBuffersHalf[name];
                std::vector<float> outputData(outputDataHalf.size());

                for (size_t i = 0; i < outputDataHalf.size(); ++i)
                {
                    outputData[i] = __half2float(outputDataHalf[i]);
                }

                postProcess(outputData.data(), name);
            }
            else if (dtype == nvinfer1::DataType::kFLOAT)
            {
                const std::vector<float>& outputData = outputDataBuffers[name];
                postProcess(outputData.data(), name);
            }
            else
            {
                std::cerr << "[Detector] Unsupported output tensor data type in inferenceThread" << std::endl;
                return;
            }
        }
    }
}

void Detector::releaseDetections()
{
    std::lock_guard<std::mutex> lock(detectionMutex);
    detectedBoxes.clear();
    detectedClasses.clear();
}

bool Detector::getLatestDetections(std::vector<cv::Rect>& boxes, std::vector<int>& classes)
{
    std::lock_guard<std::mutex> lock(detectionMutex);
    if (!detectedBoxes.empty())
    {
        boxes = detectedBoxes;
        classes = detectedClasses;
        return true;
    }
    return false;
}

void Detector::preProcess(const cv::cuda::GpuMat& frame) {
    if (frame.empty()) {
        std::cerr << "[Detector] Empty frame received" << std::endl;
        return;
    }

    void* inputBuffer = inputBindings[inputName];
    if (!inputBuffer) {
        std::cerr << "[ERROR] Input buffer not allocated for " << inputName << std::endl;
        return;
    }

    nvinfer1::Dims dims = context->getTensorShape(inputName.c_str());
    int n = dims.d[0];
    int c = dims.d[1];
    int h = dims.d[2];
    int w = dims.d[3];

    cv::cuda::GpuMat resized;
    cv::cuda::resize(frame, resized, cv::Size(w, h));

    cv::cuda::GpuMat floatResized;
    resized.convertTo(floatResized, CV_32F, 1.0f / 255.0f);

    std::vector<cv::cuda::GpuMat> gpuChannels;
    cv::cuda::split(floatResized, gpuChannels);

    for (int cc = 0; cc < c; ++cc) {
        cudaMemcpyAsync(
            static_cast<float*>(inputBuffer) + cc * h * w,
            gpuChannels[cc].ptr<float>(),
            h * w * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream
        );
    }

    cudaStreamSynchronize(stream);
}

void Detector::postProcess(const float* output, const std::string& outputName)
{
    if (numClasses <= 0)
    {
        std::cerr << "[Detector] The number of model classes is undefined or incorrect." << std::endl;
        return;
    }

    std::vector<Detection> detections;
    size_t outputSize = outputSizes[outputName] / sizeof(float);

    if (config.postprocess == "yolo10")
    {
        const std::vector<int64_t>& shape = outputShapes[outputName];
        detections = postProcessYolo10(
            output,
            shape,
            numClasses,
            config.confidence_threshold,
            config.nms_threshold
        );
    }
    else if (
        config.postprocess == "yolo8" ||
        config.postprocess == "yolo9" ||
        config.postprocess == "yolo11" ||
        config.postprocess == "yolo12"
    )
    {
        auto curShape = context->getTensorShape(outputName.c_str());
        
        std::vector<int64_t> engineShape;
        engineShape.reserve(curShape.nbDims);

        for (int i = 0; i < curShape.nbDims; ++i)
        {
            engineShape.push_back(curShape.d[i]);
        }

        detections = postProcessYolo11(
            output,
            engineShape,
            numClasses,
            config.confidence_threshold,
            config.nms_threshold
        );
    }

    {
        std::lock_guard<std::mutex> lock(detectionMutex);
        detectedBoxes.clear();
        detectedClasses.clear();
        for (const auto& det : detections)
        {
            detectedBoxes.push_back(det.box);
            detectedClasses.push_back(det.classId);
        }
        detectionVersion++;
    }
    detectionCV.notify_one();
}