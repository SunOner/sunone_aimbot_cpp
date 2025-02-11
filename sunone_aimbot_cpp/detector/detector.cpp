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
    img_scale(0.0f)
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
            nvinfer1::Dims dims = engine->getTensorShape(name);
            nvinfer1::DataType dtype = engine->getTensorDataType(name);
            size_t size = getSizeByDim(dims) * getElementSize(dtype);
            inputSizes[name] = size;
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

            nvinfer1::Dims dims = engine->getTensorShape(name);
            nvinfer1::DataType dtype = engine->getTensorDataType(name);

            size_t size = getSizeByDim(dims) * getElementSize(dtype);

            outputSizes[name] = size;
            outputTypes[name] = dtype;

            std::vector<int64_t> dim;
            for (int j = 0; j < dims.nbDims; ++j)
            {
                dim.push_back(dims.d[j]);
            }
            outputShapes[name] = dim;

            if (config.verbose)
            {
                std::cout << "[Detector] Model output name: " << name << std::endl;
                std::cout << "[Detector] Model output size: " << size << std::endl;
                std::cout << "[Detector] Model output dims: " << dims.nbDims << std::endl;
            }
        }
    }
}

void Detector::getBindings()
{
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

    for (const auto& name : inputNames)
    {
        size_t size = inputSizes[name];
        void* ptr;
        cudaMalloc(&ptr, size);
        inputBindings[name] = ptr;
    }
    for (const auto& name : outputNames)
    {
        size_t size = outputSizes[name];
        void* ptr;
        cudaMalloc(&ptr, size);
        outputBindings[name] = ptr;
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
    getBindings();

    for (const auto& name : outputNames)
    {
        size_t size = outputSizes[name];
        nvinfer1::DataType dtype = outputTypes[name];
        if (dtype == nvinfer1::DataType::kHALF)
        {
            outputDataBuffersHalf[name].resize(size / sizeof(__half));
        }
        else if (dtype == nvinfer1::DataType::kFLOAT)
        {
            outputDataBuffers[name].resize(size / sizeof(float));
        }
        else
        {
            std::cerr << "[Detector] Unsupported output tensor data type during initialization" << std::endl;
        }
    }

    if (!inputNames.empty())
    {
        inputName = inputNames[0];
        inputDims = engine->getTensorShape(inputName.c_str());

        nvinfer1::DataType dtype = engine->getTensorDataType(inputName.c_str());
        size_t inputSize = getSizeByDim(inputDims) * getElementSize(dtype);
        inputSizes[inputName] = inputSize;

        cudaMalloc(&inputBufferDevice, inputSize);

        if (config.verbose)
        {
            std::cout << "[Detector] Model input size: " << getSizeByDim(inputDims) << std::endl;
            std::cout << "[Detector] Model input dims: " << getElementSize(dtype) << std::endl;
            std::cout << "[Detector] Model input name: " << inputName << std::endl;
        }
    }
    else
    {
        std::cerr << "[Detector] No input tensors found" << std::endl;
    }

    if (!outputNames.empty())
    {
        const std::string& outputName = outputNames[0];
        const std::vector<int64_t>& shape = outputShapes[outputName];
        int dimensions = static_cast<int>(shape[1]);
        numClasses = dimensions - 4;
        std::cout << "[Detector] Number of classes: " << numClasses << std::endl;
    }
    else
    {
        std::cerr << "[Detector] No output tensors were found." << std::endl;
        numClasses = 0;
    }

    img_scale = static_cast<float>(config.detection_resolution) / config.img_size;
    std::cout << img_scale << std::endl;
}

size_t Detector::getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i)
    {
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
            std::cout << "[Detector] ONNX model detected. Building engine from: " << modelFile << ".\n[Detector] Please wait..." << std::endl;

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

        context->setTensorAddress(inputName.c_str(), inputBufferDevice);

        for (const auto& name : outputNames)
        {
            context->setTensorAddress(name.c_str(), outputBindings[name]);
        }

        context->enqueueV3(stream);

        for (const auto& name : outputNames)
        {
            size_t size = outputSizes[name];
            nvinfer1::DataType dtype = outputTypes[name];

            if (dtype == nvinfer1::DataType::kHALF)
            {
                std::vector<__half>& outputDataHalf = outputDataBuffersHalf[name];
                cudaMemcpyAsync(outputDataHalf.data(), outputBindings[name], size, cudaMemcpyDeviceToHost, stream);
            }
            else if (dtype == nvinfer1::DataType::kFLOAT)
            {
                std::vector<float>& outputData = outputDataBuffers[name];
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

void Detector::preProcess(const cv::cuda::GpuMat& frame)
{
    if (frame.empty())
    {
        std::cerr << "[Detector] An empty image was obtained." << std::endl;
        return;
    }

    cv::cuda::GpuMat procFrame;
    procFrame = frame;

    int inputC = inputDims.d[1];
    int inputH = inputDims.d[2];
    int inputW = inputDims.d[3];

    cv::cuda::GpuMat resizedImage;
    cv::cuda::resize(procFrame, resizedImage, cv::Size(inputW, inputH));
    if (resizedImage.empty())
    {
        std::cerr << "[Detector] Error when resizing the image." << std::endl;
        return;
    }

    try
    {
        resizedImage.convertTo(resizedImage, CV_32F, 1.0 / 255.0);
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "[Detector] Error when converting an image: " << e.what() << std::endl;
        return;
    }

    for (int i = 0; i < 6; i++)
    {
        d2s.value[i] = 0.0f;
    }
    d2s.value[0] = 1.0f;
    d2s.value[4] = 1.0f;

    std::vector<cv::cuda::GpuMat> gpuChannels;
    try
    {
        cv::cuda::split(resizedImage, gpuChannels);
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "[Detector] Error when dividing the image into channels: " << e.what() << std::endl;
        return;
    }
    if (gpuChannels.size() < static_cast<size_t>(inputC))
    {
        std::cerr << "[Detector] Mismatch in the number of channels. Expected: "
            << inputC << ", received: " << gpuChannels.size() << std::endl;
        return;
    }

    for (int c = 0; c < inputC; ++c)
    {
        cudaError_t err = cudaMemcpyAsync(static_cast<float*>(inputBufferDevice) + c * inputH * inputW,
            gpuChannels[c].ptr<float>(),
            inputH * inputW * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream);
        if (err != cudaSuccess)
        {
            std::cerr << "[Detector] Channel copy error " << c
                << ": " << cudaGetErrorString(err) << std::endl;
            return;
        }
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

    if (config.postprocess == "yolo8")
    {
        std::vector<float> outputVec(output, output + outputSize);
        detections = postProcessYolo8(outputVec, img_scale, config.detection_resolution, config.detection_resolution,
            numClasses, config.confidence_threshold, config.nms_threshold);
    }
    else if (config.postprocess == "yolo9")
    {
        std::vector<float> outputVec(output, output + outputSize);
        detections = postProcessYolo9(outputVec, img_scale, config.detection_resolution, config.detection_resolution,
            numClasses, config.confidence_threshold, config.nms_threshold);
    }
    else if (config.postprocess == "yolo10")
    {
        const std::vector<int64_t>& shape = outputShapes[outputName];
        detections = postProcessYolo10(output, shape, img_scale, numClasses,
            config.confidence_threshold, config.nms_threshold);
    }
    else if (config.postprocess == "yolo11")
    {
        const std::vector<int64_t>& engineShape = outputShapes[outputName];
        if (engineShape.size() != 3)
        {
            std::cerr << "[Detector] Invalid output shape dimensions for yolo11. Expected 3 dimensions, but got "
                << engineShape.size() << std::endl;
            detections.clear();
        }
        else
        {
            detections = postProcessYolo11(output,
                engineShape,
                numClasses,
                config.confidence_threshold,
                config.nms_threshold);
        }
    }
    else
    {
        std::cerr << "[Detector] Unknown postprocess method: " << config.postprocess << std::endl;
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