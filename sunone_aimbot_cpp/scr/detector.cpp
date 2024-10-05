#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <algorithm>

#include "detector.h"
#include "nvinf.h"
#include "sunone_aimbot_cpp.h"

using namespace std;

extern Logger logger;

std::mutex frameMutex;
extern std::atomic<bool> detectionPaused;

Detector::Detector()
    : runtime(nullptr), engine(nullptr), context(nullptr),
    frameReady(false), shouldExit(false), detectionVersion(0)
{
    cudaStreamCreate(&stream);
}

Detector::~Detector()
{
    if (context) delete context;
    if (engine) delete engine;
    if (runtime) delete runtime;
    cudaStreamDestroy(stream);

    for (auto& binding : inputBindings)
    {
        cudaFree(binding.second);
    }

    for (auto& binding : outputBindings)
    {
        cudaFree(binding.second);
    }
}

void Detector::initialize(const std::string& modelFile)
{
    runtime = nvinfer1::createInferRuntime(gLogger);
    loadEngine(modelFile);
    context = engine->createExecutionContext();

    getInputNames();
    getOutputNames();
    getBindings();
    getNumberOfClasses();

    if (!inputNames.empty())
    {
        const std::string& inputName = inputNames[0];
        inputDims = engine->getTensorShape(inputName.c_str());
        size_t inputSize = getSizeByDim(inputDims);
        inputBuffer.resize(inputSize);
    }
    else
    {
        // TODO: (no input tensors found)
    }

    for (const auto& name : outputNames)
    {
        size_t size = outputSizes[name];
        outputDataBuffers[name].resize(size / sizeof(float));
    }

    channels.resize(3);
    scale = static_cast<float>(config.detection_resolution) / config.engine_image_size;
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
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF:  return 2;
    case nvinfer1::DataType::kINT8:  return 1;
    case nvinfer1::DataType::kINT32: return 4;
    default: return 0;
    }
}

void Detector::getInputNames()
{
    for (int i = 0; i < engine->getNbIOTensors(); ++i)
    {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
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

            std::vector<int> dim;
            for (int j = 0; j < dims.nbDims; ++j)
            {
                dim.push_back(dims.d[j]);
            }
            outputShapes[name] = dim;
        }
    }
}

void Detector::getBindings()
{
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

void Detector::getNumberOfClasses()
{
    if (outputNames.size() > 0)
    {
        const std::string& outputName = outputNames[0];
        const std::vector<int>& dims = outputShapes[outputName];
        if (dims.size() > 0)
        {
            numClasses = dims.back() - 5;
        }
    }
}

void Detector::loadEngine(const std::string& engineFile)
{
    std::ifstream file(engineFile, std::ios::binary);
    if (!file.good())
    {
        std::cerr << "Error opening engine file" << std::endl;
        std::cin.get();
        return;
    }

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    engine = runtime->deserializeCudaEngine(engineData.data(), size);
}

void Detector::processFrame(const cv::Mat& frame)
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
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lock(inferenceMutex);
            inferenceCV.wait(lock, [this] { return frameReady || shouldExit; });
            if (shouldExit) break;
            frame = std::move(currentFrame);
            frameReady = false;
        }

        const std::string& inputName = inputNames[0];

        preProcess(frame, inputBuffer.data());

        cudaMemcpyAsync(inputBindings[inputName], inputBuffer.data(), inputSizes[inputName], cudaMemcpyHostToDevice, stream);

        context->setTensorAddress(inputName.c_str(), inputBindings[inputName]);

        for (const auto& name : outputNames)
        {
            context->setTensorAddress(name.c_str(), outputBindings[name]);
        }

        context->enqueueV3(stream);

        for (const auto& name : outputNames)
        {
            size_t size = outputSizes[name];
            std::vector<float>& outputData = outputDataBuffers[name];
            cudaMemcpyAsync(outputData.data(), outputBindings[name], size, cudaMemcpyDeviceToHost, stream);
        }
        cudaStreamSynchronize(stream);

        for (const auto& name : outputNames)
        {
            const std::vector<float>& outputData = outputDataBuffers[name];
            postProcess(outputData.data(), outputData.size());
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

void Detector::preProcess(const cv::Mat& frame, float* inputBuffer)
{
    int inputH = inputDims.d[2];
    int inputW = inputDims.d[3];

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(inputW, inputH));
    cv::Mat floatMat;
    resized.convertTo(floatMat, CV_32F, 1.0 / 255.0);

    cv::cvtColor(floatMat, floatMat, cv::COLOR_BGR2RGB);
    cv::split(floatMat, channels);

    int channelSize = inputH * inputW;
    for (int i = 0; i < 3; ++i)
    {
        memcpy(inputBuffer + i * channelSize, channels[i].data, channelSize * sizeof(float));
    }
}

void Detector::postProcess(const float* output, int outputSize)
{
    boxes.clear();
    confidences.clear();
    classes.clear();

    int numDetections = outputSize / 6;
    for (int i = 0; i < numDetections; ++i)
    {
        const float* det = output + i * 6;
        float confidence = det[4];
        if (confidence > config.confidence_threshold)
        {
            int classId = static_cast<int>(det[5]);

            float x = det[0];
            float y = det[1];
            float w = det[2];
            float h = det[3];

            int x1 = static_cast<int>(x * scale);
            int y1 = static_cast<int>(y * scale);
            int width = static_cast<int>((w - x) * scale);
            int height = static_cast<int>((h - y) * scale);

            boxes.emplace_back(x1, y1, width, height);
            confidences.push_back(confidence);
            classes.push_back(classId);
        }
    }

    std::vector<int> nmsIndices;
    cv::dnn::NMSBoxes(boxes, confidences, config.confidence_threshold, config.nms_threshold, nmsIndices);

    std::vector<cv::Rect> nmsBoxes;
    std::vector<int> nmsClasses;

    for (int idx : nmsIndices)
    {
        nmsBoxes.push_back(boxes[idx]);
        nmsClasses.push_back(classes[idx]);
    }

    {
        std::lock_guard<std::mutex> lock(detectionMutex);
        detectedBoxes = std::move(nmsBoxes);
        detectedClasses = std::move(nmsClasses);
        detectionVersion++;
    }

    detectionCV.notify_one();
}