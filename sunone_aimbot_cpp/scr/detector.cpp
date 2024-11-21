#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <algorithm>
#include <cuda_fp16.h>
#include <atomic>

#include "detector.h"
#include "nvinf.h"
#include "sunone_aimbot_cpp.h"

std::mutex frameMutex;
extern std::atomic<bool> detectionPaused;
int model_quant;
std::vector<float> outputData;

extern std::atomic<bool> detector_model_changed;

Detector::Detector()
    : frameReady(false), shouldExit(false), detectionVersion(0)
{
    cudaStreamCreate(&stream);
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
        std::cerr << "[Detector] Error loading the engine from a file " << modelFile << std::endl;
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

    if (!inputNames.empty())
    {
        const std::string& inputName = inputNames[0];
        inputDims = engine->getTensorShape(inputName.c_str());
        size_t inputSize = getSizeByDim(inputDims);
        inputBuffer.resize(inputSize);
    }
    else
    {
        std::cerr << "[Detector] No input tensors found" << std::endl;
    }

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

void Detector::loadEngine(const std::string& engineFile)
{
    std::ifstream file(engineFile, std::ios::binary);
    if (!file.good())
    {
        std::cerr << "[Detector] Error opening the engine file: " << engineFile << std::endl;
        return;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    engine.reset(runtime->deserializeCudaEngine(engineData.data(), size));

    if (!engine)
    {
        std::cerr << "[Detector] Engine deserialization error from file: " << engineFile << std::endl;
        return;
    }

    std::cout << "[Detector] The engine was successfully loaded from the file: " << engineFile << std::endl;
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
            detector_model_changed.store(false);
        }

        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lock(inferenceMutex);
            inferenceCV.wait(lock, [this] { return frameReady || shouldExit; });
            if (shouldExit) break;
            frame = std::move(currentFrame);
            frameReady = false;
        }

        if (!context)
        {
            std::cerr << "[Detector] The context is not initialized" << std::endl;
            continue;
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

                postProcess(outputData.data(), static_cast<int>(outputData.size()));
            }
            else if (dtype == nvinfer1::DataType::kFLOAT)
            {
                const std::vector<float>& outputData = outputDataBuffers[name];
                postProcess(outputData.data(), static_cast<int>(outputData.size()));
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

    const auto& shape = outputShapes[outputNames[0]];

    if (shape.size() != 3)
    {
        std::cerr << "Unsupported output shape" << std::endl;
        return;
    }

    int batch_size = shape[0];
    int dim1 = shape[1];
    int dim2 = shape[2];
    
    if (batch_size != 1)
    {
        std::cerr << "Batch size > 1 is not supported" << std::endl;
        return;
    }

    if (dim2 == 6) // [1, 300, 6]
    {
        int numDetections = dim1;

        for (int i = 0; i < numDetections; ++i)
        {
            const float* det = output + i * dim2;
            float confidence = det[4];

            if (confidence > config.confidence_threshold)
            {
                int classId = static_cast<int>(det[5]);

                float x_min = det[0];
                float y_min = det[1];
                float x_max = det[2];
                float y_max = det[3];

                int x1 = static_cast<int>(x_min * scale);
                int y1 = static_cast<int>(y_min * scale);
                int width = static_cast<int>((x_max - x_min) * scale);
                int height = static_cast<int>((y_max - y_min) * scale);

                boxes.emplace_back(cv::Rect(x1, y1, width, height));
                confidences.push_back(confidence);
                classes.push_back(classId);
            }
        }
    }
    else if (dim1 == 15 && dim2 == 8400 or dim1 == 15 && dim2 == 8400 / 4) // [1, 15, 8400] or [1, 15, 2100] Yolov11
    {
        int channels = dim1;
        int cols = dim2;

        std::vector<std::vector<float>> detections;
        for (int i = 0; i < cols; ++i)
        {
            std::vector<float> detection;
            for (int j = 0; j < channels; ++j)
            {
                detection.push_back(output[j * cols + i]);
            }
            detections.push_back(detection);
        }

        for (const auto& detection : detections)
        {
            float confidence = *std::max_element(detection.begin() + 4, detection.end());

            if (confidence > config.confidence_threshold)
            {
                int classId = std::distance(detection.begin() + 4, std::max_element(detection.begin() + 4, detection.end()));

                float x = detection[0];
                float y = detection[1];
                float w = detection[2];
                float h = detection[3];

                int x1 = static_cast<int>((x - w / 2) * scale);
                int y1 = static_cast<int>((y - h / 2) * scale);
                int width = static_cast<int>(w * scale);
                int height = static_cast<int>(h * scale);

                boxes.emplace_back(x1, y1, width, height);
                confidences.push_back(confidence);
                classes.push_back(classId);
            }
        }
    }
    else if (dim1 == 5 && dim2 == 8400) // [1, 5, 8400] with one class output
    {
        int channels = dim1;
        int cols = dim2;

        for (int i = 0; i < cols; ++i)
        {
            float x = output[i];
            float y = output[cols + i];
            float w = output[2 * cols + i];
            float h = output[3 * cols + i];
            float confidence = output[4 * cols + i];

            if (confidence > config.confidence_threshold)
            {
                int classId = 0;

                int x1 = static_cast<int>((x - w / 2) * scale);
                int y1 = static_cast<int>((y - h / 2) * scale);
                int width = static_cast<int>(w * scale);
                int height = static_cast<int>(h * scale);

                boxes.emplace_back(x1, y1, width, height);
                confidences.push_back(confidence);
                classes.push_back(classId);
            }
        }
    }
    else if (dim1 == 7 && dim2 == 8400) // [1, 7, 8400] Yolov9
    {
        int channels = dim1;
        int cols = dim2;

        for (int i = 0; i < cols; ++i)
        {
            float x_center = output[i];
            float y_center = output[cols + i];
            float width = output[2 * cols + i];
            float height = output[3 * cols + i];
            float objectness = output[4 * cols + i];
            float class_confidence = output[5 * cols + i];
            int classId = static_cast<int>(output[6 * cols + i]);

            float confidence = objectness * class_confidence;

            if (confidence > config.confidence_threshold)
            {
                int x1 = static_cast<int>((x_center - width / 2) * scale);
                int y1 = static_cast<int>((y_center - height / 2) * scale);
                int w = static_cast<int>(width * scale);
                int h = static_cast<int>(height * scale);

                boxes.emplace_back(x1, y1, w, h);
                confidences.push_back(confidence);
                classes.push_back(classId);
            }
        }
    }
    else
    {
        std::cerr << "Unknown output shape(" << shape[0] << "," << shape[1] << "," << shape[2] << ")" << std::endl;
        return;
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