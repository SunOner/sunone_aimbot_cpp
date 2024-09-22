#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <algorithm>

#include "detector.h"
#include "nvinf.h"
#include "sunone_aimbot_cpp.h"
#include "config.h"

using namespace std;

extern Logger logger;
extern Config config;

std::mutex frameMutex;
std::atomic<bool> isProcessing{ false };
extern Config config;
extern std::atomic<bool> detectionPaused;

Detector::Detector()
    : runtime(nullptr), engine(nullptr), context(nullptr),
    frameReady(false), shouldExit(false), newDetectionAvailable(false), detectionVersion(0)
{
    cudaStreamCreate(&stream);
}

Detector::~Detector()
{
    if (context) delete context;
    if (engine) delete engine;
    if (runtime) delete runtime;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
}

void Detector::initialize(const std::string& modelFile)
{
    runtime = nvinfer1::createInferRuntime(gLogger);
    loadEngine(modelFile);
    context = engine->createExecutionContext();

    auto inputDims = engine->getTensorShape("images");
    auto outputDims = engine->getTensorShape("output0");

    if (inputDims.d[0] == -1)
    {
        inputDims.d[0] = 1;
        context->setInputShape("images", inputDims);
    }

    inputSize = 1;
    for (int i = 0; i < inputDims.nbDims; ++i)
    {
        inputSize *= inputDims.d[i];
    }

    outputSize = 1;
    for (int i = 0; i < outputDims.nbDims; ++i)
    {
        outputSize *= outputDims.d[i];
    }

    cudaMalloc(&d_input, inputSize * sizeof(float));
    cudaMalloc(&d_output, outputSize * sizeof(float));
}

void Detector::loadEngine(const std::string& engineFile)
{
    std::ifstream file(engineFile, std::ios::binary);
    if (!file.good())
    {
        std::cerr << "Error opening engine file" << std::endl;
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
    latestFrame = frame.clone();
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

        std::vector<float> inputData(inputSize);
        preProcess(frame, inputData.data());

        cudaMemcpyAsync(d_input, inputData.data(), inputSize * sizeof(float), cudaMemcpyHostToDevice, stream);

        context->setTensorAddress("images", d_input);
        context->setTensorAddress("output0", d_output);

        context->enqueueV3(stream);

        std::vector<float> outputData(outputSize);
        cudaMemcpyAsync(outputData.data(), d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        postProcess(outputData.data(), outputSize);
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
    cv::Mat floatMat;
    frame.convertTo(floatMat, CV_32F, 1.0 / 255.0);

    cv::cvtColor(floatMat, floatMat, cv::COLOR_BGR2RGB);
    std::vector<cv::Mat> channels(3);
    cv::split(floatMat, channels);

    int channelSize = frame.cols * frame.rows;
    for (int i = 0; i < 3; ++i)
    {
        memcpy(inputBuffer + i * channelSize, channels[i].data, channelSize * sizeof(float));
    }
}

void Detector::postProcess(float* output, int outputSize)
{
    std::vector<cv::Rect> boxes;
    std::vector<int> classes;

    int num_boxes = outputSize / 6;

    for (int i = 0; i < num_boxes; i++)
    {
        float* box = output + i * 6;
        float confidence = box[4];
        int class_id = static_cast<int>(box[5]);

        if (confidence > config.confidence_threshold)
        {
            float x = box[0];
            float y = box[1];
            float w = box[2];
            float h = box[3];

            float scale = static_cast<float>(config.detection_resolution) / config.engine_image_size;

            int x1 = static_cast<int>(x * scale);
            int y1 = static_cast<int>(y * scale);
            int width = static_cast<int>((w - x) * scale);
            int height = static_cast<int>((h - y) * scale);

            boxes.emplace_back(x1, y1, width, height);
            classes.push_back(class_id);
        }
    }

    {
        std::lock_guard<std::mutex> lock(detectionMutex);
        detectedBoxes = std::move(boxes);
        detectedClasses = std::move(classes);
        detectionVersion++;
    }

    detectionCV.notify_one();
}