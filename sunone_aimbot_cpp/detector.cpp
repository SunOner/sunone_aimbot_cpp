#include "detector.h"
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "nvinf.h"
#include <algorithm>
#include "sunone_aimbot_cpp.h"

using namespace std;

extern Logger logger;

std::mutex frameMutex;
std::atomic<bool> isProcessing{ false };

Detector::Detector() : runtime(nullptr), engine(nullptr), context(nullptr), frameReady(false), shouldExit(false), newDetectionAvailable(false), detectionVersion(0)
{
    cudaStreamCreate(&stream);
}

Detector::~Detector()
{
    if (context) delete context;
    if (engine) delete engine;
    if (runtime) delete runtime;
    cudaStreamDestroy(stream);
}

void Detector::initialize(const std::string& modelFile)
{
    runtime = nvinfer1::createInferRuntime(gLogger);
    loadEngine(modelFile);
    context = engine->createExecutionContext();
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

        std::vector<float> inputData(3 * 640 * 640);
        preProcess(frame, inputData.data());

        void* d_input;
        cudaMalloc(&d_input, inputData.size() * sizeof(float));
        cudaMemcpyAsync(d_input, inputData.data(), inputData.size() * sizeof(float), cudaMemcpyHostToDevice, stream);

        const int outputSize = 1800;
        std::vector<float> outputData(outputSize);
        void* d_output;
        cudaMalloc(&d_output, outputSize * sizeof(float));

        context->setTensorAddress("images", d_input);
        context->setTensorAddress("output0", d_output);
        context->enqueueV3(stream);

        cudaMemcpyAsync(outputData.data(), d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        postProcess(outputData.data(), outputSize);

        cudaFree(d_input);
        cudaFree(d_output);
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

    int channels = 3;
    int height = frame.rows;
    int width = frame.cols;

    for (int c = 0; c < channels; c++)
    {
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                inputBuffer[c * height * width + h * width + w] = floatMat.at<cv::Vec3f>(h, w)[c];
            }
        }
    }
}

void Detector::postProcess(float* output, int outputSize)
{
    std::vector<cv::Rect> boxes;
    std::vector<int> classes;
    float confidence_threshold = 0.3;

    int num_boxes = outputSize / 6;

    for (int i = 0; i < num_boxes; i++)
    {
        float* box = output + i * 6;
        float confidence = box[4];
        int class_id = static_cast<int>(box[5]);

        if (confidence > confidence_threshold)
        {
            float x = box[0];
            float y = box[1];
            float w = box[2];
            float h = box[3];
            
            float scale_x = static_cast<float>(detection_window_width) / 640.0f;
            float scale_y = static_cast<float>(detection_window_height) / 640.0f;

            int x1 = static_cast<int>(x * scale_x);
            int y1 = static_cast<int>(y * scale_y);
            int x2 = static_cast<int>(w * scale_x) - x1;
            int y2 = static_cast<int>(h * scale_y) - y1;
            
            boxes.emplace_back(x1, y1, x2, y2);
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