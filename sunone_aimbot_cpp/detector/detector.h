#ifndef DETECTOR_H
#define DETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include "NvInfer.h"
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <unordered_map>
#include <cuda_fp16.h>
#include <memory>
#include <thread>
#include <chrono>
#include <functional>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <cuda_runtime_api.h>

#include "../config/config.h"

class Detector
{
public:
    Detector();
    ~Detector();
    void initialize(const std::string& modelFile);
    void processFrame(const cv::cuda::GpuMat& frame);
    void inferenceThread();
    void releaseDetections();
    bool getLatestDetections(std::vector<cv::Rect>& boxes, std::vector<int>& classes);

    std::mutex detectionMutex;

    int detectionVersion;
    std::condition_variable detectionCV;
    std::vector<cv::Rect> detectedBoxes;
    std::vector<int> detectedClasses;
    float img_scale;

    int getInputHeight() const { return inputDims.d[2]; }
    int getInputWidth() const { return inputDims.d[3]; }

    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::unordered_map<std::string, size_t> outputSizes;

private:
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    nvinfer1::Dims inputDims;

    // OpenCV CUDA streams
    cv::cuda::Stream cvStream;
    cv::cuda::Stream preprocessCvStream;
    cv::cuda::Stream postprocessCvStream;

    // Native CUDA streams (for TensorRT)
    cudaStream_t stream;
    cudaStream_t preprocessStream;
    cudaStream_t postprocessStream;

    // CUDA Graph variables
    cudaGraph_t cudaGraph;
    cudaGraphExec_t cudaGraphExec;
    bool cudaGraphCaptured;
    bool capturedBefore;
    bool useCudaGraph;

    // Pinned memory for outputs
    std::unordered_map<std::string, void*> pinnedOutputBuffers;
    bool usePinnedMemory;

    std::mutex inferenceMutex;
    std::condition_variable inferenceCV;
    std::atomic<bool> shouldExit;
    cv::cuda::GpuMat currentFrame;
    bool frameReady;

    void loadEngine(const std::string& engineFile);
    void preProcess(const cv::cuda::GpuMat& frame);
    void postProcess(const float* output, const std::string& outputName);
    void getInputNames();
    void getOutputNames();
    void getBindings();

    std::unordered_map<std::string, size_t> inputSizes;
    std::unordered_map<std::string, void*> inputBindings;
    std::unordered_map<std::string, void*> outputBindings;
    std::unordered_map<std::string, std::vector<int64_t>> outputShapes;
    int numClasses;

    size_t getSizeByDim(const nvinfer1::Dims& dims);
    size_t getElementSize(nvinfer1::DataType dtype);

    std::string inputName;
    void* inputBufferDevice;
    std::unordered_map<std::string, std::vector<float>> outputDataBuffers;
    std::unordered_map<std::string, std::vector<__half>> outputDataBuffersHalf;
    std::unordered_map<std::string, nvinfer1::DataType> outputTypes;

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classes;
    
    // Reusable GPU buffers for preprocessing
    cv::cuda::GpuMat resizedBuffer;
    cv::cuda::GpuMat floatBuffer;
    std::vector<cv::cuda::GpuMat> channelBuffers;
    
    // Helper function to synchronize OpenCV CUDA stream with CUDA stream
    void synchronizeStreams(cv::cuda::Stream& cvStream, cudaStream_t cudaStream) {
        cudaEvent_t event;
        cudaEventCreate(&event);
        
        // Record event when OpenCV stream completes
        cvStream.enqueueHostCallback([](int, void* userData) {
            cudaEventRecord(static_cast<cudaEvent_t>(userData));
        }, &event);
        
        // Make CUDA stream wait for the event
        cudaStreamWaitEvent(cudaStream, event, 0);
        
        // Clean up
        cudaEventDestroy(event);
    }
};

#endif // DETECTOR_H