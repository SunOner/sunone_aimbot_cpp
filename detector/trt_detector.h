#ifdef USE_CUDA
#ifndef TRT_DETECTOR_H
#define TRT_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <NvInfer.h>
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

#include "postProcess.h"

class TrtDetector
{
public:
    TrtDetector();
    ~TrtDetector();
    void initialize(const std::string& modelFile);
    void processFrame(const cv::Mat& frame);
    void inferenceThread();

    float img_scale;

    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::unordered_map<std::string, size_t> outputSizes;

    std::chrono::duration<double, std::milli> lastPreprocessTime;
    std::chrono::duration<double, std::milli> lastInferenceTime;
    std::chrono::duration<double, std::milli> lastCopyTime;
    std::chrono::duration<double, std::milli> lastPostprocessTime;
    std::chrono::duration<double, std::milli> lastNmsTime;

    std::vector<std::vector<Detection>> detectBatch(const std::vector<cv::Mat>& frames);

private:
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;

    cudaStream_t stream;

    bool useCudaGraph;
    bool cudaGraphCaptured;
    cudaGraph_t cudaGraph;
    cudaGraphExec_t cudaGraphExec;
    void captureCudaGraph();
    void launchCudaGraph();
    void destroyCudaGraph();

    std::unordered_map<std::string, void*> pinnedOutputBuffers;

    std::mutex inferenceMutex;
    std::condition_variable inferenceCV;
    std::atomic<bool> shouldExit;
    cv::Mat currentFrame;
    bool frameReady;

    void loadEngine(const std::string& engineFile);

    void preProcess(const cv::Mat& frame);

    void postProcess(
        const float* output,
        const std::string& outputName,
        std::chrono::duration<double, std::milli>* nmsTime
    );

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

    cv::cuda::GpuMat resizedBuffer;
    cv::cuda::GpuMat floatBuffer;
    std::vector<cv::cuda::GpuMat> channelBuffers;
};

#endif // TRT_DETECTOR_H
#endif