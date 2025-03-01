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
#include <opencv2/core/cuda.hpp>

#include <algorithm>
#include <cuda_fp16.h>
#include <atomic>
#include <numeric>
#include <vector>
#include <queue>
#include <mutex>

#include "detector.h"
#include "nvinf.h"
#include "sunone_aimbot_cpp.h"
#include "other_tools.h"
#include "postProcess.h"

// OpenCV와 호환되는 스트림 처리
cudaStream_t getStreamFromOpenCVStream(const cv::cuda::Stream&) {
    return 0;  // 항상 기본 CUDA 스트림 반환
}

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
    img_scale(1.0f),
    numClasses(0),
    useCudaGraph(false),
    cudaGraphCaptured(false)
{
    // CUDA 스트림 생성
    cudaStreamCreate(&stream);

    // OpenCV CUDA 스트림 초기화
    cvStream = cv::cuda::Stream();
    preprocessCvStream = cv::cuda::Stream();
    postprocessCvStream = cv::cuda::Stream();

    // CUDA Graph 변수 초기화 (사용하지 않음)
    cudaGraph = nullptr;
    cudaGraphExec = nullptr;
}

Detector::~Detector()
{
    // CUDA 리소스 정리
    cudaStreamDestroy(stream);
    
    if (cudaGraphCaptured) {
        cudaGraphExecDestroy(cudaGraphExec);
        cudaGraphDestroy(cudaGraph);
    }

    // 메모리 해제
    for (auto& buffer : pinnedOutputBuffers) {
        if (buffer.second) cudaFreeHost(buffer.second);
    }

    for (auto& binding : inputBindings) {
        if (binding.second) cudaFree(binding.second);
    }

    for (auto& binding : outputBindings) {
        if (binding.second) cudaFree(binding.second);
    }

    if (inputBufferDevice) {
        cudaFree(inputBufferDevice);
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
            inputNames.emplace_back(name);
            if (config.verbose) {
                std::cout << "[Detector] Detected input: " << name << std::endl;
            }
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
            outputTypes[name] = engine->getTensorDataType(name);
            
            if (config.verbose) {
                std::cout << "[Detector] Detected output: " << name << std::endl;
            }
        }
    }
}

void Detector::getBindings()
{
    // 기존 메모리 해제
    for (auto& binding : inputBindings) {
        if (binding.second) cudaFree(binding.second);
    }
    inputBindings.clear();

    for (auto& binding : outputBindings) {
        if (binding.second) cudaFree(binding.second);
    }
    outputBindings.clear();

    // 새로운 메모리 할당
    for (const auto& name : inputNames)
    {
        size_t size = inputSizes[name];
        if (size > 0) {
            void* ptr = nullptr;
            cudaError_t err = cudaMalloc(&ptr, size);
            if (err == cudaSuccess) {
                inputBindings[name] = ptr;
                if (config.verbose) {
                    std::cout << "[Detector] Allocated " << size << " bytes for input " << name << std::endl;
                }
            } else {
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
            if (err == cudaSuccess) {
                outputBindings[name] = ptr;
                if (config.verbose) {
                    std::cout << "[Detector] Allocated " << size << " bytes for output " << name << std::endl;
                }
            } else {
                std::cerr << "[Detector] Failed to allocate output memory: " << cudaGetErrorString(err) << std::endl;
            }
        }
    }
}

// 간소화된 초기화 함수
void Detector::initialize(const std::string& modelFile)
{
    // TensorRT 런타임 초기화
    runtime.reset(nvinfer1::createInferRuntime(gLogger));
    loadEngine(modelFile);

    if (!engine) {
        std::cerr << "[Detector] Engine loading failed" << std::endl;
        return;
    }

    // 실행 컨텍스트 생성
    context.reset(engine->createExecutionContext());
    if (!context) {
        std::cerr << "[Detector] Context creation failed" << std::endl;
        return;
    }

    // 입출력 텐서 정보 초기화
    getInputNames();
    getOutputNames();

    if (inputNames.empty()) {
        std::cerr << "[Detector] No input tensors found" << std::endl;
        return;
    }

    // 기본 입력 텐서 설정
    inputName = inputNames[0];
    context->setInputShape(inputName.c_str(), nvinfer1::Dims4{1, 3, 640, 640});
    if (!context->allInputDimensionsSpecified()) {
        std::cerr << "[Detector] Failed to set input dimensions" << std::endl;
        return;
    }

    // 입출력 텐서 메모리 크기 계산
    for (const auto& inName : inputNames) {
        nvinfer1::Dims dims = context->getTensorShape(inName.c_str());
        nvinfer1::DataType dtype = engine->getTensorDataType(inName.c_str());
        inputSizes[inName] = getSizeByDim(dims) * getElementSize(dtype);
    }

    for (const auto& outName : outputNames) {
        nvinfer1::Dims dims = context->getTensorShape(outName.c_str());
        nvinfer1::DataType dtype = engine->getTensorDataType(outName.c_str());
        outputSizes[outName] = getSizeByDim(dims) * getElementSize(dtype);
        
        std::vector<int64_t> shape;
        for (int j = 0; j < dims.nbDims; j++) {
            shape.push_back(dims.d[j]);
        }
        outputShapes[outName] = shape;
    }

    // GPU 메모리 할당
    getBindings();

    // 모델 클래스 수 설정
    if (!outputNames.empty()) {
        const std::string& mainOut = outputNames[0];
        nvinfer1::Dims outDims = context->getTensorShape(mainOut.c_str());

        if (config.postprocess == "yolo10") {
            numClasses = 11;
        } else {
            numClasses = outDims.d[1] - 4;
        }
    }

    // 이미지 스케일 계산
    img_scale = static_cast<float>(config.detection_resolution) / 640;
    
    // CUDA 그래프 비활성화 (OpenCV 텍스처와 호환성 문제로 인해)
    useCudaGraph = false;
    if (config.use_cuda_graph && config.verbose) {
        std::cout << "[Detector] CUDA Graph disabled due to OpenCV texture compatibility issues" << std::endl;
    }
    
    // 최적화를 위한 버퍼 미리 생성
    nvinfer1::Dims dims = context->getTensorShape(inputName.c_str());
    int c = dims.d[1];
    int h = dims.d[2];
    int w = dims.d[3];
    
    // 메모리 사전 할당
    resizedBuffer.create(h, w, CV_8UC3);
    floatBuffer.create(h, w, CV_32FC3);
    
    channelBuffers.resize(c);
    for (int i = 0; i < c; ++i) {
        channelBuffers[i].create(h, w, CV_32F);
    }
    
    // 텐서 주소 설정
    for (const auto& name : inputNames) {
        context->setTensorAddress(name.c_str(), inputBindings[name]);
    }
    for (const auto& name : outputNames) {
        context->setTensorAddress(name.c_str(), outputBindings[name]);
    }
    
    // 핀 메모리 초기화
    if (config.use_pinned_memory) {
        for (const auto& outName : outputNames) {
            size_t size = outputSizes[outName];
            void* hostBuffer = nullptr;
            cudaError_t status = cudaMallocHost(&hostBuffer, size);
            if (status == cudaSuccess) {
                pinnedOutputBuffers[outName] = hostBuffer;
            }
        }
    }
}

size_t Detector::getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) return 0;
        size *= dims.d[i];
    }
    return size;
}

size_t Detector::getElementSize(nvinfer1::DataType dtype)
{
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kINT8: return 1;
        default: return 0;
    }
}

void Detector::loadEngine(const std::string& modelFile)
{
    std::string engineFilePath;
    std::filesystem::path modelPath(modelFile);
    std::string extension = modelPath.extension().string();

    // 엔진 파일 경로 결정
    if (extension == ".engine") {
        engineFilePath = modelFile;
    } else if (extension == ".onnx") {
        // ONNX 모델에서 엔진 생성
        engineFilePath = modelPath.replace_extension(".engine").string();
        if (!fileExists(engineFilePath)) {
            std::cout << "[Detector] Building engine from ONNX model" << std::endl;
            nvinfer1::ICudaEngine* builtEngine = buildEngineFromOnnx(modelFile, gLogger);
            if (builtEngine) {
                nvinfer1::IHostMemory* serializedEngine = builtEngine->serialize();
                if (serializedEngine) {
                    // 엔진 파일 저장
                    std::ofstream engineFile(engineFilePath, std::ios::binary);
                    if (engineFile) {
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
    } else {
        std::cerr << "[Detector] Unsupported model format: " << extension << std::endl;
        return;
    }

    // 엔진 파일 로드
    std::cout << "[Detector] Loading engine: " << engineFilePath << std::endl;
    engine.reset(loadEngineFromFile(engineFilePath, runtime.get()));
}

void Detector::processFrame(const cv::cuda::GpuMat& frame)
{
    // 감지가 일시 정지된 경우
    if (detectionPaused) {
        std::lock_guard<std::mutex> lock(detectionMutex);
        detectedBoxes.clear();
        detectedClasses.clear();
        return;
    }

    // 프레임 전달 (zero-copy)
    std::unique_lock<std::mutex> lock(inferenceMutex);
    currentFrame = frame;  // 레퍼런스만 전달
    frameReady = true;
    inferenceCV.notify_one();
}

// 단순화된 추론 스레드
void Detector::inferenceThread()
{
    while (!shouldExit)
    {
        // 모델 변경 처리
        if (detector_model_changed.load()) {
            {
                std::unique_lock<std::mutex> lock(inferenceMutex);
                
                // 리소스 정리
                context.reset();
                engine.reset();
                
                // 바인딩 메모리 해제
                for (auto& binding : inputBindings) {
                    if (binding.second) cudaFree(binding.second);
                }
                inputBindings.clear();
                
                for (auto& binding : outputBindings) {
                    if (binding.second) cudaFree(binding.second);
                }
                outputBindings.clear();
            }
            
            // 새 모델 초기화
            initialize("models/" + config.ai_model);
            
            detection_resolution_changed.store(true);
            detector_model_changed.store(false);
        }
        
        // 프레임 가져오기
        cv::cuda::GpuMat frame;
        bool hasNewFrame = false;
        
        {
            std::unique_lock<std::mutex> lock(inferenceMutex);
            
            // 프레임 준비 또는 종료 신호 대기
            if (!frameReady && !shouldExit) {
                inferenceCV.wait(lock, [this] { return frameReady || shouldExit; });
            }
            
            if (shouldExit) break;
            
            if (frameReady) {
                frame = std::move(currentFrame);
                frameReady = false;
                hasNewFrame = true;
            }
        }
        
        // 컨텍스트 검사
        if (!context) {
            if (!error_logged) {
                std::cerr << "[Detector] Context not initialized" << std::endl;
                error_logged = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        } else {
            error_logged = false;
        }
        
        // 프레임 처리
        if (hasNewFrame && !frame.empty()) {
            try {
                // 1단계: 전처리
                preProcess(frame);
                
                // 전처리 완료 대기
                cudaStreamSynchronize(stream);
                
                // 2단계: 추론 실행 (직접 실행)
                context->enqueueV3(stream);
                
                // 추론 완료 대기
                cudaStreamSynchronize(stream);
                
                // 3단계: 결과 복사 및 후처리
                for (const auto& name : outputNames) {
                    size_t size = outputSizes[name];
                    nvinfer1::DataType dtype = outputTypes[name];
                    
                    if (dtype == nvinfer1::DataType::kHALF) {
                        // FP16 처리
                        size_t numElements = size / sizeof(__half);
                        std::vector<__half>& outputDataHalf = outputDataBuffersHalf[name];
                        outputDataHalf.resize(numElements);
                        
                        cudaMemcpy(
                            outputDataHalf.data(),
                            outputBindings[name],
                            size,
                            cudaMemcpyDeviceToHost);
                            
                        // FP16에서 FP32로 변환
                        std::vector<float> outputDataFloat(outputDataHalf.size());
                        for (size_t i = 0; i < outputDataHalf.size(); ++i) {
                            outputDataFloat[i] = __half2float(outputDataHalf[i]);
                        }
                        
                        postProcess(outputDataFloat.data(), name);
                    } else if (dtype == nvinfer1::DataType::kFLOAT) {
                        // FP32 직접 사용
                        std::vector<float>& outputData = outputDataBuffers[name];
                        outputData.resize(size / sizeof(float));
                        
                        cudaMemcpy(
                            outputData.data(),
                            outputBindings[name],
                            size,
                            cudaMemcpyDeviceToHost);
                            
                        postProcess(outputData.data(), name);
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "[Detector] Error during inference: " << e.what() << std::endl;
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
    if (!detectedBoxes.empty()) {
        boxes = detectedBoxes;
        classes = detectedClasses;
        return true;
    }
    return false;
}

// 최적화된 전처리 함수
void Detector::preProcess(const cv::cuda::GpuMat& frame) {
    if (frame.empty()) return;

    // 입력 버퍼 확인
    void* inputBuffer = inputBindings[inputName];
    if (!inputBuffer) return;

    // 텐서 차원 얻기
    nvinfer1::Dims dims = context->getTensorShape(inputName.c_str());
    int c = dims.d[1];  // 채널 수
    int h = dims.d[2];  // 높이
    int w = dims.d[3];  // 너비

    try {
        // 이미지 처리
        cv::cuda::resize(frame, resizedBuffer, cv::Size(w, h), 0, 0, cv::INTER_LINEAR, preprocessCvStream);
        resizedBuffer.convertTo(floatBuffer, CV_32F, 1.0f / 255.0f, 0, preprocessCvStream);
        cv::cuda::split(floatBuffer, channelBuffers, preprocessCvStream);
        
        // OpenCV 작업 완료 대기
        preprocessCvStream.waitForCompletion();
        
        // 채널 데이터 복사 - 동기화 방법 최적화
        size_t channelSize = h * w * sizeof(float);
        for (int i = 0; i < c; ++i) {
            cudaMemcpyAsync(
                static_cast<float*>(inputBuffer) + i * h * w,
                channelBuffers[i].ptr<float>(),
                channelSize,
                cudaMemcpyDeviceToDevice,
                stream  // 메인 스트림 사용
            );
        }
    } catch (const cv::Exception& e) {
        std::cerr << "[Detector] OpenCV error in preProcess: " << e.what() << std::endl;
    }
}

void Detector::postProcess(const float* output, const std::string& outputName)
{
    if (numClasses <= 0) return;

    std::vector<Detection> detections;

    // 후처리 방식 선택
    if (config.postprocess == "yolo10") {
        const std::vector<int64_t>& shape = outputShapes[outputName];
        detections = postProcessYolo10(
            output,
            shape,
            numClasses,
            config.confidence_threshold,
            config.nms_threshold
        );
    } else if (
        config.postprocess == "yolo8" ||
        config.postprocess == "yolo9" ||
        config.postprocess == "yolo11" ||
        config.postprocess == "yolo12"
    ) {
        auto shape = context->getTensorShape(outputName.c_str());
        std::vector<int64_t> engineShape;
        for (int i = 0; i < shape.nbDims; ++i) {
            engineShape.push_back(shape.d[i]);
        }

        detections = postProcessYolo11(
            output,
            engineShape,
            numClasses,
            config.confidence_threshold,
            config.nms_threshold
        );
    }

    // 결과 저장
    {
        std::lock_guard<std::mutex> lock(detectionMutex);
        detectedBoxes.clear();
        detectedClasses.clear();
        for (const auto& det : detections) {
            detectedBoxes.push_back(det.box);
            detectedClasses.push_back(det.classId);
        }
        detectionVersion++;
    }
    detectionCV.notify_one();
}