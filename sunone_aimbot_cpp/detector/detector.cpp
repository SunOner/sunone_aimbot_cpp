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

extern std::atomic<bool> detectionPaused;
int model_quant;
std::vector<float> outputData;

extern std::atomic<bool> detector_model_changed;
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
    int64_t inputH = inputDims.d[2];
    img_scale = static_cast<float>(config.detection_resolution) / inputH;
    std::cout << "[Detector] Image scale factor: " << img_scale << std::endl;
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

void Detector::preProcess(const cv::cuda::GpuMat& frame)
{
    cv::Mat frameCpu;
    frame.download(frameCpu);

    int inputC = inputDims.d[1];
    int inputH = inputDims.d[2];
    int inputW = inputDims.d[3];

    cv::Mat resizedImage;
    cv::resize(frameCpu, resizedImage, cv::Size(inputW, inputH));

    if (resizedImage.channels() == 4)
    {
        cv::cvtColor(resizedImage, resizedImage, cv::COLOR_BGRA2BGR);
    }
    else if (resizedImage.channels() != 3)
    {
        return;
    }

    cv::Mat inputBlob = cv::dnn::blobFromImage(
        resizedImage,
        1.0 / 255.0,
        cv::Size(inputW, inputH),
        cv::Scalar(),
        true,
        false
    );

    size_t inputDataSize = inputBlob.total() * inputBlob.elemSize();
    size_t expectedSize = inputSizes[inputName];

    if (inputDataSize != expectedSize)
    {
        return;
    }

    cudaMemcpyAsync(inputBufferDevice, inputBlob.ptr<float>(), inputDataSize, cudaMemcpyHostToDevice, stream);
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

    int64_t batch_size = shape[0];
    int64_t dim1 = shape[1];
    int64_t dim2 = shape[2];

    if (batch_size != 1)
    {
        std::cerr << "Batch size > 1 is not supported" << std::endl;
        return;
    }

    if (dim2 == 6) // [1, 300, 6]
    {
        int64_t numDetections = dim1;

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

                int x1 = static_cast<int>(x_min * img_scale);
                int y1 = static_cast<int>(y_min * img_scale);
                int width = static_cast<int>((x_max - x_min) * img_scale);
                int height = static_cast<int>((y_max - y_min) * img_scale);

                boxes.emplace_back(cv::Rect(x1, y1, width, height));
                confidences.push_back(confidence);
                classes.push_back(classId);
            }
        }
    }
    else if (dim1 == 15 && (dim2 == 8400 || dim2 == 2100)) // [1, 15, 8400] or [1, 15, 2100] Yolov11
    {
        int64_t channels = dim1;
        int64_t numDetections = dim2;
        const cv::Mat det_output(channels, numDetections, CV_32F, (void*)output);

        int num_classes = channels - 4;

        for (int i = 0; i < numDetections; ++i)
        {
            const cv::Mat classes_scores = det_output.col(i).rowRange(4, channels);

            double max_classes_score;
            cv::minMaxLoc(classes_scores, nullptr, &max_classes_score);

            cv::Mat exp_scores;
            cv::exp(classes_scores - max_classes_score, exp_scores);

            double sum_exp_scores = cv::sum(exp_scores)[0];

            cv::Mat probabilities = exp_scores / sum_exp_scores;

            cv::Point class_id_point;

            double max_class_score;
            cv::minMaxLoc(probabilities, nullptr, &max_class_score, nullptr, &class_id_point);

            if (max_class_score > config.confidence_threshold)
            {
                float x = det_output.at<float>(0, i);
                float y = det_output.at<float>(1, i);
                float w = det_output.at<float>(2, i);
                float h = det_output.at<float>(3, i);

                int x1 = static_cast<int>((x - w / 2) * img_scale);
                int y1 = static_cast<int>((y - h / 2) * img_scale);
                int width = static_cast<int>(w * img_scale);
                int height = static_cast<int>(h * img_scale);

                boxes.emplace_back(x1, y1, width, height);
                confidences.push_back(static_cast<float>(max_class_score));
                classes.push_back(class_id_point.y);
            }
        }
    }
    else if (dim1 == 5 && dim2 == 8400) // [1, 5, 8400] with one class output
    {
        int64_t channels = dim1;
        int64_t cols = dim2;

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

                int x1 = static_cast<int>((x - w / 2) * img_scale);
                int y1 = static_cast<int>((y - h / 2) * img_scale);
                int width = static_cast<int>(w * img_scale);
                int height = static_cast<int>(h * img_scale);

                boxes.emplace_back(x1, y1, width, height);
                confidences.push_back(confidence);
                classes.push_back(classId);
            }
        }
    }
    else if (dim1 == 7 && dim2 == 8400) // [1, 7, 8400] Yolov9
    {
        int64_t channels = dim1;
        int64_t cols = dim2;

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
                int x1 = static_cast<int>((x_center - width / 2) * img_scale);
                int y1 = static_cast<int>((y_center - height / 2) * img_scale);
                int w = static_cast<int>(width * img_scale);
                int h = static_cast<int>(height * img_scale);

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

    std::vector<int> indices(boxes.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](int i1, int i2) {
        return confidences[i1] > confidences[i2];
        });

    if (indices.size() > config.max_detections)
    {
        indices.resize(config.max_detections);
    }

    std::vector<cv::Rect> selectedBoxes;
    std::vector<float> selectedConfidences;
    std::vector<int> selectedClasses;

    for (int idx : indices)
    {
        selectedBoxes.push_back(boxes[idx]);
        selectedConfidences.push_back(confidences[idx]);
        selectedClasses.push_back(classes[idx]);
    }

    std::vector<int> nmsIndices;
    cv::dnn::NMSBoxes(selectedBoxes, selectedConfidences, config.confidence_threshold, config.nms_threshold, nmsIndices);

    std::vector<cv::Rect> finalBoxes;
    std::vector<int> finalClasses;

    for (int idx : nmsIndices)
    {
        finalBoxes.push_back(selectedBoxes[idx]);
        finalClasses.push_back(selectedClasses[idx]);
    }

    {
        std::lock_guard<std::mutex> lock(scaleMutex);
        for (auto& box : finalBoxes)
        {
            box.x = static_cast<int>(box.x * scaleX);
            box.y = static_cast<int>(box.y * scaleY);
            box.width = static_cast<int>(box.width * scaleX);
            box.height = static_cast<int>(box.height * scaleY);
        }
    }

    {
        std::lock_guard<std::mutex> lock(detectionMutex);
        detectedBoxes = std::move(finalBoxes);
        detectedClasses = std::move(finalClasses);
        detectionVersion++;
    }

    detectionCV.notify_one();
}