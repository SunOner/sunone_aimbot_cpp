#ifdef USE_CUDA
#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include "nvinf.h"
#include "sunone_aimbot_cpp.h"
#include "trt_monitor.h"

Logger gLogger;

void Logger::log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept
{
    if (severity <= nvinfer1::ILogger::Severity::kWARNING)
    {
        std::string devMsg = msg;

        std::string magicTag = "Serialization assertion plan->header.magicTag == rt::kPLAN_MAGIC_TAG failed.";
        std::string old_deserialization = "Using old deserialization call on a weight-separated plan file.";
        if (devMsg.find(magicTag) != std::string::npos || devMsg.find(old_deserialization) != std::string::npos)
        {
            std::cout << "[TensorRT] ERROR: This engine model is not suitable for execution. Please delete this engine model and set the ONNX version of this model in the settings. The program will export the model automatically." << std::endl;
        }
        else
        {
            std::cout << "[TensorRT] " << severityLevelName(severity) << ": " << msg << std::endl;
        }
    }
}

const char* Logger::severityLevelName(nvinfer1::ILogger::Severity severity)
{
    switch (severity)
    {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "INTERNAL_ERROR";
        case nvinfer1::ILogger::Severity::kERROR:          return "ERROR";
        case nvinfer1::ILogger::Severity::kWARNING:        return "WARNING";
        case nvinfer1::ILogger::Severity::kINFO:           return "INFO";
        case nvinfer1::ILogger::Severity::kVERBOSE:        return "VERBOSE";
        default:                                           return "UNKNOWN";
    }
}

nvinfer1::IBuilder* createInferBuilder()
{
    return nvinfer1::createInferBuilder(gLogger);
}

nvinfer1::INetworkDefinition* createNetwork(nvinfer1::IBuilder* builder)
{
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    return builder->createNetworkV2(explicitBatch);
}

nvinfer1::IBuilderConfig* createBuilderConfig(nvinfer1::IBuilder* builder)
{
    return builder->createBuilderConfig();
}

nvinfer1::ICudaEngine* loadEngineFromFile(const std::string& engineFile, nvinfer1::IRuntime* runtime)
{
    std::ifstream file(engineFile, std::ios::binary);
    if (!file.good())
    {
        std::cerr << "[TensorRT] Error opening the engine file: " << engineFile << std::endl;
        return nullptr;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), size);
    if (!engine)
    {
        std::cerr << "[TensorRT] Engine deserialization error from file: " << engineFile << std::endl;
        return nullptr;
    }

    if (config.verbose)
    {
        std::cout << "[TensorRT] The engine was successfully loaded from the file: " << engineFile << std::endl;
    }
    return engine;
}

nvinfer1::ICudaEngine* buildEngineFromOnnx(const std::string& onnxFile, nvinfer1::ILogger& logger)
{
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);

    nvinfer1::IBuilderConfig* cfg = builder->createBuilderConfig();

    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);

    ImGuiProgressMonitor progressMonitor;
    cfg->setProgressMonitor(&progressMonitor);
    gIsTrtExporting = true;

    if (!parser->parseFromFile(onnxFile.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        std::cerr << "[TensorRT] ERROR: Error parsing the ONNX file: " << onnxFile << std::endl;
        delete parser;
        delete network;
        delete builder;
        delete cfg;
        return nullptr;
    }

    nvinfer1::ITensor* inputTensor = network->getInput(0);
    const char* inName = inputTensor->getName();
    nvinfer1::Dims inDims = inputTensor->getDimensions();
    int H = (inDims.nbDims >= 4) ? inDims.d[2] : -1;
    int W = (inDims.nbDims >= 4) ? inDims.d[3] : -1;

    bool fixedByModel = (H > 0 && W > 0);
    bool fixedByConfig = config.fixed_input_size;
    bool makeStatic = fixedByModel || fixedByConfig;

    if (fixedByConfig && (H <= 0 || W <= 0))
        H = W = config.detection_resolution;

    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    if (makeStatic)
    {
        nvinfer1::Dims4 d{ 1, 3, H, W };
        profile->setDimensions(inName, nvinfer1::OptProfileSelector::kMIN, d);
        profile->setDimensions(inName, nvinfer1::OptProfileSelector::kOPT, d);
        profile->setDimensions(inName, nvinfer1::OptProfileSelector::kMAX, d);
        if (config.verbose)
            std::cout << "[TensorRT] Static profile " << H << "x" << W << std::endl;
    }
    else
    {
        profile->setDimensions(inName, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{ 1, 3, 160, 160 });
        profile->setDimensions(inName, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{ 1, 3, 320, 320 });
        profile->setDimensions(inName, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{ 1, 3, 640, 640 });
        if (config.verbose)
            std::cout << "[TensorRT] Dynamic profile 160/320/640" << std::endl;
    }

    cfg->addOptimizationProfile(profile);


    if (config.export_enable_fp16)
    {
        if (config.verbose)
            std::cout << "[TensorRT] Set FP16" << std::endl;
        cfg->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    if (config.export_enable_fp8)
    {
        if (config.verbose)
            std::cout << "[TensorRT] Set FP8" << std::endl;
        cfg->setFlag(nvinfer1::BuilderFlag::kFP8);
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::cout << "[TensorRT] Building engine (this may take several minutes)..." << std::endl;

    auto plan = builder->buildSerializedNetwork(*network, *cfg);
    if (!plan)
    {
        std::cerr << "[TensorRT] ERROR: Could not build the engine" << std::endl;
        delete parser;
        delete network;
        delete builder;
        delete cfg;
        return nullptr;
    }

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(plan->data(), plan->size());

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    if (!engine)
    {
        std::cerr << "[TensorRT] ERROR: Could not create engine" << std::endl;
        delete plan;
        delete runtime;
        delete parser;
        delete network;
        delete builder;
        delete cfg;
        return nullptr;
    }

    nvinfer1::IHostMemory* serializedModel = engine->serialize();
    std::string engineFile = onnxFile.substr(0, onnxFile.find_last_of('.')) + ".engine";
    std::ofstream p(engineFile, std::ios::binary);
    if (!p)
    {
        std::cerr << "[TensorRT] ERROR: Could not open file to write: " << engineFile << std::endl;
        delete serializedModel;
        delete engine;
        delete parser;
        delete network;
        delete builder;
        delete cfg;
        return nullptr;
    }
    p.write(static_cast<const char*>(plan->data()), plan->size());
    p.close();

    delete plan;
    delete runtime;
    delete parser;
    delete network;
    delete cfg;
    delete builder;

    std::cout << "[TensorRT] The engine was built and saved to the file: " << engineFile << std::endl;
    return engine;
}
#endif