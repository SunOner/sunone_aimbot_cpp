#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>
#include <iostream>
#include <fstream>

//#include <NvInferPlugin.h> // TODO: add batchedNMS
#include <NvOnnxParser.h>

#include "nvinf.h"

Logger gLogger;

void Logger::log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept
{
    if (severity <= nvinfer1::ILogger::Severity::kWARNING)
    {
        std::string devMsg = msg;

        std::string magicTag = "Serialization assertion plan->header.magicTag == rt::kPLAN_MAGIC_TAG failed.";
        std::string old_deserialization = "Using old deserialization call on a weight-separated plan file.";
        if (devMsg.find(magicTag) != std::string::npos ||
            devMsg.find(old_deserialization) != std::string::npos)
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
        std::cerr << "[nvinf] Error opening the engine file: " << engineFile << std::endl;
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
        std::cerr << "[nvinf] Engine deserialization error from file: " << engineFile << std::endl;
        return nullptr;
    }

    std::cout << "[nvinf] The engine was successfully loaded from the file: " << engineFile << std::endl;

    return engine;
}

nvinfer1::ICudaEngine* buildEngineFromOnnx(const std::string& onnxFile, nvinfer1::ILogger& logger)
{
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);

    if (!parser->parseFromFile(onnxFile.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        std::cerr << "[nvinf] ERROR: Error parsing the ONNX file: " << onnxFile << std::endl;
        
        delete parser;
        delete network;
        delete builder;
        delete config;
        
        return nullptr;
    }

    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if (!engine)
    {
        std::cerr << "[nvinf] ERROR: Could not build the engine" << std::endl;
        
        delete parser;
        delete network;
        delete builder;
        delete config;
        
        return nullptr;
    }

    nvinfer1::IHostMemory* serializedModel = engine->serialize();

    std::string engineFile = onnxFile.substr(0, onnxFile.find_last_of('.')) + ".engine";

    std::ofstream p(engineFile, std::ios::binary);
    if (!p)
    {
        std::cerr << "[nvinf] ERROR: Could not open file to write: " << engineFile << std::endl;
        
        delete serializedModel;
        delete engine;
        delete parser;
        delete network;
        delete builder;
        delete config;
        
        return nullptr;
    }

    p.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());

    delete serializedModel;
    delete parser;
    delete network;
    delete config;
    delete builder;

    std::cout << "[nvinf] The engine was built and saved to the file: " << engineFile << std::endl;

    return engine;
}