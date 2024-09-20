#include "nvinf.h"
#include <iostream>

void Logger::log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept
{
    if (severity <= nvinfer1::ILogger::Severity::kVERBOSE)
    {
        std::cout << "[TensorRT] " << severityLevelName(severity) << ": " << msg << std::endl;
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

Logger gLogger;

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