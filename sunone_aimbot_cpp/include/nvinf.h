#ifndef NVINF_H
#define NVINF_H

#include "NvInfer.h"

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override;
    static const char* severityLevelName(Severity severity);
};

extern Logger gLogger;

inline nvinfer1::IBuilder* createInferBuilder();
inline nvinfer1::INetworkDefinition* createNetwork(nvinfer1::IBuilder* builder);
inline nvinfer1::IBuilderConfig* createBuilderConfig(nvinfer1::IBuilder* builder);

#endif // NVINF_H