#ifdef USE_CUDA

#include "depth_anything_trt.h"

#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <filesystem>
#include <fstream>

namespace depth_anything
{
    namespace
    {
        constexpr bool kEnableFp16 = true;
        constexpr int kMinInputSize = 160;
        constexpr int kMaxInputSize = 640;
        constexpr int kOptInputSize = 518;
    }

    DepthAnythingTrt::DepthAnythingTrt()
        : input_w(518)
        , input_h(518)
        , min_input_size(kMinInputSize)
        , max_input_size(kMaxInputSize)
        , dynamic_input(false)
        , mean{ 123.675f, 116.28f, 103.53f }
        , std{ 58.395f, 57.12f, 57.375f }
        , runtime(nullptr)
        , engine(nullptr)
        , context(nullptr)
        , buffer{ nullptr, nullptr }
        , stream(nullptr)
        , initialized(false)
    {
    }

    DepthAnythingTrt::~DepthAnythingTrt()
    {
        reset();
    }

    bool DepthAnythingTrt::ready() const
    {
        return initialized;
    }

    const std::string& DepthAnythingTrt::lastError() const
    {
        return last_error;
    }

    void DepthAnythingTrt::reset()
    {
        initialized = false;
        last_error.clear();

        if (stream)
        {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }

        if (buffer[0])
        {
            cudaFree(buffer[0]);
            buffer[0] = nullptr;
        }
        if (buffer[1])
        {
            cudaFree(buffer[1]);
            buffer[1] = nullptr;
        }

        depth_data.clear();
        context.reset();
        engine.reset();
        runtime.reset();
    }

    bool DepthAnythingTrt::initialize(const std::string& modelPath, nvinfer1::ILogger& logger)
    {
        reset();
        dynamic_input = false;
        min_input_size = kMinInputSize;
        max_input_size = kMaxInputSize;

        if (!std::filesystem::exists(modelPath))
        {
            last_error = "Depth model file not found: " + modelPath;
            return false;
        }

        if (!loadEngine(modelPath, logger))
        {
            if (last_error.empty())
            {
                last_error = "Failed to load depth model: " + modelPath;
            }
            return false;
        }

        auto input_name = engine->getIOTensorName(0);
        auto input_dims = engine->getTensorShape(input_name);
        bool has_dynamic = false;
        for (int i = 0; i < input_dims.nbDims; i++)
        {
            if (input_dims.d[i] == -1)
            {
                has_dynamic = true;
                break;
            }
        }

        if (has_dynamic)
        {
            dynamic_input = true;
            min_input_size = kMinInputSize;
            max_input_size = kMaxInputSize;
            input_h = max_input_size;
            input_w = max_input_size;
        }
        else
        {
            input_h = input_dims.d[2];
            input_w = input_dims.d[3];
            min_input_size = input_w;
            max_input_size = input_w;
        }

        cudaStreamCreate(&stream);

        const size_t max_input = static_cast<size_t>(input_h) * static_cast<size_t>(input_w);
        cudaMalloc(&buffer[0], 3 * max_input * sizeof(float));
        cudaMalloc(&buffer[1], max_input * sizeof(float));

        depth_data.resize(max_input);

        initialized = true;
        return true;
    }

    std::vector<float> DepthAnythingTrt::preprocess(const cv::Mat& image)
    {
        cv::Mat input_image = image;
        auto resized = resize_depth(input_image, input_w, input_h);
        cv::Mat resized_image = std::get<0>(resized);

        std::vector<float> input_tensor;
        input_tensor.reserve(static_cast<size_t>(3 * input_h * input_w));
        for (int k = 0; k < 3; k++)
        {
            for (int i = 0; i < resized_image.rows; i++)
            {
                for (int j = 0; j < resized_image.cols; j++)
                {
                    input_tensor.emplace_back((static_cast<float>(resized_image.at<cv::Vec3b>(i, j)[k]) - mean[k]) / std[k]);
                }
            }
        }
        return input_tensor;
    }

    cv::Mat DepthAnythingTrt::predict(const cv::Mat& image)
    {
        if (!initialized || image.empty())
        {
            return {};
        }

        cv::Mat clone_image;
        image.copyTo(clone_image);

        int target_size = selectInputSize(clone_image);
        if (dynamic_input)
        {
            if (!setInputShape(target_size, target_size))
            {
                return {};
            }
        }

        std::vector<float> input = preprocess(clone_image);
        const size_t input_bytes = static_cast<size_t>(3) * static_cast<size_t>(input_h) * static_cast<size_t>(input_w) * sizeof(float);
        cudaMemcpyAsync(buffer[0], input.data(), input_bytes, cudaMemcpyHostToDevice, stream);

        context->executeV2(buffer);

        const size_t output_bytes = static_cast<size_t>(input_h) * static_cast<size_t>(input_w) * sizeof(float);
        cudaMemcpyAsync(depth_data.data(), buffer[1], output_bytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        cv::Mat depth_mat(input_h, input_w, CV_32FC1, depth_data.data());
        cv::Mat depth_norm;
        cv::normalize(depth_mat, depth_norm, 0, 255, cv::NORM_MINMAX, CV_8U);

        cv::Mat colormap;
        cv::applyColorMap(depth_norm, colormap, cv::COLORMAP_INFERNO);

        cv::Mat output;
        cv::resize(colormap, output, image.size());
        return output;
    }

    int DepthAnythingTrt::selectInputSize(const cv::Mat& image) const
    {
        if (min_input_size <= 0 || max_input_size <= 0)
        {
            return input_w;
        }

        int long_side = std::max(image.cols, image.rows);
        return std::clamp(long_side, min_input_size, max_input_size);
    }

    bool DepthAnythingTrt::setInputShape(int w, int h)
    {
        const char* input_name = engine->getIOTensorName(0);
        if (!context->setInputShape(input_name, nvinfer1::Dims4{ 1, 3, h, w }))
        {
            last_error = "Failed to set depth input shape.";
            return false;
        }

        input_w = w;
        input_h = h;
        return true;
    }

    bool DepthAnythingTrt::loadEngine(const std::string& modelPath, nvinfer1::ILogger& logger)
    {
        if (modelPath.find(".onnx") != std::string::npos)
        {
            if (!buildEngine(modelPath, logger))
            {
                return false;
            }
            saveEngine(modelPath);
            return true;
        }

        std::ifstream engineStream(modelPath, std::ios::binary);
        if (!engineStream.is_open())
        {
            last_error = "Unable to open depth engine: " + modelPath;
            return false;
        }

        engineStream.seekg(0, std::ios::end);
        const size_t modelSize = engineStream.tellg();
        engineStream.seekg(0, std::ios::beg);
        std::vector<char> engineData(modelSize);
        engineStream.read(engineData.data(), modelSize);
        engineStream.close();

        runtime.reset(nvinfer1::createInferRuntime(logger));
        engine.reset(runtime->deserializeCudaEngine(engineData.data(), modelSize));
        if (!engine)
        {
            last_error = "Failed to deserialize depth engine: " + modelPath;
            return false;
        }
        context.reset(engine->createExecutionContext());
        if (!context)
        {
            last_error = "Failed to create depth execution context.";
            return false;
        }
        return true;
    }

    bool DepthAnythingTrt::buildEngine(const std::string& onnxPath, nvinfer1::ILogger& logger)
    {
        auto builder = nvinfer1::createInferBuilder(logger);
        if (!builder)
        {
            last_error = "Failed to create TensorRT builder.";
            return false;
        }

        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
        nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

        if (kEnableFp16)
        {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }

        nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
        if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO)))
        {
            last_error = "Failed to parse depth ONNX model.";
            delete parser;
            delete config;
            delete network;
            delete builder;
            return false;
        }

        auto input = network->getInput(0);
        if (input)
        {
            auto input_dims = input->getDimensions();
            bool has_dynamic = false;
            for (int i = 0; i < input_dims.nbDims; i++)
            {
                if (input_dims.d[i] == -1)
                {
                    has_dynamic = true;
                    break;
                }
            }

            if (has_dynamic)
            {
                auto profile = builder->createOptimizationProfile();
                int opt_size = std::clamp(kOptInputSize, kMinInputSize, kMaxInputSize);
                const char* input_name = input->getName();
                bool ok = profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{ 1, 3, kMinInputSize, kMinInputSize });
                ok = ok && profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{ 1, 3, opt_size, opt_size });
                ok = ok && profile->setDimensions(input_name, nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{ 1, 3, kMaxInputSize, kMaxInputSize });
                if (!ok || !profile->isValid())
                {
                    last_error = "Failed to set depth input optimization profile.";
                    delete parser;
                    delete config;
                    delete network;
                    delete builder;
                    return false;
                }
                config->addOptimizationProfile(profile);
            }
        }

        nvinfer1::IHostMemory* plan = builder->buildSerializedNetwork(*network, *config);
        runtime.reset(nvinfer1::createInferRuntime(logger));
        engine.reset(runtime->deserializeCudaEngine(plan->data(), plan->size()));
        context.reset(engine->createExecutionContext());

        delete plan;
        delete parser;
        delete config;
        delete network;
        delete builder;

        if (!engine || !context)
        {
            last_error = "Failed to build depth engine from ONNX.";
            return false;
        }

        return true;
    }

    bool DepthAnythingTrt::saveEngine(const std::string& onnxPath)
    {
        if (!engine)
        {
            return false;
        }

        size_t dotIndex = onnxPath.find_last_of(".");
        if (dotIndex == std::string::npos)
        {
            return false;
        }

        std::string engine_path = onnxPath.substr(0, dotIndex) + ".engine";
        nvinfer1::IHostMemory* data = engine->serialize();
        std::ofstream file(engine_path, std::ios::binary | std::ios::out);
        if (!file.is_open())
        {
            delete data;
            return false;
        }

        file.write(reinterpret_cast<const char*>(data->data()), data->size());
        file.close();
        delete data;
        return true;
    }
}

#endif