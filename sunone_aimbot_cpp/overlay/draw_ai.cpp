#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include "imgui/imgui.h"
#include "sunone_aimbot_cpp.h"
#include "include/other_tools.h"
#include "overlay.h"

void draw_ai()
{
    std::vector<std::string> availableModels = getAvailableModels();
    if (availableModels.empty())
    {
        ImGui::Text("No models available in the 'models' folder.");
    }
    else
    {
        int currentModelIndex = 0;
        auto it = std::find(availableModels.begin(), availableModels.end(), config.ai_model);

        if (it != availableModels.end())
        {
            currentModelIndex = static_cast<int>(std::distance(availableModels.begin(), it));
        }

        std::vector<const char*> modelsItems;
        modelsItems.reserve(availableModels.size());

        for (const auto& modelName : availableModels)
        {
            modelsItems.push_back(modelName.c_str());
        }

        if (ImGui::Combo("Model", &currentModelIndex, modelsItems.data(), static_cast<int>(modelsItems.size())))
        {
            if (config.ai_model != availableModels[currentModelIndex])
            {
                config.ai_model = availableModels[currentModelIndex];
                config.saveConfig();
                detector_model_changed.store(true);
            }
        }
    }

    ImGui::Separator();

    std::vector<std::string> backendOptions = { "TRT", "DML" };
    std::vector<const char*> backendItems = { "TensorRT (CUDA)", "DirectML (CPU/GPU)" };

    int currentBackendIndex = config.backend == "DML" ? 1 : 0;

    if (ImGui::Combo("Backend", &currentBackendIndex, backendItems.data(), static_cast<int>(backendItems.size())))
    {
        std::string newBackend = backendOptions[currentBackendIndex];
        if (config.backend != newBackend)
        {
            config.backend = newBackend;
            config.saveConfig();
            detector_model_changed.store(true);
        }
    }

    ImGui::Separator();

    std::vector<std::string> postprocessOptions = { "yolo8", "yolo9", "yolo10", "yolo11", "yolo12" };
    std::vector<const char*> postprocessItems;
    for (const auto& option : postprocessOptions)
    {
        postprocessItems.push_back(option.c_str());
    }

    int currentPostprocessIndex = 0;
    for (size_t i = 0; i < postprocessOptions.size(); ++i)
    {
        if (postprocessOptions[i] == config.postprocess)
        {
            currentPostprocessIndex = static_cast<int>(i);
            break;
        }
    }

    if (ImGui::Combo("Postprocess", &currentPostprocessIndex, postprocessItems.data(), static_cast<int>(postprocessItems.size())))
    {
        config.postprocess = postprocessOptions[currentPostprocessIndex];
        config.saveConfig();
        detector_model_changed.store(true);
    }

    ImGui::Separator();
    ImGui::SliderFloat("Confidence Threshold", &config.confidence_threshold, 0.01f, 1.00f, "%.2f");
    ImGui::SliderFloat("NMS Threshold", &config.nms_threshold, 0.01f, 1.00f, "%.2f");
    ImGui::SliderInt("Max Detections", &config.max_detections, 1, 100);
}