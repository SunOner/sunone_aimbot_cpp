#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include "imgui/imgui.h"

#include "sunone_aimbot_cpp.h"
#include "include/other_tools.h"
#include "overlay.h"
#include "overlay/config_dirty.h"
#include "draw_settings.h"
#ifdef USE_CUDA
#include "trt_monitor.h"
#endif

std::string prev_backend = config.backend;
float prev_confidence_threshold = config.confidence_threshold;
float prev_nms_threshold = config.nms_threshold;
int prev_max_detections = config.max_detections;

static bool wasExporting = false;
static bool ai_state_initialized = false;

void draw_ai()
{
    if (!ai_state_initialized)
    {
        prev_backend = config.backend;
        prev_confidence_threshold = config.confidence_threshold;
        prev_nms_threshold = config.nms_threshold;
        prev_max_detections = config.max_detections;
        ai_state_initialized = true;
    }

#ifdef USE_CUDA
    if (gIsTrtExporting)
    {
        ImGui::OpenPopup("TensorRT Export Progress");
    }

    if (ImGui::BeginPopupModal("TensorRT Export Progress", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        bool hasPhases = false;
        {
            std::lock_guard<std::mutex> lock(gProgressMutex);
            hasPhases = !gProgressPhases.empty();
            if (hasPhases)
            {
                for (auto& [name, phase] : gProgressPhases)
                {
                    float percent = phase.max > 0 ? phase.current / float(phase.max) : 0.0f;
                    ImGui::Text("%s: %d/%d", name.c_str(), phase.current, phase.max);
                    ImGui::ProgressBar(percent, ImVec2(300, 0));
                }
            }
        }
        if (!hasPhases)
        {
            ImGui::CloseCurrentPopup();
        }

        ImGui::Text("Engine export in progress, please wait...");
        long long lastUpdate = gTrtExportLastUpdateMs.load();
        if (lastUpdate > 0)
        {
            double secondsSince = (TrtNowMs() - lastUpdate) / 1000.0;
            ImGui::Text("Last progress update: %.1f s ago", secondsSince);
        }
        bool cancelRequested = gTrtExportCancelRequested.load();
        if (cancelRequested)
        {
            ImGui::BeginDisabled();
        }
        if (ImGui::Button("Cancel export"))
        {
            gTrtExportCancelRequested = true;
        }
        if (cancelRequested)
        {
            ImGui::EndDisabled();
            ImGui::Text("Cancel requested...");
        }
        ImGui::EndPopup();
    }
#endif
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
                OverlayConfig_MarkDirty();
                detector_model_changed.store(true);
            }
        }
        ImGui::SameLine();
        ImGui::Text("Fixed model size: %s", config.fixed_input_size ? "Enabled" : "Disabled");
    }

    ImGui::Separator();

#ifdef USE_CUDA
    std::vector<std::string> backendOptions = { "TRT", "DML" };
    std::vector<const char*> backendItems = { "TensorRT (CUDA)", "DirectML (CPU/GPU)" };

    int currentBackendIndex = config.backend == "DML" ? 1 : 0;

    if (ImGui::Combo("Backend", &currentBackendIndex, backendItems.data(), static_cast<int>(backendItems.size())))
    {
        std::string newBackend = backendOptions[currentBackendIndex];
        if (config.backend != newBackend)
        {
            config.backend = newBackend;
            OverlayConfig_MarkDirty();
            detector_model_changed.store(true);
        }
    }

    ImGui::Separator();
#endif

    ImGui::Separator();
    ImGui::SliderFloat("Confidence Threshold", &config.confidence_threshold, 0.01f, 1.00f, "%.2f");
    ImGui::SliderFloat("NMS Threshold", &config.nms_threshold, 0.00f, 1.00f, "%.2f");
    ImGui::SliderInt("Max Detections", &config.max_detections, 1, 100);
    ImGui::Separator();
    if (ImGui::CollapsingHeader("Depth"))
    {
        ImGui::Indent();
        draw_depth();
        ImGui::Unindent();
    }
        
    if (prev_confidence_threshold != config.confidence_threshold ||
        prev_nms_threshold != config.nms_threshold ||
        prev_max_detections != config.max_detections)
    {
        prev_nms_threshold = config.nms_threshold;
        prev_confidence_threshold = config.confidence_threshold;
        prev_max_detections = config.max_detections;
        OverlayConfig_MarkDirty();
    }

    if (prev_backend != config.backend)
    {
        prev_backend = config.backend;
        detector_model_changed.store(true);
        OverlayConfig_MarkDirty();
    }
}
