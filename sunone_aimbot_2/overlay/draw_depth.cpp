#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <commdlg.h>
#include <chrono>
#include <filesystem>
#include <string>
#include <algorithm>
#include <cctype>
#include <thread>
#include <atomic>
#include <mutex>

#include "imgui/imgui.h"
#include "sunone_aimbot_2.h"
#include "overlay.h"
#include "overlay/config_dirty.h"
#include "capture.h"
#include "draw_settings.h"
#include "include/other_tools.h"
#include "overlay/ui_sections.h"

#ifdef USE_CUDA
#include "depth/depth_anything_trt.h"
#include "depth/depth_mask.h"
#include "tensorrt/nvinf.h"
#include "tensorrt/trt_monitor.h"
#endif

#ifdef USE_CUDA
static const char* kDepthColormapNames[] = {
    "Autumn",
    "Bone",
    "Jet",
    "Winter",
    "Rainbow",
    "Ocean",
    "Summer",
    "Spring",
    "Cool",
    "HSV",
    "Pink",
    "Hot",
    "Parula",
    "Magma",
    "Inferno",
    "Plasma",
    "Viridis",
    "Cividis",
    "Twilight",
    "Twilight Shifted",
    "Turbo",
    "Deepgreen"
};

namespace
{
    bool HasExtensionCaseInsensitive(const std::string& path, const char* ext)
    {
        if (!ext || !*ext)
        {
            return false;
        }

        std::filesystem::path p(path);
        std::string current = p.extension().string();
        std::string expected = ext;
        std::transform(current.begin(), current.end(), current.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        std::transform(expected.begin(), expected.end(), expected.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return current == expected;
    }
}
#endif

void draw_depth()
{
#ifndef USE_CUDA
    if (OverlayUI::BeginSection("Depth", "depth_section_unavailable"))
    {
        ImGui::TextUnformatted("Depth requires a CUDA build.");
        OverlayUI::EndSection();
    }
    return;
#else
    static std::string depthStatus = "Depth runtime is managed automatically.";
    static std::atomic<bool> depthExportRunning{ false };
    static std::thread depthExportThread;
    static std::mutex depthExportMutex;
    static std::string depthExportResult;

    if (depthExportThread.joinable() && !depthExportRunning.load())
    {
        depthExportThread.join();
    }
    {
        std::lock_guard<std::mutex> lock(depthExportMutex);
        if (!depthExportResult.empty())
        {
            depthStatus = depthExportResult;
            depthExportResult.clear();
        }
    }

    std::vector<std::string> availableDepthModels = getAvailableDepthModels();
    std::string selectedModel;
    bool hasModels = !availableDepthModels.empty();

    if (OverlayUI::BeginSection("Depth Inference", "depth_section_inference"))
    {
        if (ImGui::Checkbox("Enable Depth Inference", &config.depth_inference_enabled))
        {
            OverlayConfig_MarkDirty();
            if (!config.depth_inference_enabled)
                depthStatus = "Depth inference disabled.";
        }

        if (!hasModels)
        {
            ImGui::Text("No depth models available in 'models/depth'.");
        }
        else
        {
            int currentModelIndex = 0;
            auto it = std::find(availableDepthModels.begin(), availableDepthModels.end(), config.depth_model_path);
            if (it == availableDepthModels.end())
            {
                std::string configFile = std::filesystem::path(config.depth_model_path).filename().string();
                it = std::find(availableDepthModels.begin(), availableDepthModels.end(), configFile);
            }
            if (it != availableDepthModels.end())
            {
                currentModelIndex = static_cast<int>(std::distance(availableDepthModels.begin(), it));
            }

            std::vector<const char*> modelItems;
            modelItems.reserve(availableDepthModels.size());
            for (const auto& modelName : availableDepthModels)
            {
                modelItems.push_back(modelName.c_str());
            }

            if (ImGui::Combo("Depth model", &currentModelIndex, modelItems.data(), static_cast<int>(modelItems.size())))
            {
                if (config.depth_model_path != availableDepthModels[currentModelIndex])
                {
                    config.depth_model_path = availableDepthModels[currentModelIndex];
                    OverlayConfig_MarkDirty();
                }
            }

            selectedModel = availableDepthModels[currentModelIndex];
        }

        const bool selectedIsOnnx = hasModels && HasExtensionCaseInsensitive(selectedModel, ".onnx");
        const bool exportBusy = depthExportRunning.load();
        if (!hasModels || selectedIsOnnx || exportBusy)
        {
            ImGui::BeginDisabled();
        }
        if (ImGui::Button("Load depth model"))
        {
            if (config.depth_model_path != selectedModel)
            {
                config.depth_model_path = selectedModel;
                OverlayConfig_MarkDirty();
                depthStatus = "Depth model path applied. Runtime loader will update automatically.";
            }
            else
            {
                depthStatus = "Depth model path already selected.";
            }
        }
        if (!hasModels || selectedIsOnnx || exportBusy)
        {
            ImGui::EndDisabled();
        }

        ImGui::SameLine();

        if (!hasModels || !selectedIsOnnx || exportBusy)
        {
            ImGui::BeginDisabled();
        }
        if (ImGui::Button("Export depth engine"))
        {
            if (!depthExportRunning.load())
            {
                if (config.depth_model_path != selectedModel)
                {
                    config.depth_model_path = selectedModel;
                    OverlayConfig_MarkDirty();
                }

                std::string exportPath = selectedModel;
                if (exportPath.empty())
                {
                    depthStatus = "Set a depth ONNX path to export.";
                }
                else if (!HasExtensionCaseInsensitive(exportPath, ".onnx"))
                {
                    depthStatus = "Export expects an .onnx depth model path.";
                }
                else
                {
                    depthExportRunning.store(true);
                    depthExportThread = std::thread([exportPath] {
                        depth_anything::DepthAnythingTrt exporter;
                        std::string result;
                        if (exporter.initialize(exportPath, gLogger))
                        {
                            result = "Depth engine exported next to the ONNX file.";
                        }
                        else
                        {
                            if (gTrtExportCancelRequested.load())
                            {
                                result = "Depth export canceled.";
                            }
                            else
                            {
                                result = exporter.lastError();
                            }
                        }
                        {
                            std::lock_guard<std::mutex> lock(depthExportMutex);
                            depthExportResult = result;
                        }
                        depthExportRunning.store(false);
                    });
                }
            }
        }
        if (!hasModels || !selectedIsOnnx || exportBusy)
        {
            ImGui::EndDisabled();
        }

        OverlayUI::EndSection();
    }

    if (OverlayUI::BeginSection("Depth Runtime", "depth_section_runtime"))
    {
        if (ImGui::SliderInt("Depth FPS", &config.depth_fps, 0, 120))
        {
            OverlayConfig_MarkDirty();
        }

        if (ImGui::SliderInt("Depth Mask FPS", &config.depth_mask_fps, 1, 30))
        {
            OverlayConfig_MarkDirty();
        }
        OverlayUI::EndSection();
    }

    if (OverlayUI::BeginSection("Depth Mask", "depth_section_mask"))
    {
        if (ImGui::Checkbox("Enable Depth Mask", &config.depth_mask_enabled))
        {
            OverlayConfig_MarkDirty();
        }

        if (ImGui::SliderInt("Depth Mask Near %", &config.depth_mask_near_percent, 1, 100))
        {
            OverlayConfig_MarkDirty();
        }

        if (ImGui::SliderInt("Depth Mask Expand (px)", &config.depth_mask_expand, 0, 128))
        {
            OverlayConfig_MarkDirty();
        }

        if (ImGui::SliderInt("Depth Mask Hold Frames", &config.depth_mask_hold_frames, 0, 120))
        {
            OverlayConfig_MarkDirty();
        }

        if (ImGui::SliderInt("Depth Mask Alpha", &config.depth_mask_alpha, 0, 255))
        {
            OverlayConfig_MarkDirty();
        }
        if (config.depth_mask_enabled && config.depth_mask_alpha == 0)
        {
            ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "Depth mask is invisible: alpha is 0.");
        }

        if (ImGui::Checkbox("Depth Mask Invert", &config.depth_mask_invert))
        {
            OverlayConfig_MarkDirty();
        }

        if (ImGui::Checkbox("Depth Debug Overlay (Game)", &config.depth_debug_overlay_enabled))
        {
            OverlayConfig_MarkDirty();
        }

        int colormapIndex = config.depth_colormap;
        if (ImGui::Combo("Depth colormap", &colormapIndex, kDepthColormapNames, IM_ARRAYSIZE(kDepthColormapNames)))
        {
            config.depth_colormap = colormapIndex;
            OverlayConfig_MarkDirty();
        }

        OverlayUI::EndSection();
    }

    if (OverlayUI::BeginSection("Depth Status", "depth_section_status"))
    {
        ImGui::Text("Status: %s", depthStatus.c_str());

        if (config.depth_inference_enabled && config.depth_mask_enabled)
        {
            auto& depthMask = depth_anything::GetDepthMaskGenerator();
            const auto state = depthMask.debugState();
            const auto lastErr = depthMask.lastError();
            const auto frameSize = depthMask.lastFrameSize();

            ImGui::Separator();
            ImGui::Text("Mask runtime: %s", state.model_ready ? "ready" : "not ready");
            ImGui::Text("Mask model path: %s",
                state.last_model_path.empty() ? "(none)" : state.last_model_path.c_str());
            if (frameSize.first > 0 && frameSize.second > 0)
                ImGui::Text("Last mask frame: %dx%d", frameSize.first, frameSize.second);

            if (!lastErr.empty())
                ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "Mask error: %s", lastErr.c_str());
        }
        else if (config.depth_inference_enabled)
        {
            ImGui::Separator();
            ImGui::TextUnformatted("Depth mask is disabled.");
        }
        else
        {
            ImGui::Separator();
            ImGui::TextUnformatted("Depth inference is disabled.");
        }

        ImGui::TextUnformatted("Depth preview appears in game overlay when debug overlay is enabled.");
        OverlayUI::EndSection();
    }
#endif
}
