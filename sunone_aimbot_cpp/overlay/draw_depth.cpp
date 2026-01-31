#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <commdlg.h>
#include <chrono>
#include <string>


#include "imgui/imgui.h"
#include "sunone_aimbot_cpp.h"
#include "overlay.h"
#include "overlay/config_dirty.h"
#include "capture.h"
#include "draw_settings.h"

#ifdef USE_CUDA
#include "depth/depth_anything_trt.h"
#include "depth/depth_mask.h"
#include "tensorrt/nvinf.h"
#endif

#ifdef USE_CUDA
static depth_anything::DepthAnythingTrt g_depthModel;
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
#endif

void draw_depth()
{
#ifndef USE_CUDA
    ImGui::TextUnformatted("Depth requires a CUDA build.");
    return;
#else
    static bool bufferInit = false;
    static char modelPathBuf[260]{};
    static std::string depthStatus = "Depth model not loaded.";
    static int lastColormap = -1;

    if (!bufferInit)
    {
        memset(modelPathBuf, 0, sizeof(modelPathBuf));
        std::string initial = config.depth_model_path;
        if (initial.size() >= sizeof(modelPathBuf))
            initial = initial.substr(0, sizeof(modelPathBuf) - 1);
        memcpy(modelPathBuf, initial.c_str(), initial.size());
        bufferInit = true;
    }

    if (ImGui::Checkbox("Enable Depth Inference", &config.depth_inference_enabled))
    {
        OverlayConfig_MarkDirty();
        if (!config.depth_inference_enabled && g_depthModel.ready())
        {
            g_depthModel.reset();
            depthStatus = "Depth inference disabled.";
        }
    }

    if (ImGui::InputText("Depth model path", modelPathBuf, IM_ARRAYSIZE(modelPathBuf)))
    {
        if (config.depth_model_path != modelPathBuf)
        {
            config.depth_model_path = modelPathBuf;
            OverlayConfig_MarkDirty();
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Browse##depth_model_path"))
    {
        char filePath[MAX_PATH] = {};
        OPENFILENAMEA ofn = {};
        ofn.lStructSize = sizeof(ofn);
        ofn.hwndOwner = nullptr;
        ofn.lpstrFile = filePath;
        ofn.nMaxFile = sizeof(filePath);
        ofn.lpstrFilter = "Model Files\0*.engine;*.onnx;*.trt;*.plan\0All Files\0*.*\0";
        ofn.nFilterIndex = 1;
        ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;

        if (GetOpenFileNameA(&ofn))
        {
            strncpy_s(modelPathBuf, filePath, sizeof(modelPathBuf) - 1);
            if (config.depth_model_path != modelPathBuf)
            {
                config.depth_model_path = modelPathBuf;
                OverlayConfig_MarkDirty();
            }
        }
    }
    if (ImGui::Button("Load depth model"))
    {
        config.depth_model_path = modelPathBuf;
        OverlayConfig_MarkDirty();

        if (g_depthModel.initialize(config.depth_model_path, gLogger))
        {
            depthStatus = "Depth model loaded.";
            g_depthModel.setColormap(config.depth_colormap);
        }
        else
        {
            depthStatus = g_depthModel.lastError();
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Export depth engine"))
    {
        std::string exportPath = modelPathBuf;
        if (exportPath.empty())
        {
            depthStatus = "Set a depth ONNX path to export.";
        }
        else if (exportPath.find(".onnx") == std::string::npos)
        {
            depthStatus = "Export expects an .onnx depth model path.";
        }
        else
        {
            depth_anything::DepthAnythingTrt exporter;
            if (exporter.initialize(exportPath, gLogger))
            {
                depthStatus = "Depth engine exported next to the ONNX file.";
            }
            else
            {
                depthStatus = exporter.lastError();
            }
        }
    }

    if (ImGui::SliderInt("Depth FPS", &config.depth_fps, 0, 120))
    {
        OverlayConfig_MarkDirty();
    }

    if (ImGui::Checkbox("Enable Depth Mask", &config.depth_mask_enabled))
    {
        OverlayConfig_MarkDirty();
    }

    if (ImGui::SliderInt("Depth Mask FPS", &config.depth_mask_fps, 1, 30))
    {
        OverlayConfig_MarkDirty();
    }

    if (ImGui::SliderInt("Depth Mask Near %", &config.depth_mask_near_percent, 1, 100))
    {
        OverlayConfig_MarkDirty();
    }

    if (ImGui::SliderInt("Depth Mask Alpha", &config.depth_mask_alpha, 0, 255))
    {
        OverlayConfig_MarkDirty();
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
        g_depthModel.setColormap(config.depth_colormap);
    }

    ImGui::Text("Status: %s", depthStatus.c_str());
    if (config.depth_colormap != lastColormap)
    {
        g_depthModel.setColormap(config.depth_colormap);
        lastColormap = config.depth_colormap;
    }
    if (g_depthModel.ready())
        ImGui::TextUnformatted("Depth preview is shown in game overlay.");
#endif
}
