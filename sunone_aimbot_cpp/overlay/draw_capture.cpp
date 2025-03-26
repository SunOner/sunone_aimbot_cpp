#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <imgui/imgui.h>
#include "imgui/imgui_internal.h"

#include "config.h"
#include "sunone_aimbot_cpp.h"
#include "capture.h"
#include "include/other_tools.h"
#include "capture/virtual_camera.h"
#include "draw_settings.h"

bool disable_winrt_futures = checkwin1903();
int monitors = get_active_monitors();
std::vector<std::string> virtual_cameras = { };

void draw_capture_settings()
{
    ImGui::SliderInt("Detection Resolution", &config.detection_resolution, 50, 1280);
    if (config.detection_resolution >= 400)
    {
        ImGui::TextColored(ImVec4(255, 255, 0, 255), "WARNING: A large screen capture size can negatively affect performance.");
    }

    ImGui::SliderInt("Lock FPS", &config.capture_fps, 0, 240);
    if (config.capture_fps == 0)
    {
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(255, 0, 0, 255), "-> Disabled");
    }

    if (config.capture_fps == 0 || config.capture_fps >= 61)
    {
        ImGui::TextColored(ImVec4(255, 255, 0, 255), "WARNING: A large number of FPS can negatively affect performance.");
    }

    if (ImGui::Checkbox("Circle mask", &config.circle_mask))
    {
        capture_method_changed.store(true);
        config.saveConfig();
    }

    if (ImGui::Checkbox("Use Cuda in capture", &config.capture_use_cuda))
    {
        capture_method_changed.store(true);
        config.saveConfig();
    }

    std::vector<std::string> captureMethodOptions = { "duplication_api", "winrt", "virtual_camera" };
    std::vector<const char*> captureMethodItems;
    for (const auto& option : captureMethodOptions)
    {
        captureMethodItems.push_back(option.c_str());
    }

    int currentcaptureMethodIndex = 0;
    for (size_t i = 0; i < captureMethodOptions.size(); ++i)
    {
        if (captureMethodOptions[i] == config.capture_method)
        {
            currentcaptureMethodIndex = static_cast<int>(i);
            break;
        }
    }

    if (ImGui::Combo("Capture method", &currentcaptureMethodIndex, captureMethodItems.data(), static_cast<int>(captureMethodItems.size()))) {
        config.capture_method = captureMethodOptions[currentcaptureMethodIndex];
        config.saveConfig();
        capture_method_changed.store(true);
    }

    if (config.capture_method == "winrt")
    {
        if (disable_winrt_futures)
        {
            ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
            ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
        }

        ImGui::Checkbox("Capture Borders", &config.capture_borders);
        ImGui::Checkbox("Capture Cursor", &config.capture_cursor);

        if (disable_winrt_futures)
        {
            ImGui::PopStyleVar();
            ImGui::PopItemFlag();
        }
    }

    if (config.capture_method == "duplication_api" || config.capture_method == "winrt")
    {
        std::vector<std::string> monitorNames;
        if (monitors == -1)
        {
            monitorNames.push_back("Monitor 1");
        }
        else
        {
            for (int i = -1; i < monitors; ++i)
            {
                monitorNames.push_back("Monitor " + std::to_string(i + 1));
            }
        }

        std::vector<const char*> monitorItems;
        for (const auto& name : monitorNames)
        {
            monitorItems.push_back(name.c_str());
        }

        if (ImGui::Combo("Capture monitor (CUDA GPU)", &config.monitor_idx, monitorItems.data(), static_cast<int>(monitorItems.size())))
        {
            config.saveConfig();
            capture_method_changed.store(true);
        }
    }

    if (config.capture_method == "virtual_camera")
    {
        if (!virtual_cameras.empty())
        {
            int currentCameraIndex = 0;
            for (size_t i = 0; i < virtual_cameras.size(); i++)
            {
                if (virtual_cameras[i] == config.virtual_camera_name)
                {
                    currentCameraIndex = i;
                    break;
                }
            }

            std::vector<const char*> cameraItems;
            for (const auto& cam : virtual_cameras)
            {
                cameraItems.push_back(cam.c_str());
            }

            if (ImGui::Combo("Virtual Camera", &currentCameraIndex,
                cameraItems.data(), static_cast<int>(cameraItems.size())))
            {
                config.virtual_camera_name = virtual_cameras[currentCameraIndex];
                config.saveConfig();
                capture_method_changed.store(true);
            }
            ImGui::SameLine();
            if (ImGui::Button("Update##update_virtual_cameras"))
            {
                virtual_cameras = VirtualCameraCapture::GetAvailableVirtualCameras();
            }
        }
        else
        {
            ImGui::Text("No virtual cameras found");
            ImGui::SameLine();
            if (ImGui::Button("Update##update_virtual_cameras"))
            {
                virtual_cameras = VirtualCameraCapture::GetAvailableVirtualCameras();
            }
        }
    }
}