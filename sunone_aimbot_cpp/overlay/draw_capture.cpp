#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <string.h>
#include <algorithm>

#include <imgui/imgui.h>
#include "imgui/imgui_internal.h"

#include "config.h"
#include "sunone_aimbot_cpp.h"
#include "capture.h"
#include "other_tools.h"
#include "virtual_camera.h"
#include "draw_settings.h"
#include "overlay.h"
#include "overlay/config_dirty.h"
#include "overlay/ui_sections.h"

bool disable_winrt_futures = checkwin1903();
int monitors = get_active_monitors();

static std::vector<std::string> virtual_cameras;
static char virtual_camera_filter_buf[128] = "";
static char udp_ip_buf[64] = "";
static int udp_port_buf = 1234;
static bool udp_settings_init = false;

void ensureVirtualCamerasLoaded()
{
    if (virtual_cameras.empty())
    {
        virtual_cameras = VirtualCameraCapture::GetAvailableVirtualCameras();
    }
}

static float CaptureCompactComboWidth()
{
    return OverlayUI::AdaptiveItemWidth(0.66f);
}

void draw_capture_settings()
{
    static const int allowed_resolutions[] = { 160, 320, 640 };
    static int current_resolution_idx = 1;

    for (int i = 0; i < 3; ++i)
        if (config.detection_resolution == allowed_resolutions[i])
            current_resolution_idx = i;

    if (OverlayUI::BeginSection("General Capture", "capture_section_general"))
    {
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Detection Resolution");
        ImGui::SameLine(0.0f, 8.0f);
        ImGui::SetNextItemWidth(CaptureCompactComboWidth());
        if (ImGui::Combo("##capture_detection_resolution", &current_resolution_idx, "160\0""320\0""640\0"))
        {
            config.detection_resolution = allowed_resolutions[current_resolution_idx];
            detection_resolution_changed.store(true);
            detector_model_changed.store(true);

            globalMouseThread->updateConfig(
                config.detection_resolution,
                config.fovX,
                config.fovY,
                config.minSpeedMultiplier,
                config.maxSpeedMultiplier,
                config.predictionInterval,
                config.auto_shoot,
                config.bScope_multiplier);
            OverlayConfig_MarkDirty();
        }

        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Capture FPS");
        ImGui::SameLine(0.0f, 8.0f);
        ImGui::SetNextItemWidth(CaptureCompactComboWidth());
        if (ImGui::SliderInt("##capture_fps_slider", &config.capture_fps, 0, 240))
        {
            capture_fps_changed.store(true);
            OverlayConfig_MarkDirty();
        }

        if (config.capture_fps == 0)
        {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(255, 0, 0, 255), "-> Disabled");
        }

        if (config.capture_fps == 0 || config.capture_fps >= 61)
        {
            ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + 260.0f);
            ImGui::TextColored(ImVec4(255, 255, 0, 255), "WARNING: High FPS can reduce performance.");
            ImGui::PopTextWrapPos();
        }

        if (ImGui::Checkbox("Circle mask", &config.circle_mask))
        {
            OverlayConfig_MarkDirty();
        }

        std::vector<std::string> captureMethodOptions = { "duplication_api", "winrt", "virtual_camera", "udp_capture" };
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

        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Capture method");
        ImGui::SameLine(0.0f, 8.0f);
        ImGui::SetNextItemWidth(CaptureCompactComboWidth());
        if (ImGui::Combo("##capture_method", &currentcaptureMethodIndex, captureMethodItems.data(), static_cast<int>(captureMethodItems.size())))
        {
            config.capture_method = captureMethodOptions[currentcaptureMethodIndex];
            OverlayConfig_MarkDirty();
            capture_method_changed.store(true);
        }

        OverlayUI::EndSection();
    }

    if (config.capture_method == "winrt")
    {
        if (OverlayUI::BeginSection("WinRT", "capture_section_winrt"))
        {
            {
                std::vector<std::string> targetOptions = { "monitor", "window" };
                int currentTargetIndex = (config.capture_target == "window") ? 1 : 0;
                ImGui::AlignTextToFramePadding();
                ImGui::TextUnformatted("Capture target (WinRT)");
                ImGui::SameLine(0.0f, 8.0f);
                ImGui::SetNextItemWidth(CaptureCompactComboWidth());
                if (ImGui::Combo("##capture_target_winrt", &currentTargetIndex,
                    [](void* data, int idx, const char** out_text) {
                        const auto* v = static_cast<std::vector<std::string>*>(data);
                        if (idx < 0 || idx >= (int)v->size()) return false;
                        *out_text = v->at(idx).c_str();
                        return true;
                    }, (void*)&targetOptions, (int)targetOptions.size()))
                {
                    config.capture_target = targetOptions[currentTargetIndex];
                    OverlayConfig_MarkDirty();
                    capture_method_changed.store(true);
                    capture_window_changed.store(true);
                }
            }

            if (config.capture_target == "window")
            {
                ImGui::Separator();
                ImGui::TextUnformatted("Window Target");

                static bool initTitle = false;
                static char titleBuf[256];
                if (!initTitle)
                {
                    memset(titleBuf, 0, sizeof(titleBuf));
                    std::string t = config.capture_window_title;
                    if (t.size() >= sizeof(titleBuf)) t = t.substr(0, sizeof(titleBuf) - 1);
                    memcpy(titleBuf, t.c_str(), t.size());
                    initTitle = true;
                }

                ImGui::InputText("Window title contains", titleBuf, IM_ARRAYSIZE(titleBuf));
                ImGui::SameLine();
                if (ImGui::Button("Use Active Window"))
                {
                    wchar_t wbuf[512]{};
                    HWND fg = ::GetForegroundWindow();
                    if (fg && ::GetWindowTextW(fg, wbuf, (int)std::size(wbuf)) > 0)
                    {
                        std::wstring ws(wbuf);
                        std::string s = WideToUtf8(ws);
                        memset(titleBuf, 0, sizeof(titleBuf));
                        auto copy = s.substr(0, sizeof(titleBuf) - 1);
                        memcpy(titleBuf, copy.c_str(), copy.size());
                    }
                }
                if (ImGui::Button("Apply Window Target"))
                {
                    config.capture_window_title = titleBuf;
                    OverlayConfig_MarkDirty();
                    capture_method_changed.store(true);
                    capture_window_changed.store(true);
                }
            }

            if (disable_winrt_futures)
            {
                ImGui::BeginDisabled();
            }

            if (ImGui::Checkbox("Capture Borders", &config.capture_borders))
            {
                capture_borders_changed.store(true);
                OverlayConfig_MarkDirty();
            }

            if (ImGui::Checkbox("Capture Cursor", &config.capture_cursor))
            {
                capture_cursor_changed.store(true);
                OverlayConfig_MarkDirty();
            }

            if (disable_winrt_futures)
            {
                ImGui::EndDisabled();
            }

            OverlayUI::EndSection();
        }
    }

    if (config.capture_method == "duplication_api" || (config.capture_method == "winrt" && config.capture_target != "window"))
    {
        if (OverlayUI::BeginSection("Monitor Capture", "capture_section_monitor"))
        {
            std::vector<std::string> monitorNames;
            int monitorCount = monitors;
            if (monitorCount <= 0)
            {
                monitorNames.push_back("Monitor 1");
                monitorCount = 1;
            }
            else
            {
                for (int i = 0; i < monitorCount; ++i)
                {
                    monitorNames.push_back("Monitor " + std::to_string(i + 1));
                }
            }

            std::vector<const char*> monitorItems;
            for (const auto& name : monitorNames)
            {
                monitorItems.push_back(name.c_str());
            }

            int selectedMonitor = std::clamp(config.monitor_idx, 0, monitorCount - 1);
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted("Capture monitor");
            ImGui::SameLine(0.0f, 8.0f);
            ImGui::SetNextItemWidth(CaptureCompactComboWidth());
            if (ImGui::Combo("##capture_monitor", &selectedMonitor, monitorItems.data(), static_cast<int>(monitorItems.size())))
            {
                config.monitor_idx = selectedMonitor;
                OverlayConfig_MarkDirty();
                capture_method_changed.store(true);
            }

            OverlayUI::EndSection();
        }
    }

    if (config.capture_method == "virtual_camera")
    {
        if (OverlayUI::BeginSection("Virtual Camera", "capture_section_virtual_camera"))
        {
            ensureVirtualCamerasLoaded();
            ImGui::Text("Select virtual camera:");

            ImGui::Text("Filter:");
            if (ImGui::InputText("##VCFilter", virtual_camera_filter_buf, IM_ARRAYSIZE(virtual_camera_filter_buf)))
            {
            }

            std::string filter_lower = virtual_camera_filter_buf;
            std::transform(filter_lower.begin(), filter_lower.end(), filter_lower.begin(), ::tolower);

            std::vector<int> filtered_indices;
            for (int i = 0; i < static_cast<int>(virtual_cameras.size()); ++i)
            {
                std::string name_lower = virtual_cameras[i];
                std::transform(name_lower.begin(), name_lower.end(), name_lower.begin(), ::tolower);
                if (filter_lower.empty() || name_lower.find(filter_lower) != std::string::npos)
                {
                    filtered_indices.push_back(i);
                }
            }

            if (!filtered_indices.empty())
            {
                int currentIndex = 0;
                for (int fi = 0; fi < static_cast<int>(filtered_indices.size()); ++fi)
                {
                    if (virtual_cameras[filtered_indices[fi]] == config.virtual_camera_name)
                    {
                        currentIndex = fi;
                        break;
                    }
                }

                std::vector<const char*> items;
                items.reserve(filtered_indices.size());
                for (int idx : filtered_indices)
                {
                    items.push_back(virtual_cameras[idx].c_str());
                }

                ImGui::SetNextItemWidth(CaptureCompactComboWidth());
                if (ImGui::Combo("##virtual_camera_combo", &currentIndex, items.data(), static_cast<int>(items.size())))
                {
                    config.virtual_camera_name = virtual_cameras[filtered_indices[currentIndex]];
                    OverlayConfig_MarkDirty();
                    capture_method_changed.store(true);
                }
            }
            else
            {
                ImGui::TextDisabled("No matching virtual cameras");
            }

            ImGui::SameLine();
            if (ImGui::Button("Refresh"))
            {
                VirtualCameraCapture::ClearCachedCameraList();
                virtual_cameras = VirtualCameraCapture::GetAvailableVirtualCameras(true);
                virtual_camera_filter_buf[0] = '\0';
            }

            if (ImGui::SliderInt("Virtual camera width", &config.virtual_camera_width, 128, 3840))
            {
                OverlayConfig_MarkDirty();
                capture_method_changed.store(true);
            }

            if (ImGui::SliderInt("Virtual camera heigth", &config.virtual_camera_heigth, 128, 2160))
            {
                OverlayConfig_MarkDirty();
                capture_method_changed.store(true);
            }

            OverlayUI::EndSection();
        }
    }

    if (config.capture_method == "udp_capture")
    {
        if (OverlayUI::BeginSection("UDP Capture", "capture_section_udp"))
        {
            if (!udp_settings_init)
            {
                memset(udp_ip_buf, 0, sizeof(udp_ip_buf));
                std::string ip = config.udp_ip;
                if (ip.size() >= sizeof(udp_ip_buf))
                    ip = ip.substr(0, sizeof(udp_ip_buf) - 1);
                memcpy(udp_ip_buf, ip.c_str(), ip.size());
                udp_port_buf = config.udp_port;
                udp_settings_init = true;
            }

            ImGui::InputText("UDP IP", udp_ip_buf, IM_ARRAYSIZE(udp_ip_buf));
            ImGui::InputInt("UDP Port", &udp_port_buf);
            if (ImGui::Button("Apply UDP Settings"))
            {
                udp_port_buf = std::clamp(udp_port_buf, 1, 65535);
                config.udp_ip = udp_ip_buf;
                config.udp_port = udp_port_buf;
                OverlayConfig_MarkDirty();
                capture_method_changed.store(true);
            }

            OverlayUI::EndSection();
        }
    }
}
