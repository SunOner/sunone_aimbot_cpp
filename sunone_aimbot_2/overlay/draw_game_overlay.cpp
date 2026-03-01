#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <commdlg.h>
#include <string>
#include <cstring>
#include <algorithm>

#include "imgui/imgui.h"
#include "config.h"
#include "sunone_aimbot_2.h"
#include "capture.h"
#include "overlay/config_dirty.h"
#include "overlay/ui_sections.h"

extern std::string g_iconLastError;

void draw_game_overlay_settings()
{
    if (OverlayUI::BeginSection("General", "game_overlay_section_general"))
    {
        if (ImGui::Checkbox("Enable", &config.game_overlay_enabled))
            OverlayConfig_MarkDirty();

        ImGui::SliderInt("Overlay Max FPS (0 = uncapped)", &config.game_overlay_max_fps, 0, 256);
        if (ImGui::IsItemDeactivatedAfterEdit())
            OverlayConfig_MarkDirty();

        if (ImGui::Checkbox("Draw Detection Boxes", &config.game_overlay_draw_boxes))
            OverlayConfig_MarkDirty();

        if (ImGui::Checkbox("Draw Future Positions", &config.game_overlay_draw_future))
            OverlayConfig_MarkDirty();

        if (ImGui::Checkbox("Draw Wind Debug Tail", &config.game_overlay_draw_wind_tail))
            OverlayConfig_MarkDirty();

        if (ImGui::Checkbox("Show Target Correction", &config.game_overlay_show_target_correction))
            OverlayConfig_MarkDirty();

        OverlayUI::EndSection();
    }

    if (OverlayUI::BeginSection("Box Color", "game_overlay_section_box_color"))
    {
        bool colorChanged = false;

        ImGui::SliderInt("A##go_box_a", &config.game_overlay_box_a, 0, 255);
        colorChanged |= ImGui::IsItemEdited();
        if (ImGui::IsItemDeactivatedAfterEdit())
            OverlayConfig_MarkDirty();

        ImGui::SliderInt("R##go_box_r", &config.game_overlay_box_r, 0, 255);
        colorChanged |= ImGui::IsItemEdited();
        if (ImGui::IsItemDeactivatedAfterEdit())
            OverlayConfig_MarkDirty();

        ImGui::SliderInt("G##go_box_g", &config.game_overlay_box_g, 0, 255);
        colorChanged |= ImGui::IsItemEdited();
        if (ImGui::IsItemDeactivatedAfterEdit())
            OverlayConfig_MarkDirty();

        ImGui::SliderInt("B##go_box_b", &config.game_overlay_box_b, 0, 255);
        colorChanged |= ImGui::IsItemEdited();
        if (ImGui::IsItemDeactivatedAfterEdit())
            OverlayConfig_MarkDirty();

        ImGui::SliderFloat("Box Thickness", &config.game_overlay_box_thickness, 0.5f, 10.0f, "%.1f");
        if (ImGui::IsItemDeactivatedAfterEdit())
            OverlayConfig_MarkDirty();

        if (colorChanged)
            config.clampGameOverlayColor();

        OverlayUI::EndSection();
    }

    if (OverlayUI::BeginSection("Capture Frame", "game_overlay_section_capture_frame"))
    {
        if (ImGui::Checkbox("Draw Capture Frame", &config.game_overlay_draw_frame))
            OverlayConfig_MarkDirty();

        bool frameColorChanged = false;

        ImGui::SliderInt("A##go_frame_a", &config.game_overlay_frame_a, 0, 255);
        frameColorChanged |= ImGui::IsItemEdited();
        if (ImGui::IsItemDeactivatedAfterEdit())
            OverlayConfig_MarkDirty();

        ImGui::SliderInt("R##go_frame_r", &config.game_overlay_frame_r, 0, 255);
        frameColorChanged |= ImGui::IsItemEdited();
        if (ImGui::IsItemDeactivatedAfterEdit())
            OverlayConfig_MarkDirty();

        ImGui::SliderInt("G##go_frame_g", &config.game_overlay_frame_g, 0, 255);
        frameColorChanged |= ImGui::IsItemEdited();
        if (ImGui::IsItemDeactivatedAfterEdit())
            OverlayConfig_MarkDirty();

        ImGui::SliderInt("B##go_frame_b", &config.game_overlay_frame_b, 0, 255);
        frameColorChanged |= ImGui::IsItemEdited();
        if (ImGui::IsItemDeactivatedAfterEdit())
            OverlayConfig_MarkDirty();

        ImGui::SliderFloat("Frame Thickness", &config.game_overlay_frame_thickness, 0.5f, 10.0f, "%.1f");
        if (ImGui::IsItemDeactivatedAfterEdit())
            OverlayConfig_MarkDirty();

        if (frameColorChanged)
            config.clampGameOverlayColor();

        OverlayUI::EndSection();
    }

    if (OverlayUI::BeginSection("Future Point Style", "game_overlay_section_future_style"))
    {
        ImGui::SliderFloat("Point Radius", &config.game_overlay_future_point_radius, 1.0f, 20.0f, "%.1f");
        if (ImGui::IsItemDeactivatedAfterEdit())
            OverlayConfig_MarkDirty();

        ImGui::SliderFloat("Point Step Alpha Falloff", &config.game_overlay_future_alpha_falloff, 0.1f, 5.0f, "%.2f");
        if (ImGui::IsItemDeactivatedAfterEdit())
            OverlayConfig_MarkDirty();

        OverlayUI::EndSection();
    }

    if (OverlayUI::BeginSection("Icon Overlay", "game_overlay_section_icon"))
    {
        if (ImGui::Checkbox("Enable Icon Overlay", &config.game_overlay_icon_enabled))
            OverlayConfig_MarkDirty();

        if (!config.game_overlay_icon_enabled)
        {
            ImGui::BeginDisabled();
        }

        static bool pathInit = false;
        static char iconPathBuf[512];

        if (!pathInit)
        {
            pathInit = true;
            memset(iconPathBuf, 0, sizeof(iconPathBuf));
            std::string p = config.game_overlay_icon_path;
            if (p.size() >= sizeof(iconPathBuf)) p = p.substr(0, sizeof(iconPathBuf) - 1);
            memcpy(iconPathBuf, p.c_str(), p.size());
        }

        if (ImGui::InputText("Icon Path", iconPathBuf, IM_ARRAYSIZE(iconPathBuf)))
        {
            config.game_overlay_icon_path = iconPathBuf;
            OverlayConfig_MarkDirty();
        }

        ImGui::SameLine();
        if (ImGui::Button("Browse##icon_path"))
        {
            char filePath[MAX_PATH] = {};
            OPENFILENAMEA ofn = {};
            ofn.lStructSize = sizeof(ofn);
            ofn.hwndOwner = nullptr;
            ofn.lpstrFile = filePath;
            ofn.nMaxFile = sizeof(filePath);
            ofn.lpstrFilter = "Image Files\0*.png;*.jpg;*.jpeg;*.bmp;*.ico\0All Files\0*.*\0";
            ofn.nFilterIndex = 1;
            ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;

            if (GetOpenFileNameA(&ofn))
            {
                strncpy_s(iconPathBuf, filePath, sizeof(iconPathBuf) - 1);
                config.game_overlay_icon_path = iconPathBuf;
                OverlayConfig_MarkDirty();
            }
        }

        ImGui::SliderInt("Icon Width", &config.game_overlay_icon_width, 4, 512);
        if (ImGui::IsItemDeactivatedAfterEdit())
            OverlayConfig_MarkDirty();

        ImGui::SliderInt("Icon Height", &config.game_overlay_icon_height, 4, 512);
        if (ImGui::IsItemDeactivatedAfterEdit())
            OverlayConfig_MarkDirty();

        ImGui::SliderFloat("Icon Offset X", &config.game_overlay_icon_offset_x, -500.0f, 500.0f, "%.1f");
        if (ImGui::IsItemDeactivatedAfterEdit())
            OverlayConfig_MarkDirty();

        ImGui::SliderFloat("Icon Offset Y", &config.game_overlay_icon_offset_y, -500.0f, 500.0f, "%.1f");
        if (ImGui::IsItemDeactivatedAfterEdit())
            OverlayConfig_MarkDirty();

        if (ImGui::InputInt("Icon Class (-1 = all)", &config.game_overlay_icon_class))
        {
            if (config.game_overlay_icon_class < -1) config.game_overlay_icon_class = -1;
            OverlayConfig_MarkDirty();
        }

        const char* anchors[] = { "center", "top", "bottom", "head" };
        int currentAnchor = 0;
        for (int i = 0; i < (int)(sizeof(anchors) / sizeof(anchors[0])); ++i)
        {
            if (config.game_overlay_icon_anchor == anchors[i])
            {
                currentAnchor = i;
                break;
            }
        }

        if (ImGui::Combo("Icon Anchor", &currentAnchor, anchors, IM_ARRAYSIZE(anchors)))
        {
            config.game_overlay_icon_anchor = anchors[currentAnchor];
            OverlayConfig_MarkDirty();
        }

        if (!config.game_overlay_icon_enabled)
        {
            ImGui::EndDisabled();
            ImGui::TextDisabled("Enable Icon Overlay to edit settings.");
        }

        OverlayUI::EndSection();
    }

    if (OverlayUI::BeginSection("Aim Simulation", "game_overlay_section_aim_sim"))
    {
        if (ImGui::Checkbox("Enable Aim Simulation Window", &config.aim_sim_enabled))
            OverlayConfig_MarkDirty();

        if (!config.aim_sim_enabled)
        {
            ImGui::BeginDisabled();
        }

        ImGui::SliderInt("Sim X", &config.aim_sim_x, -3000, 3000);
        if (ImGui::IsItemDeactivatedAfterEdit())
            OverlayConfig_MarkDirty();

        ImGui::SliderInt("Sim Y", &config.aim_sim_y, -3000, 3000);
        if (ImGui::IsItemDeactivatedAfterEdit())
            OverlayConfig_MarkDirty();

        ImGui::SliderInt("Sim Width", &config.aim_sim_width, 220, 1600);
        if (ImGui::IsItemDeactivatedAfterEdit())
            OverlayConfig_MarkDirty();

        ImGui::SliderInt("Sim Height", &config.aim_sim_height, 180, 1000);
        if (ImGui::IsItemDeactivatedAfterEdit())
            OverlayConfig_MarkDirty();

        if (ImGui::SliderInt("Sim FPS Min", &config.aim_sim_fps_min, 15, 360))
        {
            if (config.aim_sim_fps_min > config.aim_sim_fps_max)
                config.aim_sim_fps_max = config.aim_sim_fps_min;
            OverlayConfig_MarkDirty();
        }

        if (ImGui::SliderInt("Sim FPS Max", &config.aim_sim_fps_max, 15, 360))
        {
            if (config.aim_sim_fps_max < config.aim_sim_fps_min)
                config.aim_sim_fps_min = config.aim_sim_fps_max;
            OverlayConfig_MarkDirty();
        }

        if (ImGui::SliderFloat("FPS Jitter", &config.aim_sim_fps_jitter, 0.0f, 0.8f, "%.3f"))
            OverlayConfig_MarkDirty();

        if (ImGui::SliderFloat("Capture Delay (ms)", &config.aim_sim_capture_delay_ms, 0.0f, 80.0f, "%.1f"))
            OverlayConfig_MarkDirty();

        static bool delayedSnapshotPending = false;
        static double delayedSnapshotApplyAt = 0.0;
        const auto apply_snapshot_metrics = []()
        {
            float current_preprocess = 0.0f;
            float current_inference = 0.0f;
            float current_copy = 0.0f;
            float current_post = 0.0f;
            float current_nms = 0.0f;
            bool hasTimingMetrics = false;

            if (config.backend == "DML" && dml_detector)
            {
                current_preprocess = static_cast<float>(dml_detector->lastPreprocessTimeDML.count());
                current_inference = static_cast<float>(dml_detector->lastInferenceTimeDML.count());
                current_copy = static_cast<float>(dml_detector->lastCopyTimeDML.count());
                current_post = static_cast<float>(dml_detector->lastPostprocessTimeDML.count());
                current_nms = static_cast<float>(dml_detector->lastNmsTimeDML.count());
                hasTimingMetrics = true;
            }
#ifdef USE_CUDA
            else
            {
                current_preprocess = static_cast<float>(trt_detector.lastPreprocessTime.count());
                current_inference = static_cast<float>(trt_detector.lastInferenceTime.count());
                current_copy = static_cast<float>(trt_detector.lastCopyTime.count());
                current_post = static_cast<float>(trt_detector.lastPostprocessTime.count());
                current_nms = static_cast<float>(trt_detector.lastNmsTime.count());
                hasTimingMetrics = true;
            }
#endif

            const auto clampf = [](float v, float lo, float hi) -> float
            {
                if (v < lo) return lo;
                if (v > hi) return hi;
                return v;
            };

            const float fpsNow = static_cast<float>(captureFps.load());
            if (fpsNow > 1.0f)
            {
                const float captureDelayMs = 1000.0f / fpsNow;
                config.aim_sim_capture_delay_ms = clampf(captureDelayMs, 0.0f, 80.0f);
            }

            if (hasTimingMetrics && current_inference > 0.0f)
                config.aim_sim_inference_delay_ms = clampf(current_inference, 0.0f, 120.0f);

            if (hasTimingMetrics)
            {
                const float extraDelayMs = current_preprocess + current_copy + current_post + current_nms;
                config.aim_sim_extra_delay_ms = clampf(extraDelayMs, 0.0f, 60.0f);
            }

            OverlayConfig_MarkDirty();
        };

        if (delayedSnapshotPending && ImGui::GetTime() >= delayedSnapshotApplyAt)
        {
            apply_snapshot_metrics();
            delayedSnapshotPending = false;
        }

        if (ImGui::Checkbox("Use Live Inference Delay", &config.aim_sim_use_live_inference))
            OverlayConfig_MarkDirty();
        ImGui::SameLine();
        if (ImGui::Button("Snapshot Metrics"))
            apply_snapshot_metrics();
        if (ImGui::IsItemHovered())
        {
            ImGui::SetTooltip(
                "Capture Delay <- 1000/FPS\n"
                "Inference Delay <- current backend inference\n"
                "Extra Delay <- preprocess + copy + postprocess + NMS"
            );
        }
        ImGui::SameLine();
        if (ImGui::Button("Snapshot in 4s"))
        {
            delayedSnapshotPending = true;
            delayedSnapshotApplyAt = ImGui::GetTime() + 4.0;
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Start 4-second timer, then snapshot metrics automatically.");

        if (delayedSnapshotPending)
        {
            const double remaining = std::max(0.0, delayedSnapshotApplyAt - ImGui::GetTime());
            ImGui::SameLine();
            ImGui::TextDisabled("Auto in %.1fs", static_cast<float>(remaining));
        }

        if (ImGui::SliderFloat("Inference Delay (ms)", &config.aim_sim_inference_delay_ms, 0.0f, 120.0f, "%.1f"))
            OverlayConfig_MarkDirty();

        if (ImGui::SliderFloat("Input Delay (ms)", &config.aim_sim_input_delay_ms, 0.0f, 60.0f, "%.1f"))
            OverlayConfig_MarkDirty();

        if (ImGui::SliderFloat("Extra Delay (ms)", &config.aim_sim_extra_delay_ms, 0.0f, 60.0f, "%.1f"))
            OverlayConfig_MarkDirty();

        if (ImGui::SliderFloat("Target Max Speed", &config.aim_sim_target_max_speed, 20.0f, 2500.0f, "%.0f"))
            OverlayConfig_MarkDirty();

        if (ImGui::SliderFloat("Target Accel", &config.aim_sim_target_accel, 20.0f, 10000.0f, "%.0f"))
            OverlayConfig_MarkDirty();

        if (ImGui::SliderFloat("Target Stop Chance", &config.aim_sim_target_stop_chance, 0.0f, 0.95f, "%.2f"))
            OverlayConfig_MarkDirty();

        if (ImGui::Checkbox("Show Delayed Observation", &config.aim_sim_show_observed))
            OverlayConfig_MarkDirty();

        if (ImGui::Checkbox("Show Trajectory History", &config.aim_sim_show_history))
            OverlayConfig_MarkDirty();

        if (!config.aim_sim_enabled)
        {
            ImGui::EndDisabled();
            ImGui::TextDisabled("Enable Aim Simulation Window to edit settings.");
        }

        OverlayUI::EndSection();
    }

    if (!g_iconLastError.empty())
    {
        if (OverlayUI::BeginSection("Errors", "game_overlay_section_errors"))
        {
            ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 100, 100, 255));
            ImGui::TextWrapped("%s", g_iconLastError.c_str());
            ImGui::PopStyleColor();
            OverlayUI::EndSection();
        }
    }

}
