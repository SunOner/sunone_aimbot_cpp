#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <commdlg.h>
#include <string>
#include <cstring>

#include "imgui/imgui.h"
#include "config.h"
#include "sunone_aimbot_cpp.h"

extern std::string g_iconLastError;

void draw_game_overlay_settings()
{
    static bool cfgDirty = false;
    static double cfgDirtyAt = 0.0;
    constexpr double kSaveDelaySec = 0.35;

    auto markDirty = [&]()
        {
            cfgDirty = true;
            cfgDirtyAt = ImGui::GetTime();
        };

    if (ImGui::Checkbox("Enable", &config.game_overlay_enabled))
        markDirty();

    ImGui::SliderInt("Overlay Max FPS (0 = uncapped)", &config.game_overlay_max_fps, 0, 256);
    if (ImGui::IsItemDeactivatedAfterEdit())
        markDirty();

    if (ImGui::Checkbox("Draw Detection Boxes", &config.game_overlay_draw_boxes))
        markDirty();

    if (ImGui::Checkbox("Draw Future Positions", &config.game_overlay_draw_future))
        markDirty();

    if (ImGui::Checkbox("Show Target Correction", &config.game_overlay_show_target_correction))
        markDirty();

    ImGui::Separator();
    ImGui::Text("Box Color (ARGB 0-255)");

    bool colorChanged = false;

    ImGui::SliderInt("A##go_box_a", &config.game_overlay_box_a, 0, 255);
    colorChanged |= ImGui::IsItemEdited();
    if (ImGui::IsItemDeactivatedAfterEdit())
        markDirty();

    ImGui::SliderInt("R##go_box_r", &config.game_overlay_box_r, 0, 255);
    colorChanged |= ImGui::IsItemEdited();
    if (ImGui::IsItemDeactivatedAfterEdit())
        markDirty();

    ImGui::SliderInt("G##go_box_g", &config.game_overlay_box_g, 0, 255);
    colorChanged |= ImGui::IsItemEdited();
    if (ImGui::IsItemDeactivatedAfterEdit())
        markDirty();

    ImGui::SliderInt("B##go_box_b", &config.game_overlay_box_b, 0, 255);
    colorChanged |= ImGui::IsItemEdited();
    if (ImGui::IsItemDeactivatedAfterEdit())
        markDirty();

    ImGui::SliderFloat("Box Thickness", &config.game_overlay_box_thickness, 0.5f, 10.0f, "%.1f");
    if (ImGui::IsItemDeactivatedAfterEdit())
        markDirty();

    if (colorChanged)
        config.clampGameOverlayColor();

    ImGui::Separator();
    ImGui::Text("Capture Frame");

    if (ImGui::Checkbox("Draw Capture Frame", &config.game_overlay_draw_frame))
        markDirty();

    bool frameColorChanged = false;

    ImGui::SliderInt("A##go_frame_a", &config.game_overlay_frame_a, 0, 255);
    frameColorChanged |= ImGui::IsItemEdited();
    if (ImGui::IsItemDeactivatedAfterEdit())
        markDirty();

    ImGui::SliderInt("R##go_frame_r", &config.game_overlay_frame_r, 0, 255);
    frameColorChanged |= ImGui::IsItemEdited();
    if (ImGui::IsItemDeactivatedAfterEdit())
        markDirty();

    ImGui::SliderInt("G##go_frame_g", &config.game_overlay_frame_g, 0, 255);
    frameColorChanged |= ImGui::IsItemEdited();
    if (ImGui::IsItemDeactivatedAfterEdit())
        markDirty();

    ImGui::SliderInt("B##go_frame_b", &config.game_overlay_frame_b, 0, 255);
    frameColorChanged |= ImGui::IsItemEdited();
    if (ImGui::IsItemDeactivatedAfterEdit())
        markDirty();

    ImGui::SliderFloat("Frame Thickness", &config.game_overlay_frame_thickness, 0.5f, 10.0f, "%.1f");
    if (ImGui::IsItemDeactivatedAfterEdit())
        markDirty();

    if (frameColorChanged)
        config.clampGameOverlayColor();

    ImGui::Separator();
    ImGui::Text("Future Point Style");

    ImGui::SliderFloat("Point Radius", &config.game_overlay_future_point_radius, 1.0f, 20.0f, "%.1f");
    if (ImGui::IsItemDeactivatedAfterEdit())
        markDirty();

    ImGui::SliderFloat("Point Step Alpha Falloff", &config.game_overlay_future_alpha_falloff, 0.1f, 5.0f, "%.2f");
    if (ImGui::IsItemDeactivatedAfterEdit())
        markDirty();

    ImGui::Separator();
    ImGui::Text("Icon Overlay");

    if (ImGui::Checkbox("Enable Icon Overlay", &config.game_overlay_icon_enabled))
        markDirty();

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
        markDirty();
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
        ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;

        if (GetOpenFileNameA(&ofn))
        {
            strncpy_s(iconPathBuf, filePath, sizeof(iconPathBuf) - 1);
            config.game_overlay_icon_path = iconPathBuf;
            markDirty();
        }
    }

    ImGui::SliderInt("Icon Width", &config.game_overlay_icon_width, 4, 512);
    if (ImGui::IsItemDeactivatedAfterEdit())
        markDirty();

    ImGui::SliderInt("Icon Height", &config.game_overlay_icon_height, 4, 512);
    if (ImGui::IsItemDeactivatedAfterEdit())
        markDirty();

    ImGui::SliderFloat("Icon Offset X", &config.game_overlay_icon_offset_x, -500.0f, 500.0f, "%.1f");
    if (ImGui::IsItemDeactivatedAfterEdit())
        markDirty();

    ImGui::SliderFloat("Icon Offset Y", &config.game_overlay_icon_offset_y, -500.0f, 500.0f, "%.1f");
    if (ImGui::IsItemDeactivatedAfterEdit())
        markDirty();

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
        markDirty();
    }

    if (!g_iconLastError.empty())
    {
        ImGui::Separator();
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 100, 100, 255));
        ImGui::TextWrapped("%s", g_iconLastError.c_str());
        ImGui::PopStyleColor();
    }

    if (cfgDirty)
    {
        const double now = ImGui::GetTime();
        if ((now - cfgDirtyAt) >= kSaveDelaySec && !ImGui::IsAnyItemActive())
        {
            config.game_overlay_icon_path = iconPathBuf;

            config.saveConfig("config.ini");
            cfgDirty = false;
        }
    }
}
