#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include "imgui/imgui.h"
#include "config.h"
#include "sunone_aimbot_cpp.h"

extern Config config;

void draw_game_overlay_settings()
{
    ImGui::Checkbox("Enable", &config.game_overlay_enabled);
    ImGui::SliderInt("Overlay Max FPS (0 = uncapped)", &config.game_overlay_max_fps, 0, 256);

    ImGui::Checkbox("Draw Detection Boxes", &config.game_overlay_draw_boxes);
    ImGui::Checkbox("Draw Future Positions", &config.game_overlay_draw_future);

    ImGui::Separator();
    ImGui::Text("Box Color (ARGB 0-255)");
    bool changed = false;
    changed |= ImGui::SliderInt("A##go_box_a", &config.game_overlay_box_a, 0, 255);
    changed |= ImGui::SliderInt("R##go_box_r", &config.game_overlay_box_r, 0, 255);
    changed |= ImGui::SliderInt("G##go_box_g", &config.game_overlay_box_g, 0, 255);
    changed |= ImGui::SliderInt("B##go_box_b", &config.game_overlay_box_b, 0, 255);
    ImGui::SliderFloat("Box Thickness", &config.game_overlay_box_thickness, 0.5f, 10.0f, "%.1f");

    ImGui::Separator();
    ImGui::Text("Future Point Style");
    ImGui::SliderFloat("Point Radius", &config.game_overlay_future_point_radius, 1.0f, 20.0f, "%.1f");
    ImGui::SliderFloat("Point Step Alpha Falloff", &config.game_overlay_future_alpha_falloff, 0.1f, 5.0f, "%.2f");

    if (changed)
    {
        config.clampGameOverlayColor();
    }

    ImGui::Separator();
    ImGui::Text("Icon Overlay");
    ImGui::Checkbox("Enable Icon Overlay", &config.game_overlay_icon_enabled);

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

    ImGui::InputText("Icon Path", iconPathBuf, IM_ARRAYSIZE(iconPathBuf));
    ImGui::SliderInt("Icon Width", &config.game_overlay_icon_width, 4, 512);
    ImGui::SliderInt("Icon Height", &config.game_overlay_icon_height, 4, 512);
    ImGui::SliderFloat("Icon Offset X", &config.game_overlay_icon_offset_x, -500.0f, 500.0f, "%.1f");
    ImGui::SliderFloat("Icon Offset Y", &config.game_overlay_icon_offset_y, -500.0f, 500.0f, "%.1f");

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
    }

    if (ImGui::Button("Save Game Overlay Config"))
    {
        config.game_overlay_icon_path = iconPathBuf;
        config.saveConfig("config.ini");
    }

}