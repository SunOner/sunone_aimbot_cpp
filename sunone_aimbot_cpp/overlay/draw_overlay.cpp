#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include "imgui/imgui.h"
#include "sunone_aimbot_cpp.h"

void draw_overlay()
{
    ImGui::SliderInt("Overlay Opacity", &config.overlay_opacity, 40, 255);

    if (ImGui::Checkbox("Enable snow theme", &config.overlay_snow_theme))
    {
        config.saveConfig();
    }

    static float ui_scale = config.overlay_ui_scale;

    if (ImGui::SliderFloat("UI Scale", &ui_scale, 0.5f, 3.0f, "%.2f"))
    {
        ImGui::GetIO().FontGlobalScale = ui_scale;

        config.overlay_ui_scale = ui_scale;
        config.saveConfig("config.ini");
    }
}