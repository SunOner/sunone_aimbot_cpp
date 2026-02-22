#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include "imgui/imgui.h"
#include "sunone_aimbot_cpp.h"
#include "overlay.h"
#include "overlay/config_dirty.h"
#include "overlay/ui_sections.h"

void draw_overlay()
{
    constexpr int kMinReadableOpacity = 220;

    if (OverlayUI::BeginSection("Visual", "overlay_section_visual"))
    {
        int prev_opacity = config.overlay_opacity;
        if (ImGui::SliderInt("Overlay Opacity", &config.overlay_opacity, kMinReadableOpacity, 255))
        {
            if (config.overlay_opacity < kMinReadableOpacity) config.overlay_opacity = kMinReadableOpacity;
            if (config.overlay_opacity > 255) config.overlay_opacity = 255;

            Overlay_SetOpacity(config.overlay_opacity);

            if (config.overlay_opacity != prev_opacity)
                OverlayConfig_MarkDirty();
        }

        static float ui_scale = config.overlay_ui_scale;

        if (ImGui::SliderFloat("UI Scale", &ui_scale, 0.5f, 3.0f, "%.2f"))
        {
            ImGui::GetIO().FontGlobalScale = ui_scale;

            config.overlay_ui_scale = ui_scale;
            OverlayConfig_MarkDirty();

            extern const int BASE_OVERLAY_WIDTH;
            extern const int BASE_OVERLAY_HEIGHT;
            overlayWidth = static_cast<int>(BASE_OVERLAY_WIDTH * ui_scale);
            overlayHeight = static_cast<int>(BASE_OVERLAY_HEIGHT * ui_scale);

            SetWindowPos(g_hwnd, NULL, 0, 0, overlayWidth, overlayHeight, SWP_NOMOVE | SWP_NOZORDER);
        }

        OverlayUI::EndSection();
    }

    if (OverlayUI::BeginSection("Capture Privacy", "overlay_section_capture_privacy"))
    {
        if (ImGui::Checkbox("Hide Overlays From Recording", &config.overlay_exclude_from_capture))
        {
            Overlay_ApplyCaptureExclusion();
            OverlayConfig_MarkDirty();
        }
        OverlayUI::EndSection();
    }
}
