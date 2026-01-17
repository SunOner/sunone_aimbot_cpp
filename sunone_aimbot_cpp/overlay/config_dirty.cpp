#include "overlay/config_dirty.h"

#include "config.h"
#include "imgui/imgui.h"

extern Config config;

namespace
{
    bool cfgDirty = false;
    double cfgDirtyAt = 0.0;
    constexpr double kSaveDelaySec = 0.35;
}

void OverlayConfig_MarkDirty()
{
    cfgDirty = true;
    cfgDirtyAt = ImGui::GetTime();
}

void OverlayConfig_TrySave(const char* filename)
{
    if (!cfgDirty)
        return;

    const double now = ImGui::GetTime();
    if ((now - cfgDirtyAt) < kSaveDelaySec)
        return;

    if (ImGui::IsAnyItemActive())
        return;

    config.saveConfig(filename ? filename : "config.ini");
    cfgDirty = false;
}
