#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <string>
#include <vector>

#include "imgui/imgui.h"
#include "sunone_aimbot_2.h"
#include "overlay.h"
#include "overlay/config_dirty.h"
#include "overlay/ui_sections.h"

namespace
{
int findKeyIndexByName(const std::string& keyName)
{
    for (size_t k = 0; k < key_names.size(); ++k)
    {
        if (key_names[k] == keyName)
            return static_cast<int>(k);
    }
    return 0;
}

bool drawButtonBindingRows(const char* comboPrefix, std::vector<std::string>& bindings, bool keepAtLeastOne)
{
    if (key_names_cstrs.empty())
    {
        ImGui::TextDisabled("No key list available.");
        return false;
    }

    bool changed = false;
    if (bindings.empty() && keepAtLeastOne)
    {
        bindings.push_back("None");
        changed = true;
    }

    for (size_t i = 0; i < bindings.size();)
    {
        std::string& currentKeyName = bindings[i];
        int currentIndex = findKeyIndexByName(currentKeyName);

        ImGui::PushID(static_cast<int>(i));

        const float rowAvail = ImGui::GetContentRegionAvail().x;
        const float actionBtnW = ImGui::GetFrameHeight();
        float comboWidth = rowAvail - (actionBtnW * 2.0f + ImGui::GetStyle().ItemSpacing.x * 2.0f);
        const float comboMin = rowAvail * 0.56f;
        if (comboWidth < comboMin)
            comboWidth = comboMin;
        if (comboWidth < 1.0f)
            comboWidth = 1.0f;
        ImGui::SetNextItemWidth(comboWidth);

        if (ImGui::Combo("##binding_combo", &currentIndex, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
        {
            currentKeyName = key_names[currentIndex];
            changed = true;
        }

        ImGui::SameLine(0.0f, 4.0f);
        if (ImGui::Button("+", ImVec2(actionBtnW, 0.0f)))
        {
            bindings.insert(bindings.begin() + static_cast<std::vector<std::string>::difference_type>(i + 1), "None");
            changed = true;
        }

        ImGui::SameLine(0.0f, 3.0f);
        bool removedCurrent = false;
        if (ImGui::Button("-", ImVec2(actionBtnW, 0.0f)))
        {
            if (bindings.size() <= 1 && keepAtLeastOne)
            {
                bindings[0] = "None";
            }
            else
            {
                bindings.erase(bindings.begin() + static_cast<std::vector<std::string>::difference_type>(i));
                removedCurrent = true;
            }
            changed = true;
        }

        ImGui::PopID();

        if (removedCurrent)
            continue;

        ++i;
    }

    return changed;
}

void drawBindingSection(const char* title, const char* sectionId, const char* comboPrefix, std::vector<std::string>& bindings, bool keepAtLeastOne = true)
{
    if (!OverlayUI::BeginSection(title, sectionId))
        return;

    if (drawButtonBindingRows(comboPrefix, bindings, keepAtLeastOne))
        OverlayConfig_MarkDirty();

    OverlayUI::EndSection();
}
}

void draw_buttons()
{
    drawBindingSection("Targeting Buttons", "buttons_section_targeting", "Targeting Button", config.button_targeting);
    drawBindingSection("Shoot Buttons", "buttons_section_shoot", "Shoot Button", config.button_shoot);
    drawBindingSection("Zoom Buttons", "buttons_section_zoom", "Zoom Button", config.button_zoom);
    drawBindingSection("Exit Buttons", "buttons_section_exit", "Exit Button", config.button_exit);
    drawBindingSection("Pause Buttons", "buttons_section_pause", "Pause Button", config.button_pause);
    drawBindingSection("Reload Config Buttons", "buttons_section_reload", "Reload config Button", config.button_reload_config);
    drawBindingSection("Overlay Buttons", "buttons_section_overlay", "Overlay Button", config.button_open_overlay);

    if (OverlayUI::BeginSection("Arrow Key Options", "buttons_section_arrows"))
    {
        if (ImGui::Checkbox("Enable arrows keys options", &config.enable_arrows_settings))
        {
            OverlayConfig_MarkDirty();
        }
        OverlayUI::EndSection();
    }
}
