#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include "imgui/imgui.h"
#include "sunone_aimbot_cpp.h"
#include "overlay.h"
#include "include/other_tools.h"

void draw_debug()
{
    ImGui::Checkbox("Show Window", &config.show_window);
    ImGui::Checkbox("Show FPS", &config.show_fps);
    ImGui::SliderInt("Window Size", &config.window_size, 10, 350);

    ImGui::Separator();
    ImGui::Text("Screenshot Buttons");

    for (size_t i = 0; i < config.screenshot_button.size(); )
    {
        std::string& current_key_name = config.screenshot_button[i];

        int current_index = -1;
        for (size_t k = 0; k < key_names.size(); ++k)
        {
            if (key_names[k] == current_key_name)
            {
                current_index = static_cast<int>(k);
                break;
            }
        }

        if (current_index == -1)
        {
            current_index = 0;
        }

        std::string combo_label = "Screenshot Button " + std::to_string(i);

        if (ImGui::Combo(combo_label.c_str(), &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
        {
            current_key_name = key_names[current_index];
            config.saveConfig("config.ini");
        }

        ImGui::SameLine();
        std::string remove_button_label = "Remove##button_screenshot" + std::to_string(i);
        if (ImGui::Button(remove_button_label.c_str()))
        {
            if (config.screenshot_button.size() <= 1)
            {
                config.screenshot_button[0] = std::string("None");
                config.saveConfig();
                continue;
            }
            else
            {
                config.screenshot_button.erase(config.screenshot_button.begin() + i);
                config.saveConfig();
                continue;
            }
        }

        ++i;
    }

    if (ImGui::Button("Add button##button_screenshot"))
    {
        config.screenshot_button.push_back("None");
        config.saveConfig();
    }

    ImGui::InputInt("Screenshot delay", &config.screenshot_delay, 50, 500);
    ImGui::Checkbox("Always On Top", &config.always_on_top);
    ImGui::Checkbox("Verbose console output", &config.verbose);

    ImGui::Separator();

    ImGui::Text("Test functions");
    if (ImGui::Button("Free terminal"))
    {
        HideConsole();
    }
    ImGui::SameLine();
    if (ImGui::Button("Restore terminal"))
    {
        ShowConsole();
    }
}