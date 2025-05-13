#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include "imgui/imgui.h"
#include "sunone_aimbot_cpp.h"
#include "overlay.h"
#include "include/other_tools.h"
#include "visuals.h"

bool prev_show_window = config.show_window;
bool prev_show_fps = config.show_fps;
int prev_window_size = config.window_size;
int prev_screenshot_delay = config.screenshot_delay;
bool prev_always_on_top = config.always_on_top;
bool prev_verbose = config.verbose;

void draw_debug()
{
    ImGui::Checkbox("Show Debug Window", &config.show_window);
    if (config.show_window)
    {
        ImGui::Checkbox("Show FPS", &config.show_fps);
        if (ImGui::Checkbox("Always on Top", &config.always_on_top))
        {
            config.saveConfig();
            setWindowAlwaysOnTop(config.window_name, config.always_on_top);
        }
        ImGui::SliderInt("Debug Window Size", &config.window_size, 10, 350);
        ImGui::Separator();
    }

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

    if (prev_show_window != config.show_window ||
        prev_always_on_top != config.always_on_top)
    {
        prev_always_on_top = config.always_on_top;
        show_window_changed.store(true);
        prev_show_window = config.show_window;
        config.saveConfig();
    }

    if (prev_show_fps != config.show_fps ||
        prev_window_size != config.window_size ||
        prev_screenshot_delay != config.screenshot_delay ||
        prev_verbose != config.verbose)
    {
        prev_show_fps = config.show_fps;
        prev_window_size = config.window_size;
        prev_screenshot_delay = config.screenshot_delay;
        prev_verbose = config.verbose;
        config.saveConfig();
    }
}