#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include "imgui/imgui.h"
#include "sunone_aimbot_cpp.h"
#include "overlay.h"

void draw_buttons()
{
    ImGui::Text("Targeting Buttons");

    for (size_t i = 0; i < config.button_targeting.size(); )
    {
        std::string& current_key_name = config.button_targeting[i];

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

        std::string combo_label = "Targeting Button " + std::to_string(i);

        if (ImGui::Combo(combo_label.c_str(), &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
        {
            current_key_name = key_names[current_index];
            config.saveConfig();
        }

        ImGui::SameLine();
        std::string remove_button_label = "Remove##button_targeting" + std::to_string(i);
        if (ImGui::Button(remove_button_label.c_str()))
        {
            if (config.button_targeting.size() <= 1)
            {
                config.button_targeting[0] = std::string("None");
                config.saveConfig();
                continue;
            }
            else
            {
                config.button_targeting.erase(config.button_targeting.begin() + i);
                config.saveConfig();
                continue;
            }
        }

        ++i;
    }

    if (ImGui::Button("Add button##targeting"))
    {
        config.button_targeting.push_back("None");
        config.saveConfig();
    }

    ImGui::Separator();

    ImGui::Text("Shoot Buttons");

    for (size_t i = 0; i < config.button_shoot.size(); )
    {
        std::string& current_key_name = config.button_shoot[i];

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

        std::string combo_label = "Shoot Button " + std::to_string(i);

        if (ImGui::Combo(combo_label.c_str(), &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
        {
            current_key_name = key_names[current_index];
            config.saveConfig();
        }

        ImGui::SameLine();
        std::string remove_button_label = "Remove##button_shoot" + std::to_string(i);
        if (ImGui::Button(remove_button_label.c_str()))
        {
            if (config.button_shoot.size() <= 1)
            {
                config.button_shoot[0] = std::string("None");
                config.saveConfig();
                continue;
            }
            else
            {
                config.button_shoot.erase(config.button_shoot.begin() + i);
                config.saveConfig();
                continue;
            }
        }

        ++i;
    }

    if (ImGui::Button("Add button##shoot"))
    {
        config.button_shoot.push_back("None");
        config.saveConfig();
    }

    ImGui::Separator();

    ImGui::Text("Zoom Buttons");

    for (size_t i = 0; i < config.button_zoom.size(); )
    {
        std::string& current_key_name = config.button_zoom[i];

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

        std::string combo_label = "Zoom Button " + std::to_string(i);

        if (ImGui::Combo(combo_label.c_str(), &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
        {
            current_key_name = key_names[current_index];
            config.saveConfig();
        }

        ImGui::SameLine();
        std::string remove_button_label = "Remove##button_zoom" + std::to_string(i);
        if (ImGui::Button(remove_button_label.c_str()))
        {
            if (config.button_zoom.size() <= 1)
            {
                config.button_zoom[0] = std::string("None");
                config.saveConfig();
                continue;
            }
            else
            {
                config.button_zoom.erase(config.button_zoom.begin() + i);
                config.saveConfig();
                continue;
            }
        }

        ++i;
    }

    if (ImGui::Button("Add button##zoom"))
    {
        config.button_zoom.push_back("None");
        config.saveConfig();
    }

    ImGui::Separator();

    ImGui::Text("Exit Buttons");

    for (size_t i = 0; i < config.button_exit.size(); )
    {
        std::string& current_key_name = config.button_exit[i];

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

        std::string combo_label = "Exit Button " + std::to_string(i);

        if (ImGui::Combo(combo_label.c_str(), &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
        {
            current_key_name = key_names[current_index];
            config.saveConfig();
        }

        ImGui::SameLine();
        std::string remove_button_label = "Remove##button_exit" + std::to_string(i);
        if (ImGui::Button(remove_button_label.c_str()))
        {
            if (config.button_exit.size() <= 1)
            {
                config.button_exit[0] = std::string("None");
                config.saveConfig();
                continue;
            }
            else
            {
                config.button_exit.erase(config.button_exit.begin() + i);
                config.saveConfig();
                continue;
            }
        }

        ++i;
    }

    if (ImGui::Button("Add button##exit"))
    {
        config.button_exit.push_back("None");
        config.saveConfig();
    }

    ImGui::Separator();

    ImGui::Text("Pause Buttons");

    for (size_t i = 0; i < config.button_pause.size(); )
    {
        std::string& current_key_name = config.button_pause[i];

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

        std::string combo_label = "Pause Button " + std::to_string(i);

        if (ImGui::Combo(combo_label.c_str(), &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
        {
            current_key_name = key_names[current_index];
            config.saveConfig();
        }

        ImGui::SameLine();
        std::string remove_button_label = "Remove##button_pause" + std::to_string(i);
        if (ImGui::Button(remove_button_label.c_str()))
        {
            if (config.button_pause.size() <= 1)
            {
                config.button_pause[0] = std::string("None");
                config.saveConfig();
                continue;
            }
            else
            {
                config.button_pause.erase(config.button_pause.begin() + i);
                config.saveConfig();
                continue;
            }
        }
        ++i;
    }

    if (ImGui::Button("Add button##pause"))
    {
        config.button_pause.push_back("None");
        config.saveConfig();
    }

    ImGui::Separator();

    ImGui::Text("Reload config Buttons");

    for (size_t i = 0; i < config.button_reload_config.size(); )
    {
        std::string& current_key_name = config.button_reload_config[i];

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

        std::string combo_label = "Reload config Button " + std::to_string(i);

        if (ImGui::Combo(combo_label.c_str(), &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
        {
            current_key_name = key_names[current_index];
            config.saveConfig();
        }

        ImGui::SameLine();
        std::string remove_button_label = "Remove##button_reload_config" + std::to_string(i);
        if (ImGui::Button(remove_button_label.c_str()))
        {
            if (config.button_reload_config.size() <= 1)
            {
                config.button_reload_config[0] = std::string("None");
                config.saveConfig();
                continue;
            }
            else
            {
                config.button_reload_config.erase(config.button_reload_config.begin() + i);
                config.saveConfig();
                continue;
            }
        }

        ++i;
    }

    if (ImGui::Button("Add button##reload_config"))
    {
        config.button_reload_config.push_back("None");
        config.saveConfig();
    }

    ImGui::Separator();

    ImGui::Text("Overlay Buttons");

    for (size_t i = 0; i < config.button_open_overlay.size(); )
    {
        std::string& current_key_name = config.button_open_overlay[i];

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

        std::string combo_label = "Overlay Button " + std::to_string(i);

        if (ImGui::Combo(combo_label.c_str(), &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
        {
            current_key_name = key_names[current_index];
            config.saveConfig();
        }

        ImGui::SameLine();
        std::string remove_button_label = "Remove##button_open_overlay" + std::to_string(i);
        if (ImGui::Button(remove_button_label.c_str()))
        {
            config.button_open_overlay.erase(config.button_open_overlay.begin() + i);
            config.saveConfig();
            continue;
        }

        ++i;
    }

    if (ImGui::Button("Add button##overlay"))
    {
        config.button_open_overlay.push_back("None");
        config.saveConfig();
    }

    ImGui::Separator();

    if (ImGui::Checkbox("Enable arrows keys options", &config.enable_arrows_settings))
    {
        config.saveConfig();
    }
}