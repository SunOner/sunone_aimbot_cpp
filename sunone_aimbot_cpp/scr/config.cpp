#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "config.h"

bool Config::loadConfig(const std::string& filename)
{
    if (!boost::filesystem::exists(filename))
    {
        std::cerr << "Error: Config file does not exist: " << filename << std::endl;
        return false;
    }

    boost::property_tree::ptree pt;
    try
    {
        boost::property_tree::ini_parser::read_ini(filename, pt);

        // Capture
        detection_resolution = pt.get<int>("detection_resolution", 0);
        capture_borders = pt.get<bool>("capture_borders", true);
        capture_cursor = pt.get<bool>("capture_cursor", true);

        // Target
        disable_headshot = pt.get<bool>("disable_headshot", false);
        body_y_offset = pt.get<float>("body_y_offset", 0.0f);
        ignore_third_person = pt.get<bool>("ignore_third_person", false);

        // Mouse
        dpi = pt.get<int>("dpi", 0);
        sensitivity = pt.get<float>("sensitivity", 0.0f);
        fovX = pt.get<int>("fovX", 0);
        fovY = pt.get<int>("fovY", 0);
        minSpeedMultiplier = pt.get<float>("minSpeedMultiplier", 0.0f);
        maxSpeedMultiplier = pt.get<float>("maxSpeedMultiplier", 0.0f);
        predictionInterval = pt.get<float>("predictionInterval", 0.0f);
        
        // Mouse shooting
        auto_shoot = pt.get<bool>("auto_shoot", false);
        bScope_multiplier = pt.get<float>("bScope_multiplier", 1.0);

        // arduino
        arduino_enable = pt.get<bool>("arduino_enable", "false");
        arduino_baudrate = pt.get<int>("arduino_baudrate", 9600);
        arduino_port = pt.get<std::string>("arduino_port", "COM0");
        arduino_16_bit_mouse = pt.get<bool>("arduino_16_bit_mouse", false);

        // AI
        ai_model = pt.get<std::string>("ai_model", "sunxds_0.5.6.engine");
        engine_image_size = pt.get<int>("engine_image_size", 0);
        confidence_threshold = pt.get<float>("confidence_threshold", 0.0f);
        nms_threshold = pt.get<float>("nms_threshold", 0.5);

        // Buttons
        button_targeting = splitString(pt.get<std::string>("button_targeting", "RightMouseButton"));
        button_exit = splitString(pt.get<std::string>("button_exit", "F2"));
        button_pause = splitString(pt.get<std::string>("button_pause", "F3"));
        button_reload_config = splitString(pt.get<std::string>("button_reload_config", "F4"));

        // overlay
        button_open_overlay = splitString(pt.get<std::string>("button_open_overlay", "Home"));

        // Debug window
        show_window = pt.get<bool>("show_window", true);
        show_fps = pt.get<bool>("show_fps", true);
        window_name = pt.get<std::string>("window_name", "Debug");
        window_size = pt.get<int>("window_size", 100);
        screenshot_button = splitString(pt.get<std::string>("screenshot_button", "RightMouseButton"));
        always_on_top = pt.get<bool>("always_on_top", true);
    }
    catch (boost::property_tree::ini_parser_error& e)
    {
        std::cerr << "Error parsing config file: " << e.what() << std::endl;
        return false;
    }
    catch (boost::property_tree::ptree_bad_path& e)
    {
        std::cerr << "Error reading config value: " << e.what() << std::endl;
        return false;
    }
    catch (boost::property_tree::ptree_bad_data& e)
    {
        std::cerr << "Error converting config value: " << e.what() << std::endl;
        return false;
    }

    return true;
}

bool Config::saveConfig(const std::string& filename)
{
    // TODO: sex
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening config.ini for writing: " << filename << std::endl;
        return false;
    }

    file << "# Detection window\n";
    file << "detection_resolution = " << detection_resolution << "\n";
    file << "capture_borders = " << (capture_borders ? "true" : "false") << "\n";
    file << "capture_cursor = " << (capture_cursor ? "true" : "false") << "\n\n";

    file << "# Target\n";
    file << "disable_headshot = " << (disable_headshot ? "true" : "false") << "\n";
    file << "body_y_offset = " << std::fixed << std::setprecision(2) << body_y_offset << "\n";
    file << "ignore_third_person = " << (ignore_third_person ? "true" : "false") << "\n\n";

    file << "# Mouse move\n";
    file << "dpi = " << dpi << "\n";
    file << "sensitivity = " << std::fixed << std::setprecision(1) << sensitivity << "\n";
    file << "fovX = " << fovX << "\n";
    file << "fovY = " << fovY << "\n";
    file << "minSpeedMultiplier = " << std::fixed << std::setprecision(1) << minSpeedMultiplier << "\n";
    file << "maxSpeedMultiplier = " << std::fixed << std::setprecision(1) << maxSpeedMultiplier << "\n";
    file << "predictionInterval = " << std::fixed << std::setprecision(1) << predictionInterval << "\n\n";

    file << "# Mouse shooting\n";
    file << "auto_shoot = " << (auto_shoot ? "true" : "false") << "\n";
    file << "bScope_multiplier = " << std::fixed << std::setprecision(1) << bScope_multiplier << "\n\n";

    file << "# Arduino\n";
    file << "arduino_enable = " << (arduino_enable ? "true" : "false") << "\n";
    file << "arduino_baudrate = " << arduino_baudrate << "\n";
    file << "arduino_port = " << arduino_port << "\n";
    file << "arduino_16_bit_mouse = " << (arduino_16_bit_mouse ? "true" : "false") << "\n";

    file << "# AI\n";
    file << "ai_model = " << ai_model << "\n";
    file << "engine_image_size = " << engine_image_size << "\n";
    file << "confidence_threshold = " << std::fixed << std::setprecision(1) << confidence_threshold << "\n";
    file << "nms_threshold = " << std::fixed << std::setprecision(1) << nms_threshold << "\n\n";

    file << "# Buttons\n";
    file << "button_targeting = " << joinStrings(button_targeting) << "\n";
    file << "button_exit = " << joinStrings(button_exit) << "\n";
    file << "button_pause = " << joinStrings(button_pause) << "\n";
    file << "button_reload_config = " << joinStrings(button_reload_config) << "\n";
    file << "button_open_overlay = " << joinStrings(button_open_overlay) << "\n\n";

    file << "# Debug window\n";
    file << "show_window = " << (show_window ? "true" : "false") << "\n";
    file << "show_fps = " << (show_fps ? "true" : "false") << "\n";
    file << "window_name = " << window_name << "\n";
    file << "window_size = " << window_size << "\n";
    file << "screenshot_button = " << joinStrings(screenshot_button) << "\n";
    file << "always_on_top = " << (always_on_top ? "true" : "false") << "\n";

    file.close();
    return true;
}

std::vector<std::string> Config::splitString(const std::string& str, char delimiter)
{
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, delimiter))
    {
        item.erase(0, item.find_first_not_of(" \t\n\r\f\v"));
        item.erase(item.find_last_not_of(" \t\n\r\f\v") + 1);
        tokens.push_back(item);
    }
    return tokens;
}

std::string Config::joinStrings(const std::vector<std::string>& vec, const std::string& delimiter)
{
    std::ostringstream oss;
    for (size_t i = 0; i < vec.size(); ++i)
    {
        if (i != 0) oss << delimiter;
        oss << vec[i];
    }
    return oss.str();
}