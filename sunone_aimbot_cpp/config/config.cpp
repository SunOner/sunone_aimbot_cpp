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
        std::cerr << "Config file does not exist, creating default config: " << filename << std::endl;
        // Capture
        capture_method = "duplication_api";
        detection_resolution = 320;
        capture_fps = 60;
        monitor_idx = 0;
        circle_mask = true;
        capture_borders = true;
        capture_cursor = true;
        virtual_camera_name = "None";

        // Target
        disable_headshot = false;
        body_y_offset = 0.15f;
        ignore_third_person = false;
        shooting_range_targets = false;
        auto_aim = false;

        // Mouse
        dpi = 1000;
        sensitivity = 4.0f;
        fovX = 50;
        fovY = 50;
        minSpeedMultiplier = 1.0f;
        maxSpeedMultiplier = 4.0f;
        predictionInterval = 0.20f;
        easynorecoil = false;
        easynorecoilstrength = 0.0f;
        input_method = "WIN32";

        // KMBOX defaults (please make a proper config for this. I can't
        kmbox_ip = "192.168.2.188";
        kmbox_port = "16896";
        kmbox_mac = "46405C53";
        
        // Arduino
        arduino_baudrate = 115200;
        arduino_port = "COM0";
        arduino_16_bit_mouse = false;
        arduino_enable_keys = false;
        
        // Mouse shooting
        auto_shoot = false;
        bScope_multiplier = 1.0f;
        
        // AI
        ai_model = "sunxds_0.5.6.engine";
        confidence_threshold = 0.15f;
        nms_threshold = 0.50;
        max_detections = 100;
        postprocess = "yolo10";
        export_enable_fp8 = false;
        export_enable_fp16 = true;

        // optical flow
        enable_optical_flow = false;
        draw_optical_flow = true;
        draw_optical_flow_steps = 16;
        optical_flow_alpha_cpu = 0.01f;
        optical_flow_magnitudeThreshold = 0.10;
        staticFrameThreshold = 4.0f;

        // Buttons
        button_targeting = splitString("RightMouseButton");
        button_shoot = splitString("LeftMouseButton");
        button_zoom = splitString("RightMouseButton");
        button_exit = splitString("F2");
        button_pause = splitString("F3");
        button_reload_config = splitString("F4");
        button_open_overlay = splitString("Home");

        // Overlay
        overlay_opacity = 225;
        overlay_snow_theme = true;
        overlay_ui_scale = 1.0f;

        // Custom classes
        class_player =  0;
        class_bot = 1;
        class_weapon = 2;
        class_outline = 3;
        class_dead_body = 4;
        class_hideout_target_human = 5;
        class_hideout_target_balls = 6;
        class_head = 7;
        class_smoke = 8;
        class_fire = 9;
        class_third_person = 10;

        // Debug
        show_window = true;
        show_fps = true;
        window_name = "Debug";
        window_size = 80;
        screenshot_button = splitString("None");
        screenshot_delay = 500;
        always_on_top = true;
        verbose = false;

        saveConfig(filename);
    }

    boost::property_tree::ptree pt;
    try
    {
        boost::property_tree::ini_parser::read_ini(filename, pt);

        // Capture
        capture_method = pt.get<std::string>("capture_method", "duplication_api");
        detection_resolution = pt.get<int>("detection_resolution", 320);
        capture_fps = pt.get<int>("capture_fps", 60);
        monitor_idx = pt.get<int>("monitor_idx", 0);
        circle_mask = pt.get<bool>("circle_mask", true);
        capture_borders = pt.get<bool>("capture_borders", true);
        capture_cursor = pt.get<bool>("capture_cursor", true);
        virtual_camera_name = pt.get<std::string>("virtual_camera_name", "None");

        // Target
        disable_headshot = pt.get<bool>("disable_headshot", false);
        body_y_offset = pt.get<float>("body_y_offset", 0.15f);
        ignore_third_person = pt.get<bool>("ignore_third_person", false);
        shooting_range_targets = pt.get<bool>("shooting_range_targets", false);
        auto_aim = pt.get<bool>("auto_aiming", false);

        // Mouse
        dpi = pt.get<int>("dpi", 1000);
        sensitivity = pt.get<float>("sensitivity", 4.0f);
        fovX = pt.get<int>("fovX", 50);
        fovY = pt.get<int>("fovY", 50);
        minSpeedMultiplier = pt.get<float>("minSpeedMultiplier", 1.0f);
        maxSpeedMultiplier = pt.get<float>("maxSpeedMultiplier", 4.0f);
        predictionInterval = pt.get<float>("predictionInterval", 0.5f);
        easynorecoil = pt.get<bool>("easynorecoil", false);
        easynorecoilstrength = pt.get<float>("easynorecoilstrength", 0.0f);
        input_method = pt.get<std::string>("input_method", "WIN32");
        kmbox_ip = pt.get<std::string>("kmbox_ip", "192.168.2.188");
        kmbox_port = pt.get<std::string>("kmbox_port", "16896");
        kmbox_mac = pt.get<std::string>("kmbox_mac", "46405C53");

        // Arduino
        arduino_baudrate = pt.get<int>("arduino_baudrate", 115200);
        arduino_port = pt.get<std::string>("arduino_port", "COM0");
        arduino_16_bit_mouse = pt.get<bool>("arduino_16_bit_mouse", false);
        arduino_enable_keys = pt.get<bool>("arduino_enable_keys", false);

        // Mouse shooting
        auto_shoot = pt.get<bool>("auto_shoot", false);
        bScope_multiplier = pt.get<float>("bScope_multiplier", 1.2f);

        // AI
        ai_model = pt.get<std::string>("ai_model", "sunxds_0.5.6.engine");
        confidence_threshold = pt.get<float>("confidence_threshold", 0.15f);
        nms_threshold = pt.get<float>("nms_threshold", 0.50);
        max_detections = pt.get<int>("max_detections", 20);
        postprocess = pt.get<std::string>("postprocess", "yolo11");
        export_enable_fp8 = pt.get<bool>("export_enable_fp8", true);
        export_enable_fp16 = pt.get<bool>("export_enable_fp16", true);

        // Optical Flow
        enable_optical_flow = pt.get<bool>("enable_optical_flow", false);
        draw_optical_flow = pt.get<bool>("draw_optical_flow", true);
        draw_optical_flow_steps = pt.get<int>("draw_optical_flow_steps", 16);
        optical_flow_alpha_cpu = pt.get<float>("optical_flow_alpha_cpu", 0.06f);
        optical_flow_magnitudeThreshold = pt.get<double>("optical_flow_magnitudeThreshold", 2.08);
        staticFrameThreshold = pt.get<float>("staticFrameThreshold", 4.0f);

        // Buttons
        button_targeting = splitString(pt.get<std::string>("button_targeting", "RightMouseButton"));
        button_shoot = splitString(pt.get<std::string>("button_shoot", "LeftMouseButton"));
        button_zoom = splitString(pt.get<std::string>("button_zoom", "RightMouseButton"));
        button_exit = splitString(pt.get<std::string>("button_exit", "F2"));
        button_pause = splitString(pt.get<std::string>("button_pause", "F3"));
        button_reload_config = splitString(pt.get<std::string>("button_reload_config", "F4"));
        button_open_overlay = splitString(pt.get<std::string>("button_open_overlay", "Home"));

        // Overlay
        overlay_opacity = pt.get<int>("overlay_opacity", 225);
        overlay_snow_theme = pt.get<bool>("overlay_snow_theme", true);
        overlay_ui_scale = pt.get<float>("overlay_ui_scale", 1.0f);

        // Custom Classes
        class_player = pt.get<int>("class_player", 0);
        class_bot = pt.get<int>("class_bot", 1);
        class_weapon = pt.get<int>("class_weapon", 2);
        class_outline = pt.get<int>("class_outline", 3);
        class_dead_body = pt.get<int>("class_dead_body", 4);
        class_hideout_target_human = pt.get<int>("class_hideout_target_human", 5);
        class_hideout_target_balls = pt.get<int>("class_hideout_target_balls", 6);
        class_head = pt.get<int>("class_head", 7);
        class_smoke = pt.get<int>("class_smoke", 8);
        class_fire = pt.get<int>("class_fire", 9);
        class_third_person = pt.get<int>("class_third_person", 10);

        // Debug window
        show_window = pt.get<bool>("show_window", true);
        show_fps = pt.get<bool>("show_fps", true);
        window_name = pt.get<std::string>("window_name", "Debug");
        window_size = pt.get<int>("window_size", 80);
        screenshot_button = splitString(pt.get<std::string>("screenshot_button", "None"));
        screenshot_delay = pt.get<int>("screenshot_delay", 500);
        always_on_top = pt.get<bool>("always_on_top", true);
        verbose = pt.get<bool>("verbose", false);
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
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening config for writing: " << filename << std::endl;
        return false;
    }

    file << "# An explanation of the options can be found at the link\n";
    file << "# https://github.com/SunOner/sunone_aimbot_docs/blob/main/config/config_cpp.md\n\n";
    file << "# Capture\n";
    file << "# duplication_api, winrt, virtual_camera\n";
    file << "capture_method = " << capture_method << "\n";
    file << "detection_resolution = " << detection_resolution << "\n";
    file << "capture_fps = " << capture_fps << "\n";
    file << "monitor_idx = " << monitor_idx << "\n";
    file << "circle_mask = " << (circle_mask ? "true" : "false") << "\n";
    file << "capture_borders = " << (capture_borders ? "true" : "false") << "\n";
    file << "capture_cursor = " << (capture_cursor ? "true" : "false") << "\n";
    file << "virtual_camera_name = " << virtual_camera_name << "\n\n";

    file << "# Target\n";
    file << "disable_headshot = " << (disable_headshot ? "true" : "false") << "\n";
    file << "body_y_offset = " << std::fixed << std::setprecision(2) << body_y_offset << "\n";
    file << "ignore_third_person = " << (ignore_third_person ? "true" : "false") << "\n";
    file << "shooting_range_targets = " << (shooting_range_targets ? "true" : "false") << "\n";
    file << "auto_aim = " << (auto_aim ? "true" : "false") << "\n\n";

    file << "# Mouse move\n";
    file << "dpi = " << dpi << "\n";
    file << "sensitivity = " << std::fixed << std::setprecision(1) << sensitivity << "\n";
    file << "fovX = " << fovX << "\n";
    file << "fovY = " << fovY << "\n";
    file << "minSpeedMultiplier = " << std::fixed << std::setprecision(1) << minSpeedMultiplier << "\n";
    file << "maxSpeedMultiplier = " << std::fixed << std::setprecision(1) << maxSpeedMultiplier << "\n";
    file << "predictionInterval = " << std::fixed << std::setprecision(2) << predictionInterval << "\n";
    file << "easynorecoil = " << (easynorecoil ? "true" : "false") << "\n";
    file << "easynorecoilstrength = " << std::fixed << std::setprecision(1) << easynorecoilstrength << "\n";
    file << "# WIN32, GHUB, ARDUINO\n";
    file << "input_method = " << input_method << "\n\n";
    file << "# KMBOX configuration\n";
    file << "kmbox_ip = " << kmbox_ip << "\n";
    file << "kmbox_port = " << kmbox_port << "\n";
    file << "kmbox_mac = " << kmbox_mac << "\n\n";

    file << "# Arduino\n";
    file << "arduino_baudrate = " << arduino_baudrate << "\n";
    file << "arduino_port = " << arduino_port << "\n";
    file << "arduino_16_bit_mouse = " << (arduino_16_bit_mouse ? "true" : "false") << "\n";
    file << "arduino_enable_keys = " << (arduino_enable_keys ? "true" : "false") << "\n\n";

    file << "# Mouse shooting\n";
    file << "auto_shoot = " << (auto_shoot ? "true" : "false") << "\n";
    file << "bScope_multiplier = " << std::fixed << std::setprecision(1) << bScope_multiplier << "\n\n";

    file << "# AI\n";
    file << "ai_model = " << ai_model << "\n";
    file << "confidence_threshold = " << std::fixed << std::setprecision(2) << confidence_threshold << "\n";
    file << "nms_threshold = " << std::fixed << std::setprecision(2) << nms_threshold << "\n";
    file << "max_detections = " << max_detections << "\n";
    file << "postprocess = " << postprocess << "\n";
    file << "export_enable_fp8 = " << (export_enable_fp8 ? "true" : "false") << "\n";
    file << "export_enable_fp16 = " << (export_enable_fp16 ? "true" : "false") << "\n\n";

    file << "# Optical Flow\n";
    file << "enable_optical_flow = " << (enable_optical_flow ? "true" : "false") << "\n";
    file << "draw_optical_flow = " << (draw_optical_flow ? "true" : "false") << "\n";
    file << "draw_optical_flow_steps = " << draw_optical_flow_steps << "\n";
    file << "optical_flow_alpha_cpu = " << std::fixed << std::setprecision(2) << optical_flow_alpha_cpu << "\n";
    file << "optical_flow_magnitudeThreshold = " << std::fixed << std::setprecision(2) << optical_flow_magnitudeThreshold << "\n";
    file << "staticFrameThreshold = " << std::fixed << std::setprecision(2) << staticFrameThreshold << "\n\n";

    file << "# Buttons\n";
    file << "button_targeting = " << joinStrings(button_targeting) << "\n";
    file << "button_shoot = " << joinStrings(button_shoot) << "\n";
    file << "button_zoom = " << joinStrings(button_zoom) << "\n";
    file << "button_exit = " << joinStrings(button_exit) << "\n";
    file << "button_pause = " << joinStrings(button_pause) << "\n";
    file << "button_reload_config = " << joinStrings(button_reload_config) << "\n";
    file << "button_open_overlay = " << joinStrings(button_open_overlay) << "\n\n";

    file << "# Overlay\n";
    file << "overlay_opacity = " << overlay_opacity << "\n";
    file << "overlay_snow_theme = " << (overlay_snow_theme ? "true" : "false") << "\n";
    file << "overlay_ui_scale = " << std::fixed << std::setprecision(2) << overlay_ui_scale << "\n\n";

    file << "# Custom Classes\n";
    file << "class_player = " << class_player << "\n";
    file << "class_bot = " << class_bot << "\n";
    file << "class_weapon = " << class_weapon << "\n";
    file << "class_outline = " << class_outline << "\n";
    file << "class_dead_body = " << class_dead_body << "\n";
    file << "class_hideout_target_human = " << class_hideout_target_human << "\n";
    file << "class_hideout_target_balls = " << class_hideout_target_balls << "\n";
    file << "class_head = " << class_head << "\n";
    file << "class_smoke = " << class_smoke << "\n";
    file << "class_fire = " << class_fire << "\n";
    file << "class_third_person = " << class_third_person << "\n\n";

    file << "# Debug window\n";
    file << "show_window = " << (show_window ? "true" : "false") << "\n";
    file << "show_fps = " << (show_fps ? "true" : "false") << "\n";
    file << "window_name = " << window_name << "\n";
    file << "window_size = " << window_size << "\n";
    file << "screenshot_button = " << joinStrings(screenshot_button) << "\n";
    file << "screenshot_delay = " << screenshot_delay << "\n";
    file << "always_on_top = " << (always_on_top ? "true" : "false") << "\n";
    file << "verbose = " << (verbose ? "true" : "false");

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
