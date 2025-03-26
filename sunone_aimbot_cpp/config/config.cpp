#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <windows.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <filesystem>

#include "config.h"
#include "modules/SimpleIni.h"

std::vector<std::string> Config::splitString(const std::string& str, char delimiter)
{
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, delimiter))
    {
        while (!item.empty() && (item.front() == ' ' || item.front() == '\t'))
            item.erase(item.begin());
        while (!item.empty() && (item.back() == ' ' || item.back() == '\t'))
            item.pop_back();

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

bool Config::loadConfig(const std::string& filename)
{
    if (!std::filesystem::exists(filename))
    {
        std::cerr << "[Config] Config file does not exist, creating default config: " << filename << std::endl;

        // Capture
        capture_method = "duplication_api";
        detection_resolution = 320;
        capture_fps = 60;
        capture_use_cuda = true;
        monitor_idx = 0;
        circle_mask = true;
        capture_borders = true;
        capture_cursor = true;
        virtual_camera_name = "None";

        // Target
        disable_headshot = false;
        body_y_offset = 0.15f;
        head_y_offset = 0.05f;
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

        // Arduino
        arduino_baudrate = 115200;
        arduino_port = "COM0";
        arduino_16_bit_mouse = false;
        arduino_enable_keys = false;

        // Kmbox
        kmbox_baudrate = 115200;
        kmbox_port = "COM0";
        kmbox_enable_keys = false;

        // Mouse shooting
        auto_shoot = false;
        bScope_multiplier = 1.0f;

        // AI
        backend = "TRT";
        ai_model = "sunxds_0.5.6.engine";
        confidence_threshold = 0.15f;
        nms_threshold = 0.50f;
        max_detections = 100;
        postprocess = "yolo10";
        export_enable_fp8 = false;
        export_enable_fp16 = true;

        // CUDA
        use_cuda_graph = true;
        use_pinned_memory = true;

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
        enable_arrows_settings = false;

        // Overlay
        overlay_opacity = 225;
        overlay_snow_theme = true;
        overlay_ui_scale = 1.0f;

        // Custom classes
        class_player = 0;
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
        return true;
    }

    CSimpleIniA ini;
    ini.SetUnicode();
    SI_Error rc = ini.LoadFile(filename.c_str());
    if (rc < 0) {
        std::cerr << "[Config] Error parsing INI file: " << filename << std::endl;
        return false;
    }

    auto get_string = [&](const char* key, const char* defval)
    {
        const char* val = ini.GetValue("", key, defval);
        return std::string(val ? val : "");
    };

    auto get_bool = [&](const char* key, bool defval)
    {
        return ini.GetBoolValue("", key, defval);
    };

    auto get_long = [&](const char* key, long defval)
    {
        return (int)ini.GetLongValue("", key, defval);
    };

    auto get_double = [&](const char* key, double defval)
    {
        return ini.GetDoubleValue("", key, defval);
    };

    // Capture
    capture_method = get_string("capture_method", "duplication_api");
    detection_resolution = get_long("detection_resolution", 320);
    capture_fps = get_long("capture_fps", 60);
    capture_use_cuda = get_bool("capture_use_cuda", true);
    monitor_idx = get_long("monitor_idx", 0);
    circle_mask = get_bool("circle_mask", true);
    capture_borders = get_bool("capture_borders", true);
    capture_cursor = get_bool("capture_cursor", true);
    virtual_camera_name = get_string("virtual_camera_name", "None");

    // Target
    disable_headshot = get_bool("disable_headshot", false);
    body_y_offset = (float)get_double("body_y_offset", 0.15);
    head_y_offset = (float)get_double("head_y_offset", 0.05);
    ignore_third_person = get_bool("ignore_third_person", false);
    shooting_range_targets = get_bool("shooting_range_targets", false);
    auto_aim = get_bool("auto_aim", false);

    // Mouse
    dpi = get_long("dpi", 1000);
    sensitivity = (float)get_double("sensitivity", 4.0);
    fovX = get_long("fovX", 50);
    fovY = get_long("fovY", 50);
    minSpeedMultiplier = (float)get_double("minSpeedMultiplier", 1.0);
    maxSpeedMultiplier = (float)get_double("maxSpeedMultiplier", 4.0);
    predictionInterval = (float)get_double("predictionInterval", 0.2);
    easynorecoil = get_bool("easynorecoil", false);
    easynorecoilstrength = (float)get_double("easynorecoilstrength", 0.0);
    input_method = get_string("input_method", "WIN32");

    // Arduino
    arduino_baudrate = get_long("arduino_baudrate", 115200);
    arduino_port = get_string("arduino_port", "COM0");
    arduino_16_bit_mouse = get_bool("arduino_16_bit_mouse", false);
    arduino_enable_keys = get_bool("arduino_enable_keys", false);

    // Kmbox
    kmbox_baudrate = get_long("kmbox_baudrate", 115200);
    kmbox_port = get_string("kmbox_port", "COM0");
    kmbox_enable_keys = get_bool("kmbox_enable_keys", false);

    // Mouse shooting
    auto_shoot = get_bool("auto_shoot", false);
    bScope_multiplier = (float)get_double("bScope_multiplier", 1.2);

    // AI
    backend = get_string("backend", "TRT");
    ai_model = get_string("ai_model", "sunxds_0.5.6.engine");
    confidence_threshold = (float)get_double("confidence_threshold", 0.15);
    nms_threshold = (float)get_double("nms_threshold", 0.50);
    max_detections = get_long("max_detections", 20);
    postprocess = get_string("postprocess", "yolo11");
    export_enable_fp8 = get_bool("export_enable_fp8", true);
    export_enable_fp16 = get_bool("export_enable_fp16", true);

    // CUDA
    use_cuda_graph = get_bool("use_cuda_graph", true);
    use_pinned_memory = get_bool("use_pinned_memory", true);

    // Optical Flow
    enable_optical_flow = get_bool("enable_optical_flow", false);
    draw_optical_flow = get_bool("draw_optical_flow", true);
    draw_optical_flow_steps = get_long("draw_optical_flow_steps", 16);
    optical_flow_alpha_cpu = (float)get_double("optical_flow_alpha_cpu", 0.06);
    optical_flow_magnitudeThreshold = get_double("optical_flow_magnitudeThreshold", 2.08);
    staticFrameThreshold = (float)get_double("staticFrameThreshold", 4.0);

    // Buttons
    button_targeting = splitString(get_string("button_targeting", "RightMouseButton"));
    button_shoot = splitString(get_string("button_shoot", "LeftMouseButton"));
    button_zoom = splitString(get_string("button_zoom", "RightMouseButton"));
    button_exit = splitString(get_string("button_exit", "F2"));
    button_pause = splitString(get_string("button_pause", "F3"));
    button_reload_config = splitString(get_string("button_reload_config", "F4"));
    button_open_overlay = splitString(get_string("button_open_overlay", "Home"));
    enable_arrows_settings = get_bool("enable_arrows_settings", false);

    // Overlay
    overlay_opacity = get_long("overlay_opacity", 225);
    overlay_snow_theme = get_bool("overlay_snow_theme", true);
    overlay_ui_scale = (float)get_double("overlay_ui_scale", 1.0);

    // Custom Classes
    class_player = get_long("class_player", 0);
    class_bot = get_long("class_bot", 1);
    class_weapon = get_long("class_weapon", 2);
    class_outline = get_long("class_outline", 3);
    class_dead_body = get_long("class_dead_body", 4);
    class_hideout_target_human = get_long("class_hideout_target_human", 5);
    class_hideout_target_balls = get_long("class_hideout_target_balls", 6);
    class_head = get_long("class_head", 7);
    class_smoke = get_long("class_smoke", 8);
    class_fire = get_long("class_fire", 9);
    class_third_person = get_long("class_third_person", 10);

    // Debug window
    show_window = get_bool("show_window", true);
    show_fps = get_bool("show_fps", true);
    window_name = get_string("window_name", "Debug");
    window_size = get_long("window_size", 80);
    screenshot_button = splitString(get_string("screenshot_button", "None"));
    screenshot_delay = get_long("screenshot_delay", 500);
    always_on_top = get_bool("always_on_top", true);
    verbose = get_bool("verbose", false);

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

    file << "# An explanation of the options can be found at:\n";
    file << "# https://github.com/SunOner/sunone_aimbot_docs/blob/main/config/config_cpp.md\n\n";

    // Capture
    file << "# Capture\n"
        << "capture_method = " << capture_method << "\n"
        << "detection_resolution = " << detection_resolution << "\n"
        << "capture_fps = " << capture_fps << "\n"
        << "capture_use_cuda = " << (capture_use_cuda ? "true" : "false") << "\n"
        << "monitor_idx = " << monitor_idx << "\n"
        << "circle_mask = " << (circle_mask ? "true" : "false") << "\n"
        << "capture_borders = " << (capture_borders ? "true" : "false") << "\n"
        << "capture_cursor = " << (capture_cursor ? "true" : "false") << "\n"
        << "virtual_camera_name = " << virtual_camera_name << "\n\n";

    // Target
    file << "# Target\n"
        << "disable_headshot = " << (disable_headshot ? "true" : "false") << "\n"
        << std::fixed << std::setprecision(2)
        << "body_y_offset = " << body_y_offset << "\n"
        << "head_y_offset = " << head_y_offset << "\n"
        << "ignore_third_person = " << (ignore_third_person ? "true" : "false") << "\n"
        << "shooting_range_targets = " << (shooting_range_targets ? "true" : "false") << "\n"
        << "auto_aim = " << (auto_aim ? "true" : "false") << "\n\n";

    // Mouse
    file << "# Mouse move\n"
        << "dpi = " << dpi << "\n"
        << std::fixed << std::setprecision(1)
        << "sensitivity = " << sensitivity << "\n"
        << "fovX = " << fovX << "\n"
        << "fovY = " << fovY << "\n"
        << "minSpeedMultiplier = " << minSpeedMultiplier << "\n"
        << "maxSpeedMultiplier = " << maxSpeedMultiplier << "\n"
        << std::fixed << std::setprecision(2)
        << "predictionInterval = " << predictionInterval << "\n"
        << "easynorecoil = " << (easynorecoil ? "true" : "false") << "\n"
        << std::fixed << std::setprecision(1)
        << "easynorecoilstrength = " << easynorecoilstrength << "\n"
        << "# WIN32, GHUB, ARDUINO\n"
        << "input_method = " << input_method << "\n\n";

    // Arduino
    file << "# Arduino\n"
        << "arduino_baudrate = " << arduino_baudrate << "\n"
        << "arduino_port = " << arduino_port << "\n"
        << "arduino_16_bit_mouse = " << (arduino_16_bit_mouse ? "true" : "false") << "\n"
        << "arduino_enable_keys = " << (arduino_enable_keys ? "true" : "false") << "\n\n";

    // Kmbox
    file << "# Kmbox\n"
        << "kmbox_baudrate = " << kmbox_baudrate << "\n"
        << "kmbox_port = " << kmbox_port << "\n"
        << "kmbox_enable_keys = " << (kmbox_enable_keys ? "true" : "false") << "\n\n";

    // Mouse shooting
    file << "# Mouse shooting\n"
        << "auto_shoot = " << (auto_shoot ? "true" : "false") << "\n"
        << std::fixed << std::setprecision(1)
        << "bScope_multiplier = " << bScope_multiplier << "\n\n";

    // AI
    file << "# AI\n"
        << "backend = " << backend << "\n"
        << "ai_model = " << ai_model << "\n"
        << std::fixed << std::setprecision(2)
        << "confidence_threshold = " << confidence_threshold << "\n"
        << "nms_threshold = " << nms_threshold << "\n"
        << std::setprecision(0)
        << "max_detections = " << max_detections << "\n"
        << "postprocess = " << postprocess << "\n"
        << "export_enable_fp8 = " << (export_enable_fp8 ? "true" : "false") << "\n"
        << "export_enable_fp16 = " << (export_enable_fp16 ? "true" : "false") << "\n\n";

    // CUDA
    file << "# CUDA\n"
        << "use_cuda_graph = " << (use_cuda_graph ? "true" : "false") << "\n"
        << "use_pinned_memory = " << (use_pinned_memory ? "true" : "false") << "\n\n";

    // Optical Flow
    file << "# Optical Flow\n"
        << "enable_optical_flow = " << (enable_optical_flow ? "true" : "false") << "\n"
        << "draw_optical_flow = " << (draw_optical_flow ? "true" : "false") << "\n"
        << "draw_optical_flow_steps = " << draw_optical_flow_steps << "\n"
        << std::fixed << std::setprecision(2)
        << "optical_flow_alpha_cpu = " << optical_flow_alpha_cpu << "\n"
        << "optical_flow_magnitudeThreshold = " << optical_flow_magnitudeThreshold << "\n"
        << "staticFrameThreshold = " << staticFrameThreshold << "\n\n";

    // Buttons
    file << "# Buttons\n"
        << "button_targeting = " << joinStrings(button_targeting) << "\n"
        << "button_shoot = " << joinStrings(button_shoot) << "\n"
        << "button_zoom = " << joinStrings(button_zoom) << "\n"
        << "button_exit = " << joinStrings(button_exit) << "\n"
        << "button_pause = " << joinStrings(button_pause) << "\n"
        << "button_reload_config = " << joinStrings(button_reload_config) << "\n"
        << "button_open_overlay = " << joinStrings(button_open_overlay) << "\n"
        << "enable_arrows_settings = " << (enable_arrows_settings ? "true" : "false") << "\n\n";

    // Overlay
    file << "# Overlay\n"
        << "overlay_opacity = " << overlay_opacity << "\n"
        << "overlay_snow_theme = " << (overlay_snow_theme ? "true" : "false") << "\n"
        << std::fixed << std::setprecision(2)
        << "overlay_ui_scale = " << overlay_ui_scale << "\n\n";

    // Custom Classes
    file << "# Custom Classes\n"
        << "class_player = " << class_player << "\n"
        << "class_bot = " << class_bot << "\n"
        << "class_weapon = " << class_weapon << "\n"
        << "class_outline = " << class_outline << "\n"
        << "class_dead_body = " << class_dead_body << "\n"
        << "class_hideout_target_human = " << class_hideout_target_human << "\n"
        << "class_hideout_target_balls = " << class_hideout_target_balls << "\n"
        << "class_head = " << class_head << "\n"
        << "class_smoke = " << class_smoke << "\n"
        << "class_fire = " << class_fire << "\n"
        << "class_third_person = " << class_third_person << "\n\n";

    // Debug
    file << "# Debug window\n"
        << "show_window = " << (show_window ? "true" : "false") << "\n"
        << "show_fps = " << (show_fps ? "true" : "false") << "\n"
        << "window_name = " << window_name << "\n"
        << "window_size = " << window_size << "\n"
        << "screenshot_button = " << joinStrings(screenshot_button) << "\n"
        << "screenshot_delay = " << screenshot_delay << "\n"
        << "always_on_top = " << (always_on_top ? "true" : "false") << "\n"
        << "verbose = " << (verbose ? "true" : "false") << "\n";

    file.close();
    return true;
}