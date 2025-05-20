#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <windows.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <filesystem>
#include <unordered_map>

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
        capture_use_cuda = false;
        monitor_idx = 0;
        circle_mask = true;
        capture_borders = true;
        capture_cursor = true;
        virtual_camera_name = "None";
        virtual_camera_width = 1920;
        virtual_camera_heigth = 1080;

        // Target
        disable_headshot = false;
        body_y_offset = 0.15f;
        head_y_offset = 0.05f;
        ignore_third_person = false;
        shooting_range_targets = false;
        auto_aim = false;

        // Mouse
        fovX = 106;
        fovY = 74;
        minSpeedMultiplier = 0.1f;
        maxSpeedMultiplier = 0.1f;

        predictionInterval = 0.01f;
        prediction_futurePositions = 20;
        draw_futurePositions = true;

        snapRadius = 1.5f;
        nearRadius = 25.0f;
        speedCurveExponent = 3.0f;
        snapBoostFactor = 1.15f;

        easynorecoil = false;
        easynorecoilstrength = 0.0f;
        input_method = "WIN32";

        // Wind mouse
        wind_mouse_enabled = false;
        wind_G = 18.0f;
        wind_W = 15.0f;
        wind_M = 10.0f;
        wind_D = 8.0f;

        // Arduino
        arduino_baudrate = 115200;
        arduino_port = "COM0";
        arduino_16_bit_mouse = false;
        arduino_enable_keys = false;

        // kmbox_B
        kmbox_b_baudrate = 115200;
        kmbox_b_port = "COM0";

        // Mouse shooting
        auto_shoot = false;
        bScope_multiplier = 1.0f;

        // AI
        backend = "TRT";
        dml_device_id = 0;
        ai_model = "sunxds_0.5.6.engine";
        confidence_threshold = 0.10f;
        nms_threshold = 0.50f;
        max_detections = 100;
        postprocess = "yolo10";
        batch_size = 1;
        export_enable_fp8 = false;
        export_enable_fp16 = true;

        // CUDA
        use_cuda_graph = false;
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
        window_size = 80;
        screenshot_button = splitString("None");
        screenshot_delay = 500;
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

    auto get_string = [&](const char* key, const std::string& defval)
    {
        const char* val = ini.GetValue("", key, defval.c_str());
        return val ? std::string(val) : defval;
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

    game_profiles.clear();

    CSimpleIniA::TNamesDepend keys;
    ini.GetAllKeys("Games", keys);

    for (const auto& k : keys)
    {
        std::string name = k.pItem;
        std::string val = ini.GetValue("Games", k.pItem, "");
        auto parts = splitString(val, ',');

        try
        {
            if (parts.size() < 2)
                throw std::runtime_error("not enough values");

            GameProfile gp;
            gp.name = name;
            gp.sens = std::stod(parts[0]);
            gp.yaw = std::stod(parts[1]);
            gp.pitch = parts.size() > 2 ? std::stod(parts[2]) : gp.yaw;
            gp.fovScaled = parts.size() > 3 && (parts[3] == "true" || parts[3] == "1");
            gp.baseFOV = parts.size() > 4 ? std::stod(parts[4]) : 0.0;

            game_profiles[name] = gp;
        }
        catch (const std::exception& e)
        {
            std::cerr << "[Config] Failed to parse profile: " << name
                << " = " << val << " (" << e.what() << ")" << std::endl;
        }
    }

    if (!game_profiles.count("UNIFIED"))
    {
        GameProfile uni;
        uni.name = "UNIFIED";
        uni.yaw = 0.022;
        uni.pitch = uni.yaw;
        uni.fovScaled = false;
        uni.baseFOV = 0.0;
        game_profiles[uni.name] = uni;
    }

    active_game = get_string("active_game", active_game);
    if (!game_profiles.count(active_game) && !game_profiles.empty())
        active_game = game_profiles.begin()->first;

    // Capture
    capture_method = get_string("capture_method", "duplication_api");
    detection_resolution = get_long("detection_resolution", 320);
    if (detection_resolution != 160 && detection_resolution != 320 && detection_resolution != 640)
        detection_resolution = 320;

    capture_fps = get_long("capture_fps", 60);
    capture_use_cuda = get_bool("capture_use_cuda", false);
    monitor_idx = get_long("monitor_idx", 0);
    circle_mask = get_bool("circle_mask", true);
    capture_borders = get_bool("capture_borders", true);
    capture_cursor = get_bool("capture_cursor", true);
    virtual_camera_name = get_string("virtual_camera_name", "None");
    virtual_camera_width = get_long("virtual_camera_width", 1920);
    virtual_camera_heigth = get_long("virtual_camera_heigth", 1080);

    // Target
    disable_headshot = get_bool("disable_headshot", false);
    body_y_offset = (float)get_double("body_y_offset", 0.15);
    head_y_offset = (float)get_double("head_y_offset", 0.05);
    ignore_third_person = get_bool("ignore_third_person", false);
    shooting_range_targets = get_bool("shooting_range_targets", false);
    auto_aim = get_bool("auto_aim", false);

    // Mouse
    fovX = get_long("fovX", 106);
    fovY = get_long("fovY", 74);
    minSpeedMultiplier = (float)get_double("minSpeedMultiplier", 0.1);
    maxSpeedMultiplier = (float)get_double("maxSpeedMultiplier", 0.1);

    predictionInterval = (float)get_double("predictionInterval", 0.01);
    prediction_futurePositions = get_long("prediction_futurePositions", 20);
    draw_futurePositions = get_bool("draw_futurePositions", true);
    
    snapRadius = (float)get_double("snapRadius", 1.5);
    nearRadius = (float)get_double("nearRadius", 25.0);
    speedCurveExponent = (float)get_double("speedCurveExponent", 3.0);
    snapBoostFactor = (float)get_double("snapBoostFactor", 1.15);

    easynorecoil = get_bool("easynorecoil", false);
    easynorecoilstrength = (float)get_double("easynorecoilstrength", 0.0);
    input_method = get_string("input_method", "WIN32");

    // Wind mouse
    wind_mouse_enabled = get_bool("wind_mouse_enabled", false);
    wind_G = (float)get_double("wind_G", 18.0f);
    wind_W = (float)get_double("wind_W", 15.0f);
    wind_M = (float)get_double("wind_M", 10.0f);
    wind_D = (float)get_double("wind_D", 8.0f);

    // Arduino
    arduino_baudrate = get_long("arduino_baudrate", 115200);
    arduino_port = get_string("arduino_port", "COM0");
    arduino_16_bit_mouse = get_bool("arduino_16_bit_mouse", false);
    arduino_enable_keys = get_bool("arduino_enable_keys", false);

    // kmbox_B
    kmbox_b_baudrate = get_long("kmbox_baudrate", 115200);
    kmbox_b_port = get_string("kmbox_port", "COM0");

    // Mouse shooting
    auto_shoot = get_bool("auto_shoot", false);
    bScope_multiplier = (float)get_double("bScope_multiplier", 1.2);

    // AI
    backend = get_string("backend", "TRT");
    dml_device_id = get_long("dml_device_id", 0);
    ai_model = get_string("ai_model", "sunxds_0.5.6.engine");
    confidence_threshold = (float)get_double("confidence_threshold", 0.15);
    nms_threshold = (float)get_double("nms_threshold", 0.50);
    max_detections = get_long("max_detections", 20);
    postprocess = get_string("postprocess", "yolo11");
    batch_size = get_long("batch_size", 1);
    if (batch_size < 1) batch_size = 1;
    if (batch_size > 8) batch_size = 8;

    export_enable_fp8 = get_bool("export_enable_fp8", true);
    export_enable_fp16 = get_bool("export_enable_fp16", true);

    // CUDA
    use_cuda_graph = get_bool("use_cuda_graph", false);
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
    window_size = get_long("window_size", 80);
    screenshot_button = splitString(get_string("screenshot_button", "None"));
    screenshot_delay = get_long("screenshot_delay", 500);
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
        << "virtual_camera_name = " << virtual_camera_name << "\n"
        << "virtual_camera_width = " << virtual_camera_width << "\n"
        << "virtual_camera_heigth = " << virtual_camera_heigth << "\n\n";

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
        << "fovX = " << fovX << "\n"
        << "fovY = " << fovY << "\n"
        << "minSpeedMultiplier = " << minSpeedMultiplier << "\n"
        << "maxSpeedMultiplier = " << maxSpeedMultiplier << "\n"

        << std::fixed << std::setprecision(2)
        << "predictionInterval = " << predictionInterval << "\n"
        << "prediction_futurePositions = " << prediction_futurePositions << "\n"
        << "draw_futurePositions = " << (draw_futurePositions ? "true" : "false") << "\n"

        << "snapRadius = " << snapRadius << "\n"
        << "nearRadius = " << nearRadius << "\n"
        << "speedCurveExponent = " << speedCurveExponent << "\n"
        << std::fixed << std::setprecision(2)
        << "snapBoostFactor = " << snapBoostFactor << "\n"

        << "easynorecoil = " << (easynorecoil ? "true" : "false") << "\n"
        << std::fixed << std::setprecision(1)
        << "easynorecoilstrength = " << easynorecoilstrength << "\n"

        << "# WIN32, GHUB, ARDUINO, KMBOX_B\n"
        << "input_method = " << input_method << "\n\n";

    // Wind mouse
    file << "# Wind mouse\n"
        << "wind_mouse_enabled = " << (wind_mouse_enabled ? "true" : "false") << "\n"
        << "wind_G = " << wind_G << "\n"
        << "wind_W = " << wind_W << "\n"
        << "wind_M = " << wind_M << "\n"
        << "wind_D = " << wind_D << "\n\n";

    // Arduino
    file << "# Arduino\n"
        << "arduino_baudrate = " << arduino_baudrate << "\n"
        << "arduino_port = " << arduino_port << "\n"
        << "arduino_16_bit_mouse = " << (arduino_16_bit_mouse ? "true" : "false") << "\n"
        << "arduino_enable_keys = " << (arduino_enable_keys ? "true" : "false") << "\n\n";

    // kmbox_B
    file << "# Kmbox_B\n"
        << "kmbox_baudrate = " << kmbox_b_baudrate << "\n"
        << "kmbox_port = " << kmbox_b_port << "\n\n";

    // Mouse shooting
    file << "# Mouse shooting\n"
        << "auto_shoot = " << (auto_shoot ? "true" : "false") << "\n"
        << std::fixed << std::setprecision(1)
        << "bScope_multiplier = " << bScope_multiplier << "\n\n";

    // AI
    file << "# AI\n"
        << "backend = " << backend << "\n"
        << "dml_device_id = " << dml_device_id << "\n"
        << "ai_model = " << ai_model << "\n"
        << std::fixed << std::setprecision(2)
        << "confidence_threshold = " << confidence_threshold << "\n"
        << "nms_threshold = " << nms_threshold << "\n"
        << std::setprecision(0)
        << "max_detections = " << max_detections << "\n"
        << "postprocess = " << postprocess << "\n"
        << "batch_size = " << batch_size << "\n"
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
    file << "# Debug\n"
        << "show_window = " << (show_window ? "true" : "false") << "\n"
        << "show_fps = " << (show_fps ? "true" : "false") << "\n"
        << "window_size = " << window_size << "\n"
        << "screenshot_button = " << joinStrings(screenshot_button) << "\n"
        << "screenshot_delay = " << screenshot_delay << "\n"
        << "verbose = " << (verbose ? "true" : "false") << "\n\n";

    // Active game
    file << "# Active game profile\n";
    file << "active_game = " << active_game << "\n\n";
    file << "[Games]\n";
    for (auto& kv : game_profiles)
    {
        auto & gp = kv.second;
        file << gp.name << " = "
             << gp.sens << "," << gp.yaw;
        file << "," << gp.pitch;
        if (gp.fovScaled)
            file << ",true," << gp.baseFOV;
        file << "\n";
    }

    file.close();
    return true;
}

const Config::GameProfile& Config::currentProfile() const
{
    auto it = game_profiles.find(active_game);
    if (it != game_profiles.end()) return it->second;
    throw std::runtime_error("Active game profile not found: " + active_game);
}

std::pair<double, double> Config::degToCounts(double degX, double degY, double fovNow) const
{
    const auto& gp = currentProfile();
    double scale = (gp.fovScaled && gp.baseFOV > 1.0) ? (fovNow / gp.baseFOV) : 1.0;

    if (gp.sens == 0.0 || gp.yaw == 0.0 || gp.pitch == 0.0)
        return { 0.0, 0.0 };

    double cx = degX / (gp.sens * gp.yaw * scale);
    double cy = degY / (gp.sens * gp.pitch * scale);
    return { cx, cy };
}