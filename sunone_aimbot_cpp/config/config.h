#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <vector>

class Config
{
public:
    // Capture
    std::string capture_method;
    int detection_resolution;
    int capture_fps;
    int monitor_idx;
    bool circle_mask;
    bool capture_borders;
    bool capture_cursor;
    std::string virtual_camera_name;

    // Target
    bool disable_headshot;
    float body_y_offset;
    bool ignore_third_person;
    bool shooting_range_targets;
    bool auto_aim;

    // Mouse
    int dpi;
    float sensitivity;
    int fovX;
    int fovY;
    float minSpeedMultiplier;
    float maxSpeedMultiplier;
    float predictionInterval;
    std::string input_method;   // "WIN32", "GHUB", "ARDUINO"
    
    // Arduino
    int arduino_baudrate;
    std::string arduino_port;
    bool arduino_16_bit_mouse;
    bool arduino_enable_keys;

    //Mouse shooting
    bool auto_shoot;
    float bScope_multiplier;

    // AI
    std::string ai_model;
    float confidence_threshold;
    float nms_threshold;
    int max_detections;
    std::string postprocess;

    // Optical Flow
    bool enable_optical_flow;
    bool draw_optical_flow;
    int draw_optical_flow_steps;
    float optical_flow_alpha_cpu;
    double optical_flow_magnitudeThreshold;

    // Buttons
    std::vector<std::string> button_targeting;
    std::vector<std::string> button_exit;
    std::vector<std::string> button_pause;
    std::vector<std::string> button_reload_config;
    std::vector<std::string> button_open_overlay;

    // Overlay
    int overlay_opacity;

    // Custom Classes
    int class_player;                  // 0
    int class_bot;                     // 1
    int class_weapon;                  // 2
    int class_outline;                 // 3
    int class_dead_body;               // 4
    int class_hideout_target_human;    // 5
    int class_hideout_target_balls;    // 6
    int class_head;                    // 7
    int class_smoke;                   // 8
    int class_fire;                    // 9
    int class_third_person;            // 10
    
    // Debug
    bool show_window;
    bool show_fps;
    std::string window_name;
    int window_size;
    std::vector<std::string> screenshot_button;
    int screenshot_delay;
    bool always_on_top;
    bool verbose;

    bool loadConfig(const std::string& filename);
    bool saveConfig(const std::string& filename);
    std::string joinStrings(const std::vector<std::string>& vec, const std::string& delimiter = ",");
private:
    std::vector<std::string> splitString(const std::string& str, char delimiter = ',');
};

#endif // CONFIG_H