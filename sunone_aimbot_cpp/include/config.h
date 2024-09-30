#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <vector>

class Config
{
public:
    // Capture
    int detection_resolution;
    bool capture_borders;
    bool capture_cursor;
    // Target
    bool disable_headshot;
    float body_y_offset;
    bool ignore_third_person;

    // Mouse
    int dpi;
    float sensitivity;
    int fovX;
    int fovY;
    float minSpeedMultiplier;
    float maxSpeedMultiplier;
    float predictionInterval;
    
    // arduino
    bool arduino_enable;
    int arduino_baudrate;
    std::string arduino_port;
    bool arduino_16_bit_mouse;

    //Mouse shooting
    bool auto_shoot;
    float bScope_multiplier;

    // AI
    std::string ai_model;
    int engine_image_size;
    float confidence_threshold;

    // Buttons
    std::vector<std::string> button_targeting;
    std::vector<std::string> button_exit;
    std::vector<std::string> button_pause;
    std::vector<std::string> button_reload_config;

    // overlay
    std::vector<std::string> button_open_overlay;

    // Debug window
    bool show_window;
    bool show_fps;
    std::string window_name;
    int window_size;
    std::vector<std::string> screenshot_button;
    bool always_on_top;

    bool loadConfig(const std::string& filename);
    bool saveConfig(const std::string& filename);
private:
    std::vector<std::string> splitString(const std::string& str, char delimiter = ',');
    std::string joinStrings(const std::vector<std::string>& vec, const std::string& delimiter = ",");
};

#endif // CONFIG_H