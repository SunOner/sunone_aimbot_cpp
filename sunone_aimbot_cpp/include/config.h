#ifndef CONFIG_H
#define CONFIG_H

#include <string>

class Config
{
public:
    // Detection window
    int detection_resolution;

    // Target
    bool disable_headshot;
    float body_y_offset;
    
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

    // Debug window
    bool show_window;
    bool show_fps;
    std::string window_name;
    int window_size;

    bool loadConfig(const std::string& filename);
};

#endif // CONFIG_H