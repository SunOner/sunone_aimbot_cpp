#ifndef CONFIG_H
#define CONFIG_H

#include <string>

class Config
{
public:
    int detection_window_width;
    int detection_window_height;

    bool disable_headshot;
    float body_y_offset;
    
    int dpi;
    float sensitivity;
    int fovX;
    int fovY;
    float minSpeedMultiplier;
    float maxSpeedMultiplier;
    float predictionInterval;
    
    bool arduino_enable;
    int arduino_baudrate;
    std::string arduino_port;

    std::string ai_model;
    int engine_image_size;
    float confidence_threshold;

    bool loadConfig(const std::string& filename);
};

#endif // CONFIG_H