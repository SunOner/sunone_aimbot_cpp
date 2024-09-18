#ifndef CONFIG_H
#define CONFIG_H

#include <string>

class Config
{
public:
    int detection_window_width;
    int detection_window_height;
    int dpi;
    float sensitivity;
    int fovX;
    int fovY;
    bool disable_headshot;
    float body_y_offset;
    float minSpeedMultiplier;
    float maxSpeedMultiplier;
    float predictionInterval;
    int engine_image_size;
    float confidence_threshold;

    bool loadConfig(const std::string& filename);
};

#endif // CONFIG_H