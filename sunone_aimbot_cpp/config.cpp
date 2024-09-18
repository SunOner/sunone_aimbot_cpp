#include <fstream>
#include <sstream>
#include <iostream>

#include "config.h"

// TODO REWRITE
bool Config::loadConfig(const std::string& filename)
{
    std::ifstream infile(filename);
    if (!infile.is_open())
    {
        std::cerr << "Error with config file: " << filename << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(infile, line)) {
        auto comment_pos = line.find('#');
        if (comment_pos != std::string::npos)
            line = line.substr(0, comment_pos);

        line.erase(0, line.find_first_not_of(" \t"));
        if (line.empty())
            continue;

        auto equal_pos = line.find('=');
        if (equal_pos == std::string::npos)
            continue;

        std::string key = line.substr(0, equal_pos);
        std::string value = line.substr(equal_pos + 1);

        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);

        if (key == "disable_headshot") {
            disable_headshot = (value == "true" || value == "1");
        }
        else if (key == "body_y_offset") {
            body_y_offset = std::stof(value);
        }
        else if (key == "dpi") {
            dpi = std::stof(value);
        }
        else if (key == "sensitivity") {
            sensitivity = std::stof(value);
        }
        else if (key == "fovX") {
            fovX = std::stof(value);
        }
        else if (key == "fovY") {
            fovY = std::stof(value);
        }
        else if (key == "minSpeedMultiplier") {
            minSpeedMultiplier = std::stof(value);
        }
        else if (key == "maxSpeedMultiplier") {
            maxSpeedMultiplier = std::stof(value);
        }
        else if (key == "predictionInterval") {
            predictionInterval = std::stof(value);
        }
        else if (key == "detection_window_width") {
            detection_window_width = std::stoi(value);
        }
        else if (key == "detection_window_height") {
            detection_window_height = std::stoi(value);
        }
        else if (key == "engine_image_size") {
            engine_image_size = std::stoi(value);
        }
        else if (key == "confidence_threshold") {
            confidence_threshold = std::stof(value);
        }
        else {
            std::cerr << "Unknown option: " << key << std::endl;
        }
    }

    infile.close();
    return true;
}