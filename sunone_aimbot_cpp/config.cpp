#include <iostream>
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

        // Detection window
        detection_window_width = pt.get<int>("detection_window_width", 0);
        detection_window_height = pt.get<int>("detection_window_height", 0);

        // Target
        disable_headshot = pt.get<bool>("disable_headshot", false);
        body_y_offset = pt.get<float>("body_y_offset", 0.0f);
        
        // Mouse
        dpi = pt.get<int>("dpi", 0);
        sensitivity = pt.get<float>("sensitivity", 0.0f);
        fovX = pt.get<int>("fovX", 0);
        fovY = pt.get<int>("fovY", 0);
        minSpeedMultiplier = pt.get<float>("minSpeedMultiplier", 0.0f);
        maxSpeedMultiplier = pt.get<float>("maxSpeedMultiplier", 0.0f);
        predictionInterval = pt.get<float>("predictionInterval", 0.0f);
        
        // arduino
        arduino_enable = pt.get<bool>("arduino_enable", "false");
        arduino_baudrate = pt.get<int>("arduino_baudrate", 9600);
        arduino_port = pt.get<std::string>("arduino_port", "COM0");

        // AI
        ai_model = pt.get<std::string>("ai_model", "sunxds_0.5.6.engine");
        engine_image_size = pt.get<int>("engine_image_size", 0);
        confidence_threshold = pt.get<float>("confidence_threshold", 0.0f);

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