#ifndef MOUSE_H
#define MOUSE_H

#include "target.h"
#include "SerialConnection.h"

class MouseThread
{
private:
    double prev_x, prev_y, prev_velocity_x, prev_velocity_y;
    std::chrono::time_point<std::chrono::steady_clock> prev_time;

    SerialConnection* serial;

    double prediction_interval;
    double max_distance;
    double min_speed_multiplier;
    double max_speed_multiplier;
    double screen_width;
    double screen_height;
    double center_x;
    double center_y;
    double dpi;
    double mouse_sensitivity;
    double fov_x;
    double fov_y;

public:
    MouseThread(int resolution, int dpi, double sensitivity, int fovX, int fovY,
        double minSpeedMultiplier, double maxSpeedMultiplier, double predictionInterval,
        SerialConnection* serialConnection = nullptr);

    void updateConfig(int resolution, double dpi, double sensitivity, double fovX, double fovY,
        double minSpeedMultiplier, double maxSpeedMultiplier, double predictionInterval, bool auto_shoot, float bScope_multiplier);

    std::pair<double, double> predict_target_position(double target_x, double target_y);
    std::pair<double, double> calc_movement(double target_x, double target_y);
    double calculate_speed_multiplier(double distance);
    bool check_target_in_scope(double target_x, double target_y, double target_w, double target_h, double reduction_factor);
    void moveMouse(const Target& target);
    void shootMouse(const Target& target);
};

#endif // MOUSE_H