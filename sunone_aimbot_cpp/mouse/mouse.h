#ifndef MOUSE_H
#define MOUSE_H

#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include "AimbotTarget.h"
#include "SerialConnection.h"
#include "ghub.h"

class MouseThread
{
private:
    double prev_x, prev_y, prev_velocity_x, prev_velocity_y;
    std::chrono::time_point<std::chrono::steady_clock> prev_time;

    SerialConnection* serial;
    GhubMouse* gHub;

    double screen_width;
    double screen_height;
    double dpi;
    double prediction_interval;
    double mouse_sensitivity;
    double fov_x;
    double fov_y;
    double max_distance;
    double min_speed_multiplier;
    double max_speed_multiplier;
    double center_x;
    double center_y;
    bool auto_shoot;
    float bScope_multiplier;
    // KMBOX mode flag (useKmbox replaces the earlier kmboxMode)
    bool useKmbox = false;

    std::chrono::steady_clock::time_point last_target_time;
    std::atomic<bool> target_detected{ false };
    std::atomic<bool> mouse_pressed{ false };

public:
    MouseThread(int resolution, int dpi, double sensitivity, int fovX, int fovY,
        double minSpeedMultiplier, double maxSpeedMultiplier, double predictionInterval,
        bool auto_shoot, float bScope_multiplier,
        SerialConnection* serialConnection = nullptr,
        GhubMouse* gHub = nullptr);

    void updateConfig(int resolution, double dpi, double sensitivity, int fovX, int fovY,
        double minSpeedMultiplier, double maxSpeedMultiplier, double predictionInterval,
        bool auto_shoot, float bScope_multiplier);

    std::pair<double, double> predict_target_position(double target_x, double target_y);
    std::pair<double, double> calc_movement(double target_x, double target_y);
    double calculate_speed_multiplier(double distance);
    bool check_target_in_scope(double target_x, double target_y, double target_w, double target_h, double reduction_factor);
    void moveMouse(const AimbotTarget& target);
    void pressMouse(const AimbotTarget& target);
    void releaseMouse();
    void resetPrediction();
    void checkAndResetPredictions();
    std::mutex input_method_mutex;
    void setSerialConnection(SerialConnection* newSerial);
    void setGHubMouse(GhubMouse* newGHub);
};

#endif // MOUSE_H
