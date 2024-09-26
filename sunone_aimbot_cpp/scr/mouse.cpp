#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <cmath>
#include <algorithm>
#include <chrono>

#include "mouse.h"
#include "capture.h"
#include "SerialConnection.h"
#include "sunone_aimbot_cpp.h"

using namespace std;

extern std::atomic<bool> aiming;
extern std::mutex configMutex;

MouseThread::MouseThread(
    int resolution,
    int dpi,
    double sensitivity,
    int fovX,
    int fovY,
    double minSpeedMultiplier,
    double maxSpeedMultiplier,
    double predictionInterval,
    bool auto_shoot,
    float bScope_multiplier,
    SerialConnection* serialConnection)
    :
    screen_width(resolution),
    screen_height(resolution),
    dpi(dpi),
    mouse_sensitivity(sensitivity),
    fov_x(fovX),
    fov_y(fovY),
    min_speed_multiplier(minSpeedMultiplier),
    max_speed_multiplier(maxSpeedMultiplier),
    prediction_interval(predictionInterval),
    auto_shoot(auto_shoot),
    bScope_multiplier(bScope_multiplier),
    prev_x(0),
    prev_y(0),
    prev_velocity_x(0),
    prev_velocity_y(0),
    max_distance(std::sqrt(resolution* resolution + resolution * resolution) / 2),
    center_x(resolution / 2),
    center_y(resolution / 2),
    serial(serialConnection)
{
}

void MouseThread::updateConfig(int resolution, double dpi, double sensitivity, int fovX, int fovY,
    double minSpeedMultiplier, double maxSpeedMultiplier, double predictionInterval,
    bool auto_shoot, float bScope_multiplier)
{
    this->screen_width = resolution;
    this->screen_height = resolution;
    this->dpi = dpi;
    this->mouse_sensitivity = sensitivity;
    this->fov_x = fovX;
    this->fov_y = fovY;
    this->min_speed_multiplier = minSpeedMultiplier;
    this->max_speed_multiplier = maxSpeedMultiplier;
    this->prediction_interval = predictionInterval;
    this->auto_shoot = auto_shoot;
    this->bScope_multiplier = bScope_multiplier;
    this->center_x = resolution / 2;
    this->center_y = resolution / 2;
    this->max_distance = std::sqrt(resolution * resolution + resolution * resolution) / 2;
}

std::pair<double, double> MouseThread::predict_target_position(double target_x, double target_y)
{
    auto current_time = std::chrono::steady_clock::now();
    if (prev_time.time_since_epoch().count() == 0)
    {
        prev_time = current_time;
        prev_x = target_x;
        prev_y = target_y;

        return { target_x, target_y };
    }

    double delta_time = std::chrono::duration<double>(current_time - prev_time).count();

    double velocity_x = (target_x - prev_x) / delta_time;
    double velocity_y = (target_y - prev_y) / delta_time;

    double acceleration_x = (velocity_x - prev_velocity_x) / delta_time;
    double acceleration_y = (velocity_y - prev_velocity_y) / delta_time;

    double prediction_interval_scaled = delta_time * prediction_interval;

    double predicted_x = target_x + velocity_x * prediction_interval_scaled + 0.5 * acceleration_x * (prediction_interval_scaled * prediction_interval_scaled);
    double predicted_y = target_y + velocity_y * prediction_interval_scaled + 0.5 * acceleration_y * (prediction_interval_scaled * prediction_interval_scaled);

    prev_x = target_x;
    prev_y = target_y;
    prev_velocity_x = velocity_x;
    prev_velocity_y = velocity_y;
    prev_time = current_time;

    return { predicted_x, predicted_y };
}

double MouseThread::calculate_speed_multiplier(double distance)
{
    double normalized_distance = std::min(distance / max_distance, 1.0);
    double speed_multiplier = min_speed_multiplier + (max_speed_multiplier - min_speed_multiplier) * (1 - normalized_distance);
    return speed_multiplier;
}

std::pair<double, double> MouseThread::calc_movement(double target_x, double target_y)
{
    double offset_x = target_x - center_x;
    double offset_y = target_y - center_y;

    double distance = std::sqrt(offset_x * offset_x + offset_y * offset_y);

    double speed_multiplier = calculate_speed_multiplier(distance);
    
    double degrees_per_pixel_x = fov_x / screen_width;
    double degrees_per_pixel_y = fov_y / screen_height;
    
    double mouse_move_x = offset_x * degrees_per_pixel_x;
    double mouse_move_y = offset_y * degrees_per_pixel_y;

    double move_x = (mouse_move_x / 360) * (dpi * (1 / mouse_sensitivity)) * speed_multiplier;
    double move_y = (mouse_move_y / 360) * (dpi * (1 / mouse_sensitivity)) * speed_multiplier;
    
    return { move_x, move_y };
}

bool MouseThread::check_target_in_scope(double target_x, double target_y, double target_w, double target_h, double reduction_factor)
{
    double reduced_w = target_w * reduction_factor / 2;
    double reduced_h = target_h * reduction_factor / 2;

    double x1 = target_x - reduced_w;
    double x2 = target_x + reduced_w;
    double y1 = target_y - reduced_h;
    double y2 = target_y + reduced_h;

    return center_x > x1 && center_x < x2 && center_y > y1 && center_y < y2;
}

void MouseThread::moveMouse(const Target& target)
{
    auto predicted_position = predict_target_position(target.x + target.w / 2, target.y + target.h / 2);
    auto movement = calc_movement(predicted_position.first, predicted_position.second);

    if (config.arduino_enable && serial)
    {
        serial->move(static_cast<INT>(movement.first), static_cast<INT>(movement.second));
    }
    else
    {
        INPUT input = { 0 };
        input.type = INPUT_MOUSE;
        input.mi.dx = static_cast<INT>(movement.first);
        input.mi.dy = static_cast<INT>(movement.second);
        input.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_VIRTUALDESK;

        SendInput(1, &input, sizeof(INPUT));
    }
}

void MouseThread::shootMouse(const Target& target)
{
    auto bScope = check_target_in_scope(target.x, target.y, target.w, target.w, config.bScope_multiplier);
    if (bScope)
    {
        if (config.arduino_enable && serial)
        {
            serial->press();
        }
    }
    else
    {
        serial->release();
    }
}