#include "mouse.h"
#include <cmath>
#include <algorithm>
#include <chrono>
#include "capture.h"
#include "SerialConnection.h"

SerialConnection serial("COM6", 115200);

MouseThread::MouseThread(double screenWidth, double screenHeight, double dpi, double sensitivity, double fovX, double fovY, double minSpeedMultiplier, double maxSpeedMultiplier, double predictionInterval)
    : screen_width(screenWidth), screen_height(screenHeight), dpi(dpi), mouse_sensitivity(sensitivity), fov_x(fovX), fov_y(fovY),
    min_speed_multiplier(minSpeedMultiplier), max_speed_multiplier(maxSpeedMultiplier), prediction_interval(predictionInterval),
    prev_x(0), prev_y(0), prev_velocity_x(0), prev_velocity_y(0), max_distance(std::sqrt(screenWidth* screenWidth + screenHeight * screenHeight) / 2)
{
    center_x = screenWidth / 2;
    center_y = screenHeight / 2;
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

#undef min

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

template <typename T>
T clamp(const T& value, const T& low, const T& high)
{
    return (value < low) ? low : (high < value) ? high : value;
}

void MouseThread::moveMouseToTarget(const Target& target)
{
    auto predicted_position = predict_target_position(target.x + target.w / 2, target.y + target.h / 2);
    
    auto movement = calc_movement(predicted_position.first, predicted_position.second);
    
    const double MAX_MOVE = 50;
    double move_x = clamp(movement.first, -MAX_MOVE, MAX_MOVE);
    double move_y = clamp(movement.second, -MAX_MOVE, MAX_MOVE);
    
    int dx = static_cast<INT>(move_x * 65535.0f / screen_width);
    int dy = static_cast<INT>(move_y * 65535.0f / screen_height);

    serial.move(dx, dy);

    //INPUT input = { 0 };
    //input.type = INPUT_MOUSE;
    //input.mi.dx = dx;
    //input.mi.dy = dy;
    //input.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_VIRTUALDESK;
    //
    //SendInput(1, &input, sizeof(INPUT));
}