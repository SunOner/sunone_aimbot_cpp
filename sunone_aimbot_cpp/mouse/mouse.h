#ifndef MOUSE_H
#define MOUSE_H

#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>
#include "../modules/eigen/include/Eigen/Dense"
#include <shared_mutex>
#include <memory>

#include "AimbotTarget.h"
#include "SerialConnection.h"
#include "ghub.h"
#include "InputMethod.h"

// 2D Kalman Filter for position and velocity tracking
class KalmanFilter2D
{
private:
    Eigen::Matrix4d A;             // State transition matrix
    Eigen::Matrix<double, 2, 4> H; // Measurement matrix
    Eigen::Matrix4d Q;             // Process noise covariance
    Eigen::Matrix2d R;             // Measurement noise covariance
    Eigen::Matrix4d P;             // Estimation error covariance
    Eigen::Vector4d x;             // State vector [x, y, vx, vy]

public:
    KalmanFilter2D(double process_noise_q, double measurement_noise_r, double initial_p);
    void predict(double dt);
    void update(const Eigen::Vector2d &measurement);
    Eigen::Vector4d getState() const { return x; }
    void reset();
    void updateParameters(double process_noise_q, double measurement_noise_r, double estimation_error_p);
};

// 2D PID Controller
class PIDController2D
{
private:
    double kp, ki, kd;
    double max_output, min_output;
    Eigen::Vector2d previous_error;
    Eigen::Vector2d integral;
    Eigen::Vector2d last_derivative;
    std::chrono::steady_clock::time_point last_time;
    bool first_run;

public:
    PIDController2D(double kp, double ki, double kd, double max_output, double min_output);
    Eigen::Vector2d calculate(const Eigen::Vector2d &setpoint, const Eigen::Vector2d &current_pos);
    void reset();
    void updateParameters(double kp, double ki, double kd, double max_output, double min_output);
};

class MouseThread
{
private:
    std::unique_ptr<KalmanFilter2D> kalman_filter;
    std::unique_ptr<PIDController2D> pid_controller;

    // 기존 포인터 제거
    // SerialConnection *serial;
    // GhubMouse *gHub;

    // InputMethod 사용으로 변경
    std::unique_ptr<InputMethod> input_method;

    double screen_width;
    double screen_height;
    double dpi;
    double mouse_sensitivity;
    double fov_x;
    double fov_y;
    double center_x;
    double center_y;
    bool auto_shoot;
    float bScope_multiplier;

    std::chrono::steady_clock::time_point last_target_time;
    std::atomic<bool> target_detected{false};
    std::atomic<bool> mouse_pressed{false};

    // Simplified target tracking
    AimbotTarget *current_target;

    double calculateTargetDistance(const AimbotTarget &target) const;
    AimbotTarget *findClosestTarget(const std::vector<AimbotTarget> &targets) const;

public:
    MouseThread(int resolution, int dpi, double sensitivity, int fovX, int fovY,
                double kp, double ki, double kd, double pid_max_output, double pid_min_output,
                double process_noise_q, double measurement_noise_r, double estimation_error_p,
                bool auto_shoot, float bScope_multiplier,
                SerialConnection *serialConnection = nullptr,
                GhubMouse *gHub = nullptr);

    void updateConfig(int resolution, int dpi, double sensitivity, int fovX, int fovY,
                      double kp, double ki, double kd, double pid_max_output, double pid_min_output,
                      double process_noise_q, double measurement_noise_r, double estimation_error_p,
                      bool auto_shoot, float bScope_multiplier);

    Eigen::Vector2d predictTargetPosition(double target_x, double target_y);
    Eigen::Vector2d calculateMovement(const Eigen::Vector2d &target_pos);
    bool checkTargetInScope(double target_x, double target_y, double target_w, double target_h, double reduction_factor);
    void moveMouse(const AimbotTarget &target);
    void pressMouse(const AimbotTarget &target);
    void releaseMouse();
    void resetPrediction();
    void checkAndResetPredictions();
    void applyRecoilCompensation(float strength);

    std::mutex input_method_mutex;

    // 단일 setter 메서드만 유지
    void setInputMethod(std::unique_ptr<InputMethod> new_method);
};

#endif // MOUSE_H