#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <cmath>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <atomic>
#include <immintrin.h> // For AVX2 intrinsics

#include "mouse.h"
#include "capture.h"
#include "SerialConnection.h"
#include "sunone_aimbot_cpp.h"
#include "ghub.h"

extern std::atomic<bool> aiming;
extern std::mutex configMutex;

// KalmanFilter2D Implementation
KalmanFilter2D::KalmanFilter2D(double process_noise_q, double measurement_noise_r, double initial_p)
{
    // Initialize state transition matrix (constant velocity model)
    A = Eigen::Matrix4d::Identity();

    // Initialize measurement matrix
    H = Eigen::Matrix<double, 2, 4>::Zero();
    H(0, 0) = 1.0; // x position measurement
    H(1, 1) = 1.0; // y position measurement

    // Initialize noise covariances
    Q = Eigen::Matrix4d::Identity() * process_noise_q;
    R = Eigen::Matrix2d::Identity() * measurement_noise_r;

    // Initialize error covariance
    P = Eigen::Matrix4d::Identity() * initial_p;

    // Initialize state vector
    x = Eigen::Vector4d::Zero();
}

void KalmanFilter2D::predict(double dt)
{
    // Update state transition matrix with dt only when needed
    A(0, 2) = dt; // x position affected by x velocity
    A(1, 3) = dt; // y position affected by y velocity

    // Optimized state prediction using direct calculation instead of full matrix multiplication
    // Original: x = A * x;
    // Optimized calculation for specific Kalman filter structure
    double new_x0 = x(0) + dt * x(2);
    double new_x1 = x(1) + dt * x(3);
    x(0) = new_x0;
    x(1) = new_x1;
    // Velocity elements remain unchanged
    // x(2) = x(2);
    // x(3) = x(3);

    // Optimized error covariance calculation for specific Kalman filter structure
    // Exploit the sparsity of A matrix (mostly zeros except diagonal and dt values)

    // Original: P = A * P * A.transpose() + Q;

    // Create temporary storage for intermediate results
    Eigen::Matrix4d temp = Eigen::Matrix4d::Zero();

    // Manually compute A*P by exploiting the structure of A
    // A has the form:
    // [1 0 dt 0]
    // [0 1 0 dt]
    // [0 0 1 0]
    // [0 0 0 1]

    // First row of result = 1*P.row(0) + dt*P.row(2)
    temp.row(0) = P.row(0) + dt * P.row(2);

    // Second row = 1*P.row(1) + dt*P.row(3)
    temp.row(1) = P.row(1) + dt * P.row(3);

    // Third row = P.row(2) (unchanged)
    temp.row(2) = P.row(2);

    // Fourth row = P.row(3) (unchanged)
    temp.row(3) = P.row(3);

    // Now compute temp * A.transpose() by exploiting structure again
    Eigen::Matrix4d new_P = Eigen::Matrix4d::Zero();

    // First column = 1*temp.col(0) + 0*temp.col(1) + 0*temp.col(2) + 0*temp.col(3) = temp.col(0)
    new_P.col(0) = temp.col(0);

    // Second column = 0*temp.col(0) + 1*temp.col(1) + 0*temp.col(2) + 0*temp.col(3) = temp.col(1)
    new_P.col(1) = temp.col(1);

    // Third column = dt*temp.col(0) + 0*temp.col(1) + 1*temp.col(2) + 0*temp.col(3) = dt*temp.col(0) + temp.col(2)
    new_P.col(2) = dt * temp.col(0) + temp.col(2);

    // Fourth column = 0*temp.col(0) + dt*temp.col(1) + 0*temp.col(2) + 1*temp.col(3) = dt*temp.col(1) + temp.col(3)
    new_P.col(3) = dt * temp.col(1) + temp.col(3);

    // Add process noise
    P = new_P + Q;
}

void KalmanFilter2D::update(const Eigen::Vector2d &measurement)
{
    // Step 1: Calculate S = HPH' + R directly without forming intermediate HP
    // Since H has a special structure [I_2x2 0_2x2], we can extract the upper-left 2x2 block of P
    Eigen::Matrix2d Pxy = P.block<2, 2>(0, 0); // Upper-left 2x2 block of P
    Eigen::Matrix2d S = Pxy + R;               // S = H*P*H' + R

    // Step 2: Calculate S inverse for 2x2 matrix directly (much faster than using .inverse())
    double det_S = S(0, 0) * S(1, 1) - S(0, 1) * S(1, 0);
    double inv_det_S = 1.0 / det_S;

    Eigen::Matrix2d S_inv;
    S_inv(0, 0) = S(1, 1) * inv_det_S;
    S_inv(0, 1) = -S(0, 1) * inv_det_S;
    S_inv(1, 0) = -S(1, 0) * inv_det_S;
    S_inv(1, 1) = S(0, 0) * inv_det_S;

    // Step 3: Calculate Kalman gain efficiently by utilizing H's structure
    // K = P*H'*S_inv where H = [I 0]
    Eigen::Matrix<double, 4, 2> K;
    K.block<2, 2>(0, 0) = P.block<2, 2>(0, 0) * S_inv;
    K.block<2, 2>(2, 0) = P.block<2, 2>(2, 0) * S_inv;

    // Step 4: Calculate innovation
    Eigen::Vector2d innovation = measurement - Eigen::Vector2d(x(0), x(1)); // H*x = [x(0), x(1)]

    // Step 5: Update state vector
    x += K * innovation;

    // Step 6: Update error covariance using simplified Joseph form
    // Regular simplified form: P = (I - KH)P

    // Since H = [I 0], KH has the structure:
    // KH = [ K(0:1,0:1)    0 ]
    //      [ K(2:3,0:1)    0 ]

    // Therefore I-KH has the structure:
    // I-KH = [ I-K(0:1,0:1)    0 ]
    //        [ -K(2:3,0:1)     I ]

    // Create the I-KH matrix more efficiently
    Eigen::Matrix4d I_KH = Eigen::Matrix4d::Identity();
    I_KH.block<2, 2>(0, 0) = Eigen::Matrix2d::Identity() - K.block<2, 2>(0, 0);
    I_KH.block<2, 2>(2, 0) = -K.block<2, 2>(2, 0);

    // Compute (I-KH)P
    // We can do this more efficiently by considering the block structure
    Eigen::Matrix4d new_P;

    // First two rows of new_P
    new_P.block<2, 4>(0, 0) = I_KH.block<2, 2>(0, 0) * P.block<2, 4>(0, 0) +
                              I_KH.block<2, 2>(0, 2) * P.block<2, 4>(2, 0);

    // Last two rows of new_P
    new_P.block<2, 4>(2, 0) = I_KH.block<2, 2>(2, 0) * P.block<2, 4>(0, 0) +
                              I_KH.block<2, 2>(2, 2) * P.block<2, 4>(2, 0);

    // To maintain symmetry and ensure numerical stability, we should apply:
    // P = 0.5 * (P + P.transpose())
    P = 0.5 * (new_P + new_P.transpose());
}

void KalmanFilter2D::reset()
{
    x = Eigen::Vector4d::Zero();
    P = Eigen::Matrix4d::Identity() * P(0, 0); // Reset to initial error covariance
}

void KalmanFilter2D::updateParameters(double process_noise_q, double measurement_noise_r, double estimation_error_p)
{
    Q = Eigen::Matrix4d::Identity() * process_noise_q;
    R = Eigen::Matrix2d::Identity() * measurement_noise_r;
    P = Eigen::Matrix4d::Identity() * estimation_error_p;
}

// PIDController2D Implementation
PIDController2D::PIDController2D(double kp, double ki, double kd, double max_output, double min_output)
    : kp(kp), ki(ki), kd(kd), max_output(max_output), min_output(min_output), first_run(true)
{
    integral = Eigen::Vector2d::Zero();
    previous_error = Eigen::Vector2d::Zero();
    last_derivative = Eigen::Vector2d::Zero();
    last_time = std::chrono::steady_clock::now();
}

Eigen::Vector2d PIDController2D::calculate(const Eigen::Vector2d &setpoint, const Eigen::Vector2d &current_pos)
{
    const auto current_time = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(current_time - last_time).count();

    // Ensure reasonable dt value
    if (first_run || dt <= 0.0 || dt > 0.1)
    {
        dt = 0.001; // Small initial or reset dt
        first_run = false;
    }

    // Compute error term
    const Eigen::Vector2d error = setpoint - current_pos;

    // More efficient integral term calculation with anti-windup
    // Only accumulate integral when within reasonable range
    const double max_integral = max_output / ki; // Prevent windup

    for (int i = 0; i < 2; i++)
    {
        integral[i] += error[i] * dt;
        // Prevent integral windup
        integral[i] = std::clamp(integral[i], -max_integral, max_integral);
    }

    // Compute derivative with improved filtering
    Eigen::Vector2d derivative;
    if (dt > 0)
    {
        // Using backward differences for derivative
        derivative = (error - previous_error) / dt;

        // Improved low-pass filter with adjusted coefficients
        // Alpha = 0.2 to filter out high-frequency noise but maintain responsiveness
        constexpr double alpha = 0.2;
        last_derivative = derivative * alpha + last_derivative * (1.0 - alpha);
    }
    else
    {
        derivative = last_derivative;
    }

    // Calculate PID output components individually for better control
    Eigen::Vector2d p_term = kp * error;
    Eigen::Vector2d i_term = ki * integral;
    Eigen::Vector2d d_term = kd * last_derivative;

    // Combine terms
    Eigen::Vector2d output = p_term + i_term + d_term;

    // Apply output constraints
    for (int i = 0; i < 2; i++)
    {
        output[i] = std::clamp(output[i], min_output, max_output);
    }

    // Update state
    previous_error = error;
    last_time = current_time;

    return output;
}

void PIDController2D::reset()
{
    integral = Eigen::Vector2d::Zero();
    previous_error = Eigen::Vector2d::Zero();
    last_derivative = Eigen::Vector2d::Zero();
    first_run = true;
}

void PIDController2D::updateParameters(double kp, double ki, double kd, double max_output, double min_output)
{
    this->kp = kp;
    this->ki = ki;
    this->kd = kd;
    this->max_output = max_output;
    this->min_output = min_output;
}

// MouseThread Implementation
MouseThread::MouseThread(
    int resolution,
    int dpi,
    double sensitivity,
    int fovX,
    int fovY,
    double kp,
    double ki,
    double kd,
    double pid_max_output,
    double pid_min_output,
    double process_noise_q,
    double measurement_noise_r,
    double estimation_error_p,
    bool auto_shoot,
    float bScope_multiplier,
    SerialConnection *serialConnection,
    GhubMouse *gHub) : screen_width(static_cast<double>(resolution * 16) / 9.0),
                       screen_height(static_cast<double>(resolution)),
                       dpi(static_cast<double>(dpi)),
                       mouse_sensitivity(sensitivity),
                       fov_x(static_cast<double>(fovX)),
                       fov_y(static_cast<double>(fovY)),
                       center_x(screen_width / 2),
                       center_y(screen_height / 2),
                       auto_shoot(auto_shoot),
                       bScope_multiplier(bScope_multiplier),
                       current_target(nullptr)
{
    // Kalman 필터와 PID 컨트롤러 초기화 (기존 코드와 동일)
    kalman_filter = std::make_unique<KalmanFilter2D>(process_noise_q, measurement_noise_r, estimation_error_p);
    pid_controller = std::make_unique<PIDController2D>(kp, ki, kd, pid_max_output, pid_min_output);

    // InputMethod 초기화 (새로운 코드)
    if (serialConnection && serialConnection->isOpen())
    {
        input_method = std::make_unique<SerialInputMethod>(serialConnection);
    }
    else if (gHub)
    {
        input_method = std::make_unique<GHubInputMethod>(gHub);
    }
    else
    {
        input_method = std::make_unique<Win32InputMethod>();
    }
}

void MouseThread::updateConfig(
    int resolution,
    int dpi,
    double sensitivity,
    int fovX,
    int fovY,
    double kp,
    double ki,
    double kd,
    double pid_max_output,
    double pid_min_output,
    double process_noise_q,
    double measurement_noise_r,
    double estimation_error_p,
    bool auto_shoot,
    float bScope_multiplier)
{
    this->screen_width = resolution;
    this->screen_height = resolution;
    this->dpi = dpi;
    this->mouse_sensitivity = sensitivity;
    this->fov_x = fovX;
    this->fov_y = fovY;
    this->auto_shoot = auto_shoot;
    this->bScope_multiplier = bScope_multiplier;
    this->center_x = resolution / 2;
    this->center_y = resolution / 2;

    kalman_filter->updateParameters(process_noise_q, measurement_noise_r, estimation_error_p);
    pid_controller->updateParameters(kp, ki, kd, pid_max_output, pid_min_output);
}

Eigen::Vector2d MouseThread::predictTargetPosition(double target_x, double target_y)
{
    // Cache current time and reuse to avoid multiple system calls
    const auto current_time = std::chrono::steady_clock::now();

    // Update target detection timestamp
    last_target_time = current_time;
    target_detected = true;

    // Create measurement vector
    const Eigen::Vector2d measurement(target_x, target_y);

    // Calculate time delta more efficiently
    static auto last_prediction_time = current_time;
    const double dt = std::chrono::duration<double>(current_time - last_prediction_time).count();
    last_prediction_time = current_time;

    // Reset filter if time delta is too large (likely a hiccup or pause)
    if (dt > 0.1)
    { // Reduced from 0.5 to 0.1 for more responsive tracking
        kalman_filter->reset();
        // Use a small non-zero dt to avoid division issues
        const double safe_dt = 0.001;

        // Predict and update with fixed dt
        kalman_filter->predict(safe_dt);
        kalman_filter->update(measurement);

        // Return the measurement directly after reset
        return measurement;
    }

    // Normal prediction path - use actual dt
    kalman_filter->predict(dt);
    kalman_filter->update(measurement);

    // Get predicted state vector
    const Eigen::Vector4d state = kalman_filter->getState();

    // Extract position components
    return Eigen::Vector2d(state[0], state[1]);
}

Eigen::Vector2d MouseThread::calculateMovement(const Eigen::Vector2d &target_pos)
{
#if defined(__AVX2__)
    // AVX2 optimized version - uses 256-bit registers for more efficient calculation
    // Prepare data for AVX - load 4 doubles at once
    // Format: [center_x, center_y, target_x, target_y]
    __m256d positions = _mm256_setr_pd(center_x, center_y, target_pos[0], target_pos[1]);

    // Calculate error: target - current
    // Use AVX2 shuffle to create [target_x, target_y, target_x, target_y]
    __m256d targets = _mm256_permute4x64_pd(positions, _MM_SHUFFLE(3, 2, 3, 2));
    // Use AVX2 shuffle to create [center_x, center_y, center_x, center_y]
    __m256d centers = _mm256_permute4x64_pd(positions, _MM_SHUFFLE(1, 0, 1, 0));

    // Calculate error: targets - centers
    __m256d errors = _mm256_sub_pd(targets, centers);

    // Extract the 2D error vector to pass to PID
    double error_array[4];
    _mm256_storeu_pd(error_array, errors);
    Eigen::Vector2d error_vector(error_array[0], error_array[1]);

    // Calculate PID output - this could also be AVX2 optimized in the future
    Eigen::Vector2d pid_output = pid_controller->calculate(target_pos, Eigen::Vector2d(center_x, center_y));

    // Scale output with FOV and sensitivity using AVX2
    // [pid_x, pid_y, 0, 0]
    __m256d pid_vec = _mm256_setr_pd(pid_output[0], pid_output[1], 0.0, 0.0);

    // [fov_x/width, fov_y/height, 0, 0]
    __m256d fov_scale = _mm256_setr_pd(
        fov_x / screen_width,
        fov_y / screen_height,
        0.0,
        0.0);

    // [dpi/sens/360, dpi/sens/360, 0, 0]
    __m256d sens_scale = _mm256_set1_pd(dpi * (1.0 / mouse_sensitivity) / 360.0);

    // Apply both scaling factors: pid * fov_scale * sens_scale
    __m256d result = _mm256_mul_pd(_mm256_mul_pd(pid_vec, fov_scale), sens_scale);

    // Extract final result
    double output[4];
    _mm256_storeu_pd(output, result);

    return Eigen::Vector2d(output[0], output[1]);
#else
    // Pre-compute constants for better performance (fallback to SSE)
    static const __m128d sensitivity_scale = _mm_set1_pd(dpi * (1.0 / mouse_sensitivity) / 360.0);

    // Load current and target positions with SIMD
    __m128d current = _mm_set_pd(center_y, center_x);
    __m128d target = _mm_load_pd(target_pos.data());

    // Calculate PID output
    Eigen::Vector2d pid_output = pid_controller->calculate(target_pos, Eigen::Vector2d(center_x, center_y));

    // Convert to mouse movement with SIMD optimization
    __m128d fov_factors = _mm_set_pd(fov_y / screen_height, fov_x / screen_width);
    __m128d pid_vector = _mm_load_pd(pid_output.data());

    // Combine operations to reduce instruction count
    __m128d movement = _mm_mul_pd(_mm_mul_pd(pid_vector, fov_factors), sensitivity_scale);

    // Store result
    Eigen::Vector2d result;
    _mm_store_pd(result.data(), movement);

    return result;
#endif
}

bool MouseThread::checkTargetInScope(double target_x, double target_y, double target_w, double target_h, double reduction_factor)
{
    // Fast path: first check if the target center is within reasonable bounds
    double center_target_x = target_x + target_w / 2;
    double center_target_y = target_y + target_h / 2;

    // Quick check against screen center with a margin
    double dx = std::abs(center_target_x - center_x);
    double dy = std::abs(center_target_y - center_y);

    // If center is far away, avoid more complex calculations
    if (dx > screen_width / 4 || dy > screen_height / 4)
    {
        return false;
    }

    // More precise check using the reduced target dimensions
    double reduced_w = target_w * reduction_factor;
    double reduced_h = target_h * reduction_factor;

    // Fast AABB check using pre-calculated boundaries
    double min_x = center_target_x - reduced_w / 2;
    double max_x = center_target_x + reduced_w / 2;
    double min_y = center_target_y - reduced_h / 2;
    double max_y = center_target_y + reduced_h / 2;

    // Check if screen center is within the reduced target bounds
    return (center_x >= min_x && center_x <= max_x &&
            center_y >= min_y && center_y <= max_y);
}

double MouseThread::calculateTargetDistance(const AimbotTarget &target) const
{
    // SIMD optimized distance calculation
    __m128d pos = _mm_set_pd(target.y - center_y, target.x - center_x);
    __m128d squared = _mm_mul_pd(pos, pos);
    __m128d sum = _mm_hadd_pd(squared, squared);
    return _mm_cvtsd_f64(_mm_sqrt_pd(sum));
}

AimbotTarget *MouseThread::findClosestTarget(const std::vector<AimbotTarget> &targets) const
{
    if (targets.empty())
    {
        return nullptr;
    }

    AimbotTarget *closest = nullptr;
    double min_distance = std::numeric_limits<double>::max();

    for (const auto &target : targets)
    {
        double distance = calculateTargetDistance(target);
        if (distance < min_distance)
        {
            min_distance = distance;
            closest = const_cast<AimbotTarget *>(&target);
        }
    }

    return closest;
}

void MouseThread::moveMouse(const AimbotTarget &target)
{
    // Calculate target center
    double center_target_x = target.x + target.w / 2;
    double center_target_y = target.y + target.h / 2;

    // Predict next position (with thread safety using target's copy)
    Eigen::Vector2d predicted_pos = predictTargetPosition(center_target_x, center_target_y);

    // Calculate movement based on predicted position
    Eigen::Vector2d movement = calculateMovement(predicted_pos);

    // InputMethod를 사용하여 마우스 이동
    {
        std::lock_guard<std::mutex> lock(input_method_mutex);
        if (input_method)
        {
            input_method->move(static_cast<int>(movement[0]), static_cast<int>(movement[1]));
        }
    }

    // Update tracking state
    target_detected = true;
    last_target_time = std::chrono::steady_clock::now();
}

void MouseThread::pressMouse(const AimbotTarget &target)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);

    auto bScope = checkTargetInScope(target.x, target.y, target.w, target.h, bScope_multiplier);

    if (bScope && !mouse_pressed)
    {
        if (input_method)
        {
            input_method->press();
        }
        mouse_pressed = true;
    }
}

void MouseThread::releaseMouse()
{
    // No need to release if not pressed
    if (!mouse_pressed)
        return;

    std::lock_guard<std::mutex> lock(input_method_mutex);

    if (input_method)
    {
        input_method->release();
    }
    mouse_pressed = false;
}

void MouseThread::resetPrediction()
{
    kalman_filter->reset();
    pid_controller->reset();
    target_detected = false;
}

void MouseThread::checkAndResetPredictions()
{
    // Only check if target was previously detected
    if (target_detected)
    {
        // Get current time once
        const auto current_time = std::chrono::steady_clock::now();

        // Calculate time since last target detection
        const double elapsed = std::chrono::duration<double>(current_time - last_target_time).count();

        // Reset prediction if no target detected for a while
        // Using a shorter timeout for more responsive behavior
        if (elapsed > 0.25) // 250ms timeout
        {
            resetPrediction();
            target_detected = false;
        }
    }
}

void MouseThread::setInputMethod(std::unique_ptr<InputMethod> new_method)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);
    input_method = std::move(new_method);
}

void MouseThread::applyRecoilCompensation(float strength)
{
    std::lock_guard<std::mutex> lock(input_method_mutex);

    // Pre-compute the scaling factor
    static const double vertical_scale = (fov_y / screen_height) * (dpi * (1.0 / mouse_sensitivity)) / 360.0;

    // Apply strength with pre-computed scale
    int compensation = static_cast<int>(strength * vertical_scale);

    if (input_method)
    {
        input_method->move(0, compensation);
    }
}