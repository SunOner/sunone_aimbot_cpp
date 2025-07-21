#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <cmath>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <atomic>
#include <vector>
#include <random>
#include <opencv2/video/tracking.hpp>

#include "mouse.h"
#include "capture.h"
#include "SerialConnection.h"
#include "ghub.h"
#include "Kmbox_b.h"
#include "KmboxNetConnection.h"
#include "AimbotTarget.h"
#include <sunone_aimbot_cpp.h>

// Constructor, Destructor y updateConfig sin cambios...
MouseThread::MouseThread(
    int resolution, int fovX, int fovY,
    double minSpeedMultiplier, double maxSpeedMultiplier,
    double predictionInterval, bool auto_shoot, float bScope_multiplier,
    SerialConnection* serialConnection, GhubMouse* gHubMouse,
    Kmbox_b_Connection* kmboxConnection, KmboxNetConnection* kmboxNetConn)
    : head_(0), tail_(0), workerStop(false),
    prev_x(0.0), prev_y(0.0), prev_velocity_x(0.0), prev_velocity_y(0.0),
    target_detected(false), mouse_pressed(false),
    serial(serialConnection), kmbox(kmboxConnection), kmbox_net(kmboxNetConn), gHub(gHubMouse)
{
    updateConfig(resolution, fovX, fovY, minSpeedMultiplier, maxSpeedMultiplier, predictionInterval, auto_shoot, bScope_multiplier);
    prev_time = std::chrono::steady_clock::time_point();
    last_target_time = std::chrono::steady_clock::now();
    moveWorker = std::thread(&MouseThread::moveWorkerLoop, this);
}

MouseThread::~MouseThread()
{
    workerStop.store(true);
    if (moveWorker.joinable()) {
        moveWorker.join();
    }
}

void MouseThread::updateConfig(
    int resolution, int fovX, int fovY,
    double minSpeedMultiplier, double maxSpeedMultiplier,
    double predictionInterval, bool new_auto_shoot, float new_bScope_multiplier)
{
    std::lock_guard<std::mutex> lock(config_mutex);
    this->screen_width = static_cast<double>(resolution);
    this->screen_height = static_cast<double>(resolution);
    this->fov_x = static_cast<double>(fovX);
    this->fov_y = static_cast<double>(fovY);
    this->min_speed_multiplier = minSpeedMultiplier;
    this->max_speed_multiplier = maxSpeedMultiplier;
    this->prediction_interval = predictionInterval;
    this->auto_shoot = new_auto_shoot;
    this->bScope_multiplier = new_bScope_multiplier;
    this->center_x = static_cast<double>(resolution) / 2.0;
    this->center_y = static_cast<double>(resolution) / 2.0;
    this->max_distance = std::hypot(static_cast<double>(resolution), static_cast<double>(resolution)) / 2.0;
    this->wind_mouse_enabled = config.wind_mouse_enabled;
    this->wind_G = config.wind_G;
    this->wind_W = config.wind_W;
    this->wind_M = config.wind_M;
    this->wind_D = config.wind_D;
}


void MouseThread::initKalmanFilter(double x, double y) {
    kf.emplace(4, 2, 0, CV_64F); 
    last_kf_update_time = std::chrono::steady_clock::now();

    cv::setIdentity(kf->transitionMatrix);
    kf->measurementMatrix = cv::Mat::zeros(2, 4, CV_64F);
    kf->measurementMatrix.at<double>(0, 0) = 1.0;
    kf->measurementMatrix.at<double>(1, 1) = 1.0;
    
    // *** AJUSTE DE ESTABILIDAD: Usamos los valores de la configuración ***
    // Estos valores son la clave para eliminar el jitter.
    // Q: Cuánto confiamos en el modelo (más bajo = más suave).
    // R: Cuánto confiamos en la medición de YOLO (más alto = más suave).
    cv::setIdentity(kf->processNoiseCov, cv::Scalar::all(config.kalman_q));
    cv::setIdentity(kf->measurementNoiseCov, cv::Scalar::all(config.kalman_r));
    
    cv::setIdentity(kf->errorCovPost, cv::Scalar::all(1.0));
    kf->statePost.at<double>(0) = x;
    kf->statePost.at<double>(1) = y;
    kf->statePost.at<double>(2) = 0;
    kf->statePost.at<double>(3) = 0;
}


// *** MODIFICADO: calculate_speed_multiplier con AMORTIGUACIÓN (DAMPING) ***
double MouseThread::calculate_speed_multiplier(double distance)
{
    // *** LÓGICA DE SNAP & LOCK CON AMORTIGUACIÓN ***
    if (distance < config.snapRadius) {
        // El boost agresivo se mantiene, pero la amortiguación en calc_movement evitará el temblor.
        return this->min_speed_multiplier * config.snapBoostFactor; 
    }
    
    if (distance < config.nearRadius) {
        double t = (distance - config.snapRadius) / (config.nearRadius - config.snapRadius);
        double curve = 1.0 - std::pow(1.0 - t, config.speedCurveExponent);
        return this->min_speed_multiplier + (this->max_speed_multiplier - this->min_speed_multiplier) * curve;
    }
    
    double norm = std::clamp(distance / this->max_distance, 0.0, 1.0);
    return this->min_speed_multiplier + (this->max_speed_multiplier - this->min_speed_multiplier) * norm;
}

void MouseThread::moveInstant(int dx, int dy)
{
    if (dx != 0 || dy != 0) {
        queueMove(dx, dy);
    }
}

// *** NUEVO: Función para movimiento suavizado (lógica extraída de calc_movement) ***
void MouseThread::moveSmooth(double target_x, double target_y)
{
    // La lógica de interpolación que ya teníamos
    if (!smoothing_initialized) {
        smoothed_x = this->center_x;
        smoothed_y = this->center_y;
        smoothing_initialized = true;
    }

    double final_target_x, final_target_y;
    double dist_to_target = std::hypot(target_x - center_x, target_y - center_y);

    if (config.smoothing_level > 1 && dist_to_target >= config.snapRadius) {
        float lerp_alpha = 1.0f / static_cast<float>(config.smoothing_level);
        smoothed_x = smoothed_x + (target_x - smoothed_x) * lerp_alpha;
        smoothed_y = smoothed_y + (target_y - smoothed_y) * lerp_alpha;
        final_target_x = smoothed_x;
        final_target_y = smoothed_y;
    } else {
        smoothed_x = target_x;
        smoothed_y = target_y;
        final_target_x = target_x;
        final_target_y = target_y;
    }

    // Calcular el delta de píxeles necesario para llegar al objetivo suavizado
    auto move_pair = calc_movement(final_target_x, final_target_y);
    int move_x = static_cast<int>(move_pair.first);
    int move_y = static_cast<int>(move_pair.second);
    
    if (move_x != 0 || move_y != 0) {
        queueMove(move_x, move_y);
    }
}


// *** MODIFICADO: moveMousePivot AHORA ES UN DESPACHADOR ***
void MouseThread::moveMousePivot(double pivotX, double pivotY)
{
    // --- PASO 1: PREDICCIÓN (Común a todos los modos) ---
    // Esta parte no cambia. Siempre calculamos el mejor punto futuro para apuntar.
    double finalTargetX = pivotX;
    double finalTargetY = pivotY;

    if (config.prediction_method == "kalman")
    {
        if (!kf.has_value()) { initKalmanFilter(pivotX, pivotY); }

        auto current_time = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(current_time - last_kf_update_time).count();
        last_kf_update_time = current_time;
        dt = std::max(dt, 1e-5); 

        kf->transitionMatrix.at<double>(0, 2) = dt;
        kf->transitionMatrix.at<double>(1, 3) = dt;
        kf->predict();
        cv::Mat measurement = cv::Mat::zeros(2, 1, CV_64F);
        measurement.at<double>(0) = pivotX;
        measurement.at<double>(1) = pivotY;
        cv::Mat estimated = kf->correct(measurement);

        finalTargetX = estimated.at<double>(0) + estimated.at<double>(2) * this->prediction_interval;
        finalTargetY = estimated.at<double>(1) + estimated.at<double>(3) * this->prediction_interval;

        prev_velocity_x = estimated.at<double>(2);
        prev_velocity_y = estimated.at<double>(3);
        prev_x = estimated.at<double>(0);
        prev_y = estimated.at<double>(1);
    }
    else // "linear"
    {
        // ... (lógica de predicción lineal) ...
        auto current_time = std::chrono::steady_clock::now();
        if (prev_time.time_since_epoch().count() == 0 || !target_detected.load(std::memory_order_relaxed)) {
            prev_time = current_time;
            prev_x = pivotX; prev_y = pivotY;
            prev_velocity_x = prev_velocity_y = 0.0;
        } else {
            double dt = std::chrono::duration<double>(current_time - prev_time).count();
            dt = std::max(dt, 1e-8); 
            double vx = std::clamp((pivotX - prev_x) / dt, -20000.0, 20000.0);
            double vy = std::clamp((pivotY - prev_y) / dt, -20000.0, 20000.0);
            prev_velocity_x = vx;  prev_velocity_y = vy;
            finalTargetX = pivotX + vx * this->prediction_interval;
            finalTargetY = pivotY + vy * this->prediction_interval;
        }
        prev_time = current_time;
        prev_x = pivotX; prev_y = pivotY;
    }

    // --- PASO 2: DESPACHO DE MOVIMIENTO (Selección de método) ---
    // Basado en el config, decidimos CÓMO movernos hacia finalTargetX/Y.
    if (config.mouse_move_method == "wind")
    {
        auto move_pair = calc_movement(finalTargetX, finalTargetY);
        windMouseMoveRelative(static_cast<int>(move_pair.first), static_cast<int>(move_pair.second));
    }
    else if (config.mouse_move_method == "smooth")
    {
        moveSmooth(finalTargetX, finalTargetY);
    }
    else // "instant"
    {
        auto move_pair = calc_movement(finalTargetX, finalTargetY);
        moveInstant(static_cast<int>(move_pair.first), static_cast<int>(move_pair.second));
    }
}

std::pair<double, double> MouseThread::calc_movement(double tx, double ty)
{
    // Esta función ahora solo convierte el objetivo (tx, ty) en un delta de píxeles de ratón.
    double final_offx = tx - this->center_x;
    double final_offy = ty - this->center_y;
    double dist_to_final_target = std::hypot(final_offx, final_offy);
    
    double speed = calculate_speed_multiplier(dist_to_final_target);
    
    double degPerPxX = this->fov_x / this->screen_width;
    double degPerPxY = this->fov_y / this->screen_height;
    
    double mmx = final_offx * degPerPxX;
    double mmy = final_offy * degPerPxY;
    
    double corr = 1.0;
    double fps = captureFps.load(std::memory_order_relaxed);
    if (fps > 30.0) { corr = 30.0 / fps; }
    
    auto counts_pair = config.degToCounts(mmx, mmy, this->fov_x);
    double move_x = counts_pair.first * speed * corr;
    double move_y = counts_pair.second * speed * corr;

    if (dist_to_final_target < config.snapRadius && config.snapRadius > 0) {
        double damping_factor = dist_to_final_target / config.snapRadius;
        move_x *= damping_factor;
        move_y *= damping_factor;
    }

    return { move_x, move_y };
}

void MouseThread::moveMouseWithSmoothing(double targetX, double targetY)
{
    // =========================================================================
    //  RUTA RÁPIDA (FAST PATH) PARA MOVIMIENTO BRUSCO / INSTANTÁNEO
    // =========================================================================
    if (config.smoothing_level <= 1)
    {
        // 1. Calcular el desplazamiento total necesario desde la mira (centro) al objetivo.
        double raw_delta_x = targetX - center_x;
        double raw_delta_y = targetY - center_y;

        // 2. Usar 'addOverflow' para manejar píxeles fraccionarios y mantener la precisión.
        auto move_pair = addOverflow(raw_delta_x, raw_delta_y, move_overflow_x, move_overflow_y);
        int move_x = static_cast<int>(move_pair.first);
        int move_y = static_cast<int>(move_pair.second);

        // 3. Enviar el movimiento en un solo paso, sin bucles ni retrasos.
        if (move_x != 0 || move_y != 0) {
            sendMovementToDriver(move_x, move_y);
        }

        // NO HAY BUCLE. NO HAY 'easeInOut'. Y LO MÁS IMPORTANTE, NO HAY 'sleep_for'.
        return; // Salimos de la función inmediatamente.
    }

    // =========================================================================
    //  RUTA LENTA (SLOW PATH) - LÓGICA ORIGINAL PARA MOVIMIENTO SUAVE
    //  Este código solo se ejecutará si 'smoothness' es 2 o mayor.
    // =========================================================================

    double start_x = center_x;
    double start_y = center_y;
    double previous_x = start_x;
    double previous_y = start_y;

    // El bloqueo/desbloqueo del ratón solo tiene sentido en un movimiento de varios pasos.
    if (kmbox_net) {
        kmbox_net->maskMouseX(1);
        kmbox_net->maskMouseY(1);
    }

    // Proceso de movimiento suave original
    for (int i = 1; i <= config.smoothing_level; i++) {
        double t = static_cast<double>(i) / config.smoothing_level;
        double progress = easeInOut(t);

        double current_x = start_x + (targetX - start_x) * progress;
        double current_y = start_y + (targetY - start_y) * progress;

        double raw_delta_x = current_x - previous_x;
        double raw_delta_y = current_y - previous_y;

        auto move_pair = addOverflow(raw_delta_x, raw_delta_y, move_overflow_x, move_overflow_y);
        int move_x = static_cast<int>(move_pair.first);
        int move_y = static_cast<int>(move_pair.second);

        if (move_x != 0 || move_y != 0) {
            sendMovementToDriver(move_x, move_y);
        }

        previous_x = current_x;
        previous_y = current_y;

        // El retraso deliberado que crea el movimiento lento
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    if (kmbox_net) {
        kmbox_net->maskMouseX(0);
        kmbox_net->maskMouseY(0);
    }
}


void MouseThread::resetPrediction()
{
    kf.reset(); // Funciona para ambos métodos, resetea el filtro si existe.
    prev_time = std::chrono::steady_clock::time_point();
    prev_x = 0; prev_y = 0;
    prev_velocity_x = 0; prev_velocity_y = 0;
    target_detected.store(false, std::memory_order_relaxed);
}

void MouseThread::queueMove(int dx, int dy)
{
    size_t current_head = head_.load(std::memory_order_relaxed);
    size_t next_head = (current_head + 1) % QUEUE_SIZE;
    if (next_head == tail_.load(std::memory_order_acquire)) {
        return;
    }
    moveQueue_[current_head] = { dx, dy };
    head_.store(next_head, std::memory_order_release);
}

void MouseThread::moveWorkerLoop()
{
    while (!workerStop.load(std::memory_order_relaxed))
    {
        size_t current_tail = tail_.load(std::memory_order_relaxed);
        if (current_tail == head_.load(std::memory_order_acquire))
        {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            continue;
        }
        const Move& m = moveQueue_[current_tail];
        sendMovementToDriver(m.dx, m.dy);
        tail_.store((current_tail + 1) % QUEUE_SIZE, std::memory_order_release);
    }
}

void MouseThread::sendMovementToDriver(int dx, int dy)
{
    if (dx == 0 && dy == 0) return;
    std::lock_guard<std::mutex> lock(input_method_mutex);
    if (kmbox) { kmbox->move(dx, dy); }
    else if (kmbox_net) { kmbox_net->move(dx, dy); }
    else if (serial) { serial->move(dx, dy); }
    else if (gHub) { gHub->mouse_xy(dx, dy); }
    else {
        INPUT in{ 0 };
        in.type = INPUT_MOUSE;
        in.mi.dx = dx;
        in.mi.dy = dy;
        in.mi.dwFlags = MOUSEEVENTF_MOVE;
        SendInput(1, &in, sizeof(INPUT));
    }
}

void MouseThread::windMouseMoveRelative(int dx, int dy)
{
    if (dx == 0 && dy == 0) return;
    thread_local static std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    constexpr double SQRT3 = 1.7320508075688772;
    constexpr double SQRT5 = 2.23606797749979;
    double sx = 0, sy = 0;
    double dxF = static_cast<double>(dx), dyF = static_cast<double>(dy);
    double vx = 0, vy = 0, wX = 0, wY = 0;
    int cx = 0, cy = 0;
    double current_wind_M = this->wind_M, current_wind_W = this->wind_W;
    double current_wind_G = this->wind_G, current_wind_D = this->wind_D;

    while (std::hypot(dxF - sx, dyF - sy) >= 1.0)
    {
        double dist = std::hypot(dxF - sx, dyF - sy);
        double wMag = std::min(current_wind_W, dist);
        if (dist >= current_wind_D) {
            wX = wX / SQRT3 + distribution(generator) * wMag / SQRT5;
            wY = wY / SQRT3 + distribution(generator) * wMag / SQRT5;
        }
        else {
            wX /= SQRT3; wY /= SQRT3;
            if (current_wind_M < 3.0) {
                std::uniform_real_distribution<double> small_m_dist(3.0, 6.0);
                current_wind_M = small_m_dist(generator);
            }
            else { current_wind_M /= SQRT5; }
        }
        vx += wX + current_wind_G * (dxF - sx) / dist;
        vy += wY + current_wind_G * (dyF - sy) / dist;
        double vMag = std::hypot(vx, vy);
        if (vMag > current_wind_M) {
            std::uniform_real_distribution<double> clip_dist(current_wind_M / 2.0, current_wind_M);
            double vClip = clip_dist(generator);
            vx = (vx / vMag) * vClip;
            vy = (vy / vMag) * vClip;
        }
        sx += vx; sy += vy;
        int rx = static_cast<int>(std::round(sx)), ry = static_cast<int>(std::round(sy));
        int step_x = rx - cx, step_y = ry - cy;
        if (step_x != 0 || step_y != 0) {
            queueMove(step_x, step_y);
            cx = rx; cy = ry;
        }
    }
}

bool MouseThread::check_target_in_scope(double target_x, double target_y, double target_w, double target_h)
{
    double center_target_x = target_x + target_w / 2.0;
    double center_target_y = target_y + target_h / 2.0;
    double reduced_w = target_w * (this->bScope_multiplier / 2.0);
    double reduced_h = target_h * (this->bScope_multiplier / 2.0);
    double x1 = center_target_x - reduced_w;
    double x2 = center_target_x + reduced_w;
    double y1 = center_target_y - reduced_h;
    double y2 = center_target_y + reduced_h;
    return (this->center_x > x1 && this->center_x < x2 && this->center_y > y1 && this->center_y < y2);
}

void MouseThread::pressMouse(const AimbotTarget& target)
{
    bool in_scope = check_target_in_scope(target.x, target.y, target.w, target.h);
    if (in_scope && !mouse_pressed.load(std::memory_order_relaxed)) {
        std::lock_guard<std::mutex> lock(input_method_mutex);
        if (kmbox) kmbox->press(0);
        else if (kmbox_net) kmbox_net->keyDown(0);
        else if (serial) serial->press();
        else if (gHub) gHub->mouse_down();
        else {
            INPUT input = { 0 }; input.type = INPUT_MOUSE; input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
            SendInput(1, &input, sizeof(INPUT));
        }
        mouse_pressed.store(true, std::memory_order_relaxed);
    }
    else if (!in_scope && mouse_pressed.load(std::memory_order_relaxed)) {
        releaseMouse();
    }
}

void MouseThread::releaseMouse()
{
    if (mouse_pressed.load(std::memory_order_relaxed)) {
        std::lock_guard<std::mutex> lock(input_method_mutex);
        if (kmbox) kmbox->release(0);
        else if (kmbox_net) kmbox_net->keyUp(0);
        else if (serial) serial->release();
        else if (gHub) gHub->mouse_up();
        else {
            INPUT input = { 0 }; input.type = INPUT_MOUSE; input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
            SendInput(1, &input, sizeof(INPUT));
        }
        mouse_pressed.store(false, std::memory_order_relaxed);
    }
}

void MouseThread::checkAndResetPredictions()
{
    auto current_time = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(current_time - last_target_time).count();
    if (elapsed > 0.5 && target_detected.load(std::memory_order_relaxed)) {
        resetPrediction();
    }
}

std::vector<std::pair<double, double>> MouseThread::predictFuturePositions(double pivotX, double pivotY, int frames)
{
    std::vector<std::pair<double, double>> result;
    if (frames <= 0) return result;

    result.reserve(frames);
    const double frame_time = 1.0 / 60.0;

    double current_x = prev_x;
    double current_y = prev_y;
    double vel_x = prev_velocity_x;
    double vel_y = prev_velocity_y;

    if (config.prediction_method == "kalman" && kf.has_value()) {
        current_x = kf->statePost.at<double>(0);
        current_y = kf->statePost.at<double>(1);
        vel_x = kf->statePost.at<double>(2);
        vel_y = kf->statePost.at<double>(3);
    }

    for (int i = 1; i <= frames; i++) {
        double t_future = frame_time * i;
        result.push_back({ current_x + vel_x * t_future, current_y + vel_y * t_future });
    }
    return result;
}

void MouseThread::storeFuturePositions(const std::vector<std::pair<double, double>>& positions)
{
    std::lock_guard<std::mutex> lock(futurePositionsMutex);
    futurePositions = positions;
}

void MouseThread::clearFuturePositions()
{
    std::lock_guard<std::mutex> lock(futurePositionsMutex);
    futurePositions.clear();
}

std::vector<std::pair<double, double>> MouseThread::getFuturePositions()
{
    std::lock_guard<std::mutex> lock(futurePositionsMutex);
    return futurePositions;
}

void MouseThread::setSerialConnection(SerialConnection* newSerial) { std::lock_guard<std::mutex> lock(input_method_mutex); serial = newSerial; }
void MouseThread::setKmboxConnection(Kmbox_b_Connection* newKmbox) { std::lock_guard<std::mutex> lock(input_method_mutex); kmbox = newKmbox; }
void MouseThread::setKmboxNetConnection(KmboxNetConnection* newKmbox_net) { std::lock_guard<std::mutex> lock(input_method_mutex); kmbox_net = newKmbox_net; }
void MouseThread::setGHubMouse(GhubMouse* newGHub) { std::lock_guard<std::mutex> lock(input_method_mutex); gHub = newGHub; }

// mouse.cpp