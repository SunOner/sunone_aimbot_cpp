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

#include "mouse.h" // Incluye tu cabecera
#include "capture.h"
#include "SerialConnection.h"
#include "ghub.h"
#include "Kmbox_b.h"
#include "KmboxNetConnection.h"

#include "AimbotTarget.h"
#include <sunone_aimbot_cpp.h>

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

std::pair<double, double> MouseThread::calc_movement(double tx, double ty)
{
    double offx = tx - this->center_x;
    double offy = ty - this->center_y;
    double dist = std::hypot(offx, offy);
    double speed = calculate_speed_multiplier(dist);
    double degPerPxX = this->fov_x / this->screen_width;
    double degPerPxY = this->fov_y / this->screen_height;
    double mmx = offx * degPerPxX, mmy = offy * degPerPxY;
    double corr = 1.0;
    double fps = captureFps.load(std::memory_order_relaxed);
    if (fps > 30.0) { corr = 30.0 / fps; }
    auto counts_pair = config.degToCounts(mmx, mmy, this->fov_x);
    double move_x = counts_pair.first * speed * corr;
    double move_y = counts_pair.second * speed * corr;
    return { move_x, move_y };
}

double MouseThread::calculate_speed_multiplier(double distance)
{
    if (distance < config.snapRadius)
        return this->min_speed_multiplier * config.snapBoostFactor;
    if (distance < config.nearRadius) {
        double t = distance / config.nearRadius;
        double curve = 1.0 - std::pow(1.0 - t, config.speedCurveExponent);
        return this->min_speed_multiplier + (this->max_speed_multiplier - this->min_speed_multiplier) * curve;
    }
    double norm = std::clamp(distance / this->max_distance, 0.0, 1.0);
    return this->min_speed_multiplier + (this->max_speed_multiplier - this->min_speed_multiplier) * norm;
}

// *** SOLUCIÓN: Implementación de la versión de 4 argumentos ***
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

void MouseThread::moveMousePivot(double pivotX, double pivotY)
{
    auto current_time = std::chrono::steady_clock::now();
    if (prev_time.time_since_epoch().count() == 0 || !target_detected.load(std::memory_order_relaxed)) {
        prev_time = current_time;
        prev_x = pivotX; prev_y = pivotY;
        prev_velocity_x = prev_velocity_y = 0.0;
        auto m0 = calc_movement(pivotX, pivotY);
        queueMove(static_cast<int>(m0.first), static_cast<int>(m0.second));
        return;
    }
    double dt = std::chrono::duration<double>(current_time - prev_time).count();
    dt = std::max(dt, 1e-8);
    double vx = std::clamp((pivotX - prev_x) / dt, -20000.0, 20000.0);
    double vy = std::clamp((pivotY - prev_y) / dt, -20000.0, 20000.0);
    prev_time = current_time;
    prev_x = pivotX; prev_y = pivotY;
    prev_velocity_x = vx;  prev_velocity_y = vy;
    double predX = pivotX + vx * this->prediction_interval;
    double predY = pivotY + vy * this->prediction_interval;
    auto mv = calc_movement(predX, predY);
    int mx = static_cast<int>(mv.first);
    int my = static_cast<int>(mv.second);
    if (this->wind_mouse_enabled) windMouseMoveRelative(mx, my);
    else queueMove(mx, my);
}

void MouseThread::pressMouse(const AimbotTarget& target)
{
    // *** SOLUCIÓN: La llamada ahora coincide con la nueva firma de 4 argumentos. ***
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

void MouseThread::resetPrediction()
{
    prev_time = std::chrono::steady_clock::time_point();
    prev_x = 0; prev_y = 0;
    prev_velocity_x = 0; prev_velocity_y = 0;
    target_detected.store(false, std::memory_order_relaxed);
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
    const double frame_time = 1.0 / 30.0;
    auto current_time = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(current_time - prev_time).count();
    if (prev_time.time_since_epoch().count() == 0 || dt > 0.5) return result;
    for (int i = 1; i <= frames; i++) {
        double t = frame_time * i;
        result.push_back({ pivotX + prev_velocity_x * t, pivotY + prev_velocity_y * t });
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