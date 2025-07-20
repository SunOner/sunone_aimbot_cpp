#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>
#include <immintrin.h> // Para _mm_pause

#include <cmath>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <atomic>
#include <vector>
#include <numeric> // Para std::accumulate

#include "mouse.h"
#include "capture.h"
#include "SerialConnection.h"
#include "sunone_aimbot_cpp.h"
#include "ghub.h"
#include "Kmbox_b.h"
#include "KmboxNetConnection.h"

MouseThread::MouseThread(
    int resolution, int fovX, int fovY,
    double minSpeedMultiplier, double maxSpeedMultiplier,
    double predictionInterval, bool auto_shoot, float bScope_multiplier,
    SerialConnection* serialConnection, GhubMouse* gHubMouse,
    Kmbox_b_Connection* kmboxConnection, KmboxNetConnection* kmboxNetConn)
    : screen_width(resolution), screen_height(resolution),
    prediction_interval(predictionInterval), fov_x(fovX), fov_y(fovY),
    max_distance(std::hypot(static_cast<double>(resolution), static_cast<double>(resolution)) / 2.0),
    min_speed_multiplier(minSpeedMultiplier), max_speed_multiplier(maxSpeedMultiplier),
    center_x(static_cast<double>(resolution) / 2.0), center_y(static_cast<double>(resolution) / 2.0),
    auto_shoot(auto_shoot), bScope_multiplier(bScope_multiplier),
    serial(serialConnection), kmbox(kmboxConnection), kmbox_net(kmboxNetConn), gHub(gHubMouse),
    prev_velocity_x(0.0), prev_velocity_y(0.0), prev_x(0.0), prev_y(0.0),
    workerStop(false), mouse_pressed(false),
    head_{0}, tail_{0} // *** OPTIMIZACIÓN: Inicializar índices de la cola lock-free ***
{
    prev_time = std::chrono::steady_clock::time_point();
    last_target_time = std::chrono::steady_clock::now();

    wind_mouse_enabled = config.wind_mouse_enabled;
    wind_G = config.wind_G; wind_W = config.wind_W;
    wind_M = config.wind_M; wind_D = config.wind_D;

    moveWorker = std::thread(&MouseThread::moveWorkerLoop, this);
}

MouseThread::~MouseThread()
{
    workerStop = true;
    if (moveWorker.joinable()) moveWorker.join();
}

void MouseThread::updateConfig(
    int resolution, int fovX, int fovY,
    double minSpeedMultiplier, double maxSpeedMultiplier,
    double predictionInterval, bool new_auto_shoot, float new_bScope_multiplier)
{
    std::lock_guard<std::mutex> lock(config_mutex);
    screen_width = screen_height = resolution;
    fov_x = fovX;  fov_y = fovY;
    min_speed_multiplier = minSpeedMultiplier;
    max_speed_multiplier = maxSpeedMultiplier;
    prediction_interval = predictionInterval;
    this->auto_shoot = new_auto_shoot;
    this->bScope_multiplier = new_bScope_multiplier;

    center_x = center_y = static_cast<double>(resolution) / 2.0;
    max_distance = std::hypot(static_cast<double>(resolution), static_cast<double>(resolution)) / 2.0;

    wind_mouse_enabled = config.wind_mouse_enabled;
    wind_G = config.wind_G; wind_W = config.wind_W;
    wind_M = config.wind_M; wind_D = config.wind_D;
}

// *** OPTIMIZACIÓN: Cola de movimiento lock-free ***
void MouseThread::queueMove(int dx, int dy)
{
    size_t current_head = head_.load(std::memory_order_relaxed);
    size_t next_head = (current_head + 1) % QUEUE_SIZE;

    // Si la cola está llena, el productor espera. En la práctica, esto casi nunca debería ocurrir.
    while (next_head == tail_.load(std::memory_order_acquire)) {
        _mm_pause();
    }

    moveQueue_[current_head] = { dx, dy };
    head_.store(next_head, std::memory_order_release);
}

// *** OPTIMIZACIÓN: Bucle de trabajo con spin-wait ***
void MouseThread::moveWorkerLoop()
{
    while (!workerStop)
    {
        size_t current_tail = tail_.load(std::memory_order_relaxed);
        if (current_tail == head_.load(std::memory_order_acquire))
        {
            // La cola está vacía, esperar eficientemente.
            _mm_pause();
            continue;
        }

        const Move& m = moveQueue_[current_tail];
        sendMovementToDriver(m.dx, m.dy);
        
        tail_.store((current_tail + 1) % QUEUE_SIZE, std::memory_order_release);
    }
}

// *** OPTIMIZACIÓN: Lógica de envío centralizada ***
void MouseThread::sendMovementToDriver(int dx, int dy)
{
    if (dx == 0 && dy == 0) return;

    if (kmbox) { kmbox->move(dx, dy); }
    else if (kmbox_net) { kmbox_net->move(dx, dy); }
    else if (serial) { serial->move(dx, dy); }
    else if (gHub) { gHub->mouse_xy(dx, dy); }
    else {
        INPUT in{ 0 };
        in.type = INPUT_MOUSE;
        in.mi.dx = dx;  in.mi.dy = dy;
        in.mi.dwFlags = MOUSEEVENTF_MOVE; // MOUSEEVENTF_VIRTUALDESK es para coordenadas absolutas
        SendInput(1, &in, sizeof(INPUT));
    }
}

// Lógica de WindMouse sin cambios funcionales, pero ahora usa la cola lock-free
void MouseThread::windMouseMoveRelative(int dx, int dy)
{
    if (dx == 0 && dy == 0) return;

    constexpr double SQRT3 = 1.7320508075688772;
    constexpr double SQRT5 = 2.23606797749979;

    double sx = 0, sy = 0;
    double dxF = static_cast<double>(dx);
    double dyF = static_cast<double>(dy);
    double vx = 0, vy = 0, wX = 0, wY = 0;
    int    cx = 0, cy = 0;

    while (std::hypot(dxF - sx, dyF - sy) >= 1.0)
    {
        double dist = std::hypot(dxF - sx, dyF - sy);
        double wMag = std::min(wind_W, dist);

        if (dist >= wind_D) {
            wX = wX / SQRT3 + ((double)rand() / RAND_MAX * 2.0 - 1.0) * wMag / SQRT5;
            wY = wY / SQRT3 + ((double)rand() / RAND_MAX * 2.0 - 1.0) * wMag / SQRT5;
        } else {
            wX /= SQRT3;  wY /= SQRT3;
            if (wind_M < 3.0) wind_M = ((double)rand() / RAND_MAX) * 3.0 + 3.0;
            else wind_M /= SQRT5;
        }

        vx += wX + wind_G * (dxF - sx) / dist;
        vy += wY + wind_G * (dyF - sy) / dist;

        double vMag = std::hypot(vx, vy);
        if (vMag > wind_M) {
            double vClip = wind_M / 2.0 + ((double)rand() / RAND_MAX) * wind_M / 2.0;
            vx = (vx / vMag) * vClip;
            vy = (vy / vMag) * vClip;
        }

        sx += vx;  sy += vy;
        int rx = static_cast<int>(std::round(sx));
        int ry = static_cast<int>(std::round(sy));
        int step_x = rx - cx;
        int step_y = ry - cy;
        if (step_x || step_y) {
            queueMove(step_x, step_y);
            cx = rx; cy = ry;
        }
    }
}

void MouseThread::moveMousePivot(double pivotX, double pivotY)
{
    // La lógica de predicción no cambia, solo el método de encolado
    auto current_time = std::chrono::steady_clock::now();
    
    if (prev_time.time_since_epoch().count() == 0 || !target_detected.load()) {
        prev_time = current_time;
        prev_x = pivotX; prev_y = pivotY;
        prev_velocity_x = prev_velocity_y = 0.0;

        auto m0 = calc_movement(pivotX, pivotY);
        queueMove(static_cast<int>(m0.first), static_cast<int>(m0.second));
        return;
    }

    double dt = std::chrono::duration<double>(current_time - prev_time).count();
    dt = std::max(dt, 1e-8); // Evitar división por cero

    double vx = std::clamp((pivotX - prev_x) / dt, -20000.0, 20000.0);
    double vy = std::clamp((pivotY - prev_y) / dt, -20000.0, 20000.0);
    
    prev_time = current_time;
    prev_x = pivotX; prev_y = pivotY;
    prev_velocity_x = vx;  prev_velocity_y = vy;

    double predX = pivotX + vx * prediction_interval;
    double predY = pivotY + vy * prediction_interval;

    auto mv = calc_movement(predX, predY);
    int mx = static_cast<int>(mv.first);
    int my = static_cast<int>(mv.second);

    if (wind_mouse_enabled) windMouseMoveRelative(mx, my);
    else queueMove(mx, my);
}


// --- El resto de las funciones (pressMouse, releaseMouse, etc.) no requieren cambios drásticos ---
// --- ya que la lógica de enviar el comando es ahora manejada por sendMovementToDriver o ---
// --- los métodos del propio dispositivo, que ya están optimizados. ---

void MouseThread::pressMouse(const AimbotTarget& target)
{
    bool bScope = check_target_in_scope(target.x, target.y, target.w, target.h, bScope_multiplier);
    if (bScope && !mouse_pressed)
    {
        if (kmbox) { kmbox->press(0); }
        else if (kmbox_net) { kmbox_net->keyDown(0); }
        else if (serial) { serial->press(); }
        else if (gHub) { gHub->mouse_down(); }
        else {
            INPUT input = { 0 };
            input.type = INPUT_MOUSE;
            input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
            SendInput(1, &input, sizeof(INPUT));
        }
        mouse_pressed = true;
    }
    else if (!bScope && mouse_pressed)
    {
        releaseMouse(); // Reutilizar la lógica de liberación
    }
}

void MouseThread::releaseMouse()
{
    if (mouse_pressed)
    {
        if (kmbox) { kmbox->release(0); }
        else if (kmbox_net) { kmbox_net->keyUp(0); }
        else if (serial) { serial->release(); }
        else if (gHub) { gHub->mouse_up(); }
        else {
            INPUT input = { 0 };
            input.type = INPUT_MOUSE;
            input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
            SendInput(1, &input, sizeof(INPUT));
        }
        mouse_pressed = false;
    }
}

// Las funciones restantes se mantienen ya que su lógica es correcta y no son cuellos de botella
std::pair<double, double> MouseThread::calc_movement(double tx, double ty)
{
    std::lock_guard<std::mutex> lock(config_mutex);
    double offx = tx - center_x;
    double offy = ty - center_y;
    double dist = std::hypot(offx, offy);
    double speed = calculate_speed_multiplier(dist);

    double degPerPxX = fov_x / static_cast<double>(screen_width);
    double degPerPxY = fov_y / static_cast<double>(screen_height);

    double mmx = offx * degPerPxX;
    double mmy = offy * degPerPxY;

    double corr = 1.0;
    double fps = captureFps.load(std::memory_order_relaxed);
    if (fps > 30.0) corr = 30.0 / fps;

    auto counts_pair = config.degToCounts(mmx, mmy, fov_x);
    double move_x = counts_pair.first * speed * corr;
    double move_y = counts_pair.second * speed * corr;

    return { move_x, move_y };
}

double MouseThread::calculate_speed_multiplier(double distance)
{
    if (distance < config.snapRadius)
        return min_speed_multiplier * config.snapBoostFactor;

    if (distance < config.nearRadius)
    {
        double t = distance / config.nearRadius;
        double curve = 1.0 - std::pow(1.0 - t, config.speedCurveExponent);
        return min_speed_multiplier + (max_speed_multiplier - min_speed_multiplier) * curve;
    }

    double norm = std::clamp(distance / max_distance, 0.0, 1.0);
    return min_speed_multiplier + (max_speed_multiplier - min_speed_multiplier) * norm;
}

bool MouseThread::check_target_in_scope(double target_x, double target_y, double target_w, double target_h, double reduction_factor)
{
    double center_target_x = target_x + target_w / 2.0;
    double center_target_y = target_y + target_h / 2.0;
    double reduced_w = target_w * (reduction_factor / 2.0);
    double reduced_h = target_h * (reduction_factor / 2.0);
    double x1 = center_target_x - reduced_w;
    double x2 = center_target_x + reduced_w;
    double y1 = center_target_y - reduced_h;
    double y2 = center_target_y + reduced_h;

    return (center_x > x1 && center_x < x2 && center_y > y1 && center_y < y2);
}

void MouseThread::resetPrediction()
{
    prev_time = std::chrono::steady_clock::time_point();
    prev_x = 0; prev_y = 0;
    prev_velocity_x = 0; prev_velocity_y = 0;
    target_detected.store(false);
}

void MouseThread::checkAndResetPredictions()
{
    auto current_time = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(current_time - last_target_time).count();
    if (elapsed > 0.5 && target_detected.load())
    {
        resetPrediction();
    }
}

std::vector<std::pair<double, double>> MouseThread::predictFuturePositions(double pivotX, double pivotY, int frames)
{
    std::vector<std::pair<double, double>> result;
    result.reserve(frames);
    const double frame_time = 1.0 / 30.0; // Assume fixed 30fps for prediction steps
    auto current_time = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(current_time - prev_time).count();
    if (prev_time.time_since_epoch().count() == 0 || dt > 0.5) return result;

    for (int i = 1; i <= frames; i++)
    {
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

// Setters para los dispositivos de entrada
void MouseThread::setSerialConnection(SerialConnection* newSerial) { serial = newSerial; }
void MouseThread::setKmboxConnection(Kmbox_b_Connection* newKmbox) { kmbox = newKmbox; }
void MouseThread::setKmboxNetConnection(KmboxNetConnection* newKmbox_net) { kmbox_net = newKmbox_net; }
void MouseThread::setGHubMouse(GhubMouse* newGHub) { gHub = newGHub; }// mouse.cpp