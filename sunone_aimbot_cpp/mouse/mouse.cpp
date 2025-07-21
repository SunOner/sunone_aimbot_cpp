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
#include <opencv2/video/tracking.hpp> // *** NUEVO: Incluir la cabecera para el Filtro de Kalman ***

#include "mouse.h"
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

// *** NUEVO: Inicializador del Filtro de Kalman ***
void MouseThread::initKalmanFilter(double x, double y) {
    // 4 estados (x, y, vx, vy), 2 mediciones (x, y)
    kf.emplace(4, 2, 0, CV_64F); 
    last_kf_update_time = std::chrono::steady_clock::now();

    // Matriz de transición (modelo de movimiento)
    // Se actualiza dinámicamente con 'dt' en cada paso del bucle.
    cv::setIdentity(kf->transitionMatrix);

    // Matriz de medición (z_x = x, z_y = y)
    kf->measurementMatrix = cv::Mat::zeros(2, 4, CV_64F);
    kf->measurementMatrix.at<double>(0, 0) = 1.0;
    kf->measurementMatrix.at<double>(1, 1) = 1.0;

    // Covarianza del ruido del proceso (Q) - Confianza en el modelo de velocidad constante.
    cv::setIdentity(kf->processNoiseCov, cv::Scalar::all(1e-2));

    // Covarianza del ruido de la medición (R) - Confianza en la detección de YOLO (más ruido = valor más alto).
    cv::setIdentity(kf->measurementNoiseCov, cv::Scalar::all(1e-1));

    // Covarianza del error a posteriori (P) - Incertidumbre inicial.
    cv::setIdentity(kf->errorCovPost, cv::Scalar::all(1.0));

    // Estado inicial del filtro.
    kf->statePost.at<double>(0) = x;
    kf->statePost.at<double>(1) = y;
    kf->statePost.at<double>(2) = 0; // Velocidad X inicial
    kf->statePost.at<double>(3) = 0; // Velocidad Y inicial
}

// *** MODIFICADO: moveMousePivot AHORA USA EL FILTRO DE KALMAN ***
void MouseThread::moveMousePivot(double pivotX, double pivotY)
{
    if (!kf.has_value()) {
        // Primera detección del objetivo, inicializamos el filtro.
        initKalmanFilter(pivotX, pivotY);
        // Realizamos un movimiento reactivo inmediato para no perder el primer frame.
        auto m0 = calc_movement(pivotX, pivotY);
        queueMove(static_cast<int>(m0.first), static_cast<int>(m0.second));
        return;
    }

    auto current_time = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(current_time - last_kf_update_time).count();
    last_kf_update_time = current_time;
    
    // dt debe ser pequeño pero positivo para la estabilidad del filtro.
    dt = std::max(dt, 1e-5); 

    // 1. Actualizar la matriz de transición con el delta de tiempo actual.
    kf->transitionMatrix.at<double>(0, 2) = dt;
    kf->transitionMatrix.at<double>(1, 3) = dt;

    // 2. Predicción: El filtro estima dónde debería estar el objetivo ahora.
    kf->predict();

    // 3. Medición: Empaquetamos la nueva detección de YOLO.
    cv::Mat measurement = cv::Mat::zeros(2, 1, CV_64F);
    measurement.at<double>(0) = pivotX;
    measurement.at<double>(1) = pivotY;

    // 4. Corrección: El filtro fusiona su predicción con la medición real para obtener un estado suavizado.
    cv::Mat estimated = kf->correct(measurement);

    // 5. Apuntar: Usamos el estado suavizado para predecir hacia el futuro.
    //    Apuntamos a la posición estimada + velocidad_suavizada * intervalo_de_prediccion.
    double predX = estimated.at<double>(0) + estimated.at<double>(2) * this->prediction_interval;
    double predY = estimated.at<double>(1) + estimated.at<double>(3) * this->prediction_interval;

    // Actualizamos las variables `prev_` con los datos suavizados del filtro para la visualización.
    prev_velocity_x = estimated.at<double>(2);
    prev_velocity_y = estimated.at<double>(3);

    // Calcular el movimiento del mouse hacia el punto futuro predicho.
    auto mv = calc_movement(predX, predY);
    int mx = static_cast<int>(mv.first);
    int my = static_cast<int>(mv.second);

    if (this->wind_mouse_enabled) {
        windMouseMoveRelative(mx, my);
    } else {
        queueMove(mx, my);
    }
}


// *** MODIFICADO: calculate_speed_multiplier ahora implementa SNAP & LOCK ***
double MouseThread::calculate_speed_multiplier(double distance)
{
    // *** LÓGICA DE SNAP & LOCK ***
    // Si la distancia al objetivo es menor que el radio de "snap",
    // aplicamos un multiplicador de velocidad muy agresivo para "pegar" la mira.
    if (distance < config.snapRadius) {
        return this->min_speed_multiplier * config.snapBoostFactor; 
    }
    
    // *** LÓGICA DE TRACKING SUAVE (ZONA DE TRANSICIÓN) ***
    // Si estamos dentro del "nearRadius" pero fuera del "snapRadius",
    // usamos una curva para una transición suave en la velocidad.
    if (distance < config.nearRadius) {
        // Normalizamos la distancia dentro de esta zona de transición.
        double t = (distance - config.snapRadius) / (config.nearRadius - config.snapRadius);
        double curve = 1.0 - std::pow(1.0 - t, config.speedCurveExponent);
        return this->min_speed_multiplier + (this->max_speed_multiplier - this->min_speed_multiplier) * curve;
    }
    
    // *** LÓGICA A LARGA DISTANCIA ***
    // Fuera de todas las zonas, la velocidad es proporcional a la distancia.
    double norm = std::clamp(distance / this->max_distance, 0.0, 1.0);
    return this->min_speed_multiplier + (this->max_speed_multiplier - this->min_speed_multiplier) * norm;
}

// *** MODIFICADO: Reinicia el filtro de Kalman ***
void MouseThread::resetPrediction()
{
    kf.reset(); // Invalida y destruye el objeto KalmanFilter.
    prev_time = std::chrono::steady_clock::time_point();
    prev_x = 0; prev_y = 0;
    prev_velocity_x = 0; prev_velocity_y = 0;
    target_detected.store(false, std::memory_order_relaxed);
}

// *** MODIFICADO: predictFuturePositions usa la velocidad del filtro ***
std::vector<std::pair<double, double>> MouseThread::predictFuturePositions(double pivotX, double pivotY, int frames)
{
    std::vector<std::pair<double, double>> result;
    // Solo predecir si el filtro está activo y tenemos un estado válido.
    if (frames <= 0 || !kf.has_value()) return result;

    result.reserve(frames);
    const double frame_time = 1.0 / 60.0; // Frecuencia de refresco para visualización

    // Usamos el estado y velocidad actual del filtro, que están suavizados.
    double current_x = kf->statePost.at<double>(0);
    double current_y = kf->statePost.at<double>(1);
    double vel_x = kf->statePost.at<double>(2);
    double vel_y = kf->statePost.at<double>(3);

    for (int i = 1; i <= frames; i++) {
        double t_future = frame_time * i;
        result.push_back({ current_x + vel_x * t_future, current_y + vel_y * t_future });
    }
    return result;
}

// --- El resto del código no requiere modificaciones funcionales para la nueva lógica ---

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