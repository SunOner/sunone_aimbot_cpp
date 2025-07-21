#include "mouse.h" // Incluir la cabecera corregida

#include <cmath>
#include <algorithm>
#include <numeric>

#include "SerialConnection.h"
#include "ghub.h"
#include "Kmbox_b.h"
#include "KmboxNetConnection.h"
#include "AimbotTarget.h"

// --- Constructor ---
// Inicializa todos los miembros a valores seguros.
MouseThread::MouseThread(
    SerialConnection* serialConnection, GhubMouse* gHubMouse,
    Kmbox_b_Connection* kmboxConnection, KmboxNetConnection* kmboxNetConn)
    : m_detection_resolution_w(0), m_detection_resolution_h(0), m_fov_radius_pixels(0.0f),
      m_minSpeedMultiplier(0.0), m_maxSpeedMultiplier(0.0), m_predictionInterval(0.0),
      m_bScope_multiplier(1.0f), m_auto_shoot_enabled(false), m_target_detected(false),
      mouse_pressed(false), serial(serialConnection), kmbox(kmboxConnection),
      kmbox_net(kmboxNetConn), gHub(gHubMouse)
{
    m_last_prediction_time = std::chrono::steady_clock::now();
    m_last_target_seen_time = std::chrono::steady_clock::now();
}

// --- Destructor ---
// (No necesita hacer nada especial ahora que no hay hilos de trabajo)
MouseThread::~MouseThread()
{
}

// --- updateConfig ---
// Centraliza la actualización de todos los parámetros desde el config
void MouseThread::updateConfig(
    int resolution, float fovX,
    double minSpeedMultiplier, double maxSpeedMultiplier,
    double predictionInterval, bool auto_shoot, float bScope_multiplier)
{
    std::lock_guard<std::mutex> lock(m_prediction_mutex); // Protege el acceso a los miembros

    m_detection_resolution_w = resolution;
    m_detection_resolution_h = resolution;
    m_minSpeedMultiplier = minSpeedMultiplier;
    m_maxSpeedMultiplier = maxSpeedMultiplier;
    m_predictionInterval = predictionInterval;
    m_auto_shoot_enabled = auto_shoot;
    m_bScope_multiplier = bScope_multiplier;

    // Pre-calcula el radio del FOV en píxeles para optimizar los cálculos en cada frame
    m_fov_radius_pixels = (static_cast<float>(m_detection_resolution_w) * (fovX / 100.0f)) / 2.0f;
}

// --- storeTargetPosition ---
// Guarda la posición actual del objetivo para el cálculo de predicción
void MouseThread::storeTargetPosition(double x, double y)
{
    std::lock_guard<std::mutex> lock(m_prediction_mutex);
    m_last_target_positions.push_back({x, y});
    // Mantiene el historial a un tamaño razonable para evitar que crezca indefinidamente
    if (m_last_target_positions.size() > 10) {
        m_last_target_positions.erase(m_last_target_positions.begin());
    }
    m_last_target_seen_time = std::chrono::steady_clock::now();
    m_target_detected = true;
}

// --- calculateAimMovement ---
// El cerebro del aimbot: calcula el delta de movimiento (dx, dy)
std::pair<int, int> MouseThread::calculateAimMovement(double target_pivot_x, double target_pivot_y)
{
    storeTargetPosition(target_pivot_x, target_pivot_y);

    double predicted_x = target_pivot_x;
    double predicted_y = target_pivot_y;
    
    // --- Paso 1: Predicción de Movimiento ---
    {
        std::lock_guard<std::mutex> lock(m_prediction_mutex);
        if (m_last_target_positions.size() > 1)
        {
            double dx_sum = 0.0, dy_sum = 0.0;
            for (size_t i = 1; i < m_last_target_positions.size(); ++i) {
                dx_sum += m_last_target_positions[i].first - m_last_target_positions[i - 1].first;
                dy_sum += m_last_target_positions[i].second - m_last_target_positions[i - 1].second;
            }
            double avg_dx = dx_sum / (m_last_target_positions.size() - 1);
            double avg_dy = dy_sum / (m_last_target_positions.size() - 1);

            auto now = std::chrono::steady_clock::now();
            double time_delta_ms = std::chrono::duration_cast<std::chrono::microseconds>(now - m_last_prediction_time).count() / 1000.0;
            m_last_prediction_time = now;
            
            double prediction_factor = (m_predictionInterval + time_delta_ms) / 16.666; // Normalizado a un frametime de 60Hz
            
            predicted_x += avg_dx * prediction_factor;
            predicted_y += avg_dy * prediction_factor;
        }
    }

    // --- Paso 2: Calcular Vector al Objetivo ---
    const double screen_center_x = static_cast<double>(m_detection_resolution_w) / 2.0;
    const double screen_center_y = static_cast<double>(m_detection_resolution_h) / 2.0;
    double dx = predicted_x - screen_center_x;
    double dy = predicted_y - screen_center_y;

    // --- Paso 3: Chequeo de FOV ---
    const double distance = std::hypot(dx, dy);
    if (distance > m_fov_radius_pixels || distance < 1.0)
    {
        return {0, 0}; // Fuera de FOV o ya en el objetivo, no mover.
    }

    // --- Paso 4: Aplicar Multiplicador de Zoom ---
    if (zooming.load(std::memory_order_relaxed))
    {
        dx *= m_bScope_multiplier;
        dy *= m_bScope_multiplier;
    }

    // --- Paso 5: Velocidad Dinámica y Suavizado ---
    float speed_factor = static_cast<float>(distance / m_fov_radius_pixels);
    speed_factor = std::min(1.0f, speed_factor);
    double final_speed_multiplier = m_minSpeedMultiplier + (m_maxSpeedMultiplier - m_minSpeedMultiplier) * speed_factor;
    double smoothed_dx = dx * final_speed_multiplier;
    double smoothed_dy = dy * final_speed_multiplier;

    // --- Paso 6: Evitar "Píxel Muerto" ---
    if (std::abs(smoothed_dx) < 1.0 && std::abs(dx) > 0.5) smoothed_dx = (smoothed_dx > 0) ? 1.0 : -1.0;
    if (std::abs(smoothed_dy) < 1.0 && std::abs(dy) > 0.5) smoothed_dy = (smoothed_dy > 0) ? 1.0 : -1.0;
    
    return { static_cast<int>(smoothed_dx), static_cast<int>(smoothed_dy) };
}

// --- sendMovementToDriver ---
// Envía el movimiento de forma síncrona al dispositivo de hardware seleccionado
void MouseThread::sendMovementToDriver(int dx, int dy)
{
    if (dx == 0 && dy == 0) return;
    std::lock_guard<std::mutex> lock(input_method_mutex);
    if (kmbox) { kmbox->move(dx, dy); }
    else if (kmbox_net) { kmbox_net->move(dx, dy); }
    else if (serial) { serial->move(dx, dy); }
    else if (gHub) { gHub->mouse_xy(dx, dy); }
    else { mouse_event(MOUSEEVENTF_MOVE, dx, dy, 0, 0); }
}

// --- isCursorInTarget ---
// Función auxiliar para comprobar si el centro de la pantalla está dentro de la caja del objetivo
bool MouseThread::isCursorInTarget(double target_x, double target_y, double target_w, double target_h)
{
    const double center_x = static_cast<double>(m_detection_resolution_w) / 2.0;
    const double center_y = static_cast<double>(m_detection_resolution_h) / 2.0;
    return (center_x >= target_x && center_x <= (target_x + target_w) &&
            center_y >= target_y && center_y <= (target_y + target_h));
}

// --- pressMouse ---
// Presiona el botón del ratón si el auto-disparo está activo y el cursor está sobre el objetivo
void MouseThread::pressMouse(const AimbotTarget& target)
{
    if (!m_auto_shoot_enabled) return;

    if (isCursorInTarget(target.x, target.y, target.w, target.h) && !mouse_pressed.load(std::memory_order_relaxed))
    {
        std::lock_guard<std::mutex> lock(input_method_mutex);
        if (kmbox) kmbox->press(0);
        else if (kmbox_net) kmbox_net->keyDown(0);
        else if (serial) serial->press();
        else if (gHub) gHub->mouse_down();
        else { /* WinAPI press */ INPUT input = {0}; input.type = INPUT_MOUSE; input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN; SendInput(1, &input, sizeof(INPUT)); }
        mouse_pressed.store(true, std::memory_order_relaxed);
    }
}

// --- releaseMouse ---
// Libera el botón del ratón si estaba presionado
void MouseThread::releaseMouse()
{
    if (!m_auto_shoot_enabled) return;

    if (mouse_pressed.load(std::memory_order_relaxed))
    {
        std::lock_guard<std::mutex> lock(input_method_mutex);
        if (kmbox) kmbox->release(0);
        else if (kmbox_net) kmbox_net->keyUp(0);
        else if (serial) serial->release();
        else if (gHub) gHub->mouse_up();
        else { /* WinAPI release */ INPUT input = {0}; input.type = INPUT_MOUSE; input.mi.dwFlags = MOUSEEVENTF_LEFTUP; SendInput(1, &input, sizeof(INPUT)); }
        mouse_pressed.store(false, std::memory_order_relaxed);
    }
}

// --- clearPredictionHistory ---
// Limpia el historial de posiciones del objetivo
void MouseThread::clearPredictionHistory()
{
    std::lock_guard<std::mutex> lock(m_prediction_mutex);
    m_last_target_positions.clear();
    m_target_detected = false;
}

// --- checkAndResetPredictions ---
// Si no se ha visto un objetivo por un tiempo, se resetea la predicción
void MouseThread::checkAndResetPredictions()
{
    auto now = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_last_target_seen_time).count();
    
    if (m_target_detected && elapsed_ms > 200) // Si han pasado 200ms sin ver un objetivo
    {
        clearPredictionHistory();
    }
}

// --- Setters para dispositivos ---
void MouseThread::setSerialConnection(SerialConnection* newSerial) { std::lock_guard<std::mutex> lock(input_method_mutex); serial = newSerial; }
void MouseThread::setKmboxConnection(Kmbox_b_Connection* newKmbox) { std::lock_guard<std::mutex> lock(input_method_mutex); kmbox = newKmbox; }
void MouseThread::setKmboxNetConnection(KmboxNetConnection* newKmbox_net) { std::lock_guard<std::mutex> lock(input_method_mutex); kmbox_net = newKmbox_net; }
void MouseThread::setGHubMouse(GhubMouse* newGHub) { std::lock_guard<std::mutex> lock(input_method_mutex); gHub = newGHub; }

// mouse.cpp