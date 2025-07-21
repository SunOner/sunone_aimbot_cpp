#ifndef MOUSE_H
#define MOUSE_H

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <vector>
#include <utility>

// Declaraciones adelantadas para evitar inclusiones circulares
class SerialConnection;
class GhubMouse;
class Kmbox_b_Connection;
class KmboxNetConnection;
struct AimbotTarget;

// Declaración externa para acceder a la variable global de zoom
extern std::atomic<bool> zooming;

class MouseThread
{
public:
    // Constructor simplificado
    MouseThread(
        SerialConnection* serialConnection = nullptr, GhubMouse* gHubMouse = nullptr,
        Kmbox_b_Connection* kmboxConnection = nullptr, KmboxNetConnection* Kmbox_Net_Connection = nullptr
    );
    
    // Destructor
    ~MouseThread();

    // Actualiza toda la configuración relevante para el aimbot
    void updateConfig(
        int resolution, float fovX,
        double minSpeedMultiplier, double maxSpeedMultiplier,
        double predictionInterval, bool auto_shoot, float bScope_multiplier
    );

    // La función principal que calcula el movimiento de apuntado
    std::pair<int, int> calculateAimMovement(double target_pivot_x, double target_pivot_y);
    
    // Envía el movimiento directamente al driver de hardware
    void sendMovementToDriver(int dx, int dy);
    
    // Maneja la pulsación y liberación del botón del ratón para auto-disparo
    void pressMouse(const AimbotTarget& target);
    void releaseMouse();

    // Lógica de predicción
    void storeTargetPosition(double x, double y);
    void checkAndResetPredictions();
    void clearPredictionHistory();

    // Setters para los dispositivos de entrada
    void setSerialConnection(SerialConnection* newSerial);
    void setKmboxConnection(Kmbox_b_Connection* newKmbox);
    void setKmboxNetConnection(KmboxNetConnection* newKmbox_net);
    void setGHubMouse(GhubMouse* newGHub);

private:
    // --- Miembros para la Lógica de Aimbot ---
    int   m_detection_resolution_w;
    int   m_detection_resolution_h;
    float m_fov_radius_pixels;
    double m_minSpeedMultiplier;
    double m_maxSpeedMultiplier;
    double m_predictionInterval;
    float m_bScope_multiplier;
    bool  m_auto_shoot_enabled;
    
    // --- Miembros para la Predicción de Movimiento ---
    std::mutex m_prediction_mutex;
    std::vector<std::pair<double, double>> m_last_target_positions;
    std::chrono::steady_clock::time_point m_last_prediction_time;
    std::chrono::steady_clock::time_point m_last_target_seen_time;
    bool m_target_detected;

    // --- Miembros para Control de Hardware ---
    std::mutex input_method_mutex;
    std::atomic<bool> mouse_pressed;
    SerialConnection* serial;
    Kmbox_b_Connection* kmbox;
    KmboxNetConnection* kmbox_net;
    GhubMouse* gHub;

    // Función auxiliar para comprobar si el cursor está sobre el objetivo
    bool isCursorInTarget(double target_x, double target_y, double target_w, double target_h);
};

#endif // MOUSE_H