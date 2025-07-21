#ifndef MOUSE_H
#define MOUSE_H

#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>
#include <utility>
#include <optional>                     // *** NUEVO: Para el filtro opcional ***
#include <opencv2/video/tracking.hpp>   // *** NUEVO: Para el Filtro de Kalman de OpenCV ***

// Adelantar declaraciones
class SerialConnection;
class GhubMouse;
class Kmbox_b_Connection;
class KmboxNetConnection;
struct AimbotTarget;

struct Move {
    int dx;
    int dy;
};

class MouseThread
{
public:
    MouseThread(
        int resolution, int fovX, int fovY,
        double minSpeedMultiplier, double maxSpeedMultiplier,
        double predictionInterval, bool auto_shoot, float bScope_multiplier,
        SerialConnection* serialConnection = nullptr, GhubMouse* gHubMouse = nullptr,
        Kmbox_b_Connection* kmboxConnection = nullptr, KmboxNetConnection* Kmbox_Net_Connection = nullptr
    );
    ~MouseThread();

    void updateConfig(
        int resolution, int fovX, int fovY,
        double minSpeedMultiplier, double maxSpeedMultiplier,
        double predictionInterval, bool auto_shoot, float bScope_multiplier
    );

    void moveMousePivot(double pivotX, double pivotY);
    void pressMouse(const AimbotTarget& target);
    void releaseMouse();
    void resetPrediction(); // *** MODIFICADO: Ahora reiniciará el filtro de Kalman ***
    void checkAndResetPredictions();
    void sendMovementToDriver(int dx, int dy);

    std::vector<std::pair<double, double>> predictFuturePositions(double pivotX, double pivotY, int frames);
    void storeFuturePositions(const std::vector<std::pair<double, double>>& positions);
    void clearFuturePositions();
    std::vector<std::pair<double, double>> getFuturePositions();

    void setSerialConnection(SerialConnection* newSerial);
    void setKmboxConnection(Kmbox_b_Connection* newKmbox);
    void setKmboxNetConnection(KmboxNetConnection* newKmbox_net);
    void setGHubMouse(GhubMouse* newGHub);

    void setTargetDetected(bool detected) { target_detected.store(detected); }
    void setLastTargetTime(const std::chrono::steady_clock::time_point& t) { last_target_time = t; }

private:
    static const size_t QUEUE_SIZE = 256; 
    Move moveQueue_[QUEUE_SIZE];
    std::atomic<size_t> head_;
    std::atomic<size_t> tail_;
    
    std::thread moveWorker;
    std::atomic<bool> workerStop;

    void moveWorkerLoop();
    void queueMove(int dx, int dy);

    // *** NUEVO: Lógica del Filtro de Kalman para predicción avanzada ***
    std::optional<cv::KalmanFilter> kf;
    std::chrono::steady_clock::time_point last_kf_update_time;
    void initKalmanFilter(double x, double y);
    // *** FIN NUEVO ***

    std::mutex config_mutex;
    double screen_width, screen_height;
    double prediction_interval;
    double fov_x, fov_y;
    double max_distance;
    double min_speed_multiplier, max_speed_multiplier;
    double center_x, center_y;
    bool   auto_shoot;
    float  bScope_multiplier;

    // Estas variables ahora se actualizan con los datos suavizados del filtro
    double prev_x, prev_y;
    double prev_velocity_x, prev_velocity_y;

    int m_detection_resolution_w;
    int m_detection_resolution_h;
    float m_fov_radius_pixels;
    
    std::chrono::time_point<std::chrono::steady_clock> prev_time;
    std::chrono::steady_clock::time_point last_target_time;
    std::atomic<bool> target_detected;
    std::atomic<bool> mouse_pressed;

    SerialConnection* serial;
    Kmbox_b_Connection* kmbox;
    KmboxNetConnection* kmbox_net;
    GhubMouse* gHub;

    bool   wind_mouse_enabled;
    double wind_G, wind_W, wind_M, wind_D;
    void   windMouseMoveRelative(int dx, int dy);

    std::pair<double, double> calc_movement(double target_x, double target_y);
    double calculate_speed_multiplier(double distance);
    
    bool check_target_in_scope(double target_x, double target_y, double target_w, double target_h);
    
    std::vector<std::pair<double, double>> futurePositions;
    std::mutex futurePositionsMutex;
    std::mutex input_method_mutex;
};

#endif // MOUSE_H