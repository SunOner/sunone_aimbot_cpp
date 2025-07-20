#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <optional> // *** OPTIMIZACIÓN: Incluir para std::optional ***
#include <filesystem> // Para std::filesystem::exists

#include "capture.h"
#include "mouse.h"
#include "sunone_aimbot_cpp.h"
#include "keyboard_listener.h"
#include "overlay.h"
#include "ghub.h"
#include "other_tools.h"
#include "virtual_camera.h"
#include "Kmbox_b.h"
#include "KmboxNetConnection.h"
#include "SerialConnection.h"
#ifdef USE_CUDA
#include "trt_detector.h"
#endif
#include "dml_detector.h"


// --- Variables Globales ---
std::condition_variable frameCV;
std::atomic<bool> shouldExit(false);
std::atomic<bool> aiming(false);
std::atomic<bool> detectionPaused(false);
std::mutex configMutex;

#ifdef USE_CUDA
TrtDetector trt_detector;
#endif

DirectMLDetector* dml_detector = nullptr;
MouseThread* globalMouseThread = nullptr;
Config config;

GhubMouse* gHub = nullptr;
SerialConnection* arduinoSerial = nullptr;
Kmbox_b_Connection* kmboxSerial = nullptr;
KmboxNetConnection* kmboxNetSerial = nullptr;

// Flags de cambio de configuración
std::atomic<bool> detection_resolution_changed(false);
std::atomic<bool> capture_method_changed(false);
std::atomic<bool> capture_cursor_changed(false);
std::atomic<bool> capture_borders_changed(false);
std::atomic<bool> capture_fps_changed(false);
std::atomic<bool> capture_window_changed(false);
std::atomic<bool> detector_model_changed(false);
std::atomic<bool> show_window_changed(false);
std::atomic<bool> input_method_changed(false);

std::atomic<bool> zooming(false);
std::atomic<bool> shooting(false);

// --- Funciones de Gestión de Dispositivos (sin cambios funcionales, solo limpieza) ---

void createInputDevices()
{
    if (arduinoSerial) { delete arduinoSerial; arduinoSerial = nullptr; }
    if (gHub) { gHub->mouse_close(); delete gHub; gHub = nullptr; }
    if (kmboxSerial) { delete kmboxSerial; kmboxSerial = nullptr; }
    if (kmboxNetSerial) { delete kmboxNetSerial; kmboxNetSerial = nullptr; }

    if (config.input_method == "ARDUINO")
    {
        std::cout << "[Mouse] Using Arduino method input." << std::endl;
        arduinoSerial = new SerialConnection(config.arduino_port, config.arduino_baudrate);
    }
    else if (config.input_method == "GHUB")
    {
        std::cout << "[Mouse] Using Ghub method input." << std::endl;
        gHub = new GhubMouse();
        if (!gHub->mouse_xy(0, 0)) {
            std::cerr << "[Ghub] Error with opening mouse." << std::endl;
            delete gHub; gHub = nullptr;
        }
    }
    else if (config.input_method == "KMBOX_B")
    {
        std::cout << "[Mouse] Using KMBOX_B method input." << std::endl;
        kmboxSerial = new Kmbox_b_Connection(config.kmbox_b_port, config.kmbox_b_baudrate);
        if (!kmboxSerial->isOpen()) {
            std::cerr << "[Kmbox] Error connecting to Kmbox serial." << std::endl;
            delete kmboxSerial; kmboxSerial = nullptr;
        }
    }
    else if (config.input_method == "KMBOX_NET")
    {
        std::cout << "[Mouse] Using KMBOX_NET input." << std::endl;
        kmboxNetSerial = new KmboxNetConnection(config.kmbox_net_ip, config.kmbox_net_port, config.kmbox_net_uuid);
        if (!kmboxNetSerial->isOpen()) {
            std::cerr << "[KmboxNet] Error connecting." << std::endl;
            delete kmboxNetSerial; kmboxNetSerial = nullptr;
        }
    }
    else
    {
        std::cout << "[Mouse] Using default Win32 method input." << std::endl;
    }
}

void assignInputDevices()
{
    if (globalMouseThread)
    {
        globalMouseThread->setSerialConnection(arduinoSerial);
        globalMouseThread->setGHubMouse(gHub);
        globalMouseThread->setKmboxConnection(kmboxSerial);
        globalMouseThread->setKmboxNetConnection(kmboxNetSerial);
    }
}

void handleEasyNoRecoil(MouseThread& mouseThread)
{
    if (config.easynorecoil && shooting.load(std::memory_order_relaxed) && zooming.load(std::memory_order_relaxed))
    {
        std::lock_guard<std::mutex> lock(mouseThread.input_method_mutex);
        int recoil_compensation = static_cast<int>(config.easynorecoilstrength);

        mouseThread.sendMovementToDriver(0, recoil_compensation); // Simplificado para usar el método central
    }
}

// *** FUNCIÓN DEL HILO PRINCIPAL OPTIMIZADA ***
void mouseThreadFunction(MouseThread& mouseThread)
{
    int lastVersion = -1;

    while (!shouldExit)
    {
        std::vector<cv::Rect> boxes;
        std::vector<int> classes;

        {
            std::unique_lock<std::mutex> lock(detectionBuffer.mutex);
            detectionBuffer.cv.wait(lock, [&] {
                return detectionBuffer.version > lastVersion || shouldExit;
            });
            if (shouldExit) break;
            boxes = detectionBuffer.boxes;
            classes = detectionBuffer.classes;
            lastVersion = detectionBuffer.version;
        }

        if (input_method_changed.load(std::memory_order_relaxed))
        {
            createInputDevices();
            assignInputDevices();
            input_method_changed.store(false, std::memory_order_relaxed);
        }

        if (detection_resolution_changed.load(std::memory_order_relaxed))
        {
            {
                std::lock_guard<std::mutex> cfgLock(configMutex);
                mouseThread.updateConfig(
                    config.detection_resolution, config.fovX, config.fovY,
                    config.minSpeedMultiplier, config.maxSpeedMultiplier,
                    config.predictionInterval, config.auto_shoot, config.bScope_multiplier);
            }
            detection_resolution_changed.store(false, std::memory_order_relaxed);
        }

        // *** OPTIMIZACIÓN: Usar std::optional para evitar alocaciones en el heap (new/delete) ***
        std::optional<AimbotTarget> target = sortTargets(
            boxes,
            classes,
            config.detection_resolution,
            config.detection_resolution,
            config.disable_headshot
        );

        if (target.has_value()) // o simplemente `if (target)`
        {
            mouseThread.setLastTargetTime(std::chrono::steady_clock::now());
            mouseThread.setTargetDetected(true);

            auto futurePositions = mouseThread.predictFuturePositions(
                target->pivotX,
                target->pivotY,
                config.prediction_futurePositions
            );
            mouseThread.storeFuturePositions(futurePositions);
        }
        else
        {
            mouseThread.clearFuturePositions();
            mouseThread.setTargetDetected(false);
        }

        if (aiming.load(std::memory_order_relaxed))
        {
            if (target) // Se puede usar como un booleano
            {
                mouseThread.moveMousePivot(target->pivotX, target->pivotY);

                if (config.auto_shoot)
                {
                    // El operador `*` devuelve la referencia al objeto contenido
                    mouseThread.pressMouse(*target);
                }
            }
            else
            {
                if (config.auto_shoot)
                {
                    mouseThread.releaseMouse();
                }
            }
        }
        else
        {
            if (config.auto_shoot)
            {
                mouseThread.releaseMouse();
            }
        }

        handleEasyNoRecoil(mouseThread);
        mouseThread.checkAndResetPredictions();
        
        // No hay 'delete target', la memoria se libera automáticamente al salir del ámbito del bucle.
    }
}

int main()
{
    try
    {
#ifdef USE_CUDA
        int cuda_devices = 0;
        if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0)
        {
            std::cerr << "[MAIN] CUDA required but no devices found." << std::endl;
            std::cin.get();
            return -1;
        }
#endif

        SetConsoleOutputCP(CP_UTF8);
        cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);

        if (!CreateDirectory(L"screenshots", NULL) && GetLastError() != ERROR_ALREADY_EXISTS)
        {
            std::cout << "[MAIN] Error with screenshot folder" << std::endl;
            std::cin.get();
            return -1;
        }

        if (!CreateDirectory(L"models", NULL) && GetLastError() != ERROR_ALREADY_EXISTS)
        {
            std::cout << "[MAIN] Error with models folder" << std::endl;
            std::cin.get();
            return -1;
        }

        if (!config.loadConfig())
        {
            std::cerr << "[Config] Error with loading config!" << std::endl;
            std::cin.get();
            return -1;
        }

        if (config.capture_method == "virtual_camera")
        {
            auto cams = VirtualCameraCapture::GetAvailableVirtualCameras();
            if (!cams.empty())
            {
                if (config.virtual_camera_name == "None" ||
                    std::find(cams.begin(), cams.end(), config.virtual_camera_name) == cams.end())
                {
                    config.virtual_camera_name = cams[0];
                    config.saveConfig("config.ini");
                    std::cout << "[MAIN] Set virtual_camera_name = " << config.virtual_camera_name << std::endl;
                }
                std::cout << "[MAIN] Virtual cameras loaded: " << cams.size() << std::endl;
            }
            else
            {
                std::cerr << "[MAIN] No virtual cameras found" << std::endl;
            }
        }
        
        std::string modelPath = "models/" + config.ai_model;
        if (!std::filesystem::exists(modelPath))
        {
            std::cerr << "[MAIN] Specified model does not exist: " << modelPath << std::endl;
            std::vector<std::string> modelFiles = getModelFiles();
            if (!modelFiles.empty())
            {
                config.ai_model = modelFiles[0];
                config.saveConfig();
                std::cout << "[MAIN] Loaded first available model: " << config.ai_model << std::endl;
            }
            else
            {
                std::cerr << "[MAIN] No models found in 'models' directory." << std::cin.get(); return -1;
            }
        }

        createInputDevices();

        MouseThread mouseThread(
            config.detection_resolution, config.fovX, config.fovY,
            config.minSpeedMultiplier, config.maxSpeedMultiplier,
            config.predictionInterval, config.auto_shoot, config.bScope_multiplier,
            arduinoSerial, gHub, kmboxSerial, kmboxNetSerial
        );

        globalMouseThread = &mouseThread;
        assignInputDevices();

        std::thread dml_detThread;
        std::thread trt_detThread;

        if (config.backend == "DML")
        {
            dml_detector = new DirectMLDetector("models/" + config.ai_model);
            std::cout << "[MAIN] DML detector initialized." << std::endl;
            dml_detThread = std::thread(&DirectMLDetector::dmlInferenceThread, dml_detector);
        }
#ifdef USE_CUDA
        else if (config.backend == "TRT")
        {
            trt_detector.initialize("models/" + config.ai_model);
            trt_detThread = std::thread(&TrtDetector::inferenceThread, &trt_detector);
        }
#endif

        detection_resolution_changed.store(true);
        std::thread keyThread(keyboardListener);
        std::thread capThread(captureThread, config.detection_resolution, config.detection_resolution);
        std::thread mouseMovThread(mouseThreadFunction, std::ref(mouseThread));
        std::thread overlayThread(OverlayThread);

        welcome_message();

        keyThread.join();
        capThread.join();

        if (dml_detThread.joinable())
        {
            if (dml_detector) dml_detector->shouldExit = true;
            if (dml_detector) dml_detector->inferenceCV.notify_all();
            dml_detThread.join();
        }

#ifdef USE_CUDA
        if (trt_detThread.joinable())
        {
            trt_detector.notifyExit();
            trt_detThread.join();
        }
#endif
        mouseMovThread.join();
        overlayThread.join();

        // Limpieza final
        if (arduinoSerial) delete arduinoSerial;
        if (gHub) { gHub->mouse_close(); delete gHub; }
        if (kmboxSerial) delete kmboxSerial;
        if (kmboxNetSerial) delete kmboxNetSerial;
        if (dml_detector) delete dml_detector;

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[MAIN] An error has occurred in the main stream: " << e.what() << std::endl;
        std::cout << "Press Enter to exit...";
        std::cin.get();
        return -1;
    }
} // sunone_aimbot_cpp.cpp