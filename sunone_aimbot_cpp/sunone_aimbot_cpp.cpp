#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

#include "capture.h"
#include "visuals.h"
#include "detector.h"
#include "mouse.h"
#include "sunone_aimbot_cpp.h"
#include "keyboard_listener.h"
#include "overlay.h"
#include "SerialConnection.h"
#include "ghub.h"
#include "other_tools.h"
#include "optical_flow.h"
#include "kmboxNet.h"   // Added for KMBOX support

std::condition_variable frameCV;
std::atomic<bool> shouldExit(false);
std::atomic<bool> aiming(false);
std::atomic<bool> detectionPaused(false);
std::mutex configMutex;

Detector detector;
MouseThread* globalMouseThread = nullptr;
Config config;

GhubMouse* gHub = nullptr;
SerialConnection* serial = nullptr;

OpticalFlow opticalFlow;

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

void initializeInputMethod()
{
    {
        std::lock_guard<std::mutex> lock(globalMouseThread->input_method_mutex);

        if (serial)
        {
            delete serial;
            serial = nullptr;
        }

        if (gHub)
        {
            gHub->mouse_close();
            delete gHub;
            gHub = nullptr;
        }
    }

    if (config.input_method == "ARDUINO")
    {
        std::cout << "[Mouse] Using Arduino method input." << std::endl;
        serial = new SerialConnection(config.arduino_port, config.arduino_baudrate);
    }
    else if (config.input_method == "GHUB")
    {
        std::cout << "[Mouse] Using Ghub method input." << std::endl;
        gHub = new GhubMouse();
        if (!gHub->mouse_xy(0, 0))
        {
            std::cerr << "[Ghub] Error with opening mouse." << std::endl;
            delete gHub;
            gHub = nullptr;
        }
    }
    else if (config.input_method == "KMBOX")
    {
        std::cout << "[Mouse] Using KMBOX method input." << std::endl;
        // Call kmNet_init with KMBOX connection parameters.
        // Ensure that config has kmbox_ip, kmbox_port, and kmbox_mac fields.
        int ret = kmNet_init(const_cast<char*>(config.kmbox_ip.c_str()),
            const_cast<char*>(config.kmbox_port.c_str()),
            const_cast<char*>(config.kmbox_mac.c_str()));
        if (ret != 0)
        {
            std::cerr << "[KMBOX] Error connecting to KMBOX device. Return code: " << ret << std::endl;
        }
        else
        {
            std::cout << "[KMBOX] Connection established." << std::endl;
        }
        // Set the MouseThread to use KMBOX mode.
        globalMouseThread->setKmboxMode(true);
    }
    else
    {
        std::cout << "[Mouse] Using default Win32 method input." << std::endl;
    }

    globalMouseThread->setSerialConnection(serial);
    globalMouseThread->setGHubMouse(gHub);
}

void handleEasyNoRecoil(MouseThread& mouseThread)
{
    if (config.easynorecoil && shooting.load() && zooming.load())
    {
        std::lock_guard<std::mutex> lock(mouseThread.input_method_mutex);
        int recoil_compensation = static_cast<int>(config.easynorecoilstrength);
        
        if (serial)
        {
            serial->move(0, recoil_compensation);
        }
        else if (gHub)
        {
            gHub->mouse_xy(0, recoil_compensation);
        }
        else
        {
            INPUT input = { 0 };
            input.type = INPUT_MOUSE;
            input.mi.dx = 0;
            input.mi.dy = recoil_compensation;
            input.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_VIRTUALDESK;
            SendInput(1, &input, sizeof(INPUT));
        }
    }
}

void mouseThreadFunction(MouseThread& mouseThread)
{
    int lastDetectionVersion = -1;

    std::chrono::milliseconds timeout(30);

    while (!shouldExit)
    {
        std::vector<cv::Rect> boxes;
        std::vector<int> classes;

        std::unique_lock<std::mutex> lock(detector.detectionMutex);

        detector.detectionCV.wait_for(lock, timeout, [&]() { return detector.detectionVersion > lastDetectionVersion || shouldExit; });

        if (shouldExit) break;

        if (detector.detectionVersion <= lastDetectionVersion) continue;

        lastDetectionVersion = detector.detectionVersion;

        boxes = detector.detectedBoxes;
        classes = detector.detectedClasses;

        if (input_method_changed.load())
        {
            initializeInputMethod();
            input_method_changed.store(false);
        }

        if (detection_resolution_changed.load())
        {
            {
                std::lock_guard<std::mutex> lock(configMutex);
                mouseThread.updateConfig(
                    config.detection_resolution,
                    config.dpi,
                    config.sensitivity,
                    config.fovX,
                    config.fovY,
                    config.minSpeedMultiplier,
                    config.maxSpeedMultiplier,
                    config.predictionInterval,
                    config.auto_shoot,
                    config.bScope_multiplier
                );
            }
            detection_resolution_changed.store(false);
        }

        AimbotTarget* target = sortTargets(boxes, classes, config.detection_resolution, config.detection_resolution, config.disable_headshot);

        if (aiming)
        {
            if (target)
            {
                mouseThread.moveMouse(*target);

                if (config.auto_shoot)
                {
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
        delete target;
    }
}

int main()
{
    try
    {
        int cuda_devices = 0;
        cudaError_t err = cudaGetDeviceCount(&cuda_devices);

        if (err != cudaSuccess)
        {
            std::cout << "[MAIN] No GPU devices with CUDA support available." << std::endl;
            std::cin.get();
            return -1;
        }

        if (!CreateDirectory(L"screenshots", NULL) && GetLastError() != ERROR_ALREADY_EXISTS)
        {
            std::cout << "[MAIN] Error with screenshoot folder" << std::endl;
            std::cin.get();
            return -1;
        }

        if (!config.loadConfig())
        {
            std::cerr << "[Config] Error with loading config!" << std::endl;
            std::cin.get();
            return -1;
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
                std::cerr << "[MAIN] No models found in 'models' directory." << std::endl;
                std::cin.get();
                return -1;
            }
        }

        if (config.input_method == "ARDUINO")
        {
            serial = new SerialConnection(config.arduino_port, config.arduino_baudrate);
        }
        else if (config.input_method == "GHUB")
        {
            gHub = new GhubMouse();
            if (!gHub->mouse_xy(0, 0))
            {
                std::cerr << "[Ghub] Error with opening mouse." << std::endl;
                delete gHub;
                gHub = nullptr;
            }
        }

        MouseThread mouseThread(
            config.detection_resolution,
            config.dpi,
            config.sensitivity,
            config.fovX,
            config.fovY,
            config.minSpeedMultiplier,
            config.maxSpeedMultiplier,
            config.predictionInterval,
            config.auto_shoot,
            config.bScope_multiplier,
            serial,
            gHub
        );

        globalMouseThread = &mouseThread;

        std::vector<std::string> availableModels = getAvailableModels();

        if (!config.ai_model.empty())
        {
            std::string modelPath = "models/" + config.ai_model;
            if (!std::filesystem::exists(modelPath))
            {
                std::cerr << "[MAIN] Specified model does not exist: " << modelPath << std::endl;

                if (!availableModels.empty())
                {
                    config.ai_model = availableModels[0];
                    config.saveConfig("config.ini");
                    std::cout << "[MAIN] Loaded first available model: " << config.ai_model << std::endl;
                }
                else
                {
                    std::cerr << "[MAIN] No models found in 'models' directory." << std::endl;
                    std::cin.get();
                    return -1;
                }
            }
        }
        else
        {
            if (!availableModels.empty())
            {
                config.ai_model = availableModels[0];
                config.saveConfig();
                std::cout << "[MAIN] No AI model specified in config. Loaded first available model: " << config.ai_model << std::endl;
            }
            else
            {
                std::cerr << "[MAIN] No AI models found in 'models' directory." << std::endl;
                std::cin.get();
                return -1;
            }
        }

        detector.initialize("models/" + config.ai_model);
        detection_resolution_changed.store(true);

        initializeInputMethod();

        std::thread keyThread(keyboardListener);
        std::thread capThread(captureThread, config.detection_resolution, config.detection_resolution);
        std::thread detThread(&Detector::inferenceThread, &detector);
        std::thread mouseMovThread(mouseThreadFunction, std::ref(mouseThread));
        std::thread overlayThread(OverlayThread);
        opticalFlow.startOpticalFlowThread();

        welcome_message();

        displayThread();

        keyThread.join();
        capThread.join();
        detThread.join();
        mouseMovThread.join();
        overlayThread.join();

        if (serial)
        {
            delete serial;
        }

        if (gHub)
        {
            gHub->mouse_close();
            delete gHub;
        }

        opticalFlow.stopOpticalFlowThread();

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[MAIN] An error has occurred in the main stream: " << e.what() << std::endl;
        std::cout << "Press Enter to exit...";
        std::cin.get();
        return -1;
    }
}
