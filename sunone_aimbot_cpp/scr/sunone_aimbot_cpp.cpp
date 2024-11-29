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

std::atomic<bool> detection_resolution_changed(false);
std::atomic<bool> capture_method_changed(false);
std::atomic<bool> capture_cursor_changed(false);
std::atomic<bool> capture_borders_changed(false);
std::atomic<bool> capture_fps_changed(false);
std::atomic<bool> detector_model_changed(false);
std::atomic<bool> show_window_changed(false);
std::atomic<bool> input_method_changed(false);

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
    else
    {
        std::cout << "[Mouse] Using default Win32 method input." << std::endl;
    }

    globalMouseThread->setSerialConnection(serial);
    globalMouseThread->setGHubMouse(gHub);
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

        if (aiming)
        {
            Target* target = sortTargets(boxes, classes, config.detection_resolution, config.detection_resolution, config.disable_headshot);
            if (target)
            {
                mouseThread.moveMouse(*target);
                if (config.auto_shoot)
                {
                    mouseThread.pressMouse(*target);
                }
                delete target;
            }
            else
            {
                // release mouse button
                if (!aiming && config.auto_shoot)
                {
                    mouseThread.releaseMouse();
                }
            }
        }

        mouseThread.checkAndResetPredictions();
    }
}

int main()
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

    if (!config.loadConfig("config.ini"))
    {
        std::cerr << "[Config] Error with loading config.ini" << std::endl;
        std::cin.get();
        return -1;
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

    detector.initialize("models/" + config.ai_model);

    initializeInputMethod();

    std::thread keyThread(keyboardListener);
    std::thread capThread(captureThread, config.detection_resolution, config.detection_resolution);
    std::thread detThread(&Detector::inferenceThread, &detector);
    std::thread mouseMovThread(mouseThreadFunction, std::ref(mouseThread));
    std::thread overlayThread(OverlayThread);

    displayThread();

    if (config.ai_model.empty())
    {
        std::cout << "[MAIN] No AI model specified in config. Please select an AI model in the overlay." << std::endl;
    }

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

    return 0;
}