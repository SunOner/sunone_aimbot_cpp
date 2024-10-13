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

using namespace std;

cv::Mat latestFrame;
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

void mouseThreadFunction(MouseThread& mouseThread)
{
    int lastDetectionVersion = -1;

    while (!shouldExit)
    {
        std::vector<cv::Rect> boxes;
        std::vector<int> classes;

        std::unique_lock<std::mutex> lock(detector.detectionMutex);
        detector.detectionCV.wait(lock, [&]() { return detector.detectionVersion > lastDetectionVersion || shouldExit; });
        if (shouldExit) break;

        lastDetectionVersion = detector.detectionVersion;

        boxes = detector.detectedBoxes;
        classes = detector.detectedClasses;
    
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
                // If no detections release mouse
                if (config.auto_shoot)
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
    if (!CreateDirectory(L"screenshots", NULL) && GetLastError() != ERROR_ALREADY_EXISTS)
    {
        return -1;
    }

    if (!config.loadConfig("config.ini"))
    {
        std::cerr << "[Config] Error with loading config.ini" << std::endl;
        cin.get();
        return -1;
    }

    if (config.arduino_enable)
    {
        cout << "[Mouse] Using Arduino method input." << endl;
        serial = new SerialConnection(config.arduino_port, config.arduino_baudrate);
    }

    if (config.arduino_enable == false && config.ghub)
    {
        cout << "[Mouse] Using Ghub method input." << endl;
        gHub = new GhubMouse();
        if (!gHub->mouse_xy(0, 0))
        {
            cerr << "[Ghub] Error with opening mouse." << endl;
            delete gHub;
            gHub = nullptr;
        }
    }

    if (config.arduino_enable && config.ghub)
    {
        cerr << "[Mouse] You use more than one mouse input method." << endl;
        cin.get();
        return -1;
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

    std::thread keyThread(keyboardListener);
    std::thread capThread(captureThread, config.detection_resolution, config.detection_resolution);
    std::thread detThread(&Detector::inferenceThread, &detector);
    std::thread dispThread(displayThread);
    std::thread mouseMovThread(mouseThreadFunction, std::ref(mouseThread));
    std::thread overlayThread(OverlayThread);

    keyThread.join();
    capThread.join();
    detThread.join();
    dispThread.join();
    mouseMovThread.join();
    overlayThread.join();

    if (serial)
    {
        delete serial;
    }

    if (gHub)
    {
        delete gHub;
    }

    return 0;
}