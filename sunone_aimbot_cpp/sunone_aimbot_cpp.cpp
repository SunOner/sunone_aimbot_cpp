#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <Windows.h>

#include "capture.h"
#include "visuals.h"
#include "detector.h"
#include "mouse.h"
#include "target.h" 
#include "sunone_aimbot_cpp.h"
#include "detector.h"
#include "config.h"
#include "keyboard_listener.h"

#pragma comment(lib, "nvinfer_10.lib")
#pragma comment(lib, "nvonnxparser_10.lib")
#pragma comment(lib, "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/lib/x64/cudart.lib")

using namespace cv;
using namespace std;

Mat latestFrame;
std::condition_variable frameCV;
std::atomic<bool> shouldExit(false);
std::atomic<bool> aiming(false);
std::atomic<bool> detectionPaused(false);

Config config;
Detector detector;
MouseThread* globalMouseThread = nullptr;

void mouseThreadFunction(MouseThread& mouseThread)
{
    int lastDetectionVersion = -1;

    while (!shouldExit)
    {
        std::vector<cv::Rect> boxes;
        std::vector<int> classes;

        {
            std::unique_lock<std::mutex> lock(detector.detectionMutex);
            detector.detectionCV.wait(lock, [&]() { return detector.detectionVersion > lastDetectionVersion || shouldExit; });
            if (shouldExit) break;

            lastDetectionVersion = detector.detectionVersion;

            boxes = detector.detectedBoxes;
            classes = detector.detectedClasses;
        }
        if (aiming)
        {
            Target* target = sortTargets(boxes, classes, config.detection_window_width, config.detection_window_height, false);
            if (target)
            {
                mouseThread.moveMouseToTarget(*target);
                delete target;
            }
        }
    }
}

int main()
{
    if (!config.loadConfig("config.ini"))
    {
        std::cerr << "Error with loading config.ini" << std::endl;
        return -1;
    }
    MouseThread mouseThread(
        config.detection_window_width,
        config.detection_window_height,
        config.dpi,
        config.sensitivity,
        config.fovX,
        config.fovY,
        config.minSpeedMultiplier,
        config.maxSpeedMultiplier,
        config.predictionInterval
    );

    globalMouseThread = &mouseThread;

    detector.initialize("models/sunxds_0.6.3.engine");

    std::thread keyThread(keyboardListener);
    std::thread capThread(captureThread, config.detection_window_width, config.detection_window_height);
    std::thread detThread(&Detector::inferenceThread, &detector);
    std::thread dispThread(displayThread);
    std::thread mouseMovThread(mouseThreadFunction, std::ref(mouseThread));

    keyThread.join();
    capThread.join();
    detThread.join();
    dispThread.join();
    mouseMovThread.join();

    return 0;
}