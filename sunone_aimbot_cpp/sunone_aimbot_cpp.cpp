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
#include "target.h" 
#include "sunone_aimbot_cpp.h"
#include "detector.h"
#include <Windows.h>

#pragma comment(lib, "nvinfer_10.lib")
#pragma comment(lib, "nvonnxparser_10.lib")
#pragma comment(lib, "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/lib/x64/cudart.lib")

using namespace cv;
using namespace std;

Mat latestFrame;
std::condition_variable frameCV;
std::atomic<bool> shouldExit(false);

Detector detector;

int detection_window_width = 480;
int detection_window_height = 320;

double dpi = 1000;
double sensitivity = 100.0;
double fovX = 50.0;
double fovY = 50.0;
double minSpeedMultiplier = 0.5;
double maxSpeedMultiplier = 1.5;
double predictionInterval = 0.3;

MouseThread mouseThread(detection_window_width, detection_window_height, dpi, sensitivity, fovX, fovY, minSpeedMultiplier, maxSpeedMultiplier, predictionInterval);

void mouseThreadFunction()
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

        if (GetAsyncKeyState(VK_RBUTTON) & 0x8000)
        {
            Target* target = sortTargets(boxes, classes, detection_window_width, detection_window_height, false);
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
    detector.initialize("models/sunxds_0.6.3.engine");

    std::thread capThread(captureThread, detection_window_width, detection_window_height);
    std::thread detThread(&Detector::inferenceThread, &detector);
    std::thread dispThread(displayThread);
    std::thread mouseMovThread(mouseThreadFunction);

    capThread.join();
    detThread.join();
    dispThread.join();
    mouseMovThread.join();

    return 0;
}