#include "visuals.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>

#include "config.h"

using namespace cv;
using namespace std;

extern Detector detector;
extern Mat latestFrame;
extern std::mutex frameMutex;
extern std::condition_variable frameCV;
extern std::atomic<bool> shouldExit;
extern Config config;

void displayThread()
{
    std::vector<cv::Rect> boxes;
    std::vector<int> classes;

    int frameCount = 0;
    double fps = 0.0;
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;

    if (config.show_fps)
    {
        frameCount = 0;
        fps = 0.0;
        startTime = std::chrono::high_resolution_clock::now();
    }

    if (!config.show_window) { return; }

    while (!shouldExit)
    {
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lock(frameMutex);
            frameCV.wait(lock, [] { return !latestFrame.empty() || shouldExit; });
            if (shouldExit) break;
            frame = latestFrame.clone();
        }

        if (detector.getLatestDetections(boxes, classes))
        {
            for (size_t i = 0; i < boxes.size(); ++i)
            {
                cv::Rect adjustedBox = boxes[i];
                adjustedBox.x = std::max(0, adjustedBox.x);
                adjustedBox.y = std::max(0, adjustedBox.y);
                adjustedBox.width = std::min(frame.cols - adjustedBox.x, adjustedBox.width);
                adjustedBox.height = std::min(frame.rows - adjustedBox.y, adjustedBox.height);

                rectangle(frame, adjustedBox, cv::Scalar(0, 255, 0), 2);
                putText(frame, std::to_string(classes[i]), adjustedBox.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            }
        }

        if (config.show_fps)
        {
            frameCount++;
            auto currentTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = currentTime - startTime;

            if (elapsed.count() >= 1.0)
            {
                fps = static_cast<double>(frameCount) / elapsed.count();
                frameCount = 0;
                startTime = currentTime;
            }

            putText(frame, "FPS: " + std::to_string(static_cast<int>(fps)), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 0), 2);
        }

        imshow("Desktop", frame);
        if (waitKey(1) == 27) shouldExit = true;
    }
}