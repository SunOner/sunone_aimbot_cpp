#include <opencv2/opencv.hpp>
#include <iostream>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>

#include "visuals.h"
#include "config.h"
#include "sunone_aimbot_cpp.h"

using namespace cv;
using namespace std;

extern Mat latestFrame;
extern std::mutex frameMutex;
extern std::condition_variable frameCV;
extern std::atomic<bool> shouldExit;

void displayThread()
{
    if (!config.show_window) { return; }

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

    
    namedWindow(config.window_name);
    
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
            float scale = static_cast<float>(config.detection_resolution) / config.engine_image_size;

            for (size_t i = 0; i < boxes.size(); ++i)
            {
                cv::Rect adjustedBox = boxes[i];

                adjustedBox.x = static_cast<int>(adjustedBox.x / scale);
                adjustedBox.y = static_cast<int>(adjustedBox.y / scale);
                adjustedBox.width = static_cast<int>(adjustedBox.width / scale);
                adjustedBox.height = static_cast<int>(adjustedBox.height / scale);

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

        if (config.window_size != 100)
        {
            int size = static_cast<int>((config.detection_resolution * config.window_size) / 100);

            Mat resized;
            cv::resize(frame, resized, cv::Size(size, size));

            resizeWindow(config.window_name, size, size);
            imshow(config.window_name, resized);
        }
        else
        {
            imshow(config.window_name, frame);
        }

        if (waitKey(1) == 27) shouldExit = true;
    }
}