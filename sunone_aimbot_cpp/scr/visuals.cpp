#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

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
#include "capture.h"

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
    std::vector<std::string> cv_classes{ "player", "bot", "weapon", "outline",
                                         "dead_body", "hideout_target_human",
                                         "hideout_target_balls", "head", "smoke", "fire",
                                         "third_person" };

    namedWindow(config.window_name, WINDOW_NORMAL);
    if (config.always_on_top)
    {
        setWindowProperty(config.window_name, WND_PROP_TOPMOST, 1);
    }

    int currentSize = config.window_size != 100 ?
        static_cast<int>((config.detection_resolution * config.window_size) / 100)
        : config.detection_resolution;

    if (config.window_size != 100)
    {
        resizeWindow(config.window_name, currentSize, currentSize);
    }

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
                
                std::string className = cv_classes[classes[i]];
                
                int baseline = 0;
                cv::Size textSize = cv::getTextSize(className, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                cv::Point textOrg(adjustedBox.x, adjustedBox.y - 5);
                
                putText(frame, className, textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            }
        }

        if (config.show_fps)
        {
            putText(frame, "FPS: " + std::to_string(static_cast<int>(captureFps)), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 0), 2);
        }

        cv::Mat displayFrame;

        if (config.window_size != 100)
        {
            int desiredSize = static_cast<int>((config.detection_resolution * config.window_size) / 100);
            if (desiredSize != currentSize)
            {
                resizeWindow(config.window_name, desiredSize, desiredSize);
                currentSize = desiredSize;
            }

            cv::resize(frame, displayFrame, cv::Size(currentSize, currentSize));
        }
        else
        {
            displayFrame = frame;
        }

        try
        {
            imshow(config.window_name, displayFrame);
        }
        catch (cv::Exception&)
        {
            break;
        }

        if (waitKey(1) == 27) shouldExit = true;
    }
}