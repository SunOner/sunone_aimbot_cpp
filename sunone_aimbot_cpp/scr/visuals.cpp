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

extern std::atomic<bool> show_window_changed;

void displayThread()
{
    std::vector<cv::Rect> boxes;
    std::vector<int> classes;
    std::vector<int> cv_classes{
        config.class_player,
        config.class_bot,
        config.class_weapon,
        config.class_outline,
        config.class_dead_body,
        config.class_hideout_target_human,
        config.class_hideout_target_balls,
        config.class_head,
        config.class_smoke,
        config.class_fire,
        config.class_third_person
    };

    if (config.show_window)
    {
        namedWindow(config.window_name, WINDOW_NORMAL);

        if (config.always_on_top)
        {
            setWindowProperty(config.window_name, WND_PROP_TOPMOST, 1);
        }
        else
        {
            setWindowProperty(config.window_name, WND_PROP_TOPMOST, 0);
        }
    }

    while (!shouldExit)
    {
        int currentSize = static_cast<int>((config.detection_resolution * config.window_size) / 100);
        if (show_window_changed.load())
        {
            if (config.show_window)
            {
                namedWindow(config.window_name, WINDOW_NORMAL);
                if (config.always_on_top)
                {
                    setWindowProperty(config.window_name, WND_PROP_TOPMOST, 1);
                }
                else
                {
                    setWindowProperty(config.window_name, WND_PROP_TOPMOST, 0);
                }

                resizeWindow(config.window_name, currentSize, currentSize);
            }
            else
            {
                destroyWindow(config.window_name);
            }
            show_window_changed.store(false);
        }

        if (config.show_window)
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
                    cv::Rect box = boxes[i];

                    box.x = static_cast<int>(box.x / scale);
                    box.y = static_cast<int>(box.y / scale);
                    box.width = static_cast<int>(box.width / scale);
                    box.height = static_cast<int>(box.height / scale);

                    box.x = std::max(0, box.x);
                    box.y = std::max(0, box.y);
                    box.width = std::min(frame.cols - box.x, box.width);
                    box.height = std::min(frame.rows - box.y, box.height);

                    rectangle(frame, box, cv::Scalar(0, 255, 0), 2);

                    std::string className = std::to_string(cv_classes[classes[i]]);

                    int baseline = 0;
                    cv::Size textSize = cv::getTextSize(className, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                    cv::Point textOrg(box.x, box.y - 5);

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
    }