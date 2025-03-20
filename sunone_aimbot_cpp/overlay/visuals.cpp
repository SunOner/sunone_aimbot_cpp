#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
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
#include "optical_flow.h"

extern std::atomic<bool> show_window_changed;

void displayThread()
{
    std::vector<cv::Rect> boxes;
    std::vector<int> classes;

    if (config.show_window)
    {
        cv::namedWindow(config.window_name, cv::WINDOW_NORMAL);
        if (config.always_on_top)
            cv::setWindowProperty(config.window_name, cv::WND_PROP_TOPMOST, 1);
        else
            cv::setWindowProperty(config.window_name, cv::WND_PROP_TOPMOST, 0);
    }

    int currentSize = 0;

    while (!shouldExit)
    {
        if (show_window_changed.load())
        {
            if (config.show_window)
            {
                cv::namedWindow(config.window_name, cv::WINDOW_NORMAL);
                if (config.always_on_top)
                    cv::setWindowProperty(config.window_name, cv::WND_PROP_TOPMOST, 1);
                else
                    cv::setWindowProperty(config.window_name, cv::WND_PROP_TOPMOST, 0);
            }
            else
            {
                cv::destroyWindow(config.window_name);
            }
            show_window_changed.store(false);
        }

        if (config.show_window)
        {
            cv::Mat frame;
            {
                std::unique_lock<std::mutex> lock(frameMutex);
                frameCV.wait(lock, [] { return !latestFrameCpu.empty() || shouldExit; });
                if (shouldExit) break;
                frame = latestFrameCpu.clone();
            }

            cv::Mat displayFrame;
            if (config.window_size != 100)
            {
                int desiredSize = static_cast<int>((640 * config.window_size) / 100);
                if (desiredSize != currentSize)
                {
                    cv::resizeWindow(config.window_name, desiredSize, desiredSize);
                    currentSize = desiredSize;
                }
                cv::resize(frame, displayFrame, cv::Size(desiredSize, desiredSize));
            }
            else
            {
                displayFrame = frame.clone();
            }

            if (detector.getLatestDetections(boxes, classes))
            {
                float box_scale_x = static_cast<float>(frame.cols) / config.detection_resolution;
                float box_scale_y = static_cast<float>(frame.rows) / config.detection_resolution;

                float resize_scale_x = static_cast<float>(displayFrame.cols) / frame.cols;
                float resize_scale_y = static_cast<float>(displayFrame.rows) / frame.rows;

                for (size_t i = 0; i < boxes.size(); ++i)
                {
                    cv::Rect box = boxes[i];
                    box.x = static_cast<int>(box.x * box_scale_x);
                    box.y = static_cast<int>(box.y * box_scale_y);
                    box.width = static_cast<int>(box.width * box_scale_x);
                    box.height = static_cast<int>(box.height * box_scale_y);

                    box.x = static_cast<int>(box.x * resize_scale_x);
                    box.y = static_cast<int>(box.y * resize_scale_y);
                    box.width = static_cast<int>(box.width * resize_scale_x);
                    box.height = static_cast<int>(box.height * resize_scale_y);

                    cv::rectangle(displayFrame, box, cv::Scalar(0, 255, 0), 2);
                    cv::putText(displayFrame, std::to_string(classes[i]), cv::Point(box.x, box.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                }
            }

            if (config.enable_optical_flow && config.draw_optical_flow)
            {
                opticalFlow.drawOpticalFlow(displayFrame);
            }

            if (globalMouseThread)
            {
                auto futurePts = globalMouseThread->getFuturePositions();
                if (!futurePts.empty())
                {
                    float scale_x = static_cast<float>(displayFrame.cols) / config.detection_resolution;
                    float scale_y = static_cast<float>(displayFrame.rows) / config.detection_resolution;

                    cv::Point prevPt(-1, -1);
                    for (size_t i = 0; i < futurePts.size(); i++)
                    {
                        int px = static_cast<int>(futurePts[i].first * scale_x);
                        int py = static_cast<int>(futurePts[i].second * scale_y);
                        cv::Point pt(px, py);

                        cv::circle(displayFrame, pt, 5, cv::Scalar(0, 0, 255), -1);
                        cv::putText(displayFrame, std::to_string(i + 1), cv::Point(px + 5, py + 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);

                        if (prevPt.x != -1)
                        {
                            cv::line(displayFrame, prevPt, pt, cv::Scalar(0, 0, 255), 2);
                        }
                        prevPt = pt;
                    }
                }
            }

            if (config.show_fps)
            {
                cv::putText(displayFrame, "FPS: " + std::to_string(static_cast<int>(captureFps)),
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                    1.0, cv::Scalar(255, 255, 0), 2);
            }

            try
            {
                cv::imshow(config.window_name, displayFrame);
            }
            catch (cv::Exception& e)
            {
                std::cerr << "[Visuals]: " << e.what() << std::endl;
                break;
            }

            if (cv::waitKey(1) == 27)
                shouldExit = true;
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }
}