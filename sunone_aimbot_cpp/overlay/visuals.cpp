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

            if (config.capture_use_cuda)
            {
                {
                    std::unique_lock<std::mutex> lock(frameMutex);
                    while (latestFrameGpu.empty() && !shouldExit)
                    {
                        lock.unlock();
                        std::this_thread::sleep_for(std::chrono::milliseconds(2));
                        lock.lock();
                    }
                    if (shouldExit) break;

                    latestFrameGpu.download(frame);
                }
            }
            else
            {
                std::unique_lock<std::mutex> lock(frameMutex);
                frameCV.wait(lock, [] { return !latestFrameCpu.empty() || shouldExit; });
                if (shouldExit) break;

                frame = latestFrameCpu.clone();
            }

            if (frame.empty())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
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
                float box_scale_x = static_cast<float>(displayFrame.cols) / config.detection_resolution;
                float box_scale_y = static_cast<float>(displayFrame.rows) / config.detection_resolution;

                for (size_t i = 0; i < boxes.size(); ++i)
                {
                    cv::Rect box = boxes[i];
                    box.x = static_cast<int>(box.x * box_scale_x);
                    box.y = static_cast<int>(box.y * box_scale_y);
                    box.width = static_cast<int>(box.width * box_scale_x);
                    box.height = static_cast<int>(box.height * box_scale_y);

                    cv::rectangle(displayFrame, box, cv::Scalar(0, 255, 0), 2);
                    cv::putText(displayFrame,
                        std::to_string(classes[i]),
                        cv::Point(box.x, box.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.5,
                        cv::Scalar(0, 255, 0),
                        1);
                }
            }

            if (config.enable_optical_flow && config.draw_optical_flow)
            {
                opticalFlow.drawOpticalFlow(displayFrame);
            }

            if (globalMouseThread)
            {
                if (config.draw_futurePositions)
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

                            int totalPts = static_cast<int>(futurePts.size());
                            for (size_t i = 0; i < futurePts.size(); i++)
                            {
                                int px = static_cast<int>(futurePts[i].first * scale_x);
                                int py = static_cast<int>(futurePts[i].second * scale_y);
                                cv::Point pt(px, py);

                                int b = static_cast<int>(255 - (i * 255.0 / totalPts));
                                int r = static_cast<int>(i * 255.0 / totalPts);
                                int g = 50;

                                cv::circle(displayFrame, pt, 4, cv::Scalar(b, g, r), cv::FILLED);

                                cv::circle(displayFrame, pt, 4, cv::Scalar(255, 255, 255), 1);
                            }
                        }
                    }
                }
            }

            if (config.show_fps)
            {
                cv::putText(displayFrame,
                    "FPS: " + std::to_string(static_cast<int>(captureFps)),
                    cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX,
                    1.0,
                    cv::Scalar(255, 255, 0),
                    2);
            }

            cv::imshow(config.window_name, displayFrame);
            
            if (cv::waitKey(1) == 27)
                shouldExit = true;
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }
}