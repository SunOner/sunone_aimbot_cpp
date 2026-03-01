#pragma once
#include <vector>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>

struct DetectionBuffer
{
    std::mutex mutex;
    std::condition_variable cv;
    int version = 0;
    std::vector<cv::Rect> boxes;
    std::vector<int> classes;

    void set(const std::vector<cv::Rect>& newBoxes, const std::vector<int>& newClasses)
    {
        std::lock_guard<std::mutex> lock(mutex);
        boxes = newBoxes;
        classes = newClasses;
        ++version;
        cv.notify_all();
    }

    void get(std::vector<cv::Rect>& outBoxes, std::vector<int>& outClasses, int& outVersion)
    {
        std::lock_guard<std::mutex> lock(mutex);
        outBoxes = boxes;
        outClasses = classes;
        outVersion = version;
    }
};