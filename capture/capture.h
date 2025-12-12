#ifndef CAPTURE_H
#define CAPTURE_H

#include <opencv2/opencv.hpp>
#include <atomic>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <deque>

extern std::atomic<bool> detection_resolution_changed;
extern std::atomic<bool> capture_method_changed;
extern std::atomic<bool> capture_cursor_changed;
extern std::atomic<bool> capture_borders_changed;
extern std::atomic<bool> capture_fps_changed;
extern std::deque<cv::Mat> frameQueue;

void captureThread(int CAPTURE_WIDTH, int CAPTURE_HEIGHT);
extern int screenWidth;
extern int screenHeight;

extern std::atomic<int> captureFrameCount;
extern std::atomic<int> captureFps;
extern std::chrono::time_point<std::chrono::high_resolution_clock> captureFpsStartTime;

extern cv::Mat latestFrame;

extern std::mutex frameMutex;
extern std::condition_variable frameCV;
extern std::atomic<bool> shouldExit;
extern std::atomic<bool> show_window_changed;

class IScreenCapture
{
public:
    virtual ~IScreenCapture() {}
    virtual cv::Mat GetNextFrameCpu() = 0;
};

#endif // CAPTURE_H