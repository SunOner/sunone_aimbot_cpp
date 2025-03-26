#ifndef CAPTURE_H
#define CAPTURE_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>
#include <atomic>
#include <chrono>
#include <mutex>
#include <condition_variable>

extern std::atomic<bool> detection_resolution_changed;
extern std::atomic<bool> capture_method_changed;
extern std::atomic<bool> capture_cursor_changed;
extern std::atomic<bool> capture_borders_changed;
extern std::atomic<bool> capture_fps_changed;

void captureThread(int CAPTURE_WIDTH, int CAPTURE_HEIGHT);
extern int screenWidth;
extern int screenHeight;

extern std::atomic<int> captureFrameCount;
extern std::atomic<int> captureFps;
extern std::chrono::time_point<std::chrono::high_resolution_clock> captureFpsStartTime;

extern cv::cuda::GpuMat latestFrameGpu;
extern cv::Mat          latestFrameCpu;

extern std::mutex frameMutex;
extern std::condition_variable frameCV;
extern std::atomic<bool> shouldExit;
extern std::atomic<bool> show_window_changed;

class IScreenCapture
{
public:
    virtual ~IScreenCapture() {}

    virtual cv::cuda::GpuMat GetNextFrameGpu() = 0;
    virtual cv::Mat GetNextFrameCpu() = 0;
};

#endif // CAPTURE_H