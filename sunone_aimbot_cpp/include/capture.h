#ifndef CAPTURE_H
#define CAPTURE_H

#include <opencv2/opencv.hpp>
#include <atomic>
#include <chrono>

void captureThread(int CAPTURE_WIDTH, int CAPTURE_HEIGHT);

extern int screenWidth;
extern int screenHeight;

extern std::atomic<int> captureFrameCount;
extern std::atomic<double> captureFps;
extern std::chrono::time_point<std::chrono::high_resolution_clock> captureFpsStartTime;

class IScreenCapture {
public:
    virtual ~IScreenCapture() {}
    virtual cv::Mat GetNextFrame() = 0;
};

#endif // CAPTURE_H