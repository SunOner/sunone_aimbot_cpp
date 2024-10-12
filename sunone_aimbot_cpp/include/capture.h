#ifndef CAPTURE_H
#define CAPTURE_H

#include <opencv2/opencv.hpp>

void captureThread(int CAPTURE_WIDTH, int CAPTURE_HEIGHT);

extern int screenWidth;
extern int screenHeight;

void CloseCapture();

extern std::atomic<int> captureFrameCount;
extern std::atomic<double> captureFps;
extern std::chrono::time_point<std::chrono::high_resolution_clock> captureFpsStartTime;

#endif // CAPTURE_H