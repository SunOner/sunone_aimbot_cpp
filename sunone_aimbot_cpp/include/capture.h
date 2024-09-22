#ifndef CAPTURE_H
#define CAPTURE_H

#include <opencv2/opencv.hpp>

void captureThread(int CAPTURE_WIDTH, int CAPTURE_HEIGHT);

extern int screenWidth;
extern int screenHeight;

#endif // CAPTURE_H