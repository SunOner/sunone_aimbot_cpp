#ifndef VISUALS_H
#define VISUALS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "detector.h"

void displayThread();
void setWindowAlwaysOnTop(const std::string& winName, bool onTop);

#endif // VISUALS_H