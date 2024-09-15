#ifndef TARGET_H
#define TARGET_H

#include <opencv2/opencv.hpp>
#include <vector>

class Target
{
public:
    int x, y, w, h, cls;

    Target(int x, int y, int w, int h, int cls);
};

Target* sortTargets(const std::vector<cv::Rect>& boxes, const std::vector<int>& classes, int screenWidth, int screenHeight, bool disableHeadshot);

#endif // TARGET_H