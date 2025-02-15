#ifndef AIMBOTTARGET_H
#define AIMBOTTARGET_H

#include <opencv2/opencv.hpp>
#include <vector>

class AimbotTarget
{
public:
    int x, y, w, h;
    int classId;

    AimbotTarget(int x, int y, int w, int h, int classId);
};

AimbotTarget* sortTargets(const std::vector<cv::Rect>& boxes,
    const std::vector<int>& classes,
    int screenWidth,
    int screenHeight,
    bool disableHeadshot);

#endif // AIMBOTTARGET_H