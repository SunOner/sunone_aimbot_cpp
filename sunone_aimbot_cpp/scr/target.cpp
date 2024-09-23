#include <cmath>
#include <limits>
#include <opencv2/opencv.hpp>

#include "sunone_aimbot_cpp.h"
#include "target.h"

using namespace std;

Target::Target(int x, int y, int w, int h, int cls) : x(x), y(y), w(w), h(h), cls(cls) {}

Target* sortTargets(const std::vector<cv::Rect>& boxes, const std::vector<int>& classes, int screenWidth, int screenHeight, bool disableHeadshot)
{
    if (boxes.empty() || classes.empty())
    {
        return nullptr;
    }

    cv::Point center(screenWidth / 2, screenHeight / 2);

    double minDistance = std::numeric_limits<double>::max();
    int nearestIdx = -1;
    bool headFound = false;
    int targetY = 0;

    for (size_t i = 0; i < boxes.size(); ++i)
    {
        if (classes[i] == 7)
        {
            cv::Point targetPoint(boxes[i].x + boxes[i].width / 2, boxes[i].y + boxes[i].height / 2);
            double distance = std::pow(targetPoint.x - center.x, 2) + std::pow(targetPoint.y - center.y, 2);

            if (distance < minDistance)
            {
                minDistance = distance;
                nearestIdx = i;
                headFound = true;
                targetY = targetPoint.y;
            }
        }
    }

    if (!headFound)
    {
        minDistance = std::numeric_limits<double>::max();
        for (size_t i = 0; i < boxes.size(); ++i)
        {
            if (classes[i] == 0 || classes[i] == 1 || classes[i] == 5 || classes[i] == 6)
            {
                int offsetY = static_cast<int>(boxes[i].height * config.body_y_offset);
                cv::Point targetPoint(boxes[i].x + boxes[i].width / 2, boxes[i].y + offsetY);
                double distance = std::pow(targetPoint.x - center.x, 2) + std::pow(targetPoint.y - center.y, 2);

                if (distance < minDistance)
                {
                    minDistance = distance;
                    nearestIdx = i;
                    targetY = targetPoint.y;
                }
            }
        }
    }

    if (nearestIdx == -1)
    {
        return nullptr;
    }

    int y = (classes[nearestIdx] == 7) ? boxes[nearestIdx].y : targetY - boxes[nearestIdx].height / 2;

    return new Target(
        boxes[nearestIdx].x,
        y,
        boxes[nearestIdx].width,
        boxes[nearestIdx].height,
        classes[nearestIdx]
    );
}