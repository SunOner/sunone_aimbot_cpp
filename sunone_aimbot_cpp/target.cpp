#include <cmath>
#include <limits>
#include <opencv2/opencv.hpp>

#include "sunone_aimbot_cpp.h"
#include "target.h"
#include "config.h"

using namespace std;

extern Config config;

float body_y_offset = 0.95f;

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

    for (size_t i = 0; i < boxes.size(); ++i)
    {
        if (classes[i] == 2 || classes[i] == 3 || classes[i] == 4 || classes[i] == 8 || classes[i] == 9)
        {
            continue;
        }

        cv::Point targetPoint;

        if (classes[i] == 7)
        {
            headFound = true;
            targetPoint = cv::Point(boxes[i].x + boxes[i].width / 2, boxes[i].y + boxes[i].height / 2);
        }
        else if (!headFound)
        {
            targetPoint = cv::Point(boxes[i].x + boxes[i].width / 2, boxes[i].y + static_cast<int>(boxes[i].height * config.body_y_offset));
        }
        else
        {
            continue;
        }

        double distance = std::pow(targetPoint.x - center.x, 2) + std::pow(targetPoint.y - center.y, 2);

        if (distance < minDistance)
        {
            minDistance = distance;
            nearestIdx = i;
        }
    }

    if (nearestIdx == -1)
    {
        return nullptr;
    }

    return new Target(boxes[nearestIdx].x, boxes[nearestIdx].y, boxes[nearestIdx].width, boxes[nearestIdx].height, classes[nearestIdx]);
}