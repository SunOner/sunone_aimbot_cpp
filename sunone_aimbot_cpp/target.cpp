#include "target.h"
#include <cmath>
#include <limits>
#include <opencv2/opencv.hpp>
#include "sunone_aimbot_cpp.h"

using namespace std;

int screen_x_center = detection_window_width / 2;
int screen_y_center = detection_window_height / 2;
bool disable_headshot = false;
float body_y_offset = 0.33f;

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

    for (size_t i = 0; i < boxes.size(); ++i)
    {
        if (classes[i] == 2 || classes[i] == 3 || classes[i] == 4 || classes[i] == 8 || classes[i] == 9)
        {
            continue;
        }

        int adjustedY = classes[i] == 7 ? boxes[i].y : static_cast<int>(boxes[i].y - body_y_offset * boxes[i].height);

        float scale_x = static_cast<float>(screenWidth) / detection_window_width;
        float scale_y = static_cast<float>(screenHeight) / detection_window_height;

        cv::Point boxCenter(
            (boxes[i].x + boxes[i].width / 2) * scale_x,
            (adjustedY + boxes[i].height / 2) * scale_y
        );

        double distance = std::pow(boxCenter.x - center.x, 2) + std::pow(boxCenter.y - center.y, 2);

        if (disableHeadshot && classes[i] == 7)
        {
            continue;
        }

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

    int adjustedY = classes[nearestIdx] == 7 ? boxes[nearestIdx].y : static_cast<int>(boxes[nearestIdx].y - body_y_offset * boxes[nearestIdx].height);

    return new Target(boxes[nearestIdx].x, adjustedY, boxes[nearestIdx].width, boxes[nearestIdx].height, classes[nearestIdx]);
}