#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <cmath>
#include <limits>
#include <opencv2/opencv.hpp>

#include "sunone_aimbot_cpp.h"
#include "AimbotTarget.h"
#include "config.h"

AimbotTarget::AimbotTarget(int x_, int y_, int w_, int h_, int cls, double px, double py)
    : x(x_), y(y_), w(w_), h(h_), classId(cls), pivotX(px), pivotY(py)
{
}

AimbotTarget* sortTargets(
    const std::vector<cv::Rect>& boxes,
    const std::vector<int>& classes,
    int screenWidth,
    int screenHeight,
    bool disableHeadshot)
{
    if (boxes.empty() || classes.empty())
    {
        return nullptr;
    }

    const float maxDistance = 120 * 120; // 120 pixels squared

    cv::Point center(screenWidth / 2, screenHeight / 2);

    double minDistance = std::numeric_limits<double>::max();
    int nearestIdx = -1;
    int targetY = 0;

    if (!disableHeadshot)
    {
        for (size_t i = 0; i < boxes.size(); i++)
        {
            if (classes[i] == config.class_head)
            {
                int headOffsetY = static_cast<int>(boxes[i].height * config.head_y_offset);
                cv::Point targetPoint(boxes[i].x + boxes[i].width / 2, boxes[i].y + headOffsetY);
                double distance = std::pow(targetPoint.x - center.x, 2) + std::pow(targetPoint.y - center.y, 2);
                if (distance <= maxDistance && distance < minDistance)
                {
                    minDistance = distance;
                    nearestIdx = static_cast<int>(i);
                    targetY = targetPoint.y;
                }
            }
        }
    }

    if (disableHeadshot || nearestIdx == -1)
    {
        minDistance = std::numeric_limits<double>::max();
        for (size_t i = 0; i < boxes.size(); i++)
        {
            if (disableHeadshot && classes[i] == config.class_head)
                continue;

            if (classes[i] == config.class_player ||
                classes[i] == config.class_bot ||
                (classes[i] == config.class_hideout_target_human && config.shooting_range_targets) ||
                (classes[i] == config.class_hideout_target_balls && config.shooting_range_targets) ||
                (classes[i] == config.class_third_person && !config.ignore_third_person))
            {
                int offsetY = static_cast<int>(boxes[i].height * config.body_y_offset);
                cv::Point targetPoint(boxes[i].x + boxes[i].width / 2, boxes[i].y + offsetY);
                double distance = std::pow(targetPoint.x - center.x, 2) + std::pow(targetPoint.y - center.y, 2);
                if (distance <= maxDistance && distance < minDistance)
                {
                    minDistance = distance;
                    nearestIdx = static_cast<int>(i);
                    targetY = targetPoint.y;
                }
            }
        }
    }

    if (nearestIdx == -1)
    {
        return nullptr;
    }
   
    int finalX = boxes[nearestIdx].x;
    int finalW = boxes[nearestIdx].width;
    int finalH = boxes[nearestIdx].height;
    int finalClass = classes[nearestIdx];
    // Calcular el centro exacto del área objetivo (head/body)
    double pivotY;
    if (finalClass == config.class_head) {
        int headOffsetY = static_cast<int>(finalH * config.head_y_offset);
        pivotY = boxes[nearestIdx].y + headOffsetY;
    } else {
        int offsetY = static_cast<int>(finalH * config.body_y_offset);
        pivotY = boxes[nearestIdx].y + offsetY;
    }
    

    double pivotX = finalX + (finalW / 2.0);
    

    return new AimbotTarget(finalX, boxes[nearestIdx].y, finalW, finalH, finalClass, pivotX, pivotY);
}
