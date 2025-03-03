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

AimbotTarget::AimbotTarget(int x, int y, int w, int h, int cls) : x(x), y(y), w(w), h(h), classId(cls) {}

AimbotTarget* sortTargets(const std::vector<cv::Rect>& boxes, const std::vector<int>& classes, int screenWidth, int screenHeight, bool disableHeadshot)
{
    if (boxes.empty() || classes.empty())
    {
        return nullptr;
    }

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

                if (distance < minDistance)
                {
                    minDistance = distance;
                    nearestIdx = i;
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
            {
                continue;
            }

            if (classes[i] == config.class_player ||
                classes[i] == config.class_bot ||
                classes[i] == config.class_hideout_target_human && config.shooting_range_targets ||
                classes[i] == config.class_hideout_target_balls && config.shooting_range_targets ||
                (classes[i] == config.class_third_person && !config.ignore_third_person))
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

    int y;
    if (classes[nearestIdx] == config.class_head)
    {
        int headOffsetY = static_cast<int>(boxes[nearestIdx].height * config.head_y_offset);
        y = boxes[nearestIdx].y + headOffsetY - boxes[nearestIdx].height / 2;
    }
    else
    {
        y = targetY - boxes[nearestIdx].height / 2;
    }

    return new AimbotTarget(
        boxes[nearestIdx].x,
        y,
        boxes[nearestIdx].width,
        boxes[nearestIdx].height,
        classes[nearestIdx]
    );
}