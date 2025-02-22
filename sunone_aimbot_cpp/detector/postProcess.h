#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <vector>
#include <opencv2/opencv.hpp>

#include "detector.h"

struct Detection
{
    cv::Rect box;
    float confidence;
    int classId;
};

void NMS(std::vector<Detection>& detections, float nmsThreshold);

std::vector<Detection> postProcessYolo10(
    const float* output,
    const std::vector<int64_t>& shape,
    int numClasses,
    float confThreshold,
    float nmsThreshold);

std::vector<Detection> postProcessYolo11(
    const float* output,
    const std::vector<int64_t>& shape,
    int numClasses,
    float confThreshold,
    float nmsThreshold);

#endif // POSTPROCESS_H