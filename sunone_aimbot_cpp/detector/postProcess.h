#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <vector>
#include <opencv2/opencv.hpp>

struct Detection
{
    cv::Rect box;
    float confidence;
    int classId;
};

void NMS(std::vector<Detection>& detections, float nmsThreshold);

std::vector<Detection> postProcessYolo8(const std::vector<float>& output, float ratio, int imgWidth, int imgHeight, int numClasses, float confThreshold, float nmsThreshold);
std::vector<Detection> postProcessYolo9(const std::vector<float>& output, float ratio, int imgWidth, int imgHeight, int numClasses, float confThreshold, float nmsThreshold);
std::vector<Detection> postProcessYolo10(const float* output, const std::vector<int64_t>& shape, float factor, int numClasses, float confThreshold, float nmsThreshold);
std::vector<Detection> postProcessYolo11(const float* output, const std::vector<int64_t>& shape, int numClasses, float confThreshold, float nmsThreshold, float imgScale);

#endif // POSTPROCESS_H