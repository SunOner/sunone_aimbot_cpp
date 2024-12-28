#include "postProcess.h"
#include <algorithm>
#include <numeric>

void NMS(std::vector<Detection>& detections, float nmsThreshold)
{
    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b)
        {
            return a.confidence > b.confidence;
        }
    );

    std::vector<bool> suppress(detections.size(), false);

    for (size_t i = 0; i < detections.size(); ++i)
    {
        if (suppress[i]) continue;
        for (size_t j = i + 1; j < detections.size(); ++j)
        {
            if (suppress[j])
            {
                continue;
            }

            float iou = (detections[i].box & detections[j].box).area() / (float)(detections[i].box | detections[j].box).area();
            if (iou > nmsThreshold)
            {
                suppress[j] = true;
            }
        }
    }

    std::vector<Detection> result;
    for (size_t i = 0; i < detections.size(); ++i)
    {
        if (!suppress[i])
        {
            result.push_back(detections[i]);
        }
    }

    detections = result;
}

std::vector<Detection> postProcessYolo8(const std::vector<float>& output, float ratio, int imgWidth, int imgHeight, int numClasses, float confThreshold, float nmsThreshold)
{
    int numChannels = numClasses + 4;
    int numAnchors = output.size() / numChannels;

    std::vector<Detection> detections;

    cv::Mat outputMat(numChannels, numAnchors, CV_32F, (void*)output.data());
    outputMat = outputMat.t();

    for (int i = 0; i < numAnchors; i++)
    {
        const float* rowPtr = outputMat.row(i).ptr<float>();
        const float* bboxesPtr = rowPtr;
        const float* scoresPtr = rowPtr + 4;

        auto maxSPtr = std::max_element(scoresPtr, scoresPtr + numClasses);
        float score = *maxSPtr;
        if (score > confThreshold)
        {
            float x = *bboxesPtr++;
            float y = *bboxesPtr++;
            float w = *bboxesPtr++;
            float h = *bboxesPtr;

            float x0 = std::clamp((x - 0.5f * w) * ratio, 0.f, (float)imgWidth);
            float y0 = std::clamp((y - 0.5f * h) * ratio, 0.f, (float)imgHeight);
            float x1 = std::clamp((x + 0.5f * w) * ratio, 0.f, (float)imgWidth);
            float y1 = std::clamp((y + 0.5f * h) * ratio, 0.f, (float)imgHeight);

            int label = static_cast<int>(maxSPtr - scoresPtr);

            Detection det;
            det.box = cv::Rect(cv::Point(x0, y0), cv::Point(x1, y1));
            det.confidence = score;
            det.classId = label;

            detections.push_back(det);
        }
    }

    NMS(detections, nmsThreshold);

    return detections;
}

std::vector<Detection> postProcessYolo9(const std::vector<float>& output, float ratio, int imgWidth, int imgHeight, int numClasses, float confThreshold, float nmsThreshold)
{
    return postProcessYolo8(output, ratio, imgWidth, imgHeight, numClasses, confThreshold, nmsThreshold);
}

std::vector<Detection> postProcessYolo10(const float* output, const std::vector<int64_t>& shape, float factor, int numClasses, float confThreshold, float nmsThreshold)
{
    std::vector<Detection> detections;

    int64_t numDetections = shape[1];

    for (int i = 0; i < numDetections; ++i)
    {
        const float* det = output + i * shape[2];
        float confidence = det[4];

        if (confidence > confThreshold)
        {
            int classId = static_cast<int>(det[5]);

            float cx = det[0];
            float cy = det[1];
            float dx = det[2];
            float dy = det[3];

            int x = static_cast<int>(cx * factor);
            int y = static_cast<int>(cy * factor);
            int width = static_cast<int>((dx - cx) * factor);
            int height = static_cast<int>((dy - cy) * factor);

            cv::Rect box(x, y, width, height);

            Detection det;
            det.box = box;
            det.confidence = confidence;
            det.classId = classId;

            detections.push_back(det);
        }
    }

    NMS(detections, nmsThreshold);

    return detections;
}

std::vector<Detection> postProcessYolo11(const float* output, const std::vector<int64_t>& shape, int numClasses, float confThreshold, float nmsThreshold, float imgScale)
{
    if (shape.size() != 3)
    {
        std::cerr << "[postProcess] Unsupported output shape" << std::endl;
        return std::vector<Detection>();
    }

    std::vector<Detection> detections;

    cv::Mat det_output(shape[1], shape[2], CV_32F, (void*)output);

    for (int i = 0; i < det_output.cols; ++i)
    {
        if (det_output.rows < 4 + numClasses)
        {
            std::cerr << "[postProcess] Number of classes exceeds available rows in det_output" << std::endl;
            continue;
        }

        cv::Mat classes_scores = det_output.col(i).rowRange(4, 4 + numClasses);

        cv::Point class_id_point;
        double score;

        cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        if (score > confThreshold)
        {
            float cx = det_output.at<float>(0, i);
            float cy = det_output.at<float>(1, i);
            float ow = det_output.at<float>(2, i);
            float oh = det_output.at<float>(3, i);

            cv::Rect box;
            box.x = static_cast<int>((cx - 0.5f * ow) * imgScale);
            box.y = static_cast<int>((cy - 0.5f * oh) * imgScale);
            box.width = static_cast<int>(ow * imgScale);
            box.height = static_cast<int>(oh * imgScale);

            Detection detection;
            detection.box = box;
            detection.confidence = static_cast<float>(score);
            detection.classId = class_id_point.y;

            detections.push_back(detection);
        }
    }

    NMS(detections, nmsThreshold);

    return detections;
}