#include <algorithm>
#include <numeric>
#include <chrono>

#include "postProcess.h"
#include "sunone_aimbot_cpp.h"
#ifdef USE_CUDA
#include "trt_detector.h"
#endif

void NMS(std::vector<Detection>& detections, float nmsThreshold, std::chrono::duration<double, std::milli>* nmsTime)
{
    if (detections.empty()) return;

    auto t0 = std::chrono::steady_clock::now();

    std::sort(
        detections.begin(),
        detections.end(),
        [](const Detection& a, const Detection& b)
        {
            return a.confidence > b.confidence;
        }
    );

    std::vector<bool> suppress(detections.size(), false);
    std::vector<Detection> result;
    result.reserve(detections.size());

    for (size_t i = 0; i < detections.size(); ++i)
    {
        if (suppress[i]) continue;

        result.push_back(detections[i]);

        const cv::Rect& box_i = detections[i].box;
        const float area_i = static_cast<float>(box_i.area());

        for (size_t j = i + 1; j < detections.size(); ++j)
        {
            if (suppress[j]) continue;

            const cv::Rect& box_j = detections[j].box;
            const cv::Rect intersection = box_i & box_j;

            if (intersection.width > 0 && intersection.height > 0)
            {
                const float intersection_area = static_cast<float>(intersection.area());
                const float union_area = area_i + static_cast<float>(box_j.area()) - intersection_area;

                if (intersection_area / union_area > nmsThreshold)
                {
                    suppress[j] = true;
                }
            }
        }
    }

    detections = std::move(result);

    auto t1 = std::chrono::steady_clock::now();
    if (nmsTime)
    {
        *nmsTime = t1 - t0;
    }
}

#ifdef USE_CUDA
std::vector<Detection> postProcessYolo10(
    const float* output,
    const std::vector<int64_t>& shape,
    int numClasses,
    float confThreshold,
    float nmsThreshold,
    std::chrono::duration<double, std::milli>* nmsTime
)
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

            int x = static_cast<int>(cx * trt_detector.img_scale);
            int y = static_cast<int>(cy * trt_detector.img_scale);
            int width = static_cast<int>((dx - cx) * trt_detector.img_scale);
            int height = static_cast<int>((dy - cy) * trt_detector.img_scale);

            cv::Rect box(x, y, width, height);

            Detection det;
            det.box = box;
            det.confidence = confidence;
            det.classId = classId;

            detections.push_back(det);
        }
    }

    NMS(detections, nmsThreshold, nmsTime);

    return detections;
}

std::vector<Detection> postProcessYolo11(
    const float* output,
    const std::vector<int64_t>& shape,
    int numClasses,
    float confThreshold,
    float nmsThreshold,
    std::chrono::duration<double, std::milli>* nmsTime
)
{
    if (shape.size() != 3)
    {
        std::cerr << "[postProcess] Unsupported output shape" << std::endl;
        return std::vector<Detection>();
    }

    std::vector<Detection> detections;
    detections.reserve(shape[2]);

    int64_t rows = shape[1];
    int64_t cols = shape[2];
    
    if (rows < 4 + numClasses)
    {
        std::cerr << "[postProcess] Number of classes exceeds available rows in det_output" << std::endl;
        return detections;
    }
    
    cv::Mat det_output(rows, cols, CV_32F, (void*)output);

    const float img_scale = trt_detector.img_scale;
    
    for (int i = 0; i < cols; ++i)
    {
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

            const float half_ow = 0.5f * ow;
            const float half_oh = 0.5f * oh;
            
            cv::Rect box;
            box.x = static_cast<int>((cx - half_ow) * img_scale);
            box.y = static_cast<int>((cy - half_oh) * img_scale);
            box.width = static_cast<int>(ow * img_scale);
            box.height = static_cast<int>(oh * img_scale);

            Detection detection;
            detection.box = box;
            detection.confidence = static_cast<float>(score);
            detection.classId = class_id_point.y;

            detections.push_back(detection);
        }
    }

    if (!detections.empty())
    {
        NMS(detections, nmsThreshold, nmsTime);
    }

    return detections;
}
#endif

std::vector<Detection> postProcessYolo10DML(
    const float* output,
    const std::vector<int64_t>& shape,
    int numClasses,
    float confThreshold,
    float nmsThreshold
) {
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

            int x = static_cast<int>(cx);
            int y = static_cast<int>(cy);
            int width = static_cast<int>(dx - cx);
            int height = static_cast<int>(dy - cy);

            cv::Rect box(x, y, width, height);
            detections.push_back(Detection{ box, confidence, classId });
        }
    }
    NMS(detections, nmsThreshold);
    return detections;
}

std::vector<Detection> postProcessYolo11DML(
    const float* output,
    const std::vector<int64_t>& shape,
    int numClasses,
    float confThreshold,
    float nmsThreshold
) {
    std::vector<Detection> detections;
    if (shape.size() != 2) return detections;

    int64_t rows = shape[0];
    int64_t cols = shape[1];

    cv::Mat det_output(rows, cols, CV_32F, (void*)output);
    for (int i = 0; i < cols; ++i) {
        cv::Mat classes_scores = det_output.col(i).rowRange(4, 4 + numClasses);
        cv::Point class_id_point;
        double score;
        cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);
        if (score > confThreshold) {
            float cx = det_output.at<float>(0, i);
            float cy = det_output.at<float>(1, i);
            float ow = det_output.at<float>(2, i);
            float oh = det_output.at<float>(3, i);
            const float half_ow = 0.5f * ow;
            const float half_oh = 0.5f * oh;
            cv::Rect box;
            box.x = static_cast<int>(cx - half_ow);
            box.y = static_cast<int>(cy - half_oh);
            box.width = static_cast<int>(ow);
            box.height = static_cast<int>(oh);
            detections.push_back(Detection{ box, static_cast<float>(score), class_id_point.y });
        }
    }
    if (!detections.empty()) NMS(detections, nmsThreshold);
    return detections;
}