#include <algorithm>
#include <numeric>

#include "postProcess.h"
#include "sunone_aimbot_cpp.h"
#include "detector.h"

void NMS(std::vector<Detection>& detections, float nmsThreshold)
{
    if (detections.empty()) return;
    
    // Sort detections by confidence (highest first)
    std::sort(detections.begin(), detections.end(), [](const Detection& a, const Detection& b)
        {
            return a.confidence > b.confidence;
        }
    );

    std::vector<bool> suppress(detections.size(), false);
    std::vector<Detection> result;
    result.reserve(detections.size());  // Pre-allocate memory

    for (size_t i = 0; i < detections.size(); ++i)
    {
        if (suppress[i]) continue;
        
        // Keep this detection
        result.push_back(detections[i]);
        
        // Efficiently suppress overlapping detections
        const cv::Rect& box_i = detections[i].box;
        const float area_i = static_cast<float>(box_i.area());
        
        for (size_t j = i + 1; j < detections.size(); ++j)
        {
            if (suppress[j]) break;
            
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
    
    detections = std::move(result);  // Use move semantics to avoid copy
}

std::vector<Detection> postProcessYolo10(
    const float* output,
    const std::vector<int64_t>& shape,
    int numClasses,
    float confThreshold,
    float nmsThreshold
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

            int x = static_cast<int>(cx * detector.img_scale);
            int y = static_cast<int>(cy * detector.img_scale);
            int width = static_cast<int>((dx - cx) * detector.img_scale);
            int height = static_cast<int>((dy - cy) * detector.img_scale);

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

std::vector<Detection> postProcessYolo11(
    const float* output,
    const std::vector<int64_t>& shape,
    int numClasses,
    float confThreshold,
    float nmsThreshold
)
{
    if (shape.size() != 3)
    {
        std::cerr << "[postProcess] Unsupported output shape" << std::endl;
        return std::vector<Detection>();
    }

    // Pre-allocate memory - this is likely an overestimation but prevents reallocations
    std::vector<Detection> detections;
    detections.reserve(shape[2]);  // Reserve based on expected number of detections

    int rows = shape[1];
    int cols = shape[2];
    
    // Check if we have enough data to process
    if (rows < 4 + numClasses)
    {
        std::cerr << "[postProcess] Number of classes exceeds available rows in det_output" << std::endl;
        return detections;
    }
    
    // Create a cv::Mat view of the output data for easier processing
    cv::Mat det_output(rows, cols, CV_32F, (void*)output);

    // Cache the img_scale value to avoid multiple lookups
    const float img_scale = detector.img_scale;
    
    // Process each detection
    for (int i = 0; i < cols; ++i)
    {
        // Get the class scores for this detection
        cv::Mat classes_scores = det_output.col(i).rowRange(4, 4 + numClasses);

        // Find the class with the highest confidence
        cv::Point class_id_point;
        double score;
        cv::minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);

        // Only process detections with confidence above threshold
        if (score > confThreshold)
        {
            // Get bounding box coordinates
            float cx = det_output.at<float>(0, i);
            float cy = det_output.at<float>(1, i);
            float ow = det_output.at<float>(2, i);
            float oh = det_output.at<float>(3, i);

            // Calculate scaled box coordinates
            const float half_ow = 0.5f * ow;
            const float half_oh = 0.5f * oh;
            
            // Create bounding box
            cv::Rect box;
            box.x = static_cast<int>((cx - half_ow) * img_scale);
            box.y = static_cast<int>((cy - half_oh) * img_scale);
            box.width = static_cast<int>(ow * img_scale);
            box.height = static_cast<int>(oh * img_scale);

            // Create detection object
            Detection detection;
            detection.box = box;
            detection.confidence = static_cast<float>(score);
            detection.classId = class_id_point.y;

            detections.push_back(detection);
        }
    }

    // Apply NMS to filter overlapping detections
    if (!detections.empty())
    {
        NMS(detections, nmsThreshold);
    }

    return detections;
}