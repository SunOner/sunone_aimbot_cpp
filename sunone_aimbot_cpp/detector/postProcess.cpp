// *** CORRECCIÓN: Definir NOMINMAX antes de cualquier inclusión de Windows ***
#define NOMINMAX

#include <algorithm>
#include <numeric>
#include <chrono>
#include <vector>
#include <immintrin.h>

#include "postProcess.h"
#include "sunone_aimbot_cpp.h"
#ifdef USE_CUDA
#include "trt_detector.h"
#endif

// --- Implementación NMS Optimizada con AVX2 (sin cambios, ya era correcta) ---
void NMS(std::vector<Detection>& detections, float nmsThreshold, std::chrono::duration<double, std::milli>* nmsTime)
{
    if (detections.empty()) return;

    auto t0 = std::chrono::steady_clock::now();

    std::sort(
        detections.begin(),
        detections.end(),
        [](const Detection& a, const Detection& b) {
            return a.confidence > b.confidence;
        }
    );

    const size_t num_boxes = detections.size();
    std::vector<float> x1(num_boxes);
    std::vector<float> y1(num_boxes);
    std::vector<float> x2(num_boxes);
    std::vector<float> y2(num_boxes);
    std::vector<float> areas(num_boxes);
    
    for (size_t i = 0; i < num_boxes; ++i) {
        x1[i] = static_cast<float>(detections[i].box.x);
        y1[i] = static_cast<float>(detections[i].box.y);
        x2[i] = static_cast<float>(detections[i].box.x + detections[i].box.width);
        y2[i] = static_cast<float>(detections[i].box.y + detections[i].box.height);
        areas[i] = static_cast<float>(detections[i].box.width * detections[i].box.height);
    }

    std::vector<bool> suppress(num_boxes, false);
    const __m256 nms_threshold_vec = _mm256_set1_ps(nmsThreshold);

    for (size_t i = 0; i < num_boxes; ++i)
    {
        if (suppress[i]) continue;

        const __m256 i_x1_vec = _mm256_set1_ps(x1[i]);
        const __m256 i_y1_vec = _mm256_set1_ps(y1[i]);
        const __m256 i_x2_vec = _mm256_set1_ps(x2[i]);
        const __m256 i_y2_vec = _mm256_set1_ps(y2[i]);
        const __m256 i_area_vec = _mm256_set1_ps(areas[i]);

        size_t j = i + 1;
        for (; j + 7 < num_boxes; j += 8)
        {
            __m256 j_x1_vec = _mm256_loadu_ps(&x1[j]);
            __m256 j_y1_vec = _mm256_loadu_ps(&y1[j]);
            __m256 j_x2_vec = _mm256_loadu_ps(&x2[j]);
            __m256 j_y2_vec = _mm256_loadu_ps(&y2[j]);
            __m256 j_area_vec = _mm256_loadu_ps(&areas[j]);

            __m256 xx1 = _mm256_max_ps(i_x1_vec, j_x1_vec);
            __m256 yy1 = _mm256_max_ps(i_y1_vec, j_y1_vec);
            __m256 xx2 = _mm256_min_ps(i_x2_vec, j_x2_vec);
            __m256 yy2 = _mm256_min_ps(i_y2_vec, j_y2_vec);

            __m256 width = _mm256_sub_ps(xx2, xx1);
            __m256 height = _mm256_sub_ps(yy2, yy1);

            width = _mm256_max_ps(_mm256_setzero_ps(), width);
            height = _mm256_max_ps(_mm256_setzero_ps(), height);

            __m256 inter_area = _mm256_mul_ps(width, height);
            __m256 union_area = _mm256_sub_ps(_mm256_add_ps(i_area_vec, j_area_vec), inter_area);

            __m256 iou = _mm256_div_ps(inter_area, union_area);
            __m256 suppress_mask_vec = _mm256_cmp_ps(iou, nms_threshold_vec, _CMP_GT_OQ);
            
            int suppress_mask = _mm256_movemask_ps(suppress_mask_vec);
            if (suppress_mask != 0) {
                for(int k=0; k<8; ++k) {
                    if((suppress_mask >> k) & 1 && !suppress[j+k]) {
                        suppress[j+k] = true;
                    }
                }
            }
        }
        
        for (; j < num_boxes; ++j)
        {
            if (suppress[j]) continue;

            // *** CORRECCIÓN: Usar (std::max) y (std::min) para evitar conflicto de macros ***
            // O, mejor aún, haber definido NOMINMAX al principio del archivo.
            float xx1 = std::max(x1[i], x1[j]);
            float yy1 = std::max(y1[i], y1[j]);
            float xx2 = std::min(x2[i], x2[j]);
            float yy2 = std::min(y2[i], y2[j]);

            float w = std::max(0.0f, xx2 - xx1);
            float h = std::max(0.0f, yy2 - yy1);

            float inter_area = w * h;
            float union_area = areas[i] + areas[j] - inter_area;

            if (union_area > 0 && inter_area / union_area > nmsThreshold)
            {
                suppress[j] = true;
            }
        }
    }

    std::vector<Detection> result;
    result.reserve(detections.size()); // Reservar el tamaño original es una buena heurística
    for (size_t i = 0; i < num_boxes; ++i) {
        if (!suppress[i]) {
            result.push_back(std::move(detections[i])); // Usar move para eficiencia
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
    const float* output, const std::vector<int64_t>& shape, int numClasses,
    float confThreshold, float nmsThreshold, std::chrono::duration<double, std::milli>* nmsTime)
{
    std::vector<Detection> detections;
    int64_t numDetections = shape[1];
    detections.reserve(numDetections);

    for (int i = 0; i < numDetections; ++i) {
        const float* det = output + i * shape[2];
        if (det[4] > confThreshold) {
            detections.push_back({
                cv::Rect(
                    static_cast<int>(det[0] * trt_detector.img_scale),
                    static_cast<int>(det[1] * trt_detector.img_scale),
                    static_cast<int>((det[2] - det[0]) * trt_detector.img_scale),
                    static_cast<int>((det[3] - det[1]) * trt_detector.img_scale)
                ),
                det[4],
                static_cast<int>(det[5])
            });
        }
    }
    NMS(detections, nmsThreshold, nmsTime);
    return detections;
}

std::vector<Detection> postProcessYolo11(
    const float* output, const std::vector<int64_t>& shape, int numClasses,
    float confThreshold, float nmsThreshold, std::chrono::duration<double, std::milli>* nmsTime)
{
    std::vector<Detection> detections;
    if (shape.size() != 3) return {};
    // La forma es [batch, rows, cols], las detecciones están en la última dimensión
    detections.reserve(shape[2]); 
    cv::Mat det_output(shape[1], shape[2], CV_32F, (void*)output);
    const float img_scale = trt_detector.img_scale;
    
    for (int i = 0; i < shape[2]; ++i) {
        cv::Mat scores = det_output.col(i).rowRange(4, 4 + numClasses);
        cv::Point class_id;
        double score;
        cv::minMaxLoc(scores, nullptr, &score, nullptr, &class_id);

        if (score > confThreshold) {
            float cx = det_output.at<float>(0, i);
            float cy = det_output.at<float>(1, i);
            float ow = det_output.at<float>(2, i);
            float oh = det_output.at<float>(3, i);
            detections.push_back({
                cv::Rect(
                    static_cast<int>((cx - 0.5f * ow) * img_scale),
                    static_cast<int>((cy - 0.5f * oh) * img_scale),
                    static_cast<int>(ow * img_scale),
                    static_cast<int>(oh * img_scale)
                ),
                static_cast<float>(score),
                class_id.y
            });
        }
    }
    NMS(detections, nmsThreshold, nmsTime);
    return detections;
}
#endif

std::vector<Detection> postProcessYolo10DML(
    const float* output, const std::vector<int64_t>& shape, int numClasses,
    float confThreshold, float nmsThreshold, std::chrono::duration<double, std::milli>* nmsTime)
{
    std::vector<Detection> detections;
    int64_t numDetections = shape[1];
    detections.reserve(numDetections);

    for (int i = 0; i < numDetections; ++i) {
        const float* det = output + i * shape[2];
        if (det[4] > confThreshold) {
            detections.push_back({
                cv::Rect(
                    static_cast<int>(det[0]), static_cast<int>(det[1]),
                    static_cast<int>(det[2] - det[0]), static_cast<int>(det[3] - det[1])
                ),
                det[4],
                static_cast<int>(det[5])
            });
        }
    }
    NMS(detections, nmsThreshold, nmsTime);
    return detections;
}

std::vector<Detection> postProcessYolo11DML(
    const float* output, const std::vector<int64_t>& shape, int numClasses,
    float confThreshold, float nmsThreshold, std::chrono::duration<double, std::milli>* nmsTime)
{
    std::vector<Detection> detections;
    if (shape.size() != 2) return detections;
    
    // *** CORRECCIÓN: La forma es [rows, cols], el número de detecciones está en shape[1] ***
    detections.reserve(shape[1]); 
    
    const int HEAD_CLASS_ID = 1;
    const float HEAD_MIN_CONFIDENCE = confThreshold - 0.10f;
    cv::Mat det_output(shape[0], shape[1], CV_32F, (void*)output);

    for (int i = 0; i < shape[1]; ++i) { // Iterar sobre las columnas (detecciones)
        cv::Mat scores = det_output.col(i).rowRange(4, 4 + numClasses);
        cv::Point class_id;
        double score;
        cv::minMaxLoc(scores, nullptr, &score, nullptr, &class_id);
        float min_conf = (class_id.y == HEAD_CLASS_ID) ? HEAD_MIN_CONFIDENCE : confThreshold;

        if (score > min_conf) {
            float cx = det_output.at<float>(0, i);
            float cy = det_output.at<float>(1, i);
            float ow = det_output.at<float>(2, i);
            float oh = det_output.at<float>(3, i);
            detections.push_back({
                cv::Rect(
                    static_cast<int>(cx - 0.5f * ow), static_cast<int>(cy - 0.5f * oh),
                    static_cast<int>(ow), static_cast<int>(oh)
                ),
                static_cast<float>(score),
                class_id.y
            });
        }
    }
    NMS(detections, nmsThreshold, nmsTime);
    return detections;
}