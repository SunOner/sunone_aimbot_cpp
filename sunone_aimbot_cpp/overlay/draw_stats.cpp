#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include "imgui/imgui.h"
#include "sunone_aimbot_cpp.h"
#include "overlay.h"
#include "capture.h"

void draw_stats()
{
    // all stages
    static float preprocess_times[120] = {};
    static float inference_times[120] = {};
    static float copy_times[120] = {};
    static float postprocess_times[120] = {};
    static float nms_times[120] = {};
    static int index_inf = 0;

    float current_preprocess = 0.0f;
    float current_inference = 0.0f;
    float current_copy = 0.0f;
    float current_post = 0.0f;
    float current_nms = 0.0f;

    if (config.backend == "DML" && dml_detector)
    {
        current_preprocess = static_cast<float>(dml_detector->lastPreprocessTimeDML.count());
        current_inference = static_cast<float>(dml_detector->lastInferenceTimeDML.count());
        current_copy = static_cast<float>(dml_detector->lastCopyTimeDML.count());
        current_post = static_cast<float>(dml_detector->lastPostprocessTimeDML.count());
        current_nms = static_cast<float>(dml_detector->lastNmsTimeDML.count());
    }
#ifdef USE_CUDA
    else
    {
        current_preprocess = static_cast<float>(trt_detector.lastPreprocessTime.count());
        current_inference = static_cast<float>(trt_detector.lastInferenceTime.count());
        current_copy = static_cast<float>(trt_detector.lastCopyTime.count());
        current_post = static_cast<float>(trt_detector.lastPostprocessTime.count());
        current_nms = static_cast<float>(trt_detector.lastNmsTime.count());
    }
#endif
    preprocess_times[index_inf] = current_preprocess;
    inference_times[index_inf] = current_inference;
    copy_times[index_inf] = current_copy;
    postprocess_times[index_inf] = current_post;
    nms_times[index_inf] = current_nms;
    index_inf = (index_inf + 1) % IM_ARRAYSIZE(inference_times);

    auto avg = [](const float* arr, int n) -> float {
        float sum = 0.0f; int cnt = 0;
        for (int i = 0; i < n; ++i) if (arr[i] > 0.0f) { sum += arr[i]; ++cnt; }
        return cnt ? sum / cnt : 0.0f;
        };

    float avg_preprocess = avg(preprocess_times, IM_ARRAYSIZE(preprocess_times));
    float avg_inference = avg(inference_times, IM_ARRAYSIZE(inference_times));
    float avg_copy = avg(copy_times, IM_ARRAYSIZE(copy_times));
    float avg_post = avg(postprocess_times, IM_ARRAYSIZE(postprocess_times));
    float avg_nms = avg(nms_times, IM_ARRAYSIZE(nms_times));

    ImGui::SeparatorText("Time Breakdown");

    ImGui::PlotLines("Preprocess", preprocess_times, IM_ARRAYSIZE(preprocess_times), index_inf, nullptr, 0.0f, 20.0f, ImVec2(0, 40));
    ImGui::SameLine(); ImGui::Text("%.2f | Avg: %.2f", current_preprocess, avg_preprocess);

    ImGui::PlotLines("Inference", inference_times, IM_ARRAYSIZE(inference_times), index_inf, nullptr, 0.0f, 20.0f, ImVec2(0, 40));
    ImGui::SameLine(); ImGui::Text("%.2f | Avg: %.2f", current_inference, avg_inference);

    ImGui::PlotLines("Copy", copy_times, IM_ARRAYSIZE(copy_times), index_inf, nullptr, 0.0f, 10.0f, ImVec2(0, 40));
    ImGui::SameLine(); ImGui::Text("%.2f | Avg: %.2f", current_copy, avg_copy);

    ImGui::PlotLines("Postprocess", postprocess_times, IM_ARRAYSIZE(postprocess_times), index_inf, nullptr, 0.0f, 10.0f, ImVec2(0, 40));
    ImGui::SameLine(); ImGui::Text("%.2f | Avg: %.2f", current_post, avg_post);

    ImGui::PlotLines("NMS", nms_times, IM_ARRAYSIZE(nms_times), index_inf, nullptr, 0.0f, 5.0f, ImVec2(0, 40));
    ImGui::SameLine(); ImGui::Text("%.2f | Avg: %.2f", current_nms, avg_nms);

    // Capture FPS
    static float capture_fps_vals[120] = {};
    static int index_fps = 0;

    float current_fps = static_cast<float>(captureFps.load());
    capture_fps_vals[index_fps] = current_fps;
    index_fps = (index_fps + 1) % IM_ARRAYSIZE(capture_fps_vals);

    float sum_fps = 0.0f;
    int count_fps = 0;
    for (float f : capture_fps_vals)
    {
        if (f > 0.0f) { sum_fps += f; ++count_fps; }
    }
    float avg_fps = (count_fps > 0) ? (sum_fps / count_fps) : 0.0f;

    ImGui::SeparatorText("Capture FPS");
    ImGui::PlotLines("##fps_plot", capture_fps_vals, IM_ARRAYSIZE(capture_fps_vals), index_fps, nullptr, 0.0f, 144.0f, ImVec2(0, 60));
    ImGui::SameLine();
    ImGui::Text("Now: %.1f | Avg: %.1f", current_fps, avg_fps);
}
