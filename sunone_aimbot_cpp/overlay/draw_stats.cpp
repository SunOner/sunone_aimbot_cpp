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
    static float preprocess_times[120] = {};
    static float inference_times[120] = {};
    static float copy_times[120] = {};
    static float postprocess_times[120] = {};
    static float nms_times[120] = {};
    static int index_inf = 0;

    static float capture_fps_vals[120] = {};
    static int index_fps = 0;

    static float avg_preprocess_cached = 0.0f;
    static float avg_inference_cached = 0.0f;
    static float avg_copy_cached = 0.0f;
    static float avg_post_cached = 0.0f;
    static float avg_nms_cached = 0.0f;
    static float avg_fps_cached = 0.0f;
    static double last_avg_update_time = 0.0;

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

    float current_fps = static_cast<float>(captureFps.load());
    capture_fps_vals[index_fps] = current_fps;
    index_fps = (index_fps + 1) % IM_ARRAYSIZE(capture_fps_vals);

    auto avg = [](const float* arr, int n) -> float {
        float sum = 0.0f; int cnt = 0;
        for (int i = 0; i < n; ++i)
            if (arr[i] > 0.0f) { sum += arr[i]; ++cnt; }
        return cnt ? (sum / cnt) : 0.0f;
        };

    const double now = ImGui::GetTime();
    if (last_avg_update_time == 0.0 || (now - last_avg_update_time) >= 1.0)
    {
        avg_preprocess_cached = avg(preprocess_times, IM_ARRAYSIZE(preprocess_times));
        avg_inference_cached = avg(inference_times, IM_ARRAYSIZE(inference_times));
        avg_copy_cached = avg(copy_times, IM_ARRAYSIZE(copy_times));
        avg_post_cached = avg(postprocess_times, IM_ARRAYSIZE(postprocess_times));
        avg_nms_cached = avg(nms_times, IM_ARRAYSIZE(nms_times));
        avg_fps_cached = avg(capture_fps_vals, IM_ARRAYSIZE(capture_fps_vals));

        last_avg_update_time = now;
    }

    ImGui::SeparatorText("Time Breakdown");

    ImGui::PlotLines("Preprocess", preprocess_times, IM_ARRAYSIZE(preprocess_times), index_inf, nullptr, 0.0f, 20.0f, ImVec2(0, 40));
    ImGui::SameLine(); ImGui::Text("%.2f | Avg: %.2f", current_preprocess, avg_preprocess_cached);

    ImGui::PlotLines("Inference", inference_times, IM_ARRAYSIZE(inference_times), index_inf, nullptr, 0.0f, 20.0f, ImVec2(0, 40));
    ImGui::SameLine();

    ImGui::Text("%.2f | Avg:", current_inference);
    ImGui::SameLine();

    const bool inf_slow = (avg_inference_cached > 20.0f);
    if (inf_slow)
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.2f, 0.2f, 1.0f));

    ImGui::Text("%.2f", avg_inference_cached);

    if (inf_slow)
        ImGui::PopStyleColor();

    ImGui::PlotLines("Copy", copy_times, IM_ARRAYSIZE(copy_times), index_inf, nullptr, 0.0f, 10.0f, ImVec2(0, 40));
    ImGui::SameLine(); ImGui::Text("%.2f | Avg: %.2f", current_copy, avg_copy_cached);

    ImGui::PlotLines("Postprocess", postprocess_times, IM_ARRAYSIZE(postprocess_times), index_inf, nullptr, 0.0f, 10.0f, ImVec2(0, 40));
    ImGui::SameLine(); ImGui::Text("%.2f | Avg: %.2f", current_post, avg_post_cached);

    ImGui::PlotLines("NMS", nms_times, IM_ARRAYSIZE(nms_times), index_inf, nullptr, 0.0f, 5.0f, ImVec2(0, 40));
    ImGui::SameLine(); ImGui::Text("%.2f | Avg: %.2f", current_nms, avg_nms_cached);

    ImGui::SeparatorText("Capture FPS");
    ImGui::PlotLines("##fps_plot", capture_fps_vals, IM_ARRAYSIZE(capture_fps_vals), index_fps, nullptr, 0.0f, 144.0f, ImVec2(0, 60));
    ImGui::SameLine();
    ImGui::Text("Now: %.1f | Avg: %.1f", current_fps, avg_fps_cached);
}