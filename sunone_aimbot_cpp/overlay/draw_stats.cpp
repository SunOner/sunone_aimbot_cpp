#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include "imgui/imgui.h"
#include "sunone_aimbot_cpp.h"
#include "overlay.h"
#include <capture.h>

void draw_stats()
{
    // Inference speed
    static float inference_times[120] = {};
    static int index_inf = 0;

    float current_time = static_cast<float>(detector.lastInferenceTime.count());
    inference_times[index_inf] = current_time;
    index_inf = (index_inf + 1) % IM_ARRAYSIZE(inference_times);

    float sum_inf = 0.0f;
    int count_inf = 0;
    for (float t : inference_times) {
        if (t > 0.0f) { sum_inf += t; ++count_inf; }
    }
    float avg_inf = (count_inf > 0) ? (sum_inf / count_inf) : 0.0f;

    ImGui::SeparatorText("Inference Time");
    ImGui::PlotLines("##inference_plot", inference_times, IM_ARRAYSIZE(inference_times), index_inf, nullptr, 0.0f, 50.0f, ImVec2(0, 60));
    ImGui::SameLine();
    ImGui::Text("Now: %.2f ms | Avg: %.2f ms", current_time, avg_inf);

    // Capture FPS
    static float capture_fps_vals[120] = {};
    static int index_fps = 0;

    float current_fps = static_cast<float>(captureFps.load());
    capture_fps_vals[index_fps] = current_fps;
    index_fps = (index_fps + 1) % IM_ARRAYSIZE(capture_fps_vals);

    float sum_fps = 0.0f;
    int count_fps = 0;
    for (float f : capture_fps_vals) {
        if (f > 0.0f) { sum_fps += f; ++count_fps; }
    }
    float avg_fps = (count_fps > 0) ? (sum_fps / count_fps) : 0.0f;

    ImGui::SeparatorText("Capture FPS");
    ImGui::PlotLines("##fps_plot", capture_fps_vals, IM_ARRAYSIZE(capture_fps_vals), index_fps, nullptr, 0.0f, 144.0f, ImVec2(0, 60));
    ImGui::SameLine();
    ImGui::Text("Now: %.1f | Avg: %.1f", current_fps, avg_fps);
}