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

    ImGui::SeparatorText("Inference Time");
    ImGui::PlotLines("Inference:", inference_times, IM_ARRAYSIZE(inference_times), index_inf, nullptr, 0.0f, 50.0f, ImVec2(0, 60));
    ImGui::SameLine();
    ImGui::Text("%.2f ms", current_time);

    // Capture FPS
    static float capture_fps_vals[120] = {};
    static int index_fps = 0;

    float current_fps = static_cast<float>(captureFps.load());
    capture_fps_vals[index_fps] = current_fps;
    index_fps = (index_fps + 1) % IM_ARRAYSIZE(capture_fps_vals);

    ImGui::SeparatorText("Capture FPS");
    ImGui::PlotLines("FPS:", capture_fps_vals, IM_ARRAYSIZE(capture_fps_vals), index_fps, nullptr, 0.0f, 144.0f, ImVec2(0, 60));
    ImGui::SameLine();
    ImGui::Text("%.1f", current_fps);
}