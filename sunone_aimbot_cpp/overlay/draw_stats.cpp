#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <algorithm>
#include <string>

#include "imgui/imgui.h"
#include "sunone_aimbot_cpp.h"
#include "overlay.h"
#include "capture.h"
#include "overlay/ui_sections.h"

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

    if (OverlayUI::BeginSection("Time Breakdown", "stats_section_time_breakdown"))
    {
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

        OverlayUI::EndSection();
    }

    if (OverlayUI::BeginSection("Capture FPS", "stats_section_capture_fps"))
    {
        ImGui::PlotLines("##fps_plot", capture_fps_vals, IM_ARRAYSIZE(capture_fps_vals), index_fps, nullptr, 0.0f, 144.0f, ImVec2(0, 60));
        ImGui::SameLine();
        ImGui::Text("Now: %.1f | Avg: %.1f", current_fps, avg_fps_cached);
        OverlayUI::EndSection();
    }

    int latestWidth = 0;
    int latestHeight = 0;
    size_t queueDepth = 0;
    {
        std::lock_guard<std::mutex> lk(frameMutex);
        if (!latestFrame.empty())
        {
            latestWidth = latestFrame.cols;
            latestHeight = latestFrame.rows;
        }
        queueDepth = frameQueue.size();
    }

    const int captureFpsLimit = std::max(0, config.capture_fps);
    const float currentFrameTimeMs = (current_fps > 0.01f) ? (1000.0f / current_fps) : 0.0f;
    const float avgFrameTimeMs = (avg_fps_cached > 0.01f) ? (1000.0f / avg_fps_cached) : 0.0f;

    std::string captureSource = "Unknown";
    if (config.capture_method == "duplication_api")
    {
        captureSource = "Monitor " + std::to_string(std::max(0, config.monitor_idx) + 1);
    }
    else if (config.capture_method == "winrt")
    {
        if (config.capture_target == "window")
        {
            captureSource = config.capture_window_title.empty()
                ? "Window target is empty"
                : "Window: " + config.capture_window_title;
        }
        else
        {
            captureSource = "Monitor " + std::to_string(std::max(0, config.monitor_idx) + 1);
        }
    }
    else if (config.capture_method == "virtual_camera")
    {
        captureSource =
            "Camera: " + config.virtual_camera_name + " (" +
            std::to_string(config.virtual_camera_width) + "x" +
            std::to_string(config.virtual_camera_heigth) + ")";
    }
    else if (config.capture_method == "udp_capture")
    {
        captureSource = "UDP " + config.udp_ip + ":" + std::to_string(config.udp_port);
    }

    if (OverlayUI::BeginSection("Capture Details", "stats_section_capture_details"))
    {
        ImGui::Text("Method: %s", config.capture_method.c_str());
        ImGui::Text("Backend: %s", config.backend.c_str());
        ImGui::TextWrapped("Source: %s", captureSource.c_str());

        if (screenWidth > 0 && screenHeight > 0)
            ImGui::Text("Desktop size: %dx%d", screenWidth, screenHeight);
        else
            ImGui::TextDisabled("Desktop size: n/a");

        if (latestWidth > 0 && latestHeight > 0)
            ImGui::Text("Latest frame: %dx%d", latestWidth, latestHeight);
        else
            ImGui::TextDisabled("Latest frame: n/a");

        ImGui::Text("Detection resolution: %d", config.detection_resolution);
        if (captureFpsLimit > 0)
            ImGui::Text("Capture FPS limit: %d", captureFpsLimit);
        else
            ImGui::Text("Capture FPS limit: unlimited");

        if (currentFrameTimeMs > 0.0f || avgFrameTimeMs > 0.0f)
            ImGui::Text("Frame time: now %.2f ms | avg %.2f ms", currentFrameTimeMs, avgFrameTimeMs);
        else
            ImGui::TextDisabled("Frame time: n/a");

        ImGui::Text("Frame queue depth: %d", static_cast<int>(queueDepth));
        ImGui::Text("Circle mask: %s", config.circle_mask ? "on" : "off");

#ifdef USE_CUDA
        if (config.backend == "TRT")
        {
            const bool depthMaskEnabled = config.depth_inference_enabled && config.depth_mask_enabled;
            const bool canUseCudaCapture = (config.capture_method == "duplication_api");
            const bool directCaptureActive =
                canUseCudaCapture &&
                config.capture_use_cuda &&
                !config.circle_mask &&
                !depthMaskEnabled;

            std::string directCaptureStatus;
            if (!canUseCudaCapture)
                directCaptureStatus = "N/A (requires duplication_api)";
            else if (!config.capture_use_cuda)
                directCaptureStatus = "Disabled by user";
            else if (config.circle_mask)
                directCaptureStatus = "CPU fallback (circle mask is enabled)";
            else if (depthMaskEnabled)
                directCaptureStatus = "CPU fallback (depth mask is enabled)";
            else
                directCaptureStatus = "Active";

            ImGui::Separator();
            ImGui::Text("CUDA Direct Capture: %s", config.capture_use_cuda ? "enabled" : "disabled");
            ImGui::Text("Depth mask: %s", depthMaskEnabled ? "on" : "off");
            ImGui::Text("Capture pipeline: %s", directCaptureActive ? "GPU direct path" : "CPU readback");

            if (directCaptureActive)
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.35f, 1.0f, 0.45f, 1.0f));
            else
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.78f, 0.28f, 1.0f));
            ImGui::TextWrapped("Direct capture status: %s", directCaptureStatus.c_str());
            ImGui::PopStyleColor();
        }
#endif

        OverlayUI::EndSection();
    }
}
