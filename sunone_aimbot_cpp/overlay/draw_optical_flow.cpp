#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include "imgui/imgui.h"
#include "sunone_aimbot_cpp.h"

void draw_optical_flow()
{
    ImGui::Text("This is an experimental feature");

    if (ImGui::Checkbox("Enable Optical Flow", &config.enable_optical_flow))
    {
        config.saveConfig();
        opticalFlow.manageOpticalFlowThread();
    }

    ImGui::Separator();
    if (ImGui::Checkbox("Draw in debug window", &config.draw_optical_flow))
    {
        config.saveConfig();
    }

    if (config.draw_optical_flow)
    {
        ImGui::Separator();
        if (ImGui::SliderInt("Draw steps", &config.draw_optical_flow_steps, 2, 32))
        {
            config.saveConfig();
        }
        ImGui::Separator();
    }

    if (ImGui::SliderFloat("Alpha CPU", &config.optical_flow_alpha_cpu, 0.01f, 1.00f, "%.2f"))
    {
        config.saveConfig();
    }

    float magnitudeThreshold = static_cast<float>(config.optical_flow_magnitudeThreshold);
    if (ImGui::SliderFloat("Magnitude Threshold", &magnitudeThreshold, 0.01f, 10.00f, "%.2f"))
    {
        config.optical_flow_magnitudeThreshold = static_cast<double>(magnitudeThreshold);
        config.saveConfig();
    }

    if (ImGui::SliderFloat("Static Frame Threshold", &config.staticFrameThreshold, 0.01f, 10.00f, "%.2f"))
    {
        config.saveConfig();
    }
}