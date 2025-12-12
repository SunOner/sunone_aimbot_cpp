#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <d3d11.h>

#include "imgui/imgui.h"
#include "sunone_aimbot_cpp.h"
#include "overlay.h"
#include "include/other_tools.h"
#include "capture.h"

#ifndef SAFE_RELEASE
#define SAFE_RELEASE(p)       \
    do {                      \
        if ((p) != nullptr) { \
            (p)->Release();   \
            (p) = nullptr;    \
        }                     \
    } while (0)
#endif

int prev_screenshot_delay = config.screenshot_delay;
bool prev_verbose = config.verbose;

static ID3D11Texture2D* g_debugTex = nullptr;
static ID3D11ShaderResourceView* g_debugSRV = nullptr;
static int texW = 0, texH = 0;


static float debug_scale = 1.0f;

static void uploadDebugFrame(const cv::Mat& bgr)
{
    if (bgr.empty()) return;

    if (!g_debugTex || bgr.cols != texW || bgr.rows != texH)
    {
        SAFE_RELEASE(g_debugTex);
        SAFE_RELEASE(g_debugSRV);

        texW = bgr.cols;  texH = bgr.rows;

        D3D11_TEXTURE2D_DESC td = {};
        td.Width = texW;
        td.Height = texH;
        td.MipLevels = td.ArraySize = 1;
        td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DYNAMIC;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        td.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

        g_pd3dDevice->CreateTexture2D(&td, nullptr, &g_debugTex);

        D3D11_SHADER_RESOURCE_VIEW_DESC sd = {};
        sd.Format = td.Format;
        sd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        sd.Texture2D.MipLevels = 1;
        g_pd3dDevice->CreateShaderResourceView(g_debugTex, &sd, &g_debugSRV);
    }

    static cv::Mat rgba;
    cv::cvtColor(bgr, rgba, cv::COLOR_BGR2RGBA);

    D3D11_MAPPED_SUBRESOURCE ms;
    if (SUCCEEDED(g_pd3dDeviceContext->Map(g_debugTex, 0,
        D3D11_MAP_WRITE_DISCARD, 0, &ms)))
    {
        for (int y = 0; y < texH; ++y)
            memcpy((uint8_t*)ms.pData + ms.RowPitch * y,
                rgba.ptr(y), texW * 4);
        g_pd3dDeviceContext->Unmap(g_debugTex, 0);
    }
}

void draw_debug_frame()
{
    cv::Mat frameCopy;
    {
        std::lock_guard<std::mutex> lk(frameMutex);
        if (!latestFrame.empty())
            latestFrame.copyTo(frameCopy);
    }

    uploadDebugFrame(frameCopy);

    if (!g_debugSRV) return;

    ImGui::SliderFloat("Debug scale", &debug_scale, 0.1f, 2.0f, "%.1fx");

    ImVec2 image_size(texW * debug_scale, texH * debug_scale);
    ImGui::Image(g_debugSRV, image_size);

    ImVec2 image_pos = ImGui::GetItemRectMin();
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    {
        std::lock_guard<std::mutex> lock(detectionBuffer.mutex);
        for (size_t i = 0; i < detectionBuffer.boxes.size(); ++i)
        {
            const cv::Rect& box = detectionBuffer.boxes[i];

            ImVec2 p1(image_pos.x + box.x * debug_scale,
                image_pos.y + box.y * debug_scale);
            ImVec2 p2(p1.x + box.width * debug_scale,
                p1.y + box.height * debug_scale);

            ImU32 color = IM_COL32(255, 0, 0, 255);

            draw_list->AddRect(p1, p2, color, 0.0f, 0, 2.0f);

            if (i < detectionBuffer.classes.size())
            {
                std::string label = "Class " + std::to_string(detectionBuffer.classes[i]);
                draw_list->AddText(ImVec2(p1.x, p1.y - 16), IM_COL32(255, 255, 0, 255), label.c_str());
            }
        }
    }

    if (config.draw_futurePositions && globalMouseThread)
    {
        auto futurePts = globalMouseThread->getFuturePositions();
        if (!futurePts.empty())
        {
            float scale_x = static_cast<float>(texW) / config.detection_resolution;
            float scale_y = static_cast<float>(texH) / config.detection_resolution;

            ImVec2 clip_min = image_pos;
            ImVec2 clip_max = ImVec2(image_pos.x + texW * debug_scale,
                image_pos.y + texH * debug_scale);
            draw_list->PushClipRect(clip_min, clip_max, true);

            int totalPts = static_cast<int>(futurePts.size());
            for (size_t i = 0; i < futurePts.size(); ++i)
            {
                int px = static_cast<int>(futurePts[i].first * scale_x);
                int py = static_cast<int>(futurePts[i].second * scale_y);
                ImVec2 pt(image_pos.x + px * debug_scale,
                    image_pos.y + py * debug_scale);

                int b = static_cast<int>(255 - (i * 255.0 / totalPts));
                int r = static_cast<int>(i * 255.0 / totalPts);
                int g = 50;

                ImU32 fillColor = IM_COL32(r, g, b, 255);
                ImU32 outlineColor = IM_COL32(255, 255, 255, 255);

                draw_list->AddCircleFilled(pt, 4.0f * debug_scale, fillColor);
                draw_list->AddCircle(pt, 4.0f * debug_scale, outlineColor, 0, 1.0f);
            }

            draw_list->PopClipRect();
        }
    }
}

void draw_debug()
{
    ImGui::Checkbox("Show Debug Window", &config.show_window);
    if (config.show_window)
    {
        draw_debug_frame();
        ImGui::Separator();
    }

    ImGui::Text("Screenshot Buttons");

    for (size_t i = 0; i < config.screenshot_button.size(); )
    {
        std::string& current_key_name = config.screenshot_button[i];

        int current_index = -1;
        for (size_t k = 0; k < key_names.size(); ++k)
        {
            if (key_names[k] == current_key_name)
            {
                current_index = static_cast<int>(k);
                break;
            }
        }

        if (current_index == -1)
        {
            current_index = 0;
        }

        std::string combo_label = "Screenshot Button " + std::to_string(i);

        if (ImGui::Combo(combo_label.c_str(), &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
        {
            current_key_name = key_names[current_index];
            config.saveConfig("config.ini");
        }

        ImGui::SameLine();
        std::string remove_button_label = "Remove##button_screenshot" + std::to_string(i);
        if (ImGui::Button(remove_button_label.c_str()))
        {
            if (config.screenshot_button.size() <= 1)
            {
                config.screenshot_button[0] = std::string("None");
                config.saveConfig();
                continue;
            }
            else
            {
                config.screenshot_button.erase(config.screenshot_button.begin() + i);
                config.saveConfig();
                continue;
            }
        }

        ++i;
    }

    if (ImGui::Button("Add button##button_screenshot"))
    {
        config.screenshot_button.push_back("None");
        config.saveConfig();
    }

    ImGui::InputInt("Screenshot delay", &config.screenshot_delay, 50, 500);
    ImGui::Checkbox("Verbose console output", &config.verbose);

    ImGui::Separator();

    ImGui::Text("Test functions");
    if (ImGui::Button("Free terminal"))
    {
        HideConsole();
    }
    ImGui::SameLine();
    if (ImGui::Button("Restore terminal"))
    {
        ShowConsole();
    }

    if (prev_screenshot_delay != config.screenshot_delay ||
        prev_verbose != config.verbose)
    {
        prev_screenshot_delay = config.screenshot_delay;
        prev_verbose = config.verbose;
        config.saveConfig();
    }
}