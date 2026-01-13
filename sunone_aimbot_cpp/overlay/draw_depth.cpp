#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <commdlg.h>
#include <chrono>
#include <string>

#include <d3d11.h>

#include "imgui/imgui.h"
#include "sunone_aimbot_cpp.h"
#include "overlay.h"
#include "capture.h"
#include "draw_settings.h"

#ifdef USE_CUDA
#include "depth/depth_anything_trt.h"
#include "tensorrt/nvinf.h"
#endif

#ifndef SAFE_RELEASE
#define SAFE_RELEASE(p)       \
    do {                      \
        if ((p) != nullptr) { \
            (p)->Release();   \
            (p) = nullptr;    \
        }                     \
    } while (0)
#endif

static ID3D11Texture2D* g_depthTex = nullptr;
static ID3D11ShaderResourceView* g_depthSRV = nullptr;
static int depthTexW = 0;
static int depthTexH = 0;
static float depth_scale = 0.5f;

#ifdef USE_CUDA
static depth_anything::DepthAnythingTrt g_depthModel;
#endif

static void uploadDepthFrame(const cv::Mat& bgr)
{
    if (bgr.empty())
        return;

    if (!g_depthTex || bgr.cols != depthTexW || bgr.rows != depthTexH)
    {
        SAFE_RELEASE(g_depthTex);
        SAFE_RELEASE(g_depthSRV);

        depthTexW = bgr.cols;
        depthTexH = bgr.rows;

        D3D11_TEXTURE2D_DESC td = {};
        td.Width = depthTexW;
        td.Height = depthTexH;
        td.MipLevels = td.ArraySize = 1;
        td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DYNAMIC;
        td.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        td.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

        g_pd3dDevice->CreateTexture2D(&td, nullptr, &g_depthTex);

        D3D11_SHADER_RESOURCE_VIEW_DESC sd = {};
        sd.Format = td.Format;
        sd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        sd.Texture2D.MipLevels = 1;
        g_pd3dDevice->CreateShaderResourceView(g_depthTex, &sd, &g_depthSRV);
    }

    static cv::Mat rgba;
    cv::cvtColor(bgr, rgba, cv::COLOR_BGR2RGBA);

    D3D11_MAPPED_SUBRESOURCE ms;
    if (SUCCEEDED(g_pd3dDeviceContext->Map(g_depthTex, 0, D3D11_MAP_WRITE_DISCARD, 0, &ms)))
    {
        for (int y = 0; y < depthTexH; ++y)
        {
            memcpy(reinterpret_cast<uint8_t*>(ms.pData) + ms.RowPitch * y, rgba.ptr(y), depthTexW * 4);
        }
        g_pd3dDeviceContext->Unmap(g_depthTex, 0);
    }
}

void draw_depth()
{
#ifndef USE_CUDA
    ImGui::TextUnformatted("Depth requires a CUDA build.");
    return;
#else
    static bool bufferInit = false;
    static char modelPathBuf[260]{};
    static std::string depthStatus = "Depth model not loaded.";
    static std::chrono::steady_clock::time_point lastUpdate = std::chrono::steady_clock::now();
    static float lastInferenceMs = 0.0f;

    if (!bufferInit)
    {
        memset(modelPathBuf, 0, sizeof(modelPathBuf));
        std::string initial = config.depth_model_path;
        if (initial.size() >= sizeof(modelPathBuf))
            initial = initial.substr(0, sizeof(modelPathBuf) - 1);
        memcpy(modelPathBuf, initial.c_str(), initial.size());
        bufferInit = true;
    }

    ImGui::InputText("Depth model path", modelPathBuf, IM_ARRAYSIZE(modelPathBuf));
    ImGui::SameLine();
    if (ImGui::Button("Browse##depth_model_path"))
    {
        char filePath[MAX_PATH] = {};
        OPENFILENAMEA ofn = {};
        ofn.lStructSize = sizeof(ofn);
        ofn.hwndOwner = nullptr;
        ofn.lpstrFile = filePath;
        ofn.nMaxFile = sizeof(filePath);
        ofn.lpstrFilter = "Model Files\0*.engine;*.onnx;*.trt;*.plan\0All Files\0*.*\0";
        ofn.nFilterIndex = 1;
        ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;

        if (GetOpenFileNameA(&ofn))
        {
            strncpy_s(modelPathBuf, filePath, sizeof(modelPathBuf) - 1);
        }
    }
    if (ImGui::Button("Load depth model"))
    {
        config.depth_model_path = modelPathBuf;
        config.saveConfig();

        if (g_depthModel.initialize(config.depth_model_path, gLogger))
        {
            depthStatus = "Depth model loaded.";
        }
        else
        {
            depthStatus = g_depthModel.lastError();
        }
    }

    if (ImGui::SliderInt("Depth FPS", &config.depth_fps, 0, 120))
    {
        config.saveConfig();
    }

    ImGui::Text("Status: %s", depthStatus.c_str());
    if (g_depthModel.ready())
    {
        cv::Mat frameCopy;
        {
            std::lock_guard<std::mutex> lk(frameMutex);
            if (!latestFrame.empty())
                latestFrame.copyTo(frameCopy);
        }

        if (!frameCopy.empty())
        {
            auto now = std::chrono::steady_clock::now();
            bool shouldUpdate = true;
            if (config.depth_fps > 0)
            {
                auto interval = std::chrono::milliseconds(1000 / config.depth_fps);
                shouldUpdate = (now - lastUpdate) >= interval;
            }

            if (shouldUpdate)
            {
                auto start = std::chrono::steady_clock::now();
                cv::Mat depthFrame = g_depthModel.predict(frameCopy);
                auto end = std::chrono::steady_clock::now();
                lastInferenceMs = std::chrono::duration<float, std::milli>(end - start).count();

                if (!depthFrame.empty())
                {
                    uploadDepthFrame(depthFrame);
                    lastUpdate = now;
                }
            }
        }

        if (g_depthSRV)
        {
            ImGui::SliderFloat("Depth scale", &depth_scale, 0.1f, 2.0f, "%.1fx");
            ImGui::Text("Last inference: %.2f ms", lastInferenceMs);
            ImVec2 image_size(depthTexW * depth_scale, depthTexH * depth_scale);
            ImGui::Image(g_depthSRV, image_size);
        }
        else
        {
            ImGui::TextUnformatted("Depth not ready yet.");
        }
    }
#endif
}
