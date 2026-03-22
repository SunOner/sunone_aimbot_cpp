#include <imgui.h>
#include <vector>
#include <string>
#include <filesystem>
#include <d3d11.h>
#include <mutex>
#include <opencv2/opencv.hpp> 
#include "config.h"
#include "overlay/config_dirty.h"

#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h" 

extern Config config;
extern ID3D11Device* g_pd3dDevice; 

cv::Mat latestFrameCpu;
extern std::mutex frameMutex;

struct CrosshairIcon {
    std::string filename;
    ID3D11ShaderResourceView* texture;
    int width, height;
};

static std::vector<CrosshairIcon> crosshair_icons;
static bool crosshairs_loaded = false;
static ID3D11ShaderResourceView* active_crosshair_tex = nullptr;
static ImVec4 currentSmoothColor = ImVec4(0, 1, 1, 1);

#define CONFIG_CX config.crosshair_x
#define CONFIG_CY config.crosshair_y

void RGBtoHSV(float r, float g, float b, float& h, float& s, float& v) {
    ImGui::ColorConvertRGBtoHSV(r, g, b, h, s, v);
}

void HSVtoRGB(float h, float s, float v, float& r, float& g, float& b) {
    ImGui::ColorConvertHSVtoRGB(h, s, v, r, g, b);
}

ImVec4 GetSmartColor() {
    ImVec4 targetColor = currentSmoothColor;

    if (!latestFrameCpu.empty() && frameMutex.try_lock()) {
        int cx = latestFrameCpu.cols / 2;
        int cy = latestFrameCpu.rows / 2;
        int radius = 30 + (int)(config.crosshair_scale * 20.0f); 
        
        if (cx > radius && cx < latestFrameCpu.cols - radius && cy > radius && cy < latestFrameCpu.rows - radius) {
            float rTot = 0, gTot = 0, bTot = 0;
            int count = 0;
            int diag = (int)(radius * 0.707f);
            int offsets[8][2] = {
                {radius, 0}, {-radius, 0}, {0, radius}, {0, -radius},
                {diag, diag}, {-diag, -diag}, {diag, -diag}, {-diag, diag}
            };

            for (int i = 0; i < 8; i++) {
                cv::Vec4b p = latestFrameCpu.at<cv::Vec4b>(cy + offsets[i][1], cx + offsets[i][0]);
                bTot += p[0]; gTot += p[1]; rTot += p[2];
                count++;
            }

            float avgR = (rTot / count) / 255.0f;
            float avgG = (gTot / count) / 255.0f;
            float avgB = (bTot / count) / 255.0f;

            float h, s, v;
            RGBtoHSV(avgR, avgG, avgB, h, s, v);

            if (s < 0.2f) { 
                if (v > 0.6f) {
                    targetColor = ImVec4(0.05f, 0.05f, 0.05f, 1.0f); 
                } else {
                    targetColor = ImVec4(0.0f, 1.0f, 1.0f, 1.0f);
                }
            } else {
                float targetR = 1.0f - avgR;
                float targetG = 1.0f - avgG;
                float targetB = 1.0f - avgB;
                float th, ts, tv;

                RGBtoHSV(targetR, targetG, targetB, th, ts, tv);
                HSVtoRGB(th, 1.0f, 1.0f, targetColor.x, targetColor.y, targetColor.z);
                targetColor.w = 1.0f;
            }
        }
        frameMutex.unlock();
    }

    float dt = ImGui::GetIO().DeltaTime; 
    float step = dt * 15.0f; 
    if (step > 1.0f) step = 1.0f; 

    currentSmoothColor.x += (targetColor.x - currentSmoothColor.x) * step;
    currentSmoothColor.y += (targetColor.y - currentSmoothColor.y) * step;
    currentSmoothColor.z += (targetColor.z - currentSmoothColor.z) * step;
    currentSmoothColor.w = 1.0f; 

    return currentSmoothColor;
}

bool LoadTextureFromFile(const char* filename, ID3D11ShaderResourceView** out_srv, int* out_width, int* out_height) {
    int image_width = 0; 
    int image_height = 0;
    unsigned char* image_data = stbi_load(filename, &image_width, &image_height, NULL, 4);
    if (image_data == NULL) return false;
    
    D3D11_TEXTURE2D_DESC desc; 
    ZeroMemory(&desc, sizeof(desc));
    desc.Width = image_width; 
    desc.Height = image_height; 
    desc.MipLevels = 1; 
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM; 
    desc.SampleDesc.Count = 1; 
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE; 
    desc.CPUAccessFlags = 0;
    
    ID3D11Texture2D* pTexture = NULL;
    D3D11_SUBRESOURCE_DATA subResource; 
    subResource.pSysMem = image_data; 
    subResource.SysMemPitch = desc.Width * 4; 
    subResource.SysMemSlicePitch = 0;
    
    g_pd3dDevice->CreateTexture2D(&desc, &subResource, &pTexture);
    
    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc; 
    ZeroMemory(&srvDesc, sizeof(srvDesc));
    srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM; 
    srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D; 
    srvDesc.Texture2D.MipLevels = desc.MipLevels;
    
    g_pd3dDevice->CreateShaderResourceView(pTexture, &srvDesc, out_srv);
    pTexture->Release();
    
    *out_width = image_width; 
    *out_height = image_height;
    stbi_image_free(image_data);
    return true;
}

void RefreshCrosshairs() {
    for (auto& icon : crosshair_icons) {
        if (icon.texture) icon.texture->Release();
    }
    crosshair_icons.clear();
    
    if (!std::filesystem::exists("crosshairs")) {
        std::filesystem::create_directory("crosshairs");
    }
    
    for (const auto& entry : std::filesystem::directory_iterator("crosshairs")) {
        if (!entry.is_regular_file()) continue;
        std::string filename = entry.path().filename().string();
        ID3D11ShaderResourceView* tex = nullptr; 
        int w, h;
        if (LoadTextureFromFile(entry.path().string().c_str(), &tex, &w, &h)) {
            crosshair_icons.push_back({ filename, tex, w, h });
            if (filename == config.current_crosshair) active_crosshair_tex = tex;
        }
    }
    
    if (!active_crosshair_tex && !crosshair_icons.empty()) {
        active_crosshair_tex = crosshair_icons[0].texture;
        config.current_crosshair = crosshair_icons[0].filename;
    }
    crosshairs_loaded = true;
}

void draw_crosshair_settings() {
    if (!crosshairs_loaded) RefreshCrosshairs();

    ImGui::TextColored(ImVec4(1.0f, 0.2f, 0.2f, 1.0f), "CROSSHAIR CONFIGURATION");
    ImGui::Separator();
    ImGui::Spacing();

    if (ImGui::Checkbox("Enable Crosshair Overlay", &config.show_crosshair))
        OverlayConfig_MarkDirty();
    
    ImGui::TextDisabled("(Ctrl+Click sliders to type)");
    
    ImGui::SliderFloat("X Position", &CONFIG_CX, -0.5f, 0.5f, "%.3f");
    if (ImGui::IsItemDeactivatedAfterEdit()) OverlayConfig_MarkDirty();
    
    ImGui::SliderFloat("Y Position", &CONFIG_CY, -0.5f, 0.5f, "%.3f");
    if (ImGui::IsItemDeactivatedAfterEdit()) OverlayConfig_MarkDirty();
    
    ImGui::SliderFloat("Size Scale", &config.crosshair_scale, 0.1f, 5.0f, "%.1f");
    if (ImGui::IsItemDeactivatedAfterEdit()) OverlayConfig_MarkDirty();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    if (ImGui::Checkbox("Smart Contrast (Auto-Color)", &config.crosshair_smart_color))
        OverlayConfig_MarkDirty();

    if (!config.crosshair_smart_color) {
        ImGui::SliderFloat("Hue", &config.crosshair_hue, 0.0f, 1.0f, "");
        if (ImGui::IsItemDeactivatedAfterEdit()) OverlayConfig_MarkDirty();
        
        ImVec2 p = ImGui::GetCursorScreenPos();
        ImDrawList* dl = ImGui::GetWindowDrawList();
        float w = ImGui::GetContentRegionAvail().x;
        
        for(int i = 0; i < 64; i++) {
            float h = i / 64.0f;
            ImColor c; 
            ImGui::ColorConvertHSVtoRGB(h, 1.0f, 1.0f, c.Value.x, c.Value.y, c.Value.z);
            dl->AddRectFilled(ImVec2(p.x + (w * (i / 64.0f)), p.y - 5), ImVec2(p.x + (w * ((i + 1) / 64.0f)), p.y), c);
        }

        ImGui::SliderFloat("Saturation", &config.crosshair_saturation, 0.0f, 1.0f, "%.2f");
        if (ImGui::IsItemDeactivatedAfterEdit()) OverlayConfig_MarkDirty();
    } else {
        ImGui::SameLine();
        ImGui::ColorButton("##SmartCol", currentSmoothColor, ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoPicker, ImVec2(20, 20));
        ImGui::SameLine();
        ImGui::TextDisabled("Auto-Fading");
    }

    ImGui::SliderFloat("Opacity", &config.crosshair_alpha, 0.0f, 1.0f, "%.2f");
    if (ImGui::IsItemDeactivatedAfterEdit()) OverlayConfig_MarkDirty();

    ImGui::Spacing();
    if (ImGui::Button("Reset Defaults")) {
        CONFIG_CX = 0.0f; 
        CONFIG_CY = 0.0f;
        config.crosshair_scale = 1.0f;
        config.crosshair_hue = 0.0f;        
        config.crosshair_saturation = 0.0f; 
        config.crosshair_alpha = 1.0f;
        config.crosshair_smart_color = false;
        OverlayConfig_MarkDirty();
    }

    ImGui::Dummy(ImVec2(0, 10));
    ImGui::Separator();
    
    if (ImGui::Button("Refresh List")) {
        RefreshCrosshairs();
    }

    ImGui::Spacing();

    ImGui::BeginChild("Grid", ImVec2(0, 300), true, ImGuiWindowFlags_AlwaysVerticalScrollbar);
    float window_visible_x2 = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
    ImGuiStyle& style = ImGui::GetStyle();
    int buttons_count = (int)crosshair_icons.size();
    
    ImVec4 tintColor;
    ImGui::ColorConvertHSVtoRGB(config.crosshair_hue, config.crosshair_saturation, 1.0f, tintColor.x, tintColor.y, tintColor.z);
    tintColor.w = 1.0f;

    for (int n = 0; n < buttons_count; n++) {
        ImGui::PushID(n);
        
        bool is_selected = (config.current_crosshair == crosshair_icons[n].filename);
        
        if (is_selected) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(1.0f, 0.2f, 0.2f, 0.4f)); 
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1.0f, 0.2f, 0.2f, 0.6f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(1.0f, 0.2f, 0.2f, 0.8f));
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.1f, 0.1f, 0.1f, 0.5f)); 
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.3f, 0.3f, 0.5f));
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.4f, 0.4f, 0.4f, 0.5f));
        }

        if (ImGui::ImageButton("##btn", crosshair_icons[n].texture, ImVec2(64, 64), ImVec2(0, 0), ImVec2(1, 1), ImVec4(0, 0, 0, 0), tintColor)) {
            config.current_crosshair = crosshair_icons[n].filename;
            active_crosshair_tex = crosshair_icons[n].texture;
            OverlayConfig_MarkDirty();
        }

        ImGui::PopStyleColor(3);

        float last_button_x2 = ImGui::GetItemRectMax().x;
        float next_button_x2 = last_button_x2 + style.ItemSpacing.x + 64.0f;
        if (n + 1 < buttons_count && next_button_x2 < window_visible_x2) {
            ImGui::SameLine();
        }

        ImGui::PopID();
    }
    ImGui::EndChild();
}

void RenderActiveCrosshair(int screenWidth, int screenHeight) {
    if (!crosshairs_loaded) RefreshCrosshairs();
    if (!config.show_crosshair || !active_crosshair_tex) return;

    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    
    float baseW = 64.0f; 
    float baseH = 64.0f;
    
    float screenCenterX = screenWidth / 2.0f;
    float screenCenterY = screenHeight / 2.0f;

    float finalX = screenCenterX + (screenWidth * CONFIG_CX);
    float finalY = screenCenterY - (screenHeight * CONFIG_CY);

    float finalW = baseW * config.crosshair_scale;
    float finalH = baseH * config.crosshair_scale;

    ImVec4 mainColor;
    if (config.crosshair_smart_color) {
        mainColor = GetSmartColor();
    } else {
        ImGui::ColorConvertHSVtoRGB(config.crosshair_hue, config.crosshair_saturation, 1.0f, mainColor.x, mainColor.y, mainColor.z);
        mainColor.w = 1.0f;
    }
    
    mainColor.w *= config.crosshair_alpha;

    dl->AddImage(active_crosshair_tex, 
        ImVec2(finalX - (finalW / 2), finalY - (finalH / 2)), 
        ImVec2(finalX + (finalW / 2), finalY + (finalH / 2)),
        ImVec2(0, 0), ImVec2(1, 1), 
        ImColor(mainColor));
}