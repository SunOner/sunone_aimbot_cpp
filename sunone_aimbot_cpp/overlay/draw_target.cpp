#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include "d3d11.h"
#include "imgui/imgui.h"

#include "overlay.h"
#include "draw_settings.h"
#include "sunone_aimbot_cpp.h"
#include "other_tools.h"
#include "memory_images.h"

ID3D11ShaderResourceView* bodyTexture = nullptr;
ImVec2 bodyImageSize;

void draw_target()
{
    ImGui::Checkbox("Disable Headshot", &config.disable_headshot);

    ImGui::Separator();

    ImGui::SliderFloat("Approximate Body Y Offset", &config.body_y_offset, 0.0f, 1.0f, "%.2f");
    if (bodyTexture)
    {
        ImGui::Image((void*)bodyTexture, bodyImageSize);

        ImVec2 image_pos = ImGui::GetItemRectMin();
        ImVec2 image_size = ImGui::GetItemRectSize();

        ImDrawList* draw_list = ImGui::GetWindowDrawList();

        float normalized_value = (config.body_y_offset - 1.0f) / 1.0f;

        float line_y = image_pos.y + (1.0f + normalized_value) * image_size.y;

        ImVec2 line_start = ImVec2(image_pos.x, line_y);
        ImVec2 line_end = ImVec2(image_pos.x + image_size.x, line_y);

        draw_list->AddLine(line_start, line_end, IM_COL32(255, 0, 0, 255), 2.0f);
    }
    else
    {
        ImGui::Text("Image not found!");
    }
    ImGui::Text("Note: There is a different value for each game, as the sizes of the player models may vary.");
    ImGui::Separator();
    ImGui::Checkbox("Ignore Third Person", &config.ignore_third_person);
    ImGui::Checkbox("Shooting range targets", &config.shooting_range_targets);
    ImGui::Checkbox("Auto Aim", &config.auto_aim);
}

void load_body_texture()
{
    int image_width = 0;
    int image_height = 0;

    std::string body_image = std::string(bodyImageBase64_1) + std::string(bodyImageBase64_2) + std::string(bodyImageBase64_3);

    bool ret = LoadTextureFromMemory(body_image, g_pd3dDevice, &bodyTexture, &image_width, &image_height);
    if (!ret)
    {
        std::cerr << "[Overlay] Can't load image!" << std::endl;
    }
    else
    {
        bodyImageSize = ImVec2((float)image_width, (float)image_height);
    }
}

void release_body_texture()
{
    if (bodyTexture)
    {
        bodyTexture->Release();
        bodyTexture = nullptr;
    }
}