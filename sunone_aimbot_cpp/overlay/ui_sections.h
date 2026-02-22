#ifndef OVERLAY_UI_SECTIONS_H
#define OVERLAY_UI_SECTIONS_H

#include "imgui/imgui.h"

namespace OverlayUI
{
inline void DrawBodyFrame(const ImVec2& min, const ImVec2& max) noexcept
{
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    drawList->AddRectFilled(min, max, IM_COL32(12, 16, 24, 28), 9.0f);
    drawList->AddRect(min, max, IM_COL32(98, 122, 158, 245), 9.0f, 0, 1.35f);
    drawList->AddLine(
        ImVec2(min.x + 1.0f, min.y + 1.0f),
        ImVec2(max.x - 1.0f, min.y + 1.0f),
        IM_COL32(160, 186, 220, 120),
        1.0f
    );
    drawList->AddLine(
        ImVec2(min.x + 2.0f, min.y + 4.0f),
        ImVec2(min.x + 2.0f, max.y - 4.0f),
        IM_COL32(208, 158, 168, 255),
        2.0f
    );
}

inline void BeginBodyGroup() noexcept
{
    ImGui::BeginGroup();
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(10.0f, 10.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, IM_COL32(22, 28, 40, 250));
    ImGui::PushStyleColor(ImGuiCol_Border, IM_COL32(92, 114, 146, 220));
    ImGui::Dummy(ImVec2(0.0f, 5.0f));
    ImGui::Indent(13.0f);
}

inline void EndBodyGroup() noexcept
{
    ImGui::Unindent(13.0f);
    ImGui::Dummy(ImVec2(0.0f, 5.0f));
    ImGui::PopStyleColor(2);
    ImGui::PopStyleVar(2);
    ImGui::EndGroup();

    const ImVec2 groupMin = ImGui::GetItemRectMin();
    const ImVec2 groupMax = ImGui::GetItemRectMax();
    const ImVec2 boxMin(groupMin.x - 9.0f, groupMin.y - 7.0f);
    const ImVec2 boxMax(groupMax.x + 9.0f, groupMax.y + 7.0f);
    DrawBodyFrame(boxMin, boxMax);
}

inline bool BeginSection(const char* label, const char* id = nullptr, bool defaultOpen = true) noexcept
{
    ImGui::PushID(id ? id : label);

    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.2f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 8.0f);
    ImGui::PushStyleColor(ImGuiCol_Header, IM_COL32(52, 72, 102, 245));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, IM_COL32(66, 90, 128, 250));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, IM_COL32(78, 104, 145, 255));
    ImGui::PushStyleColor(ImGuiCol_Border, IM_COL32(128, 154, 196, 230));
    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(236, 241, 250, 255));

    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_SpanAvailWidth;
    if (defaultOpen)
        flags |= ImGuiTreeNodeFlags_DefaultOpen;

    const bool open = ImGui::TreeNodeEx("##section_header", flags, "%s", label);

    ImGui::PopStyleColor(5);
    ImGui::PopStyleVar(2);

    if (!open)
    {
        ImGui::PopID();
        ImGui::Dummy(ImVec2(0.0f, 6.0f));
        return false;
    }

    BeginBodyGroup();
    return true;
}

inline void EndSection() noexcept
{
    EndBodyGroup();
    ImGui::TreePop();
    ImGui::PopID();
    ImGui::Dummy(ImVec2(0.0f, 7.0f));
}

inline bool BeginSubsection(const char* label, bool defaultOpen = true) noexcept
{
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f);
    ImGui::PushStyleColor(ImGuiCol_Header, IM_COL32(34, 50, 74, 235));
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, IM_COL32(48, 68, 97, 245));
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, IM_COL32(58, 82, 116, 250));
    ImGui::PushStyleColor(ImGuiCol_Border, IM_COL32(94, 120, 156, 215));
    ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(220, 228, 240, 245));

    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_SpanAvailWidth;
    if (defaultOpen)
        flags |= ImGuiTreeNodeFlags_DefaultOpen;

    const bool open = ImGui::TreeNodeEx(label, flags);

    ImGui::PopStyleColor(5);
    ImGui::PopStyleVar(2);

    if (!open)
        return false;

    ImGui::PushID(label);
    BeginBodyGroup();
    return true;
}

inline void EndSubsection() noexcept
{
    EndBodyGroup();
    ImGui::TreePop();
    ImGui::PopID();
    ImGui::Dummy(ImVec2(0.0f, 6.0f));
}
}

#endif // OVERLAY_UI_SECTIONS_H
