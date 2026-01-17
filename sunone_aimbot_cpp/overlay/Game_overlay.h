#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <atomic>
#include <memory>

struct OverlayRect { float x, y, w, h; };
struct OverlayLine { float x1, y1, x2, y2; };
struct OverlayCircle { float cx, cy, r; };

using OverlayColor = uint32_t; // ARGB: 0xAARRGGBB
inline constexpr OverlayColor ARGB(uint8_t a, uint8_t r, uint8_t g, uint8_t b) {
    return (OverlayColor(a) << 24) | (OverlayColor(r) << 16) | (OverlayColor(g) << 8) | OverlayColor(b);
}

class Game_overlay
{
public:
    bool Start();
    void Stop();
    bool IsRunning() const;

    void SetVisible(bool visible);
    bool GetVisible() const;

    void BeginFrame();
    void EndFrame();

    void AddLine(const OverlayLine& line, OverlayColor color, float thickness = 1.0f);
    void AddRect(const OverlayRect& rc, OverlayColor color, float thickness = 1.0f);
    void FillRect(const OverlayRect& rc, OverlayColor color);
    void AddCircle(const OverlayCircle& c, OverlayColor color, float thickness = 1.0f);
    void FillCircle(const OverlayCircle& c, OverlayColor color);
    void AddText(float x, float y, const std::wstring& text, float sizePx,
        OverlayColor color, const std::wstring& font = L"Segoe UI");

    int  LoadImageFromFile(const std::wstring& path);
    void UnloadImage(int imageId);
    void DrawImage(int imageId, float x, float y, float w, float h, float opacity = 1.0f);
    int  UpdateImageFromBGRA(const void* data, int width, int height, int strideBytes, int imageId = 0);

    void UseVirtualScreen();
    void SetWindowBounds(int x, int y, int w, int h);

    void SetMaxFPS(unsigned fps);

    Game_overlay();
    ~Game_overlay();

private:
    Game_overlay(const Game_overlay&) = delete;
    Game_overlay& operator=(const Game_overlay&) = delete;

    struct Impl;
    std::unique_ptr<Impl> impl_;
};
