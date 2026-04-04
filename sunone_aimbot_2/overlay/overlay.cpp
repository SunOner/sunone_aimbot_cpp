#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <Windows.h>
#include <tchar.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <d3d11.h>
#include <dxgi.h>
#include <dxgi1_2.h>
#include <dwmapi.h>
#include <dcomp.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <chrono>
#include <cmath>

#include <imgui.h>
#include <imgui_impl_dx11.h>
#include <imgui_impl_win32.h>

#include "overlay.h"
#include "overlay/draw_settings.h"
#include "overlay/config_dirty.h"
#include "include/other_tools.h"
#include "config.h"
#include "keycodes.h"
#include "keyboard_listener.h"

#ifdef USE_CUDA
#include "trt_detector.h"
#endif

#pragma comment(lib, "dwmapi.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "dcomp.lib")
#pragma comment(lib, "d3d11.lib")

#define GET_X_LPARAM(lp) ((int)(short)LOWORD(lp))
#define GET_Y_LPARAM(lp) ((int)(short)HIWORD(lp))

ID3D11Device* g_pd3dDevice = NULL;
ID3D11DeviceContext* g_pd3dDeviceContext = NULL;
IDXGISwapChain1* g_pSwapChain = NULL;
IDCompositionDevice* g_dcompDevice = NULL;
IDCompositionTarget* g_dcompTarget = NULL;
IDCompositionVisual* g_dcompVisual = NULL;
ID3D11RenderTargetView* g_mainRenderTargetView = NULL;
HWND g_hwnd = NULL;

extern Config config;
extern std::mutex configMutex;
extern std::atomic<bool> shouldExit;

extern void draw_crosshair_settings();
extern void RenderActiveCrosshair(int screenWidth, int screenHeight);

bool CreateDeviceD3D(HWND hWnd);
void CleanupDeviceD3D();
void CreateRenderTarget();
void CleanupRenderTarget();

ID3D11BlendState* g_pBlendState = nullptr;

LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

const int BASE_OVERLAY_WIDTH = 720;
const int BASE_OVERLAY_HEIGHT = 500;
static const int MIN_EDITOR_OPACITY = 220;

int overlayWidth = 0;
int overlayHeight = 0;

float g_menuX = 0;
float g_menuY = 0;
float g_menuW = 720;
float g_menuH = 500;
bool g_menuOpen = false;

static bool g_autoResizeEnabled = true;
static ImGuiStyle g_baseStyle{};
static bool g_baseStyleReady = false;
static float g_runtimeUiScale = -1.0f;

std::vector<std::string> availableModels;
std::vector<std::string> key_names;
std::vector<const char*> key_names_cstrs;

ID3D11ShaderResourceView* body_texture = nullptr;

void load_body_texture();
void release_body_texture();
std::vector<std::string> getAvailableModels();

static inline int ClampInt(int v, int lo, int hi)
{
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

static float ComputeRuntimeUiScale(float windowW, float windowH)
{
    const float safeW = (windowW > 1.0f) ? windowW : static_cast<float>(BASE_OVERLAY_WIDTH);
    const float safeH = (windowH > 1.0f) ? windowH : static_cast<float>(BASE_OVERLAY_HEIGHT);
    const float refW = static_cast<float>(BASE_OVERLAY_WIDTH);
    const float refH = static_cast<float>(BASE_OVERLAY_HEIGHT);

    const float wFactor = safeW / refW;
    const float hFactor = safeH / refH;
    float autoFactor = std::sqrt(wFactor * hFactor);
    autoFactor = std::clamp(autoFactor, 0.85f, 1.90f);

    const float userFactor = std::clamp(config.overlay_ui_scale, 0.85f, 1.35f);
    return std::clamp(autoFactor * userFactor, 0.80f, 2.20f);
}

static void ApplyRuntimeUiScale(float windowW, float windowH)
{
    if (!g_baseStyleReady) return;

    ImGuiIO& io = ImGui::GetIO();
    const float targetScale = ComputeRuntimeUiScale(windowW, windowH);
    if (std::fabs(targetScale - g_runtimeUiScale) > 0.01f)
    {
        ImGuiStyle& style = ImGui::GetStyle();
        style = g_baseStyle;
        style.ScaleAllSizes(targetScale);
        g_runtimeUiScale = targetScale;
    }
    io.FontGlobalScale = targetScale;
}

void Overlay_SetOpacity(int opacity255)
{
    if (!g_hwnd) return;

    opacity255 = ClampInt(opacity255, MIN_EDITOR_OPACITY, 255);

    LONG exStyle = GetWindowLong(g_hwnd, GWL_EXSTYLE);
    if ((exStyle & WS_EX_LAYERED) == 0)
        SetWindowLong(g_hwnd, GWL_EXSTYLE, exStyle | WS_EX_LAYERED);

    SetLayeredWindowAttributes(g_hwnd, RGB(0, 0, 0), 0, LWA_COLORKEY);
}

static void Overlay_SetDisplayAffinity(HWND hwnd, bool excludeFromCapture)
{
    if (!hwnd) return;

    const DWORD wanted = excludeFromCapture ? WDA_EXCLUDEFROMCAPTURE : WDA_NONE;
    if (SetWindowDisplayAffinity(hwnd, wanted)) return;

    if (excludeFromCapture)
    {
        if (!SetWindowDisplayAffinity(hwnd, WDA_MONITOR))
        {
            std::cerr << "[OverlayUI] SetWindowDisplayAffinity failed." << std::endl;
        }
    }
}

void Overlay_ApplyCaptureExclusion()
{
    Overlay_SetDisplayAffinity(g_hwnd, config.overlay_exclude_from_capture);
}

static inline ImVec4 RGBA(int r, int g, int b, int a = 255) {
    return ImVec4(r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f);
}

static void ApplyTheme_RoseDark()
{
    ImGuiStyle& style = ImGui::GetStyle();
    style.Alpha = 1.0f;

    style.WindowRounding = 0.0f;
    style.ChildRounding = 0.0f;
    style.PopupRounding = 0.0f;
    style.FrameRounding = 0.0f;
    style.TabRounding = 0.0f;
    style.ScrollbarRounding = 0.0f;
    style.GrabRounding = 0.0f;

    style.WindowBorderSize = 1.0f;
    style.ChildBorderSize = 1.0f;
    style.FrameBorderSize = 1.0f;
    style.PopupBorderSize = 1.0f;
    style.TabBorderSize = 1.0f;

    style.WindowPadding = ImVec2(9.0f, 8.0f);
    style.FramePadding = ImVec2(6.0f, 3.0f);
    style.ItemSpacing = ImVec2(7.0f, 5.0f);
    style.ItemInnerSpacing = ImVec2(6.0f, 4.0f);
    style.CellPadding = ImVec2(6.0f, 5.0f);
    style.ScrollbarSize = 10.0f;
    style.GrabMinSize = 10.0f;
    style.IndentSpacing = 12.0f;

    ImVec4* c = style.Colors;

    const ImVec4 bg0 = RGBA(4, 4, 4, 250);
    const ImVec4 bg1 = RGBA(10, 10, 10, 250);
    const ImVec4 bg2 = RGBA(16, 16, 16, 245);
    const ImVec4 stroke = RGBA(255, 255, 255, 56);
    const ImVec4 strokeHi = RGBA(255, 255, 255, 92);

    const ImVec4 text = RGBA(232, 237, 245, 255);
    const ImVec4 textDim = RGBA(143, 160, 182, 255);
    const ImVec4 bright = RGBA(245, 245, 245, 255);

    c[ImGuiCol_Text] = text;
    c[ImGuiCol_TextDisabled] = textDim;

    c[ImGuiCol_WindowBg] = RGBA(0, 0, 0, 0);
    c[ImGuiCol_ChildBg] = RGBA(0, 0, 0, 0);
    c[ImGuiCol_PopupBg] = bg1;

    c[ImGuiCol_Border] = stroke;
    c[ImGuiCol_BorderShadow] = RGBA(0, 0, 0, 0);

    c[ImGuiCol_FrameBg] = bg2;
    c[ImGuiCol_FrameBgHovered] = RGBA(24, 24, 24, 250);
    c[ImGuiCol_FrameBgActive] = RGBA(31, 31, 31, 252);

    c[ImGuiCol_TitleBg] = bg1;
    c[ImGuiCol_TitleBgActive] = bg1;
    c[ImGuiCol_TitleBgCollapsed] = bg1;
    c[ImGuiCol_MenuBarBg] = bg0;

    c[ImGuiCol_ScrollbarBg] = RGBA(0, 0, 0, 95);
    c[ImGuiCol_ScrollbarGrab] = RGBA(96, 96, 96, 170);
    c[ImGuiCol_ScrollbarGrabHovered] = RGBA(122, 122, 122, 210);
    c[ImGuiCol_ScrollbarGrabActive] = RGBA(145, 145, 145, 232);

    c[ImGuiCol_CheckMark] = bright;
    c[ImGuiCol_SliderGrab] = RGBA(236, 236, 236, 236);
    c[ImGuiCol_SliderGrabActive] = bright;

    c[ImGuiCol_Button] = RGBA(14, 14, 14, 246);
    c[ImGuiCol_ButtonHovered] = RGBA(20, 20, 20, 250);
    c[ImGuiCol_ButtonActive] = RGBA(28, 28, 28, 252);

    c[ImGuiCol_Header] = RGBA(18, 18, 18, 244);
    c[ImGuiCol_HeaderHovered] = RGBA(24, 24, 24, 250);
    c[ImGuiCol_HeaderActive] = RGBA(32, 32, 32, 252);

    c[ImGuiCol_Separator] = stroke;
    c[ImGuiCol_SeparatorHovered] = strokeHi;
    c[ImGuiCol_SeparatorActive] = RGBA(168, 168, 168, 228);

    c[ImGuiCol_Tab] = RGBA(14, 14, 14, 248);
    c[ImGuiCol_TabHovered] = RGBA(22, 22, 22, 250);
    c[ImGuiCol_TabActive] = RGBA(30, 30, 30, 252);
    c[ImGuiCol_TabUnfocused] = RGBA(12, 12, 12, 240);
    c[ImGuiCol_TabUnfocusedActive] = RGBA(22, 22, 22, 248);

    c[ImGuiCol_ResizeGrip] = RGBA(0, 0, 0, 0);
    c[ImGuiCol_ResizeGripHovered] = RGBA(0, 0, 0, 0);
    c[ImGuiCol_ResizeGripActive] = RGBA(0, 0, 0, 0);

    c[ImGuiCol_PlotLines] = RGBA(216, 216, 216, 255);
    c[ImGuiCol_PlotHistogram] = RGBA(216, 216, 216, 255);

    c[ImGuiCol_TableHeaderBg] = bg1;
    c[ImGuiCol_TableBorderStrong] = stroke;
    c[ImGuiCol_TableBorderLight] = RGBA(0, 0, 0, 0);
    c[ImGuiCol_TableRowBg] = RGBA(0, 0, 0, 0);
    c[ImGuiCol_TableRowBgAlt] = RGBA(255, 255, 255, 6);

    c[ImGuiCol_NavHighlight] = RGBA(255, 255, 255, 110);
    c[ImGuiCol_NavWindowingHighlight] = RGBA(255, 255, 255, 90);
    c[ImGuiCol_NavWindowingDimBg] = RGBA(0, 0, 0, 110);

    c[ImGuiCol_TextSelectedBg] = RGBA(255, 255, 255, 56);
    c[ImGuiCol_DragDropTarget] = RGBA(255, 255, 255, 188);
}

struct OverlayTabItem
{
    const char* label;
    const char* group;
    const char* description;
    void (*draw)();
};

static const OverlayTabItem kOverlayTabs[] = {
    { "Capture",           "Core",    "Frame source and input feed settings.",              draw_capture_settings },
    { "Target",            "Core",    "Target selection and aim point offsets.",            draw_target },
    { "Mouse",             "Core",    "Mouse behavior, input backend and motion profile.",  draw_mouse },
    { "AI",                "Core",    "Model and detector thresholds.",                     draw_ai },
    { "Buttons",           "Control", "Hotkeys for features and runtime actions.",          draw_buttons },
    { "Overlay",           "Control", "Editor appearance and privacy options.",             draw_overlay },
    { "Game Overlay",      "Control", "In-game render visuals and simulation options.",     draw_game_overlay_settings },
    { "Crosshair Overlay", "Control", "Dynamic crosshair configuration and auto-color.",    draw_crosshair_settings },
    { "Stats",             "Monitor", "Performance and timing graphs.",                     draw_stats },
    { "Debug",             "Monitor", "Screenshot bindings and diagnostics.",               draw_debug },
};

static void DrawMainPanelBackground(const ImVec2& pos, const ImVec2& size)
{
    ImDrawList* draw = ImGui::GetWindowDrawList();
    const ImVec2 max(pos.x + size.x, pos.y + size.y);
    draw->AddRectFilled(pos, max, IM_COL32(4, 4, 4, 248), 0.0f);
    draw->AddRect(pos, max, IM_COL32(255, 255, 255, 56), 0.0f, 0, 1.0f);
}

static bool DrawSidebarTabButton(const char* label, bool selected)
{
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    const ImGuiStyle& style = ImGui::GetStyle();
    ImVec2 size = ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetFrameHeight() + style.ItemSpacing.y * 0.15f);
    if (size.x < 1.0f) size.x = 1.0f;

    const std::string id = std::string("##nav_") + label;
    const bool pressed = ImGui::InvisibleButton(id.c_str(), size);
    const bool hovered = ImGui::IsItemHovered();

    ImDrawList* draw = ImGui::GetWindowDrawList();
    const ImVec2 max(pos.x + size.x, pos.y + size.y);

    ImU32 rowBg = IM_COL32(0, 0, 0, 0);
    if (selected)
        rowBg = IM_COL32(60, 60, 60, 190);
    else if (hovered)
        rowBg = IM_COL32(28, 28, 28, 210);

    if ((rowBg >> IM_COL32_A_SHIFT) != 0)
        draw->AddRectFilled(pos, max, rowBg, 0.0f);
    if (selected)
        draw->AddRect(pos, max, IM_COL32(255, 255, 255, 76), 0.0f, 0, 1.0f);

    const float textY = pos.y + (size.y - ImGui::GetTextLineHeight()) * 0.5f;
    const ImU32 textCol = selected ? IM_COL32(245, 245, 245, 255) : (hovered ? IM_COL32(226, 226, 226, 255) : IM_COL32(192, 200, 214, 240));
    draw->AddText(ImVec2(pos.x + style.FramePadding.x + 2.0f, textY), textCol, label);

    return pressed;
}

bool InitializeBlendState()
{
    D3D11_BLEND_DESC blendDesc;
    ZeroMemory(&blendDesc, sizeof(blendDesc));

    blendDesc.AlphaToCoverageEnable = FALSE;
    blendDesc.RenderTarget[0].BlendEnable = TRUE;
    blendDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
    blendDesc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    blendDesc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    blendDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    blendDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
    blendDesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    blendDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

    HRESULT hr = g_pd3dDevice->CreateBlendState(&blendDesc, &g_pBlendState);
    if (FAILED(hr)) return false;

    float blendFactor[4] = { 0.f, 0.f, 0.f, 0.f };
    g_pd3dDeviceContext->OMSetBlendState(g_pBlendState, blendFactor, 0xffffffff);
    return true;
}

bool CreateDeviceD3D(HWND hWnd)
{
    UINT createDeviceFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    D3D_FEATURE_LEVEL featureLevel;
    const D3D_FEATURE_LEVEL featureLevelArray[] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0 };

    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, createDeviceFlags, featureLevelArray, ARRAYSIZE(featureLevelArray), D3D11_SDK_VERSION, &g_pd3dDevice, &featureLevel, &g_pd3dDeviceContext);
    if (FAILED(hr)) return false;

    IDXGIDevice* dxgiDev = nullptr;
    hr = g_pd3dDevice->QueryInterface(IID_PPV_ARGS(&dxgiDev));
    if (FAILED(hr) || !dxgiDev) return false;

    IDXGIAdapter* adapter = nullptr;
    hr = dxgiDev->GetAdapter(&adapter);
    if (FAILED(hr) || !adapter) { dxgiDev->Release(); return false; }

    IDXGIFactory2* factory2 = nullptr;
    hr = adapter->GetParent(IID_PPV_ARGS(&factory2));
    if (FAILED(hr) || !factory2) { adapter->Release(); dxgiDev->Release(); return false; }

    DXGI_SWAP_CHAIN_DESC1 scd = {};
    scd.Width = overlayWidth;
    scd.Height = overlayHeight;
    scd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    scd.SampleDesc.Count = 1;
    scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scd.BufferCount = 2;
    scd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    hr = factory2->CreateSwapChainForHwnd(g_pd3dDevice, hWnd, &scd, nullptr, nullptr, &g_pSwapChain);
    
    factory2->Release(); 
    adapter->Release();
    dxgiDev->Release();

    if (FAILED(hr) || !g_pSwapChain) return false;

    if (!InitializeBlendState()) return false;
    CreateRenderTarget();
    return true;
}

void CreateRenderTarget()
{
    ID3D11Texture2D* pBackBuffer = NULL;
    g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&pBackBuffer));
    g_pd3dDevice->CreateRenderTargetView(pBackBuffer, NULL, &g_mainRenderTargetView);
    pBackBuffer->Release();
}

void CleanupRenderTarget()
{
    if (g_mainRenderTargetView) { g_mainRenderTargetView->Release(); g_mainRenderTargetView = NULL; }
}

void CleanupDeviceD3D()
{
    CleanupRenderTarget();
    if (g_pSwapChain) { g_pSwapChain->Release(); g_pSwapChain = NULL; }
    if (g_pd3dDeviceContext) { g_pd3dDeviceContext->Release(); g_pd3dDeviceContext = NULL; }
    if (g_pd3dDevice) { g_pd3dDevice->Release(); g_pd3dDevice = NULL; }
    if (g_pBlendState) { g_pBlendState->Release(); g_pBlendState = nullptr; }
}

LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam))
        return true;

    switch (msg)
    {
    case WM_NCHITTEST:
    {
        if (!g_menuOpen) return HTTRANSPARENT; 

        POINT pt = { GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam) };
        ::ScreenToClient(hWnd, &pt);

        if (pt.x >= g_menuX && pt.x <= g_menuX + g_menuW &&
            pt.y >= g_menuY && pt.y <= g_menuY + g_menuH)
        {
            return HTCLIENT;
        }
        
        return HTTRANSPARENT;
    }
    case WM_SIZE:
        if (g_pd3dDevice != NULL && wParam != SIZE_MINIMIZED)
        {
            overlayWidth = (UINT)LOWORD(lParam);
            overlayHeight = (UINT)HIWORD(lParam);

            CleanupRenderTarget();
            g_pSwapChain->ResizeBuffers(0, overlayWidth, overlayHeight, DXGI_FORMAT_UNKNOWN, 0);
            CreateRenderTarget();
        }
        return 0;

    case WM_DESTROY:
        shouldExit = true;
        ::PostQuitMessage(0);
        return 0;

    default:
        return ::DefWindowProc(hWnd, msg, wParam, lParam);
    }
}

void SetupImGui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.FontGlobalScale = 1.0f;
    ImFontConfig fontConfig{};
    fontConfig.OversampleH = 3;
    fontConfig.OversampleV = 2;
    fontConfig.PixelSnapH = true;
    
    io.IniFilename = "sunone_ui_layout.ini"; 
    io.LogFilename = nullptr;

    if (!io.Fonts->AddFontFromFileTTF("C:\\Windows\\Fonts\\segoeui.ttf", 14.0f, &fontConfig)) {
        io.Fonts->AddFontDefault();
    }

    ImGui_ImplWin32_Init(g_hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    ApplyTheme_RoseDark();
    g_baseStyle = ImGui::GetStyle();
    g_baseStyleReady = true;
    g_runtimeUiScale = -1.0f;
    load_body_texture();
}

bool CreateOverlayWindow()
{
    overlayWidth = GetSystemMetrics(SM_CXSCREEN);
    overlayHeight = GetSystemMetrics(SM_CYSCREEN);

    WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, WndProc, 0L, 0L, GetModuleHandle(NULL), NULL, NULL, NULL, NULL, _T("Chrome"), NULL };
    ::RegisterClassEx(&wc);

    const DWORD exStyle = WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_LAYERED;
    const DWORD style = WS_POPUP;

    g_hwnd = ::CreateWindowEx(
        exStyle | WS_EX_TRANSPARENT, 
        wc.lpszClassName, _T("Chrome"),
        style,
        0, 0, overlayWidth, overlayHeight,
        NULL, NULL, wc.hInstance, NULL);

    if (g_hwnd == NULL) return false;

    Overlay_SetOpacity(255);

    if (!CreateDeviceD3D(g_hwnd))
    {
        CleanupDeviceD3D();
        ::UnregisterClass(wc.lpszClassName, wc.hInstance);
        return false;
    }

    SetWindowPos(g_hwnd, HWND_TOPMOST, 0, 0, overlayWidth, overlayHeight, SWP_SHOWWINDOW);
    Overlay_ApplyCaptureExclusion();

    return true;
}

void OverlayThread()
{
    if (!CreateOverlayWindow())
    {
        std::cout << "[Overlay] Can't create overlay window!" << std::endl;
        return;
    }

    SetupImGui();

    bool show_overlay = false;
    int frames_to_clear = 0;

    for (const auto& pair : KeyCodes::key_code_map) key_names.push_back(pair.first);
    std::sort(key_names.begin(), key_names.end());
    key_names_cstrs.reserve(key_names.size());
    for (const auto& name : key_names) key_names_cstrs.push_back(name.c_str());

    availableModels = getAvailableModels();

    MSG msg;
    ZeroMemory(&msg, sizeof(msg));
    bool lastExcludeFromCapture = config.overlay_exclude_from_capture;
    Overlay_SetDisplayAffinity(g_hwnd, lastExcludeFromCapture);

    using clock = std::chrono::high_resolution_clock;
    auto last_render_time = clock::now();

    while (!shouldExit)
    {
        while (::PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE))
        {
            ::TranslateMessage(&msg);
            ::DispatchMessage(&msg);
            if (msg.message == WM_QUIT)
            {
                shouldExit = true;
                break;
            }
        }
        if (shouldExit) break;

        if (lastExcludeFromCapture != config.overlay_exclude_from_capture)
        {
            lastExcludeFromCapture = config.overlay_exclude_from_capture;
            Overlay_SetDisplayAffinity(g_hwnd, lastExcludeFromCapture);
        }

        if (isAnyKeyPressed(config.button_open_overlay) & 0x1)
        {
            show_overlay = !show_overlay;
            g_menuOpen = show_overlay; 
            
            if (!show_overlay) {
                frames_to_clear = 2;
            }

            LONG exStyle = GetWindowLong(g_hwnd, GWL_EXSTYLE);
            if (show_overlay)
            {
                SetWindowLong(g_hwnd, GWL_EXSTYLE, exStyle & ~WS_EX_TRANSPARENT);
                SetForegroundWindow(g_hwnd);
            }
            else
            {
                SetWindowLong(g_hwnd, GWL_EXSTYLE, exStyle | WS_EX_TRANSPARENT);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        bool should_render = false;
        bool use_vsync = false; 

        if (show_overlay || frames_to_clear > 0)
        {
            should_render = true;
            use_vsync = true;
        }
        else if (config.show_crosshair)
        {
            auto now = clock::now();
            double fps_target = config.crosshair_smart_color ? 60.0 : 1.0;
            double frame_delay_ms = 1000.0 / fps_target;

            std::chrono::duration<double, std::milli> elapsed = now - last_render_time;
            if (elapsed.count() > frame_delay_ms)
            {
                should_render = true;
                last_render_time = now;
            }
        }

        if (!should_render)
        {
            if (show_overlay) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            else if (config.show_crosshair && config.crosshair_smart_color) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            else if (config.show_crosshair && !config.crosshair_smart_color) std::this_thread::sleep_for(std::chrono::milliseconds(32));
            else std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();

        if (config.show_crosshair)
        {
            RenderActiveCrosshair(overlayWidth, overlayHeight);
        }

        if (show_overlay)
        {
            ApplyRuntimeUiScale(BASE_OVERLAY_WIDTH, BASE_OVERLAY_HEIGHT);
            const float sidebarWidth = std::clamp((float)BASE_OVERLAY_WIDTH * 0.23f, (float)BASE_OVERLAY_WIDTH * 0.18f, (float)BASE_OVERLAY_WIDTH * 0.30f);

            ImGui::SetNextWindowPos(ImVec2((overlayWidth - BASE_OVERLAY_WIDTH) / 2.0f, (overlayHeight - BASE_OVERLAY_HEIGHT) / 2.0f), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(BASE_OVERLAY_WIDTH, BASE_OVERLAY_HEIGHT), ImGuiCond_FirstUseEver);
            ImGui::PushStyleColor(ImGuiCol_WindowBg, IM_COL32(0, 0, 0, 0));
            
            ImGui::Begin("##editor_root", nullptr,
                ImGuiWindowFlags_NoDecoration |
                ImGuiWindowFlags_NoBringToFrontOnFocus |
                ImGuiWindowFlags_NoScrollbar |
                ImGuiWindowFlags_NoScrollWithMouse);
            ImGui::PopStyleColor();

            ImVec2 windowPos = ImGui::GetWindowPos();
            g_menuX = windowPos.x;
            g_menuY = windowPos.y;
            g_menuW = ImGui::GetWindowSize().x;
            g_menuH = ImGui::GetWindowSize().y;

            DrawMainPanelBackground(ImGui::GetWindowPos(), ImGui::GetWindowSize());

            {
                std::lock_guard<std::mutex> lock(configMutex);

                static int activeTab = 0;
                const int tabCount = (int)(sizeof(kOverlayTabs) / sizeof(kOverlayTabs[0]));
                if (activeTab < 0 || activeTab >= tabCount)
                    activeTab = 0;

                ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 0.0f);
                ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32(11, 11, 11, 245));
                ImGui::PushStyleColor(ImGuiCol_Border, IM_COL32(255, 255, 255, 56));
                ImGui::BeginChild("##options_nav", ImVec2(sidebarWidth, 0.0f), true,
                    ImGuiWindowFlags_AlwaysUseWindowPadding | ImGuiWindowFlags_AlwaysVerticalScrollbar);

                ImGui::TextUnformatted("SunOne Overlay");
                ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(143, 160, 182, 255));
                ImGui::TextUnformatted("HOME to open/close");
                ImGui::PopStyleColor();
                ImGui::Dummy(ImVec2(0.0f, 2.0f));

                const char* lastGroup = nullptr;
                for (int i = 0; i < tabCount; ++i)
                {
                    const char* group = kOverlayTabs[i].group;
                    if (!lastGroup || std::strcmp(lastGroup, group) != 0)
                    {
                        if (lastGroup)
                            ImGui::Dummy(ImVec2(0.0f, 2.0f));
                        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(165, 180, 199, 228));
                        ImGui::TextUnformatted(group);
                        ImGui::PopStyleColor();
                    }
                    if (DrawSidebarTabButton(kOverlayTabs[i].label, activeTab == i))
                        activeTab = i;
                    lastGroup = group;
                }
                ImGui::EndChild();
                ImGui::PopStyleColor(2);
                ImGui::PopStyleVar();

                ImGui::SameLine(0.0f, 6.0f);

                ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 0.0f);
                ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32(12, 12, 12, 245));
                ImGui::PushStyleColor(ImGuiCol_Border, IM_COL32(255, 255, 255, 56));
                ImGui::BeginChild("##options_content", ImVec2(0.0f, 0.0f), true,
                    ImGuiWindowFlags_AlwaysUseWindowPadding | ImGuiWindowFlags_AlwaysVerticalScrollbar);

                ImGui::TextUnformatted(kOverlayTabs[activeTab].label);
                ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(143, 160, 182, 255));
                ImGui::TextWrapped("%s", kOverlayTabs[activeTab].description);
                ImGui::PopStyleColor();
                ImGui::Separator();

                kOverlayTabs[activeTab].draw();

                ImGui::EndChild();
                ImGui::PopStyleColor(2);
                ImGui::PopStyleVar();

                OverlayConfig_TrySave();
            }

            ImGui::End();
        }

        ImGui::Render();

        const float clear_color_with_alpha[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, NULL);
        g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTargetView, clear_color_with_alpha);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

        if (use_vsync)
        {
            HRESULT result = g_pSwapChain->Present(1, 0);
            if (result == DXGI_STATUS_OCCLUDED || result == DXGI_ERROR_ACCESS_LOST)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        else
        {
            HRESULT result = g_pSwapChain->Present(0, 0);
            if (result == DXGI_STATUS_OCCLUDED || result == DXGI_ERROR_ACCESS_LOST)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        if (frames_to_clear > 0) {
            frames_to_clear--;
        }
    }

    release_body_texture();

    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    CleanupDeviceD3D();
    ::DestroyWindow(g_hwnd);
    ::UnregisterClass(_T("Chrome"), GetModuleHandle(NULL));
}

int APIENTRY _tWinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPTSTR lpCmdLine, _In_ int nCmdShow)
{
    std::thread overlay(OverlayThread);
    overlay.join();
    return 0;
}
