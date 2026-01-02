#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include <tchar.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <d3d11.h>
#include <dxgi.h>
#include <filesystem>

#include <imgui.h>
#include <imgui_impl_dx11.h>
#include <imgui_impl_win32.h>
#include <imgui/imgui_internal.h>

#include "overlay.h"
#include "overlay/draw_settings.h"
#include "config.h"
#include "keycodes.h"
#include "sunone_aimbot_cpp.h"
#include "capture.h"
#include "keyboard_listener.h"
#include "other_tools.h"
#include "virtual_camera.h"
#include "draw_game_overlay.h"
#ifdef USE_CUDA
#include "trt_detector.h"
#endif

ID3D11Device* g_pd3dDevice = NULL;
ID3D11DeviceContext* g_pd3dDeviceContext = NULL;
IDXGISwapChain* g_pSwapChain = NULL;
ID3D11RenderTargetView* g_mainRenderTargetView = NULL;
HWND g_hwnd = NULL;

extern Config config;
extern std::mutex configMutex;
extern std::atomic<bool> shouldExit;

bool CreateDeviceD3D(HWND hWnd);
void CleanupDeviceD3D();
void CreateRenderTarget();
void CleanupRenderTarget();

ID3D11BlendState* g_pBlendState = nullptr;
LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

const int BASE_OVERLAY_WIDTH = 720;
const int BASE_OVERLAY_HEIGHT = 500;
int overlayWidth = 0;
int overlayHeight = 0;

std::vector<std::string> availableModels;
std::vector<std::string> key_names;
std::vector<const char*> key_names_cstrs;

ID3D11ShaderResourceView* body_texture = nullptr;

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
    if (FAILED(hr))
    {
        return false;
    }

    float blendFactor[4] = { 0.f, 0.f, 0.f, 0.f };
    g_pd3dDeviceContext->OMSetBlendState(g_pBlendState, blendFactor, 0xffffffff);

    return true;
}

bool CreateDeviceD3D(HWND hWnd)
{
    DXGI_SWAP_CHAIN_DESC sd;
    ZeroMemory(&sd, sizeof(sd));
    sd.BufferCount = 2;
    sd.BufferDesc.Width = overlayWidth;
    sd.BufferDesc.Height = overlayHeight;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 0;
    sd.BufferDesc.RefreshRate.Denominator = 0;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hWnd;
    sd.SampleDesc.Count = 1;
    sd.SampleDesc.Quality = 0;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;

    UINT createDeviceFlags = 0;

    D3D_FEATURE_LEVEL featureLevel;
    const D3D_FEATURE_LEVEL featureLevelArray[2] =
    {
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_0,
    };

    HRESULT res = D3D11CreateDeviceAndSwapChain(NULL,
        D3D_DRIVER_TYPE_HARDWARE,
        NULL,
        createDeviceFlags,
        featureLevelArray,
        2,
        D3D11_SDK_VERSION,
        &sd,
        &g_pSwapChain,
        &g_pd3dDevice,
        &featureLevel,
        &g_pd3dDeviceContext);
    if (res != S_OK)
        return false;

    if (!InitializeBlendState())
        return false;

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
    case WM_SIZE:
        if (g_pd3dDevice != NULL && wParam != SIZE_MINIMIZED)
        {
            RECT rect;
            GetWindowRect(hWnd, &rect);
            UINT width = rect.right - rect.left;
            UINT height = rect.bottom - rect.top;

            CleanupRenderTarget();
            g_pSwapChain->ResizeBuffers(0, width, height, DXGI_FORMAT_UNKNOWN, 0);
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

static inline ImVec4 RGBA(int r, int g, int b, int a = 255)
{
    return ImVec4(r / 255.0f, g / 255.0f, b / 255.0f, a / 255.0f);
}

static void ApplyTheme_RoseDark()
{
    ImGuiStyle& style = ImGui::GetStyle();

    style.WindowRounding = 10.0f;
    style.ChildRounding = 10.0f;
    style.PopupRounding = 10.0f;
    style.FrameRounding = 8.0f;
    style.TabRounding = 8.0f;
    style.ScrollbarRounding = 10.0f;
    style.GrabRounding = 10.0f;

    style.WindowBorderSize = 1.0f;
    style.FrameBorderSize = 1.0f;
    style.PopupBorderSize = 1.0f;
    style.TabBorderSize = 1.0f;

    style.WindowPadding = ImVec2(14, 12);
    style.FramePadding = ImVec2(10, 8);
    style.ItemSpacing = ImVec2(10, 8);
    style.ItemInnerSpacing = ImVec2(8, 6);
    style.ScrollbarSize = 14.0f;

    ImVec4* c = style.Colors;

    const ImVec4 bg0 = RGBA(12, 12, 13, 245);
    const ImVec4 bg1 = RGBA(18, 18, 20, 245);
    const ImVec4 bg2 = RGBA(24, 24, 28, 245);
    const ImVec4 stroke = RGBA(46, 46, 52, 255);
    const ImVec4 strokeHi = RGBA(64, 64, 74, 255);

    const ImVec4 text = RGBA(230, 230, 235, 255);
    const ImVec4 textDim = RGBA(160, 160, 170, 255);

    const ImVec4 acc = RGBA(168, 125, 135, 255);
    const ImVec4 accHover = RGBA(190, 145, 155, 255);
    const ImVec4 accActive = RGBA(145, 105, 115, 255);

    c[ImGuiCol_Text] = text;
    c[ImGuiCol_TextDisabled] = textDim;

    c[ImGuiCol_WindowBg] = bg0;
    c[ImGuiCol_ChildBg] = RGBA(0, 0, 0, 0);
    c[ImGuiCol_PopupBg] = RGBA(16, 16, 18, 250);

    c[ImGuiCol_Border] = stroke;
    c[ImGuiCol_BorderShadow] = RGBA(0, 0, 0, 0);

    c[ImGuiCol_FrameBg] = bg2;
    c[ImGuiCol_FrameBgHovered] = RGBA(30, 30, 36, 255);
    c[ImGuiCol_FrameBgActive] = RGBA(34, 34, 42, 255);

    c[ImGuiCol_TitleBg] = bg1;
    c[ImGuiCol_TitleBgActive] = bg1;
    c[ImGuiCol_TitleBgCollapsed] = bg1;

    c[ImGuiCol_ScrollbarBg] = RGBA(0, 0, 0, 80);
    c[ImGuiCol_ScrollbarGrab] = RGBA(70, 70, 80, 180);
    c[ImGuiCol_ScrollbarGrabHovered] = RGBA(90, 90, 105, 200);
    c[ImGuiCol_ScrollbarGrabActive] = RGBA(110, 110, 130, 220);

    c[ImGuiCol_CheckMark] = acc;

    c[ImGuiCol_SliderGrab] = acc;
    c[ImGuiCol_SliderGrabActive] = accHover;

    c[ImGuiCol_Button] = RGBA(32, 32, 38, 255);
    c[ImGuiCol_ButtonHovered] = RGBA(42, 42, 50, 255);
    c[ImGuiCol_ButtonActive] = RGBA(48, 48, 58, 255);

    c[ImGuiCol_Header] = RGBA(34, 34, 40, 255);
    c[ImGuiCol_HeaderHovered] = RGBA(44, 44, 54, 255);
    c[ImGuiCol_HeaderActive] = RGBA(52, 52, 64, 255);

    c[ImGuiCol_Separator] = stroke;
    c[ImGuiCol_SeparatorHovered] = strokeHi;
    c[ImGuiCol_SeparatorActive] = acc;

    c[ImGuiCol_Tab] = RGBA(20, 20, 24, 255);
    c[ImGuiCol_TabHovered] = RGBA(40, 40, 48, 255);
    c[ImGuiCol_TabActive] = RGBA(28, 28, 34, 255);
    c[ImGuiCol_TabUnfocused] = RGBA(18, 18, 22, 255);
    c[ImGuiCol_TabUnfocusedActive] = RGBA(24, 24, 30, 255);

    c[ImGuiCol_ResizeGrip] = RGBA(0, 0, 0, 0);
    c[ImGuiCol_ResizeGripHovered] = RGBA(0, 0, 0, 0);
    c[ImGuiCol_ResizeGripActive] = RGBA(0, 0, 0, 0);

    c[ImGuiCol_PlotLines] = acc;
    c[ImGuiCol_PlotHistogram] = acc;

    c[ImGuiCol_TableHeaderBg] = bg1;
    c[ImGuiCol_TableBorderStrong] = stroke;
    c[ImGuiCol_TableBorderLight] = RGBA(0, 0, 0, 0);
    c[ImGuiCol_TableRowBg] = RGBA(0, 0, 0, 0);
    c[ImGuiCol_TableRowBgAlt] = RGBA(255, 255, 255, 6);

    c[ImGuiCol_NavHighlight] = RGBA(255, 255, 255, 40);
    c[ImGuiCol_NavWindowingHighlight] = RGBA(255, 255, 255, 40);
    c[ImGuiCol_NavWindowingDimBg] = RGBA(0, 0, 0, 90);
}

void SetupImGui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.FontGlobalScale = config.overlay_ui_scale;

    ImGui_ImplWin32_Init(g_hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    ApplyTheme_RoseDark();

    load_body_texture();
}

bool CreateOverlayWindow()
{
    overlayWidth = static_cast<int>(BASE_OVERLAY_WIDTH * config.overlay_ui_scale);
    overlayHeight = static_cast<int>(BASE_OVERLAY_HEIGHT * config.overlay_ui_scale);

    WNDCLASSEX wc = {
        sizeof(WNDCLASSEX),
        CS_CLASSDC,
        WndProc,
        0L,
        0L,
        GetModuleHandle(NULL),
        NULL,
        NULL,
        NULL,
        NULL,
        _T("Chrome"),
        NULL
    };
    ::RegisterClassEx(&wc);

    g_hwnd = ::CreateWindowEx(
        WS_EX_TOPMOST | WS_EX_LAYERED,
        wc.lpszClassName, _T("Chrome"),
        WS_POPUP, 0, 0, overlayWidth, overlayHeight,
        NULL, NULL, wc.hInstance, NULL);

    if (g_hwnd == NULL)
        return false;
    
    if (config.overlay_opacity <= 20)
    {
        config.overlay_opacity = 20;
        config.saveConfig("config.ini");
    }

    if (config.overlay_opacity >= 256)
    {
        config.overlay_opacity = 255;
        config.saveConfig("config.ini");
    }

    BYTE opacity = config.overlay_opacity;

    SetLayeredWindowAttributes(g_hwnd, 0, opacity, LWA_ALPHA);

    if (!CreateDeviceD3D(g_hwnd))
    {
        CleanupDeviceD3D();
        ::UnregisterClass(wc.lpszClassName, wc.hInstance);
        return false;
    }

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
    int prev_opacity = config.overlay_opacity;

    for (const auto& pair : KeyCodes::key_code_map)
    {
        key_names.push_back(pair.first);
    }
    std::sort(key_names.begin(), key_names.end());

    key_names_cstrs.reserve(key_names.size());
    for (const auto& name : key_names)
    {
        key_names_cstrs.push_back(name.c_str());
    }
    
    std::vector<std::string> availableModels = getAvailableModels();

    MSG msg;
    ZeroMemory(&msg, sizeof(msg));
    while (!shouldExit)
    {
        while (::PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE))
        {
            ::TranslateMessage(&msg);
            ::DispatchMessage(&msg);
            if (msg.message == WM_QUIT)
            {
                shouldExit = true;
                return;
            }
        }

        if (isAnyKeyPressed(config.button_open_overlay) & 0x1)
        {
            show_overlay = !show_overlay;

            if (show_overlay)
            {
                ShowWindow(g_hwnd, SW_SHOW);
                SetForegroundWindow(g_hwnd);
            }
            else
            {
                ShowWindow(g_hwnd, SW_HIDE);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }

        if (show_overlay)
        {
            ImGui_ImplDX11_NewFrame();
            ImGui_ImplWin32_NewFrame();
            ImGui::NewFrame();

            ImGui::SetNextWindowPos(ImVec2(0, 0));
            ImGui::SetNextWindowSize(ImVec2((float)overlayWidth, (float)overlayHeight));

            ImGui::Begin("Options", &show_overlay, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoTitleBar);
            {
                std::lock_guard<std::mutex> lock(configMutex);

                if (ImGui::BeginTabBar("Options tab bar"))
                {
                    if (ImGui::BeginTabItem("Capture")) { draw_capture_settings(); ImGui::EndTabItem(); }
                    if (ImGui::BeginTabItem("Target")) { draw_target(); ImGui::EndTabItem(); }
                    if (ImGui::BeginTabItem("Mouse")) { draw_mouse(); ImGui::EndTabItem();}
                    if (ImGui::BeginTabItem("AI")) { draw_ai(); ImGui::EndTabItem(); }
                    if (ImGui::BeginTabItem("Buttons")) { draw_buttons(); ImGui::EndTabItem(); }
                    if (ImGui::BeginTabItem("Overlay")) { draw_overlay(); ImGui::EndTabItem(); }
                    if (ImGui::BeginTabItem("Game Overlay")) { draw_game_overlay_settings(); ImGui::EndTabItem(); }
                    if (ImGui::BeginTabItem("Stats")) { draw_stats(); ImGui::EndTabItem(); }
                    if (ImGui::BeginTabItem("Debug")) { draw_debug(); ImGui::EndTabItem(); }

                    if (prev_opacity != config.overlay_opacity)
                    {
                        BYTE opacity = config.overlay_opacity;
                        SetLayeredWindowAttributes(g_hwnd, 0, opacity, LWA_ALPHA);
                        config.saveConfig();
                    }

                ImGui::EndTabBar();
                }
            }

            ImGui::End();
            ImGui::Render();

            const float clear_color_with_alpha[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRenderTargetView, NULL);
            g_pd3dDeviceContext->ClearRenderTargetView(g_mainRenderTargetView, clear_color_with_alpha);
            ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

            HRESULT result = g_pSwapChain->Present(0, 0);

            if (result == DXGI_STATUS_OCCLUDED || result == DXGI_ERROR_ACCESS_LOST)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
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

int APIENTRY _tWinMain(_In_ HINSTANCE hInstance,
    _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPTSTR    lpCmdLine,
    _In_ int       nCmdShow)
{
    std::thread overlay(OverlayThread);
    overlay.join();
    return 0;
}