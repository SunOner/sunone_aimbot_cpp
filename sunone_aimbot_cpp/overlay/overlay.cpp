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
#include "Snowflake.hpp"

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

const int BASE_OVERLAY_WIDTH = 680;
const int BASE_OVERLAY_HEIGHT = 480;
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

void SetupImGui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.FontGlobalScale = config.overlay_ui_scale;

    ImGui_ImplWin32_Init(g_hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    ImGui::StyleColorsDark();

    load_body_texture();
}

bool CreateOverlayWindow()
{
    overlayWidth = static_cast<int>(BASE_OVERLAY_WIDTH * config.overlay_ui_scale);
    overlayHeight = static_cast<int>(BASE_OVERLAY_HEIGHT * config.overlay_ui_scale);

    WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, WndProc, 0L, 0L,
                      GetModuleHandle(NULL), NULL, NULL, NULL, NULL,
                      _T("Edge"), NULL };
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

    // Capture
    std::string prev_capture_method = config.capture_method;
    int prev_detection_resolution = config.detection_resolution;
    int prev_capture_fps = config.capture_fps;
    int prev_monitor_idx = config.monitor_idx;
    bool prev_circle_mask = config.circle_mask;

    bool prev_capture_borders = config.capture_borders;
    bool prev_capture_cursor = config.capture_cursor;

    // Target
    bool prev_disable_headshot = config.disable_headshot;
    float prev_body_y_offset = config.body_y_offset;
    float prev_head_y_offset = config.head_y_offset;
    bool prev_ignore_third_person = config.ignore_third_person;
    bool prev_shooting_range_targets = config.shooting_range_targets;
    bool prev_auto_aim = config.auto_aim;

    // Mouse
    int prev_dpi = config.dpi;
    float prev_sensitivity = config.sensitivity;
    int prev_fovX = config.fovX;
    int prev_fovY = config.fovY;
    float prev_minSpeedMultiplier = config.minSpeedMultiplier;
    float prev_maxSpeedMultiplier = config.maxSpeedMultiplier;
    float prev_predictionInterval = config.predictionInterval;

    //Mouse shooting
    bool prev_auto_shoot = config.auto_shoot;
    float prev_bScope_multiplier = config.bScope_multiplier;

    // AI
    float prev_confidence_threshold = config.confidence_threshold;
    float prev_nms_threshold = config.nms_threshold;
    int prev_max_detections = config.max_detections;

    // Overlay
    int prev_opacity = config.overlay_opacity;

    // Debug
    bool prev_show_window = config.show_window;
    bool prev_show_fps = config.show_fps;
    int prev_window_size = config.window_size;
    int prev_screenshot_delay = config.screenshot_delay;
    bool prev_always_on_top = config.always_on_top;
    bool prev_verbose = config.verbose;

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

    int input_method_index = 0;
    if (config.input_method == "WIN32")
        input_method_index = 0;
    else if (config.input_method == "GHUB")
        input_method_index = 1;
    else if (config.input_method == "ARDUINO")
        input_method_index = 2;
    else
        input_method_index = 0;
    
    std::vector<std::string> availableModels = getAvailableModels();

    std::vector<Snowflake::Snowflake> snow;
    static auto lastTime = std::chrono::high_resolution_clock::now();
    POINT mouse;

    Snowflake::CreateSnowFlakes(
        snow,
        80,
        2.f,
        5.f,
        0,
        0,
        overlayWidth,
        overlayHeight,
        Snowflake::vec3(0.f, 0.005f),
        IM_COL32(255, 255, 255, 255)
    );

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

                if (config.overlay_snow_theme)
                {
                    auto currentTime = std::chrono::high_resolution_clock::now();
                    float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
                    lastTime = currentTime;

                    Snowflake::Update(snow, Snowflake::vec3(0, 0), deltaTime);
                }

                if (ImGui::BeginTabBar("Options tab bar"))
                {
                    if (ImGui::BeginTabItem("Capture"))
                    {
                        draw_capture_settings();

                        ImGui::EndTabItem();
                    }

                    if (ImGui::BeginTabItem("Target"))
                    {
                        draw_target();

                        ImGui::EndTabItem();
                    }

                    if (ImGui::BeginTabItem("Mouse"))
                    {
                        draw_mouse();

                        ImGui::EndTabItem();
                    }

                    if (ImGui::BeginTabItem("AI"))
                    {
                        draw_ai();

                        ImGui::EndTabItem();
                    }

                    if (ImGui::BeginTabItem("Optical flow"))
                    {
                        draw_optical_flow();

                        ImGui::EndTabItem();
                    }

                    if (ImGui::BeginTabItem("Buttons"))
                    {
                        draw_buttons();

                        ImGui::EndTabItem();
                    }

                    if (ImGui::BeginTabItem("Overlay"))
                    {
                        draw_overlay();

                        ImGui::EndTabItem();
                    }

                    if (ImGui::BeginTabItem("Debug"))
                    {
                        draw_debug();

                        ImGui::EndTabItem();
                    }

                    // ******************************************* APPLY VARS *******************************************
                    // DETECTION RESOLUTION
                    if (prev_detection_resolution != config.detection_resolution)
                    {
                        prev_detection_resolution = config.detection_resolution;
                        detection_resolution_changed.store(true);
                        detector_model_changed.store(true); // reboot vars for visuals

                        // apply new detection_resolution
                        globalMouseThread->updateConfig(
                            config.detection_resolution,
                            config.dpi,
                            config.sensitivity,
                            config.fovX,
                            config.fovY,
                            config.minSpeedMultiplier,
                            config.maxSpeedMultiplier,
                            config.predictionInterval,
                            config.auto_shoot,
                            config.bScope_multiplier);
                        config.saveConfig();
                    }

                    // CAPTURE CURSOR
                    if (prev_capture_cursor != config.capture_cursor && config.capture_method == "winrt")
                    {
                        capture_cursor_changed.store(true);
                        prev_capture_cursor = config.capture_cursor;
                        config.saveConfig();
                    }

                    // CAPTURE BORDERS
                    if (prev_capture_borders != config.capture_borders && config.capture_method == "winrt")
                    {
                        capture_borders_changed.store(true);
                        prev_capture_borders = config.capture_borders;
                        config.saveConfig();
                    }

                    // CAPTURE_FPS
                    if (prev_capture_fps != config.capture_fps ||
                        prev_monitor_idx != config.monitor_idx)
                    {
                        capture_fps_changed.store(true);
                        prev_monitor_idx = config.monitor_idx;
                        prev_capture_fps = config.capture_fps;
                        config.saveConfig();
                    }

                    // DISABLE_HEADSHOT / BODY_Y_OFFSET / HEAD_Y_OFFSET / IGNORE_THIRD_PERSON / SHOOTING_RANGE_TARGETS / AUTO_AIM
                    if (prev_disable_headshot != config.disable_headshot ||
                        prev_body_y_offset != config.body_y_offset ||
                        prev_head_y_offset != config.head_y_offset ||
                        prev_ignore_third_person != config.ignore_third_person ||
                        prev_shooting_range_targets != config.shooting_range_targets ||
                        prev_auto_aim != config.auto_aim)
                    {
                        prev_disable_headshot = config.disable_headshot;
                        prev_body_y_offset = config.body_y_offset;
                        prev_head_y_offset = config.head_y_offset;
                        prev_ignore_third_person = config.ignore_third_person;
                        prev_shooting_range_targets = config.shooting_range_targets;
                        prev_auto_aim = config.auto_aim;
                        config.saveConfig();
                    }

                    // DPI / SENSITIVITY / FOVX / FOVY / MINSPEEDMULTIPLIER / MAXSPEEDMULTIPLIER / PREDICTIONINTERVAL
                    if (prev_dpi != config.dpi ||
                        prev_sensitivity != config.sensitivity ||
                        prev_fovX != config.fovX ||
                        prev_fovY != config.fovY ||
                        prev_minSpeedMultiplier != config.minSpeedMultiplier ||
                        prev_maxSpeedMultiplier != config.maxSpeedMultiplier ||
                        prev_predictionInterval != config.predictionInterval)
                    {
                        prev_dpi = config.dpi;
                        prev_sensitivity = config.sensitivity;
                        prev_fovX = config.fovX;
                        prev_fovY = config.fovY;
                        prev_minSpeedMultiplier = config.minSpeedMultiplier;
                        prev_maxSpeedMultiplier = config.maxSpeedMultiplier;
                        prev_predictionInterval = config.predictionInterval;

                        globalMouseThread->updateConfig(
                            config.detection_resolution,
                            config.dpi,
                            config.sensitivity,
                            config.fovX,
                            config.fovY,
                            config.minSpeedMultiplier,
                            config.maxSpeedMultiplier,
                            config.predictionInterval,
                            config.auto_shoot,
                            config.bScope_multiplier);

                        config.saveConfig();
                    }

                    // AUTO_SHOOT / BSCOPE_MULTIPLIER / AUTO_SHOOT_REPLAY
                    if (prev_auto_shoot != config.auto_shoot ||
                        prev_bScope_multiplier != config.bScope_multiplier)
                    {
                        prev_auto_shoot = config.auto_shoot;
                        prev_bScope_multiplier = config.bScope_multiplier;

                        globalMouseThread->updateConfig(
                            config.detection_resolution,
                            config.dpi,
                            config.sensitivity,
                            config.fovX,
                            config.fovY,
                            config.minSpeedMultiplier,
                            config.maxSpeedMultiplier,
                            config.predictionInterval,
                            config.auto_shoot,
                            config.bScope_multiplier);

                        config.saveConfig();
                    }

                    // OVERLAY OPACITY
                    if (prev_opacity != config.overlay_opacity)
                    {
                        BYTE opacity = config.overlay_opacity;
                        SetLayeredWindowAttributes(g_hwnd, 0, opacity, LWA_ALPHA);
                        config.saveConfig();
                    }

                    // CONFIDENCE THERSHOLD / NMS THRESHOLD / MAX DETECTIONS
                    if (prev_confidence_threshold != config.confidence_threshold ||
                        prev_nms_threshold != config.nms_threshold ||
                        prev_max_detections != config.max_detections)
                    {
                        prev_nms_threshold = config.nms_threshold;
                        prev_confidence_threshold = config.confidence_threshold;
                        prev_max_detections = config.max_detections;
                        config.saveConfig();
                    }

                    // SHOW WINDOW / ALWAYS_ON_TOP
                    if (prev_show_window != config.show_window ||
                        prev_always_on_top != config.always_on_top)
                    {
                        prev_always_on_top = config.always_on_top;
                        show_window_changed.store(true);
                        prev_show_window = config.show_window;
                        config.saveConfig();
                    }
                    
                    // SHOW_FPS / WINDOW_SIZE / SCREENSHOT_DELAY / VERBOSE
                    if (prev_show_fps != config.show_fps ||
                        prev_window_size != config.window_size ||
                        prev_screenshot_delay != config.screenshot_delay ||
                        prev_verbose != config.verbose)
                    {
                        prev_show_fps = config.show_fps;
                        prev_window_size = config.window_size;
                        prev_screenshot_delay = config.screenshot_delay;
                        prev_verbose = config.verbose;
                        config.saveConfig();
                    }

                ImGui::EndTabBar();
                }
            }

            ImGui::Separator();
            ImGui::TextColored(ImVec4(255, 255, 255, 100), "Do not test shooting and aiming with the overlay and debug window is open.");

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
    ::UnregisterClass(_T("Edge"), GetModuleHandle(NULL));
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