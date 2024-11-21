#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

// for ai models search
#include <string>
#include <iostream>
#include <filesystem>
#include <algorithm>

#include <tchar.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <d3d11.h>
#include <dxgi.h>

#include "imgui.h"
#include "imgui_impl_dx11.h"
#include "imgui_impl_win32.h"

#include "config.h"
#include "keycodes.h"
#include "sunone_aimbot_cpp.h"
#include "capture.h"
#include "keyboard_listener.h"

static ID3D11Device* g_pd3dDevice = NULL;
static ID3D11DeviceContext* g_pd3dDeviceContext = NULL;
static IDXGISwapChain* g_pSwapChain = NULL;
static ID3D11RenderTargetView* g_mainRenderTargetView = NULL;
static HWND g_hwnd = NULL;

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

int overlayWidth = 500;
int overlayHeight = 300;

// init vars
std::vector<std::string> engine_models;
int prev_ai_model_index;
int current_ai_model_index;
int prev_imgsz_index;
int selected_imgsz;

// Realtime config vars
extern std::atomic<bool> detection_resolution_changed;
extern std::atomic<bool> capture_method_changed;
extern std::atomic<bool> capture_cursor_changed;
extern std::atomic<bool> capture_borders_changed;
extern std::atomic<bool> capture_fps_changed;
extern std::atomic<bool> detector_model_changed;
extern std::atomic<bool> show_window_changed;

std::vector<std::string> getEngineFiles()
{
    std::vector<std::string> engineFiles;

    for (const auto& entry : std::filesystem::directory_iterator("models/"))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".engine")
        {
            engineFiles.push_back(entry.path().filename().string());
        }
    }
    return engineFiles;
}

int getModelIndex(std::vector<std::string> engine_models)
{
    auto it = std::find(engine_models.begin(), engine_models.end(), config.ai_model);

    if (it != engine_models.end())
    {
        return std::distance(engine_models.begin(), it);
    }
    else
    {
        return 0; // not found
    }
}

std::string intToString(int value) {
    return std::to_string(value);
}

int getImageSizeIndex(int engine_image_size, const int* model_sizes, int model_sizes_count)
{
    for (int i = 0; i < model_sizes_count; ++i)
    {
        if (model_sizes[i] == engine_image_size)
        {
            return i;
        }
    }
    return 0; // not found
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
    if (FAILED(hr))
        return false;

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
#ifdef _DEBUG
    createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif

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
            if (GetClientRect(hWnd, &rect))
            {
                UINT width = rect.right - rect.left;
                UINT height = rect.bottom - rect.top;

                CleanupRenderTarget();
                g_pSwapChain->ResizeBuffers(0, width, height, DXGI_FORMAT_UNKNOWN, 0);
                CreateRenderTarget();
            }
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
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    ImGui_ImplWin32_Init(g_hwnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    ImGui::StyleColorsDark();

    // other setups
    engine_models = getEngineFiles();
    prev_ai_model_index = getModelIndex(engine_models);
    current_ai_model_index = prev_ai_model_index;

    int model_sizes[] = { 320, 480, 640 };
    const int model_sizes_count = sizeof(model_sizes) / sizeof(model_sizes[0]);

    selected_imgsz = getImageSizeIndex(config.engine_image_size, model_sizes, model_sizes_count);
    prev_imgsz_index = selected_imgsz;
}

bool CreateOverlayWindow()
{
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
    
    // Opacity
    if (config.overlay_opacity <= 0)
    {
        std::cout << "[Overlay] The transparency value of the overlay is set to less than one, this value is unacceptable." << std::endl;
        std::cin.get();
        return -1;
    }

    if (config.overlay_opacity >= 256)
    {
        std::cout << "[Overlay] The transparency value of the overlay is set to more than 255, this value is unacceptable." << std::endl;
        std::cin.get();
        return -1;
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
        std::cerr << "Can't create overlay window" << std::endl;
        return;
    }

    SetupImGui();

    bool show_overlay = false;

    // Real time settings vars
    static int prev_detection_resolution = config.detection_resolution;
    
    bool prev_capture_method = config.duplication_api;
    bool prev_capture_cursor = config.capture_cursor;
    bool prev_capture_borders = config.capture_borders;
    float prev_capture_fps = config.capture_fps;

    bool prev_show_window = config.show_window;
    int prev_opacity = config.overlay_opacity;

    int current_ai_model_index = prev_ai_model_index;
    float prev_confidence_threshold = config.confidence_threshold;
    float prev_nms_threshold = config.nms_threshold;

    bool prev_disable_headshot = config.disable_headshot;
    float prev_body_y_offset = config.body_y_offset;
    bool prev_ignore_third_person = config.ignore_third_person;

    int prev_dpi = config.dpi;
    float prev_sensitivity = config.sensitivity;
    int prev_fovX = config.fovX;
    int prev_fovY = config.fovY;
    float prev_minSpeedMultiplier = config.minSpeedMultiplier;
    float prev_maxSpeedMultiplier = config.maxSpeedMultiplier;
    float prev_predictionInterval = config.predictionInterval;

    bool prev_auto_shoot = config.auto_shoot;
    float prev_bScope_multiplier = config.bScope_multiplier;

    bool prev_show_fps = config.show_fps;
    int prev_window_size = config.window_size;
    int prev_screenshot_delay = config.screenshot_delay;
    bool prev_always_on_top = config.always_on_top;

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

            ImGui::Begin("Options", &show_overlay, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
            {
                std::lock_guard<std::mutex> lock(configMutex);
                if (ImGui::BeginTabBar("Options tab bar"))
                {
                    if (ImGui::BeginTabItem("Capture"))
                    {
                        ImGui::SliderInt("Detection Resolution", &config.detection_resolution, 50, 720);
                        ImGui::SliderInt("Lock FPS", &config.capture_fps, 0, 240);
                        ImGui::Checkbox("Duplication API", &config.duplication_api);
                        if (config.duplication_api == false)
                        {
                            ImGui::Checkbox("Capture Borders", &config.capture_borders);
                            ImGui::Checkbox("Capture Cursor", &config.capture_cursor);
                        }

                        ImGui::EndTabItem();
                    }

                    if (ImGui::BeginTabItem("Target"))
                    {
                        ImGui::Checkbox("Disable Headshot", &config.disable_headshot);
                        ImGui::SliderFloat("Body Y Offset", &config.body_y_offset, -2.0f, 2.0f, "%.2f");
                        ImGui::Checkbox("Ignore Third Person", &config.ignore_third_person);

                        ImGui::EndTabItem();
                    }

                    if (ImGui::BeginTabItem("Mouse"))
                    {
                        ImGui::SliderInt("DPI", &config.dpi, 800, 5000);
                        ImGui::SliderFloat("Sensitivity", &config.sensitivity, 0.1f, 10.0f, "%.1f");
                        ImGui::SliderInt("FOV X", &config.fovX, 60, 120);
                        ImGui::SliderInt("FOV Y", &config.fovY, 40, 100);
                        ImGui::SliderFloat("Min Speed Multiplier", &config.minSpeedMultiplier, 0.1f, 5.0f, "%.1f");
                        ImGui::SliderFloat("Max Speed Multiplier", &config.maxSpeedMultiplier, 0.1f, 5.0f, "%.1f");
                        ImGui::SliderFloat("Prediction Interval", &config.predictionInterval, 0.1f, 3.0f, "%.1f");

                        ImGui::Checkbox("Auto Shoot", &config.auto_shoot);
                        ImGui::SliderFloat("bScope Multiplier", &config.bScope_multiplier, 0.5f, 2.0f, "%.1f");

                        ImGui::EndTabItem();
                    }

                    if (ImGui::BeginTabItem("AI"))
                    {
                        // ai models
                        std::vector<const char*> models_items;
                        models_items.reserve(engine_models.size());
                        for (const auto& item : engine_models)
                        {
                            models_items.push_back(item.c_str());
                        }

                        if (ImGui::ListBox("Model", &current_ai_model_index, models_items.data(), static_cast<int>(models_items.size())))
                        {
                            if (current_ai_model_index != prev_ai_model_index)
                            {
                                config.ai_model = engine_models[current_ai_model_index];
                                detector_model_changed.store(true);
                                prev_ai_model_index = current_ai_model_index;
                                config.saveConfig("config.ini");
                            }
                        }

                        // selected model text
                        const std::string msg_text_model = "Selected model: ";
                        const std::string msg_text_current_selected_model = engine_models[current_ai_model_index];
                        const std::string msg_used_model = msg_text_model + msg_text_current_selected_model;
                        ImGui::Text("%s", msg_used_model.c_str());

                        // model size
                        int model_sizes[] = { 320, 480, 640 };
                        const int model_sizes_count = sizeof(model_sizes) / sizeof(model_sizes[0]);

                        const char* model_sizes_str[model_sizes_count];
                        std::string model_sizes_buffer[model_sizes_count];

                        for (int i = 0; i < model_sizes_count; ++i)
                        {
                            model_sizes_buffer[i] = intToString(model_sizes[i]);
                            model_sizes_str[i] = model_sizes_buffer[i].c_str();
                        }

                        if (ImGui::ListBox("Engine Image Size", &selected_imgsz, model_sizes_str, model_sizes_count))
                        {
                            if (selected_imgsz != prev_imgsz_index)
                            {
                                std::cout << "Image size changed to: " << model_sizes[selected_imgsz] << std::endl;

                                config.engine_image_size = model_sizes[selected_imgsz];
                                detector_model_changed.store(true);
                                prev_imgsz_index = selected_imgsz;

                                config.saveConfig("config.ini");
                            }
                        }

                        ImGui::SliderFloat("Confidence Threshold", &config.confidence_threshold, 0.0f, 1.0f);
                        ImGui::SliderFloat("NMS Threshold", &config.nms_threshold, 0.0f, 1.0f);

                        ImGui::EndTabItem();
                    }

                    if (ImGui::BeginTabItem("Overlay"))
                    {
                        ImGui::SliderInt("Overlay Opacity", &config.overlay_opacity, 40, 255);

                        ImGui::EndTabItem();
                    }
                    
                    // TODO CUSTOM CLASSES

                    if (ImGui::BeginTabItem("Debug"))
                    {
                        ImGui::Checkbox("Show Window", &config.show_window);
                        ImGui::Checkbox("Show FPS", &config.show_fps);
                        //ImGui::InputText("Window Name", &config.window_name[0], config.window_name.capacity() + 1); // TODO
                        ImGui::InputInt("Window Size", &config.window_size);
                        ImGui::Checkbox("Always On Top", &config.always_on_top);

                        ImGui::EndTabItem();
                    }

                    // DETECTION RESOLUTION
                    if (prev_detection_resolution != config.detection_resolution)
                    {
                        detection_resolution_changed.store(true);
                        detector_model_changed.store(true); // reboot vars for visuals
                        prev_detection_resolution = config.detection_resolution;
                        config.saveConfig("config.ini");
                    }

                    // CAPTURE METHOD
                    if (prev_capture_method != config.duplication_api)
                    {
                        capture_method_changed.store(true);
                        prev_capture_method = config.duplication_api;
                        config.saveConfig("config.ini");
                    }

                    // CAPTURE CURSOR
                    if (prev_capture_cursor != config.capture_cursor && config.duplication_api == false)
                    {
                        capture_cursor_changed.store(true);
                        prev_capture_cursor = config.capture_cursor;
                        config.saveConfig("config.ini");
                    }

                    // CAPTURE BORDERS
                    if (prev_capture_borders != config.capture_borders && config.duplication_api == false)
                    {
                        capture_borders_changed.store(true);
                        prev_capture_borders = config.capture_borders;
                        config.saveConfig("config.ini");
                    }

                    // CAPTURE_FPS
                    if (prev_capture_fps != config.capture_fps)
                    {
                        capture_fps_changed.store(true);
                        prev_capture_fps = config.capture_fps;
                        config.saveConfig("config.ini");
                    }

                    // DISABLE_HEADSHOT / BODY_Y_OFFSET / IGNORE_THIRD_PERSON
                    if (prev_disable_headshot != config.disable_headshot ||
                        prev_body_y_offset != config.body_y_offset ||
                        prev_ignore_third_person != config.ignore_third_person)
                    {
                        prev_disable_headshot = config.disable_headshot;
                        prev_body_y_offset = config.body_y_offset;
                        prev_ignore_third_person = config.ignore_third_person;
                        config.saveConfig("config.ini");
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
                        config.saveConfig("config.ini");
                    }

                    // AUTO_SHOOT / BSCOPE_MULTIPLIER
                    if (prev_auto_shoot != config.auto_shoot ||
                        prev_bScope_multiplier != config.bScope_multiplier)
                    {
                        prev_auto_shoot = config.auto_shoot;
                        prev_bScope_multiplier = config.bScope_multiplier;
                        config.saveConfig("config.ini");
                    }

                    // OVERLAY OPACITY
                    if (prev_opacity != config.overlay_opacity)
                    {
                        BYTE opacity = config.overlay_opacity;
                        SetLayeredWindowAttributes(g_hwnd, 0, opacity, LWA_ALPHA);
                        config.saveConfig("config.ini");
                    }

                    // CONFIDENCE THERSHOLD / NMS THRESHOLD
                    if (prev_confidence_threshold != config.confidence_threshold ||
                        prev_nms_threshold != config.nms_threshold)
                    {
                        prev_nms_threshold = config.nms_threshold;
                        prev_confidence_threshold = config.confidence_threshold;
                        config.saveConfig("config.ini");
                    }

                    // SHOW WINDOW
                    if (prev_show_window != config.show_window)
                    {
                        show_window_changed.store(true);
                        prev_show_window = config.show_window;
                        config.saveConfig("config.ini");
                    }
                    
                    // ALWAYS_ON_TOP
                    if (prev_always_on_top != config.always_on_top)
                    {
                        // TODO: update window property
                        prev_always_on_top = config.always_on_top;
                    }

                    // SHOW_FPS / WINDOW_SIZE / SCREENSHOT_DELAY
                    if (prev_show_fps != config.show_fps ||
                        prev_window_size != config.window_size ||
                        prev_screenshot_delay != config.screenshot_delay)
                    {
                        prev_show_fps = config.show_fps;
                        prev_window_size = config.window_size;
                        prev_screenshot_delay = config.screenshot_delay;
                        config.saveConfig("config.ini");
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

            if (result == DXGI_STATUS_OCCLUDED) // TODO
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

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