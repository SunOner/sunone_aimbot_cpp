#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include "shellapi.h"
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

#include "config.h"
#include "keycodes.h"
#include "sunone_aimbot_cpp.h"
#include "capture.h"
#include "keyboard_listener.h"
#include "other_tools.h"
#include "memory_images.h"
#include "virtual_camera.h"
#include "Snowflake.hpp"

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

int overlayWidth = 680;
int overlayHeight = 480;

// init vars
std::vector<std::string> availableModels;
int prev_imgsz_index;
int selected_imgsz;

// Realtime config vars
extern std::atomic<bool> detection_resolution_changed;
extern std::atomic<bool> capture_method_changed;
extern std::atomic<bool> capture_cursor_changed;
extern std::atomic<bool> capture_borders_changed;
extern std::atomic<bool> capture_fps_changed;
extern std::atomic<bool> capture_window_changed;
extern std::atomic<bool> detector_model_changed;
extern std::atomic<bool> show_window_changed;

// textures
ID3D11ShaderResourceView* bodyTexture = nullptr;
ImVec2 bodyImageSize;

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

    // Load body texture
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
    
    // Opacity checks and init
    if (config.overlay_opacity <= 20)
    {
        std::cout << "[Overlay] The transparency value of the overlay is set to less than 20, this value is unacceptable." << std::endl;
        config.overlay_opacity = 20;
        config.saveConfig("config.ini");
    }

    if (config.overlay_opacity >= 256)
    {
        std::cout << "[Overlay] The transparency value of the overlay is set to more than 255, this value is unacceptable." << std::endl;
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
        std::cerr << "Can't create overlay window" << std::endl;
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
    
    bool disable_winrt_futures = checkwin1903();

    bool prev_capture_borders = config.capture_borders;
    bool prev_capture_cursor = config.capture_cursor;
    int monitors = get_active_monitors();
    std::vector<std::string> virtual_cameras = { };

    // Target
    bool prev_disable_headshot = config.disable_headshot;
    float prev_body_y_offset = config.body_y_offset;
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

    // keycodes
    std::vector<std::string> key_names;
    for (const auto& pair : KeyCodes::key_code_map)
    {
        key_names.push_back(pair.first);
    }
    std::sort(key_names.begin(), key_names.end());

    std::vector<const char*> key_names_cstrs;
    key_names_cstrs.reserve(key_names.size());
    for (const auto& name : key_names)
    {
        key_names_cstrs.push_back(name.c_str());
    }

    // input methods
    int input_method_index = 0;
    if (config.input_method == "WIN32")
        input_method_index = 0;
    else if (config.input_method == "GHUB")
        input_method_index = 1;
    else if (config.input_method == "ARDUINO")
        input_method_index = 2;
    else
        input_method_index = 0;

    std::string ghub_version = get_ghub_version();
    
    std::vector<std::string> availableModels = getAvailableModels();
    if (availableModels.empty())
    {
        std::cerr << "[Overlay] No models found in 'models' directory." << std::endl;
    }

    // SNOW THEME VARS
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
        680,
        480,
        Snowflake::vec3(0.f, 0.005f),
        IM_COL32(255, 255, 255, 255));

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
#pragma region CAPTURE
                    // ********************************************* CAPTURE ********************************************
                    if (ImGui::BeginTabItem("Capture"))
                    {
                        ImGui::SliderInt("Detection Resolution", &config.detection_resolution, 50, 1280);
                        if (config.detection_resolution >= 400)
                        {
                            ImGui::TextColored(ImVec4(255, 255, 0, 255), "WARNING: A large screen capture size can negatively affect performance.");
                        }

                        ImGui::SliderInt("Lock FPS", &config.capture_fps, 0, 240);
                        if (config.capture_fps == 0)
                        {
                            ImGui::SameLine();
                            ImGui::TextColored(ImVec4(255, 0, 0, 255), "-> Disabled");
                        }

                        if (config.capture_fps == 0 || config.capture_fps >= 61)
                        {
                            ImGui::TextColored(ImVec4(255, 255, 0, 255), "WARNING: A large number of FPS can negatively affect performance.");
                        }

                        if (ImGui::Checkbox("Circle mask", &config.circle_mask))
                        {
                            capture_method_changed.store(true);
                            config.saveConfig();
                        }

                        std::vector<std::string> captureMethodOptions = { "duplication_api", "winrt", "virtual_camera" };
                        std::vector<const char*> captureMethodItems;
                        for (const auto& option : captureMethodOptions)
                        {
                            captureMethodItems.push_back(option.c_str());
                        }

                        int currentcaptureMethodIndex = 0;
                        for (size_t i = 0; i < captureMethodOptions.size(); ++i)
                        {
                            if (captureMethodOptions[i] == config.capture_method)
                            {
                                currentcaptureMethodIndex = static_cast<int>(i);
                                break;
                            }
                        }

                        if (ImGui::Combo("Capture method", &currentcaptureMethodIndex, captureMethodItems.data(), static_cast<int>(captureMethodItems.size()))) {
                            config.capture_method = captureMethodOptions[currentcaptureMethodIndex];
                            config.saveConfig();
                            capture_method_changed.store(true);
                        }

                        if (config.capture_method == "winrt")
                        {
                            if (disable_winrt_futures)
                            {
                                ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
                                ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
                            }

                            ImGui::Checkbox("Capture Borders", &config.capture_borders);
                            ImGui::Checkbox("Capture Cursor", &config.capture_cursor);

                            if (disable_winrt_futures)
                            {
                                ImGui::PopStyleVar();
                                ImGui::PopItemFlag();
                            }
                        }

                        if (config.capture_method == "duplication_api" || config.capture_method == "winrt")
                        {
                            std::vector<std::string> monitorNames;
                            if (monitors == -1)
                            {
                                monitorNames.push_back("Monitor 1");
                            }
                            else
                            {
                                for (int i = -1; i < monitors; ++i)
                                {
                                    monitorNames.push_back("Monitor " + std::to_string(i + 1));
                                }
                            }

                            std::vector<const char*> monitorItems;
                            for (const auto& name : monitorNames)
                            {
                                monitorItems.push_back(name.c_str());
                            }

                            if (ImGui::Combo("Capture monitor (CUDA GPU)", &config.monitor_idx, monitorItems.data(), static_cast<int>(monitorItems.size())))
                            {
                                config.saveConfig();
                                capture_method_changed.store(true);
                            }
                        }

                        if (config.capture_method == "virtual_camera")
                        {
                            if (!virtual_cameras.empty())
                            {
                                int currentCameraIndex = 0;
                                for (size_t i = 0; i < virtual_cameras.size(); i++)
                                {
                                    if (virtual_cameras[i] == config.virtual_camera_name)
                                    {
                                        currentCameraIndex = i;
                                        break;
                                    }
                                }
                        
                                std::vector<const char*> cameraItems;
                                for (const auto& cam : virtual_cameras)
                                {
                                    cameraItems.push_back(cam.c_str());
                                }
                        
                                if (ImGui::Combo("Virtual Camera", &currentCameraIndex,
                                    cameraItems.data(), static_cast<int>(cameraItems.size())))
                                {
                                    config.virtual_camera_name = virtual_cameras[currentCameraIndex];
                                    config.saveConfig();
                                    capture_method_changed.store(true);
                                }
                                ImGui::SameLine();
                                if (ImGui::Button("Update##update_virtual_cameras"))
                                {
                                    virtual_cameras = VirtualCameraCapture::GetAvailableVirtualCameras();
                                }
                            }
                            else
                            {
                                ImGui::Text("No virtual cameras found");
                                ImGui::SameLine();
                                if (ImGui::Button("Update##update_virtual_cameras"))
                                {
                                    virtual_cameras = VirtualCameraCapture::GetAvailableVirtualCameras();
                                }
                            }
                        }

                        ImGui::EndTabItem();
                    }
#pragma endregion
#pragma region TARGET
                    // ********************************************* TARGET *********************************************
                    if (ImGui::BeginTabItem("Target"))
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

                        ImGui::EndTabItem();
                    }
#pragma endregion
#pragma region MOUSE
                    // ********************************************** MOUSE *********************************************
                    if (ImGui::BeginTabItem("Mouse"))
                    {
                        ImGui::SliderInt("DPI", &config.dpi, 800, 5000);
                        ImGui::SliderFloat("Sensitivity", &config.sensitivity, 0.1f, 10.0f, "%.1f");
                        ImGui::SliderInt("FOV X", &config.fovX, 60, 120);
                        ImGui::SliderInt("FOV Y", &config.fovY, 40, 100);
                        ImGui::SliderFloat("Min Speed Multiplier", &config.minSpeedMultiplier, 0.1f, 5.0f, "%.1f");
                        ImGui::SliderFloat("Max Speed Multiplier", &config.maxSpeedMultiplier, 0.1f, 5.0f, "%.1f");
                        ImGui::SliderFloat("Prediction Interval", &config.predictionInterval, 0.00f, 3.00f, "%.2f");
                        if (config.predictionInterval == 0.00f)
                        {
                            ImGui::SameLine();
                            ImGui::TextColored(ImVec4(255, 0, 0, 255), "-> Disabled");
                        }

                        ImGui::Checkbox("Auto Shoot", &config.auto_shoot);
                        if (config.auto_shoot)
                        {
                            ImGui::SliderFloat("bScope Multiplier", &config.bScope_multiplier, 0.5f, 2.0f, "%.1f");
                        }

                        // INPUT METHODS
                        ImGui::Separator();
                        std::vector<std::string> input_methods = {"WIN32", "GHUB", "ARDUINO" };
                        std::vector<const char*> method_items;
                        method_items.reserve(input_methods.size());
                        for (const auto& item : input_methods)
                        {
                            method_items.push_back(item.c_str());
                        }

                        std::string combo_label = "Mouse Input method";
                        int input_method_index = 0;
                        for (size_t i = 0; i < input_methods.size(); ++i)
                        {
                            if (input_methods[i] == config.input_method)
                            {
                                input_method_index = static_cast<int>(i);
                                break;
                            }
                        }

                        if (ImGui::Combo("Mouse Input Method", &input_method_index, method_items.data(), static_cast<int>(method_items.size())))
                        {
                            std::string new_input_method = input_methods[input_method_index];

                            if (new_input_method != config.input_method)
                            {
                                config.input_method = new_input_method;
                                config.saveConfig();
                                input_method_changed.store(true);
                            }
                        }

                        if (config.input_method == "ARDUINO")
                        {
                            if (serial)
                            {
                                if (serial->isOpen())
                                {
                                    ImGui::TextColored(ImVec4(0, 255, 0, 255), "Arduino connected");
                                }
                                else
                                {
                                    ImGui::TextColored(ImVec4(255, 0, 0, 255), "Arduino not connected");
                                }
                            }

                            std::vector<std::string> port_list;
                            for (int i = 1; i <= 30; ++i)
                            {
                                port_list.push_back("COM" + std::to_string(i));
                            }

                            std::vector<const char*> port_items;
                            port_items.reserve(port_list.size());
                            for (const auto& port : port_list)
                            {
                                port_items.push_back(port.c_str());
                            }

                            int port_index = 0;
                            for (size_t i = 0; i < port_list.size(); ++i)
                            {
                                if (port_list[i] == config.arduino_port)
                                {
                                    port_index = static_cast<int>(i);
                                    break;
                                }
                            }

                            if (ImGui::Combo("Arduino Port", &port_index, port_items.data(), static_cast<int>(port_items.size())))
                            {
                                config.arduino_port = port_list[port_index];
                                config.saveConfig();
                                input_method_changed.store(true);
                            }

                            std::vector<int> baud_rate_list = { 9600, 19200, 38400, 57600, 115200 };
                            std::vector<std::string> baud_rate_str_list;
                            for (const auto& rate : baud_rate_list)
                            {
                                baud_rate_str_list.push_back(std::to_string(rate));
                            }

                            std::vector<const char*> baud_rate_items;
                            baud_rate_items.reserve(baud_rate_str_list.size());
                            for (const auto& rate_str : baud_rate_str_list)
                            {
                                baud_rate_items.push_back(rate_str.c_str());
                            }

                            int baud_rate_index = 0;
                            for (size_t i = 0; i < baud_rate_list.size(); ++i)
                            {
                                if (baud_rate_list[i] == config.arduino_baudrate)
                                {
                                    baud_rate_index = static_cast<int>(i);
                                    break;
                                }
                            }

                            if (ImGui::Combo("Arduino Baudrate", &baud_rate_index, baud_rate_items.data(), static_cast<int>(baud_rate_items.size())))
                            {
                                config.arduino_baudrate = baud_rate_list[baud_rate_index];
                                config.saveConfig();
                                input_method_changed.store(true);
                            }

                            if (ImGui::Checkbox("Arduino 16-bit Mouse", &config.arduino_16_bit_mouse))
                            {
                                config.saveConfig();
                                input_method_changed.store(true);
                            }
                            if (ImGui::Checkbox("Arduino Enable Keys", &config.arduino_enable_keys))
                            {
                                config.saveConfig();
                                input_method_changed.store(true);
                            }
                        }
                        else if (config.input_method == "GHUB")
                        {
                            if (ghub_version == "13.1.4")
                            {
                                std::string ghub_version_label = "The correct version of Ghub is installed: " + ghub_version;
                                ImGui::Text(ghub_version_label.c_str());
                            }
                            else
                            {
                                if (ghub_version == "")
                                {
                                    ghub_version = "unknown";
                                }

                                std::string ghub_version_label = "Installed Ghub version: " + ghub_version;
                                ImGui::Text(ghub_version_label.c_str());
                                ImGui::Text("The wrong version of Ghub is installed or the path to Ghub is not set by default.\nDefault system path: C:\\Program Files\\LGHUB");
                                if (ImGui::Button("GHub Docs"))
                                {
                                    ShellExecute(0, 0, L"https://github.com/SunOner/sunone_aimbot_docs/blob/main/tips/ghub.md", 0, 0, SW_SHOW);
                                }
                            }
                            ImGui::TextColored(ImVec4(255, 0, 0, 255), "Use at your own risk, the method is detected in some games.");
                        }
                        else if (config.input_method == "WIN32")
                        {
                            ImGui::TextColored(ImVec4(255, 255, 255, 255), "This is a standard mouse input method, it may not work in most games. Use GHUB or ARDUINO.");
                            ImGui::TextColored(ImVec4(255, 0, 0, 255), "Use at your own risk, the method is detected in some games.");
                        }

                        ImGui::EndTabItem();
                    }
#pragma endregion
#pragma region AI
                    // *********************************************** AI ***********************************************
                    if (ImGui::BeginTabItem("AI"))
                    {
                        std::vector<std::string> availableModels = getAvailableModels();
                        if (availableModels.empty())
                        {
                            ImGui::Text("No models available in the 'models' folder.");
                        }
                        else
                        {
                            // Get index
                            int currentModelIndex = 0;
                            auto it = std::find(availableModels.begin(), availableModels.end(), config.ai_model);
                            if (it != availableModels.end())
                            {
                                currentModelIndex = static_cast<int>(std::distance(availableModels.begin(), it));
                            }

                            // Create array
                            std::vector<const char*> modelsItems;
                            modelsItems.reserve(availableModels.size());
                            for (const auto& modelName : availableModels)
                            {
                                modelsItems.push_back(modelName.c_str());
                            }

                            if (ImGui::Combo("Model", &currentModelIndex, modelsItems.data(), static_cast<int>(modelsItems.size())))
                            {
                                if (config.ai_model != availableModels[currentModelIndex])
                                {
                                    config.ai_model = availableModels[currentModelIndex];
                                    config.saveConfig();
                                    detector_model_changed.store(true);
                                }
                            }
                        }
                        ImGui::Separator();
                        std::vector<std::string> postprocessOptions = { "yolo8", "yolo9", "yolo10", "yolo11" };
                        std::vector<const char*> postprocessItems;
                        for (const auto& option : postprocessOptions) {
                            postprocessItems.push_back(option.c_str());
                        }

                        int currentPostprocessIndex = 0;
                        for (size_t i = 0; i < postprocessOptions.size(); ++i) {
                            if (postprocessOptions[i] == config.postprocess) {
                                currentPostprocessIndex = static_cast<int>(i);
                                break;
                            }
                        }

                        if (ImGui::Combo("Postprocess", &currentPostprocessIndex, postprocessItems.data(), static_cast<int>(postprocessItems.size()))) {
                            config.postprocess = postprocessOptions[currentPostprocessIndex];
                            config.saveConfig();
                            detector_model_changed.store(true);
                        }

                        ImGui::Separator();
                        ImGui::SliderFloat("Confidence Threshold", &config.confidence_threshold, 0.01f, 1.00f, "%.2f");
                        ImGui::SliderFloat("NMS Threshold", &config.nms_threshold, 0.01f, 1.00f, "%.2f");
                        ImGui::SliderInt("Max Detections", &config.max_detections, 1, 100);

                        ImGui::EndTabItem();
                    }
#pragma endregion
#pragma region OPTICAL_FLOW
                    if (ImGui::BeginTabItem("Optical flow"))
                    {
                        ImGui::Text("This is an experimental feature");
                        // enable / disable
                        if (ImGui::Checkbox("Enable Optical Flow", &config.enable_optical_flow))
                        {
                            config.saveConfig();
                            opticalFlow.manageOpticalFlowThread();
                        }

                        // draw options
                        ImGui::Separator();
                        if (ImGui::Checkbox("Draw in debug window", &config.draw_optical_flow))
                        {
                            config.saveConfig();
                        }
                        
                        if (config.draw_optical_flow)
                        {
                            ImGui::Separator();
                            if (ImGui::SliderInt("Draw steps", &config.draw_optical_flow_steps, 2, 32))
                            {
                                config.saveConfig();
                            }
                        ImGui::Separator();
                        }

                        if (ImGui::SliderFloat("Alpha CPU", &config.optical_flow_alpha_cpu, 0.01f, 1.00f, "%.2f"))
                        {
                            config.saveConfig();
                        }

                        float magnitudeThreshold = static_cast<float>(config.optical_flow_magnitudeThreshold);
                        if (ImGui::SliderFloat("Magnitude Threshold", &magnitudeThreshold, 0.01f, 10.00f, "%.2f"))
                        {
                            config.optical_flow_magnitudeThreshold = static_cast<double>(magnitudeThreshold);
                            config.saveConfig();
                        }
                        
                        if (ImGui::SliderFloat("Static Frame Threshold", &config.staticFrameThreshold, 0.01f, 10.00f, "%.2f"))
                        {
                            config.saveConfig();
                        }

                        ImGui::EndTabItem();
                    }
#pragma endregion
#pragma region BUTTONS
                    // ********************************************* BUTTONS ********************************************
                    if (ImGui::BeginTabItem("Buttons"))
                    {
                        // targeting
                        ImGui::Text("Targeting Buttons");

                        for (size_t i = 0; i < config.button_targeting.size(); )
                        {
                            std::string& current_key_name = config.button_targeting[i];

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

                            std::string combo_label = "Targeting Button " + std::to_string(i);

                            if (ImGui::Combo(combo_label.c_str(), &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
                            {
                                current_key_name = key_names[current_index];
                                config.saveConfig();
                            }

                            ImGui::SameLine();
                            std::string remove_button_label = "Remove##button_targeting" + std::to_string(i);
                            if (ImGui::Button(remove_button_label.c_str()))
                            {
                                if (config.button_targeting.size() <= 1)
                                {
                                    config.button_targeting[0] = std::string("None");
                                    config.saveConfig();
                                    continue;
                                }
                                else
                                {
                                    config.button_targeting.erase(config.button_targeting.begin() + i);
                                    config.saveConfig();
                                    continue;
                                }
                            }

                            ++i;
                        }

                        if (ImGui::Button("Add button##targeting"))
                        {
                            config.button_targeting.push_back("None");
                            config.saveConfig();
                        }

                        // exit
                        ImGui::Separator();
                        ImGui::Text("Exit Buttons");

                        for (size_t i = 0; i < config.button_exit.size(); )
                        {
                            std::string& current_key_name = config.button_exit[i];

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

                            std::string combo_label = "Exit Button " + std::to_string(i);

                            if (ImGui::Combo(combo_label.c_str(), &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
                            {
                                current_key_name = key_names[current_index];
                                config.saveConfig();
                            }

                            ImGui::SameLine();
                            std::string remove_button_label = "Remove##button_exit" + std::to_string(i);
                            if (ImGui::Button(remove_button_label.c_str()))
                            {
                                if (config.button_exit.size() <= 1)
                                {
                                    config.button_exit[0] = std::string("None");
                                    config.saveConfig();
                                    continue;
                                }
                                else
                                {
                                    config.button_exit.erase(config.button_exit.begin() + i);
                                    config.saveConfig();
                                    continue;
                                }
                            }

                            ++i;
                        }

                        if (ImGui::Button("Add button##exit"))
                        {
                            config.button_exit.push_back("None");
                            config.saveConfig();
                        }

                        // pause
                        ImGui::Separator();
                        ImGui::Text("Pause Buttons");

                        for (size_t i = 0; i < config.button_pause.size(); )
                        {
                            std::string& current_key_name = config.button_pause[i];

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

                            std::string combo_label = "Pause Button " + std::to_string(i);

                            if (ImGui::Combo(combo_label.c_str(), &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
                            {
                                current_key_name = key_names[current_index];
                                config.saveConfig();
                            }

                            ImGui::SameLine();
                            std::string remove_button_label = "Remove##button_pause" + std::to_string(i);
                            if (ImGui::Button(remove_button_label.c_str()))
                            {
                                if (config.button_pause.size() <= 1)
                                {
                                    config.button_pause[0] = std::string("None");
                                    config.saveConfig();
                                    continue;
                                }
                                else
                                {
                                    config.button_pause.erase(config.button_pause.begin() + i);
                                    config.saveConfig();
                                    continue;
                                }
                            }
                            ++i;
                        }

                        if (ImGui::Button("Add button##pause"))
                        {
                            config.button_pause.push_back("None");
                            config.saveConfig();
                        }

                        // reload config
                        ImGui::Separator();
                        ImGui::Text("Reload config Buttons");

                        for (size_t i = 0; i < config.button_reload_config.size(); )
                        {
                            std::string& current_key_name = config.button_reload_config[i];

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

                            std::string combo_label = "Reload config Button " + std::to_string(i);

                            if (ImGui::Combo(combo_label.c_str(), &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
                            {
                                current_key_name = key_names[current_index];
                                config.saveConfig();
                            }

                            ImGui::SameLine();
                            std::string remove_button_label = "Remove##button_reload_config" + std::to_string(i);
                            if (ImGui::Button(remove_button_label.c_str()))
                            {
                                if (config.button_reload_config.size() <= 1)
                                {
                                    config.button_reload_config[0] = std::string("None");
                                    config.saveConfig();
                                    continue;
                                }
                                else
                                {
                                    config.button_reload_config.erase(config.button_reload_config.begin() + i);
                                    config.saveConfig();
                                    continue;
                                }
                            }

                            ++i;
                        }

                        if (ImGui::Button("Add button##reload_config"))
                        {
                            config.button_reload_config.push_back("None");
                            config.saveConfig();
                        }

                        // overlay
                        ImGui::Separator();
                        ImGui::Text("Overlay Buttons");

                        for (size_t i = 0; i < config.button_open_overlay.size(); )
                        {
                            std::string& current_key_name = config.button_open_overlay[i];

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

                            std::string combo_label = "Overlay Button " + std::to_string(i);

                            if (ImGui::Combo(combo_label.c_str(), &current_index, key_names_cstrs.data(), static_cast<int>(key_names_cstrs.size())))
                            {
                                current_key_name = key_names[current_index];
                                config.saveConfig();
                            }

                            ImGui::SameLine();
                            std::string remove_button_label = "Remove##button_open_overlay" + std::to_string(i);
                            if (ImGui::Button(remove_button_label.c_str()))
                            {
                                config.button_open_overlay.erase(config.button_open_overlay.begin() + i);
                                config.saveConfig();
                                continue;
                            }

                            ++i;
                        }

                        if (ImGui::Button("Add button##overlay"))
                        {
                            config.button_open_overlay.push_back("None");
                            config.saveConfig();
                        }

                        ImGui::EndTabItem();
                    }
#pragma endregion
#pragma region OVERLAY
                    // ********************************************* OVERLAY ********************************************
                    if (ImGui::BeginTabItem("Overlay"))
                    {
                        ImGui::SliderInt("Overlay Opacity", &config.overlay_opacity, 40, 255);

                        if (ImGui::Checkbox("Enable snow theme", &config.overlay_snow_theme))
                        {
                            config.saveConfig();
                        }

                        ImGui::EndTabItem();
                    }
#pragma endregion
#pragma region DEBUG
                    // ********************************************** DEBUG *********************************************
                    if (ImGui::BeginTabItem("Debug"))
                    {
                        ImGui::Checkbox("Show Window", &config.show_window);
                        ImGui::Checkbox("Show FPS", &config.show_fps);
                        //ImGui::InputText("Window Name", &config.window_name[0], config.window_name.capacity() + 1); // TODO
                        ImGui::SliderInt("Window Size", &config.window_size, 10, 350);

                        // screenshot_button
                        ImGui::Separator();
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
                        ImGui::Checkbox("Always On Top", &config.always_on_top);
                        ImGui::Checkbox("Verbose console output", &config.verbose);
                        // test functions
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

                        ImGui::EndTabItem();
                    }
#pragma endregion
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

                    // DISABLE_HEADSHOT / BODY_Y_OFFSET / IGNORE_THIRD_PERSON / SHOOTING_RANGE_TARGETS / AUTO_AIM
                    if (prev_disable_headshot != config.disable_headshot ||
                        prev_body_y_offset != config.body_y_offset ||
                        prev_ignore_third_person != config.ignore_third_person ||
                        prev_shooting_range_targets != config.shooting_range_targets ||
                        prev_auto_aim != config.auto_aim)
                    {
                        prev_disable_headshot = config.disable_headshot;
                        prev_body_y_offset = config.body_y_offset;
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
            ImGui::TextColored(ImVec4(255, 255, 255, 100),
                "Do not test shooting and aiming with the overlay and debug window is open.");

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

    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();

    if (bodyTexture)
    {
        bodyTexture->Release();
        bodyTexture = nullptr;
    }

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